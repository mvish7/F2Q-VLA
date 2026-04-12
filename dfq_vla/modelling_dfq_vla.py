from dataclasses import dataclass
from typing import Optional, Union, Any

import torch
import torch.nn as nn
from transformers import (
    PreTrainedModel,
    AutoModel,
    AutoModelForCausalLM,
    GenerationMixin
)
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.modeling_outputs import ModelOutput
from dfq_vla.configuration_dfq_vla import DFQVLAConfig
from dfq_vla.trajectory_projector import TrajHistProjector, prepare_traj_input
from dfq_vla.traj_utils import TrajectoryFusionMixin
from dfq_vla.action_head import ActionChunkingHead
from dfq_vla.flex_scene_encoder import FlexSceneEncoder, create_flex_scene_encoder
import sys
import os

try:
    from dfq_vla.vqvae_tokenizer import VQVAETrajectoryTokenizer
except ImportError:
    print(f"Warning: Could not import VQVAETrajectoryTokenizer.")
    VQVAETrajectoryTokenizer = None

from dfq_vla.loss import DFQVLALoss


class DFQVLAProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        # DinoV3 hidden size -> Qwen hidden size
        self.linear_1 = nn.Linear(config.vision_config.hidden_size, config.text_config.hidden_size, bias=False)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=False)

    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states

@dataclass
class DFQVLAOutputWithPast(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
        The rope index difference between sequence length and multimodal rope.
    text_loss, xyz_loss, rot_loss: Individual sub-loss components for logging.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None
    text_loss: Optional[torch.FloatTensor] = None
    xyz_loss: Optional[torch.FloatTensor] = None
    rot_loss: Optional[torch.FloatTensor] = None


class DFQVLAPretrainedModel(PreTrainedModel):
    config: DFQVLAConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = []
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {}


class DFQVLAForConditionalGeneration(DFQVLAPretrainedModel, GenerationMixin, TrajectoryFusionMixin):
    _checkpoint_conversion_mapping = {}
    _tied_weights_keys = ["lm_head.weight"]
    _keys_to_ignore_on_load_missing = [r"^vqvae_tokenizer\..*"]
    config_class = DFQVLAConfig
    accepts_loss_kwargs = False
    _supports_flash_attn_2 = True

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # 1. Load Vision Encoder (DINOv3)
        # We use from_config to initialize empty structure, weights loaded later
        self.vision_tower = AutoModel.from_config(config.vision_config)

        # 2. Load LLM (Qwen3)
        self.language_model = AutoModel.from_config(config.text_config)

        # 3. Projector (3072 -> 1024)
        self.projector = DFQVLAProjector(config)

        # LM head
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        
        # 4. Initialize trajectory history projector
        self._initialize_traj_projector(config)
        
        # 5. Initialize action head for future trajectory prediction
        if getattr(config, 'include_action_head', True):
            self._initialize_action_head(config)
        else:
            self.action_head = None
        
        # 6. Initialize Loss Calculator
        loss_weights = getattr(config, "loss_weights", None)
        self.loss_calculator = DFQVLALoss(loss_weights)
        
        # 7. Initialize Flex Scene Encoder (optional)
        self._initialize_flex_scene_encoder(config)
        
        # 8. Initialize VQ-VAE Trajectory Decoder (Phase 2)
        if getattr(config, 'include_vqvae', True):
            self._initialize_vqvae(config)
        else:
            self.vqvae_tokenizer = None

        # 9. Tie weights if necessary (standard HF practice)
        self.post_init()
        
    def _initialize_vqvae(self, config):
        """Initialize frozen VQ-VAE for trajectory decoding."""
        vqvae_checkpoint_path = getattr(config, "vqvae_checkpoint_path", None)
        if vqvae_checkpoint_path and VQVAETrajectoryTokenizer is not None:
            print(f"Loading VQ-VAE from {vqvae_checkpoint_path}...")
            # Initialize unified tokenizer wrapper (also freezes weights)
            self.vqvae_tokenizer = VQVAETrajectoryTokenizer(
                checkpoint_path=vqvae_checkpoint_path,
                num_embeddings=getattr(config, "vqvae_num_embeddings", 768),
                embedding_dim=getattr(config, "vqvae_embedding_dim", 256),
                hidden_dim=getattr(config, "vqvae_hidden_dim", 256)
            )
            print("Successfully loaded frozen VQ-VAE tokenizer.")
        else:
            self.vqvae_tokenizer = None
            if getattr(config, "vqvae_checkpoint_path", None) is not None:
                print("Warning: config specifies vqvae_checkpoint_path but VQVAETrajectoryTokenizer import failed.")
    
    def _initialize_traj_projector(self, config):
        """Initialize MLP projector for trajectory history encoding."""
        traj_input_dim = getattr(config, "traj_input_dim", 5)
        self.traj_projector = TrajHistProjector(
            input_dim=traj_input_dim,
            hidden_size=config.hidden_size,
        )
    
    def _initialize_action_head(self, config):
        """Initialize action chunking head for future trajectory prediction."""
        self.action_head = ActionChunkingHead(
            hidden_size=config.hidden_size,
            num_queries=config.num_action_queries,
            num_layers=config.num_action_layers,
            nhead=config.action_nhead,
            dim_feedforward=config.action_dim_feedforward,
            dropout=config.action_dropout,
        )
    
    def _initialize_flex_scene_encoder(self, config):
        """Initialize Flex Scene Encoder for multi-camera, multi-timestamp encoding."""
        if config.use_flex_scene_encoder:
            self.flex_scene_encoder = create_flex_scene_encoder(config)
        else:
            self.flex_scene_encoder = None

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_placeholder_mask(self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: Optional[torch.FloatTensor] = None
    ):

        """
            Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
            equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)

        else:
            special_image_mask = input_ids == self.config.image_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}"
            )

        return special_image_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        camera_ids: Optional[torch.LongTensor] = None,  # [B, num_images] for Flex encoder
        timestamp_ids: Optional[torch.LongTensor] = None,  # [B, num_images] for Flex encoder
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ):

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # 1. Extract Image Features
        if pixel_values is not None:
            # DINOv3 forward pass - returns last_hidden_state (B*num_images, seq_len, hidden)
            # Use embedding layer dtype to ensure compatibility with QLoRA/mixed precision
            target_dtype = self.get_input_embeddings().weight.dtype
            vision_outputs = self.vision_tower(pixel_values.to(target_dtype), output_hidden_states=True)
            image_embeds = vision_outputs.last_hidden_state
            
            # 2. Flex Scene Encoding (if enabled)
            if self.flex_scene_encoder is not None and camera_ids is not None:
                # Reshape: [B*num_images, N, D] -> [B, num_images, N, D]
                B = camera_ids.shape[0]
                num_images = camera_ids.shape[1]
                N = image_embeds.shape[1]
                D = image_embeds.shape[2]
                
                image_embeds = image_embeds.view(B, num_images, N, D)
                
                # Encode scene: [B, num_images, N, D] -> [B, K, D]
                image_embeds = self.flex_scene_encoder(
                    image_embeds, 
                    camera_ids.to(image_embeds.device), 
                    timestamp_ids.to(image_embeds.device)
                )

            # Project to LLM space
            image_embeds = self.projector(image_embeds)

            image_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds.to(inputs_embeds.dtype))
        else:
            image_embeds = None

        # 2b. Fuse Trajectory History Embeddings
        ego_history_xyz = kwargs.get("ego_history_xyz")
        ego_history_rot = kwargs.get("ego_history_rot")
        if ego_history_xyz is not None and ego_history_rot is not None:
            inputs_embeds = self.fuse_traj_embeddings(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                ego_history_xyz=ego_history_xyz,
                ego_history_rot=ego_history_rot,
            )

        # 3. Pass to LLM with output_hidden_states and output_attentions
        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        loss_output = None
        
        # Calculate Causal LM Loss
        lm_loss = None
        if labels is not None:
            lm_loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)
            
        # Trajectory Prediction and Loss
        ego_future_xyz = kwargs.get("ego_future_xyz")
        ego_future_rot = kwargs.get("ego_future_rot")
        
        traj_output = None
        xyz_loss, rot_loss = None, None
        
        # Only predict trajectory if we have labels or are explicitly asked (implied by having labels in train time)
        # Note: In inference we usually call predict_future_trajectory manually
        if self.action_head is not None and ((ego_future_xyz is not None and ego_future_rot is not None) or (labels is not None and "ego_future_xyz" in kwargs)):
             # Use full hidden states for action head cross-attention
             vlm_context = hidden_states  # [B, S, hidden_size]
             
             # Create memory_key_padding_mask for action head (True = padded/ignored)
             memory_key_padding_mask = None
             if attention_mask is not None:
                 memory_key_padding_mask = ~attention_mask.bool()  # Invert: 0 -> True (pad), 1 -> False (keep)
                 
             # Phase 2: Extract VQ-VAE indices from input_ids and decode
             base_traj = None
             if hasattr(self, "vqvae_tokenizer") and self.vqvae_tokenizer is not None and input_ids is not None:
                 B = input_ids.shape[0]
                 start_id = self.config.traj_token_ids["future_start"]
                 traj_start_idx = self.config.traj_token_start_idx
                 
                 # Prepare indices tensor [B, 8]
                 vqvae_indices = torch.zeros(B, 8, dtype=torch.long, device=input_ids.device)
                 found_all = True
                 
                 for b in range(B):
                    matches = (input_ids[b] == start_id).nonzero(as_tuple=True)[0]
                    if len(matches) > 0:
                        start_pos = matches[-1].item()
                        for i in range(8):
                            if start_pos + 1 + i < input_ids.shape[1]:
                                token = input_ids[b, start_pos + 1 + i].item()
                                if traj_start_idx <= token < traj_start_idx + getattr(self.config, 'traj_vocab_size', 768):
                                    vqvae_indices[b, i] = token - traj_start_idx
                                else:
                                    found_all = False
                                    break
                    else:
                        found_all = False
                 
                 if found_all:
                     with torch.no_grad():
                         # vqvae_indices is [B, 8]
                         # Decode returns [B, 64, 5] (x, y, z, sin, cos) natively inside vqvae_tokenizer
                         base_traj = self.vqvae_tokenizer.decode(vqvae_indices).to(vlm_context.dtype).to(vlm_context.device)
             
             if base_traj is not None:
                 traj_output = self.predict_future_trajectory(
                     vlm_context,
                     base_traj=base_traj,
                     normalize_rot=True,
                     attention_mask=memory_key_padding_mask,
                 )
             else:
                 # Fallback if VQ-VAE extraction failed or not available
                 print("Warning: VQ-VAE base trajectory could not be extracted. ActionChunkingHead requires base_traj.")
             
        # Compute Total Loss
        if lm_loss is not None or (traj_output is not None):
            pred_xyz = traj_output["xyz"] if traj_output else None
            pred_rot = traj_output["rot2d"] if traj_output else None
            
            loss_output = self.loss_calculator(
                text_loss=lm_loss,
                pred_xyz=pred_xyz,
                target_xyz=ego_future_xyz,
                pred_rot=pred_rot,
                target_rot=ego_future_rot
            )
            loss = loss_output.total_loss


        return DFQVLAOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None,
            text_loss=loss_output.text_loss if loss_output is not None else None,
            xyz_loss=loss_output.xyz_loss if loss_output is not None else None,
            rot_loss=loss_output.rot_loss if loss_output is not None else None,
        )
    
    def predict_future_trajectory(
        self,
        vlm_context: torch.Tensor,
        base_traj: torch.Tensor,
        normalize_rot: bool = True,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Predict future trajectory from VLM context and coarse base representation.
        
        This method uses the action head to predict delta waypoints via cross-attention
        to a concatenated memory of VLM context and the VQ-VAE decoded base trajectory.
        
        Args:
            vlm_context: VLM hidden states. Shape: [B, S, hidden_size].
            base_traj: coarse base trajectory from VQ-VAE. Shape: [B, 64, 5].
            normalize_rot: If True, normalize the 2D rotation representation to unit circle.
            attention_mask: Optional mask for padded positions. Shape: [B, S].
                True indicates padded (ignored) positions.
            
        Returns:
            Dictionary containing:
                - "xyz": Predicted XYZ positions [B, num_queries, 3]
                - "rot2d": 2D rotation representation [B, num_queries, 2]
        """
        return self.action_head(
            vlm_context,
            base_traj=base_traj,
            normalize_rot=normalize_rot,
            memory_key_padding_mask=attention_mask,
        )



    # Required for generation (model.generate)
    def prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            inputs_embeds=None,
            cache_position=None,
            position_ids=None,
            use_cache=True,
            pixel_values=None,
            **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            use_cache=use_cache,
            **kwargs,
        )

        if cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        return model_inputs


__all__ = [
    "DFQVLAForConditionalGeneration"
]
