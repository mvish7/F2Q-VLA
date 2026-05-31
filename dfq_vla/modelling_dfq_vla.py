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
from dfq_vla.flex_scene_encoder import FlexSceneEncoder, create_flex_scene_encoder


class DFQVLAProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Vision hidden size -> LLM hidden size
        self.linear_1 = nn.Linear(config.vision_hidden_size, config.text_config.hidden_size, bias=False)
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
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None


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
    _tied_weights_keys = {"lm_head.weight": "language_model.embed_tokens.weight"}
    config_class = DFQVLAConfig
    accepts_loss_kwargs = False
    _supports_flash_attn_2 = True

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # 1. Load Vision Encoder (TIPSv2 — vision branch only)
        # TIPSv2 is a dual-encoder (vision + text); we only need the vision side.
        # Setting text_encoder = None deregisters its params from the module tree.
        self.vision_tower = AutoModel.from_config(config.vision_config, trust_remote_code=True)
        if hasattr(self.vision_tower, "text_encoder"):
            self.vision_tower.text_encoder = None

        # 2. Load LLM (LFM2.5)
        self.language_model = AutoModel.from_config(config.text_config)

        # 3. Projector (3072 -> 1024)
        self.projector = DFQVLAProjector(config)

        # LM head
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        
        # 4. Initialize trajectory history projector (optional)
        if getattr(config, 'include_traj_projector', True):
            self._initialize_traj_projector(config)
        else:
            self.traj_projector = None
        
        # 5. Initialize Flex Scene Encoder (optional)
        self._initialize_flex_scene_encoder(config)

        # 6. Tie weights if necessary (standard HF practice)
        self.post_init()
    
    def _initialize_traj_projector(self, config):
        """Initialize MLP projector for trajectory history encoding."""
        traj_input_dim = getattr(config, "traj_input_dim", 5)
        self.traj_projector = TrajHistProjector(
            input_dim=traj_input_dim,
            hidden_size=config.hidden_size,
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
            vision_outputs = self.vision_tower(pixel_values.to(target_dtype))
            image_embeds = vision_outputs.image_features.patch_tokens
            
            # # Strip CLS + register tokens, keep only patch tokens
            # # DINOv3 output layout: [CLS, reg_0, ..., reg_N, patch_0, ..., patch_P]
            # num_register = getattr(self.config.vision_config, "num_register_tokens", 0)
            # num_prefix = 1 + num_register  # 1 for CLS
            # image_embeds = image_embeds[:, num_prefix:, :]
            
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

        # 2b. Fuse Trajectory History Embeddings (only when traj_projector is present)
        if self.traj_projector is not None:
            ego_history_xyz = kwargs.get("ego_history_xyz")
            ego_history_rot = kwargs.get("ego_history_rot")
            if ego_history_xyz is not None and ego_history_rot is not None:
                inputs_embeds = self.fuse_traj_embeddings(
                    input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    ego_history_xyz=ego_history_xyz,
                    ego_history_rot=ego_history_rot,
                )

        # 3. Pass to LLM
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

        # Calculate Causal LM Loss (text-only — action reasoning is predicted as text tokens)
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

        return DFQVLAOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None,
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

        # During autoregressive decoding (past_key_values is populated), the model only
        # receives the single new token — no image placeholder tokens exist in input_ids.
        # Passing pixel_values here would cause a features/tokens mismatch in get_placeholder_mask.
        # Only process images during the prefill step (no past_key_values yet).
        is_prefill = past_key_values is None
        if not is_prefill:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None
            # Trajectory placeholder tokens only exist in the prefill input_ids.
            # Skip traj fusion during decode steps.
            model_inputs.pop("ego_history_xyz", None)
            model_inputs.pop("ego_history_rot", None)

        return model_inputs


__all__ = [
    "DFQVLAForConditionalGeneration"
]
