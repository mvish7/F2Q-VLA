from dataclasses import dataclass
from typing import Optional, Union

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
from f2q_vla.configuration_f2q_vla import F2QVLAConfig


VISION_MODEL_ID = "kevin510/fast-vit-hd"


class F2QVLAProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        # FastViT hidden size (3072) -> Qwen hidden size (1024)
        self.linear_1 = nn.Linear(config.vision_hidden_size, config.text_config.hidden_size, bias=False)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=False)

    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states

@dataclass
class F2QVLAOutputWithPast(ModelOutput):
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


class F2QVLAPretrainedModel(PreTrainedModel):
    config: F2QVLAConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = []
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {}


class F2QVLAForConditionalGeneration(F2QVLAPretrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = {}
    _tied_weights_keys = ["lm_head.weight"]
    config_class = F2QVLAConfig
    accepts_loss_kwargs = False
    _supports_flash_attn_2 = True

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # 1. Load Vision Encoder (FastViT-HD)
        # We use from_config to initialize empty structure, weights loaded later
        self.vision_tower = AutoModel.from_config(config.vision_config, trust_remote_code=True)

        # 2. Load LLM (Qwen3)
        self.language_model = AutoModel.from_config(config.text_config)

        # 3. Projector (3072 -> 1024)
        self.projector = F2QVLAProjector(config)

        # LM head
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        # 4. Tie weights if necessary (standard HF practice)
        self.post_init()

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
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ):

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # 1. Extract Image Features
        if pixel_values is not None:
            # FastViT forward pass - returns (B, num_patches, 3072)
            image_embeds = self.vision_tower.forward_images(pixel_values).to(torch.bfloat16)

            # Project to LLM space (3072 -> 1024)
            image_embeds = self.projector(image_embeds)

            image_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        else:
            image_embeds = None

        # 3. Pass to LLM
        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

        return F2QVLAOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
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
    "F2QVLAForConditionalGeneration"
]
