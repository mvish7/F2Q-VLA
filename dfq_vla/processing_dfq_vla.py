import math
from typing import Optional

import torch
from transformers import ProcessorMixin, AutoTokenizer, AutoImageProcessor
from transformers.image_processing_base import BatchFeature
from transformers.processing_utils import ImagesKwargs, ProcessingKwargs, Unpack


class Dinov3ImagesKwargs(ImagesKwargs):
    size: Optional[dict]


class DFQVLAProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: Dinov3ImagesKwargs
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_token_type_ids": False,
        },
    }


class DFQVLAProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "CLIPImageProcessor"
    tokenizer_class = "AutoTokenizer"
    
    # Trajectory token constants (history only — future is predicted as text)
    TRAJ_TOKEN = {
        "history": "<|traj_history|>",
        "history_start": "<|traj_history_start|>",
        "history_end": "<|traj_history_end|>",
    }

    # Action reasoning delimiter tokens
    ACTION_REASONING_TOKEN = {
        "start": "<|action_reasoning_start|>",
        "end": "<|action_reasoning_end|>",
    }

    # Action head tokens (for learned action embeddings)
    ACTION_TOKEN = {
        "start": "<|action_start|>",
        "end": "<|action_end|>",
        "action": "<|action|>",
    }

    def __init__(self, image_processor, tokenizer, chat_template, vision_config=None,
                 use_flex_scene_encoder=False, num_scene_tokens=512, **kwargs):
        super().__init__(image_processor, tokenizer, chat_template=chat_template)
        self.vision_config = vision_config
        self.image_token = "<|image_pad|>" if not hasattr(tokenizer, "image_token") else tokenizer.image_token
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.image_token_id = (
            tokenizer.image_token_id
            if getattr(tokenizer, "image_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.image_token)
        )
        self.vision_start_token = (
            "<|vision_start|>" if not hasattr(tokenizer, "vision_start_token") else tokenizer.vision_start_token
        )
        self.vision_end_token = (
            "<|vision_end|>" if not hasattr(tokenizer, "vision_end_token") else tokenizer.vision_end_token
        )
        self.vision_start_token_id = (
            tokenizer.vision_start_token_id
            if getattr(tokenizer, "vision_start_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.vision_start_token)
        )
        self.vision_end_token_id = (
            tokenizer.vision_end_token_id
            if getattr(tokenizer, "vision_end_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.vision_end_token)
        )
        
        # Flex Scene Encoder config
        self.use_flex_scene_encoder = use_flex_scene_encoder
        self.num_scene_tokens = num_scene_tokens
        
        # Add all custom special tokens to tokenizer
        self._add_special_tokens(tokenizer)
    
    def _add_special_tokens(self, tokenizer):
        """Add all custom special tokens to the tokenizer.
        
        Registers: trajectory history tokens, action reasoning delimiters,
        and vision tokens (for LFM2.5 which lacks them natively).
        """
        # Collect all special tokens
        special_tokens = list(self.TRAJ_TOKEN.values())
        special_tokens += list(self.ACTION_REASONING_TOKEN.values())
        special_tokens += list(self.ACTION_TOKEN.values())
        
        # Vision tokens — add if not already in tokenizer vocab
        vision_tokens = ["<|image_pad|>", "<|vision_start|>", "<|vision_end|>"]
        existing_vocab = set(tokenizer.get_vocab().keys())
        for vt in vision_tokens:
            if vt not in existing_vocab:
                special_tokens.append(vt)
        
        tokenizer.add_tokens(special_tokens, special_tokens=True)
        
        # Store trajectory token IDs
        tokenizer.traj_token_ids = {
            k: tokenizer.convert_tokens_to_ids(v) for k, v in self.TRAJ_TOKEN.items()
        }
        self.traj_token_ids = tokenizer.traj_token_ids
        
        # Store action reasoning token IDs
        tokenizer.action_reasoning_token_ids = {
            k: tokenizer.convert_tokens_to_ids(v) for k, v in self.ACTION_REASONING_TOKEN.items()
        }
        self.action_reasoning_token_ids = tokenizer.action_reasoning_token_ids
        
        # Store action head token IDs
        tokenizer.action_token_ids = {
            k: tokenizer.convert_tokens_to_ids(v) for k, v in self.ACTION_TOKEN.items()
        }
        self.action_token_ids = tokenizer.action_token_ids

    def _calculate_num_image_tokens(self, image_height, image_width):
        """Calculate number of tokens based on image dimensions.
        
        TIPSv2 formula: (H // patch_size) * (W // patch_size) + 1 (CLS token)
        Falls back to a default if vision_config is not available.
        """
        if self.vision_config is not None:
            patch_size = self.vision_config.patch_size
        else:
            patch_size = 14  # TIPSv2 default patch size
        return (image_height // patch_size) * (image_width // patch_size) + 1

    def __call__(self, text=None, images=None, return_tensors=None, **kwargs: Unpack[DFQVLAProcessorKwargs]):
        output_kwargs = self._merge_kwargs(
            DFQVLAProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if text is None and images is None:
            raise ValueError("You have to specify either text or images.")

        # Process Images
        image_inputs = {}
        image_sizes = []
        if images is not None:
            image_inputs = self.image_processor(images, return_tensors=return_tensors, **output_kwargs["images_kwargs"])

            # Get image sizes from pixel_values tensor (B, 3, H, W)
            pixel_values = image_inputs.get("pixel_values")
            if pixel_values is not None:
                # pixel_values shape is (B, 3, H, W)
                for i in range(pixel_values.shape[0]):
                    h, w = pixel_values.shape[2], pixel_values.shape[3]
                    image_sizes.append((h, w))

        if not isinstance(text, list):
            text = [text]
        text = text.copy()  # below lines change text in-place

        index = 0
        for i in range(len(text)):
            while self.image_token in text[i]:
                if self.use_flex_scene_encoder:
                    # Flex mode: Single image placeholder expands to K scene tokens
                    # The Flex encoder outputs num_scene_tokens tokens
                    num_image_tokens = self.num_scene_tokens
                else:
                    # Legacy mode: Calculate dynamic token count based on image dimensions
                    if index < len(image_sizes):
                        img_h, img_w = image_sizes[index]
                        num_image_tokens = self._calculate_num_image_tokens(img_h, img_w)
                    else:
                        # Fallback for when image sizes not available
                        num_image_tokens = 256  # default for 1024x1024
                
                text[i] = text[i].replace(self.image_token, "<|placeholder|>" * num_image_tokens, 1)
                index += 1
            text[i] = text[i].replace("<|placeholder|>", self.image_token)

        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type="pt")

    # Handle batch decoding
    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)
