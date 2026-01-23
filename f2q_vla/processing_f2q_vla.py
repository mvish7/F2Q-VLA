import math
from typing import Optional

import torch
from transformers import ProcessorMixin, AutoTokenizer, AutoImageProcessor
from transformers.image_processing_base import BatchFeature
from transformers.processing_utils import ImagesKwargs, ProcessingKwargs, Unpack


class FastViTImagesKwargs(ImagesKwargs):
    size: Optional[dict]


class F2QVLAProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: FastViTImagesKwargs
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_token_type_ids": False,
        },
    }


class F2QVLAProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "CLIPImageProcessor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")
    
    # Trajectory token constants
    TRAJ_TOKEN = {
        "history": "<|traj_history|>",
        "history_start": "<|traj_history_start|>",
        "history_end": "<|traj_history_end|>",
    }

    def __init__(self, image_processor, tokenizer, chat_template, patch_size=64, traj_vocab_size=768, **kwargs):
        super().__init__(image_processor, tokenizer, chat_template=chat_template)
        self.patch_size = patch_size
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
        
        # Add trajectory tokens to tokenizer
        self.traj_vocab_size = traj_vocab_size
        self._add_trajectory_tokens(tokenizer, traj_vocab_size)
    
    def _add_trajectory_tokens(self, tokenizer, traj_vocab_size: int):
        """Add trajectory tokens to the tokenizer.
        
        Args:
            tokenizer: The tokenizer to modify.
            traj_vocab_size: Number of discrete trajectory tokens to add.
        """
        # Add discrete trajectory tokens (<i0> to <i{traj_vocab_size-1}>)
        discrete_tokens = [f"<i{v}>" for v in range(traj_vocab_size)]
        num_new_tokens = tokenizer.add_tokens(discrete_tokens)
        
        # Store trajectory token indices on tokenizer
        tokenizer.traj_token_start_idx = tokenizer.convert_tokens_to_ids("<i0>")
        tokenizer.traj_token_end_idx = tokenizer.convert_tokens_to_ids(f"<i{traj_vocab_size - 1}>")
        
        # Add special trajectory tokens
        special_tokens = list(self.TRAJ_TOKEN.values())
        tokenizer.add_tokens(special_tokens, special_tokens=True)
        
        # Store special token IDs on tokenizer
        tokenizer.traj_token_ids = {
            k: tokenizer.convert_tokens_to_ids(v) for k, v in self.TRAJ_TOKEN.items()
        }
        
        # Store on processor for easy access
        self.traj_token_start_idx = tokenizer.traj_token_start_idx
        self.traj_token_ids = tokenizer.traj_token_ids

    def _calculate_num_image_tokens(self, image_height, image_width):
        """Calculate number of tokens based on image dimensions and patch size.
        
        Formula: ceil(H/patch_size) * ceil(W/patch_size)
        For FastViT-HD with patch_size=64
        """
        return math.ceil(image_height / self.patch_size) * math.ceil(image_width / self.patch_size)

    def __call__(self, text=None, images=None, return_tensors=None, **kwargs: Unpack[F2QVLAProcessorKwargs]):
        output_kwargs = self._merge_kwargs(
            F2QVLAProcessorKwargs,
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
                # Calculate dynamic token count based on image dimensions
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
