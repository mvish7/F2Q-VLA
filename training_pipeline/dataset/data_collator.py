from typing import Any, Dict, List, Optional
from PIL import Image
import torch
from ..configs.configs import DataConfig

class DataCollator:
    """Collator that encodes text and image pairs for VLM training."""
    
    def __init__(self, processor: Any, image_token_id: int, config: DataConfig):
        self.processor = processor
        self.image_token_id = image_token_id
        self.config = config

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Apply chat template to format conversations
        texts = [
            self.processor.apply_chat_template(example, tokenize=False) 
            for example in examples
        ]

        # Process images
        image_inputs = []
        for example in examples:
            # Extract image path from the user message (index 1) which contains the image
            content = example[1]["content"]
            image_path = None
            for item in content:
                if item["type"] == "image":
                    image_path = item["image"]
                    break
            
            if image_path:
                image = Image.open(image_path).convert("RGB")
                image_inputs.append([image])
            else:
                # Handle cases with no image if necessary, though this pipeline expects VLM pairs
                # Placeholder handling or error could go here
                pass

        # Let processor handle tokenization
        batch = self.processor(
            text=texts, 
            images=image_inputs, 
            return_tensors="pt", 
            padding=True,
            size={"height": self.config.image_size_height, "width": self.config.image_size_width}
        )
        
        # Labels for causal LM training
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100  # Mask padding tokens
        
        # Mask image tokens
        labels[labels == self.image_token_id] = -100
        
        batch["labels"] = labels

        return batch
