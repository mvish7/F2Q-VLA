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
        # 'examples' is a list of raw dataset samples now (dictionaries)
        
        # 1. Format text using the formatter
        from .dataset_formatter import format_vla_data
        
        # Convert raw samples to conversation format
        formatted_examples = [format_vla_data(sample) for sample in examples]
        
        # Apply chat template
        texts = [
            self.processor.apply_chat_template(conv, tokenize=False) 
            for conv in formatted_examples
        ]

        # 2. Process images
        flat_images = []
        for idx, sample in enumerate(examples):
            # We look at the formatted conversation for this sample
            # The user message is at index 1
            conv = formatted_examples[idx]
            user_content = conv[1]["content"]
            
            for item in user_content:
                if item["type"] == "image":
                    image_path = item["image"]
                    # Load image
                    try:
                        image = Image.open(image_path).convert("RGB")
                        flat_images.append(image)
                    except Exception as e:
                        print(f"Error loading image {image_path}: {e}")
                        # If we fail to load an image, we MUST ensure the text doesn't have the placeholder
                        # But apply_chat_template already ran. 
                        # This is a potential misalignment risk if image load fails but token exists.
                        # Ideally we crash or substitute black image.
                        # Using black image for robustness:
                        flat_images.append(Image.new('RGB', (self.config.image_size_width, self.config.image_size_height), (0, 0, 0)))

        # 3. Tokenize Text & Process Images
        # Processor expects a flat list of images corresponding to <|image_pad|> tokens in sequence
        batch = self.processor(
            text=texts, 
            images=flat_images if flat_images else None, 
            return_tensors="pt", 
            padding=True,
            size={"height": self.config.image_size_height, "width": self.config.image_size_width}
        )
        
        # 4. Prepare Labels for Causal LM
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        labels[labels == self.image_token_id] = -100
        batch["labels"] = labels

        # 5. Extract and Stack Trajectory Data (History & Future)
        # History is needed for the model input (fusing)
        # Future is needed for loss calculation
        
        ego_history_xyz_list = []
        ego_history_rot_list = []
        ego_future_xyz_list = []
        ego_future_rot_list = []
        
        for sample in examples:
            # History
            h_xyz = torch.tensor(sample["ego_history_xyz"], dtype=torch.float32)
            h_rot = torch.tensor(sample["ego_history_rot"], dtype=torch.float32)
            # Squeeze dimensions: [1, 1, T, 3] -> [T, 3]
            while h_xyz.ndim > 2: h_xyz = h_xyz.squeeze(0)
            while h_rot.ndim > 3: h_rot = h_rot.squeeze(0)
            
            ego_history_xyz_list.append(h_xyz)
            ego_history_rot_list.append(h_rot)
            
            # Future (Labels)
            f_xyz = torch.tensor(sample["ego_future_xyz"], dtype=torch.float32)
            f_rot = torch.tensor(sample["ego_future_rot"], dtype=torch.float32)
            # Squeeze: [1, 1, T, 3] -> [T, 3]
            while f_xyz.ndim > 2: f_xyz = f_xyz.squeeze(0)
            while f_rot.ndim > 3: f_rot = f_rot.squeeze(0)
            
            ego_future_xyz_list.append(f_xyz)
            ego_future_rot_list.append(f_rot)
            
        # Stack
        batch["ego_history_xyz"] = torch.stack(ego_history_xyz_list)
        batch["ego_history_rot"] = torch.stack(ego_history_rot_list)
        batch["ego_future_xyz"] = torch.stack(ego_future_xyz_list)
        batch["ego_future_rot"] = torch.stack(ego_future_rot_list)

        return batch
