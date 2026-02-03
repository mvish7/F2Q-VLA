from typing import Any, Dict, List, Optional
from PIL import Image
import os
import torch
from ..configs.configs import DataConfig

class DataCollator:
    """Collator that encodes text and image pairs for VLM training."""
    
    def __init__(self, processor: Any, image_token_id: int, config: DataConfig, use_flex: bool = False):
        self.processor = processor
        self.image_token_id = image_token_id
        self.config = config
        self.use_flex = use_flex

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 'examples' is a list of raw dataset samples now (dictionaries)
        
        # 1. Format text using the formatter
        from .dataset_formatter import format_vla_data
        
        # Convert raw samples to conversation format
        formatted_examples = [format_vla_data(sample, use_flex=self.use_flex) for sample in examples]
        
        # Apply chat template
        texts = [
            self.processor.apply_chat_template(conv, tokenize=False) 
            for conv in formatted_examples
        ]

        # 2. Process images
        flat_images = []
        for idx, sample in enumerate(examples):
            if self.use_flex:
                # Flex mode: Load ALL images from image_paths regardless of placeholder count
                # The Flex encoder needs all 16 images (4 cameras × 4 timestamps)
                if "image_paths" in sample:
                    for cam_name, paths in sample["image_paths"].items():
                        for path in paths:
                            if self.config.image_base_path and not os.path.isabs(path):
                                path = os.path.join(self.config.image_base_path, path)
                            try:
                                image = Image.open(path).convert("RGB")
                                flat_images.append(image)
                            except Exception as e:
                                print(f"Error loading image {path}: {e}")
                                flat_images.append(Image.new('RGB', (self.config.image_size_width, self.config.image_size_height), (0, 0, 0)))
            else:
                # Legacy mode: Load images based on placeholders in conversation
                conv = formatted_examples[idx]
                user_content = conv[1]["content"]
                
                for item in user_content:
                    if item["type"] == "image":
                        image_path = item["image"]
                        if self.config.image_base_path and not os.path.isabs(image_path):
                            image_path = os.path.join(self.config.image_base_path, image_path)
                        try:
                            image = Image.open(image_path).convert("RGB")
                            flat_images.append(image)
                        except Exception as e:
                            print(f"Error loading image {image_path}: {e}")
                            flat_images.append(Image.new('RGB', (self.config.image_size_width, self.config.image_size_height), (0, 0, 0)))

        # 3. Tokenize Text & Process Images
        # Processor expects a flat list of images corresponding to <|image_pad|> tokens in sequence
        batch = self.processor(
            text=texts, 
            images=flat_images if flat_images else None, 
            return_tensors="pt", 
            padding=True,
            crop_size={"height": self.config.image_size_height, "width": self.config.image_size_width}
        )
        
        # Close PIL images to free memory immediately after processing
        for img in flat_images:
            img.close()
        del flat_images
        
        # 4. Prepare Labels for Causal LM
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        labels[labels == self.image_token_id] = -100
        
        # Mask trajectory history tokens (they are inputs, not targets)
        traj_tokens_to_mask = [
            "<|traj_history|>",
            "<|traj_history_start|>",
            "<|traj_history_end|>",
        ]
        for token in traj_tokens_to_mask:
            token_id = self.processor.tokenizer.convert_tokens_to_ids(token)
            if token_id != self.processor.tokenizer.unk_token_id:  # Only mask if token exists
                labels[labels == token_id] = -100
        
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
        
        # 6. Generate camera_ids and timestamp_ids for Flex Scene Encoder
        # Order: for each camera -> for each timestamp
        # camera_ids:    [0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3]
        # timestamp_ids: [0,1,2,3, 0,1,2,3, 0,1,2,3, 0,1,2,3]
        num_cameras = getattr(self.config, "num_cameras", 4)
        num_timestamps = getattr(self.config, "num_timestamps", 4)
        
        camera_ids = []
        timestamp_ids = []
        for cam_idx in range(num_cameras):
            for ts_idx in range(num_timestamps):
                camera_ids.append(cam_idx)
                timestamp_ids.append(ts_idx)
        
        batch_size = len(examples)
        batch["camera_ids"] = torch.tensor([camera_ids] * batch_size, dtype=torch.long)
        batch["timestamp_ids"] = torch.tensor([timestamp_ids] * batch_size, dtype=torch.long)

        return batch
