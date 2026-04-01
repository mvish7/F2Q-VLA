from typing import Any, Optional
from PIL import Image
import os
import torch
from ..configs.configs import DataConfig

class DataCollator:
    """Collator that encodes text and image pairs for VLM training."""
    
    def __init__(self, processor: Any, image_token_id: int, config: DataConfig, use_flex: bool = False, vqvae_tokenizer=None):
        self.processor = processor
        self.image_token_id = image_token_id
        self.config = config
        self.use_flex = use_flex
        self.vqvae_tokenizer = vqvae_tokenizer
        self._setup_assistant_masking()
    
    def _setup_assistant_masking(self):
        """Pre-compute token IDs for masking everything before assistant content."""
        tokenizer = self.processor.tokenizer
        self.im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        # "assistant\n" may tokenize to 1-2 tokens; measure once
        self._assistant_header_len = 1 + len(
            tokenizer.encode("assistant\n", add_special_tokens=False)
        )  # 1 for <|im_start|> + len("assistant\n" tokens)
    
    def _extract_traj_data(self, curr_sample: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    
        # History
        h_xyz = torch.tensor(curr_sample["ego_history_xyz"], dtype=torch.float32)
        h_rot = torch.tensor(curr_sample["ego_history_rot"], dtype=torch.float32)
        
        # Convert 3x3 rotation matrices to 2D continuous yaw
        yaw_h = torch.atan2(h_rot[..., 1, 0], h_rot[..., 0, 0])
        h_rot = torch.stack([torch.cos(yaw_h), torch.sin(yaw_h)], dim=-1)
        
        # Squeeze dimensions: [1, 1, T, 3] -> [T, 3]
        while h_xyz.ndim > 2: h_xyz = h_xyz.squeeze(0)
        while h_rot.ndim > 2: h_rot = h_rot.squeeze(0)
        
        # Future (Labels)
        f_xyz = torch.tensor(curr_sample["ego_future_xyz"], dtype=torch.float32)
        f_rot = torch.tensor(curr_sample["ego_future_rot"], dtype=torch.float32)
        
        # Convert 3x3 rotation matrices to 2D continuous yaw
        yaw_f = torch.atan2(f_rot[..., 1, 0], f_rot[..., 0, 0])
        f_rot = torch.stack([torch.cos(yaw_f), torch.sin(yaw_f)], dim=-1)
        
        # Squeeze: [1, 1, T, 3] -> [T, 3]
        while f_xyz.ndim > 2: f_xyz = f_xyz.squeeze(0)
        while f_rot.ndim > 2: f_rot = f_rot.squeeze(0)
        
        return h_xyz, h_rot, f_xyz, f_rot
        

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        # 'examples' is a list of raw dataset samples now (dictionaries)
        
        from .dataset_formatter import format_vla_data

        # placeholders for Trajectory Data (History & Future)
        # History is needed for the model input via TrajHistProjector
        # Future is needed for loss calculation and action head
        
        ego_history_xyz_list = []
        ego_history_rot_list = []
        ego_future_xyz_list = []
        ego_future_rot_list = []

        formatted_examples = []

        for sample in examples:
            curr_hist_xyz, curr_hist_rot, curr_fut_xyz, curr_fut_rot = self._extract_traj_data(sample)
            
            # Encode future trajectory → 8 VQ-VAE codebook indices
            vqvae_indices = self.vqvae_tokenizer.encode(curr_fut_xyz, curr_fut_rot)
            
            formatted_sample = format_vla_data(sample, vqvae_indices, use_flex=self.use_flex)
            formatted_examples.append(formatted_sample)

            ego_history_xyz_list.append(curr_hist_xyz)
            ego_history_rot_list.append(curr_hist_rot)
            ego_future_xyz_list.append(curr_fut_xyz)
            ego_future_rot_list.append(curr_fut_rot)
        
        
        # 1. Format text using the formatter
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
            size={"height": self.config.image_size_height, "width": self.config.image_size_width}
        )
        
        # Close PIL images to free memory immediately after processing
        for img in flat_images:
            img.close()
        del flat_images
        
        # 4. Prepare Labels for Causal LM
        # Mask everything before assistant content so the model only learns
        # to predict the assistant response (traj tokens + delimiters + <|im_end|>).
        labels = batch["input_ids"].clone()
        
        for i in range(labels.shape[0]):
            # Find last <|im_start|> — always the assistant turn
            positions = (labels[i] == self.im_start_id).nonzero(as_tuple=True)[0]
            if len(positions) > 0:
                # Mask everything up to and including "<|im_start|>assistant\n"
                labels[i, : positions[-1].item() + self._assistant_header_len] = -100
        
        # Mask pad tokens (right-padding)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        batch["labels"] = labels

        # Stack traj data
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
