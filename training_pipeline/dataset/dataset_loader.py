import random
from datasets import load_from_disk, Dataset
from typing import Any
import physical_ai_av
from ..configs.configs import DataConfig, ModelConfig
from .dataset_formatter import format_vla_data
from .data_collator import DataCollator
from .data_extractor import get_egomotion_for_curr_t


class DatasetLoader:
    def __init__(self, config: DataConfig, processor: Any, model_config: ModelConfig = None):
        self.config = config
        self.processor = processor
        self.model_config = model_config
        self.local_avdi = physical_ai_av.PhysicalAIAVDatasetInterface(cache_dir=config.data_base_path)
        self.camera_features = [
            self.local_avdi.features.CAMERA.CAMERA_CROSS_LEFT_120FOV,
            self.local_avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV,
            self.local_avdi.features.CAMERA.CAMERA_CROSS_RIGHT_120FOV,
            self.local_avdi.features.CAMERA.CAMERA_FRONT_TELE_30FOV,
        ]

    def load_dataset(self) -> tuple[list[dict], list[dict]]:
        """Load and process the dataset."""
        dataset = load_from_disk(self.config.dataset_path)

        # Expand into list of dicts (can't use Dataset.map because camera
        # objects are not Arrow-serializable)
        samples = self._expand_dataset(dataset)

        # Shuffle and split
        random.seed(42)
        random.shuffle(samples)
        split_idx = int(len(samples) * (1 - self.config.test_split_ratio))
        train_dataset = samples[:split_idx]
        test_dataset = samples[split_idx:]

        return train_dataset, test_dataset

    def _expand_dataset(self, dataset: Dataset) -> list[dict]:
        """Expand each sample with egomotion and camera data.

        Returns a list of plain dicts since camera objects (SeekVideoReader)
        are not Arrow-serializable and cannot live in a HF Dataset.
        """
        cached_clip_id = None
        cached_egomotion = None
        cached_cameras = None
        samples = []

        for i in range(len(dataset)):
            sample = dataset[i]  # returns a fresh dict copy
            clip_id = sample["clip_id"]

            # Fetch clip-level features once per clip (single-entry cache)
            if clip_id != cached_clip_id:
                cached_clip_id = clip_id
                cached_egomotion = self.local_avdi.get_clip_feature(
                    clip_id,
                    self.local_avdi.features.LABELS.EGOMOTION,
                )
                cached_cameras = [
                    self.local_avdi.get_clip_feature(clip_id, cam)
                    for cam in self.camera_features
                ]

            ego_history_xyz, ego_history_rot, ego_future_xyz, ego_future_rot = \
                get_egomotion_for_curr_t(cached_egomotion, clip_id, sample["t_curr"], self.config)

            sample["ego_history_xyz"] = ego_history_xyz
            sample["ego_history_rot"] = ego_history_rot
            sample["ego_future_xyz"] = ego_future_xyz
            sample["ego_future_rot"] = ego_future_rot
            sample["camera"] = cached_cameras
            samples.append(sample)

        return samples

    def get_collator(self) -> DataCollator:
        """Get the data collator initialized with processor and config."""
        # Detect image token ID
        # Common convention for some VLMs
        if "<|image_pad|>" in self.processor.tokenizer.additional_special_tokens:
            image_token_id = self.processor.tokenizer.additional_special_tokens_ids[
                self.processor.tokenizer.additional_special_tokens.index("<|image_pad|>")
            ]
        elif hasattr(self.processor.tokenizer, "image_token_id") and self.processor.tokenizer.image_token_id is not None:
             image_token_id = self.processor.tokenizer.image_token_id

        # Check if processor is configured for flex scene encoder
        use_flex = getattr(self.processor, 'use_flex_scene_encoder', False)

        # Create VQ-VAE tokenizer for trajectory tokenization (runs on CPU dynamically)
        vqvae_tokenizer = None
        if self.model_config and getattr(self.model_config, "vqvae_checkpoint_path", None):
            import sys
            import os
            from pathlib import Path
            vla_root = str(Path(__file__).resolve().parents[3])
            if vla_root not in sys.path:
                sys.path.insert(0, vla_root)
                
            from dfq_vla.vqvae_tokenizer import VQVAETrajectoryTokenizer
            
            vqvae_tokenizer = VQVAETrajectoryTokenizer(
                checkpoint_path=self.model_config.vqvae_checkpoint_path,
                num_embeddings=getattr(self.model_config, "vqvae_num_embeddings", 768),
                hidden_dim=getattr(self.model_config, "vqvae_hidden_dim", 256),
                embedding_dim=getattr(self.model_config, "vqvae_embedding_dim", 256),
            )

        return DataCollator(self.processor, image_token_id, self.config, use_flex=use_flex, vqvae_tokenizer=vqvae_tokenizer)