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
            self.local_avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV,
            self.local_avdi.features.CAMERA.CAMERA_FRONT_TELE_30FOV,
        ]

    def load_dataset(self) -> tuple[list[dict], list[dict]]:
        """Load and process the dataset."""
        dataset = load_from_disk(self.config.dataset_path)
        dataset = dataset.select(range(100))

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
        samples = []

        for i in range(len(dataset)):
            sample = dataset[i]  # returns a fresh dict copy
            clip_id = sample["clip_id"]

            # Fetch clip-level egomotion once per clip (single-entry cache)
            if clip_id != cached_clip_id:
                cached_clip_id = clip_id
                cached_egomotion = self.local_avdi.get_clip_feature(
                    clip_id,
                    self.local_avdi.features.LABELS.EGOMOTION,
                )
                # NOTE: Camera features are NOT loaded here to avoid pinning
                # SeekVideoReader objects for every clip in RAM. They are
                # loaded lazily per batch in the DataCollator.

            ego_history_xyz, ego_history_rot, ego_future_xyz, ego_future_rot = \
                get_egomotion_for_curr_t(cached_egomotion, clip_id, sample["t_curr"], self.config)

            sample["ego_history_xyz"] = ego_history_xyz
            sample["ego_history_rot"] = ego_history_rot
            sample["ego_future_xyz"] = ego_future_xyz
            sample["ego_future_rot"] = ego_future_rot
            samples.append(sample)

        return samples

    def get_collator(self) -> DataCollator:
        """Get the data collator initialized with processor and config."""
        # Detect image token ID
        # Common convention for some VLMs
        vocab = self.processor.tokenizer.get_vocab()
        if "<|image_pad|>" in vocab:
            image_token_id = vocab["<|image_pad|>"]
        elif hasattr(self.processor.tokenizer, "image_token_id") and self.processor.tokenizer.image_token_id is not None:
             image_token_id = self.processor.tokenizer.image_token_id

        # Check if processor is configured for flex scene encoder
        use_flex = getattr(self.processor, 'use_flex_scene_encoder', False)

        # Resolve include_traj_projector: None → default True
        include_traj = getattr(self.model_config, "include_traj_projector", None) if self.model_config else None
        if include_traj is None:
            include_traj = True

        return DataCollator(
            self.processor, image_token_id, self.config,
            use_flex=use_flex,
            camera_features=self.camera_features,
            include_traj_projector=include_traj,
        )