from datasets import load_from_disk
from typing import Tuple, Any, Dict
from ..configs.configs import DataConfig
from .dataset_formatter import format_data, format_vla_data
from .data_collator import DataCollator

class DatasetLoader:
    def __init__(self, config: DataConfig, processor: Any):
        self.config = config
        self.processor = processor

    def load_dataset(self) -> Tuple[Any, Any]:
        """Load and process the dataset."""
        dataset = load_from_disk(self.config.dataset_path)
        
        # Split dataset
        dataset_dict = dataset.train_test_split(
            test_size=self.config.test_split_ratio, 
            seed=42, 
            shuffle=True
        )

        train_dataset = dataset_dict["train"].shuffle()
        test_dataset = dataset_dict["test"].shuffle()
        
        # Format datasets using the formatter
        # We NO LONGER map format_vla_data here because we need the raw sample in the DataCollator
        # to extract trajectory info. The formatting happens inside DataCollator.
        
        # Convert to list of dicts to be compatible with DataCollator expecting list
        # using select/indices or just iteration.
        # Since we shuffled, we can just return the dataset object if it implements __getitem__
        # But DataCollator expects a list.
        train_dataset = [sample for sample in train_dataset]
        test_dataset = [sample for sample in test_dataset]
        
        return train_dataset, test_dataset

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

        return DataCollator(self.processor, image_token_id, self.config, use_flex=use_flex)