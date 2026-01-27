import os
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments
from ..configs.configs import VLMTrainingConfig
from ..configs.config_utils import save_config

class VLMTrainer(SFTTrainer):
    """
    Custom VLM Trainer that extends SFTTrainer.
    """
    
    def __init__(
        self,
        config: VLMTrainingConfig,
        model,
        args: SFTConfig,
        train_dataset,
        eval_dataset,
        data_collator,
        processing_class,
        peft_config=None
    ):
        self.vlm_config = config
        
        # Initialize parent SFTTrainer
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            processing_class=processing_class,
            peft_config=peft_config
        )
        
    def save_model(self, output_dir=None, _internal_call=False):
        """Override save_model to also save our custom config."""
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        super().save_model(output_dir, _internal_call)
        
        # Save our custom config to the output directory
        config_path = os.path.join(output_dir, "vlm_config.yaml")
        save_config(self.vlm_config, config_path)
