import os
import torch
import gc
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
        
    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override training_step to aggressively clear CUDA cache before each step.
        
        This is needed because the model + activations + gradients consume nearly
        all GPU memory, leaving no room for the next batch's pixel_values to be
        transferred to GPU.
        """
        # Aggressive memory cleanup before processing
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Call parent training_step
        return super().training_step(model, inputs, num_items_in_batch)
        
    def save_model(self, output_dir=None, _internal_call=False):
        """Override save_model to also save our custom config."""
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        super().save_model(output_dir, _internal_call)
        
        # Save our custom config to the output directory
        config_path = os.path.join(output_dir, "vlm_config.yaml")
        save_config(self.vlm_config, config_path)

