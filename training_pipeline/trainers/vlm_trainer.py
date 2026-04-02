import os
import torch
import gc
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments
from ..configs.configs import VLMTrainingConfig
from ..configs.config_utils import save_config
from ..utils.param_groups import build_param_groups

class VLMTrainer(SFTTrainer):
    """
    Custom VLM Trainer that extends SFTTrainer.
    Supports per-module learning rate groups for staged training.
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
    
    def create_optimizer(self):
        """Override to build optimizer with per-module LR parameter groups.
        
        Uses build_param_groups to classify trainable params into
        default / LLM LoRA / Vision LoRA groups with distinct LRs.
        """
        if self.optimizer is not None:
            return self.optimizer
        
        param_groups = build_param_groups(self.model, self.vlm_config)
        
        # Resolve optimizer class from args.optim string
        optimizer_cls, optimizer_kwargs = SFTTrainer.get_optimizer_cls_and_kwargs(self.args, self.model)
        
        # Remove 'lr' from kwargs since each group has its own
        optimizer_kwargs.pop("lr", None)
        
        self.optimizer = optimizer_cls(param_groups, **optimizer_kwargs)
        return self.optimizer
        
    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override training_step with NaN gradient detection."""
        # Aggressive memory cleanup before processing
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Call parent training_step (forward + backward)
        loss = super().training_step(model, inputs, num_items_in_batch)
        
        return loss
        
    def save_model(self, output_dir=None, _internal_call=False):
        """Override save_model to also save our custom config."""
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        super().save_model(output_dir, _internal_call)
        
        # Save our custom config to the output directory
        config_path = os.path.join(output_dir, "vlm_config.yaml")
        save_config(self.vlm_config, config_path)
