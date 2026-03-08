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
        self._micro_step = 0
        self._nan_detected = False
        
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
        """Override training_step with NaN gradient detection."""
        # Aggressive memory cleanup before processing
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        self._micro_step += 1
        
        # Call parent training_step (forward + backward)
        loss = super().training_step(model, inputs, num_items_in_batch)
        
        # Check gradient health after EVERY micro-batch backward (only report first time)
        if not self._nan_detected:
            nan_grad_groups = {}
            for name, p in model.named_parameters():
                if p.requires_grad and p.grad is not None and torch.isnan(p.grad).any():
                    # Group by model component
                    for key in ["vision_tower", "language_model", "action_head", 
                                "flex_scene_encoder", "projector", "embed_tokens", "lm_head"]:
                        if key in name:
                            group = key
                            break
                    else:
                        group = "other"
                    
                    if group not in nan_grad_groups:
                        nan_grad_groups[group] = []
                    nan_grad_groups[group].append(name)
            
            if nan_grad_groups:
                self._nan_detected = True
                step = getattr(self.state, 'global_step', '?')
                print(f"\n🔴 NaN gradients first detected at micro_step={self._micro_step}, global_step={step}")
                for group, params in sorted(nan_grad_groups.items()):
                    print(f"  [{group}] {len(params)} params with NaN grads (first: {params[0]})")
                
                # Check which weights are already NaN
                nan_weights = [n for n, p in model.named_parameters() if torch.isnan(p.data).any()]
                if nan_weights:
                    print(f"  💀 {len(nan_weights)} params have NaN weights already")
                else:
                    print(f"  ✅ All weights still finite — NaN is only in gradients from this backward pass")
        
        return loss
        
    def save_model(self, output_dir=None, _internal_call=False):
        """Override save_model to also save our custom config."""
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        super().save_model(output_dir, _internal_call)
        
        # Save our custom config to the output directory
        config_path = os.path.join(output_dir, "vlm_config.yaml")
        save_config(self.vlm_config, config_path)
