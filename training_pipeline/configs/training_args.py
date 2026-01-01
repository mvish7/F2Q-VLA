from trl import SFTConfig
from typing import Optional, Any
from .configs import VLMTrainingConfig

def get_training_args(config: VLMTrainingConfig, data_collator: Any) -> SFTConfig:
    """
    Construct SFTConfig (TrainingArguments) from the VLM training configuration.
    
    Args:
        config: The main training configuration object.
        data_collator: The data collator instance, used to retrieve image_token_id.
        
    Returns:
        SFTConfig: The configuration object for SFTTrainer.
    """
    return SFTConfig(
        output_dir=config.training.output_dir,
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay, # Default or add to config
        logging_steps=config.training.logging_steps,
        save_strategy=config.training.save_strategy,
        save_steps=config.training.save_steps,
        eval_strategy=config.training.eval_strategy,
        eval_steps=config.training.eval_steps,
        report_to=config.training.report_to,
        bf16=config.training.bf16,
        max_grad_norm=config.training.max_grad_norm,
        warmup_ratio=config.training.warmup_ratio,
        gradient_checkpointing=config.training.gradient_checkpointing,
        dataloader_num_workers=config.data.dataloader_num_workers,
        dataloader_pin_memory=True,
        max_seq_length=config.data.max_len,
        dataset_kwargs={"skip_prepare_dataset": True}, # We prepare it manually
        dataset_text_field="text", # Dummy field as we use custom collator
        remove_unused_columns=False, # Essential for custom VLM collators often
    )
