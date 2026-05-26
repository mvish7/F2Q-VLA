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
        # lr_scheduler_type=config.training.lr_scheduler_type,
        # lr_scheduler_kwargs=config.training.lr_scheduler_kwargs,
        warmup_ratio=config.training.warmup_ratio,
        gradient_checkpointing=config.training.gradient_checkpointing,
        # use_reentrant=False is required when gradient checkpointing with frozen modules.
        # The default (use_reentrant=True) produces None gradients when layer inputs
        # don't have requires_grad=True, which breaks training with frozen LLM.
        gradient_checkpointing_kwargs={"use_reentrant": False} if config.training.gradient_checkpointing else None,
        dataloader_num_workers=config.data.dataloader_num_workers,
        dataloader_pin_memory=config.data.dataloader_pin_memory,
        # prefetch_factor must be None when num_workers=0
        dataloader_prefetch_factor=config.data.dataloader_prefetch_factor if config.data.dataloader_num_workers > 0 else None,
        max_seq_length=config.data.max_len,
        dataset_kwargs={"skip_prepare_dataset": True}, # We prepare it manually
        dataset_text_field="text", # Dummy field as we use custom collator
        remove_unused_columns=False, # Essential for custom VLM collators often
        use_liger_kernel=config.training.use_liger_kernel,  # Liger Kernel optimization
        torch_empty_cache_steps=config.training.torch_empty_cache_steps,  # Clear CUDA cache every N steps
        # optim=config.training.optim,  # Optimizer type,
        torch_compile=config.training.torch_compile,
        torch_compile_backend=config.training.torch_compile_backend,
        torch_compile_mode=config.training.torch_compile_mode,
    )
