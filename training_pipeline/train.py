import argparse
import os
import torch


from training_pipeline.configs import load_config, get_training_args
from training_pipeline.utils import load_model_and_processor, apply_freezing, setup_lora, print_model_parameters
from training_pipeline.dataset import DatasetLoader
from training_pipeline.trainers import VLMTrainer

def main():
    parser = argparse.ArgumentParser(description="VLA Training Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    args = parser.parse_args()

    # Load Configuration
    config = load_config(args.config)
    print(f"Loaded configuration from {args.config}")
    
    # Load Model and Processor
    print("Loading model and processor...")
    model, processor = load_model_and_processor(config)
    
    # Apply Freezing Strategy
    print("Applying freezing strategy...")
    model = apply_freezing(model, config)
    
    # Setup LoRA if enabled
    if config.lora and config.lora.enabled:
        print("Setting up LoRA adapters...")
        vision_lora = getattr(config, 'vision_lora', None)
        model = setup_lora(model, config.lora, vision_lora_config=vision_lora)
    
    # Enable input require grads AFTER LoRA setup so the hook survives PEFT wrapping.
    # This is needed for gradient checkpointing to work with frozen modules (e.g. LLM).
    model.enable_input_require_grads()
    
    # Load and Prepare Dataset
    print("Loading dataset...")
    dataset_loader = DatasetLoader(config.data, processor, model_config=config.model)
    train_dataset, eval_dataset = dataset_loader.load_dataset()
    data_collator = dataset_loader.get_collator()
    
    training_args = get_training_args(config, data_collator)
    
    # Initialize Trainer
    print("Initializing Trainer...")
    trainer = VLMTrainer(
        config=config,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=processor,
        peft_config=None # We applied PEFT manually if needed
    )
    
    # Print parameter counts per block
    print_model_parameters(model)

    # Train
    print("Starting training...")
    trainer.train()
    
    # Save Model
    print(f"Saving model to {config.training.output_dir}...")
    trainer.save_model()

if __name__ == "__main__":
    main()
