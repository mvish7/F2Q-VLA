import argparse
import logging
import os
import sys
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
from peft import PeftModel

# Attempt to import DFQ_VLA components
try:
    from dfq_vla.configuration_dfq_vla import DFQVLAConfig
    from dfq_vla.modelling_dfq_vla import DFQVLAForConditionalGeneration
    from dfq_vla.processing_dfq_vla import DFQVLAProcessor
except ImportError:
    logging.warning("Could not import dfq_vla modules directly. Ensure they are in PYTHONPATH.")
    DFQVLAConfig = None
    DFQVLAForConditionalGeneration = None
    DFQVLAProcessor = None

from training_pipeline.utils import load_model_and_processor, setup_lora
from training_pipeline.configs import load_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def register_dfq_vla():
    """Register the DFQ VLA model with Auto classes."""
    if DFQVLAConfig:
        AutoConfig.register("dfq_vla", DFQVLAConfig)
        AutoModelForCausalLM.register(DFQVLAConfig, DFQVLAForConditionalGeneration)
        AutoProcessor.register(DFQVLAConfig, DFQVLAProcessor)
        logger.info("Registered DFQVLA classes.")
    else:
        logger.error("DFQVLA classes not available for registration.")

def parse_args():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model.")
    parser.add_argument(
        "--config_path", 
        type=str, 
        required=True, 
        help="Path to the training configuration YAML file."
    )
    parser.add_argument(
        "--adapter_path", 
        type=str, 
        required=True, 
        help="Path to the LoRA adapter checkpoint directory."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True, 
        help="Directory to save the merged model."
    )
    return parser.parse_args()

def merge_adapter(config_path: str, adapter_path: str, output_dir: str):
    """
    Loads base model from config, loads adapter, merges, and saves.
    """
    logger.info(f"Loading configuration from {config_path}")
    config = load_config(config_path)

    register_dfq_vla()

    logger.info("Loading base model and processor...")
    model, processor = load_model_and_processor(config)

    # In strict merge scripts, we often don't need 'setup_lora' if we are just loading the adapter 
    # via PeftModel or load_adapter. However, if the base model needs specific LoRA prep, we keep it.
    # The original script called setup_lora.
    # model = setup_lora(model, config.lora) 
    
    logger.info(f"Loading adapter from {adapter_path}")
    try:
        # Load adapter using the model's load_adapter method (wrapper around PEFT)
        # Note: Depending on transformers version/PEFT integration, this might be all that's needed.
        model.load_adapter(adapter_path, adapter_name="default", local_files_only=True)
    except Exception as e:
        logger.error(f"Failed to load adapter from {adapter_path}: {e}")
        raise

    logger.info("Merging adapter into base model...")
    model = model.merge_and_unload()

    logger.info(f"Saving merged model and processor to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    logger.info("Merge complete.")

def main():
    args = parse_args()
    
    if not os.path.exists(args.config_path):
        logger.error(f"Config file not found: {args.config_path}")
        sys.exit(1)
        
    if not os.path.exists(args.adapter_path):
        logger.error(f"Adapter path not found: {args.adapter_path}")
        sys.exit(1)

    try:
        merge_adapter(args.config_path, args.adapter_path, args.output_dir)
    except Exception as e:
        logger.exception("An error occurred during the merge process.")
        sys.exit(1)

if __name__ == "__main__":
    main()

