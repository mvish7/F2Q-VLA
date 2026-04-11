"""Merge a LoRA adapter checkpoint back into the base DFQ-VLA model.

Usage:
    python -m training_pipeline.utils.merge_lora_adapter \
        --config_path training_pipeline/configs/examples/stage1.yaml \
        --adapter_path /path/to/checkpoint-XXXX \
        --output_dir /path/to/merged_model
"""

import argparse
import logging
import os
import sys

import torch
from peft import PeftModel

from training_pipeline.configs import load_config
from training_pipeline.utils import load_model_and_processor, setup_lora

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base DFQ-VLA model.")
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the training configuration YAML file used during LoRA training.",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Path to the saved LoRA adapter checkpoint directory (e.g. checkpoint-500).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the merged model and processor.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to load model on for merging (default: cpu).",
    )
    return parser.parse_args()


def merge_adapter(config_path: str, adapter_path: str, output_dir: str, device: str = "cpu"):
    """Load base model, apply LoRA structure, load adapter weights, merge, and save.

    The merge process mirrors the training setup:
    1. Load base model + processor via the same pipeline used in training.
    2. Wrap with PEFT using the same LoRA config (so modules_to_save are wrapped correctly).
    3. Load the adapter checkpoint weights into the PEFT model.
    4. Merge LoRA weights and unwrap to get a plain model.
    5. Save the merged model and processor.
    """
    # 1. Load training config
    logger.info(f"Loading training config from: {config_path}")
    config = load_config(config_path)

    # Disable quantization for merging — we want full-precision weights
    if config.qlora:
        config.qlora.enabled = False
        config.qlora.load_in_4bit = False
    logger.info("Quantization disabled for merge.")

    # 2. Load base model and processor (same as training pipeline)
    logger.info("Loading base model and processor...")
    model, processor = load_model_and_processor(config)
    model = model.to(device)
    logger.info(f"Base model loaded on {device}.")

    # 3. Wrap model with PEFT LoRA (same config as training)
    #    This creates the same adapter structure including modules_to_save wrappers
    if config.lora and config.lora.enabled:
        logger.info("Applying LoRA structure to match training setup...")
        model = setup_lora(model, config.lora)
    else:
        logger.error("LoRA is not enabled in the config. Nothing to merge.")
        sys.exit(1)

    # 4. Load adapter weights from checkpoint
    logger.info(f"Loading adapter weights from: {adapter_path}")
    adapter_weights_path = os.path.join(adapter_path, "adapter_model.safetensors")
    if not os.path.exists(adapter_weights_path):
        adapter_weights_path = os.path.join(adapter_path, "adapter_model.bin")

    if os.path.exists(adapter_weights_path):
        # Use PeftModel's load mechanism to load the adapter weights
        model.load_adapter(adapter_path, adapter_name="default")
        logger.info("Adapter weights loaded successfully.")
    else:
        logger.error(
            f"No adapter weights found at {adapter_path}. "
            "Expected adapter_model.safetensors or adapter_model.bin"
        )
        sys.exit(1)

    # 5. Merge LoRA into base weights and unwrap
    logger.info("Merging adapter into base model...")
    model = model.merge_and_unload()
    logger.info("Merge complete — LoRA adapters merged and PEFT wrappers removed.")

    # 6. Save merged model and processor
    logger.info(f"Saving merged model to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    processor.save_pretrained(output_dir)
    logger.info(f"Merged model and processor saved to: {output_dir}")


def main():
    args = parse_args()

    if not os.path.exists(args.config_path):
        logger.error(f"Config file not found: {args.config_path}")
        sys.exit(1)

    if not os.path.exists(args.adapter_path):
        logger.error(f"Adapter path not found: {args.adapter_path}")
        sys.exit(1)

    try:
        merge_adapter(args.config_path, args.adapter_path, args.output_dir, args.device)
    except Exception:
        logger.exception("An error occurred during the merge process.")
        sys.exit(1)


if __name__ == "__main__":
    main()
