"""DFQ VLA Evaluation Pipeline.

Evaluates action reasoning text generation quality.
Reuses model loading, dataset loading, and preprocessing from the training pipeline.

Usage:
    python -m eval_pipeline.evaluate --config eval_pipeline/configs/examples/eval.yaml
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from peft import PeftModel

# Reuse training pipeline components
from training_pipeline.configs import load_config
from training_pipeline.utils import load_model_and_processor, apply_freezing, setup_lora
from training_pipeline.dataset import DatasetLoader

# Eval pipeline components
from eval_pipeline.configs.eval_config import load_eval_config
from eval_pipeline.inference import generate_action_reasoning


def main():
    parser = argparse.ArgumentParser(description="DFQ VLA Evaluation Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to eval YAML config")
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # 1. Load Configuration
    # -------------------------------------------------------------------------
    eval_cfg = load_eval_config(args.config)
    print(f"Loaded eval configuration from {args.config}")

    # Build a training-compatible config for reusing model/dataset loading
    from training_pipeline.configs.configs import VLMTrainingConfig, TrainingConfig

    training_config = TrainingConfig(output_dir="/tmp/eval_unused")
    compat_config = VLMTrainingConfig(
        model=eval_cfg.model,
        data=eval_cfg.data,
        training=training_config,
        lora=eval_cfg.lora,
        qlora=eval_cfg.qlora,
    )

    # -------------------------------------------------------------------------
    # 2. Load Model & Processor
    # -------------------------------------------------------------------------
    print("Loading model and processor...")
    model, processor = load_model_and_processor(compat_config)

    # Apply freezing (all frozen for eval, but needed for model setup)
    model = apply_freezing(model, compat_config)

    # Load LoRA adapters if checkpoint contains them
    checkpoint_path = eval_cfg.checkpoint_path
    if checkpoint_path:
        is_lora = (Path(checkpoint_path) / "adapter_config.json").exists()
        if is_lora:
            print(f"Loading LoRA adapters from {checkpoint_path}...")
            model = PeftModel.from_pretrained(model, checkpoint_path, is_trainable=False)
        else:
            print(f"Note: checkpoint_path specified but no adapter_config.json found.")
            print(f"Assuming weights are already merged into model_path.")

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model on {device}, memory: {model.get_memory_footprint() / 1e9:.2f} GB")

    # -------------------------------------------------------------------------
    # 3. Load Dataset (eval split only)
    # -------------------------------------------------------------------------
    print("Loading dataset...")
    dataset_loader = DatasetLoader(eval_cfg.data, processor, model_config=eval_cfg.model)
    _, eval_dataset = dataset_loader.load_dataset()
    eval_dataset = eval_dataset[0:20]
    data_collator = dataset_loader.get_collator()

    if not eval_dataset:
        print("Error: eval split is empty. Check test_split_ratio in config.")
        return

    print(f"Eval dataset size: {len(eval_dataset)} samples")

    # -------------------------------------------------------------------------
    # 4. Create DataLoader
    # -------------------------------------------------------------------------
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=eval_cfg.eval.batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=0,  # Keep simple for eval
        pin_memory=False,
    )

    # -------------------------------------------------------------------------
    # 5. Evaluation Loop
    # -------------------------------------------------------------------------
    max_new_tokens = eval_cfg.eval.max_new_tokens

    all_results = []

    print(f"\nStarting evaluation (max_new_tokens={max_new_tokens})...")
    start_time = time.time()

    for batch_idx, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
        # Generate action reasoning text
        generated_texts = generate_action_reasoning(
            model=model,
            batch=batch,
            processor=processor,
            max_new_tokens=max_new_tokens,
        )

        # Store results
        for text in generated_texts:
            all_results.append({
                "generated_reasoning": text,
            })

        # Periodic logging
        if (batch_idx + 1) % 10 == 0:
            tqdm.write(
                f"  [Batch {batch_idx + 1}] "
                f"Generated {len(all_results)} samples so far"
            )

    elapsed = time.time() - start_time

    # -------------------------------------------------------------------------
    # 6. Report
    # -------------------------------------------------------------------------
    results = {
        "num_samples": len(all_results),
        "elapsed_seconds": round(elapsed, 1),
        "config": args.config,
        "predictions": all_results,
    }

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Samples evaluated: {results['num_samples']}")
    print(f"  Time elapsed:      {elapsed:.1f}s")
    print("=" * 60)

    # Print first few predictions
    for i, r in enumerate(all_results[:3]):
        print(f"\n--- Sample {i} ---")
        print(f"  Generated: {r['generated_reasoning'][:200]}...")

    # Save results to JSON
    output_file = eval_cfg.eval.output_file
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
