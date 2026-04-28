"""DFQ VLA Evaluation Pipeline.

Evaluates trajectory prediction quality using ADE and minADE metrics.
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
from eval_pipeline.metrics import compute_ade, compute_min_ade, aggregate_metrics
from eval_pipeline.inference import generate_trajectory_samples


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
    # We create a minimal VLMTrainingConfig wrapper
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
    eval_dataset = eval_dataset[:20]
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
    num_samples_k = eval_cfg.eval.num_samples
    temperature = eval_cfg.eval.temperature
    top_p = eval_cfg.eval.top_p
    max_new_tokens = eval_cfg.eval.max_new_tokens

    all_ade = []
    all_min_ade3 = []
    all_min_ade6 = []
    num_failed = 0

    print(f"\nStarting evaluation (K={num_samples_k}, T={temperature}, top_p={top_p})...")
    print(f"Generating {num_samples_k} trajectory samples per input.\n")
    start_time = time.time()

    for batch_idx, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
        # Ground truth future trajectory [B, T, 3]
        gt_xyz = batch["ego_future_xyz"].to(device)

        # Generate K trajectory samples
        traj_samples = generate_trajectory_samples(
            model=model,
            batch=batch,
            processor=processor,
            num_samples=num_samples_k,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )

        if len(traj_samples) == 0:
            num_failed += batch["input_ids"].shape[0]
            continue

        # Process per-sample in batch
        B = gt_xyz.shape[0]
        for b in range(B):
            gt_b = gt_xyz[b]  # [T, 3]

            # Collect all valid predictions for this sample
            pred_xyz_list = []
            for traj in traj_samples:
                pred_xyz_list.append(traj["xyz"][b])  # [64, 3]

            if not pred_xyz_list:
                num_failed += 1
                continue

            pred_samples = torch.stack(pred_xyz_list)  # [K_valid, 64, 3]

            # ADE: use first sample
            ade_val = compute_ade(pred_samples[0], gt_b).item()
            all_ade.append(ade_val)

            # minADE3: min over first 3 samples (if available)
            k3 = min(3, pred_samples.shape[0])
            min_ade3_val = compute_min_ade(pred_samples[:k3], gt_b).item()
            all_min_ade3.append(min_ade3_val)

            # minADE6: min over first 6 samples (if available)
            k6 = min(6, pred_samples.shape[0])
            min_ade6_val = compute_min_ade(pred_samples[:k6], gt_b).item()
            all_min_ade6.append(min_ade6_val)

        # Periodic logging
        if (batch_idx + 1) % 10 == 0:
            running = aggregate_metrics(all_ade, all_min_ade3, all_min_ade6)
            tqdm.write(
                f"  [Batch {batch_idx + 1}] "
                f"ADE={running['ADE']:.4f}  "
                f"minADE3={running['minADE3']:.4f}  "
                f"minADE6={running['minADE6']:.4f}  "
                f"(failed={num_failed})"
            )

    elapsed = time.time() - start_time

    # -------------------------------------------------------------------------
    # 6. Aggregate & Report
    # -------------------------------------------------------------------------
    results = aggregate_metrics(all_ade, all_min_ade3, all_min_ade6)
    results["num_failed"] = num_failed
    results["elapsed_seconds"] = round(elapsed, 1)
    results["config"] = args.config

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Samples evaluated: {results['num_samples']}")
    print(f"  Failed samples:    {num_failed}")
    print(f"  Time elapsed:      {elapsed:.1f}s")
    print(f"  ─────────────────────────────────")
    print(f"  ADE:               {results['ADE']:.4f}")
    print(f"  minADE3:           {results['minADE3']:.4f}")
    print(f"  minADE6:           {results['minADE6']:.4f}")
    print("=" * 60)

    # Save results to JSON
    output_file = eval_cfg.eval.output_file
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
