"""Per-module parameter group builder for multi-LR training.

Classifies trainable parameters into distinct optimizer groups with
independent learning rates to support staged VLA training.
"""

from typing import Any


def build_param_groups(model: Any, config: Any) -> list[dict]:
    """Build optimizer parameter groups with per-module learning rates.
    
    Groups trainable parameters into:
    - **default**: Projector, Flex, Traj MLP, Action Head, LM Head, etc.
      Uses `config.training.learning_rate`.
    - **llm_lora**: LLM LoRA adapter weights (lora_A, lora_B).
      Uses `config.training.llm_learning_rate` (falls back to default).
    - **vision_lora**: Vision tower LoRA adapter weights.
      Uses `config.training.vision_enc_learning_rate` (falls back to default).
    
    Args:
        model: The full model (potentially PEFT-wrapped).
        config: VLMTrainingConfig with training section containing per-module LRs.
        
    Returns:
        List of parameter group dicts for the optimizer.
    """
    base_lr = config.training.learning_rate

    llm_lora_lr = getattr(config.training, 'llm_learning_rate', None) or base_lr
    vision_lora_lr = getattr(config.training, 'vision_enc_learning_rate', None) or base_lr
    
    groups = {"default": [], "llm_lora": [], "vision_lora": []}
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if "lora_" in name and "vision_tower" in name:
            groups["vision_lora"].append(param)
        elif "lora_" in name and "language_model" in name:
            groups["llm_lora"].append(param)
        else:
            groups["default"].append(param)
    
    param_groups = []
    
    if groups["default"]:
        param_groups.append({"params": groups["default"], "lr": base_lr})
    if groups["llm_lora"]:
        param_groups.append({"params": groups["llm_lora"], "lr": llm_lora_lr})
    if groups["vision_lora"]:
        param_groups.append({"params": groups["vision_lora"], "lr": vision_lora_lr})
    
    # Summary
    print(f"=== Optimizer Parameter Groups ===")
    print(f"  Default (LR={base_lr}): {sum(p.numel() for p in groups['default']):,} params")
    print(f"  LLM LoRA (LR={llm_lora_lr}): {sum(p.numel() for p in groups['llm_lora']):,} params")
    print(f"  Vision LoRA (LR={vision_lora_lr}): {sum(p.numel() for p in groups['vision_lora']):,} params")
    print(f"==================================")
    
    return param_groups
