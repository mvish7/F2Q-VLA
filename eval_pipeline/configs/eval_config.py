"""Evaluation-specific configuration."""

import dataclasses
from dataclasses import dataclass, field
from typing import Optional

import yaml

from training_pipeline.configs.configs import (
    ModelConfig,
    DataConfig,
    LoRAConfig,
    QLoRAConfig,
)


@dataclass
class EvalConfig:
    """Evaluation hyperparameters (sampling + metric settings)."""

    num_samples: int = 6  # K for minADE-K (must be >= 6 for minADE6)
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 32  # 8 VQ indices + delimiters + margin
    batch_size: int = 1
    output_file: str = "eval_results.json"


@dataclass
class VLAEvalConfig:
    """Top-level evaluation configuration — mirrors VLMTrainingConfig layout."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    lora: Optional[LoRAConfig] = None
    qlora: Optional[QLoRAConfig] = None
    # Reuse training_pipeline's checkpoint resume field
    checkpoint_path: Optional[str] = None


def load_eval_config(config_path: str) -> VLAEvalConfig:
    """Load evaluation configuration from a YAML file."""
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    def _create(cls, data):
        if data is None:
            return None
        field_names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in field_names})

    model_config = _create(ModelConfig, config_dict.get("model", {}))
    data_config = _create(DataConfig, config_dict.get("data", {}))
    eval_config = _create(EvalConfig, config_dict.get("eval", {}))

    lora_config = None
    if config_dict.get("lora"):
        lora_config = _create(LoRAConfig, config_dict["lora"])

    qlora_config = None
    if config_dict.get("qlora"):
        qlora_config = _create(QLoRAConfig, config_dict["qlora"])

    return VLAEvalConfig(
        model=model_config,
        data=data_config,
        eval=eval_config,
        lora=lora_config,
        qlora=qlora_config,
        checkpoint_path=config_dict.get("checkpoint_path"),
    )
