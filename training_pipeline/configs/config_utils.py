import yaml
import os
import dataclasses
from typing import Type, TypeVar, Any, Dict
from .configs import VLMTrainingConfig, ModelConfig, DataConfig, TrainingConfig, LoRAConfig

T = TypeVar("T")

def load_config(config_path: str) -> VLMTrainingConfig:
    """Load configuration from a YAML file."""
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    
    # Helper to create nested config objects
    def create_config_object(cls: Type[T], data: Dict[str, Any]) -> T:
        if data is None:
            return None
        
        # Get field names for the dataclass
        field_names = {f.name for f in dataclasses.fields(cls)}
        
        # Filter data to only include valid fields
        filtered_data = {k: v for k, v in data.items() if k in field_names}
        
        return cls(**filtered_data)

    model_config = create_config_object(ModelConfig, config_dict.get("model", {}))
    data_config = create_config_object(DataConfig, config_dict.get("data", {}))
    training_config = create_config_object(TrainingConfig, config_dict.get("training", {}))
    
    lora_config = None
    if "lora" in config_dict and config_dict["lora"]:
        lora_config = create_config_object(LoRAConfig, config_dict["lora"])

    return VLMTrainingConfig(
        model=model_config,
        data=data_config,
        training=training_config,
        lora=lora_config
    )

def save_config(config: VLMTrainingConfig, save_path: str):
    """Save configuration to a YAML file."""
    config_dict = dataclasses.asdict(config)
    
    # Clean up None values if needed, or keep them to be explicit
    # For now, we dump everything
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)
