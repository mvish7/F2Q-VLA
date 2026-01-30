from dataclasses import dataclass, field
from typing import Optional, List, Tuple

@dataclass
class ModelConfig:
    model_path: str
    processor_path: Optional[str] = None
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"
    type: str = "f2q_vla"
    
    # Freezing configuration
    freeze_vision_tower: bool = True
    freeze_llm: bool = False
    freeze_projector: bool = False
    freeze_action_head: bool = False
    freeze_flex_encoder: bool = False  # Flex scene encoder freezing
    
    # Loss weights
    loss_weights: dict = field(default_factory=lambda: {"text": 1.0, "xyz": 1.0, "rot": 1.0})

@dataclass
class DataConfig:
    dataset_path: str
    image_base_path: str = ""  # Base path to prepend to relative image paths
    test_split_ratio: float = 0.01
    image_size_height: int = 360
    image_size_width: int = 640
    dataloader_num_workers: int = 6
    # Flex Scene Encoder data config
    num_cameras: int = 4  # Number of camera views
    num_timestamps: int = 4  # Number of timestamps per camera
    # max_len: int = 1024

@dataclass
class LoRAConfig:
    enabled: bool = False
    r: int = 128
    lora_alpha: int = 256
    lora_dropout: float = 0.05
    bias: str = "none"
    use_rslora: bool = True
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    modules_to_save: List[str] = field(default_factory=list)

@dataclass
class TrainingConfig:
    output_dir: str
    num_train_epochs: int = 1
    learning_rate: float = 3e-4
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 32
    gradient_checkpointing: bool = True
    report_to: str = "tensorboard"
    save_strategy: str = "steps"
    save_steps: int = 500
    eval_strategy: str = "steps"
    eval_steps: int = 500
    logging_steps: int = 100
    warmup_ratio: float = 0.03
    bf16: bool = True
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    
@dataclass
class VLMTrainingConfig:
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    lora: Optional[LoRAConfig] = None
