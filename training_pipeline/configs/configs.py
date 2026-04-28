from dataclasses import dataclass, field
from typing import Optional, List, Tuple

@dataclass
class ModelConfig:
    model_path: str
    processor_path: Optional[str] = None
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"
    type: str = "dfq_vla"
    
    # Freezing configuration
    freeze_vision_tower: bool = True
    freeze_llm: bool = False
    freeze_projector: bool = False
    freeze_action_head: bool = False
    freeze_flex_encoder: bool = False  # Flex scene encoder freezing
    freeze_traj_projector: bool = False
    
    # Component inclusion (set False to exclude from model entirely)
    include_action_head: bool = True
    include_vqvae: bool = True
    scheduled_sampling_prob: float = 0.0  # Prob of using LLM predictions for VQ-VAE decoding (0=teacher forcing)
    action_head_grad_scale: float = 0.1  # Scale factor for gradients from action_head → LLM (0=detach, 1=full)
    
    # Loss weights
    loss_weights: dict = field(default_factory=lambda: {"text": 1.0, "xyz": 0.001, "rot": 0.001})
    
    # VQ-VAE trajectory tokenizer
    vqvae_checkpoint_path: str = "/media/vishal/workspace/projects/VQ-VAE/checkpoints/codebook768_2drot_high_perp/epoch43_best.pt"     # Path to pre-trained VQ-VAE checkpoint
    vqvae_num_embeddings: int = 768     # Codebook size K
    vqvae_hidden_dim: int = 256         # Must match trained model
    vqvae_embedding_dim: int = 256      # Must match trained model

@dataclass
class DataConfig:
    dataset_path: str
    data_base_path: str = ""  # Base path to prepend to relative image paths
    test_split_ratio: float = 0.01
    image_size_height: int = 320
    image_size_width: int = 512
    dataloader_num_workers: int = 1
    dataloader_pin_memory: bool = True
    dataloader_prefetch_factor: int = 0
    # Flex Scene Encoder data config
    num_cameras: int = 4  # Number of camera views
    num_timestamps: int = 4  # Number of timestamps per camera
    max_len: int = 1024
    num_history_steps: int = 16
    num_future_steps: int = 64
    time_step: float = 0.1
    num_frames: int = 4
    traj_history_dropout_prob: float = 0.0  # Prob of zeroing out traj history per sample (regularization)

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
    # Target module expansion: resolve short names to fully-qualified layer names
    expand_target_modules: bool = False
    llm_target_modules: List[str] = field(default_factory=list)
    vision_enc_target_modules: List[str] = field(default_factory=list)

@dataclass
class QLoRAConfig:
    """Configuration for QLoRA (4-bit quantized LoRA) training."""
    enabled: bool = False
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"  # "nf4" or "fp4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True

@dataclass
class TrainingConfig:
    output_dir: str
    num_train_epochs: int = 1
    learning_rate: float = 3e-4
    llm_learning_rate: Optional[float] = None       # LLM LoRA LR (None → falls back to learning_rate)
    vision_enc_learning_rate: Optional[float] = None # Vision encoder LoRA LR (None → falls back to learning_rate)
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
    use_liger_kernel: bool = False  # Enable Liger Kernel for faster training
    torch_empty_cache_steps: int = None  # Clear CUDA cache every N steps (None = disabled)
    optim: str = "adamw_bnb_8bit"  # Optimizer type: adamw_torch, adamw_bnb_8bit,
    resume_from_checkpoint: Optional[str] = None  # Path to full checkpoint or LoRA adapter dir (auto-detected)
    torch_compile: bool = False  # Enable torch.compile for the model
    torch_compile_backend: Optional[str] = None  # Compiler backend (e.g. "inductor")
    torch_compile_mode: Optional[str] = None  # Compilation mode (e.g. "reduce-overhead", "max-autotune")
    
@dataclass
class VLMTrainingConfig:
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    lora: Optional[LoRAConfig] = None
    qlora: Optional[QLoRAConfig] = None
