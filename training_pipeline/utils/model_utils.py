import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from typing import Tuple, Any

# Import DFQ_VLA components
# Assuming these are available in the python path as top-level modules
try:
    from dfq_vla.configuration_dfq_vla import DFQVLAConfig
    from dfq_vla.modelling_dfq_vla import DFQVLAForConditionalGeneration
    from dfq_vla.processing_dfq_vla import DFQVLAProcessor
except ImportError:
    # Fallback or strict error depending on environment
    print("Warning: Could not import dfq_vla modules directly. Ensure they are in PYTHONPATH.")
    DFQVLAConfig = None
    DFQVLAForConditionalGeneration = None
    DFQVLAProcessor = None

def register_dfq_vla():
    """Register the DFQ VLA model with Auto classes."""
    if DFQVLAConfig:
        AutoConfig.register("dfq_vla", DFQVLAConfig)
        AutoModelForCausalLM.register(DFQVLAConfig, DFQVLAForConditionalGeneration)
        AutoProcessor.register(DFQVLAConfig, DFQVLAProcessor)

def load_model_and_processor(config) -> Tuple[Any, Any]:
    """Load model and processor based on configuration."""
    register_dfq_vla()
    
    # Load processor
    processor_path = config.model.processor_path or config.model.model_path
    processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
    
    # Load Model
    torch_dtype = getattr(torch, config.model.torch_dtype) if hasattr(torch, config.model.torch_dtype) else torch.bfloat16
    
    # Configure 4-bit quantization if QLoRA is enabled
    quantization_config = None
    if config.qlora and config.qlora.enabled:
        compute_dtype = getattr(torch, config.qlora.bnb_4bit_compute_dtype, torch.bfloat16)
        # Skip modules that are not compatible with 4-bit quantization
        # These are custom modules in DFQ_VLA that should not be quantized
        skip_modules = [
            "vision_tower",      # DinoV3 vision encoder
            "projector",         # Vision-to-LM projector
            "flex_scene_encoder", # Multi-camera/timestamp encoder
            "action_head",       # Action chunking head
            "lm_head",           # LM classification head (tied with embed_tokens)
            "embed_tokens",      # MUST be skipped if lm_head is skipped due to weight tying!
        ]
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=config.qlora.load_in_4bit,
            bnb_4bit_quant_type=config.qlora.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=config.qlora.bnb_4bit_use_double_quant,
            llm_int8_skip_modules=skip_modules,
        )
        print(f"Using QLoRA with 4-bit quantization: quant_type={config.qlora.bnb_4bit_quant_type}, compute_dtype={compute_dtype}")
        print(f"Skipping modules from quantization: {skip_modules}")
    
    # Load config and inject VQ-VAE attributes before model init
    hf_config = AutoConfig.from_pretrained(config.model.model_path, trust_remote_code=False)
    hf_config.vqvae_checkpoint_path = getattr(config.model, "vqvae_checkpoint_path", None)
    hf_config.vqvae_num_embeddings = getattr(config.model, "vqvae_num_embeddings", 768)
    hf_config.vqvae_hidden_dim = getattr(config.model, "vqvae_hidden_dim", 256)
    hf_config.vqvae_embedding_dim = getattr(config.model, "vqvae_embedding_dim", 256)
    
    # Inject component inclusion flags
    hf_config.include_action_head = getattr(config.model, "include_action_head", True)
    hf_config.include_vqvae = getattr(config.model, "include_vqvae", True)

    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_path,
        config=hf_config,
        torch_dtype=torch_dtype,
        attn_implementation=config.model.attn_implementation,
        quantization_config=quantization_config,
        trust_remote_code=False,
    )
    
    # Ensure custom modules are natively cast to model's torch_dtype
    # This prevents Float vs BFloat16 crashes when running purely in BFloat16 without autocast
    for module_name in ["flex_scene_encoder", "projector", "action_head", "loss_calculator"]:
        if hasattr(model, module_name) and getattr(model, module_name) is not None:
            getattr(model, module_name).to(torch_dtype)
    
    # Note: We skip prepare_model_for_kbit_training() as it casts modules to float32
    # which conflicts with bf16 training. With modern bitsandbytes, the
    # BitsAndBytesConfig handles preparation automatically.
    # Sync processor's flex encoder settings with model config
    # This ensures the processor inserts the correct number of image tokens
    if hasattr(model.config, 'use_flex_scene_encoder'):
        processor.use_flex_scene_encoder = model.config.use_flex_scene_encoder
        processor.num_scene_tokens = getattr(model.config, 'num_scene_tokens', 800)
        print(f"Synced processor flex encoder settings: use_flex_scene_encoder={processor.use_flex_scene_encoder}, num_scene_tokens={processor.num_scene_tokens}")

    # Resize token embeddings to match tokenizer
    # unique to DFQ VLA: we add trajectory tokens to the tokenizer in the processor
    # so we must resize the model embeddings to accommodate them
    model.resize_token_embeddings(len(processor.tokenizer))
    
    # Sync loss weights from training config to model's loss calculator
    # The model's DFQVLAConfig (loaded from pretrained) may not have loss_weights,
    # so we override from the training pipeline config
    if hasattr(config.model, 'loss_weights') and config.model.loss_weights:
        from dfq_vla.loss import DFQVLALoss
        model.loss_calculator = DFQVLALoss(config.model.loss_weights)
        print(f"Applied loss weights from training config: {config.model.loss_weights}")
    
    return model, processor

def apply_freezing(model, config):
    """Apply freezing strategies based on configuration."""
    
    # 1. Vision Tower
    if config.model.freeze_vision_tower:
        print("Freezing vision tower...")
        for param in model.vision_tower.parameters():
            param.requires_grad = False
    
    # 2. Language Model (LLM)
    if config.model.freeze_llm:
        print("Freezing language model...")
        for param in model.language_model.parameters():
            param.requires_grad = False
            
        # Freezing the LLM usually means freezing the LM head as well, unless specified otherwise
        # But often we want to train the head if we are fine-tuning. 
        # For now, let's treat lm_head as part of the LLM block unless we want to be very specific.
        # If the user wants to train ONLY the head, they might use LoRA or just freeze the body.
        # Given the config structure, we'll freeze the lm_head here too if freeze_llm is True.
        for param in model.lm_head.parameters():
            param.requires_grad = False

    # 3. Projector
    if config.model.freeze_projector:
        print("Freezing projector...")
        for param in model.projector.parameters():
            param.requires_grad = False

    # 4. Action Head
    # Check if attribute exists (it might not on older configs, but we added it to dataclass)
    freeze_action_head = getattr(config.model, "freeze_action_head", False)
    if freeze_action_head:
        print("Freezing action head...")
        # Check if model has action_head
        if hasattr(model, "action_head"):
            for param in model.action_head.parameters():
                param.requires_grad = False
        else:
            print("Warning: requested to freeze action_head, but model does not have 'action_head' attribute.")

    # 5. Flex Scene Encoder
    freeze_flex_encoder = getattr(config.model, "freeze_flex_encoder", False)
    if freeze_flex_encoder:
        print("Freezing flex scene encoder...")
        if hasattr(model, "flex_scene_encoder") and model.flex_scene_encoder is not None:
            for param in model.flex_scene_encoder.parameters():
                param.requires_grad = False
        else:
            print("Warning: requested to freeze flex_scene_encoder, but model does not have it.")
    
    # 6. Trajectory Projector
    freeze_traj_projector = getattr(config.model, "freeze_traj_projector", False)
    if freeze_traj_projector:
        print("Freezing trajectory projector...")
        if hasattr(model, "traj_projector") and model.traj_projector is not None:
            for param in model.traj_projector.parameters():
                param.requires_grad = False
        else:
            print("Warning: requested to freeze traj_projector, but model does not have it.")

    # Debug print to verify
    trainable_params = []
    total_params = 0
    trainable_count = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params.append(name)
            trainable_count += param.numel()
    
    # print(f"Total parameters: {total_params:,}")
    # print(f"Trainable parameters: {trainable_count:,} ({trainable_count/total_params:.2%})")
    
    # If list is too long, print summary
    if len(trainable_params) > 20:
        print(f"First 10 trainable modules:")
        for p in trainable_params[:10]:
            print(f" - {p}")
        print("...")
    else:
        print(f"Trainable modules: {trainable_params}")

    return model

def setup_lora(model, lora_config, vision_lora_config=None):
    """Apply LoRA configuration to the model.
    
    Supports applying LoRA to both LLM and Vision tower independently
    via a single get_peft_model call with regex-prefixed target modules.
    
    Args:
        model: The model to apply LoRA to.
        lora_config: LLM LoRA configuration.
        vision_lora_config: Optional vision tower LoRA configuration.
    """
    if not lora_config or not lora_config.enabled:
        return model
        
    # Build target_modules with regex prefixes to disambiguate LLM vs Vision
    llm_targets = [m for m in lora_config.target_modules]
    
    all_targets = llm_targets
    
    # Merge vision LoRA targets if enabled
    if vision_lora_config and vision_lora_config.enabled:
        vision_targets = [f"vision_tower.*{m}" for m in vision_lora_config.target_modules]
        all_targets = all_targets + vision_targets
        print(f"Vision LoRA enabled with targets: {vision_lora_config.target_modules}")
    
    peft_config = LoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        use_rslora=lora_config.use_rslora,
        bias=lora_config.bias,
        target_modules=all_targets,
        task_type="CAUSAL_LM",
        modules_to_save=lora_config.modules_to_save,
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model

def print_model_parameters(model):
    """Print the number of parameters in each block of the DFQ VLA model."""
    print("=== Model Parameter Breakdown ===")
    blocks = {
        "Vision Tower": 0,
        "Flex Scene Encoder": 0,
        "Projector": 0,
        "Traj Projector": 0,
        "Language Model": 0,
        "LM Head": 0,
        "Action Head": 0,
        "Other": 0
    }
    trainable_blocks = {k: 0 for k in blocks.keys()}
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        is_trainable = param.requires_grad
        
        if "vision_tower" in name:
            key = "Vision Tower"
        elif "flex_scene_encoder" in name:
            key = "Flex Scene Encoder"
        elif "traj_projector" in name:
            key = "Traj Projector"
        elif "projector" in name:
            key = "Projector"
        elif "language_model" in name:
            key = "Language Model"
        elif "lm_head" in name:
            key = "LM Head"
        elif "action_head" in name:
            key = "Action Head"
        else:
            key = "Other"
            
        blocks[key] += num_params
        if is_trainable:
            trainable_blocks[key] += num_params
            
    total_params = sum(blocks.values())
    total_trainable = sum(trainable_blocks.values())
    
    for key in blocks.keys():
        if blocks[key] > 0:
            print(f"{key:<20} | Total: {blocks[key]:>14,} | Trainable: {trainable_blocks[key]:>14,}")
            
    print("-" * 55)
    print(f"{'Total':<20} | Total: {total_params:>14,} | Trainable: {total_trainable:>14,}")
    print("=================================")
