import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
from peft import LoraConfig, get_peft_model
from typing import Tuple, Any

# Import F2Q_VLA components
# Assuming these are available in the python path as top-level modules
try:
    from f2q_vla.configuration_f2q_vla import F2QVLAConfig
    from f2q_vla.modelling_f2q_vla import F2QVLAForConditionalGeneration
    from f2q_vla.processing_f2q_vla import F2QVLAProcessor
except ImportError:
    # Fallback or strict error depending on environment
    print("Warning: Could not import f2q_vla modules directly. Ensure they are in PYTHONPATH.")
    F2QVLAConfig = None
    F2QVLAForConditionalGeneration = None
    F2QVLAProcessor = None

def register_f2q_vla():
    """Register the F2Q VLA model with Auto classes."""
    if F2QVLAConfig:
        AutoConfig.register("f2q_vla", F2QVLAConfig)
        AutoModelForCausalLM.register(F2QVLAConfig, F2QVLAForConditionalGeneration)
        AutoProcessor.register(F2QVLAConfig, F2QVLAProcessor)

def load_model_and_processor(config) -> Tuple[Any, Any]:
    """Load model and processor based on configuration."""
    register_f2q_vla()
    
    # Load processor
    processor_path = config.model.processor_path or config.model.model_path
    processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
    
    # Load Model
    torch_dtype = getattr(torch, config.model.torch_dtype) if hasattr(torch, config.model.torch_dtype) else torch.bfloat16
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_path,
        torch_dtype=torch_dtype,
        attn_implementation=config.model.attn_implementation,
        trust_remote_code=False
    )

    # Resize token embeddings to match tokenizer
    # unique to F2Q VLA: we add trajectory tokens to the tokenizer in the processor
    # so we must resize the model embeddings to accommodate them
    model.resize_token_embeddings(len(processor.tokenizer))
    
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

    # Debug print to verify
    trainable_params = []
    total_params = 0
    trainable_count = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params.append(name)
            trainable_count += param.numel()
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_count:,} ({trainable_count/total_params:.2%})")
    
    # If list is too long, print summary
    if len(trainable_params) > 20:
        print(f"First 10 trainable modules:")
        for p in trainable_params[:10]:
            print(f" - {p}")
        print("...")
    else:
        print(f"Trainable modules: {trainable_params}")

    return model

def setup_lora(model, lora_config):
    """Apply LoRA configuration to the model."""
    if not lora_config or not lora_config.enabled:
        return model
        
    peft_config = LoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        use_rslora=lora_config.use_rslora,
        bias=lora_config.bias,
        target_modules=lora_config.target_modules,
        task_type="CAUSAL_LM",
        modules_to_save=lora_config.modules_to_save,
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model
