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
    
    return model, processor

def apply_freezing(model, config):
    """Apply freezing strategies based on configuration."""
    
    # Freeze Vision Tower
    if config.model.freeze_vision_tower:
        for name, param in model.named_parameters():
             if "vision_tower" in name:
                param.requires_grad = False
                
    # Freeze LLM
    if config.model.freeze_llm:
        for name, param in model.named_parameters():
            # Adjust logical check based on model architecture
            # Usually LLM parts are not vision_tower
            if "vision_tower" not in name and "projector" not in name:
                 # Check if it's strictly LLM parameters. 
                 # For F2Q VLA, assuming language_model or similar containment, or everything else
                 # If we freeze LLM, we typically freeze everything EXCEPT projector if freeze_projector is False
                 param.requires_grad = False
    
    # Freeze Projector
    if config.model.freeze_projector:
        for name, param in model.named_parameters():
            if "projector" in name:
                param.requires_grad = False

    # Debug print to verify
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(name)
    
    # print(f"Model keys frozen. Trainable parameters: {len(trainable_params)}")
    # if len(trainable_params) < 50:
    print(f"Trainable params: {trainable_params}")

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
