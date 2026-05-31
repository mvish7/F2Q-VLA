"""Create DFQ VLA model artifacts.

Assembles the complete DFQ VLA model by loading pretrained weights for:
- TIPSv2 vision encoder
- LFM2.5 language model
And initializing from scratch:
- Vision-to-LLM projector
- Flex scene encoder (multi-camera, multi-timestamp compression)
- Trajectory projector (history trajectory encoding MLP)

All components are saved together as a single HuggingFace model + processor.
"""

import json

import torch
from transformers import AutoModel, AutoConfig, AutoTokenizer, CLIPImageProcessor

from dfq_vla.configuration_dfq_vla import DFQVLAConfig, VISION_MODEL_ID
from dfq_vla.modelling_dfq_vla import DFQVLAForConditionalGeneration
from dfq_vla.processing_dfq_vla import DFQVLAProcessor


# =============================================================================
# Configuration
# =============================================================================
print(VISION_MODEL_ID)
LANGUAGE_MODEL_ID = "LiquidAI/LFM2.5-350M"
print(LANGUAGE_MODEL_ID)
OUTPUT_DIR = "./dfq_vla_artifacts"


def main():
    # =========================================================================
    # 1. Load sub-model configs
    # =========================================================================
    print("Loading configs...")
    vision_config = AutoConfig.from_pretrained(VISION_MODEL_ID, trust_remote_code=True)
    text_config = AutoConfig.from_pretrained(LANGUAGE_MODEL_ID)

    # =========================================================================
    # 2. Create DFQ VLA config (includes flex encoder defaults)
    # =========================================================================
    dfq_config = DFQVLAConfig(
        vision_config=vision_config,
        text_config=text_config,
    )
    print(f"  Vision hidden size: {dfq_config.vision_hidden_size}")
    print(f"  LLM hidden size:    {dfq_config.hidden_size}")
    print(f"  Flex scene tokens:  {dfq_config.num_scene_tokens}")
    print(f"  Traj input dim:     {dfq_config.traj_input_dim}")

    # =========================================================================
    # 3. Initialize full model (all components created from config)
    # =========================================================================
    print("\nInitializing model architecture...")
    model = DFQVLAForConditionalGeneration(dfq_config)

    # =========================================================================
    # 4. Load pretrained weights for vision encoder and LLM
    # =========================================================================
    print(f"\nLoading TIPSv2 vision encoder weights from {VISION_MODEL_ID}...")
    vision_model = AutoModel.from_pretrained(VISION_MODEL_ID, trust_remote_code=True)
    missing, unexpected = model.vision_tower.load_state_dict(
        vision_model.state_dict(), strict=False
    )
    if missing:
        print(f"  Vision missing keys: {len(missing)}")
    if unexpected:
        print(f"  Vision unexpected keys: {len(unexpected)}")
        
    print("\nInitializing Flex Scene Encoder from TIPSv2 weights...")
    if getattr(dfq_config, "use_flex_scene_encoder", False) and model.flex_scene_encoder is not None:
        flex = model.flex_scene_encoder
        tips_blocks = vision_model.vision_encoder.blocks
        
        # Map 4 evenly spaced layers from TIPSv2 to Flex
        # TIPSv2 has 12 layers (0 to 11). We pick 0, 4, 8, 11
        layer_mapping = {0: 8, 1: 9, 2: 10, 3: 11}
        
        with torch.no_grad():
            for flex_idx, tips_idx in layer_mapping.items():
                flex_layer = flex.encoder.layers[flex_idx]
                tips_layer = tips_blocks[tips_idx]
                
                # Pre-attention LayerNorm
                flex_layer.norm1.weight.copy_(tips_layer.norm1.weight)
                flex_layer.norm1.bias.copy_(tips_layer.norm1.bias)
                
                # Fused QKV (TIPSv2 qkv -> Flex in_proj)
                flex_layer.self_attn.in_proj_weight.copy_(tips_layer.attn.qkv.weight)
                flex_layer.self_attn.in_proj_bias.copy_(tips_layer.attn.qkv.bias)
                
                # Output Projection
                flex_layer.self_attn.out_proj.weight.copy_(tips_layer.attn.proj.weight)
                flex_layer.self_attn.out_proj.bias.copy_(tips_layer.attn.proj.bias)
                
                # Pre-FFN LayerNorm
                flex_layer.norm2.weight.copy_(tips_layer.norm2.weight)
                flex_layer.norm2.bias.copy_(tips_layer.norm2.bias)
                
                # FFN (TIPSv2 fc1/fc2 -> Flex linear1/linear2)
                flex_layer.linear1.weight.copy_(tips_layer.mlp.fc1.weight)
                flex_layer.linear1.bias.copy_(tips_layer.mlp.fc1.bias)
                
                flex_layer.linear2.weight.copy_(tips_layer.mlp.fc2.weight)
                flex_layer.linear2.bias.copy_(tips_layer.mlp.fc2.bias)
                
        print("  FlexSceneEncoder partially initialized from TIPSv2 weights.")
        
    del vision_model

    print("Loading LFM2.5 LLM weights...")
    from transformers import AutoModelForCausalLM
    llm_causal = AutoModelForCausalLM.from_pretrained(LANGUAGE_MODEL_ID)
    
    missing, unexpected = model.language_model.load_state_dict(
        llm_causal.model.state_dict(), strict=False
    )
    if missing:
        print(f"  LLM body missing keys: {len(missing)}")
    if unexpected:
        print(f"  LLM body unexpected keys: {len(unexpected)}")
        
    missing_head, unexpected_head = model.lm_head.load_state_dict(
        llm_causal.lm_head.state_dict(), strict=False
    )
    if missing_head:
        print(f"  LM head missing keys: {len(missing_head)}")
    
    del llm_causal

    print("\nRandomly initialized components:")
    print("  - Projector (vision → LLM)")
    print("  - Trajectory projector (MLP)")

    # =========================================================================
    # 5. Create processor (tokenizer + image processor + trajectory tokens)
    # =========================================================================
    print("\nCreating processor...")
    tokenizer = AutoTokenizer.from_pretrained(LANGUAGE_MODEL_ID)
    image_processor = CLIPImageProcessor(
        size={"height": 252, "width": 448},
        crop_size={"height": 252, "width": 448},
        do_normalize=False,
        do_rescale=True,
        rescale_factor=1/255.0,
    )

    # # Add vision special tokens if not already present
    # special_tokens_to_add = [
    #     "<|vision_start|>",
    #     "<|vision_end|>",
    #     "<|im_start|>",
    #     "<|im_end|>",
    # ]
    # existing_tokens = set(tokenizer.get_vocab().keys())
    # new_special = [t for t in special_tokens_to_add if t not in existing_tokens]
    # if new_special:
    #     tokenizer.add_tokens(new_special, special_tokens=True)
    #     print(f"  Added {len(new_special)} vision special tokens")

    # Load LFM2.5's native chat template from the tokenizer
    lfm_tokenizer = AutoTokenizer.from_pretrained(LANGUAGE_MODEL_ID)
    chat_template = lfm_tokenizer.chat_template

    # Processor adds trajectory history + action reasoning + vision special
    # tokens to the tokenizer. No discrete VQ-VAE tokens are added — the LLM
    # predicts action_reasoning as natural language text.
    processor = DFQVLAProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        chat_template=chat_template,
        vision_config=dfq_config.vision_config,
        use_flex_scene_encoder=dfq_config.use_flex_scene_encoder,
        num_scene_tokens=dfq_config.num_scene_tokens,
    )
    print(f"  Final vocab size: {len(tokenizer)}")

    # Update config with special token IDs from processor
    dfq_config.traj_token_ids = processor.traj_token_ids
    dfq_config.action_reasoning_token_ids = processor.action_reasoning_token_ids
    
    # Set vision token IDs now that tokenizer is initialized
    dfq_config.image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    dfq_config.vision_start_token_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    dfq_config.vision_end_token_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")

    # =========================================================================
    # 6. Resize model embeddings to match expanded tokenizer
    # =========================================================================
    model.resize_token_embeddings(len(tokenizer))
    print(f"  Model embeddings resized to: {len(tokenizer)}")

    # =========================================================================
    # 7. Save everything
    # =========================================================================
    print(f"\nSaving to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print("Done!")


if __name__ == "__main__":
    main()
