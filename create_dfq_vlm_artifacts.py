"""Create DFQ VLA model artifacts.

Assembles the complete DFQ VLA model by loading pretrained weights for:
- DINOv3 vision encoder
- Qwen3 language model
And initializing from scratch:
- Vision-to-LLM projector
- Action chunking head (future trajectory prediction)
- Flex scene encoder (multi-camera, multi-timestamp compression)
- Trajectory tokenizer (history trajectory encoding)

All components are saved together as a single HuggingFace model + processor.
"""

import json

import torch
from transformers import AutoModel, AutoConfig, AutoTokenizer, AutoImageProcessor

from dfq_vla.configuration_dfq_vla import DFQVLAConfig, VISION_MODEL_ID
from dfq_vla.modelling_dfq_vla import DFQVLAForConditionalGeneration
from dfq_vla.processing_dfq_vla import DFQVLAProcessor


# =============================================================================
# Configuration
# =============================================================================
print(VISION_MODEL_ID)
LANGUAGE_MODEL_ID = "Qwen/Qwen3-0.6B"
print(LANGUAGE_MODEL_ID)
OUTPUT_DIR = "./dfq_vla_artifacts"
CHAT_TEMPLATE_PATH = "/media/vishal/workspace/projects/VLA/qwen3_vl_chat_template.json"


def main():
    # =========================================================================
    # 1. Load sub-model configs
    # =========================================================================
    print("Loading configs...")
    vision_config = AutoConfig.from_pretrained(VISION_MODEL_ID)
    text_config = AutoConfig.from_pretrained(LANGUAGE_MODEL_ID)

    # =========================================================================
    # 2. Create DFQ VLA config (includes action head + flex encoder defaults)
    # =========================================================================
    dfq_config = DFQVLAConfig(
        vision_config=vision_config,
        text_config=text_config,
    )
    print(f"  Vision hidden size: {dfq_config.vision_hidden_size}")
    print(f"  LLM hidden size:    {dfq_config.hidden_size}")
    print(f"  Action queries:     {dfq_config.num_action_queries}")
    print(f"  Flex scene tokens:  {dfq_config.num_scene_tokens}")
    print(f"  Traj vocab size:    {dfq_config.traj_vocab_size}")
    print(f"  Traj input dim:     {dfq_config.traj_input_dim}")

    # =========================================================================
    # 3. Initialize full model (all components created from config)
    # =========================================================================
    print("\nInitializing model architecture...")
    model = DFQVLAForConditionalGeneration(dfq_config)

    # =========================================================================
    # 4. Load pretrained weights for vision encoder and LLM
    # =========================================================================
    print("\nLoading DINOv3 vision encoder weights...")
    vision_model = AutoModel.from_pretrained(VISION_MODEL_ID)
    missing, unexpected = model.vision_tower.load_state_dict(
        vision_model.state_dict(), strict=False
    )
    if missing:
        print(f"  Vision missing keys: {len(missing)}")
    if unexpected:
        print(f"  Vision unexpected keys: {len(unexpected)}")
    del vision_model

    print("Loading Qwen3 LLM weights...")
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
    print("  - Action chunking head")
    print("  - Flex scene encoder")
    print("  - Trajectory projector (MLP)")

    # =========================================================================
    # 5. Create processor (tokenizer + image processor + trajectory tokens)
    # =========================================================================
    print("\nCreating processor...")
    tokenizer = AutoTokenizer.from_pretrained(LANGUAGE_MODEL_ID)
    image_processor = AutoImageProcessor.from_pretrained(VISION_MODEL_ID)

    # Add vision special tokens if not already present
    special_tokens_to_add = [
        "<|image_pad|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|im_start|>",
        "<|im_end|>",
    ]
    existing_tokens = set(tokenizer.get_vocab().keys())
    new_special = [t for t in special_tokens_to_add if t not in existing_tokens]
    if new_special:
        tokenizer.add_tokens(new_special, special_tokens=True)
        print(f"  Added {len(new_special)} vision special tokens")

    with open(CHAT_TEMPLATE_PATH, "r") as f:
        chat_template = json.load(f)["chat_template"]

    # Processor adds trajectory tokens (discrete + special) to the tokenizer
    processor = DFQVLAProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        chat_template=chat_template,
        vision_config=dfq_config.vision_config,
        traj_vocab_size=dfq_config.traj_vocab_size,
        use_flex_scene_encoder=dfq_config.use_flex_scene_encoder,
        num_scene_tokens=dfq_config.num_scene_tokens,
    )
    print(f"  Final vocab size: {len(tokenizer)}")
    print(f"  Traj token start idx: {processor.traj_token_start_idx}")

    # =========================================================================
    # 6. Resize model embeddings to match expanded tokenizer
    # =========================================================================
    model.resize_token_embeddings(len(tokenizer))
    print(f"  Model embeddings resized to: {len(tokenizer)}")

    # Update config with trajectory token indices from processor
    dfq_config.traj_token_start_idx = processor.traj_token_start_idx
    dfq_config.traj_token_ids = processor.traj_token_ids

    # =========================================================================
    # 7. Save everything
    # =========================================================================
    print(f"\nSaving to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print("Done!")


if __name__ == "__main__":
    main()
