import random
from typing import Any, Dict
import torch


def get_fields_from_sample(sample: Dict[str, Any]):
    """Extract image path, user text, and assistant text from a sample."""
    dataset_source = sample.get("texts", None)
    if isinstance(dataset_source, list):
        # localized narratives dataset
        return sample["image_path"], sample["texts"][0]["user"], sample["texts"][0]["assistant"]
    else:
        # pixmo dataset
        return sample["image_path"], random.choice(CAPTION_PROMPTS), sample["caption"]

# Trajectory tokens (history only — future is predicted as natural language text)
TRAJ_TOKEN = {
    "history": "<|traj_history|>",
    "history_start": "<|traj_history_start|>",
    "history_end": "<|traj_history_end|>",
}

def format_vla_data(sample: Dict[str, Any], use_flex: bool = False,
                    sample_image: torch.Tensor = None, include_traj_history: bool = True,
                    num_traj_tokens: int = 16) -> list[Dict[str, Any]]:
    """Format a VLA sample into a conversation list for DFQ VLA.
    
    The assistant target is the `action_reasoning` text from the dataset,
    which the LLM learns to predict as natural language tokens.
    
    Args:
        sample: Raw dataset sample. Must contain 'action_reasoning' key.
        use_flex: If True, use single image placeholder for Flex Scene Encoder.
                  If False, use per-image placeholders (16 total).
        sample_image: Sample image tensor for placeholder.
        include_traj_history: If True, include trajectory history placeholder tokens.
                              Set to False when traj_projector is excluded.
        num_traj_tokens: Number of history trajectory placeholder tokens.
    
    Returns:
        Conversation list for chat template.
    """
    # 1. System Prompt
    system_msg = "You are an expert self-driving system. Analyze the driving scene and describe the intended driving action."
    
    # 2. User Prompt Components
    user_content = []
    
    user_content.append({
        "type": "image",
        "image": sample_image,  # Placeholder - collator loads all images
    })
    
    # b. Trajectory History Placeholder (only when traj_projector is included)
    user_text = "Analyze the driving scene and describe the intended driving action."
    
    if include_traj_history:
        hist_traj_placeholder = (
            f"{TRAJ_TOKEN['history_start']}"
            f"{TRAJ_TOKEN['history'] * num_traj_tokens}"
            f"{TRAJ_TOKEN['history_end']}"
        )
        user_text = f"{hist_traj_placeholder}{user_text}"
    
    user_content.append({
        "type": "text",
        "text": user_text
    })

    # 3. Assistant Target — action reasoning text from dataset
    action_reasoning = sample.get("action_reasoning", "")
    
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_msg}],
        },
        {
            "role": "user",
            "content": user_content,
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": action_reasoning}],
        },
    ]
