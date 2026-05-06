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

# Trajectory tokens (mirrored from dfq_vla/traj_utils.py to avoid circular imports if needed, 
# or we can import if the package is installed. For safety in this script, defining here.)
TRAJ_TOKEN = {
    "history": "<|traj_history|>",
    "history_start": "<|traj_history_start|>",
    "history_end": "<|traj_history_end|>",
    "future_start": "<|traj_future_start|>",
    "future_end": "<|traj_future_end|>",
}

def format_vla_data(sample: Dict[str, Any], vqvae_indices: list[int], use_flex: bool = False,
                    sample_image:torch.Tensor = None, include_traj_history: bool = True) -> list[Dict[str, Any]]:
    """Format a VLA sample into a conversation list for DFQ VLA.
    
    Args:
        sample: Raw dataset sample.
        use_flex: If True, use single image placeholder for Flex Scene Encoder.
                  If False, use per-image placeholders (16 total).
        include_traj_history: If True, include trajectory history placeholder tokens.
                              Set to False when traj_projector is excluded.
    
    Returns:
        Conversation list for chat template.
    """
    # 1. System Prompt
    system_msg = "You are an expert self-driving system that generates safe and accurate future driving trajectories."
    
    # 2. User Prompt Components
    user_content = []
    
    # # a. Images from image_paths
    # if "image_paths" in sample:
    #     if use_flex:
    #         # Flex mode: Single image placeholder for entire scene
    #         # All images are still loaded, but represented by one token block
    #         # The Flex encoder compresses them into K scene tokens
    #         # We pick first image path as placeholder (collator loads all images)
    #         first_path = None
    #         for cam_name, paths in sample["image_paths"].items():
    #             if paths:
    #                 first_path = paths[0]
    #                 break
    #         if first_path:
    #             user_content.append({
    #                 "type": "image",
    #                 "image": first_path,  # Placeholder - collator loads all images
    #             })
    #     else:
    #         # Legacy mode: Per-image placeholders (4 cameras × 4 timestamps = 16)
    #         # Order: Camera, then Time
    #         for cam_name, paths in sample["image_paths"].items():
    #             for path in paths:
    #                 user_content.append({
    #                     "type": "image",
    #                     "image": path,
    #                 })
    
    user_content.append({
        "type": "image",
        "image": sample_image,  # Placeholder - collator loads all images
    })
    
    # b. Trajectory History Placeholder (only when traj_projector is included)
    user_text = "By analyzing the given images and the past trajectory, predict 8 discrete ids corresponding to the future trajectory."
    
    if include_traj_history:
        num_traj_tokens = 16
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

    # 3. Assistant Target — VQ-VAE trajectory indices
    traj_tokens = "".join(f"<i{idx}>" for idx in vqvae_indices)
    assistant_text = f"{TRAJ_TOKEN['future_start']}{traj_tokens}{TRAJ_TOKEN['future_end']}"
    assistant_text = assistant_text.replace(" ", "")
    
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
            "content": [{"type": "text", "text": assistant_text}],
        },
    ]
