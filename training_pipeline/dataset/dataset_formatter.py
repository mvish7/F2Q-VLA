import random
from typing import Dict, List, Any

# Default system message
DEFAULT_SYSTEM_MESSAGE = """You are a Vision Language Model specialized in image captioning task. Please be thorough and 
descriptive in your captions. Your task is to analyze the provided image and respond with a detailed caption that
describes the content of the image."""

# List of user prompts for variation
CAPTION_PROMPTS = [
    "please describe in detail what you observe in this image?",
    "give a detailed explanation of the scene in this picture.",
    "could you summarize the contents of this image in a great detail?",
    "how would you describe this image at length to someone who can't see it?",
    "provide a detailed summary of what this picture shows.",
    "in a few sentences, explain what's happening in this image.",
    "could you briefly explain the scene captured in this image?",
    "describe the key elements and details visible in this picture.",
    "offer a detailed description of what you notice in this image.",
    "in 4-5 sentences, summarize various aspects of this photo.",
    "give a complete overview of what this image is about.",
    "could you detail the important parts you observe in this picture?",
    "please share a precise summary of this image.",
    "write a few lines describing what you see in this photograph.",
    "summarize the overall scene depicted in this image.",
    "in 4-5 sentences, describe the important features of this image.",
    "briefly describe what is happening in this picture.",
    "what do you see in this image? describe it in a few sentences.",
    "explain what this image portrays in a short paragraph.",
    "give a clear and detailed description of this photo."
]

def get_fields_from_sample(sample: Dict[str, Any]):
    """Extract image path, user text, and assistant text from a sample."""
    dataset_source = sample.get("texts", None)
    if isinstance(dataset_source, list):
        # localized narratives dataset
        return sample["image_path"], sample["texts"][0]["user"], sample["texts"][0]["assistant"]
    else:
        # pixmo dataset
        return sample["image_path"], random.choice(CAPTION_PROMPTS), sample["caption"]

def format_data(sample: Dict[str, Any], system_message: str = DEFAULT_SYSTEM_MESSAGE) -> List[Dict[str, Any]]:
    """Format a sample into a conversation list."""
    # fetch info from sample
    image_path, user_text, assistant_text = get_fields_from_sample(sample)

    # format the message
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {
                    "type": "text",
                    "text": user_text
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": assistant_text}],
        },
    ]

# Trajectory tokens (mirrored from f2q_vla/traj_utils.py to avoid circular imports if needed, 
# or we can import if the package is installed. For safety in this script, defining here.)
TRAJ_TOKEN = {
    "history": "<|traj_history|>",
    "history_start": "<|traj_history_start|>",
    "history_end": "<|traj_history_end|>",
}

def format_vla_data(sample: Dict[str, Any], use_flex: bool = False) -> List[Dict[str, Any]]:
    """Format a VLA sample into a conversation list for F2Q VLA.
    
    Args:
        sample: Raw dataset sample.
        use_flex: If True, use single image placeholder for Flex Scene Encoder.
                  If False, use per-image placeholders (16 total).
    
    Returns:
        Conversation list for chat template.
    """
    # 1. System Prompt
    system_msg = "You are a driving assistant that generates safe and accurate actions."
    
    # 2. User Prompt Components
    user_content = []
    
    # a. Images from image_paths
    if "image_paths" in sample:
        if use_flex:
            # Flex mode: Single image placeholder for entire scene
            # All images are still loaded, but represented by one token block
            # The Flex encoder compresses them into K scene tokens
            # We pick first image path as placeholder (collator loads all images)
            first_path = None
            for cam_name, paths in sample["image_paths"].items():
                if paths:
                    first_path = paths[0]
                    break
            if first_path:
                user_content.append({
                    "type": "image",
                    "image": first_path,  # Placeholder - collator loads all images
                })
        else:
            # Legacy mode: Per-image placeholders (4 cameras × 4 timestamps = 16)
            # Order: Camera, then Time
            for cam_name, paths in sample["image_paths"].items():
                for path in paths:
                    user_content.append({
                        "type": "image",
                        "image": path,
                    })
    
    # b. Trajectory History Placeholder
    # Default 48 tokens (16 steps * 3 dims)
    num_traj_tokens = 48 
    hist_traj_placeholder = (
        f"{TRAJ_TOKEN['history_start']}"
        f"{TRAJ_TOKEN['history'] * num_traj_tokens}"
        f"{TRAJ_TOKEN['history_end']}"
    )
    
    # c. Text Prompt
    user_text = "output the chain-of-thought reasoning of the driving process, then output the future trajectory."
    
    user_content.append({
        "type": "text",
        "text": f"{hist_traj_placeholder}{user_text}"
    })

    # 3. Assistant Target (Reasoning)
    # We want the model to learn to output: <|cot_start|> reasoning
    coc_reasoning = sample.get("coc_reasoning", "")
    # Handle case where coc_reasoning is a list (e.g., from inference output)
    if isinstance(coc_reasoning, list):
        coc_reasoning = coc_reasoning[0] if coc_reasoning else ""
    assistant_text = "<|cot_start|> " + coc_reasoning
    
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
