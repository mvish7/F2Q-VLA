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
