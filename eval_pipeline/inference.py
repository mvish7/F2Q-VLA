"""Text-based trajectory inference for evaluation.

Generates action reasoning text from the model by:
1. Running model.generate() with sampling
2. Decoding the generated text tokens
"""

import torch
from torch import Tensor


def _build_generation_inputs(
    batch: dict[str, Tensor], model, processor
) -> dict[str, Tensor]:
    """Build generation-ready inputs from an eval batch.

    The training collator produces batches with full input_ids (including the
    assistant response). For generation we truncate to the prompt, then
    pre-compute fused inputs_embeds (image + trajectory already baked in)
    so that model.generate() doesn't receive unsupported kwargs.

    Args:
        batch: Batch dict from DataCollator.
        model: The VLA model (or PEFT-wrapped model).
        processor: The model's processor (for tokenizer access).

    Returns:
        Dict with generation-ready inputs: inputs_embeds, attention_mask,
        plus the truncated input_ids (needed later for context).
    """
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]

    # --- Truncate to prompt (cut before assistant response) ---
    truncated_ids = []
    truncated_masks = []

    for i in range(input_ids.shape[0]):
        valid_positions = (labels[i] != -100).nonzero(as_tuple=True)[0]
        cut_pos = valid_positions[0].item() if len(valid_positions) > 0 else input_ids.shape[1]
        truncated_ids.append(input_ids[i, :cut_pos])
        truncated_masks.append(attention_mask[i, :cut_pos])

    # Pad to equal length
    max_len = max(t.shape[0] for t in truncated_ids)
    pad_id = processor.tokenizer.pad_token_id
    device = input_ids.device

    padded_ids = torch.full((input_ids.shape[0], max_len), pad_id, dtype=input_ids.dtype, device=device)
    padded_mask = torch.zeros((input_ids.shape[0], max_len), dtype=attention_mask.dtype, device=device)

    for i in range(input_ids.shape[0]):
        seq_len = truncated_ids[i].shape[0]
        padded_ids[i, :seq_len] = truncated_ids[i]
        padded_mask[i, :seq_len] = truncated_masks[i]

    # --- Move to model device early ---
    device = next(model.parameters()).device
    padded_ids = padded_ids.to(device)
    padded_mask = padded_mask.to(device)

    # --- Pre-compute fused inputs_embeds ---
    # This replicates the model's forward-pass embedding fusion (image + traj)
    # so we can pass inputs_embeds directly to generate() without the rejected kwargs.
    inputs_embeds = model.get_input_embeddings()(padded_ids)

    # Fuse image features
    pixel_values = batch.get("pixel_values")
    if pixel_values is not None:
        pixel_values = pixel_values.to(device)
        target_dtype = model.get_input_embeddings().weight.dtype
        vision_outputs = model.vision_tower(
            pixel_values.to(target_dtype), output_hidden_states=True
        )
        image_embeds = vision_outputs.last_hidden_state

        # Flex scene encoding
        camera_ids = batch.get("camera_ids")
        timestamp_ids = batch.get("timestamp_ids")
        if model.flex_scene_encoder is not None and camera_ids is not None:
            camera_ids = camera_ids.to(device)
            timestamp_ids = timestamp_ids.to(device)
            B = camera_ids.shape[0]
            num_images = camera_ids.shape[1]
            N, D = image_embeds.shape[1], image_embeds.shape[2]
            image_embeds = image_embeds.view(B, num_images, N, D)
            image_embeds = model.flex_scene_encoder(
                image_embeds, camera_ids, timestamp_ids
            )

        image_embeds = model.projector(image_embeds)
        image_mask = model.get_placeholder_mask(padded_ids, inputs_embeds, image_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(
            image_mask, image_embeds.to(inputs_embeds.dtype)
        )

    # Fuse trajectory history (only when traj_projector is present)
    if getattr(model, "traj_projector", None) is not None:
        ego_history_xyz = batch.get("ego_history_xyz")
        ego_history_rot = batch.get("ego_history_rot")
        if ego_history_xyz is not None and ego_history_rot is not None:
            inputs_embeds = model.fuse_traj_embeddings(
                input_ids=padded_ids,
                inputs_embeds=inputs_embeds,
                ego_history_xyz=ego_history_xyz.to(device),
                ego_history_rot=ego_history_rot.to(device),
            )

    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": padded_mask,
        "_input_ids": padded_ids,  # kept for context, not passed to generate
    }


@torch.no_grad()
def generate_action_reasoning(
    model,
    batch: dict[str, Tensor],
    processor,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = False,
) -> list[str]:
    """Generate action reasoning text for a batch.

    Args:
        model: DFQVLAForConditionalGeneration model in eval mode.
        batch: Batch dict from DataCollator.
        processor: Model processor for tokenizer access.
        max_new_tokens: Max tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        do_sample: Whether to use sampling or greedy decoding.

    Returns:
        List of generated action reasoning text strings, one per batch element.
    """
    # Pre-compute fused embeddings once (image + trajectory baked in)
    precomputed = _build_generation_inputs(batch, model, processor)
    inputs_embeds = precomputed["inputs_embeds"]
    attention_mask = precomputed["attention_mask"]

    # Generate text tokens
    gen_output = model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        return_dict_in_generate=True,
    )

    generated_ids = gen_output.sequences  # [B, generated_len]

    # Decode generated tokens to text
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

    return generated_texts
