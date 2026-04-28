"""Multi-sample trajectory inference for evaluation.

Generates K diverse trajectory predictions per batch by:
1. Running model.generate() with sampling K times
2. Extracting VQ-VAE codebook indices from generated tokens
3. Decoding via VQ-VAE → coarse trajectory
4. Refining via ActionChunkingHead → final trajectory
"""

import torch
from torch import Tensor


def _extract_vqvae_indices(
    generated_ids: Tensor,
    future_start_token_id: int,
    traj_token_start_idx: int,
    traj_vocab_size: int,
    num_indices: int = 8,
) -> Tensor | None:
    """Extract VQ-VAE codebook indices from generated token IDs.

    Finds the <|traj_future_start|> marker and reads the next `num_indices`
    tokens, converting from token vocabulary space back to codebook space.

    Args:
        generated_ids: Token IDs from model.generate(), shape [B, S].
        future_start_token_id: Token ID for <|traj_future_start|>.
        traj_token_start_idx: Start of trajectory token range in vocabulary.
        traj_vocab_size: Size of the VQ-VAE codebook.
        num_indices: Number of codebook indices to extract (default 8).

    Returns:
        Tensor of shape [B, num_indices] with codebook indices,
        or None if extraction failed for any sample in the batch.
    """
    B = generated_ids.shape[0]
    indices = torch.zeros(B, num_indices, dtype=torch.long, device=generated_ids.device)

    for b in range(B):
        matches = (generated_ids[b] == future_start_token_id).nonzero(as_tuple=True)[0]
        if len(matches) == 0:
            return None

        start_pos = matches[-1].item()
        for i in range(num_indices):
            pos = start_pos + 1 + i
            if pos >= generated_ids.shape[1]:
                return None
            token = generated_ids[b, pos].item()
            if traj_token_start_idx <= token < traj_token_start_idx + traj_vocab_size:
                indices[b, i] = token - traj_token_start_idx
            else:
                return None

    return indices


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
        plus the truncated input_ids (needed later for index extraction offset).
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

    # Fuse trajectory history
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
        "_input_ids": padded_ids,  # kept for index extraction, not passed to generate
    }


@torch.no_grad()
def generate_trajectory_samples(
    model,
    batch: dict[str, Tensor],
    processor,
    num_samples: int = 6,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 32,
) -> list[dict[str, Tensor]]:
    """Generate K trajectory samples for a batch via diverse sampling.

    For each sample k:
    1. model.generate() with temperature sampling → token sequence
    2. Extract VQ-VAE indices from generated tokens
    3. Decode via VQ-VAE → coarse trajectory [B, 64, 5]
    4. Run action head with VLM context → refined trajectory

    Args:
        model: DFQVLAForConditionalGeneration model in eval mode.
        batch: Batch dict from DataCollator.
        processor: Model processor for tokenizer access.
        num_samples: Number of trajectory samples K to generate.
        temperature: Sampling temperature (higher = more diverse).
        top_p: Nucleus sampling threshold.
        max_new_tokens: Max tokens to generate per sample.

    Returns:
        List of K dicts, each containing:
            - "xyz": [B, 64, 3] predicted positions
            - "rot2d": [B, 64, 2] predicted rotations
        Failed samples are excluded (list may be shorter than K).
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # Pre-compute fused embeddings once (image + trajectory baked in)
    precomputed = _build_generation_inputs(batch, model, processor)
    inputs_embeds = precomputed["inputs_embeds"]
    attention_mask = precomputed["attention_mask"]
    prompt_input_ids = precomputed["_input_ids"]

    # Model config for trajectory token extraction
    config = model.config
    future_start_id = config.traj_token_ids["future_start"]
    traj_start_idx = config.traj_token_start_idx
    traj_vocab_size = getattr(config, "traj_vocab_size", 768)

    # Batch metadata for the post-generation forward pass
    fwd_extras = {}
    for key in ["pixel_values", "camera_ids", "timestamp_ids",
                "ego_history_xyz", "ego_history_rot"]:
        if key in batch:
            fwd_extras[key] = batch[key].to(device)

    trajectory_samples = []

    for k in range(num_samples):
        # 1. Generate tokens with sampling
        #    Pass inputs_embeds (not input_ids) — images & trajectory are already fused.
        gen_output = model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            return_dict_in_generate=True,
        )

        generated_ids = gen_output.sequences  # [B, generated_len]

        # When generate() receives inputs_embeds (not input_ids), the returned
        # sequences contain ONLY the newly generated tokens — no dummy prompt
        # prefix.  Reconstruct full_ids = real prompt IDs + generated tokens so
        # that image/traj placeholder tokens are present for the forward pass.
        new_tokens = generated_ids  # already just the generated part
        full_ids = torch.cat([prompt_input_ids, new_tokens], dim=1)

        # 2. Extract VQ-VAE indices from the full reconstructed sequence
        vqvae_indices = _extract_vqvae_indices(
            full_ids, future_start_id, traj_start_idx, traj_vocab_size
        )

        if vqvae_indices is None:
            continue  # Skip failed extractions

        # 3. Decode VQ-VAE indices → coarse trajectory [B, 64, 5]
        base_traj = model.vqvae_tokenizer.decode(vqvae_indices).to(dtype).to(device)

        # 4. Refine with action head if available, otherwise use coarse VQ-VAE output
        if getattr(model, "action_head", None) is not None:
            # Full forward pass to get hidden_states for action head cross-attention
            full_mask = torch.ones_like(full_ids)
            fwd_output = model(
                input_ids=full_ids,
                attention_mask=full_mask,
                output_hidden_states=True,
                **fwd_extras,
            )
            vlm_context = fwd_output.hidden_states[-1]  # [B, S, D]

            memory_key_padding_mask = ~full_mask.bool()
            traj_output = model.predict_future_trajectory(
                vlm_context,
                base_traj=base_traj,
                normalize_rot=True,
                attention_mask=memory_key_padding_mask,
            )
            # Action head predicts in normalized space (targets /100 during training)
            # Scale back to raw meters for eval
            traj_output["xyz"] = traj_output["xyz"] * 100.0
        else:
            # No action head — use raw VQ-VAE coarse trajectory directly
            # base_traj is [B, 64, 5] with dims (x, y, z, sin, cos)
            traj_output = {
                "xyz": base_traj[:, :, :3],    # [B, 64, 3]
                "rot2d": base_traj[:, :, 3:],  # [B, 64, 2]
            }

        trajectory_samples.append(traj_output)

    return trajectory_samples
