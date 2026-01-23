"""Trajectory utility functions for F2Q VLA.

This module provides utility functions for:
- Creating VLA messages with trajectory placeholders
- Tokenizing trajectory data
- Fusing trajectory tokens with input IDs
"""

from typing import Any

import einops
import torch


# Trajectory token constants (matching processor)
TRAJ_TOKEN = {
    "history": "<|traj_history|>",
    "history_start": "<|traj_history_start|>",
    "history_end": "<|traj_history_end|>",
}


def create_vla_message(
    frames: torch.Tensor,
    num_traj_tokens: int = 48,
    system_prompt: str = "You are a driving assistant that generates safe and accurate actions.",
    user_prompt: str = "output the chain-of-thought reasoning of the driving process, then output the future trajectory.",
) -> list[dict]:
    """Create a VLA message with image frames and trajectory placeholders.
    
    Similar to alpamayo_r1's helper.create_message().
    
    Args:
        frames: Image frames tensor of shape (N, C, H, W).
        num_traj_tokens: Number of trajectory placeholder tokens (default 48 for 16 waypoints × 3 xyz).
        system_prompt: System message content.
        user_prompt: User prompt after trajectory placeholders.
    
    Returns:
        List of message dicts in chat format.
    """
    assert frames.ndim == 4, f"{frames.ndim=}, expected (N, C, H, W)"

    # Create trajectory placeholder string
    hist_traj_placeholder = (
        f"{TRAJ_TOKEN['history_start']}"
        f"{TRAJ_TOKEN['history'] * num_traj_tokens}"
        f"{TRAJ_TOKEN['history_end']}"
    )

    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_prompt,
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "image", "image": frame} for frame in frames]
            + [
                {
                    "type": "text",
                    "text": f"{hist_traj_placeholder}{user_prompt}",
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "<|cot_start|>",
                }
            ],
        },
    ]


def tokenize_history_trajectory(
    tokenizer: Any,
    traj_data: dict[str, Any],
    start_idx: int = 0,
) -> torch.Tensor:
    """Tokenize the history trajectory.
    
    Args:
        tokenizer: Trajectory tokenizer with encode method (DeltaTrajectoryTokenizer).
        traj_data: Dict containing "ego_history_xyz" and "ego_history_rot".
                   ego_history_xyz shape: (B, n_traj, T, 3)
                   ego_history_rot shape: (B, n_traj, T, 3, 3)
        start_idx: Start of token index for the history trajectory tokens.

    Returns:
        torch.Tensor: Tokenized trajectory indices of shape [B, n_traj * tokens_per_history_traj].
    """
    assert "ego_history_xyz" in traj_data
    assert traj_data["ego_history_xyz"].ndim == 4, "ego_history_xyz must be 4D of [B, n_traj, T, 3]"

    B = traj_data["ego_history_xyz"].shape[0]
    hist_xyz = traj_data["ego_history_xyz"].flatten(start_dim=0, end_dim=1)
    hist_rot = traj_data["ego_history_rot"].flatten(start_dim=0, end_dim=1)

    # Encode history trajectory using delta tokenizer
    # Note: hist_xyz is passed to fut_xyz as we're encoding the history itself
    hist_idx = (
        tokenizer.encode(
            hist_xyz=hist_xyz[:, :1],
            hist_rot=hist_rot[:, :1],
            fut_xyz=hist_xyz,
            fut_rot=hist_rot,
        )
        + start_idx
    )  # [B*n_traj, tokens_per_history_traj]
    
    hist_idx = einops.rearrange(hist_idx, "(b n_traj) n -> b (n_traj n)", b=B)

    return hist_idx


def replace_pad_token(
    input_ids: torch.Tensor,
    new_ids: torch.Tensor,
    pad_idx: int,
) -> torch.Tensor:
    """Replace pad tokens in input_ids with new token values.
    
    Args:
        input_ids: Input token IDs of shape [B, L].
        new_ids: New token IDs to insert of shape [B, N] where N is number of pad tokens.
        pad_idx: The pad token ID to replace.
    
    Returns:
        torch.Tensor: Modified input_ids with pad tokens replaced.
    """
    mask = input_ids == pad_idx
    return input_ids.masked_scatter(mask, new_ids)


class TrajectoryFusionMixin:
    """Mixin class providing trajectory fusion functionality.
    
    This mixin should be used with F2QVLAForConditionalGeneration to add
    trajectory encoding and fusion capabilities.
    """

    def fuse_traj_tokens(
        self,
        input_ids: torch.Tensor,
        traj_data: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Fuse trajectory tokens into the input ids.
        
        Replaces <|traj_history|> placeholder tokens with actual trajectory token IDs.

        Args:
            input_ids: Input token IDs of shape [B, L].
            traj_data: Dict containing ego_history_xyz, ego_history_rot.

        Returns:
            input_ids: Modified input_ids with trajectory tokens fused.
        """
        if (
            traj_data is None
            or traj_data.get("ego_history_xyz") is None
            or traj_data.get("ego_history_rot") is None
        ):
            return input_ids

        # Validate required attributes
        if not hasattr(self, "hist_traj_tokenizer"):
            raise AttributeError("TrajectoryFusionMixin requires 'hist_traj_tokenizer' attribute")
        if not hasattr(self, "hist_token_start_idx"):
            raise AttributeError("TrajectoryFusionMixin requires 'hist_token_start_idx' attribute")
        if not hasattr(self.config, "traj_token_ids"):
            raise AttributeError("Config requires 'traj_token_ids' attribute")

        # Tokenize history trajectory
        hist_idx = tokenize_history_trajectory(
            self.hist_traj_tokenizer, traj_data, self.hist_token_start_idx
        )
        
        # Replace placeholder tokens with actual trajectory tokens
        input_ids = replace_pad_token(
            input_ids, hist_idx, self.config.traj_token_ids["history"]
        )

        return input_ids
