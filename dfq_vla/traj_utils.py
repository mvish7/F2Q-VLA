"""Trajectory utility functions for DFQ VLA.

This module provides utility functions for:
- Creating VLA messages with trajectory placeholders
- Fusing trajectory embeddings with input embeddings
"""

from typing import Any

import torch

from dfq_vla.trajectory_projector import prepare_traj_input


# Trajectory token constants (history only — future is predicted as text)
TRAJ_TOKEN = {
    "history": "<|traj_history|>",
    "history_start": "<|traj_history_start|>",
    "history_end": "<|traj_history_end|>",
}


def create_vla_message(
    frames: torch.Tensor,
    num_traj_tokens: int = 16,
    system_prompt: str = "You are a driving assistant that generates safe and accurate actions.",
    user_prompt: str = "Analyze the driving scene and describe the intended driving action.",
) -> list[dict]:
    """Create a VLA message with image frames and trajectory placeholders.
    
    Args:
        frames: Image frames tensor of shape (N, C, H, W).
        num_traj_tokens: Number of trajectory placeholder tokens (default 16, one per waypoint).
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


class TrajectoryFusionMixin:
    """Mixin class providing trajectory embedding fusion.
    
    This mixin should be used with DFQVLAForConditionalGeneration to add
    trajectory projection and embedding fusion capabilities.
    """

    def fuse_traj_embeddings(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        ego_history_xyz: torch.Tensor,
        ego_history_rot: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse projected trajectory embeddings into the input embeddings.
        
        Projects raw trajectory data through TrajHistProjector and scatters
        the resulting embeddings at <|traj_history|> placeholder positions,
        similar to how image embeddings are fused.

        Args:
            input_ids: Input token IDs of shape [B, L].
            inputs_embeds: Current input embeddings of shape [B, L, D].
            ego_history_xyz: History positions of shape [B, T, 3].
            ego_history_rot: History rotations of shape [B, T, 3, 3].

        Returns:
            inputs_embeds: Modified embeddings with trajectory embeddings fused in.
        """
        if not hasattr(self, "traj_projector") or self.traj_projector is None:
            return inputs_embeds  # No-op: trajectory projector excluded
        if not hasattr(self.config, "traj_token_ids"):
            raise AttributeError("Config requires 'traj_token_ids' attribute")

        # Get the trajectory history placeholder token ID
        traj_history_token_id = self.config.traj_token_ids["history"]
        
        # Build mask for <|traj_history|> placeholder positions
        traj_mask = input_ids == traj_history_token_id  # [B, L]
        
        if not traj_mask.any():
            return inputs_embeds

        # Prepare input: concatenate xyz + yaw → (B, T, 4)
        traj_input = prepare_traj_input(
            ego_history_xyz.to(inputs_embeds.dtype),
            ego_history_rot.to(inputs_embeds.dtype),
        )
        
        # Project through MLP → (B, T, hidden_dim)
        traj_embeds = self.traj_projector(traj_input)

        # Scatter into inputs_embeds at placeholder positions
        traj_mask_expanded = traj_mask.unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(
            traj_mask_expanded, traj_embeds.to(inputs_embeds.dtype)
        )

        return inputs_embeds
