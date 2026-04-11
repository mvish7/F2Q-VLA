"""Unified frozen VQ-VAE wrapper for trajectory encoding and decoding.

This module provides `VQVAETrajectoryTokenizer`, a frozen nn.Module wrapper around
the `TrajectoryVQVAE` model. It handles channel reordering and exposes cleanly:
  - `encode(xyz, rot)` for dataset collation (CPU)
  - `decode(indices)` for the VLA model Action Head prior (GPU)
"""

import sys
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

# Add VQ-VAE submodule to import path dynamically
_VQVAE_ROOT = str(Path(__file__).resolve().parents[2] / "VQ-VAE")
if _VQVAE_ROOT not in sys.path:
    sys.path.insert(0, _VQVAE_ROOT)

from model.vqvae import TrajectoryVQVAE


class VQVAETrajectoryTokenizer(nn.Module):
    """Frozen VQ-VAE tokenizer for continuous trajectories."""

    def __init__(
        self,
        checkpoint_path: str,
        num_embeddings: int = 768,
        hidden_dim: int = 256,
        embedding_dim: int = 256,
    ):
        """Initialize the VQ-VAE tokenizer from a checkpoint.

        Args:
            checkpoint_path: Absolute path to the VQ-VAE .pt checkpoint.
            num_embeddings: Codebook size K (must match trained model).
            hidden_dim: Encoder/decoder hidden dim (must match trained model).
            embedding_dim: Codebook embedding dim (must match trained model).
        """
        super().__init__()
        
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"VQ-VAE checkpoint not found: {checkpoint_path}")

        # Instantiate raw VQ-VAE model
        self.model = TrajectoryVQVAE(
            in_channels=5,
            hidden_dim=hidden_dim,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
        )

        # Load checkpoint weights
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        # Handle dict wrapping and torch.compile _orig_mod prefixes
        state_dict = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
        state_dict = {
            k.replace("_orig_mod.", ""): v
            for k, v in state_dict.items()
        }
        self.model.load_state_dict(state_dict, assign=True)
        self.model.to(torch.bfloat16)

        # Freeze completely
        self.eval()
        for param in self.parameters():
            param.requires_grad_(False)

    def train(self, mode: bool = True):
        """Override train to ensure it always stays in eval mode."""
        return super().train(False)

    @torch.no_grad()
    def encode(self, xyz: Tensor, rot: Tensor) -> list[int]:
        """Encode a single trajectory into 8 codebook indices.
        
        Typically used by DataCollator on CPU.

        Args:
            xyz: Position tensor of shape (T, 3) — [x, y, z].
            rot: Rotation tensor of shape (T, 2) — [cos_yaw, sin_yaw]
                 (collator convention).

        Returns:
            List of 8 integer codebook indices, each in [0, K).
        """
        # Reorder rot from [cos, sin] → VQ-VAE's [sin, cos]
        sin_yaw = rot[:, 1:2]  # (T, 1)
        cos_yaw = rot[:, 0:1]  # (T, 1)

        # Build (T, 5) feature: [x, y, z, sin_yaw, cos_yaw]
        features = torch.cat([xyz, sin_yaw, cos_yaw], dim=1).to(torch.bfloat16)  # (T, 5)

        # Transpose to (1, 5, T) for VQ-VAE encoder
        features = features.T.unsqueeze(0).to(dtype=self.model.quantizer.embeddings.dtype, device=self.model.quantizer.embeddings.device)

        # Encode → (1, 8)
        indices, _ = self.model.encode(features)

        # Ensure output is standard ints
        return indices.squeeze(0).cpu().tolist()

    @torch.no_grad()
    def decode(self, indices: Tensor) -> Tensor:
        """Decode integer indices back into continuous coarse trajectories.
        
        Typically used natively inside the VLA model's forward pass on GPU.

        Args:
            indices: Integer tensor of shape (B, 8) containing codebook IDs.

        Returns:
            Decoded trajectory tensor of shape (B, 64, 5) where the last dim is 
            [x, y, z, sin, cos]. Note: order is maintained from VQ-VAE natively 
            and splits must be handled in action head.
        """
        # Ensure indices match the device of the quantizer
        device = self.model.quantizer.embeddings.device
        indices = indices.to(device)
        
        # Decode: [B, 5, 64]
        base_traj_decoded = self.model.decode_from_indices(indices)
        
        # Permute to fit Action Head sequential requirements: [B, 64, 5]
        return base_traj_decoded.permute(0, 2, 1)
