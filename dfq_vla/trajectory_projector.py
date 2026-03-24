"""Trajectory history projector for DFQ VLA.

Projects raw continuous trajectory history (xyz + yaw) into LLM embedding space
via a lightweight MLP, replacing the discrete DeltaTrajectoryTokenizer approach.
"""

import torch
import torch.nn as nn


class TrajHistProjector(nn.Module):
    """MLP projector for trajectory history.
    
    Takes per-waypoint features (xyz + rot2d) and projects them to the
    LLM hidden dimension, producing one embedding per waypoint.
    
    Architecture mirrors DFQVLAProjector (vision projector):
        Linear(input_dim, hidden_size) → GELU → Linear(hidden_size, hidden_size)
    """

    def __init__(self, input_dim: int = 5, hidden_size: int = 1024):
        """Initialize TrajHistProjector.
        
        Args:
            input_dim: Per-waypoint feature dimension (default 5: xyz + rot2d).
            hidden_size: LLM hidden dimension to project into.
        """
        super().__init__()
        self.linear_1 = nn.Linear(input_dim, hidden_size, bias=False)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, traj_features: torch.Tensor) -> torch.Tensor:
        """Project trajectory features to LLM embedding space.
        
        Args:
            traj_features: (B, T, input_dim) — e.g. (B, 16, 5) for xyz+rot2d.
            
        Returns:
            (B, T, hidden_size) embeddings ready for scatter into input_embeds.
        """
        hidden = self.linear_1(traj_features)
        hidden = self.act(hidden)
        hidden = self.linear_2(hidden)
        return hidden


def prepare_traj_input(xyz: torch.Tensor, rot2d: torch.Tensor) -> torch.Tensor:
    """Prepare trajectory input by concatenating xyz with 2D yaw rotation.
    
    Args:
        xyz: Position coordinates of shape (B, T, 3).
        rot2d: 2D rotation representation of shape (B, T, 2).
        
    Returns:
        Combined features of shape (B, T, 5).
    """
    return torch.cat([xyz, rot2d], dim=-1)
