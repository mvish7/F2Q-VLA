"""Trajectory history projector for DFQ VLA.

Projects raw continuous trajectory history (xyz + yaw) into LLM embedding space
via a lightweight MLP, replacing the discrete DeltaTrajectoryTokenizer approach.
"""

import torch
import torch.nn as nn


class TrajHistProjector(nn.Module):
    """MLP projector for trajectory history.
    
    Takes per-waypoint features (xyz + yaw) and projects them to the
    LLM hidden dimension, producing one embedding per waypoint.
    
    Architecture mirrors DFQVLAProjector (vision projector):
        Linear(input_dim, hidden_size) → GELU → Linear(hidden_size, hidden_size)
    """

    def __init__(self, input_dim: int = 4, hidden_size: int = 1024):
        """Initialize TrajHistProjector.
        
        Args:
            input_dim: Per-waypoint feature dimension (default 4: xyz + yaw).
            hidden_size: LLM hidden dimension to project into.
        """
        super().__init__()
        self.linear_1 = nn.Linear(input_dim, hidden_size, bias=False)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, traj_features: torch.Tensor) -> torch.Tensor:
        """Project trajectory features to LLM embedding space.
        
        Args:
            traj_features: (B, T, input_dim) — e.g. (B, 16, 4) for xyz+yaw.
            
        Returns:
            (B, T, hidden_size) embeddings ready for scatter into input_embeds.
        """
        hidden = self.linear_1(traj_features)
        hidden = self.act(hidden)
        hidden = self.linear_2(hidden)
        return hidden


def extract_yaw_from_rot(rot: torch.Tensor) -> torch.Tensor:
    """Extract yaw angle from 3x3 rotation matrices.
    
    Args:
        rot: Rotation matrices of shape (..., 3, 3).
        
    Returns:
        Yaw angles of shape (..., 1).
    """
    yaw = torch.atan2(rot[..., 1, 0], rot[..., 0, 0])
    return yaw.unsqueeze(-1)


def prepare_traj_input(xyz: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    """Prepare trajectory input by concatenating xyz with yaw.
    
    Args:
        xyz: Position coordinates of shape (B, T, 3).
        rot: Rotation matrices of shape (B, T, 3, 3).
        
    Returns:
        Combined features of shape (B, T, 4).
    """
    yaw = extract_yaw_from_rot(rot)
    return torch.cat([xyz, yaw], dim=-1)
