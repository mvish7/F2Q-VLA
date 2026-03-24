"""Geometry utilities for DFQ VLA.

This module provides rotation representation conversion functions.
"""

import torch
import torch.nn.functional as F


def rotmat_to_rot2d(rot_matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to 2D continuous yaw representation [cos(yaw), sin(yaw)].
    
    Args:
        rot_matrix: Rotation matrices of shape (..., 3, 3).
        
    Returns:
        2D rotation representation of shape (..., 2).
    """
    yaw = torch.atan2(rot_matrix[..., 1, 0], rot_matrix[..., 0, 0])
    return torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)


def rot2d_to_yaw(rot2d: torch.Tensor) -> torch.Tensor:
    """Convert 2D continuous yaw representation [cos(yaw), sin(yaw)] to scalar yaw angle.
    
    Args:
        rot2d: 2D rotation representation of shape (..., 2).
        
    Returns:
        Yaw angles of shape (..., 1).
    """
    yaw = torch.atan2(rot2d[..., 1], rot2d[..., 0])
    return yaw.unsqueeze(-1)
