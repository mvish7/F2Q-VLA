"""Geometry utilities for F2Q VLA.

This module provides rotation matrix conversion functions.
"""

import torch
import torch.nn.functional as F


def compute_rotation_matrix_from_ortho6d(ortho6d: torch.Tensor) -> torch.Tensor:
    """Compute rotation matrix from 6D continuous rotation representation.
    
    Uses Gram-Schmidt orthogonalization to convert a 6D vector to a 3x3 rotation matrix.
    Based on "On the Continuity of Rotation Representations in Neural Networks" (Zhou et al., CVPR 2019).
    
    Args:
        ortho6d: 6D rotation representation of shape (..., 6).
        
    Returns:
        Rotation matrices of shape (..., 3, 3).
    """
    # Split into two 3D vectors
    x_raw = ortho6d[..., :3]  # First column
    y_raw = ortho6d[..., 3:6]  # Second column hint
    
    # Normalize first column
    x = F.normalize(x_raw, dim=-1)
    
    # Gram-Schmidt: make y orthogonal to x
    dot = (x * y_raw).sum(dim=-1, keepdim=True)
    y = y_raw - dot * x
    y = F.normalize(y, dim=-1)
    
    # Cross product for third column
    z = torch.cross(x, y, dim=-1)
    
    # Stack into rotation matrix
    # Each column is a basis vector
    rot_matrix = torch.stack([x, y, z], dim=-1)  # (..., 3, 3)
    
    return rot_matrix


def rotation_matrix_to_ortho6d(rot_matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to 6D continuous representation.
    
    Args:
        rot_matrix: Rotation matrices of shape (..., 3, 3).
        
    Returns:
        6D rotation representation of shape (..., 6).
    """
    # Take first two columns (x and y vectors)
    # rot_matrix shape: (..., 3, 3) where columns are [x, y, z]
    x = rot_matrix[..., :, 0]  # First column: (..., 3)
    y = rot_matrix[..., :, 1]  # Second column: (..., 3)
    return torch.cat([x, y], dim=-1)  # (..., 6)

