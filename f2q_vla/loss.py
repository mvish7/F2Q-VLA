"""Loss functions for F2Q VLA model.

This module provides the loss calculation logic for the F2Q VLA model,
including Geodesic Loss for rotation matrices and a composite loss class.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

@dataclass
class F2QVLALossOutput:
    """Output for F2QVLALoss."""
    total_loss: torch.Tensor
    text_loss: Optional[torch.Tensor] = None
    xyz_loss: Optional[torch.Tensor] = None
    rot_loss: Optional[torch.Tensor] = None


def compute_geodesic_loss(pred_rot: torch.Tensor, target_rot: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Compute Geodesic Loss between two batches of rotation matrices.
    
    Geodesic distance is defined as:
    theta = arccos((tr(R_pred^T * R_target) - 1) / 2)
    
    Args:
        pred_rot: Predicted rotation matrices of shape (..., 3, 3)
        target_rot: Target rotation matrices of shape (..., 3, 3)
        eps: Small epsilon for numerical stability in acos
        
    Returns:
        Loss tensor of shape (...)
    """
    # Calculate R_pred^T * R_target
    # shape: (..., 3, 3)
    m = torch.matmul(pred_rot.transpose(-1, -2), target_rot)
    
    # Compute trace: sum of diagonal elements
    # shape: (...)
    trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    
    # Clamp for numerical stability before acos
    # The argument for acos must be in [-1, 1]
    # (Trace of 3x3 rotation matrix is 1 + 2cos(theta), so (trace-1)/2 is cos(theta))
    cos_theta = (trace - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1.0 + eps, 1.0 - eps)
    
    # Compute angle
    theta = torch.acos(cos_theta)
    
    return theta.mean()


class F2QVLALoss(nn.Module):
    """Loss calculator for F2Q VLA."""
    
    def __init__(self, loss_weights: Dict[str, float] = None):
        super().__init__()
        if loss_weights is None:
            loss_weights = {"text": 1.0, "xyz": 1.0, "rot": 1.0}
        self.loss_weights = loss_weights
        
        # Loss functions
        self.xyz_loss_fn = nn.L1Loss(reduction='mean')
        
    def forward(
        self,
        text_loss: Optional[torch.Tensor],
        pred_xyz: Optional[torch.Tensor],
        target_xyz: Optional[torch.Tensor],
        pred_rot: Optional[torch.Tensor],
        target_rot: Optional[torch.Tensor],
    ) -> F2QVLALossOutput:
        """Compute total loss.
        
        Args:
            text_loss: Cross entropy loss from LM head (computed outside)
            pred_xyz: Predicted XYZ coordinates
            target_xyz: Target XYZ coordinates
            pred_rot: Predicted rotation matrices
            target_rot: Target rotation matrices
            
        Returns:
            F2QVLALossOutput containing total and individual losses
        """
        combined_loss = 0.0
        losses = {}
        
        # 1. Text Loss (Already computed by CausalLM head)
        if text_loss is not None:
             # Ensure scalar
            if text_loss.numel() > 1:
                text_loss = text_loss.mean()
            
            combined_loss += self.loss_weights.get("text", 1.0) * text_loss
            losses["text_loss"] = text_loss
            
        # 2. XYZ Loss
        # Calculate XYZ Loss
        xyz_loss = None
        if pred_xyz is not None and target_xyz is not None:
            # Cast target to match prediction dtype (e.g., bfloat16)
            target_xyz = target_xyz.to(dtype=pred_xyz.dtype)
            xyz_loss = self.xyz_loss_fn(pred_xyz, target_xyz)
            combined_loss += self.loss_weights.get("xyz", 1.0) * xyz_loss
            losses["xyz_loss"] = xyz_loss
            
        # 3. Rotation Loss
        # Calculate Rotation Loss
        rot_loss = None
        if pred_rot is not None and target_rot is not None:
            # Cast target to match prediction dtype
            target_rot = target_rot.to(dtype=pred_rot.dtype)
            rot_loss = compute_geodesic_loss(pred_rot, target_rot)
            combined_loss += self.loss_weights.get("rot", 1.0) * rot_loss
            losses["rot_loss"] = rot_loss
            
        return F2QVLALossOutput(
            total_loss=combined_loss,
            text_loss=losses.get("text_loss"),
            xyz_loss=losses.get("xyz_loss"),
            rot_loss=losses.get("rot_loss")
        )
