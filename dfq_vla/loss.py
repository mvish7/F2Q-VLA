"""Loss functions for DFQ VLA model.

This module provides the loss calculation logic for the DFQ VLA model,
including Geodesic Loss for rotation matrices and a composite loss class.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

@dataclass
class DFQVLALossOutput:
    """Output for DFQVLALoss."""
    total_loss: torch.Tensor
    text_loss: Optional[torch.Tensor] = None
    xyz_loss: Optional[torch.Tensor] = None
    rot_loss: Optional[torch.Tensor] = None


class DFQVLALoss(nn.Module):
    """Loss calculator for DFQ VLA."""
    
    def __init__(self, loss_weights: Dict[str, float] = None):
        super().__init__()
        if loss_weights is None:
            loss_weights = {"text": 1.0, "xyz": 1.0, "rot": 1.0}
        self.loss_weights = loss_weights
        
        # Loss functions
        self.xyz_loss_fn = nn.L1Loss(reduction='none')
        self.rot_loss_fn = nn.L1Loss(reduction='none')
        
    def forward(
        self,
        text_loss: Optional[torch.Tensor],
        pred_xyz: Optional[torch.Tensor],
        target_xyz: Optional[torch.Tensor],
        pred_rot: Optional[torch.Tensor],
        target_rot: Optional[torch.Tensor],
    ) -> DFQVLALossOutput:
        """Compute total loss.
        
        Args:
            text_loss: Cross entropy loss from LM head (computed outside)
            pred_xyz: Predicted XYZ coordinates
            target_xyz: Target XYZ coordinates
            pred_rot: Predicted rotation 2D representation
            target_rot: Target rotation 2D representation
            
        Returns:
            DFQVLALossOutput containing total and individual losses
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
            target_xyz = target_xyz / 100.0
            xyz_loss = self.xyz_loss_fn(pred_xyz, target_xyz).sum(dim=(1,2)).mean()
            combined_loss += self.loss_weights.get("xyz", 1.0) * xyz_loss
            losses["xyz_loss"] = xyz_loss
            
        # 3. Rotation Loss
        # Calculate Rotation Loss
        rot_loss = None
        if pred_rot is not None and target_rot is not None:
            # Cast target to match prediction dtype
            target_rot = target_rot.to(dtype=pred_rot.dtype)
            rot_loss = self.rot_loss_fn(pred_rot, target_rot).sum(dim=(1,2)).mean()
            combined_loss += self.loss_weights.get("rot", 1.0) * rot_loss
            losses["rot_loss"] = rot_loss
            
        return DFQVLALossOutput(
            total_loss=combined_loss,
            text_loss=losses.get("text_loss"),
            xyz_loss=losses.get("xyz_loss"),
            rot_loss=losses.get("rot_loss")
        )
