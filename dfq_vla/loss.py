"""Loss functions for DFQ VLA model.

This module provides the loss calculation logic for the DFQ VLA model.
With the removal of VQ-VAE and the action head, the loss is purely
text-based cross-entropy computed by the standard LM loss function.
This module is retained for extensibility.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class DFQVLALossOutput:
    """Output for DFQVLALoss."""
    total_loss: torch.Tensor
    text_loss: Optional[torch.Tensor] = None


class DFQVLALoss(nn.Module):
    """Loss calculator for DFQ VLA.
    
    Currently wraps text-only CE loss. Kept as a module for future extensibility
    (e.g. adding auxiliary losses).
    """
    
    def __init__(self, loss_weights: Dict[str, float] = None):
        super().__init__()
        if loss_weights is None:
            loss_weights = {"text": 1.0}
        self.loss_weights = loss_weights
        
    def forward(
        self,
        text_loss: Optional[torch.Tensor],
    ) -> DFQVLALossOutput:
        """Compute total loss.
        
        Args:
            text_loss: Cross entropy loss from LM head (computed outside)
            
        Returns:
            DFQVLALossOutput containing total and individual losses
        """
        combined_loss = torch.tensor(0.0)
        
        if text_loss is not None:
            if text_loss.numel() > 1:
                text_loss = text_loss.mean()
            combined_loss = self.loss_weights.get("text", 1.0) * text_loss

        return DFQVLALossOutput(
            total_loss=combined_loss,
            text_loss=text_loss,
        )
