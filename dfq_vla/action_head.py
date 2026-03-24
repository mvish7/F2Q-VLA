"""Action Chunking Head for DFQ VLA.

This module provides the ActionChunkingHead for future trajectory prediction.
Uses a standard PyTorch TransformerDecoder architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionChunkingHead(nn.Module):
    """Action Chunking Transformer Head for trajectory prediction.
    
    Predicts future trajectory waypoints using a Transformer Decoder.
    The decoder queries cross-attend to the VLM context.
    
    Attributes:
        num_queries: Number of future waypoints to predict.
        hidden_size: Hidden dimension (matching LLM).
    """
    
    def __init__(
        self,
        hidden_size: int = 1024,
        num_queries: int = 64,
        num_layers: int = 4,
        nhead: int = 16,
        dim_feedforward: int = 4096,
        dropout: float = 0.1,
    ):
        """Initialize ActionChunkingHead.
        
        Args:
            hidden_size: Hidden dimension (should match LLM hidden size).
            num_queries: Number of future waypoints to predict.
            num_layers: Number of transformer decoder layers.
            nhead: Number of attention heads.
            dim_feedforward: Feedforward network dimension.
            dropout: Dropout probability.
        """
        super().__init__()
        
        self.num_queries = num_queries
        self.hidden_size = hidden_size
        
        # Learnable queries for future waypoints
        self.query_embed = nn.Embedding(num_queries, hidden_size)
        
        # Standard PyTorch Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm for better training stability
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output heads
        self.xyz_head = nn.Linear(hidden_size, 3)
        self.rot_head = nn.Linear(hidden_size, 2)  # 2D continuous yaw
        
    def forward(
        self,
        vlm_context: torch.Tensor,
        normalize_rot: bool = True,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass to predict future trajectory.
        
        Args:
            vlm_context: VLM hidden states at context token(s). 
                Shape: [B, S, hidden_size] where S is context sequence length.
            normalize_rot: If True, normalize the 2D rotation representation to exist on unit circle.
            memory_key_padding_mask: Optional mask for padded positions in vlm_context.
                Shape: [B, S]. True indicates padded (ignored) positions.
            
        Returns:
            Dictionary containing:
                - "xyz": Predicted XYZ positions [B, num_queries, 3]
                - "rot2d": 2D rotation representation [B, num_queries, 2]
        """
        batch_size = vlm_context.shape[0]
        device = vlm_context.device
        
        # Expand queries for batch: [num_queries, hidden_size] -> [B, num_queries, hidden_size]
        queries = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Transformer decoder: queries attend to vlm_context
        # tgt=queries (what we want to decode), memory=vlm_context (what we attend to)
        decoder_output = self.decoder(
            tgt=queries,
            memory=vlm_context,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        
        # Predict outputs
        xyz = self.xyz_head(decoder_output)  # [B, num_queries, 3]
        rot2d = self.rot_head(decoder_output)  # [B, num_queries, 2]
        
        if normalize_rot:
            rot2d = F.normalize(rot2d, dim=-1)
            
        result = {
            "xyz": xyz,
            "rot2d": rot2d,
        }
            
        return result


def create_action_head(config) -> ActionChunkingHead:
    """Factory function to create ActionChunkingHead from config.
    
    Args:
        config: DFQVLAConfig with action head parameters.
        
    Returns:
        Initialized ActionChunkingHead.
    """
    return ActionChunkingHead(
        hidden_size=getattr(config, "hidden_size", 1024),
        num_queries=getattr(config, "num_action_queries", 64),
        num_layers=getattr(config, "num_action_layers", 4),
        nhead=getattr(config, "action_nhead", 16),
        dim_feedforward=getattr(config, "action_dim_feedforward", 4096),
        dropout=getattr(config, "action_dropout", 0.1),
    )
