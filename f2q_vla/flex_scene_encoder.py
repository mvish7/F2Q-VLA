"""Flex Scene Encoder for multi-camera, multi-timestamp visual encoding.

This module implements the Flex Scene Encoder from arXiv:2512.10947v2.
It compresses image tokens from multiple cameras and timestamps into a
compact set of learnable scene tokens using self-attention.
"""

import math
import torch
import torch.nn as nn


class FlexSceneEncoder(nn.Module):
    """Compress multi-view, multi-timestamp images into K scene tokens.
    
    Architecture:
    1. Add camera embeddings (learnable) to image tokens
    2. Add timestep embeddings (sinusoidal) to image tokens
    3. Prepend K learnable scene tokens as queries
    4. Run L layers of self-attention over combined sequence
    5. Return only the scene tokens (discard image tokens)
    
    Attributes:
        num_scene_tokens: Number of output scene tokens (K).
        vision_hidden_size: Hidden dimension matching vision encoder output.
    """
    
    def __init__(
        self,
        vision_hidden_size: int = 3072,
        num_scene_tokens: int = 800,
        num_cameras: int = 4,
        num_timestamps: int = 4,
        num_layers: int = 6,
        num_heads: int = 12,
        dim_feedforward: int = 4096,
        dropout: float = 0.1,
        gradient_checkpointing: bool = True,  # Default to True for memory efficiency
    ):
        """Initialize FlexSceneEncoder.
        
        Args:
            vision_hidden_size: Hidden dimension (should match vision encoder output).
            num_scene_tokens: Number of scene tokens to output (K).
            num_cameras: Number of camera views.
            num_timestamps: Number of timestamps.
            num_layers: Number of transformer encoder layers.
            num_heads: Number of attention heads.
            dim_feedforward: Feedforward network dimension.
            dropout: Dropout probability.
            gradient_checkpointing: If True, use gradient checkpointing to save memory.
        """
        super().__init__()
        
        self.num_scene_tokens = num_scene_tokens
        self.vision_hidden_size = vision_hidden_size
        self.num_cameras = num_cameras
        self.num_timestamps = num_timestamps
        self.gradient_checkpointing = gradient_checkpointing
        
        # Learnable scene tokens (queries)
        self.scene_tokens = nn.Parameter(
            torch.randn(num_scene_tokens, vision_hidden_size) * 0.02
        )
        
        # Camera embeddings indexed by camera type (learnable)
        self.camera_embeddings = nn.Embedding(num_cameras, vision_hidden_size)
        
        # Timestep embeddings (sinusoidal, fixed)
        timestep_embeddings = self._create_sinusoidal_embeddings(
            num_timestamps, vision_hidden_size
        )
        self.register_buffer("timestep_embeddings", timestep_embeddings)
        
        # Transformer encoder (self-attention)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=vision_hidden_size,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm for better training stability
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers,
            enable_nested_tensor=False,  # Disable for gradient checkpointing compatibility
        )
        
        # Layer norm for scene tokens initialization
        self.scene_token_norm = nn.LayerNorm(vision_hidden_size)
    
    def _create_sinusoidal_embeddings(
        self, num_positions: int, dim: int
    ) -> torch.Tensor:
        """Create sinusoidal positional embeddings (DiT-style).
        
        Args:
            num_positions: Number of positions (timestamps).
            dim: Embedding dimension.
            
        Returns:
            Tensor of shape [num_positions, dim].
        """
        half_dim = dim // 2
        # Frequency bands
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(0, half_dim, dtype=torch.float32) / half_dim
        )
        
        # Position indices
        positions = torch.arange(num_positions, dtype=torch.float32)
        
        # Outer product: [num_positions, half_dim]
        angles = positions.unsqueeze(1) * freqs.unsqueeze(0)
        
        # Concatenate sin and cos
        embeddings = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        
        # Handle odd dimensions
        if dim % 2 == 1:
            embeddings = torch.cat([embeddings, torch.zeros(num_positions, 1)], dim=-1)
        
        return embeddings
    
    def forward(
        self,
        image_tokens: torch.Tensor,
        camera_ids: torch.Tensor,
        timestamp_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass to encode scene.
        
        Args:
            image_tokens: Image tokens from vision encoder.
                Shape: [B, num_images, num_patches, D]
            camera_ids: Camera type indices for each image.
                Shape: [B, num_images] with values in [0, num_cameras-1]
            timestamp_ids: Timestamp indices for each image.
                Shape: [B, num_images] with values in [0, num_timestamps-1]
                
        Returns:
            Scene tokens of shape [B, K, D].
        """
        B, num_images, N, D = image_tokens.shape
        
        # Validate dimensions
        assert D == self.vision_hidden_size, (
            f"Dimension mismatch: got {D}, expected {self.vision_hidden_size}"
        )
        
        # 1. Add camera embeddings (broadcast over patches)
        # camera_ids: [B, num_images] -> cam_embed: [B, num_images, D]
        cam_embed = self.camera_embeddings(camera_ids)
        # Expand to patches: [B, num_images, N, D]
        cam_embed = cam_embed.unsqueeze(2).expand(-1, -1, N, -1)
        
        # 2. Add timestep embeddings (broadcast over patches)
        # timestamp_ids: [B, num_images] -> time_embed: [B, num_images, D]
        time_embed = self.timestep_embeddings[timestamp_ids]
        # Expand to patches: [B, num_images, N, D]
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, N, -1)
        
        # 3. Add positional info to image tokens
        image_tokens = image_tokens + cam_embed + time_embed
        
        # 4. Flatten images: [B, num_images * N, D]
        # Use reshape instead of view to handle non-contiguous tensors from expand()
        image_tokens = image_tokens.reshape(B, num_images * N, D)
        
        # 5. Prepare scene tokens: [B, K, D]
        scene_tokens = self.scene_tokens.unsqueeze(0).expand(B, -1, -1)
        scene_tokens = self.scene_token_norm(scene_tokens)
        
        # 6. Prepend scene tokens: [B, K + num_images*N, D]
        combined = torch.cat([scene_tokens, image_tokens], dim=1)
        
        # 7. Self-attention over all tokens (with optional gradient checkpointing)
        if self.gradient_checkpointing and self.training:
            # Process each layer with gradient checkpointing to save memory
            # This recomputes activations during backward instead of storing them
            from torch.utils.checkpoint import checkpoint
            
            output = combined
            for layer in self.encoder.layers:
                # Use checkpoint for each transformer layer
                output = checkpoint(layer, output, use_reentrant=False)
            # Apply final norm if present
            if self.encoder.norm is not None:
                output = self.encoder.norm(output)
        else:
            output = self.encoder(combined)
        
        # 8. Return only scene tokens (first K)
        return output[:, :self.num_scene_tokens, :]


def create_flex_scene_encoder(config) -> FlexSceneEncoder:
    """Factory function to create FlexSceneEncoder from config.
    
    Args:
        config: F2QVLAConfig with flex encoder parameters.
        
    Returns:
        Initialized FlexSceneEncoder.
    """
    return FlexSceneEncoder(
        vision_hidden_size=getattr(config, "vision_hidden_size", 3072),
        num_scene_tokens=getattr(config, "num_scene_tokens", 800),
        num_cameras=getattr(config, "num_cameras", 4),
        num_timestamps=getattr(config, "num_timestamps", 4),
        num_layers=getattr(config, "flex_encoder_layers", 6),
        num_heads=getattr(config, "flex_encoder_heads", 12),
        dim_feedforward=getattr(config, "flex_encoder_dim_feedforward", 4096),
        dropout=getattr(config, "flex_encoder_dropout", 0.1),
        gradient_checkpointing=getattr(config, "flex_encoder_gradient_checkpointing", True),
    )
