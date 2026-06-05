"""Action Head for DFQ VLA — Transformer Decoder for Trajectory Prediction.

Takes two inputs:
  A) 16 action token hidden states from LFM2.5-350M  (batch, 16, 1024)
  B) 512 Flex scene token embeddings                  (batch, 512, 768)

Produces one output:
  64 waypoints as (x, y, sin_yaw, cos_yaw) in ego-centric frame  (batch, 64, 4)
  x/y are normalized by NORM_SCALE (default 50.0) — unnormalize during inference.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. ActionHeadConfig
# ---------------------------------------------------------------------------

@dataclass
class ActionHeadConfig:
    """All hyperparameters for the action head. Constants live here, nowhere else."""

    D_LLM: int = 1024          # LFM2.5-350M hidden_size
    D_FLEX: int = 768           # TIPSv2 / Flex output dimension
    D_ACTION: int = 384         # Internal dimension of the action head
    N_HEADS: int = 6            # Attention heads (384 / 6 = 64 head_dim)
    D_FFN: int = 1536           # Feed-forward expansion (4 × 384)
    DROPOUT: float = 0.1       # Dropout rate everywhere
    N_LAYERS: int = 5           # Number of decoder layers
    N_TRAJ_QUERIES: int = 64    # One per output timestep
    N_ACTION_TOKENS: int = 16   # Action embeddings fed to LLM
    OUTPUT_DIM: int = 4         # (x, y, sin_yaw, cos_yaw)
    OUTPUT_HIDDEN: int = 128    # Intermediate dim in output MLP
    TEMPORAL_WEIGHT_MAX: float = 1.0   # Loss weight for timestep 0
    TEMPORAL_WEIGHT_MIN: float = 0.7   # Loss weight for timestep 63
    NORM_SCALE: float = 50.0   # x/y normalization scale (meters → [-1, 1])
    UNIT_CIRCLE_LAMBDA: float = 0.1  # Weight for unit circle regularization

    @property
    def HEAD_DIM(self) -> int:
        assert self.D_ACTION % self.N_HEADS == 0, (
            f"D_ACTION ({self.D_ACTION}) must be divisible by N_HEADS ({self.N_HEADS})"
        )
        return self.D_ACTION // self.N_HEADS


# ---------------------------------------------------------------------------
# 2. build_sinusoidal_pe
# ---------------------------------------------------------------------------

def build_sinusoidal_pe(n_positions: int, d_model: int) -> torch.Tensor:
    """Build sinusoidal positional encoding (not learnable).

    Args:
        n_positions: Number of positions (e.g. 64 trajectory queries).
        d_model: Embedding dimension (e.g. 384).

    Returns:
        Tensor of shape (n_positions, d_model).
    """
    pe = torch.zeros(n_positions, d_model)
    position = torch.arange(0, n_positions, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
    )

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


# ---------------------------------------------------------------------------
# 3. MultiHeadAttention
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """Multi-head attention using F.scaled_dot_product_attention.

    Used for both self-attention and cross-attention (same class, different inputs).
    No causal mask — trajectory queries are a SET, not a sequence.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: (batch, n_q, d_model)
            key:   (batch, n_kv, d_model)
            value: (batch, n_kv, d_model)

        Returns:
            (batch, n_q, d_model)
        """
        B, n_q, _ = query.shape

        # Project and reshape to (batch, n_heads, seq_len, head_dim)
        q = self.W_q(query).view(B, n_q, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(key).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(value).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention (no causal mask — fully bidirectional)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            is_causal=False,
            dropout_p=self.dropout if self.training else 0.0,
        )

        # Reshape back to (batch, n_q, d_model)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, n_q, -1)
        return self.W_o(attn_out)


# ---------------------------------------------------------------------------
# 4. FFN
# ---------------------------------------------------------------------------

class FFN(nn.Module):
    """Two-layer MLP with GELU activation."""

    def __init__(self, d_model: int, d_ffn: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ffn)
        self.w2 = nn.Linear(d_ffn, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(self.dropout(F.gelu(self.w1(x)))))


# ---------------------------------------------------------------------------
# 5. ActionDecoderLayer
# ---------------------------------------------------------------------------

class ActionDecoderLayer(nn.Module):
    """One decoder layer with 4 sub-layers in this exact order:

    A) Self-attention over trajectory queries (inter-waypoint coherence)
    B) Cross-attention to Flex scene tokens   (spatial grounding)
    C) Cross-attention to LLM action tokens   (intent grounding)
    D) Feed-forward network                   (nonlinear integration)

    Each sub-layer uses PRE-LAYERNORM with RESIDUAL CONNECTION:
        output = x + sublayer(LayerNorm(x))
    """

    def __init__(self, d_action: int, n_heads: int, d_ffn: int, dropout: float = 0.1):
        super().__init__()

        # Sub-layer A: Self-attention
        self.self_attn = MultiHeadAttention(d_action, n_heads, dropout)
        self.norm_self = nn.LayerNorm(d_action)

        # Sub-layer B: Cross-attention to scene
        self.cross_attn_scene = MultiHeadAttention(d_action, n_heads, dropout)
        self.norm_scene = nn.LayerNorm(d_action)

        # Sub-layer C: Cross-attention to LLM action tokens
        self.cross_attn_llm = MultiHeadAttention(d_action, n_heads, dropout)
        self.norm_llm = nn.LayerNorm(d_action)

        # Sub-layer D: FFN
        self.ffn = FFN(d_action, d_ffn, dropout)
        self.norm_ffn = nn.LayerNorm(d_action)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        queries: torch.Tensor,
        scene_ctx: torch.Tensor,
        action_ctx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            queries:    (batch, 64, D_ACTION)  — trajectory queries
            scene_ctx:  (batch, N_scene, D_ACTION)  — already projected from 768
            action_ctx: (batch, 16, D_ACTION)  — already projected from 1024

        Returns:
            (batch, 64, D_ACTION)
        """
        # A: Self-attention
        q_norm = self.norm_self(queries)
        queries = queries + self.dropout(self.self_attn(q_norm, q_norm, q_norm))

        # B: Cross-attention to scene
        q_norm = self.norm_scene(queries)
        queries = queries + self.dropout(self.cross_attn_scene(q_norm, scene_ctx, scene_ctx))

        # C: Cross-attention to LLM action tokens
        q_norm = self.norm_llm(queries)
        queries = queries + self.dropout(self.cross_attn_llm(q_norm, action_ctx, action_ctx))

        # D: FFN
        q_norm = self.norm_ffn(queries)
        queries = queries + self.dropout(self.ffn(q_norm))

        return queries


# ---------------------------------------------------------------------------
# 6. OutputHead
# ---------------------------------------------------------------------------

class OutputHead(nn.Module):
    """Projects trajectory queries from D_ACTION to OUTPUT_DIM.

    Architecture: LayerNorm → Linear(384, 128) → GELU → Linear(128, 4)
    No dropout — output layer should not have stochastic noise.
    """

    def __init__(self, d_action: int, output_hidden: int, output_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_action)
        self.fc1 = nn.Linear(d_action, output_hidden)
        self.fc2 = nn.Linear(output_hidden, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(self.norm(x))))


# ---------------------------------------------------------------------------
# 7. LearnedActionTokens
# ---------------------------------------------------------------------------

class LearnedActionTokens(nn.Module):
    """16 learned embedding vectors injected into LLM input at <|action|> positions.

    NOT part of the action head decoder — lives in the LLM input pipeline.
    These are always the same fixed (but learnable) inputs. We only care about
    their OUTPUT hidden states after the LLM forward pass.
    """

    def __init__(self, config: ActionHeadConfig):
        super().__init__()
        self.embeddings = nn.Parameter(
            torch.empty(config.N_ACTION_TOKENS, config.D_LLM)
        )
        nn.init.xavier_uniform_(self.embeddings)

    def forward(self, batch_size: int) -> torch.Tensor:
        """Returns (batch, N_ACTION_TOKENS, D_LLM)."""
        return self.embeddings.unsqueeze(0).expand(batch_size, -1, -1)


# ---------------------------------------------------------------------------
# 8. ActionHead (main module)
# ---------------------------------------------------------------------------

class ActionHead(nn.Module):
    """Transformer Decoder action head for trajectory prediction.

    Composes all sub-modules: input projections, learned trajectory queries,
    sinusoidal PE, decoder layers, and output head.
    """

    def __init__(self, config: ActionHeadConfig | None = None):
        super().__init__()
        if config is None:
            config = ActionHeadConfig()
        self.config = config

        # Input projections (computed ONCE, reused across all decoder layers)
        self.proj_llm = nn.Sequential(
            nn.Linear(config.D_LLM, config.D_ACTION),
            nn.LayerNorm(config.D_ACTION),
        )
        self.proj_scene = nn.Sequential(
            nn.Linear(config.D_FLEX, config.D_ACTION),
            nn.LayerNorm(config.D_ACTION),
        )

        # Trajectory queries — one per output timestep
        self.traj_queries = nn.Embedding(config.N_TRAJ_QUERIES, config.D_ACTION)

        # Sinusoidal positional encoding (fixed, not learnable)
        self.register_buffer(
            "temporal_pe",
            build_sinusoidal_pe(config.N_TRAJ_QUERIES, config.D_ACTION),
        )

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            ActionDecoderLayer(config.D_ACTION, config.N_HEADS, config.D_FFN, config.DROPOUT)
            for _ in range(config.N_LAYERS)
        ])

        # Output head
        self.output_head = OutputHead(config.D_ACTION, config.OUTPUT_HIDDEN, config.OUTPUT_DIM)

        # Temporal loss weights (near-future weighted higher)
        self.register_buffer(
            "temporal_weights",
            torch.linspace(config.TEMPORAL_WEIGHT_MAX, config.TEMPORAL_WEIGHT_MIN, config.N_TRAJ_QUERIES),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize all weights following GPT-2 conventions."""
        cfg = self.config

        # Generic initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

        # Special: near-zero initial output predictions
        nn.init.xavier_uniform_(self.output_head.fc2.weight, gain=0.1)
        nn.init.zeros_(self.output_head.fc2.bias)

        # Special: GPT-2 residual scaling on all W_o projections
        gain = 1.0 / math.sqrt(2.0 * cfg.N_LAYERS)
        for layer in self.decoder_layers:
            for attn in [layer.self_attn, layer.cross_attn_scene, layer.cross_attn_llm]:
                nn.init.xavier_uniform_(attn.W_o.weight, gain=gain)

    def forward(
        self,
        llm_action_hidden: torch.Tensor,
        flex_scene_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            llm_action_hidden:     (batch, 16, 1024)  — LLM hidden states at action positions
            flex_scene_embeddings: (batch, N_scene, 768)  — raw Flex output (NOT projected)

        Returns:
            waypoints: (batch, 64, 4)  — (x, y, sin_yaw, cos_yaw), x/y in normalized space
        """
        batch = llm_action_hidden.shape[0]

        # Step 1: Project inputs (computed once, reused across all decoder layers)
        action_ctx = self.proj_llm(llm_action_hidden)       # (batch, 16, 384)
        scene_ctx = self.proj_scene(flex_scene_embeddings)   # (batch, N_scene, 384)

        # Step 2: Initialize queries with temporal PE
        queries = self.traj_queries.weight.unsqueeze(0).expand(batch, -1, -1)
        queries = queries + self.temporal_pe.unsqueeze(0)    # (batch, 64, 384)

        # Step 3: Run decoder layers
        for layer in self.decoder_layers:
            queries = layer(queries, scene_ctx, action_ctx)

        # Step 4: Predict waypoints
        waypoints = self.output_head(queries)                # (batch, 64, 4)

        return waypoints

    def compute_loss(
        self,
        predicted: torch.Tensor,
        target_xy: torch.Tensor,
        target_rot: torch.Tensor,
    ) -> torch.Tensor:
        """Compute temporally-weighted L1 loss + unit circle regularization.

        Args:
            predicted:  (batch, 64, 4)  — (x, y, sin, cos) in normalized space
            target_xy:  (batch, 64, 3)  — ego-centric (x, y, z) in meters
            target_rot: (batch, 64, 2)  — (cos_yaw, sin_yaw) continuous representation

        Returns:
            Scalar loss.
        """
        cfg = self.config

        # Normalize GT x/y by NORM_SCALE, drop z
        gt_xy = target_xy[..., :2] / cfg.NORM_SCALE       # (batch, 64, 2)
        # Reorder GT rot from (cos, sin) to (sin, cos) to match output head format
        gt_sin = target_rot[..., 1:2]                       # (batch, 64, 1)
        gt_cos = target_rot[..., 0:1]                       # (batch, 64, 1)
        gt = torch.cat([gt_xy, gt_sin, gt_cos], dim=-1)    # (batch, 64, 4)

        # Per-point L1
        per_point_l1 = torch.abs(predicted - gt)            # (batch, 64, 4)
        per_point_l1 = per_point_l1.mean(dim=-1)            # (batch, 64)

        # Apply temporal weights
        weighted = per_point_l1 * self.temporal_weights      # (batch, 64)
        l1_loss = weighted.mean()

        # Unit circle regularization: sin² + cos² should equal 1
        pred_sin = predicted[..., 2]
        pred_cos = predicted[..., 3]
        unit_deviation = (pred_sin ** 2 + pred_cos ** 2 - 1.0) ** 2
        unit_loss = unit_deviation.mean()

        return l1_loss + cfg.UNIT_CIRCLE_LAMBDA * unit_loss


# ---------------------------------------------------------------------------
# 9. sanity_check
# ---------------------------------------------------------------------------

def sanity_check():
    """Validate action head in isolation."""
    print("=" * 60)
    print("Action Head Sanity Check")
    print("=" * 60)

    config = ActionHeadConfig()
    action_head = ActionHead(config)
    learned_tokens = LearnedActionTokens(config)

    # Random inputs
    B = 4
    llm_hidden = torch.randn(B, config.N_ACTION_TOKENS, config.D_LLM)
    scene_emb = torch.randn(B, 512, config.D_FLEX)
    gt_xy = torch.randn(B, config.N_TRAJ_QUERIES, 3)
    gt_rot = torch.randn(B, config.N_TRAJ_QUERIES, 2)

    # Forward pass
    waypoints = action_head(llm_hidden, scene_emb)
    print(f"Output shape: {waypoints.shape}")
    assert waypoints.shape == (B, config.N_TRAJ_QUERIES, config.OUTPUT_DIM), (
        f"Expected ({B}, {config.N_TRAJ_QUERIES}, {config.OUTPUT_DIM}), got {waypoints.shape}"
    )

    # Loss
    loss = action_head.compute_loss(waypoints, gt_xy, gt_rot)
    print(f"Loss value: {loss.item():.4f}")
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    # Backward
    loss.backward()

    # Check gradients
    all_grads_ok = True
    for name, param in action_head.named_parameters():
        if param.requires_grad and param.grad is None:
            print(f"  MISSING gradient: {name}")
            all_grads_ok = False

    for name, param in learned_tokens.named_parameters():
        if param.requires_grad and param.grad is None:
            # LearnedActionTokens wasn't in the forward graph — expected
            pass

    # Param count breakdown
    def count_params(module):
        return sum(p.numel() for p in module.parameters())

    print(f"\n--- Parameter Breakdown ---")
    print(f"  proj_llm:        {count_params(action_head.proj_llm):>10,}")
    print(f"  proj_scene:      {count_params(action_head.proj_scene):>10,}")
    print(f"  traj_queries:    {count_params(action_head.traj_queries):>10,}")
    print(f"  decoder_layers:  {count_params(action_head.decoder_layers):>10,}")
    print(f"  output_head:     {count_params(action_head.output_head):>10,}")
    total_ah = count_params(action_head)
    total_lat = count_params(learned_tokens)
    print(f"  ---")
    print(f"  ActionHead total:         {total_ah:>10,}")
    print(f"  LearnedActionTokens:      {total_lat:>10,}")
    print(f"  Combined total:           {total_ah + total_lat:>10,}")

    # Final verdict
    print(f"\n{'=' * 60}")
    if all_grads_ok and torch.isfinite(loss):
        print("PASSED ✓")
    else:
        print("FAILED ✗")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    sanity_check()
