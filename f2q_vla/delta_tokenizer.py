"""Delta trajectory tokenizer for F2Q VLA.

This module provides trajectory tokenization functionality, adapted from alpamayo_r1.
The tokenizer encodes trajectory waypoints as discrete tokens using delta encoding.
"""

import einops
import numpy as np
import torch


class DeltaTrajectoryTokenizer:
    """Delta trajectory tokenizer.
    
    Encodes trajectory waypoints as discrete tokens by:
    1. Computing delta positions between consecutive waypoints
    2. Quantizing the deltas into discrete bins
    """

    def __init__(
        self,
        ego_xyz_min: tuple[float, float, float] = (-4, -4, -10),
        ego_xyz_max: tuple[float, float, float] = (4, 4, 10),
        ego_yaw_min: float = -np.pi,
        ego_yaw_max: float = np.pi,
        num_bins: int = 1000,
        predict_yaw: bool = False,
    ):
        """Initialize the tokenizer.
        
        Args:
            ego_xyz_min: Minimum delta values for xyz clipping.
            ego_xyz_max: Maximum delta values for xyz clipping.
            ego_yaw_min: Minimum yaw angle for clipping.
            ego_yaw_max: Maximum yaw angle for clipping.
            num_bins: Number of discrete bins for quantization.
            predict_yaw: Whether to encode yaw angles (adds 1 token per waypoint).
        """
        self.ego_xyz_min = ego_xyz_min
        self.ego_xyz_max = ego_xyz_max
        self.num_bins = num_bins
        self._predict_yaw = predict_yaw
        self.ego_yaw_min = ego_yaw_min
        self.ego_yaw_max = ego_yaw_max

    @property
    def vocab_size(self) -> int:
        """Tokens are integers from the set {0, 1, ..., vocab_size - 1}"""
        return self.num_bins

    def encode(
        self,
        hist_xyz: torch.Tensor,
        hist_rot: torch.Tensor,
        fut_xyz: torch.Tensor,
        fut_rot: torch.Tensor,
        hist_tstamp: torch.Tensor | None = None,
        fut_tstamp: torch.Tensor | None = None,
    ) -> torch.LongTensor:
        """Encode trajectories as discrete tokens.
        
        The model conditions on historical waypoints to tokenize future/target waypoints.
        Trajectories can be provided in any coordinate frame.
        
        Args:
            hist_xyz: Historical locations XYZ. Shape: (B, Th, 3).
            hist_rot: Historical rotations. Shape: (B, Th, 3, 3).
            fut_xyz: Future/target locations XYZ. Shape: (B, Tf, 3).
            fut_rot: Future/target rotations. Shape: (B, Tf, 3, 3).
            hist_tstamp: Historical timestamps. Shape: (B, Th). Unused.
            fut_tstamp: Future timestamps. Shape: (B, Tf). Unused.

        Returns:
            torch.LongTensor: Token indices. Shape: (B, num_tokens_per_trajectory).
        """
        del hist_xyz, hist_rot, hist_tstamp, fut_tstamp
        
        # Compute delta positions
        xyz = torch.nn.functional.pad(fut_xyz, [0, 0, 1, 0, 0, 0])
        xyz = xyz[:, 1:] - xyz[:, :-1]
        
        # Normalize to [0, 1] range
        ego_xyz_max = torch.tensor(self.ego_xyz_max, dtype=xyz.dtype, device=xyz.device)
        ego_xyz_min = torch.tensor(self.ego_xyz_min, dtype=xyz.dtype, device=xyz.device)
        xyz = (xyz - ego_xyz_min) / (ego_xyz_max - ego_xyz_min)
        
        # Quantize to discrete bins
        xyz = (xyz * (self.num_bins - 1)).round().long()
        xyz = xyz.clamp(0, self.num_bins - 1)
        
        if not self._predict_yaw:
            return einops.rearrange(xyz, "b n m -> b (n m)")
        
        # Extract yaw angles from rotation matrices
        yaw = torch.atan2(fut_rot[..., 0, 1], fut_rot[..., 0, 0])

        # Calculate delta yaw
        yaw_padded = torch.nn.functional.pad(yaw, [1, 0, 0, 0])
        delta_yaw = yaw_padded[:, 1:] - yaw_padded[:, :-1]

        # Normalize delta yaw to [-pi, pi]
        delta_yaw = torch.atan2(torch.sin(delta_yaw), torch.cos(delta_yaw))

        # Scale and quantize delta yaw
        delta_yaw = (delta_yaw - self.ego_yaw_min) / (self.ego_yaw_max - self.ego_yaw_min)
        delta_yaw = (delta_yaw * (self.num_bins - 1)).round().long()
        delta_yaw = delta_yaw.clamp(0, self.num_bins - 1)

        xyzw = torch.cat([xyz, delta_yaw.unsqueeze(-1)], dim=-1)  # Shape: (B, Tf, 4)
        return einops.rearrange(xyzw, "b n m -> b (n m)")

    def decode(
        self,
        hist_xyz: torch.Tensor,
        hist_rot: torch.Tensor,
        tokens: torch.LongTensor,
        hist_tstamp: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Decode tokens into future trajectories.
        
        The future trajectory is returned in the same coordinate frame as the
        historical trajectory.
        
        Args:
            hist_xyz: Historical locations XYZ. Shape: (B, Th, 3).
            hist_rot: Historical rotations. Shape: (B, Th, 3, 3).
            tokens: Token indices. Shape: (B, num_tokens_per_trajectory).
            hist_tstamp: Historical timestamps. Shape: (B, Th). Unused.

        Returns:
            fut_xyz: Future locations XYZ. Shape: (B, Tf, 3).
            fut_rot: Future rotations. Shape: (B, Tf, 3, 3).
            fut_tstamp: Future timestamps. Shape: (B, Tf). Always None.
        """
        del hist_tstamp
        m = 4 if self._predict_yaw else 3
        xyzw = einops.rearrange(tokens, "b (n m) -> b n m", m=m).to(hist_xyz.dtype)
        xyz = xyzw[..., :3]
        xyz = xyz / (self.num_bins - 1)
        ego_xyz_max = torch.tensor(self.ego_xyz_max, dtype=xyz.dtype, device=xyz.device)
        ego_xyz_min = torch.tensor(self.ego_xyz_min, dtype=xyz.dtype, device=xyz.device)
        xyz = xyz * (ego_xyz_max - ego_xyz_min) + ego_xyz_min
        fut_xyz = torch.cumsum(xyz, dim=1)
        
        if not self._predict_yaw:
            xyz_cpu = fut_xyz.cpu().numpy().astype(float)
            fut_rot = _get_yaw_rotation_matrices(xyz_cpu)
            fut_rot = torch.tensor(fut_rot, device=fut_xyz.device, dtype=fut_xyz.dtype)
            return fut_xyz, fut_rot, None
        
        yaw_tokens = xyzw[..., 3]
        yaw = yaw_tokens.float() / (self.num_bins - 1)
        yaw = yaw * (self.ego_yaw_max - self.ego_yaw_min) + self.ego_yaw_min
        yaw = torch.cumsum(yaw, dim=1)

        # Convert yaw angles to rotation matrices
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        zeros = torch.zeros_like(cos_yaw)
        ones = torch.ones_like(cos_yaw)

        fut_rot = torch.stack(
            [
                torch.stack([cos_yaw, -sin_yaw, zeros], dim=-1),
                torch.stack([sin_yaw, cos_yaw, zeros], dim=-1),
                torch.stack([zeros, zeros, ones], dim=-1),
            ],
            dim=-2,
        ).to(device=hist_rot.device, dtype=hist_rot.dtype)
        return fut_xyz, fut_rot, None


def _get_yaw_rotation_matrices(trajectory: np.ndarray, window_size: int = 10, poly_order: int = 3) -> np.ndarray:
    """Calculate yaw rotation matrices using polynomial fitting.
    
    Args:
        trajectory: Array of shape (B, N, 3) for batch of x,y,z coordinates.
        window_size: Size of window for polynomial fitting.
        poly_order: Order of polynomial to fit.

    Returns:
        rotation_matrices: Rotation matrices at each point, shape (B, N, 3, 3).
    """
    B, N = trajectory.shape[:2]
    rotation_matrices = []

    for b in range(B):
        traj_batch = trajectory[b]  # (N, 3)
        batch_matrices = []

        for i in range(N):
            # Get window indices with padding for edges
            start_idx = max(0, i - window_size // 2)
            end_idx = min(N, start_idx + window_size)

            # Adjust window if at edges
            if end_idx - start_idx < window_size:
                start_idx = max(0, end_idx - window_size)

            # Get points in window
            window_points = traj_batch[start_idx:end_idx]

            # Use time parameter t
            t = np.arange(len(window_points))

            # Fit polynomials to both x(t) and y(t)
            x_coeffs = np.polyfit(t, window_points[:, 0], poly_order)
            y_coeffs = np.polyfit(t, window_points[:, 1], poly_order)

            # Calculate derivatives at center point
            center_t = min(i - start_idx, window_size - 1)
            x_deriv = np.polyder(x_coeffs)
            y_deriv = np.polyder(y_coeffs)

            dx = np.polyval(x_deriv, center_t)
            dy = np.polyval(y_deriv, center_t)

            # Calculate yaw angle from dx, dy
            yaw = np.arctan2(dy, dx)

            # Create 3x3 rotation matrix for yaw
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)
            rotation_matrix = np.array([[cos_yaw, -sin_yaw, 0], [sin_yaw, cos_yaw, 0], [0, 0, 1]])

            batch_matrices.append(rotation_matrix)

        rotation_matrices.append(batch_matrices)

    return np.array(rotation_matrices)
