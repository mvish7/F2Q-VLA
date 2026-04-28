"""Trajectory evaluation metrics: ADE and minADE."""

import torch
from torch import Tensor


def compute_ade(pred_xyz: Tensor, gt_xyz: Tensor) -> Tensor:
    """Average Displacement Error between a single predicted and ground-truth trajectory.

    Args:
        pred_xyz: Predicted positions, shape [T, 3].
        gt_xyz:   Ground-truth positions, shape [T, 3].

    Returns:
        Scalar ADE (mean L2 distance across T timesteps).
    """
    return torch.norm(pred_xyz - gt_xyz, dim=-1).mean()


def compute_min_ade(pred_samples_xyz: Tensor, gt_xyz: Tensor) -> Tensor:
    """Minimum ADE across K trajectory samples.

    Args:
        pred_samples_xyz: K predicted trajectories, shape [K, T, 3].
        gt_xyz:           Ground-truth positions, shape [T, 3].

    Returns:
        Scalar minADE — the ADE of the best-matching sample.
    """
    # [K, T] per-timestep distances, then mean over T → [K]
    ades = torch.norm(pred_samples_xyz - gt_xyz.unsqueeze(0), dim=-1).mean(dim=-1)
    return ades.min()


def aggregate_metrics(
    all_ade: list[float],
    all_min_ade3: list[float],
    all_min_ade6: list[float],
) -> dict[str, float]:
    """Aggregate per-sample metrics into dataset-level statistics.

    Args:
        all_ade:      Per-sample ADE values.
        all_min_ade3: Per-sample minADE3 values.
        all_min_ade6: Per-sample minADE6 values.

    Returns:
        Dictionary with mean values for each metric.
    """
    def _mean(vals):
        return sum(vals) / len(vals) if vals else float("nan")

    return {
        "ADE": _mean(all_ade),
        "minADE3": _mean(all_min_ade3),
        "minADE6": _mean(all_min_ade6),
        "num_samples": len(all_ade),
    }
