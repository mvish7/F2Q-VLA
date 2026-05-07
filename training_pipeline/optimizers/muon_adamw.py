"""MuonPlusAdamW: hybrid optimizer routing 2D weights to Muon, rest to AdamW.

Muon applies Newton-Schulz orthogonalization to weight matrices, which
improves gradient conditioning for linear layers, attention projections,
and MLPs. Non-matrix parameters (biases, norms, embeddings) use AdamW.

Requires PyTorch >= 2.8 for torch.optim.Muon.
"""

import torch
from typing import Optional, Callable


class MuonPlusAdamW(torch.optim.Optimizer):
    """Hybrid optimizer that splits parameters by dimensionality.

    - 2D weight matrices → torch.optim.Muon (Newton-Schulz orthogonalization)
    - Everything else (1D biases, norms, embeddings) → torch.optim.AdamW

    Accepts standard parameter groups (list of dicts with 'params' and 'lr')
    to preserve per-module learning rate support.

    Args:
        param_groups: Iterable of param groups, each a dict with 'params' and 'lr'.
        weight_decay: Decoupled weight decay for both sub-optimizers.
        muon_momentum: Momentum for Muon optimizer.
    """

    def __init__(
        self,
        param_groups: list[dict],
        weight_decay: float = 0.01,
        muon_momentum: float = 0.95,
    ):
        # Flatten all params for the parent Optimizer registration
        all_params = []
        for group in param_groups:
            all_params.extend(group["params"])
        super().__init__(all_params, defaults={"lr": 1e-3})

        # Split each incoming LR-group into Muon (2D) and AdamW (non-2D) subgroups
        muon_groups = []
        adamw_groups = []

        total_muon = 0
        total_adamw = 0

        for group in param_groups:
            lr = group.get("lr", 1e-3)
            muon_params = [p for p in group["params"] if p.ndim == 2]
            adamw_params = [p for p in group["params"] if p.ndim != 2]

            if muon_params:
                muon_groups.append({"params": muon_params, "lr": lr})
                total_muon += sum(p.numel() for p in muon_params)
            if adamw_params:
                adamw_groups.append({"params": adamw_params, "lr": lr})
                total_adamw += sum(p.numel() for p in adamw_params)

        # Create inner optimizers targeting the SAME param tensors
        self._muon = torch.optim.Muon(
            muon_groups, lr=1e-3, momentum=muon_momentum, weight_decay=weight_decay
        ) if muon_groups else None

        self._adamw = torch.optim.AdamW(
            adamw_groups, lr=1e-3, weight_decay=weight_decay
        ) if adamw_groups else None

        print(f"=== MuonPlusAdamW Optimizer ===")
        print(f"  Muon (2D weights): {total_muon:,} params across {len(muon_groups)} LR-groups")
        print(f"  AdamW (non-2D):    {total_adamw:,} params across {len(adamw_groups)} LR-groups")
        print(f"  Weight decay: {weight_decay}")
        print(f"  Muon momentum: {muon_momentum}")
        print(f"===============================")

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = closure() if closure else None
        if self._adamw is not None:
            self._adamw.step()
        if self._muon is not None:
            self._muon.step()
        return loss

    def zero_grad(self, set_to_none: bool = True):
        if self._adamw is not None:
            self._adamw.zero_grad(set_to_none)
        if self._muon is not None:
            self._muon.zero_grad(set_to_none)
