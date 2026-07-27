"""
=============================================================================
Tatvaani — Weighted Focal Loss
=============================================================================
File: training/focal_loss.py

Focal Loss (Lin et al., 2017) with per-class alpha weighting.
Designed to handle the severe class imbalance in deepfake detection
(~90% spoof, ~10% bonafide in current manifest).

gamma=2 down-weights easy samples (correctly classified with high confidence)
so the model focuses learning budget on hard/ambiguous examples.

alpha per class accounts for frequency imbalance on top of focal weighting.
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss with optional per-class alpha weighting.

    Args:
        gamma (float): Focusing parameter. 0 = standard CE. 2 = recommended.
        alpha (Tensor, optional): Per-class weight tensor of shape [num_classes].
                                  If None, all classes weighted equally.
        reduction (str): 'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma     = gamma
        self.reduction = reduction

        # Register alpha as a buffer so it moves to the right device
        # automatically with .to(device) and is included in state_dict
        if alpha is not None:
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  [B, C] — raw model output (NOT softmax-ed)
            targets: [B]    — integer class labels [0, C-1]
        Returns:
            Scalar loss (if reduction='mean')
        """
        # Standard cross-entropy gives log-probabilities
        # F.cross_entropy = -log(p_t)
        log_probs = F.log_softmax(logits, dim=-1)            # [B, C]
        probs     = torch.exp(log_probs)                     # [B, C] — actual probs

        # Gather log-prob and prob for the TRUE class at each sample
        log_pt = log_probs.gather(dim=1, index=targets.view(-1, 1)).squeeze(1)  # [B]
        pt     = probs.gather(dim=1, index=targets.view(-1, 1)).squeeze(1)      # [B]

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1.0 - pt) ** self.gamma               # [B]

        # Base focal loss per sample
        loss = -focal_weight * log_pt                          # [B]

        # Apply per-class alpha weights
        if self.alpha is not None:
            at   = self.alpha.gather(0, targets)               # [B]
            loss = at * loss

        # Reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss   # [B]


# Quick self-test
if __name__ == "__main__":
    print("Testing FocalLoss...")
    B, C = 8, 3
    logits  = torch.randn(B, C)
    targets = torch.randint(0, C, (B,))
    alpha   = torch.tensor([3.4, 1.0, 0.37])

    loss_fn = FocalLoss(gamma=2.0, alpha=alpha)
    loss    = loss_fn(logits, targets)
    print(f"  logits shape:  {logits.shape}")
    print(f"  targets:       {targets.tolist()}")
    print(f"  loss:          {loss.item():.4f}")
    assert not torch.isnan(loss), "NaN loss in FocalLoss self-test!"
    print("  ✅ FocalLoss OK")