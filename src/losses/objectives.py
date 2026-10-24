"""
Training objectives and loss functions for OCDP:
- Generalized Cross-Entropy (GCE) loss for context sink
- Area prior regularization for object gate
- Counterfactual consistency loss
- Dynamic context-error sample reweighting
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneralizedCrossEntropyLoss(nn.Module):
    """
    Generalized Cross Entropy (GCE) loss (Zhang & Sabuncu, NeurIPS 2018).
    L_q(p, y) = (1 - p_y^q) / q
    As q -> 0, L_q approaches Cross-Entropy.
    For q in (0, 1] (e.g. q=0.7 as in LfF), emphasizes confident, easy examples (shortcuts).
    """
    def __init__(self, q: float = 0.7, eps: float = 1e-7, reduction: str = "mean"):
        super().__init__()
        self.q = q
        self.eps = eps
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        p_y = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        p_y = torch.clamp(p_y, min=self.eps, max=1.0)
        
        if abs(self.q) < 1e-5:
            loss = -torch.log(p_y)
        else:
            loss = (1.0 - torch.pow(p_y, self.q)) / self.q

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "none":
            return loss
        return loss.sum()


class AreaPriorLoss(nn.Module):
    """
    Area prior loss: penalizes deviations of the average gate value from target ratio rho.
    L_area = (mean(a) - rho)^2
    Prevents gate saturation (a -> 0 or a -> 1).
    """
    def __init__(self, rho: float = 0.3):
        super().__init__()
        self.rho = rho

    def forward(self, gate: torch.Tensor) -> torch.Tensor:
        # gate: [B, 1, H, W]
        mean_gate_per_sample = gate.mean(dim=(1, 2, 3))
        loss = (mean_gate_per_sample - self.rho) ** 2
        return loss.mean()


class DynamicSampleReweighting(nn.Module):
    """
    Calculates per-example weights based on context sink error vs object path error:
    w_i = CE_ctx,i / (CE_ctx,i + CE_obj,i)
    Bias-conflicting examples (which the shortcut sink gets wrong) receive higher weights.
    """
    def __init__(
        self,
        w_min: float = 0.1,
        w_max: float = 5.0,
        eps: float = 1e-6
    ):
        super().__init__()
        self.w_min = w_min
        self.w_max = w_max
        self.eps = eps

    def forward(
        self,
        ce_obj_per_sample: torch.Tensor,
        ce_ctx_per_sample: torch.Tensor
    ) -> torch.Tensor:
        # Normalized per-sample relative error of the sink
        raw_weights = ce_ctx_per_sample / (ce_ctx_per_sample + ce_obj_per_sample + self.eps)
        # Scale to mean 1.0 before clipping
        scaled_weights = raw_weights / (raw_weights.mean() + self.eps)
        clipped_weights = torch.clamp(scaled_weights, min=self.w_min, max=self.w_max)
        # Re-normalize batch weights so sum(w) = B
        normalized_weights = clipped_weights / (clipped_weights.mean() + self.eps)
        return normalized_weights


class CounterfactualConsistencyLoss(nn.Module):
    """
    Consistency regularizer between clean prediction and masked prediction.
    KL(p_obj(x) || p_obj(x * M(a)))
    """
    def __init__(self):
        super().__init__()

    def forward(
        self,
        clean_logits: torch.Tensor,
        masked_logits: torch.Tensor
    ) -> torch.Tensor:
        p_clean = F.softmax(clean_logits, dim=-1)
        log_p_masked = F.log_softmax(masked_logits, dim=-1)
        return F.kl_div(log_p_masked, p_clean, reduction="batchmean")
