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
    Supports L2 penalty ((mean(a) - rho)^2) and L1/smooth penalty with optional boundary sharpness.
    """
    def __init__(self, rho: float = 0.3, mode: str = "l2", sharpness_coeff: float = 0.0):
        super().__init__()
        self.rho = rho
        self.mode = mode
        self.sharpness_coeff = sharpness_coeff

    def forward(self, gate: torch.Tensor) -> torch.Tensor:
        # gate: [B, 1, H, W]
        mean_gate_per_sample = gate.mean(dim=(1, 2, 3))
        if self.mode == "l1":
            area_loss = torch.abs(mean_gate_per_sample - self.rho).mean()
        else:
            area_loss = ((mean_gate_per_sample - self.rho) ** 2).mean()

        if self.sharpness_coeff > 0.0:
            # Penalizes indecisive gate values near 0.5 (encourages 0 or 1)
            sharp_loss = (gate * (1.0 - gate)).mean()
            return area_loss + self.sharpness_coeff * sharp_loss
        return area_loss


class DynamicSampleReweighting(nn.Module):
    """
    Calculates per-example weights based on context sink error vs object path error:
    w_i = (CE_ctx,i / (CE_ctx,i + CE_obj,i))^gamma
    Tracks weights with Exponential Moving Average (EMA) across epochs to eliminate batch noise.
    Bias-conflicting examples receive elevated weights.
    """
    def __init__(
        self,
        w_min: float = 0.1,
        w_max: float = 10.0,
        gamma: float = 1.0,
        momentum: float = 0.7,
        num_samples: Optional[int] = None,
        eps: float = 1e-6
    ):
        super().__init__()
        self.w_min = w_min
        self.w_max = w_max
        self.gamma = gamma
        self.momentum = momentum
        self.eps = eps
        if num_samples is not None:
            self.register_buffer("ema_weights", torch.ones(num_samples))
        else:
            self.ema_weights = None

    def forward(
        self,
        ce_obj_per_sample: torch.Tensor,
        ce_ctx_per_sample: torch.Tensor,
        sample_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Relative error of the context sink
        ratio = ce_ctx_per_sample / (ce_ctx_per_sample + ce_obj_per_sample + self.eps)
        if self.gamma != 1.0:
            raw_weights = torch.pow(ratio, self.gamma)
        else:
            raw_weights = ratio

        # EMA smoothing across epochs if sample indices are tracked
        if sample_indices is not None and self.ema_weights is not None:
            device = raw_weights.device
            if self.ema_weights.device != device:
                self.ema_weights = self.ema_weights.to(device)
            old_w = self.ema_weights[sample_indices]
            updated_w = self.momentum * old_w + (1.0 - self.momentum) * raw_weights
            self.ema_weights[sample_indices] = updated_w.detach()
            target_weights = updated_w
        else:
            target_weights = raw_weights

        # Scale to mean 1.0 before clipping
        scaled_weights = target_weights / (target_weights.mean() + self.eps)
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
