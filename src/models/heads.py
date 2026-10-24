"""
Pooling heads for ResNet-50 backbone:
- GAPHead: Standard Global Average Pooling baseline
- ABMILHead: Attention-based Multiple Instance Learning (Ilse et al.)
- TopKHead: Top-K spatial logit pooling (FocusMIL style)
- DARHead: Deep Attention Reweighting spatial attention (no context sink)
- OCDPHead: Object-Context Decoupled Pooling (Proposed)
"""

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class GAPHead(nn.Module):
    """Standard Global Average Pooling + Linear Classifier."""
    def __init__(self, in_features: int = 2048, num_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] or [B, C]
        if x.dim() == 4:
            x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        return self.fc(x)


class ABMILHead(nn.Module):
    """
    Attention-based Multiple Instance Learning (Ilse et al. 2018).
    Applies spatial attention over grid locations to pool features before classification.
    """
    def __init__(self, in_features: int = 2048, hidden_dim: int = 256, num_classes: int = 2):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_features, hidden_dim, kernel_size=1),
            nn.Tanh(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1)
        )
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        attn_logits = self.attention(x).view(B, 1, H * W)  # [B, 1, HW]
        attn_weights = F.softmax(attn_logits, dim=-1)  # [B, 1, HW]
        
        flat_x = x.view(B, C, H * W)  # [B, C, HW]
        pooled = torch.bmm(flat_x, attn_weights.transpose(1, 2)).squeeze(-1)  # [B, C]
        return self.fc(pooled)


class TopKHead(nn.Module):
    """
    Top-K spatial logit pooling (FocusMIL control).
    Computes per-location class logits, then averages the top-k highest logits per class.
    """
    def __init__(self, in_features: int = 2048, num_classes: int = 2, k: int = 5):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_features, num_classes, kernel_size=1)
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        loc_logits = self.conv1x1(x)  # [B, K, H, W]
        B, K, H, W = loc_logits.shape
        flat_logits = loc_logits.view(B, K, H * W)
        k_val = min(self.k, H * W)
        topk_logits, _ = torch.topk(flat_logits, k=k_val, dim=-1)  # [B, K, k_val]
        return topk_logits.mean(dim=-1)


class DARHead(nn.Module):
    """
    DAR-style spatial attention head without context sink.
    Computes per-location logits and attention gate, pooling only via attention.
    """
    def __init__(self, in_features: int = 2048, hidden_dim: int = 256, num_classes: int = 2):
        super().__init__()
        self.evidence_conv = nn.Conv2d(in_features, num_classes, kernel_size=1)
        self.gate_mlp = nn.Sequential(
            nn.Conv2d(in_features, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.evidence_conv(x)  # [B, K, H, W]
        a = self.gate_mlp(x)       # [B, 1, H, W]
        B, K, H, W = z.shape
        z_flat = z.view(B, K, H * W)
        a_flat = a.view(B, 1, H * W)
        eps = 1e-6
        pooled = (a_flat * z_flat).sum(dim=-1) / (a_flat.sum(dim=-1) + eps)
        return pooled


class OCDPHead(nn.Module):
    """
    Object-Context Decoupled Pooling (OCDP) Head.
    - Evidence Head: 1x1 conv mapping C -> K (logits z_u)
    - Gate Head: 1x1 conv MLP (C -> 256 -> 1) -> Sigmoid (object gate a_u)
    - Context Head: 1x1 conv mapping C -> K (logits c_u)
    - Object Pooling: Gate-weighted mean of z_u (inference & test prediction)
    - Context Sink: (1 - a_u)-weighted mean of c_u (training shortcut absorber)
    """
    def __init__(
        self,
        in_features: int = 2048,
        hidden_dim: int = 256,
        num_classes: int = 2,
        eps: float = 1e-6
    ):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps
        
        # 1. Evidence head: Wo * f_u + bo
        self.evidence_conv = nn.Conv2d(in_features, num_classes, kernel_size=1)
        
        # 2. Gate head: MLP (2048 -> 256 -> 1) + Sigmoid
        self.gate_conv1 = nn.Conv2d(in_features, hidden_dim, kernel_size=1)
        self.gate_act = nn.ReLU(inplace=True)
        self.gate_conv2 = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        
        # 3. Context head: Wc * f_u + bc
        self.context_conv = nn.Conv2d(in_features, num_classes, kernel_size=1)

    def compute_gate(self, x: torch.Tensor) -> torch.Tensor:
        """Computes spatial object gate a in (0, 1)."""
        h = self.gate_act(self.gate_conv1(x))
        return torch.sigmoid(self.gate_conv2(h))

    def forward(
        self,
        x: torch.Tensor,
        return_context: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass.
        Args:
            x: Feature map [B, C, H, W]
            return_context: If True, also computes and returns context sink logits.
        Returns:
            y_obj: Object logits [B, K]
            y_ctx: Context sink logits [B, K] (if return_context=True, else None)
            a: Spatial gate map [B, 1, H, W]
        """
        B, C, H, W = x.shape
        
        # Evidence logits: [B, K, H, W]
        z = self.evidence_conv(x)
        
        # Object gate: [B, 1, H, W]
        a = self.compute_gate(x)
        
        # Reshape to spatial vectors
        z_flat = z.view(B, self.num_classes, H * W)
        a_flat = a.view(B, 1, H * W)
        
        # Object pooling: sum(a_u * z_u) / (sum(a_u) + eps)
        a_sum = a_flat.sum(dim=-1) + self.eps  # [B, 1]
        y_obj = (a_flat * z_flat).sum(dim=-1) / a_sum  # [B, K]
        
        y_ctx = None
        if return_context:
            c = self.context_conv(x)  # [B, K, H, W]
            c_flat = c.view(B, self.num_classes, H * W)
            inv_a_flat = 1.0 - a_flat
            inv_a_sum = inv_a_flat.sum(dim=-1) + self.eps  # [B, 1]
            y_ctx = (inv_a_flat * c_flat).sum(dim=-1) / inv_a_sum  # [B, K]
            
        return y_obj, y_ctx, a

    def get_evidence_map(self, x: torch.Tensor, class_idx: int) -> torch.Tensor:
        """
        Computes faithful local evidence map for target class: a_u * z_u[class_idx].
        Returns map of shape [B, H, W].
        """
        z = self.evidence_conv(x)  # [B, K, H, W]
        a = self.compute_gate(x)    # [B, 1, H, W]
        z_class = z[:, class_idx:class_idx+1, :, :]  # [B, 1, H, W]
        evidence = (a * z_class).squeeze(1)  # [B, H, W]
        return evidence
