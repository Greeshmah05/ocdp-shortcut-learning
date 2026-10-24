"""
Unit tests for all pooling heads, OCDP components, loss functions, and evaluation metrics.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from src.models.heads import GAPHead, ABMILHead, TopKHead, DARHead, OCDPHead
from src.losses.objectives import (
    GeneralizedCrossEntropyLoss,
    AreaPriorLoss,
    DynamicSampleReweighting,
    CounterfactualConsistencyLoss
)
from src.evaluation.metrics import (
    compute_group_metrics,
    compute_calibration_metrics,
    compute_minority_detection_metrics
)


def test_head_shapes_and_gradients():
    """Verify that all 5 pooling heads accept [B, 2048, 7, 7] and produce [B, 2] with valid gradients."""
    B, C, H, W = 4, 2048, 7, 7
    num_classes = 2
    heads = [
        GAPHead(in_features=C, num_classes=num_classes),
        ABMILHead(in_features=C, hidden_dim=256, num_classes=num_classes),
        TopKHead(in_features=C, num_classes=num_classes, k=5),
        DARHead(in_features=C, hidden_dim=256, num_classes=num_classes),
        OCDPHead(in_features=C, hidden_dim=256, num_classes=num_classes)
    ]

    for head in heads:
        x = torch.randn(B, C, H, W, requires_grad=True)
        if isinstance(head, OCDPHead):
            out, ctx, gate = head(x, return_context=True)
            assert out.shape == (B, num_classes)
            assert ctx.shape == (B, num_classes)
            assert gate.shape == (B, 1, H, W)
            assert (gate >= 0.0).all() and (gate <= 1.0).all()
            loss = out.sum() + ctx.sum()
        else:
            out = head(x)
            assert out.shape == (B, num_classes)
            loss = out.sum()

        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


def test_ocdp_parameter_overhead():
    """Verify that OCDP adds approximately 0.53M parameters (~2% of ResNet-50)."""
    ocdp = OCDPHead(in_features=2048, hidden_dim=256, num_classes=2)
    total_params = sum(p.numel() for p in ocdp.parameters())
    # Expected: Conv1(2048*256 + 256) + Conv2(256*1 + 1) + EvConv(2048*2 + 2) + CtxConv(2048*2 + 2)
    # Approx 524k + 257 + 4098 + 4098 = ~533k parameters
    assert 500_000 < total_params < 600_000


def test_faithful_evidence_map():
    """Verify that OCDP computes local evidence maps [B, H, W] without NaN."""
    B, C, H, W = 2, 2048, 7, 7
    ocdp = OCDPHead(in_features=C, hidden_dim=256, num_classes=2)
    x = torch.randn(B, C, H, W)
    ev_map_0 = ocdp.get_evidence_map(x, class_idx=0)
    ev_map_1 = ocdp.get_evidence_map(x, class_idx=1)
    assert ev_map_0.shape == (B, H, W)
    assert ev_map_1.shape == (B, H, W)
    assert not torch.isnan(ev_map_0).any()


def test_generalized_cross_entropy_loss():
    """Verify GCE loss behaves properly and matches CE when q -> 0."""
    logits = torch.tensor([[2.0, -1.0], [-0.5, 1.5]])
    targets = torch.tensor([0, 1])
    
    gce = GeneralizedCrossEntropyLoss(q=0.7)
    loss = gce(logits, targets)
    assert loss.item() > 0.0
    assert not torch.isnan(loss)

    # When q -> 0, GCE approaches standard CE
    gce_zero = GeneralizedCrossEntropyLoss(q=0.0)
    ce_loss = nn.CrossEntropyLoss()(logits, targets)
    assert torch.isclose(gce_zero(logits, targets), ce_loss, atol=1e-4)


def test_area_prior_loss():
    """Verify area prior loss is 0 when mean gate equals target rho."""
    area_loss = AreaPriorLoss(rho=0.3)
    perfect_gate = torch.full((4, 1, 7, 7), 0.3)
    assert torch.isclose(area_loss(perfect_gate), torch.tensor(0.0), atol=1e-6)

    deviated_gate = torch.full((4, 1, 7, 7), 0.8)
    expected_penalty = (0.8 - 0.3) ** 2
    assert torch.isclose(area_loss(deviated_gate), torch.tensor(expected_penalty), atol=1e-5)


def test_dynamic_sample_reweighting():
    """Verify dynamic reweighting assigns higher weight to high-context-error samples."""
    reweighter = DynamicSampleReweighting(w_min=0.2, w_max=4.0)
    # Sample 0: bias-aligned (ctx loss low, obj loss low)
    # Sample 1: bias-conflicting (ctx loss high, obj loss low)
    ce_obj = torch.tensor([0.2, 0.2])
    ce_ctx = torch.tensor([0.1, 2.0])

    weights = reweighter(ce_obj, ce_ctx)
    assert weights.shape == (2,)
    # Sample 1 should receive significantly higher weight than Sample 0
    assert weights[1].item() > weights[0].item()
    # Batch mean must be close to 1.0
    assert torch.isclose(weights.mean(), torch.tensor(1.0), atol=1e-4)


def test_evaluation_metrics_computation():
    """Verify group metrics, WGA, MGA, and minority detection AUROC."""
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 0, 1, 1])  # G1 correct, G2 incorrect
    groups = np.array([0, 1, 2, 3, 0, 3])

    res = compute_group_metrics(y_true, y_pred, groups)
    assert "worst_group_acc" in res
    assert "mean_group_acc" in res
    assert "weighted_avg_acc" in res
    assert 0.0 <= res["worst_group_acc"] <= 1.0

    # Calibration test
    probs = np.array([
        [0.9, 0.1],
        [0.8, 0.2],
        [0.1, 0.9],
        [0.6, 0.4],
        [0.7, 0.3],
        [0.2, 0.8]
    ])
    calib = compute_calibration_metrics(y_true, probs)
    assert 0.0 <= calib["ece"] <= 1.0
    assert calib["brier_score"] >= 0.0

    # Minority detection test
    # True minority are groups 1 and 2
    shortcut_scores = np.array([0.9, 0.2, 0.1, 0.95, 0.85, 0.9])
    det = compute_minority_detection_metrics(shortcut_scores, groups)
    assert "minority_detection_auroc" in det
    assert det["minority_detection_auroc"] >= 0.8
