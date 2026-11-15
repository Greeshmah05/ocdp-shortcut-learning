"""
Unit tests for Stage 4 End-to-End OCDP architecture and counterfactual consistency.
Verifies that gradients flow from the combined objective (Object loss, GCE Context Sink,
Area Prior, and Counterfactual Consistency) back into ResNet-50 conv1 through layer4.
"""

import pytest
import torch
import torch.nn.functional as F

from src.models.backbone import ClassifierWithHead
from src.losses.objectives import (
    GeneralizedCrossEntropyLoss,
    AreaPriorLoss,
    DynamicSampleReweighting,
    CounterfactualConsistencyLoss
)


def test_stage4_end_to_end_gradient_flow():
    """Verify gradients propagate across all ResNet-50 layers and the OCDP head."""
    B, C_in, H, W = 2, 3, 224, 224
    model = ClassifierWithHead(
        head_type="ocdp",
        num_classes=2,
        pretrained_backbone=False,
        freeze_backbone=False,
        detach_context_gate=True
    )

    imgs = torch.randn(B, C_in, H, W, requires_grad=True)
    targets = torch.tensor([0, 1])

    # Clean pass
    y_obj, y_ctx, gate = model(imgs, return_context=True)
    assert y_obj.shape == (B, 2)
    assert y_ctx.shape == (B, 2)
    assert gate.shape == (B, 1, 7, 7)

    # Counterfactual consistency pass
    gate_2d = F.interpolate(gate.detach(), size=(H, W), mode="bilinear", align_corners=False)
    imgs_masked = imgs * gate_2d
    y_obj_masked = model(imgs_masked, return_context=False)

    gce_fn = GeneralizedCrossEntropyLoss(q=0.7)
    area_fn = AreaPriorLoss(rho=0.3)
    cons_fn = CounterfactualConsistencyLoss()

    loss_obj = F.cross_entropy(y_obj, targets)
    loss_ctx = gce_fn(y_ctx, targets)
    loss_area = area_fn(gate)
    loss_cons = cons_fn(y_obj.detach(), y_obj_masked)

    total_loss = loss_obj + loss_ctx + loss_area + loss_cons
    total_loss.backward()

    # Verify input gradient
    assert imgs.grad is not None
    assert not torch.isnan(imgs.grad).any()

    # Verify ResNet-50 backbone gradients (conv1, layer1, layer4)
    assert model.backbone.conv1.weight.grad is not None
    assert model.backbone.layer1[0].conv1.weight.grad is not None
    assert model.backbone.layer4[0].conv1.weight.grad is not None

    # Verify OCDP head gradients
    assert model.head.evidence_conv.weight.grad is not None
    assert model.head.gate_conv1.weight.grad is not None
    assert model.head.context_conv.weight.grad is not None


def test_stage4_checkpoint_save_and_load(tmp_path):
    """Verify end-to-end model checkpoint saving and loading without state discrepancy."""
    model1 = ClassifierWithHead(head_type="ocdp", num_classes=2, pretrained_backbone=False, freeze_backbone=False)
    ckpt_path = tmp_path / "test_e2e_model.pt"

    torch.save(model1.state_dict(), ckpt_path)

    model2 = ClassifierWithHead(head_type="ocdp", num_classes=2, pretrained_backbone=False, freeze_backbone=False)
    model2.load_state_dict(torch.load(ckpt_path, weights_only=True))

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out1 = model1(x, return_context=False)
        out2 = model2(x, return_context=False)
    assert torch.allclose(out1, out2, atol=1e-6)

