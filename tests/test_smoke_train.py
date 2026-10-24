"""
Smoke test: executes 5 batches of forward/backward, checkpoint save/load, and validation evaluation.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.models.backbone import ClassifierWithHead
from src.models.heads import OCDPHead
from src.losses.objectives import GeneralizedCrossEntropyLoss, AreaPriorLoss, DynamicSampleReweighting
from src.evaluation.metrics import compute_group_metrics


def test_smoke_training_loop(tmp_path):
    """Executes a 5-batch smoke run verifying training step, checkpointing, and evaluation."""
    device = torch.device("cpu")
    num_classes = 2
    head = OCDPHead(in_features=2048, hidden_dim=256, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    
    gce_fn = GeneralizedCrossEntropyLoss(q=0.7, reduction="none")
    area_fn = AreaPriorLoss(rho=0.3)
    reweighter = DynamicSampleReweighting(w_min=0.2, w_max=4.0)
    ce_loss_none = nn.CrossEntropyLoss(reduction="none")

    # Synthetic batches representing extracted features [B, 2048, 7, 7]
    B = 4
    feats = torch.randn(B * 5, 2048, 7, 7)
    labels = torch.randint(0, 2, (B * 5,))
    groups = torch.randint(0, 4, (B * 5,))
    loader = DataLoader(TensorDataset(feats, labels, groups), batch_size=B)

    # 1. Run 5 training steps
    head.train()
    step_count = 0
    for b_feats, b_labels, b_groups in loader:
        optimizer.zero_grad()
        y_obj, y_ctx, gate = head(b_feats, return_context=True)
        ce_obj = ce_loss_none(y_obj, b_labels)
        ce_ctx = ce_loss_none(y_ctx, b_labels)
        weights = reweighter(ce_obj.detach(), ce_ctx.detach())
        loss_obj = (weights * ce_obj).mean()
        loss_ctx = gce_fn(y_ctx, b_labels).mean()
        loss_area = area_fn(gate)
        loss = loss_obj + loss_ctx + 0.5 * loss_area
        loss.backward()
        optimizer.step()
        step_count += 1

    assert step_count == 5

    # 2. Checkpoint save and load
    ckpt_path = os.path.join(tmp_path, "smoke_checkpoint.pt")
    torch.save({"model_state_dict": head.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, ckpt_path)
    assert os.path.exists(ckpt_path)

    new_head = OCDPHead(in_features=2048, hidden_dim=256, num_classes=num_classes)
    checkpoint = torch.load(ckpt_path, weights_only=True)
    new_head.load_state_dict(checkpoint["model_state_dict"])

    # 3. Validation evaluation
    new_head.eval()
    with torch.no_grad():
        val_obj, val_ctx, _ = new_head(feats[:8], return_context=True)
        preds = torch.argmax(val_obj, dim=-1).numpy()
        metrics = compute_group_metrics(labels[:8].numpy(), preds, groups[:8].numpy())
        assert "worst_group_acc" in metrics
        assert "overall_acc" in metrics
