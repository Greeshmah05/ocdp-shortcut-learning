"""
Stage 4: End-to-End OCDP Training Engine.
Fine-tunes the ResNet-50 backbone alongside the OCDP head with:
1. Differential learning rates (lr_backbone=1e-5, lr_head=1e-4)
2. Counterfactual Consistency Loss (KL divergence between clean & gate-masked inputs)
3. Dataset-level EMA sample reweighting
4. Stop-gradient context routing and sharpened area prior
5. Tier 3 validation selection strictly via worst-class accuracy
"""

import os
import sys
import json
import random
import argparse
from typing import Dict, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import pandas as pd

from src.data.waterbirds import build_dataloaders
from src.models.backbone import ClassifierWithHead
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


def set_seed(seed: int) -> None:
    """Sets random seeds across Python, NumPy, and PyTorch for exact reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    gce_loss_fn: nn.Module,
    area_loss_fn: nn.Module,
    reweighter: nn.Module,
    cons_loss_fn: nn.Module,
    epoch: int,
    warmup_epochs: int,
    lambda_ctx: float,
    lambda_area: float,
    lambda_cons: float,
    grad_clip: float,
    device: torch.device,
    smoke_batches: Optional[int] = None
) -> Dict[str, float]:
    """Executes one epoch of end-to-end training."""
    model.train()
    total_loss = 0.0
    total_obj = 0.0
    total_ctx = 0.0
    total_area = 0.0
    total_cons = 0.0
    gate_means = []
    batch_count = 0

    ce_none = nn.CrossEntropyLoss(reduction="none")

    pbar = tqdm(loader, desc=f"Epoch {epoch:2d} [Train]", leave=False)
    for batch_idx, (imgs, targets, indices) in enumerate(pbar):
        if smoke_batches and batch_idx >= smoke_batches:
            break

        imgs = imgs.to(device)
        targets = targets.to(device)
        indices = indices.to(device)
        optimizer.zero_grad()

        # 1. Clean forward pass
        y_obj, y_ctx, gate = model(imgs, return_context=True)
        mean_gate = gate.mean().item()
        gate_means.append(mean_gate)

        ce_obj = ce_none(y_obj, targets)
        ce_ctx = ce_none(y_ctx, targets)

        # 2. Dynamic Reweighting & Counterfactual Consistency
        if epoch <= warmup_epochs:
            loss_obj = ce_obj.mean()
            loss_cons = torch.tensor(0.0, device=device)
        else:
            weights = reweighter(ce_obj.detach(), ce_ctx.detach(), sample_indices=indices)
            loss_obj = (weights * ce_obj).mean()

            if lambda_cons > 0.0:
                # Upsample gate to raw image resolution [B, 1, H, W]
                gate_2d = F.interpolate(
                    gate.detach(),
                    size=(imgs.shape[2], imgs.shape[3]),
                    mode="bilinear",
                    align_corners=False
                )
                # Counterfactual masked image
                imgs_masked = imgs * gate_2d
                y_obj_masked = model(imgs_masked, return_context=False)
                loss_cons = cons_loss_fn(y_obj.detach(), y_obj_masked)
            else:
                loss_cons = torch.tensor(0.0, device=device)

        loss_ctx = gce_loss_fn(y_ctx, targets).mean()
        loss_area = area_loss_fn(gate)

        loss = loss_obj + lambda_ctx * loss_ctx + lambda_area * loss_area + lambda_cons * loss_cons

        loss.backward()
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_obj += loss_obj.item()
        total_ctx += loss_ctx.item()
        total_area += loss_area.item()
        total_cons += loss_cons.item()
        batch_count += 1

    count = max(1, batch_count)
    return {
        "train_loss": total_loss / count,
        "loss_obj": total_obj / count,
        "loss_ctx": total_ctx / count,
        "loss_area": total_area / count,
        "loss_cons": total_cons / count,
        "mean_gate": float(np.mean(gate_means)) if gate_means else 0.0
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    smoke_batches: Optional[int] = None
) -> Dict[str, Any]:
    """Evaluates model across validation groups."""
    model.eval()
    all_targets = []
    all_groups = []
    all_obj_probs = []
    all_ctx_probs = []

    with torch.no_grad():
        for batch_idx, (imgs, targets, groups, places, indices) in enumerate(loader):
            if smoke_batches and batch_idx >= smoke_batches:
                break
            imgs = imgs.to(device)
            y_obj, y_ctx, gate = model(imgs, return_context=True)
            prob_obj = F.softmax(y_obj, dim=-1).cpu().numpy()
            prob_ctx = F.softmax(y_ctx, dim=-1).cpu().numpy()

            all_targets.append(targets.numpy())
            all_groups.append(groups.numpy())
            all_obj_probs.append(prob_obj)
            all_ctx_probs.append(prob_ctx)

    y_true = np.concatenate(all_targets, axis=0)
    groups = np.concatenate(all_groups, axis=0)
    obj_probs = np.concatenate(all_obj_probs, axis=0)
    ctx_probs = np.concatenate(all_ctx_probs, axis=0)

    y_pred = np.argmax(obj_probs, axis=1)

    metrics = compute_group_metrics(y_true, y_pred, groups)
    calib = compute_calibration_metrics(y_true, obj_probs)
    metrics.update(calib)

    # Shortcut score: s_i = p_ctx(y_i | x_i)
    shortcut_scores = ctx_probs[np.arange(len(y_true)), y_true]
    det = compute_minority_detection_metrics(shortcut_scores, groups)
    metrics.update(det)

    return metrics


def train_single_seed_e2e(
    config: Dict[str, Any],
    seed: int,
    device: torch.device,
    smoke_test: bool = False
) -> Dict[str, Any]:
    """Trains a complete end-to-end OCDP model for a single seed."""
    set_seed(seed)
    print(f"\n==========================================")
    print(f"STAGE 4 END-TO-END OCDP (Seed {seed})")
    print(f"==========================================")

    data_dir = config.get("data", {}).get("data_dir", "data/raw")
    batch_size = config.get("training", {}).get("batch_size", 32)
    train_loader, val_loader, _ = build_dataloaders(data_dir=data_dir, batch_size=batch_size)

    ocdp_cfg = config.get("ocdp_hyperparameters", {})
    model = ClassifierWithHead(
        head_type="ocdp",
        num_classes=config.get("model", {}).get("num_classes", 2),
        pretrained_backbone=config.get("model", {}).get("pretrained", True),
        freeze_backbone=False,
        detach_context_gate=ocdp_cfg.get("detach_context_gate", True)
    ).to(device)

    # Differential learning rates
    lr_backbone = config.get("training", {}).get("lr_backbone", 1e-5)
    lr_head = config.get("training", {}).get("lr_head", 1e-4)
    weight_decay = config.get("training", {}).get("weight_decay", 1e-4)

    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": lr_backbone},
        {"params": model.head.parameters(), "lr": lr_head}
    ], weight_decay=weight_decay)

    # Losses
    num_train_samples = len(train_loader.dataset)
    gce_loss_fn = GeneralizedCrossEntropyLoss(q=ocdp_cfg.get("gce_q", 0.7), reduction="none")
    area_loss_fn = AreaPriorLoss(
        rho=ocdp_cfg.get("rho_area", 0.3),
        mode=ocdp_cfg.get("area_mode", "l2"),
        sharpness_coeff=ocdp_cfg.get("sharpness_coeff", 0.1)
    )
    reweighter = DynamicSampleReweighting(
        w_min=ocdp_cfg.get("w_min", 0.1),
        w_max=ocdp_cfg.get("w_max", 10.0),
        gamma=ocdp_cfg.get("weight_gamma", 1.5),
        momentum=ocdp_cfg.get("ema_momentum", 0.7),
        num_samples=num_train_samples
    )
    cons_loss_fn = CounterfactualConsistencyLoss()

    epochs = 1 if smoke_test else config.get("training", {}).get("epochs", 50)
    warmup_epochs = 0 if smoke_test else ocdp_cfg.get("warmup_epochs", 5)
    lambda_ctx = ocdp_cfg.get("lambda_ctx", 1.0)
    lambda_area = ocdp_cfg.get("lambda_area", 1.0)
    lambda_cons = ocdp_cfg.get("lambda_cons", 1.0)
    grad_clip = config.get("training", {}).get("grad_clip", 1.0)

    output_dir = config.get("output_dir", "results/stage4_e2e")
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    best_t3_metric = -1.0
    best_t2_metric = -1.0
    best_val_metrics = {}
    epoch_logs = []

    smoke_batches = 2 if smoke_test else None

    for epoch in range(1, epochs + 1):
        train_info = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            gce_loss_fn=gce_loss_fn,
            area_loss_fn=area_loss_fn,
            reweighter=reweighter,
            cons_loss_fn=cons_loss_fn,
            epoch=epoch,
            warmup_epochs=warmup_epochs,
            lambda_ctx=lambda_ctx,
            lambda_area=lambda_area,
            lambda_cons=lambda_cons,
            grad_clip=grad_clip,
            device=device,
            smoke_batches=smoke_batches
        )

        val_metrics = evaluate(model=model, loader=val_loader, device=device, smoke_batches=smoke_batches)

        t3_score = val_metrics["worst_class_acc"]
        t2_score = val_metrics["worst_group_acc"]

        is_best_t3 = (t3_score > best_t3_metric)
        if is_best_t3:
            best_t3_metric = t3_score
            best_val_metrics = dict(val_metrics)
            best_val_metrics["selected_epoch"] = epoch
            best_val_metrics["selection_tier"] = "T3 (Worst-Class Acc)"
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f"best_ocdp_e2e_seed{seed}.pt"))

        if t2_score > best_t2_metric:
            best_t2_metric = t2_score

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_info["train_loss"],
            "loss_obj": train_info["loss_obj"],
            "loss_ctx": train_info["loss_ctx"],
            "loss_area": train_info["loss_area"],
            "loss_cons": train_info["loss_cons"],
            "mean_gate": train_info["mean_gate"],
            "val_wga": val_metrics["worst_group_acc"],
            "val_mga": val_metrics["mean_group_acc"],
            "val_wca": val_metrics["worst_class_acc"],
            "val_overall": val_metrics["overall_acc"]
        }
        epoch_logs.append(epoch_record)
        print(f"Epoch {epoch:2d}/{epochs} | Loss: {train_info['train_loss']:.4f} (Obj: {train_info['loss_obj']:.3f}, Cons: {train_info['loss_cons']:.3f}) | Gate: {train_info['mean_gate']:.3f} | Val WGA: {val_metrics['worst_group_acc']*100:.2f}% | Val WCA: {val_metrics['worst_class_acc']*100:.2f}%")

    return {
        "seed": seed,
        "supervision_tier": "T3",
        "best_val_metrics": best_val_metrics,
        "best_t2_val_wga": best_t2_metric,
        "epoch_logs": epoch_logs
    }


def run_stage4_experiments(config_path: str = "configs/stage4_e2e.yaml", smoke_test: bool = False):
    """Orchestrates Stage 4 experiments across seeds."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Stage 4 End-to-End on device: {device} (Smoke: {smoke_test})")

    seeds = [42] if smoke_test else config.get("seeds", [42, 100, 2024])
    output_dir = config.get("output_dir", "results/stage4_e2e")
    os.makedirs(output_dir, exist_ok=True)

    all_seed_results = []
    summary_rows = []

    for seed in seeds:
        res = train_single_seed_e2e(config=config, seed=seed, device=device, smoke_test=smoke_test)
        all_seed_results.append(res)
        m = res["best_val_metrics"]
        summary_rows.append({
            "model": "OCDP (End-to-End)",
            "seed": seed,
            "tier": "T3",
            "overall_acc": m["overall_acc"] * 100,
            "G0_acc": m["G0_acc"] * 100,
            "G1_acc": m["G1_acc"] * 100,
            "G2_acc": m["G2_acc"] * 100,
            "G3_acc": m["G3_acc"] * 100,
            "mean_group_acc": m["mean_group_acc"] * 100,
            "worst_group_acc": m["worst_group_acc"] * 100,
            "weighted_avg_acc": m["weighted_avg_acc"] * 100,
            "t2_val_wga": res["best_t2_val_wga"] * 100
        })

    # Save detailed JSON and CSV summaries
    with open(os.path.join(output_dir, "stage4_e2e_results.json"), "w", encoding="utf-8") as f:
        json.dump(all_seed_results, f, indent=2)

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(output_dir, "stage4_e2e_summary.csv"), index=False)

    print("\n==========================================")
    print("STAGE 4 END-TO-END SUMMARY TABLE")
    print("==========================================")
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 4 End-to-End OCDP Training Engine")
    parser.add_argument("--config", type=str, default="configs/stage4_e2e.yaml", help="Path to config file")
    parser.add_argument("--smoke", action="store_true", help="Run rapid smoke test (2 batches, 1 epoch)")
    args = parser.parse_args()

    run_stage4_experiments(config_path=args.config, smoke_test=args.smoke)

