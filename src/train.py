"""
Unified Training and Evaluation Engine for OCDP and Baseline Pooling Heads.
Supports feature extraction caching, Pilot P1 head comparisons, and end-to-end training.
Complies with Tier 3 supervision rules and evaluation protocols.
"""

import os
import sys
import random
import json
import argparse
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from tqdm import tqdm
import yaml

from src.data.waterbirds import build_dataloaders, WaterbirdsDataset
from src.models.heads import GAPHead, ABMILHead, TopKHead, DARHead, OCDPHead
from src.models.backbone import ResNet50FeatureExtractor
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


def extract_spatial_features(
    backbone: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Passes images through ResNet-50 backbone to extract layer4 feature maps [N, 2048, 7, 7].
    Collects images, labels, and groups (groups for validation evaluation only).
    """
    backbone.eval()
    features_list = []
    labels_list = []
    groups_list = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting spatial features"):
            if len(batch) == 5:
                # Evaluation loader: img, y, g, place, idx
                imgs, ys, gs, _, _ = batch
            else:
                # Training loader: img, y, idx
                imgs, ys, _ = batch
                gs = torch.zeros_like(ys)

            imgs = imgs.to(device)
            feats = backbone(imgs).cpu()  # [B, 2048, 7, 7]
            features_list.append(feats)
            labels_list.append(ys)
            groups_list.append(gs)

    features = torch.cat(features_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    groups = torch.cat(groups_list, dim=0)
    return features, labels, groups


def build_head(head_type: str, in_features: int = 2048, num_classes: int = 2) -> nn.Module:
    """Factory for pooling heads."""
    head_type = head_type.lower()
    if head_type == "gap":
        return GAPHead(in_features=in_features, num_classes=num_classes)
    elif head_type == "abmil":
        return ABMILHead(in_features=in_features, hidden_dim=256, num_classes=num_classes)
    elif head_type == "topk":
        return TopKHead(in_features=in_features, num_classes=num_classes, k=5)
    elif head_type == "dar":
        return DARHead(in_features=in_features, hidden_dim=256, num_classes=num_classes)
    elif head_type == "ocdp":
        return OCDPHead(in_features=in_features, hidden_dim=256, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown head type: {head_type}")


def train_head_on_features(
    head_type: str,
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    val_feats: torch.Tensor,
    val_labels: torch.Tensor,
    val_groups: torch.Tensor,
    config: Dict[str, Any],
    seed: int,
    device: torch.device
) -> Dict[str, Any]:
    """
    Trains a pooling head on extracted spatial features.
    Implements Pilot P1 protocol with Tier 3 model selection.
    """
    set_seed(seed)
    num_classes = 2
    head = build_head(head_type, in_features=2048, num_classes=num_classes).to(device)

    epochs = config.get("training", {}).get("epochs", 15)
    batch_size = config.get("training", {}).get("batch_size", 64)
    lr = config.get("training", {}).get("lr", 0.001)
    weight_decay = config.get("training", {}).get("weight_decay", 1e-4)

    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)

    # OCDP specific parameters
    is_ocdp = (head_type.lower() == "ocdp")
    ocdp_cfg = config.get("ocdp_hyperparameters", {})
    gce_loss_fn = GeneralizedCrossEntropyLoss(q=ocdp_cfg.get("gce_q", 0.7), reduction="none")
    area_loss_fn = AreaPriorLoss(rho=ocdp_cfg.get("rho_area", 0.3))
    reweighter = DynamicSampleReweighting(
        w_min=ocdp_cfg.get("w_min", 0.2),
        w_max=ocdp_cfg.get("w_max", 4.0)
    )
    lambda_ctx = ocdp_cfg.get("lambda_ctx", 1.0)
    lambda_area = ocdp_cfg.get("lambda_area", 0.5)
    warmup_epochs = ocdp_cfg.get("warmup_epochs", 2)

    ce_loss_none = nn.CrossEntropyLoss(reduction="none")
    ce_loss_mean = nn.CrossEntropyLoss(reduction="mean")

    train_ds = TensorDataset(train_feats, train_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    epoch_logs = []
    best_t3_metric = -1.0  # Best worst-class accuracy (T3 rule)
    best_t2_metric = -1.0  # Best worst-group accuracy (T2 logged for comparison)
    best_val_metrics = {}

    for epoch in range(1, epochs + 1):
        head.train()
        total_loss = 0.0
        gate_means = []
        batch_count = 0

        for feats_b, ys_b in train_loader:
            feats_b = feats_b.to(device)
            ys_b = ys_b.to(device)
            optimizer.zero_grad()

            if is_ocdp:
                y_obj, y_ctx, gate = head(feats_b, return_context=True)
                mean_gate = gate.mean().item()
                gate_means.append(mean_gate)

                ce_obj_per_sample = ce_loss_none(y_obj, ys_b)
                ce_ctx_per_sample = ce_loss_none(y_ctx, ys_b)

                if epoch <= warmup_epochs:
                    loss_obj = ce_obj_per_sample.mean()
                else:
                    weights = reweighter(ce_obj_per_sample.detach(), ce_ctx_per_sample.detach())
                    loss_obj = (weights * ce_obj_per_sample).mean()

                loss_ctx = gce_loss_fn(y_ctx, ys_b).mean()
                loss_area = area_loss_fn(gate)

                loss = loss_obj + lambda_ctx * loss_ctx + lambda_area * loss_area
            else:
                logits = head(feats_b)
                loss = ce_loss_mean(logits, ys_b)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1

        avg_loss = total_loss / max(1, batch_count)
        avg_gate = float(np.mean(gate_means)) if gate_means else 0.0

        # Validation evaluation (at epoch end)
        head.eval()
        with torch.no_grad():
            v_feats = val_feats.to(device)
            if is_ocdp:
                val_obj, val_ctx, val_gate = head(v_feats, return_context=True)
                val_probs = F.softmax(val_obj, dim=-1).cpu().numpy()
                val_ctx_probs = F.softmax(val_ctx, dim=-1).cpu().numpy()
                # Built-in shortcut score: s_i = p_ctx(y_i|x_i)
                val_true_labels = val_labels.numpy()
                shortcut_scores = val_ctx_probs[np.arange(len(val_true_labels)), val_true_labels]
            else:
                val_logits = head(v_feats)
                val_probs = F.softmax(val_logits, dim=-1).cpu().numpy()
                shortcut_scores = None

        val_preds = np.argmax(val_probs, axis=1)
        val_metrics = compute_group_metrics(
            val_labels.numpy(),
            val_preds,
            val_groups.numpy()
        )

        calib_metrics = compute_calibration_metrics(val_labels.numpy(), val_probs)
        val_metrics.update(calib_metrics)

        if shortcut_scores is not None:
            det_metrics = compute_minority_detection_metrics(shortcut_scores, val_groups.numpy())
            val_metrics.update(det_metrics)

        # Checkpoint selection (Tier 3: select on worst_class_acc)
        t3_score = val_metrics["worst_class_acc"]
        t2_score = val_metrics["worst_group_acc"]

        is_best_t3 = (t3_score > best_t3_metric)
        if is_best_t3:
            best_t3_metric = t3_score
            best_val_metrics = dict(val_metrics)
            best_val_metrics["selected_epoch"] = epoch
            best_val_metrics["selection_tier"] = "T3 (Worst-Class Acc)"

        if t2_score > best_t2_metric:
            best_t2_metric = t2_score

        epoch_info = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "mean_gate": avg_gate,
            "val_wga": val_metrics["worst_group_acc"],
            "val_mga": val_metrics["mean_group_acc"],
            "val_wca": val_metrics["worst_class_acc"],
            "val_overall": val_metrics["overall_acc"]
        }
        epoch_logs.append(epoch_info)

    return {
        "head": head_type,
        "seed": seed,
        "supervision_tier": "T3",
        "best_val_metrics": best_val_metrics,
        "best_t2_val_wga": best_t2_metric,
        "epoch_logs": epoch_logs
    }


def run_pilot_p1_experiments(
    data_dir: str = "data/raw",
    config_path: str = "configs/pilot_p1.yaml",
    cache_dir: str = "checkpoints/cache"
) -> Dict[str, Any]:
    """
    Executes the full Pilot P1 benchmark across all 5 heads and 3 seeds.
    Extracts features once and caches them, then trains each head under identical conditions.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Pilot P1 on device: {device}")

    os.makedirs(cache_dir, exist_ok=True)
    train_cache = os.path.join(cache_dir, "train_feats.pt")
    val_cache = os.path.join(cache_dir, "val_feats.pt")

    if os.path.exists(train_cache) and os.path.exists(val_cache):
        print("Loading cached spatial features...")
        train_data = torch.load(train_cache, weights_only=True)
        val_data = torch.load(val_cache, weights_only=True)
        train_feats, train_labels, _ = train_data
        val_feats, val_labels, val_groups = val_data
    else:
        print("Extracting features with ImageNet-pretrained ResNet-50...")
        train_loader, val_loader, _ = build_dataloaders(data_dir=data_dir, batch_size=32)
        backbone = ResNet50FeatureExtractor(pretrained=True).to(device)

        train_feats, train_labels, train_groups = extract_spatial_features(backbone, train_loader, device)
        val_feats, val_labels, val_groups = extract_spatial_features(backbone, val_loader, device)

        torch.save((train_feats, train_labels, train_groups), train_cache)
        torch.save((val_feats, val_labels, val_groups), val_cache)
        print("Cached features successfully.")

    heads = config.get("heads", ["gap", "abmil", "topk", "dar", "ocdp"])
    seeds = config.get("seeds", [42, 100, 2024])
    output_dir = config.get("output_dir", "results/pilot_p1")
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}
    summary_rows = []

    for head_type in heads:
        print(f"\n==========================================")
        print(f"Benchmarking Head: {head_type.upper()}")
        print(f"==========================================")
        head_results = []

        for seed in seeds:
            print(f"--> Training {head_type} (Seed {seed})...")
            res = train_head_on_features(
                head_type=head_type,
                train_feats=train_feats,
                train_labels=train_labels,
                val_feats=val_feats,
                val_labels=val_labels,
                val_groups=val_groups,
                config=config,
                seed=seed,
                device=device
            )
            head_results.append(res)
            best_m = res["best_val_metrics"]
            summary_rows.append({
                "head": head_type,
                "seed": seed,
                "tier": "T3",
                "overall_acc": best_m["overall_acc"] * 100,
                "G0_acc": best_m["G0_acc"] * 100,
                "G1_acc": best_m["G1_acc"] * 100,
                "G2_acc": best_m["G2_acc"] * 100,
                "G3_acc": best_m["G3_acc"] * 100,
                "mean_group_acc": best_m["mean_group_acc"] * 100,
                "worst_group_acc": best_m["worst_group_acc"] * 100,
                "weighted_avg_acc": best_m["weighted_avg_acc"] * 100,
                "t2_val_wga": res["best_t2_val_wga"] * 100
            })

        all_results[head_type] = head_results

    # Save detailed JSON and CSV summaries
    with open(os.path.join(output_dir, "pilot_p1_results.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(output_dir, "pilot_p1_summary.csv"), index=False)

    print("\n==========================================")
    print("PILOT P1 EXPERIMENT SUMMARY TABLE")
    print("==========================================")
    grouped = df_summary.groupby("head").agg({
        "worst_group_acc": ["mean", "std"],
        "mean_group_acc": ["mean", "std"],
        "overall_acc": ["mean", "std"],
        "t2_val_wga": ["mean", "std"]
    })
    print(grouped)
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCDP & Pooling Heads Training Engine")
    parser.add_argument("--pilot", action="store_true", help="Run Pilot P1 experiments across 5 heads and 3 seeds")
    parser.add_argument("--config", type=str, default="configs/pilot_p1.yaml", help="Config file path")
    args = parser.parse_args()

    if args.pilot:
        run_pilot_p1_experiments(config_path=args.config)
    else:
        # Default to running pilot
        run_pilot_p1_experiments(config_path=args.config)
