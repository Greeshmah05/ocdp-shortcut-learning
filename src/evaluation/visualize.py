"""
Visualization Suite for OCDP and Baseline Pooling Heads:
Generates publication-quality figures:
1. Head comparison bar chart (WGA & MGA with error bars across 3 seeds)
2. Group-by-group accuracy breakdown (G0..G3)
3. Minority detection ROC curve
4. Faithful evidence map visualizations
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_head_wga_comparison(
    summary_csv_path: str = "results/pilot_p1/pilot_p1_summary.csv",
    output_path: str = "results/pilot_p1/head_wga_comparison.png"
) -> None:
    """Generates comparison bar chart with error bars for WGA and MGA."""
    if not os.path.exists(summary_csv_path):
        print(f"File not found: {summary_csv_path}")
        return

    df = pd.read_csv(summary_csv_path)
    heads = ["gap", "abmil", "topk", "dar", "ocdp"]
    head_labels = ["GAP (Baseline)", "ABMIL", "Top-K (FocusMIL)", "DAR-Style", "OCDP (Ours)"]

    wga_means, wga_stds = [], []
    mga_means, mga_stds = [], []

    for h in heads:
        sub = df[df["head"] == h]
        if len(sub) > 0:
            wga_means.append(sub["worst_group_acc"].mean())
            wga_stds.append(sub["worst_group_acc"].std())
            mga_means.append(sub["mean_group_acc"].mean())
            mga_stds.append(sub["mean_group_acc"].std())
        else:
            wga_means.append(0.0)
            wga_stds.append(0.0)
            mga_means.append(0.0)
            mga_stds.append(0.0)

    x = np.arange(len(heads))
    width = 0.35

    plt.figure(figsize=(9, 5.5), dpi=200)
    colors_wga = ["#4A90E2", "#50E3C2", "#F5A623", "#9013FE", "#E94E77"]
    colors_mga = ["#A1C4FD", "#B5F2E6", "#FAD8A5", "#D7B5FD", "#F7A8BE"]

    bars1 = plt.bar(x - width/2, wga_means, width, yerr=wga_stds, capsize=5, label="Worst-Group Acc (WGA)", color=colors_wga, edgecolor="black", alpha=0.9)
    bars2 = plt.bar(x + width/2, mga_means, width, yerr=mga_stds, capsize=5, label="Mean-Group Acc (MGA)", color=colors_mga, edgecolor="black", alpha=0.9)

    plt.ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    plt.title("Waterbirds Robustness Across Pooling Heads (3 Seeds, Tier 3)", fontsize=13, fontweight="bold", pad=15)
    plt.xticks(x, head_labels, fontsize=10, fontweight="bold")
    plt.ylim(0, 105)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=10)

    # Annotate bar values
    for b in bars1:
        h_val = b.get_height()
        if h_val > 0:
            plt.text(b.get_x() + b.get_width()/2., h_val + 2, f"{h_val:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_group_accuracy_breakdown(
    summary_csv_path: str = "results/pilot_p1/pilot_p1_summary.csv",
    output_path: str = "results/pilot_p1/group_accuracy_breakdown.png"
) -> None:
    """Generates per-group accuracy comparison chart across heads."""
    if not os.path.exists(summary_csv_path):
        return

    df = pd.read_csv(summary_csv_path)
    heads = ["gap", "abmil", "topk", "dar", "ocdp"]
    head_names = ["GAP", "ABMIL", "Top-K", "DAR", "OCDP"]

    group_keys = ["G0_acc", "G1_acc", "G2_acc", "G3_acc"]
    group_titles = [
        "G0: Land/Land (Maj)",
        "G1: Land/Water (Min)",
        "G2: Water/Land (Min)",
        "G3: Water/Water (Maj)"
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), sharey=True, dpi=200)

    for i, (gk, title) in enumerate(zip(group_keys, group_titles)):
        ax = axes[i]
        means = [df[df["head"] == h][gk].mean() if len(df[df["head"] == h]) > 0 else 0 for h in heads]
        stds = [df[df["head"] == h][gk].std() if len(df[df["head"] == h]) > 0 else 0 for h in heads]
        
        bars = ax.bar(head_names, means, yerr=stds, capsize=4, color=["#4A90E2", "#50E3C2", "#F5A623", "#9013FE", "#E94E77"], edgecolor="black", alpha=0.85)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 105)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        if i == 0:
            ax.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")

        for b in bars:
            h_val = b.get_height()
            if h_val > 0:
                ax.text(b.get_x() + b.get_width()/2., h_val + 2, f"{h_val:.0f}%", ha="center", va="bottom", fontsize=8)

    plt.suptitle("Per-Group Accuracy Breakdown across Pooling Heads", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    plot_head_wga_comparison()
    plot_group_accuracy_breakdown()
