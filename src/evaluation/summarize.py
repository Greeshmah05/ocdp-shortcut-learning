"""
Result summarization and report generation.
Reads results from results/ directory and compiles markdown tables with tier tags and statistics.
"""

import os
import json
import numpy as np
import pandas as pd


# Reference published numbers from docs/literature_matrix.md
PUBLISHED_BASELINES = [
    {"method": "Group DRO (Sagawa et al. 2020)", "tier": "T1", "WGA": "91.4", "source": "Literature"},
    {"method": "DFR (Kirichenko et al. 2023)", "tier": "T2", "WGA": "92.1 - 92.3", "source": "Literature"},
    {"method": "JTT (Liu et al. 2021)", "tier": "T2", "WGA": "86.7", "source": "Literature"},
    {"method": "Decompose-and-Compose (CVPR 2024)", "tier": "T2", "WGA": "92.3", "source": "Literature"},
    {"method": "ExMap (CVPR 2024)", "tier": "T3", "WGA": "92.5", "source": "Literature"},
    {"method": "ShortcutProbe (IJCAI 2025)", "tier": "T3", "WGA": "90.8 ± 0.6", "source": "Literature"},
    {"method": "EVaLS (arXiv 2024)", "tier": "T3", "WGA": "88.4 ± 3.1", "source": "Literature"},
    {"method": "ERM Published (Various)", "tier": "T3", "WGA": "70.8 - 80.3", "source": "Literature"}
]


def summarize_pilot_results(summary_csv_path: str = "results/pilot_p1/pilot_p1_summary.csv") -> str:
    """Generates markdown summary of the Pilot P1 head comparisons."""
    if not os.path.exists(summary_csv_path):
        return "No Pilot P1 summary file found."

    df = pd.read_csv(summary_csv_path)
    grouped = df.groupby("head")

    rows = []
    for head, group in grouped:
        wga_mean = group["worst_group_acc"].mean()
        wga_std = group["worst_group_acc"].std()
        mga_mean = group["mean_group_acc"].mean()
        mga_std = group["mean_group_acc"].std()
        overall_mean = group["overall_acc"].mean()
        overall_std = group["overall_acc"].std()
        g0_mean = group["G0_acc"].mean()
        g1_mean = group["G1_acc"].mean()
        g2_mean = group["G2_acc"].mean()
        g3_mean = group["G3_acc"].mean()

        rows.append({
            "Head": head.upper(),
            "Tier": "T3",
            "WGA (Mean ± Std)": f"{wga_mean:.2f} ± {wga_std:.2f}%",
            "MGA (Mean ± Std)": f"{mga_mean:.2f} ± {mga_std:.2f}%",
            "Overall Acc": f"{overall_mean:.2f} ± {overall_std:.2f}%",
            "G0 (Land/Land)": f"{g0_mean:.1f}%",
            "G1 (Land/Water)": f"{g1_mean:.1f}%",
            "G2 (Water/Land)": f"{g2_mean:.1f}%",
            "G3 (Water/Water)": f"{g3_mean:.1f}%"
        })

    summary_df = pd.DataFrame(rows)
    return summary_df.to_markdown(index=False)


def generate_comparative_table(pilot_csv_path: str = "results/pilot_p1/pilot_p1_summary.csv") -> str:
    """Generates full comparative table incorporating measured results and published literature."""
    table_lines = [
        "| Method / Head | Architecture | Supervision Tier | Val Worst-Group Acc (WGA) | Reference / Source |",
        "| :--- | :--- | :--- | :--- | :--- |"
    ]

    # Published baselines
    for b in PUBLISHED_BASELINES:
        table_lines.append(f"| {b['method']} | ResNet-50 | {b['tier']} | {b['WGA']} | {b['source']} |")

    # Measured heads
    if os.path.exists(pilot_csv_path):
        df = pd.read_csv(pilot_csv_path)
        for head, group in df.groupby("head"):
            wga_mean = group["worst_group_acc"].mean()
            wga_std = group["worst_group_acc"].std()
            table_lines.append(f"| **{head.upper()} (Ours)** | ResNet-50 | T3 | **{wga_mean:.2f} ± {wga_std:.2f}%** | Measured (3 Seeds) |")

    return "\n".join(table_lines)


if __name__ == "__main__":
    print("=== PILOT P1 MEASURED RESULTS ===")
    print(summarize_pilot_results())
    print("\n=== COMPREHENSIVE COMPARISON WITH LITERATURE ===")
    print(generate_comparative_table())
