"""
Interactive Sample Inference and Explainability Tool for OCDP.
Loads a trained OCDP model, runs inference on a chosen image sample,
computes the Object Gate spatial heatmap and Context Sink shortcut scores,
and saves a comprehensive visual explanation plot.
"""

import os
import io
import argparse
import tempfile
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.data.waterbirds import WaterbirdsDataset, get_default_transform
from src.models.backbone import ResNet50FeatureExtractor
from src.models.heads import OCDPHead
from src.losses.objectives import (
    GeneralizedCrossEntropyLoss,
    AreaPriorLoss,
    DynamicSampleReweighting
)


GROUP_NAMES = {
    0: "Landbird on Land (Majority G0)",
    1: "Landbird on Water (Minority G1)",
    2: "Waterbird on Land (Minority G2)",
    3: "Waterbird on Water (Majority G3)"
}

CLASS_NAMES = {0: "Landbird", 1: "Waterbird"}
PLACE_NAMES = {0: "Land", 1: "Water"}


def train_quick_ocdp_head(
    cache_path: str = "checkpoints/cache/train_feats.pt",
    save_path: str = "checkpoints/best_ocdp_head.pt",
    device: torch.device = torch.device("cpu")
) -> OCDPHead:
    """Trains OCDP on cached features if checkpoint does not already exist."""
    print("[*] No existing checkpoint found. Training OCDP head on cached features (~15 seconds)...")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cached features not found at {cache_path}. Run Pilot P1 first.")

    train_data = torch.load(cache_path, weights_only=True)
    train_feats, train_labels, _ = train_data

    head = OCDPHead(in_features=2048, hidden_dim=256, num_classes=2).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)

    gce_fn = GeneralizedCrossEntropyLoss(q=0.7, reduction="none")
    area_fn = AreaPriorLoss(rho=0.3)
    reweighter = DynamicSampleReweighting(w_min=0.2, w_max=4.0)
    ce_loss_none = torch.nn.CrossEntropyLoss(reduction="none")

    train_ds = torch.utils.data.TensorDataset(train_feats, train_labels)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)

    head.train()
    epochs = 15
    for ep in range(1, epochs + 1):
        for feats_b, ys_b in loader:
            feats_b = feats_b.to(device)
            ys_b = ys_b.to(device)
            optimizer.zero_grad()

            y_obj, y_ctx, gate = head(feats_b, return_context=True)
            ce_obj = ce_loss_none(y_obj, ys_b)
            ce_ctx = ce_loss_none(y_ctx, ys_b)

            if ep <= 2:
                loss_obj = ce_obj.mean()
            else:
                w = reweighter(ce_obj.detach(), ce_ctx.detach())
                loss_obj = (w * ce_obj).mean()

            loss_ctx = gce_fn(y_ctx, ys_b).mean()
            loss_area = area_fn(gate)
            loss = loss_obj + 1.0 * loss_ctx + 0.5 * loss_area
            loss.backward()
            optimizer.step()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(head.state_dict(), save_path)
    print(f"[OK] Trained OCDP head and saved checkpoint to: {save_path}")
    return head


def load_ocdp_model(
    ckpt_path: str = "checkpoints/best_ocdp_head.pt",
    device: torch.device = torch.device("cpu")
) -> OCDPHead:
    """Loads OCDP head weights, training on cached features if necessary."""
    head = OCDPHead(in_features=2048, hidden_dim=256, num_classes=2).to(device)
    if os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
        head.load_state_dict(state_dict)
    else:
        head = train_quick_ocdp_head(save_path=ckpt_path, device=device)
    head.eval()
    return head


def run_sample_inference(
    split: str = "validation",
    idx: int = 0,
    target_group: int = None,
    data_dir: str = "data/raw",
    ckpt_path: str = "checkpoints/best_ocdp_head.pt",
    output_plot: str = "results/sample_inference.png"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing evaluation on device: {device}...")

    # Load dataset
    dataset = WaterbirdsDataset(split=split, data_dir=data_dir, return_group=True)

    # Select sample index
    if target_group is not None:
        matching_indices = [i for i, g in enumerate(dataset.groups) if g == target_group]
        if not matching_indices:
            raise ValueError(f"No samples found for group {target_group} in {split} split.")
        sample_idx = matching_indices[min(idx, len(matching_indices) - 1)]
    else:
        sample_idx = min(idx, len(dataset) - 1)

    # Load raw PIL image and metadata
    raw_bytes = dataset.images[sample_idx].as_py()["bytes"]
    pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    true_label = dataset.labels[sample_idx]
    true_place = dataset.places[sample_idx]
    group_id = dataset.groups[sample_idx]

    # Preprocess image for ResNet-50
    transform = get_default_transform(split=split, img_size=224)
    img_tensor = transform(pil_img).unsqueeze(0).to(device)  # [1, 3, 224, 224]

    # Extract backbone features
    backbone = ResNet50FeatureExtractor(pretrained=True).to(device).eval()
    with torch.no_grad():
        feat_map = backbone(img_tensor)  # [1, 2048, 7, 7]

    # Load and run OCDP head
    ocdp = load_ocdp_model(ckpt_path=ckpt_path, device=device)
    with torch.no_grad():
        y_obj, y_ctx, gate = ocdp(feat_map, return_context=True)
        prob_obj = F.softmax(y_obj, dim=-1).squeeze(0).cpu().numpy()
        prob_ctx = F.softmax(y_ctx, dim=-1).squeeze(0).cpu().numpy()
        gate_map = gate.squeeze().cpu().numpy()  # [7, 7]

    # Upsample gate to 224x224
    gate_t = torch.tensor(gate_map).unsqueeze(0).unsqueeze(0)  # [1, 1, 7, 7]
    gate_upsampled = F.interpolate(gate_t, size=(224, 224), mode="bilinear", align_corners=False)
    gate_2d = gate_upsampled.squeeze().numpy()

    # Predictions
    pred_obj = int(np.argmax(prob_obj))
    pred_ctx = int(np.argmax(prob_ctx))
    shortcut_confidence = prob_ctx[true_label]

    # Print summary
    print("\n" + "=" * 50)
    print(f"SAMPLE INFERENCE REPORT (Index: {sample_idx}, Split: {split})")
    print("=" * 50)
    print(f"True Class        : {CLASS_NAMES[true_label]} (y = {true_label})")
    print(f"True Background   : {PLACE_NAMES[true_place]} (place = {true_place})")
    print(f"Group Category    : {GROUP_NAMES[group_id]}")
    print("-" * 50)
    print(f"OCDP Object Pred  : {CLASS_NAMES[pred_obj]} (Confidence: {prob_obj[pred_obj]*100:.2f}%)")
    print(f"Context Sink Pred : {CLASS_NAMES[pred_ctx]} (Confidence: {prob_ctx[pred_ctx]*100:.2f}%)")
    print(f"Shortcut Score s_i: {shortcut_confidence:.4f}")
    print(f"Object Gate Mean  : {gate_map.mean():.4f} (Area coverage)")
    correct_str = "[CORRECT]" if pred_obj == true_label else "[FAILED]"
    print(f"Verdict           : {correct_str}")
    print("=" * 50)

    # Generate multi-panel figure
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), dpi=200)

    # Panel 1: Original Image
    resized_pil = pil_img.resize((224, 224))
    axes[0].imshow(resized_pil)
    axes[0].set_title(f"Input: {CLASS_NAMES[true_label]}\n(Group {group_id}: {PLACE_NAMES[true_place]})", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: OCDP Object Gate Heatmap
    axes[1].imshow(resized_pil)
    im = axes[1].imshow(gate_2d, cmap="jet", alpha=0.55, vmin=0.0, vmax=1.0)
    axes[1].set_title(f"OCDP Object Gate Map\nMean Coverage: {gate_map.mean()*100:.1f}%", fontsize=11, fontweight="bold")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: Foreground Isolated
    img_arr = np.array(resized_pil).astype(float) / 255.0
    masked_img = img_arr * np.expand_dims(gate_2d, axis=-1)
    axes[2].imshow(np.clip(masked_img, 0, 1))
    axes[2].set_title("Gate-Isolated Object\n(Background Suppressed)", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    # Panel 4: Probability Comparison
    classes = ["Landbird", "Waterbird"]
    y_pos = np.arange(len(classes))
    bar_width = 0.35

    axes[3].barh(y_pos - bar_width/2, prob_obj, bar_width, label="Object Head (y_obj)", color="#2ECC71", edgecolor="black")
    axes[3].barh(y_pos + bar_width/2, prob_ctx, bar_width, label="Context Sink (y_ctx)", color="#E74C3C", edgecolor="black")
    axes[3].set_yticks(y_pos)
    axes[3].set_yticklabels(classes, fontsize=10, fontweight="bold")
    axes[3].set_xlim(0, 1.05)
    axes[3].set_xlabel("Predicted Probability", fontsize=10, fontweight="bold")
    axes[3].set_title(f"Decoupled Predictions\nVerdict: {correct_str}", fontsize=11, fontweight="bold")
    axes[3].legend(loc="lower right", fontsize=8.5)
    axes[3].grid(axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_plot), exist_ok=True)
    plt.savefig(output_plot)
    plt.close()
    print(f"\n[OK] Visual explanation plot saved to: {output_plot}")
    return output_plot


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test OCDP on individual dataset samples")
    parser.add_argument("--idx", type=int, default=0, help="Sample index in split")
    parser.add_argument("--group", type=int, default=None, choices=[0, 1, 2, 3], help="Filter by group: 0=Land/Land, 1=Land/Water, 2=Water/Land, 3=Water/Water")
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test"], help="Dataset split")
    parser.add_argument("--output", type=str, default="results/sample_inference.png", help="Path to save explanation figure")
    args = parser.parse_args()

    run_sample_inference(
        split=args.split,
        idx=args.idx,
        target_group=args.group,
        output_plot=args.output
    )

