"""
Unit tests for Waterbirds Dataset Loader and Tier 3 Supervision Guard.
"""

import pytest
import torch
from src.data.waterbirds import WaterbirdsDataset, get_group_id


def test_group_mapping_formula():
    """Verify group ID definition: g = 2*y + place."""
    assert get_group_id(0, 0) == 0  # Landbird on Land
    assert get_group_id(0, 1) == 1  # Landbird on Water
    assert get_group_id(1, 0) == 2  # Waterbird on Land
    assert get_group_id(1, 1) == 3  # Waterbird on Water


def test_split_counts_and_groups():
    """Verify exact count parity with official Group DRO benchmarks."""
    train_ds = WaterbirdsDataset(split="train", data_dir="data/raw", return_group=True)
    val_ds = WaterbirdsDataset(split="validation", data_dir="data/raw", return_group=True)
    test_ds = WaterbirdsDataset(split="test", data_dir="data/raw", return_group=True)

    assert len(train_ds) == 4795
    assert len(val_ds) == 1199
    assert len(test_ds) == 5794

    train_counts = train_ds.get_group_counts()
    assert train_counts[0] == 3498
    assert train_counts[1] == 184
    assert train_counts[2] == 56
    assert train_counts[3] == 1057

    val_counts = val_ds.get_group_counts()
    assert val_counts[0] == 467
    assert val_counts[1] == 466
    assert val_counts[2] == 133
    assert val_counts[3] == 133

    test_counts = test_ds.get_group_counts()
    assert test_counts[0] == 2255
    assert test_counts[1] == 2255
    assert test_counts[2] == 642
    assert test_counts[3] == 642


def test_tier3_training_guard():
    """Verify that training dataset strictly hides group and place labels."""
    train_ds = WaterbirdsDataset(split="train", data_dir="data/raw", return_group=False)
    item = train_ds[0]
    # Must only return: img, label, index
    assert len(item) == 3
    img, y, idx = item
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 224, 224)
    assert isinstance(y, int)
    assert y in (0, 1)
    assert isinstance(idx, int)


def test_eval_dataset_returns_groups():
    """Verify evaluation dataset exposes group labels for metric computation."""
    val_ds = WaterbirdsDataset(split="validation", data_dir="data/raw", return_group=True)
    item = val_ds[0]
    assert len(item) == 5
    img, y, g, place, idx = item
    assert g == 2 * y + place
    assert g in (0, 1, 2, 3)
