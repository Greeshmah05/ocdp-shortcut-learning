"""
Waterbirds Dataset Loader
Implements dataset access for official Waterbirds benchmark splits.
Enforces Tier 3 training isolation: training loaders strictly return (image, label, idx).
Group labels (g = 2*y + place) are only exposed when return_group=True for evaluation.
"""

import io
import os
from typing import Optional, Callable, Tuple, Dict, Any
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import pyarrow.parquet as pq
import pandas as pd
import torchvision.transforms as transforms


def get_default_transform(split: str = "train", img_size: int = 224) -> Callable:
    """Standard ResNet-50 ImageNet transformations."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    if split == "train":
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    else:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize
        ])


def get_group_id(label: int, place: int) -> int:
    """
    Computes group index g = 2 * label + place.
    G0: Landbird on Land (y=0, a=0) - Majority
    G1: Landbird on Water (y=0, a=1) - Minority
    G2: Waterbird on Land (y=1, a=0) - Minority
    G3: Waterbird on Water (y=1, a=1) - Majority
    """
    return 2 * int(label) + int(place)


class WaterbirdsDataset(Dataset):
    """
    PyTorch Dataset for Waterbirds.
    Loads directly from parquet archives or cached tables.
    """
    def __init__(
        self,
        split: str = "train",
        data_dir: str = "data/raw",
        transform: Optional[Callable] = None,
        return_group: bool = False
    ):
        assert split in ["train", "validation", "val", "test"], f"Invalid split: {split}"
        actual_split = "validation" if split == "val" else split
        self.split = actual_split
        self.return_group = return_group
        self.transform = transform or get_default_transform(split=split)
        
        parquet_path = os.path.join(data_dir, f"{actual_split}.parquet")
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Parquet file not found at: {parquet_path}")
            
        self.table = pq.read_table(parquet_path)
        self.images = self.table.column("image")
        self.labels = self.table.column("label").to_pylist()
        self.places = self.table.column("place").to_pylist()
        self.groups = [get_group_id(y, a) for y, a in zip(self.labels, self.places)]
        self.length = len(self.labels)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[Any, ...]:
        img_struct = self.images[idx].as_py()
        raw_bytes = img_struct["bytes"]
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        
        if self.transform:
            img = self.transform(img)
            
        y = self.labels[idx]
        
        if self.return_group:
            # Evaluation only
            g = self.groups[idx]
            place = self.places[idx]
            return img, y, g, place, idx
            
        # Training (Tier 3: no group/place labels returned)
        return img, y, idx

    def get_group_counts(self) -> Dict[int, int]:
        """Returns group count mapping."""
        counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for g in self.groups:
            counts[g] = counts.get(g, 0) + 1
        return counts


def build_dataloaders(
    data_dir: str = "data/raw",
    batch_size: int = 32,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Builds standard DataLoaders for train, validation, and test.
    Train loader has return_group=False (Tier 3).
    Val and Test loaders have return_group=True (for evaluation metrics).
    """
    train_dataset = WaterbirdsDataset(split="train", data_dir=data_dir, return_group=False)
    val_dataset = WaterbirdsDataset(split="validation", data_dir=data_dir, return_group=True)
    test_dataset = WaterbirdsDataset(split="test", data_dir=data_dir, return_group=True)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    return train_loader, val_loader, test_loader
