"""
Robust downloader and validator for the Waterbirds benchmark parquet dataset.
Handles partial/corrupted downloads safely via temporary files (.tmp) and verifies PyArrow readability.
"""

import os
import requests
import pyarrow.parquet as pq
import pandas as pd
from tqdm import tqdm

DATA_DIR = os.path.join("data", "raw")
SPLITS = ["train", "validation", "test"]
EXPECTED_COUNTS = {
    "train": 4795,
    "validation": 1199,
    "test": 5794
}

def verify_file(filepath: str, split: str) -> bool:
    """Check if the parquet file exists and is valid/untruncated."""
    if not os.path.exists(filepath):
        return False
    try:
        table = pq.read_table(filepath)
        count = len(table)
        expected = EXPECTED_COUNTS.get(split)
        if expected is not None and count != expected:
            print(f"Warning: {filepath} has {count} rows (expected {expected}).")
            return False
        return True
    except Exception:
        return False

def download_file(url: str, dest: str):
    """Download a file with progress bar and atomic rename via .tmp."""
    tmp_dest = f"{dest}.tmp"
    if os.path.exists(tmp_dest):
        os.remove(tmp_dest)
        
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    chunk_size = 1024 * 1024  # 1 MB chunks

    with open(tmp_dest, "wb") as f, tqdm(
        desc=os.path.basename(dest),
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

    os.replace(tmp_dest, dest)

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    for split in SPLITS:
        dest = os.path.join(DATA_DIR, f"{split}.parquet")
        url = f"https://huggingface.co/datasets/grodino/waterbirds/resolve/main/data/{split}-00000-of-00001.parquet"
        
        if verify_file(dest, split):
            print(f"[OK] {dest} is valid and complete.")
        else:
            if os.path.exists(dest):
                print(f"[!] {dest} is incomplete or corrupt ({os.path.getsize(dest)} bytes). Re-downloading...")
                os.remove(dest)
            else:
                print(f"[*] Downloading {split} split...")
            download_file(url, dest)
            print(f"[OK] {split} downloaded successfully ({os.path.getsize(dest):,} bytes).")

    # Generate metadata.csv
    print("[*] Generating data/metadata.csv...")
    dfs = []
    for split_idx, split in enumerate(SPLITS):
        p = os.path.join(DATA_DIR, f"{split}.parquet")
        table = pq.read_table(p)
        df = table.select(["label", "place"]).to_pandas()
        df["split"] = split_idx  # 0: train, 1: val, 2: test
        df["group"] = 2 * df["label"] + df["place"]
        df["img_id"] = range(len(df))
        dfs.append(df)

    metadata_path = os.path.join("data", "metadata.csv")
    full_meta = pd.concat(dfs, ignore_index=True)
    full_meta.to_csv(metadata_path, index=False)
    print(f"[OK] Saved {metadata_path} with {len(full_meta):,} rows.")
    print("\nDataset verification complete. You can now run: python -m pytest tests/")

if __name__ == "__main__":
    main()

