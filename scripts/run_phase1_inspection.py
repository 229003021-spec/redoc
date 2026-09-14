"""
Phase 1 Data Inspection Script for Kepler Photometry Datasets.
"""

import glob
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exoplanet_kepler.cleaning import clean_lightcurve


def inspect_dataset(data_dir, labels_csv, truth_csv, dataset_name="TRAIN"):
    print(f"\n==========================================")
    print(f"       PHASE 1 INSPECTION: {dataset_name} SET")
    print(f"==========================================")

    labels = pd.read_csv(labels_csv)
    truth = pd.read_csv(truth_csv)

    print(f"Total Star Parquet Files: {len(glob.glob(os.path.join(data_dir, '*.parquet')))}")
    print(f"Labels CSV Shape: {labels.shape}")
    print(f"Labels Columns: {list(labels.columns)}")
    print(f"Planet Class Distribution:\n{labels['label'].value_counts().to_string()}")

    print(f"\nTruth CSV Shape: {truth.shape}")
    print(f"Truth Columns: {list(truth.columns)}")
    print(f"Injected Planet Signals: {truth['injected'].sum()} / {len(truth)}")

    # Inspect first parquet file
    files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    sample_file = files[0]
    star_id = os.path.basename(sample_file)[:-8]

    df = pd.read_parquet(sample_file)
    print(f"\nSample Star: {star_id}")
    print(f"  Raw Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Time Range: {df.time.min():.2f} to {df.time.max():.2f} BKJD ({df.time.max() - df.time.min():.1f} days)")
    print(f"  Quarters Present: {sorted(df.quarter.unique())}")
    print(f"  Quality Flag Zero Cadences: {(df.quality == 0).mean():.2%}")
    print(f"  Flux Stats (SAP/PDC): median={df.flux.median():.2f}, std={df.flux.std():.2f}, min={df.flux.min():.2f}, max={df.flux.max():.2f}")

    t, f, ferr, q = clean_lightcurve(df)
    print(f"\nCleaned & Normalized Lightcurve:")
    print(f"  Valid Cadences Retained: {len(t)} / {len(df)} ({len(t)/len(df):.2%})")
    print(f"  Normalized Flux Scatter (STD): {np.std(f) * 1e6:.1f} ppm")


if __name__ == "__main__":
    base_data = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    inspect_dataset(
        os.path.join(base_data, "train"),
        os.path.join(base_data, "train_labels.csv"),
        os.path.join(base_data, "train_truth.csv"),
        "TRAIN"
    )
    inspect_dataset(
        os.path.join(base_data, "dev"),
        os.path.join(base_data, "dev_labels.csv"),
        os.path.join(base_data, "dev_truth.csv"),
        "DEVELOPMENT"
    )
