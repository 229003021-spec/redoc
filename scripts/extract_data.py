"""
Script to extract train_pack.zip and dev_pack.zip into local data directory.
"""

import os
import zipfile

TRAIN_ZIP = r"C:\Users\Arvind\OneDrive\Documents\train_pack.zip"
DEV_ZIP = r"C:\Users\Arvind\OneDrive\Documents\dev_pack.zip"
DATA_DIR = r"c:\Users\Arvind\OneDrive\Documents\exoplanet_kepler_pipeline\data"


def extract_zips():
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(TRAIN_ZIP):
        print(f"Extracting {TRAIN_ZIP} to {DATA_DIR}...")
        with zipfile.ZipFile(TRAIN_ZIP, 'r') as z:
            z.extractall(DATA_DIR)

    if os.path.exists(DEV_ZIP):
        print(f"Extracting {DEV_ZIP} to {DATA_DIR}...")
        with zipfile.ZipFile(DEV_ZIP, 'r') as z:
            z.extractall(DATA_DIR)

    print("Data extraction complete!")
    print("Data directory contents:", os.listdir(DATA_DIR))


if __name__ == "__main__":
    extract_zips()
