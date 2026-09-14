"""
Script to extract train_pack.zip and dev_pack.zip into local data directory.
"""

import os
import zipfile

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PARENT_DIR = os.path.dirname(BASE_DIR)

TRAIN_ZIP = os.path.join(PARENT_DIR, "train_pack.zip") if os.path.exists(os.path.join(PARENT_DIR, "train_pack.zip")) else os.path.join(BASE_DIR, "train_pack.zip")
DEV_ZIP = os.path.join(PARENT_DIR, "dev_pack.zip") if os.path.exists(os.path.join(PARENT_DIR, "dev_pack.zip")) else os.path.join(BASE_DIR, "dev_pack.zip")
PRIVATE_ZIP = os.path.join(PARENT_DIR, "private_pack.zip") if os.path.exists(os.path.join(PARENT_DIR, "private_pack.zip")) else os.path.join(BASE_DIR, "private_pack.zip")


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
