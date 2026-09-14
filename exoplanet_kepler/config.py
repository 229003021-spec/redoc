"""
Configuration parameters for the Kepler Exoplanet Detection Pipeline.
"""

import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PRIVATE_DIR = os.path.join(DATA_DIR, "private")

# Zip Archive Paths (relative to BASE_DIR or parent directory if present)
TRAIN_ZIP_PATH = os.path.join(BASE_DIR, "train_pack.zip") if os.path.exists(os.path.join(BASE_DIR, "train_pack.zip")) else os.path.join(os.path.dirname(BASE_DIR), "train_pack.zip")
DEV_ZIP_PATH = os.path.join(BASE_DIR, "dev_pack.zip") if os.path.exists(os.path.join(BASE_DIR, "dev_pack.zip")) else os.path.join(os.path.dirname(BASE_DIR), "dev_pack.zip")
PRIVATE_ZIP_PATH = os.path.join(BASE_DIR, "private_pack.zip") if os.path.exists(os.path.join(BASE_DIR, "private_pack.zip")) else os.path.join(os.path.dirname(BASE_DIR), "private_pack.zip")

# Preprocessing & Quality Filtering
QUALITY_BITMASK_DROP = 0  # Default: drop any non-zero quality cadence
MIN_CADENCES = 1000       # Minimum valid cadences required to process a star
SIGMA_CLIP_OUTLIERS = 6.0 # Extreme positive outlier clip (cosmic rays)

# Detrending Settings
DETREND_WINDOW_DAYS = 1.0 # Baseline running median window length in days
SAVGOL_WINDOW_DAYS = 1.0  # Savitzky-Golay window length in days
SAVGOL_POLYORDER = 2      # Savitzky-Golay polynomial order

# Coarse-to-Fine Box Least Squares (BLS) Search Grid Settings
PERIOD_MIN = 3.0
PERIOD_MAX = 400.0
N_COARSE = 10000         # Number of coarse trial periods (log-spaced)
N_PEAKS = 8               # Top coarse periodogram peaks to refine
N_FINE = 600              # Refinement trial periods around each coarse peak
FINE_WINDOW_FRAC = 0.02   # +/- 2% period refinement window
DURATIONS_DAYS = [0.05, 0.10, 0.20, 0.40, 0.80] # Candidate transit durations

# Candidate Vetting & Scoring Thresholds
SDE_BASELINE_THRESHOLD = 10.0
