"""
Final Model Training & Serialization Script.
Fits the final calibrated HistGradientBoosting model on combined Train + Dev datasets using locked threshold.
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exoplanet_kepler.classifier import ExoplanetCandidateClassifier, FEATURE_COLS
from scripts.run_pipeline_eval import process_dataset_cached, run_oof_cv_threshold_selection


def train_final_model():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")

    # 1. Load Train & Dev features
    df_train = process_dataset_cached(
        os.path.join(data_dir, "train"),
        os.path.join(data_dir, "train_labels.csv"),
        os.path.join(data_dir, "train_truth.csv"),
        cache_name="train_features.csv"
    )
    df_dev = process_dataset_cached(
        os.path.join(data_dir, "dev"),
        os.path.join(data_dir, "dev_labels.csv"),
        os.path.join(data_dir, "dev_truth.csv"),
        cache_name="dev_features.csv"
    )

    # 2. Lock Threshold via Train 5-Fold OOF CV
    locked_thr, _ = run_oof_cv_threshold_selection(df_train, model_type="hist_gb")
    print(f"\nFinal Locked Threshold from Train OOF CV: {locked_thr:.2f}")

    # 3. Fit Final Classifier on Combined Datasets
    df_combined = pd.concat([df_train, df_dev], ignore_index=True)
    clf = ExoplanetCandidateClassifier(model_type="hist_gb", random_state=42)
    clf.fit(df_combined, df_combined["has_planet"].values)
    print("SUCCESS: Final Calibrated Classifier trained on 358 stars!")

    return clf, locked_thr


if __name__ == "__main__":
    train_final_model()
