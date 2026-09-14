"""
Pipeline Evaluation and Development Set Benchmark Runner (Parallelized).
"""

import glob
import os
import sys
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exoplanet_kepler.cleaning import clean_lightcurve
from exoplanet_kepler.detrending import detrend_lightcurve
from exoplanet_kepler.search import coarse_fine_bls_search
from exoplanet_kepler.features import extract_candidate_features
from exoplanet_kepler.classifier import ExoplanetCandidateClassifier, FEATURE_COLS
from exoplanet_kepler.metrics import evaluate_detection_metrics


def process_single_star(path, detrend_method="savgol", window_days=1.0):
    """
    Helper function to process a single star parquet file.
    """
    star_id = os.path.basename(path)[:-8]
    kepid = int(star_id.replace("KIC_", "")) if star_id.startswith("KIC_") else star_id

    try:
        df = pd.read_parquet(path)
        t_raw, f_raw, ferr_raw, q_raw = clean_lightcurve(df)
        if t_raw is None or len(t_raw) < 500:
            feats = {col: 0.0 for col in FEATURE_COLS}
        else:
            t, f, trend = detrend_lightcurve(t_raw, f_raw, method=detrend_method, window_days=window_days)
            candidate = coarse_fine_bls_search(t, f)
            feats = extract_candidate_features(t, f, q_raw[:len(t)], candidate)
    except Exception as e:
        feats = {col: 0.0 for col in FEATURE_COLS}

    return {"star_id": star_id, "kepid": kepid, **feats}


def process_dataset(data_dir, labels_csv, truth_csv, detrend_method="savgol", window_days=1.0, n_jobs=-1):
    """
    Extracts BLS transit search features for all stars using multi-core parallel processing.
    """
    files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    labels = pd.read_csv(labels_csv)
    truth = pd.read_csv(truth_csv) if os.path.exists(truth_csv) else None

    n_files = len(files)
    t_start = time.time()
    print(f"Processing {n_files} stars in {data_dir} using {os.cpu_count()} CPU cores ('{detrend_method}' detrending)...", flush=True)

    records = Parallel(n_jobs=n_jobs, batch_size=4)(
        delayed(process_single_star)(path, detrend_method, window_days) for path in files
    )

    elapsed = time.time() - t_start
    rate = n_files / elapsed if elapsed > 0 else 0
    print(f"Completed {n_files} stars in {elapsed:.1f}s ({rate:.1f} stars/sec).", flush=True)

    df_feats = pd.DataFrame(records)

    # Merge labels & truth ground truth
    df_feats = df_feats.merge(labels[["kepid", "label"]], on="kepid", how="left")
    if truth is not None:
        df_feats = df_feats.merge(
            truth[["kepid", "injected", "period_days", "depth_ppm"]],
            on="kepid", how="left", suffixes=("", "_true")
        )
        df_feats["injected"] = df_feats["injected"].fillna(0)
        df_feats["has_planet"] = ((df_feats["label"] == 1) | (df_feats["injected"] == 1)).astype(int)
    else:
        df_feats["has_planet"] = df_feats["label"].fillna(0).astype(int)

    return df_feats


def run_pipeline_benchmark():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")

    # 1. Process Train Set
    df_train = process_dataset(
        os.path.join(data_dir, "train"),
        os.path.join(data_dir, "train_labels.csv"),
        os.path.join(data_dir, "train_truth.csv"),
        detrend_method="savgol", window_days=1.0
    )

    # 2. Train ML Classifier
    clf = ExoplanetCandidateClassifier(model_type="hist_gb")
    clf.fit(df_train, df_train["has_planet"].values)
    print("\n[ML Model] HistGradientBoosting candidate classifier successfully trained!")

    # 3. Process Development Set
    df_dev = process_dataset(
        os.path.join(data_dir, "dev"),
        os.path.join(data_dir, "dev_labels.csv"),
        os.path.join(data_dir, "dev_truth.csv"),
        detrend_method="savgol", window_days=1.0
    )

    # 4. Predict Confidence Probabilities
    dev_probas = clf.predict_proba(df_dev)
    df_dev["confidence"] = np.round(dev_probas, 4)

    # Pick optimal detection threshold based on dev F1 score
    best_thr, best_f1 = 0.5, 0.0
    for thr in np.linspace(0.1, 0.9, 81):
        preds = (df_dev["confidence"] >= thr).astype(int)
        df_dev_temp = df_dev.copy()
        df_dev_temp["prediction"] = preds
        m = evaluate_detection_metrics(df_dev_temp)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_thr = thr

    df_dev["prediction"] = (df_dev["confidence"] >= best_thr).astype(int)
    metrics = evaluate_detection_metrics(df_dev)

    # Print Final Evaluation Results
    print("\n=======================================================")
    print("      DEVELOPMENT SET BENCHMARK RESULTS (89 STARS)     ")
    print("=======================================================")
    print(f"  Optimal Confidence Threshold: {best_thr:.2f}")
    print(f"  Precision:         {metrics['precision']:.4f}")
    print(f"  Recall:            {metrics['recall']:.4f}")
    print(f"  F1 Score:          {metrics['f1']:.4f}")
    print(f"  PR-AUC:            {metrics['pr_auc']:.4f}")
    print(f"  Average Precision: {metrics['avg_precision']:.4f}")
    print(f"  Period Accuracy:   {metrics['period_accuracy']:.2%}")
    print(f"  Mean Depth Error:  {metrics['mean_depth_error']:.2%}")

    print("\n--- Recall Breakdown by Transit Depth Bracket ---")
    for b_name, (rec, count) in metrics["depth_brackets"].items():
        print(f"  {b_name:25s}: {rec:6.1%}  (Count: {count})")

    # Save Dev Set Predictions CSV
    dev_out_path = os.path.join(base_dir, "dev_benchmark_predictions.csv")
    df_dev.to_csv(dev_out_path, index=False)
    print(f"\nSaved Development Set Predictions to: {dev_out_path}")


if __name__ == "__main__":
    run_pipeline_benchmark()
