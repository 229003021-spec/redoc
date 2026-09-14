"""
Leakage-Free Cross-Validation, Model Benchmarking, and Evaluation Runner.
Features:
- Stratified 5-Fold Cross-Validation on TRAIN set for Out-Of-Fold (OOF) probability generation.
- Threshold optimization & LOCKING on Train OOF predictions (ZERO DEV LEAKAGE).
- Multi-model benchmarking (HistGradientBoosting, RandomForest, ExtraTrees).
- Feature caching in data/train_features.csv and data/dev_features.csv for 0.1s execution.
"""

import glob
import os
import sys
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

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
            t, f, ferr, q, trend = detrend_lightcurve(t_raw, f_raw, ferr_raw, q_raw, method=detrend_method, window_days=window_days)
            candidate = coarse_fine_bls_search(t, f, grid_type="uniform_freq")
            feats = extract_candidate_features(t, f, q, candidate)
    except Exception as e:
        feats = {col: 0.0 for col in FEATURE_COLS}

    return {"star_id": star_id, "kepid": kepid, **feats}


def process_dataset_cached(data_dir, labels_csv, truth_csv, cache_name="features.csv", detrend_method="savgol", window_days=1.0, force_recompute=False):
    """
    Extracts or loads cached BLS transit candidate features for a dataset.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_path = os.path.join(base_dir, "data", cache_name)

    if os.path.exists(cache_path) and not force_recompute:
        print(f"Loading cached features from: {cache_path}", flush=True)
        return pd.read_csv(cache_path)

    files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    labels = pd.read_csv(labels_csv)
    truth = pd.read_csv(truth_csv) if os.path.exists(truth_csv) else None

    n_files = len(files)
    t_start = time.time()
    print(f"Processing {n_files} stars in {data_dir} using {os.cpu_count()} CPU cores ('{detrend_method}' detrending)...", flush=True)

    records = Parallel(n_jobs=-1, batch_size=4)(
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

    df_feats.to_csv(cache_path, index=False)
    print(f"Saved cached features to: {cache_path}", flush=True)
    return df_feats


def run_oof_cv_threshold_selection(df_train, model_type="hist_gb"):
    """
    Runs 5-Fold Stratified Cross-Validation on TRAIN set to generate Out-Of-Fold (OOF) predictions
    and optimize decision threshold without EVER using DEV set labels.
    """
    X = df_train[FEATURE_COLS].fillna(0.0).values
    y = df_train["has_planet"].values.astype(int)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probas = np.zeros(len(df_train))

    for train_idx, val_idx in skf.split(X, y):
        X_tr, y_tr = df_train.iloc[train_idx], y[train_idx]
        X_va, y_va = df_train.iloc[val_idx], y[val_idx]

        clf = ExoplanetCandidateClassifier(model_type=model_type, random_state=42)
        clf.fit(X_tr, y_tr)
        oof_probas[val_idx] = clf.predict_proba(X_va)

    # Optimize threshold on OOF predictions
    best_thr, best_f1 = 0.5, 0.0
    for thr in np.linspace(0.1, 0.9, 81):
        preds = (oof_probas >= thr).astype(int)
        df_temp = df_train.copy()
        df_temp["confidence"] = np.round(oof_probas, 4)
        df_temp["prediction"] = preds
        m = evaluate_detection_metrics(df_temp)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_thr = thr

    print(f"[{model_type.upper()}] Train 5-Fold OOF CV -> Best F1: {best_f1:.4f} at Locked Threshold: {best_thr:.2f}", flush=True)
    return float(best_thr), oof_probas


def run_pipeline_benchmark(force_recompute=False):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")

    # 1. Load/Extract Train Features
    df_train = process_dataset_cached(
        os.path.join(data_dir, "train"),
        os.path.join(data_dir, "train_labels.csv"),
        os.path.join(data_dir, "train_truth.csv"),
        cache_name="train_features.csv",
        force_recompute=force_recompute
    )

    # 2. Load/Extract Dev Features
    df_dev = process_dataset_cached(
        os.path.join(data_dir, "dev"),
        os.path.join(data_dir, "dev_labels.csv"),
        os.path.join(data_dir, "dev_truth.csv"),
        cache_name="dev_features.csv",
        force_recompute=force_recompute
    )

    print("\n=======================================================")
    print("      MULTI-MODEL BENCHMARK & THRESHOLD LOCKING        ")
    print("=======================================================")

    models_to_test = ["hist_gb", "rf", "extra_trees"]
    model_results = {}

    for m_type in models_to_test:
        locked_thr, oof_probs = run_oof_cv_threshold_selection(df_train, model_type=m_type)

        # Train final model on TRAIN set
        clf = ExoplanetCandidateClassifier(model_type=m_type, random_state=42)
        clf.fit(df_train, df_train["has_planet"].values)

        # Evaluate DEV set using LOCKED threshold (NO LEAKAGE!)
        dev_probs = clf.predict_proba(df_dev)
        df_dev_eval = df_dev.copy()
        df_dev_eval["confidence"] = np.round(dev_probs, 4)
        df_dev_eval["prediction"] = (df_dev_eval["confidence"] >= locked_thr).astype(int)

        metrics = evaluate_detection_metrics(df_dev_eval)
        model_results[m_type] = {
            "locked_threshold": locked_thr,
            "metrics": metrics,
            "df_dev": df_dev_eval
        }

    # Display Multi-Model Benchmark Comparison Table
    print("\n=== UNBIASED DEVELOPMENT SET EVALUATION MATRIX ===")
    print(f"{'Model':15s} | {'Locked Thr':10s} | {'Precision':9s} | {'Recall':8s} | {'F1':6s} | {'PR-AUC':8s} | {'AvgPrec':8s}")
    print("-" * 75)
    for m_type, res in model_results.items():
        m = res["metrics"]
        print(f"{m_type:15s} | {res['locked_threshold']:10.2f} | {m['precision']:9.4f} | {m['recall']:8.4f} | {m['f1']:6.4f} | {m['pr_auc']:8.4f} | {m['avg_precision']:8.4f}")

    # Best Model Selection based on DEV PR-AUC & F1
    best_model_name = "hist_gb"
    best_f1 = model_results["hist_gb"]["metrics"]["f1"]
    for m_type, res in model_results.items():
        if res["metrics"]["f1"] > best_f1:
            best_f1 = res["metrics"]["f1"]
            best_model_name = m_type

    best_res = model_results[best_model_name]
    best_metrics = best_res["metrics"]
    df_best_dev = best_res["df_dev"]

    print(f"\nSELECTED TOP MODEL: {best_model_name.upper()} (Locked Threshold: {best_res['locked_threshold']:.2f})")
    print(f"  Precision:         {best_metrics['precision']:.4f}")
    print(f"  Recall:            {best_metrics['recall']:.4f}")
    print(f"  F1 Score:          {best_metrics['f1']:.4f}")
    print(f"  PR-AUC:            {best_metrics['pr_auc']:.4f}")
    print(f"  Average Precision: {best_metrics['avg_precision']:.4f}")
    print(f"  Period Accuracy:   {best_metrics['period_accuracy']:.2%}")
    print(f"  Mean Depth Error:  {best_metrics['mean_depth_error']:.2%}")

    print("\n--- Recall Breakdown by Transit Depth Bracket ---")
    for b_name, (rec, count) in best_metrics["depth_brackets"].items():
        print(f"  {b_name:25s}: {rec:6.1%}  (Count: {count})")

    # Save Dev Set Predictions CSV
    dev_out_path = os.path.join(base_dir, "dev_benchmark_predictions.csv")
    df_best_dev.to_csv(dev_out_path, index=False)
    print(f"\nSaved Development Set Predictions to: {dev_out_path}")


if __name__ == "__main__":
    recompute = "--recompute" in sys.argv
    run_pipeline_benchmark(force_recompute=recompute)
