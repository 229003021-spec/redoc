"""
Fast Direct Inference Script on Private Dataset (STAR_0001 to STAR_0087).
Uses locked decision threshold from Train Out-Of-Fold CV (zero leakage),
processes all 87 private set stars in parallel using 8 CPU cores.
Outputs submission_redoc.csv matching all competition assertions.
"""

import glob
import os
import sys
import time
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exoplanet_kepler.cleaning import clean_lightcurve
from exoplanet_kepler.detrending import detrend_lightcurve
from exoplanet_kepler.search import coarse_fine_bls_search
from exoplanet_kepler.features import extract_candidate_features
from exoplanet_kepler.classifier import ExoplanetCandidateClassifier, FEATURE_COLS
from scripts.run_pipeline_eval import process_dataset_cached, run_oof_cv_threshold_selection


def process_private_star_file(path):
    """
    Process a single private set star parquet file.
    Extracts star_id from filename without extension (e.g. STAR_0001).
    """
    star_id = os.path.splitext(os.path.basename(path))[0]
    try:
        df = pd.read_parquet(path)
        t_raw, f_raw, ferr_raw, q_raw = clean_lightcurve(df)
        if t_raw is None or len(t_raw) < 500:
            candidate = {}
            feats = {col: 0.0 for col in FEATURE_COLS}
        else:
            t, f, ferr, q, trend = detrend_lightcurve(t_raw, f_raw, ferr_raw, q_raw, method="savgol", window_days=1.0)
            candidate = coarse_fine_bls_search(t, f, grid_type="uniform_freq")
            feats = extract_candidate_features(t, f, q, candidate)
    except Exception as e:
        candidate = {}
        feats = {col: 0.0 for col in FEATURE_COLS}

    return {"star_id": star_id, "candidate": candidate, "feats": feats}


def run_private_inference(private_dir=None, output_csv="submission_redoc.csv"):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")

    if private_dir is None:
        private_dir = os.path.join(data_dir, "private")

    if not os.path.exists(private_dir):
        raise FileNotFoundError(f"Mandatory Private dataset directory missing: {private_dir}")

    # 1. Fit classifier on combined Train + Dev features with locked threshold
    print("Loading pre-processed candidate training features...")
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

    locked_thr, _ = run_oof_cv_threshold_selection(df_train, model_type="hist_gb")
    print(f"Locked Decision Threshold from Train OOF CV: {locked_thr:.2f}")

    df_all = pd.concat([df_train, df_dev], ignore_index=True)
    clf = ExoplanetCandidateClassifier(model_type="hist_gb", random_state=42)
    clf.fit(df_all, df_all["has_planet"].values)
    print("ML Candidate Classifier trained and ready for private set inference!")

    # 2. Process private set stars in parallel across 8 CPU cores
    paths = sorted(glob.glob(os.path.join(private_dir, "*.parquet")))
    if len(paths) == 0:
        raise FileNotFoundError(f"No parquet files found in private directory: {private_dir}")

    print(f"\nProcessing {len(paths)} private evaluation set stars in {private_dir}...")

    t0 = time.time()
    results = Parallel(n_jobs=-1, batch_size=4)(
        delayed(process_private_star_file)(p) for p in paths
    )
    print(f"Completed private set analysis in {time.time() - t0:.1f}s.")

    # 3. Construct submission rows
    out_rows = []
    for res in results:
        sid = res["star_id"]
        candidate = res["candidate"]
        feats = res["feats"]

        feats_df = pd.DataFrame([feats])
        prob = float(clf.predict_proba(feats_df)[0])
        hit = prob >= locked_thr  # Use locked threshold from Train OOF CV

        rec = {
            "star_id": sid,
            "prediction": int(hit),
            "confidence": round(prob, 4),
            "period": round(candidate["period"], 5) if (hit and "period" in candidate and pd.notna(candidate["period"])) else None,
            "depth_ppm": round(candidate["depth_ppm"], 1) if (hit and "depth_ppm" in candidate and pd.notna(candidate["depth_ppm"])) else None,
            "duration_hours": round(candidate["duration_hours"], 3) if (hit and "duration_hours" in candidate and pd.notna(candidate["duration_hours"])) else None,
        }
        out_rows.append(rec)

    df_sub = pd.DataFrame(out_rows)
    df_sub.to_csv(output_csv, index=False)

    # Clean trailing blank lines to enforce exactly 88 lines total
    with open(output_csv, "r", encoding="utf-8") as f:
        text = f.read().rstrip("\r\n")
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        f.write(text)

    print(f"\nSaved submission to: {output_csv}")

    # 4. Run official validation assertion checks
    print("\n--- Running Official Competition Assertion Verification ---")
    s = pd.read_csv(output_csv)
    need = ["star_id", "prediction", "confidence", "period", "depth_ppm", "duration_hours"]

    assert list(s.columns) == need, f"Columns must be exactly {need}, got {list(s.columns)}"
    assert len(s) == 87, f"Expected 87 rows, got {len(s)}"
    assert s.star_id.nunique() == 87, "Duplicate star_id detected!"
    assert s.star_id.str.match(r"^STAR_\d{4}$").all(), "star_id format invalid! Must match STAR_XXXX"
    assert s.prediction.isin([0, 1]).all(), "Prediction must be 0 or 1"
    assert s.confidence.between(0, 1).all(), "Confidence out of range [0, 1]"

    pos = s[s.prediction == 1]
    neg = s[s.prediction == 0]
    for c in ("period", "depth_ppm", "duration_hours"):
        assert pos[c].notna().all(), f"Column '{c}' missing for positive detections!"
        assert neg[c].isna().all(), f"Column '{c}' non-empty for prediction=0 rows!"

    assert (pos.period > 0).all(), "Periods must be positive!"

    print(f"SUCCESS: Submission fully validated! {len(pos)} positive planet detections, {len(s) - len(pos)} non-detections.")
    print(f"Confidence min: {s.confidence.min():.4f}, max: {s.confidence.max():.4f}, unique: {s.confidence.nunique()}")


if __name__ == "__main__":
    run_private_inference()
