"""
Final Submission Generator & Validation Script (Parallelized).
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
from scripts.run_pipeline_eval import process_dataset, process_single_star


def process_private_star(path):
    """
    Helper function to process a single private set star.
    """
    sid = os.path.basename(path)[:-8] # e.g. STAR_0001
    try:
        df = pd.read_parquet(path)
        t_raw, f_raw, ferr_raw, q_raw = clean_lightcurve(df)
        if t_raw is None or len(t_raw) < 500:
            candidate = {}
            feats = {col: 0.0 for col in FEATURE_COLS}
        else:
            t, f, trend = detrend_lightcurve(t_raw, f_raw, method="savgol", window_days=1.0)
            candidate = coarse_fine_bls_search(t, f)
            feats = extract_candidate_features(t, f, q_raw[:len(t)], candidate)
    except Exception as e:
        candidate = {}
        feats = {col: 0.0 for col in FEATURE_COLS}

    return {"star_id": sid, "candidate": candidate, "feats": feats}


def generate_and_validate_submission(private_dir, output_csv="submission.csv"):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")

    # 1. Train ML model on combined Train + Dev sets for maximum data efficiency
    print("Training final ML classifier on combined Train + Dev datasets...")
    df_train = process_dataset(
        os.path.join(data_dir, "train"),
        os.path.join(data_dir, "train_labels.csv"),
        os.path.join(data_dir, "train_truth.csv")
    )
    df_dev = process_dataset(
        os.path.join(data_dir, "dev"),
        os.path.join(data_dir, "dev_labels.csv"),
        os.path.join(data_dir, "dev_truth.csv")
    )
    df_all = pd.concat([df_train, df_dev], ignore_index=True)

    clf = ExoplanetCandidateClassifier(model_type="hist_gb")
    clf.fit(df_all, df_all["has_planet"].values)
    print("Final ML model successfully trained!")

    # 2. Process Private Evaluation Set
    paths = sorted(glob.glob(os.path.join(private_dir, "*.parquet")))
    if len(paths) == 0:
        print(f"Warning: No parquet files found in {private_dir}")
        return

    n_paths = len(paths)
    print(f"Generating predictions for {n_paths} private evaluation set stars using {os.cpu_count()} CPU cores...")

    t0 = time.time()
    results = Parallel(n_jobs=-1, batch_size=4)(
        delayed(process_private_star)(p) for p in paths
    )
    print(f"Completed private set processing in {time.time() - t0:.1f}s.")

    out_rows = []
    for res in results:
        sid = res["star_id"]
        candidate = res["candidate"]
        feats = res["feats"]

        feats_df = pd.DataFrame([feats])
        prob = float(clf.predict_proba(feats_df)[0])
        hit = prob >= 0.5

        rec = {
            "star_id": sid,
            "prediction": int(hit),
            "confidence": round(prob, 4),
            "period": round(candidate["period"], 5) if (hit and "period" in candidate) else None,
            "depth_ppm": round(candidate["depth_ppm"], 1) if (hit and "depth_ppm" in candidate) else None,
            "duration_hours": round(candidate["duration_hours"], 3) if (hit and "duration_hours" in candidate) else None,
        }
        out_rows.append(rec)

    df_sub = pd.DataFrame(out_rows)
    df_sub.to_csv(output_csv, index=False)
    print(f"\nSaved submission to: {output_csv}")

    # 3. Official Format Validation Assertions
    print("\n--- Running Official Submission Assertion Verification ---")
    s = pd.read_csv(output_csv)
    need = ["star_id", "prediction", "confidence", "period", "depth_ppm", "duration_hours"]

    assert list(s.columns) == need, f"Columns must be exactly {need}, got {list(s.columns)}"
    assert len(s) == n_paths, f"Expected {n_paths} rows, got {len(s)}"
    assert s.star_id.nunique() == n_paths, "Duplicate star_id detected!"
    assert s.prediction.isin([0, 1]).all(), "Prediction must be 0 or 1"
    assert s.confidence.between(0, 1).all(), "Confidence out of range [0, 1]"

    pos = s[s.prediction == 1]
    for c in ("period", "depth_ppm", "duration_hours"):
        assert pos[c].notna().all(), f"Column '{c}' missing for positive detections!"

    print(f"SUCCESS: Submission fully validated! {len(pos)} positive planet detections, {len(s) - len(pos)} non-detections.")


if __name__ == "__main__":
    base_d = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    priv_d = os.path.join(base_d, "data", "private") if os.path.exists(os.path.join(base_d, "data", "private")) else os.path.join(base_d, "data", "dev")
    if len(sys.argv) > 1:
        priv_d = sys.argv[1]
    out_csv = sys.argv[2] if len(sys.argv) > 2 else "submission_redoc.csv"
    generate_and_validate_submission(priv_d, output_csv=out_csv)
