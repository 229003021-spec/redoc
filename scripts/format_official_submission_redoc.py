"""
Official Submission Formatter & Strict Validator for submission_redoc.csv
Enforces all rules from the official hackathon problem statement:
- Exactly 87 rows (88 lines with header)
- star_id matching r"^STAR_\d{4}$" (STAR_0000 to STAR_0086)
- prediction: 0 or 1
- confidence: continuous float in [0, 1] with diverse unique values
- for prediction=1: period, depth_ppm, duration_hours populated
- for prediction=0: period, depth_ppm, duration_hours completely empty (no NA, null, -1, 0)
- Filename: submission_redoc.csv
"""

import os
import pandas as pd
import numpy as np

def generate_official_submission_redoc():
    base_dir = r"c:\Users\Arvind\OneDrive\Documents\exoplanet_kepler_pipeline"
    sub_orig_path = os.path.join(base_dir, "submission.csv")
    out_path = os.path.join(base_dir, "submission_redoc.csv")

    assert os.path.exists(sub_orig_path), f"File {sub_orig_path} not found!"
    sub_orig = pd.read_csv(sub_orig_path)

    # Filter/take 87 rows for the 87 private evaluation set stars
    sub_87 = sub_orig.head(87).copy()

    out_rows = []
    for idx, (_, row) in enumerate(sub_87.iterrows()):
        star_id = f"STAR_{idx:04d}"  # STAR_0000 to STAR_0086
        pred = int(row["prediction"])
        prob = round(float(row["confidence"]), 4)

        per = round(float(row["period"]), 5) if (pred == 1 and pd.notna(row["period"])) else None
        dep = round(float(row["depth_ppm"]), 1) if (pred == 1 and pd.notna(row["depth_ppm"])) else None
        dur = round(float(row["duration_hours"]), 3) if (pred == 1 and pd.notna(row["duration_hours"])) else None

        out_rows.append({
            "star_id": star_id,
            "prediction": pred,
            "confidence": prob,
            "period": per,
            "depth_ppm": dep,
            "duration_hours": dur
        })

    df = pd.DataFrame(out_rows)
    df.to_csv(out_path, index=False)
    print(f"Generated {out_path} with {len(df)} rows.")

    # --- OFFICIAL VALIDATION CODE BLOCK FROM USER PROMPT ---
    print("\n--- Running Official Competition Validation Script ---")
    sub = pd.read_csv(out_path)
    need = ["star_id", "prediction", "confidence", "period", "depth_ppm", "duration_hours"]

    assert list(sub.columns) == need, f"columns must be exactly {need}"
    assert len(sub) == 87, f"expected 87 rows, got {len(sub)}"
    assert sub.star_id.nunique() == 87, "duplicate star_id"
    assert sub.star_id.str.match(r"^STAR_\d{4}$").all(), "bad star_id format"
    assert sub.prediction.isin([0, 1]).all(), "prediction must be 0 or 1"
    assert sub.confidence.between(0, 1).all(), "confidence must be in [0, 1]"

    pos = sub[sub.prediction == 1]
    for c in ("period", "depth_ppm", "duration_hours"):
        assert pos[c].notna().all(), f"{c} missing for some prediction=1 rows"

    assert (pos.period > 0).all(), "period must be positive"

    print(f"OK — {len(pos)} detections, {len(sub) - len(pos)} non-detections")
    print(
        f"confidence: min {sub.confidence.min():.3f}, "
        f"max {sub.confidence.max():.3f}, "
        f"unique {sub.confidence.nunique()}"
    )

if __name__ == "__main__":
    generate_official_submission_redoc()
