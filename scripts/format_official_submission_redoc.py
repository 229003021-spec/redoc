"""
Official Submission Formatter & Validator for submission_redoc.csv
Enforces EXACTLY 88 lines (1 header + 87 data rows) with 87 unique star IDs.
"""

import os
import pandas as pd

def generate_official_submission_redoc():
    base_dir = r"c:\Users\Arvind\OneDrive\Documents\exoplanet_kepler_pipeline"
    sub_orig_path = os.path.join(base_dir, "submission.csv")
    out_path = os.path.join(base_dir, "submission_redoc.csv")

    assert os.path.exists(sub_orig_path), f"File {sub_orig_path} not found!"
    sub_orig = pd.read_csv(sub_orig_path)

    # Filter to exactly 87 target rows
    sub_87 = sub_orig.head(87).copy()

    out_rows = []
    for _, row in sub_87.iterrows():
        star_id = str(row["star_id"])
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

    # Clean trailing blank lines to enforce exactly 88 lines in raw file
    with open(out_path, "r", encoding="utf-8") as f:
        text = f.read().rstrip("\r\n")
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write(text)

    # File line count assertion check
    with open(out_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    assert len(lines) == 88, f"Expected exactly 88 lines, got {len(lines)}"
    assert df["star_id"].nunique() == 87, "Duplicate star_id detected!"

    print(f"Generated {out_path} with exactly 88 lines (1 header + 87 data rows).")

    # --- VALIDATION SCRIPT ---
    print("\n--- Running Official Submission Validation ---")
    sub = pd.read_csv(out_path)
    need = ["star_id", "prediction", "confidence", "period", "depth_ppm", "duration_hours"]

    assert list(sub.columns) == need, f"columns must be exactly {need}"
    assert len(sub) == 87, f"expected 87 rows, got {len(sub)}"
    assert sub.star_id.nunique() == 87, "duplicate star_id"
    assert sub.prediction.isin([0, 1]).all(), "prediction must be 0 or 1"
    assert sub.confidence.between(0, 1).all(), "confidence must be in [0, 1]"

    pos = sub[sub.prediction == 1]
    neg = sub[sub.prediction == 0]

    for c in ("period", "depth_ppm", "duration_hours"):
        assert pos[c].notna().all(), f"{c} missing for some prediction=1 rows"
        assert neg[c].isna().all(), f"{c} non-empty for prediction=0 rows"

    assert (pos.period > 0).all(), "period must be positive"

    print(f"OK — {len(pos)} detections, {len(sub) - len(pos)} non-detections across {len(sub)} stars")
    print(
        f"confidence: min {sub.confidence.min():.3f}, "
        f"max {sub.confidence.max():.3f}, "
        f"unique {sub.confidence.nunique()}"
    )

if __name__ == "__main__":
    generate_official_submission_redoc()
