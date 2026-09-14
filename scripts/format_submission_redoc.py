"""
Formats and generates submission_redoc.csv for the 87 private stars (STAR_0001 - STAR_0087).
Applies strict formatting rules:
- star_id format: STAR_XXXX
- prediction: 0 or 1
- confidence: [0, 1]
- for prediction=1: period, depth_ppm, duration_hours populated
- for prediction=0: period, depth_ppm, duration_hours empty (NaN)
- 87 data rows
"""

import os
import glob
import pandas as pd

def format_submission_redoc():
    base_dir = r"c:\Users\Arvind\OneDrive\Documents\exoplanet_kepler_pipeline"
    sub_orig_path = os.path.join(base_dir, "submission.csv")
    priv_dir = os.path.join(base_dir, "data", "private")
    out_path = os.path.join(base_dir, "submission_redoc.csv")

    assert os.path.exists(sub_orig_path), f"Original submission {sub_orig_path} not found!"
    sub_orig = pd.read_csv(sub_orig_path)

    priv_files = sorted(glob.glob(os.path.join(priv_dir, "*.parquet")))
    print(f"Total private set files: {len(priv_files)}")
    assert len(priv_files) == 87, f"Expected 87 files, found {len(priv_files)}"

    out_rows = []
    for idx in range(87):
        star_id = f"STAR_{idx+1:04d}"
        row = sub_orig.iloc[idx]
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
    print(f"SUCCESS: Generated {out_path} with {len(df)} rows!")

if __name__ == "__main__":
    format_submission_redoc()
