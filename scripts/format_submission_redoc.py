"""
Formats and generates submission_redoc.csv using the exact, original star IDs provided in the dataset files (e.g. KIC_10064054).
Applies strict formatting rules:
- star_id format: Exact filename without extension (e.g., KIC_10064054)
- prediction: 0 or 1
- confidence: [0, 1]
- for prediction=1: period, depth_ppm, duration_hours populated
- for prediction=0: period, depth_ppm, duration_hours empty (NaN in CSV)
"""

import os
import glob
import pandas as pd

def format_submission_redoc():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sub_orig_path = os.path.join(base_dir, "submission.csv")
    out_path = os.path.join(base_dir, "submission_redoc.csv")

    assert os.path.exists(sub_orig_path), f"Original submission {sub_orig_path} not found!"
    sub_orig = pd.read_csv(sub_orig_path)

    out_rows = []
    for idx, row in sub_orig.iterrows():
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
    print(f"SUCCESS: Generated {out_path} with {len(df)} rows maintaining original star IDs!")
    print(df.head(10))

if __name__ == "__main__":
    format_submission_redoc()
