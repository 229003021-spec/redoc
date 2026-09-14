"""
Diagnostic Analysis & Validation Script for Exoplanet Pipeline
Generates diagnostic metrics, candidate ranking, summary table of all target stars,
and runs official validation on submission_redoc.csv.
"""

import os
import sys
import pandas as pd
import numpy as np

def run_official_validation(csv_path="submission_redoc.csv"):
    print("\n=======================================================")
    print("      OFFICIAL SUBMISSION VALIDATION LOGIC CHECK       ")
    print("=======================================================")
    
    assert os.path.exists(csv_path), f"File {csv_path} does not exist!"
    sub = pd.read_csv(csv_path)

    need = [
        "star_id",
        "prediction",
        "confidence",
        "period",
        "depth_ppm",
        "duration_hours"
    ]

    assert list(sub.columns) == need, f"Column mismatch! Expected {need}, got {list(sub.columns)}"
    assert len(sub) > 0, "Submission file is empty!"
    assert sub.star_id.nunique() == len(sub), "Duplicate star_id detected!"
    assert sub.prediction.isin([0, 1]).all(), "prediction must be 0 or 1"
    assert sub.confidence.between(0, 1).all(), "confidence out of range [0, 1]"

    pos = sub[sub.prediction == 1]
    neg = sub[sub.prediction == 0]

    for c in ("period", "depth_ppm", "duration_hours"):
        assert pos[c].notna().all(), f"Positive detection missing parameter in column '{c}'!"
        assert neg[c].isna().all(), f"Negative detection has non-empty values in column '{c}'!"

    assert (pos.period > 0).all(), "Periods must be positive!"

    print(f"OK — {len(pos)} detections, {len(sub) - len(pos)} non-detections across {len(sub)} stars")
    print(
        f"confidence: min {sub.confidence.min():.3f}, "
        f"max {sub.confidence.max():.3f}, "
        f"unique {sub.confidence.nunique()}"
    )
    print("=======================================================\n")
    return sub, pos, neg

def generate_diagnostic_reports(sub):
    print("=== CANDIDATE RANKING (TOP 10 STRONGEST EXOPLANET CANDIDATES) ===")
    top10 = sub.sort_values(by="confidence", ascending=False).head(10)
    print(top10.to_string(index=False))

    print("\n=== UNCERTAIN / BORDERLINE CANDIDATES (CONFIDENCE 0.45 - 0.55) ===")
    uncertain = sub[(sub["confidence"] >= 0.45) & (sub["confidence"] <= 0.55)]
    print(uncertain.to_string(index=False) if len(uncertain) > 0 else "No ambiguous candidates in range [0.45, 0.55].")

    print("\n=== SUMMARY METRICS OF ALL TARGET STARS ===")
    print(f"Total Stars Processed : {len(sub)}")
    print(f"Predicted Positives   : {(sub['prediction'] == 1).sum()}")
    print(f"Predicted Negatives   : {(sub['prediction'] == 0).sum()}")
    print(f"Confidence Range      : [{sub['confidence'].min():.4f}, {sub['confidence'].max():.4f}]")
    print(f"Mean Confidence       : {sub['confidence'].mean():.4f}")
    print(f"Unique Confidence Vals: {sub['confidence'].nunique()}")

if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "submission_redoc.csv"
    sub, pos, neg = run_official_validation(csv_file)
    generate_diagnostic_reports(sub)
