"""
Evaluation metrics module for detection performance and planet characterization accuracy.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, auc, average_precision_score


def is_period_accurate(found_period: float, true_period: float, tol: float = 0.02) -> bool:
    """
    Checks if detected period is within tol (2%) of true period or an accepted alias (2x, 1/2x, 3x, 1/3x).
    """
    if np.isnan(found_period) or np.isnan(true_period) or true_period <= 0:
        return False

    aliases = [1.0, 2.0, 0.5, 3.0, 1/3.0, 4.0, 0.25]
    for mult in aliases:
        target = true_period * mult
        if abs(found_period - target) / target <= tol:
            return True
    return False


def evaluate_detection_metrics(df_results: pd.DataFrame) -> dict:
    """
    Computes challenge evaluation metrics on development set:
    - Precision, Recall, F1
    - PR-AUC, Average Precision
    - Period Accuracy (overall and for true positive transits)
    - Relative Depth Error
    - Metric breakdown across depth brackets
    """
    y_true = df_results["has_planet"].values.astype(int)
    y_pred = df_results["prediction"].values.astype(int)
    y_conf = df_results["confidence"].values.astype(float)

    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_conf)
        pr_auc = float(auc(rec_curve, prec_curve))
        avg_prec = float(average_precision_score(y_true, y_conf))
    except Exception:
        pr_auc, avg_prec = 0.0, 0.0

    # Characterization Accuracy on True Planets
    true_planets = df_results[df_results["has_planet"] == 1].copy()
    n_true_planets = len(true_planets)

    period_correct_count = 0
    depth_errors = []

    for idx, row in true_planets.iterrows():
        p_found = row.get("period", np.nan)
        p_true = row.get("period_days", np.nan)
        d_found = row.get("depth_ppm", np.nan)
        d_true = row.get("depth_ppm_true", np.nan)

        if is_period_accurate(p_found, p_true):
            period_correct_count += 1

        if not np.isnan(d_found) and not np.isnan(d_true) and d_true > 0:
            depth_errors.append(abs(d_found - d_true) / d_true)

    period_acc = float(period_correct_count / n_true_planets) if n_true_planets > 0 else 0.0
    mean_depth_err = float(np.mean(depth_errors)) if len(depth_errors) > 0 else 0.0

    # Depth Bracket Breakdown
    brackets = {
        "Deep (>1000 ppm)": true_planets[true_planets["depth_ppm_true"] > 1000],
        "Medium (500-1000 ppm)": true_planets[(true_planets["depth_ppm_true"] > 500) & (true_planets["depth_ppm_true"] <= 1000)],
        "Shallow (200-500 ppm)": true_planets[(true_planets["depth_ppm_true"] > 200) & (true_planets["depth_ppm_true"] <= 500)],
        "Very Shallow (<=200 ppm)": true_planets[true_planets["depth_ppm_true"] <= 200]
    }

    bracket_recall = {}
    for name, sub in brackets.items():
        if len(sub) > 0:
            rec_b = float((sub["prediction"] == 1).mean())
            bracket_recall[name] = (rec_b, len(sub))
        else:
            bracket_recall[name] = (0.0, 0)

    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "pr_auc": pr_auc,
        "avg_precision": avg_prec,
        "period_accuracy": period_acc,
        "mean_depth_error": mean_depth_err,
        "depth_brackets": bracket_recall
    }
