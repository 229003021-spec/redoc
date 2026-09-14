"""
Feature extraction module for machine learning candidate vetting.
"""

import numpy as np
from .vetting import check_odd_even_consistency, check_secondary_eclipse, check_quarter_recurrence


def extract_candidate_features(t: np.ndarray, f: np.ndarray, q: np.ndarray, candidate: dict) -> dict:
    """
    Extracts a feature dictionary from light curve and candidate BLS output.
    """
    period = candidate.get("period", np.nan)
    depth_ppm = candidate.get("depth_ppm", np.nan)
    duration_hours = candidate.get("duration_hours", np.nan)
    t0 = candidate.get("t0", np.nan)
    sde = candidate.get("sde", 0.0)
    snr = candidate.get("snr", 0.0)
    n_transits_exp = candidate.get("n_transits_expected", 0)
    n_in_transit = candidate.get("n_in_transit_points", 0)

    scatter_ppm = float(np.nanstd(f) * 1e6) if f is not None and len(f) > 0 else 0.0
    depth_to_scatter = depth_ppm / scatter_ppm if (scatter_ppm > 0 and not np.isnan(depth_ppm)) else 0.0

    # Radius ratio estimate: (Rp / R*)^2 = depth (in fraction) => Rp / R* = sqrt(depth_ppm / 1e6)
    radius_ratio = float(np.sqrt(max(0.0, depth_ppm) / 1e6)) if not np.isnan(depth_ppm) else 0.0

    # Scientific vetting metrics
    if t is not None and len(t) > 500 and not np.isnan(period) and period > 0:
        odd_even_ratio, odd_even_diff_sig = check_odd_even_consistency(t, f, period, t0, duration_hours)
        sec_ratio = check_secondary_eclipse(t, f, period, t0, duration_hours)
        quarter_rec = check_quarter_recurrence(t, f, q, period, t0, duration_hours)
    else:
        odd_even_ratio, odd_even_diff_sig = 1.0, 0.0
        sec_ratio = 0.0
        quarter_rec = 1.0

    return {
        "sde": float(sde),
        "snr": float(snr),
        "period": float(period) if not np.isnan(period) else 0.0,
        "depth_ppm": float(depth_ppm) if not np.isnan(depth_ppm) else 0.0,
        "duration_hours": float(duration_hours) if not np.isnan(duration_hours) else 0.0,
        "n_transits_expected": int(n_transits_exp),
        "n_in_transit_points": int(n_in_transit),
        "odd_even_ratio": float(odd_even_ratio),
        "odd_even_diff_sig": float(odd_even_diff_sig),
        "sec_depth_ratio": float(sec_ratio),
        "quarter_recurrence": float(quarter_rec),
        "scatter_ppm": float(scatter_ppm),
        "depth_to_scatter": float(depth_to_scatter),
        "radius_ratio": float(radius_ratio)
    }
