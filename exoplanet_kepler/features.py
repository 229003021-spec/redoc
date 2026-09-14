"""
Feature extraction module for machine learning candidate vetting.
Extracts astrophysical, signal-to-noise, and vetting features.
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
    r_half = candidate.get("alias_ratio_half", 0.0)
    r_double = candidate.get("alias_ratio_double", 0.0)
    r_triple = candidate.get("alias_ratio_triple", 0.0)

    # Out-of-transit scatter evaluation
    if f is not None and len(f) > 0:
        scatter_std_ppm = float(np.nanstd(f) * 1e6)
        mad = np.nanmedian(np.abs(f - np.nanmedian(f)))
        scatter_mad_ppm = float(1.4826 * mad * 1e6)
    else:
        scatter_std_ppm, scatter_mad_ppm = 0.0, 0.0

    depth_to_scatter = depth_ppm / scatter_mad_ppm if (scatter_mad_ppm > 0 and not np.isnan(depth_ppm)) else 0.0
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
        "scatter_ppm": float(scatter_std_ppm),
        "scatter_mad_ppm": float(scatter_mad_ppm),
        "depth_to_scatter": float(depth_to_scatter),
        "radius_ratio": float(radius_ratio),
        "alias_ratio_half": float(r_half),
        "alias_ratio_double": float(r_double),
        "alias_ratio_triple": float(r_triple)
    }
