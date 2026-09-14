"""
Scientific false-positive vetting heuristics for exoplanet candidate verification.
"""

import numpy as np


def check_odd_even_consistency(t: np.ndarray, f: np.ndarray, period: float, t0: float, duration_hours: float):
    """
    Computes odd vs even transit depth consistency.

    Eclipsing binaries (EBs) with different primary/secondary eclipse depths will exhibit
    a large difference between odd and even transit depths when folded at half period.
    True planets exhibit odd/even depth ratio near 1.0.

    Returns:
    - odd_even_depth_ratio (float)
    - odd_even_diff_sig (float, statistical significance of difference)
    """
    if np.isnan(period) or period <= 0 or duration_hours <= 0:
        return 1.0, 0.0

    transit_num = np.floor((t - t0 + 0.5 * period) / period).astype(int)
    duration_days = duration_hours / 24.0

    # In-transit phase mask
    phase = ((t - t0) / period) % 1.0
    phase = np.where(phase > 0.5, phase - 1.0, phase)
    in_transit = np.abs(phase) < (duration_days / (2.0 * period))

    odd_mask = in_transit & (transit_num % 2 != 0)
    even_mask = in_transit & (transit_num % 2 == 0)

    # Baseline out-of-transit mask
    out_transit = ~in_transit

    if np.sum(odd_mask) < 3 or np.sum(even_mask) < 3 or np.sum(out_transit) < 50:
        return 1.0, 0.0

    med_out = np.median(f[out_transit])
    depth_odd = (med_out - np.median(f[odd_mask])) * 1e6
    depth_even = (med_out - np.median(f[even_mask])) * 1e6

    std_odd = np.std(f[odd_mask]) * 1e6 / np.sqrt(np.sum(odd_mask))
    std_even = np.std(f[even_mask]) * 1e6 / np.sqrt(np.sum(even_mask))

    diff = abs(depth_odd - depth_even)
    diff_err = np.sqrt(std_odd**2 + std_even**2)
    diff_sig = diff / diff_err if diff_err > 0 else 0.0

    ratio = depth_odd / depth_even if depth_even > 0 else 1.0
    return float(np.clip(ratio, 0.0, 10.0)), float(diff_sig)


def check_secondary_eclipse(t: np.ndarray, f: np.ndarray, period: float, t0: float, duration_hours: float):
    """
    Checks for a secondary eclipse at phase 0.5.
    Returns secondary_depth_ratio (depth_sec / depth_pri).
    """
    if np.isnan(period) or period <= 0 or duration_hours <= 0:
        return 0.0

    duration_days = duration_hours / 24.0
    phase = ((t - t0) / period) % 1.0
    phase = np.where(phase > 0.5, phase - 1.0, phase)

    primary_mask = np.abs(phase) < (duration_days / (2.0 * period))
    secondary_mask = np.abs(phase - 0.5) < (duration_days / (2.0 * period))
    out_transit = (~primary_mask) & (~secondary_mask)

    if np.sum(secondary_mask) < 3 or np.sum(primary_mask) < 3 or np.sum(out_transit) < 50:
        return 0.0

    med_out = np.median(f[out_transit])
    depth_pri = (med_out - np.median(f[primary_mask])) * 1e6
    depth_sec = (med_out - np.median(f[secondary_mask])) * 1e6

    if depth_pri <= 0:
        return 0.0

    return float(max(0.0, depth_sec / depth_pri))


def check_quarter_recurrence(t: np.ndarray, f: np.ndarray, q: np.ndarray, period: float, t0: float, duration_hours: float):
    """
    Measures the fraction of Kepler mission quarters in which the transit is observed.
    Instrumental artifacts often appear in only a single quarter.
    """
    if np.isnan(period) or period <= 0 or len(q) == 0:
        return 1.0

    duration_days = duration_hours / 24.0
    phase = ((t - t0) / period) % 1.0
    phase = np.where(phase > 0.5, phase - 1.0, phase)
    in_transit = np.abs(phase) < (duration_days / (2.0 * period))

    quarters_total = np.unique(q)
    quarters_with_transit = np.unique(q[in_transit])

    if len(quarters_total) == 0:
        return 1.0

    return float(len(quarters_with_transit) / len(quarters_total))
