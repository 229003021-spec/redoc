"""
Scientific false-positive vetting heuristics for exoplanet candidate verification.
Includes secondary eclipse phase wrapping, odd/even depth consistency, and quarter recurrence.
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
    if np.isnan(period) or period <= 0 or duration_hours <= 0 or len(t) < 50:
        return 1.0, 0.0

    transit_num = np.floor((t - t0 + 0.5 * period) / period).astype(int)
    duration_days = duration_hours / 24.0

    # Phase mapped to [0, 1)
    phase_0_1 = ((t - t0) / period) % 1.0
    in_transit = (phase_0_1 < (duration_days / (2.0 * period))) | (phase_0_1 > (1.0 - (duration_days / (2.0 * period))))

    odd_mask = in_transit & (transit_num % 2 != 0)
    even_mask = in_transit & (transit_num % 2 == 0)
    out_transit = ~in_transit

    # Require minimum in-transit cadences (N >= 5) per parity to avoid tiny point instabilities
    if np.sum(odd_mask) < 5 or np.sum(even_mask) < 5 or np.sum(out_transit) < 50:
        return 1.0, 0.0

    med_out = np.nanmedian(f[out_transit])
    depth_odd = (med_out - np.nanmedian(f[odd_mask])) * 1e6
    depth_even = (med_out - np.nanmedian(f[even_mask])) * 1e6

    std_odd = (np.nanstd(f[odd_mask]) * 1e6 / np.sqrt(np.sum(odd_mask))) if np.sum(odd_mask) > 1 else 1e5
    std_even = (np.nanstd(f[even_mask]) * 1e6 / np.sqrt(np.sum(even_mask))) if np.sum(even_mask) > 1 else 1e5

    diff = abs(depth_odd - depth_even)
    diff_err = np.sqrt(std_odd**2 + std_even**2)
    diff_sig = diff / diff_err if diff_err > 0 else 0.0

    ratio = depth_odd / depth_even if depth_even > 0 else 1.0
    return float(np.clip(ratio, 0.0, 10.0)), float(diff_sig)


def check_secondary_eclipse(t: np.ndarray, f: np.ndarray, period: float, t0: float, duration_hours: float):
    """
    Checks for a secondary eclipse around phase 0.5.
    Phase is represented in [0, 1) so phase 0.5 is centered without phase boundary issues.

    Returns secondary_depth_ratio (depth_sec / depth_pri).
    """
    if np.isnan(period) or period <= 0 or duration_hours <= 0 or len(t) < 50:
        return 0.0

    duration_days = duration_hours / 24.0
    half_width_phase = duration_days / (2.0 * period)

    # Map phase strictly to [0, 1)
    phase_0_1 = ((t - t0) / period) % 1.0

    # Primary eclipse mask around phase 0.0 (or 1.0)
    primary_mask = (phase_0_1 < half_width_phase) | (phase_0_1 > (1.0 - half_width_phase))

    # Secondary eclipse mask centered around phase 0.5
    secondary_mask = np.abs(phase_0_1 - 0.5) < half_width_phase

    out_transit = (~primary_mask) & (~secondary_mask)

    if np.sum(secondary_mask) < 5 or np.sum(primary_mask) < 5 or np.sum(out_transit) < 50:
        return 0.0

    med_out = np.nanmedian(f[out_transit])
    depth_pri = (med_out - np.nanmedian(f[primary_mask])) * 1e6
    depth_sec = (med_out - np.nanmedian(f[secondary_mask])) * 1e6

    if depth_pri <= 0:
        return 0.0

    return float(max(0.0, depth_sec / depth_pri))


def check_quarter_recurrence(t: np.ndarray, f: np.ndarray, q: np.ndarray, period: float, t0: float, duration_hours: float):
    """
    Measures the fraction of Kepler mission quarters in which at least 3 in-transit points occur.
    Prevents single noisy cadence spikes from counting as quarter recurrence.
    """
    if np.isnan(period) or period <= 0 or len(q) == 0 or len(t) < 50:
        return 1.0

    duration_days = duration_hours / 24.0
    half_width_phase = duration_days / (2.0 * period)

    phase_0_1 = ((t - t0) / period) % 1.0
    in_transit = (phase_0_1 < half_width_phase) | (phase_0_1 > (1.0 - half_width_phase))

    quarters_total = np.unique(q)
    if len(quarters_total) == 0:
        return 1.0

    quarters_with_transit = 0
    for qq in quarters_total:
        q_mask = (q == qq)
        if np.sum(in_transit & q_mask) >= 3:  # Require at least 3 valid in-transit cadences
            quarters_with_transit += 1

    return float(quarters_with_transit / len(quarters_total))
