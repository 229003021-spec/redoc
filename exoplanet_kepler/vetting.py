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


def check_secondary_eclipse_details(t: np.ndarray, f: np.ndarray, period: float, t0: float, duration_hours: float):
    """
    Checks for secondary eclipse around phase 0.5.
    Returns:
    - sec_depth_ratio (float, depth_sec / depth_pri)
    - sec_depth_sig (float, statistical significance of secondary depth in units of sigma)
    """
    if np.isnan(period) or period <= 0 or duration_hours <= 0 or len(t) < 50:
        return 0.0, 0.0

    duration_days = duration_hours / 24.0
    half_width_phase = duration_days / (2.0 * period)

    phase_0_1 = ((t - t0) / period) % 1.0
    primary_mask = (phase_0_1 < half_width_phase) | (phase_0_1 > (1.0 - half_width_phase))
    secondary_mask = np.abs(phase_0_1 - 0.5) < half_width_phase
    out_transit = (~primary_mask) & (~secondary_mask)

    if np.sum(secondary_mask) < 5 or np.sum(primary_mask) < 5 or np.sum(out_transit) < 50:
        return 0.0, 0.0

    med_out = np.nanmedian(f[out_transit])
    mad_out = np.nanmedian(np.abs(f[out_transit] - med_out)) * 1.4826 * 1e6
    depth_pri = (med_out - np.nanmedian(f[primary_mask])) * 1e6
    depth_sec = (med_out - np.nanmedian(f[secondary_mask])) * 1e6

    sec_err = mad_out / np.sqrt(np.sum(secondary_mask)) if np.sum(secondary_mask) > 1 else 1e5
    sec_sig = float(max(0.0, depth_sec / sec_err)) if sec_err > 0 else 0.0

    if depth_pri <= 0:
        return 0.0, sec_sig

    return float(max(0.0, depth_sec / depth_pri)), sec_sig


def check_secondary_eclipse(t: np.ndarray, f: np.ndarray, period: float, t0: float, duration_hours: float):
    """Backward compatible wrapper returning secondary depth ratio."""
    sec_ratio, _ = check_secondary_eclipse_details(t, f, period, t0, duration_hours)
    return sec_ratio


def check_transit_shape_metric(t: np.ndarray, f: np.ndarray, period: float, t0: float, duration_hours: float):
    """
    Computes V-shape vs U-shape transit metric.
    Flat-bottomed U-shaped planetary transits have mean in-transit depth ~ median in-transit depth (ratio ~ 1.0).
    V-shaped grazing eclipsing binary transits have steep central dips (ratio > 1.25).
    """
    if np.isnan(period) or period <= 0 or duration_hours <= 0 or len(t) < 50:
        return 1.0

    duration_days = duration_hours / 24.0
    half_width_phase = duration_days / (2.0 * period)

    phase_0_1 = ((t - t0) / period) % 1.0
    in_transit = (phase_0_1 < half_width_phase) | (phase_0_1 > (1.0 - half_width_phase))
    out_transit = ~in_transit

    if np.sum(in_transit) < 5 or np.sum(out_transit) < 50:
        return 1.0

    med_out = np.nanmedian(f[out_transit])
    dips = (med_out - f[in_transit]) * 1e6
    mean_dip = np.nanmean(dips)
    max_dip = np.nanmax(dips)

    if mean_dip <= 0:
        return 1.0

    return float(np.clip(max_dip / mean_dip, 0.5, 5.0))


def check_local_snr(t: np.ndarray, f: np.ndarray, period: float, t0: float, duration_hours: float):
    """
    Computes transit signal-to-noise ratio evaluated in a narrow local phase window (+/- 2x transit duration).
    """
    if np.isnan(period) or period <= 0 or duration_hours <= 0 or len(t) < 50:
        return 0.0

    duration_days = duration_hours / 24.0
    half_width_phase = duration_days / (2.0 * period)

    phase_0_1 = ((t - t0) / period) % 1.0
    phase_centered = np.where(phase_0_1 > 0.5, phase_0_1 - 1.0, phase_0_1)

    in_transit = np.abs(phase_centered) < half_width_phase
    local_out = (np.abs(phase_centered) >= half_width_phase) & (np.abs(phase_centered) <= 3.0 * half_width_phase)

    if np.sum(in_transit) < 5 or np.sum(local_out) < 20:
        return 0.0

    med_local_out = np.nanmedian(f[local_out])
    local_mad = np.nanmedian(np.abs(f[local_out] - med_local_out)) * 1.4826 * 1e6
    depth_ppm = (med_local_out - np.nanmedian(f[in_transit])) * 1e6

    if local_mad <= 0 or depth_ppm <= 0:
        return 0.0

    sem = local_mad / np.sqrt(np.sum(in_transit))
    return float(depth_ppm / sem) if sem > 0 else 0.0


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
