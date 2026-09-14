"""
Coarse-to-fine Box Least Squares (BLS) period search module with alias power evaluation.
"""

import numpy as np
from astropy.timeseries import BoxLeastSquares
from .config import PERIOD_MIN, PERIOD_MAX, N_COARSE, N_PEAKS, N_FINE, FINE_WINDOW_FRAC, DURATIONS_DAYS


def calculate_sde(power: np.ndarray, peak_idx: int) -> float:
    """
    Signal Detection Efficiency (SDE): peak height above the periodogram floor
    in robust standard deviations (using Median Absolute Deviation).
    """
    if power is None or len(power) == 0 or peak_idx >= len(power):
        return 0.0
    med = np.nanmedian(power)
    mad = np.nanmedian(np.abs(power - med))
    if mad > 0:
        return float((power[peak_idx] - med) / (1.4826 * mad))
    return 0.0


def evaluate_period_alias_powers(bls: BoxLeastSquares, period: float, durations: np.ndarray) -> tuple:
    """
    Evaluates BLS periodogram power at period harmonics P/2, 2P, and 3P relative to P.
    Helps discriminate true orbital periods from eclipsing binary aliases.
    """
    if np.isnan(period) or period <= 0:
        return 0.0, 0.0, 0.0

    p_half = max(PERIOD_MIN, period / 2.0)
    p_double = period * 2.0
    p_triple = period * 3.0

    test_periods = np.array([period, p_half, p_double, p_triple])
    try:
        res = bls.power(test_periods, durations, objective="likelihood")
        powers = np.asarray(res.power)
        p_base = max(1e-6, powers[0])
        r_half = float(powers[1] / p_base) if len(powers) > 1 else 0.0
        r_double = float(powers[2] / p_base) if len(powers) > 2 else 0.0
        r_triple = float(powers[3] / p_base) if len(powers) > 3 else 0.0
        return r_half, r_double, r_triple
    except Exception:
        return 0.0, 0.0, 0.0


def coarse_fine_bls_search(t: np.ndarray, f: np.ndarray, grid_type: str = "uniform_freq", verbose: bool = False) -> dict:
    """
    Performs coarse-to-fine Box Least Squares period search on cleaned, detrended flux.

    Parameters
    ----------
    t : np.ndarray
        Time array in BKJD days.
    f : np.ndarray
        Cleaned detrended flux array.
    grid_type : str
        'uniform_freq' (frequency-spaced grid scaling as P^2) or 'log' (log-spaced grid).

    Returns
    -------
    dict containing candidate parameters and signal diagnostics.
    """
    if t is None or len(t) < 500:
        return {
            "period": np.nan, "depth_ppm": np.nan, "duration_hours": np.nan,
            "t0": np.nan, "sde": 0.0, "snr": 0.0, "n_transits_expected": 0, "n_in_transit_points": 0,
            "alias_ratio_half": 0.0, "alias_ratio_double": 0.0, "alias_ratio_triple": 0.0
        }

    bls = BoxLeastSquares(t, f)
    baseline = t.max() - t.min()
    pmax = min(PERIOD_MAX, baseline / 2.0)  # Support periods up to baseline / 2

    if pmax <= PERIOD_MIN:
        pmax = PERIOD_MIN + 1.0

    # 1. --- Coarse Sweep Grid Construction ---
    durations = np.array(DURATIONS_DAYS)
    if grid_type == "uniform_freq":
        # Uniform frequency grid: dfreq = 1 / (4 * baseline) => trial period resolution scales as P^2
        f_min = 1.0 / pmax
        f_max = 1.0 / PERIOD_MIN
        dfreq = 1.0 / (4.0 * baseline)
        n_freqs = min(N_COARSE, max(2000, int((f_max - f_min) / dfreq)))
        freqs = np.linspace(f_min, f_max, n_freqs)
        coarse_periods = 1.0 / freqs[::-1]
    else:
        coarse_periods = np.exp(np.linspace(np.log(PERIOD_MIN), np.log(pmax), N_COARSE))

    coarse_res = bls.power(coarse_periods, durations, objective="likelihood")
    coarse_power = np.asarray(coarse_res.power)

    # 2. --- Peak Extraction with Neighborhood Exclusion ---
    order = np.argsort(coarse_power)[::-1]
    peaks, used = [], np.zeros(len(coarse_periods), dtype=bool)

    for idx in order:
        if used[idx]:
            continue
        peaks.append(idx)
        lo = np.searchsorted(coarse_periods, coarse_periods[idx] * 0.9)
        hi = np.searchsorted(coarse_periods, coarse_periods[idx] * 1.1)
        used[lo:hi] = True
        if len(peaks) >= N_PEAKS:
            break

    # 3. --- Fine Refinement Around Promising Peaks ---
    best_candidate = None

    for idx in peaks:
        p0 = coarse_periods[idx]
        half_width = FINE_WINDOW_FRAC * p0  # +/- 2%
        fine_periods = np.linspace(p0 - half_width, p0 + half_width, N_FINE)
        fine_periods = fine_periods[(fine_periods >= PERIOD_MIN) & (fine_periods <= pmax)]

        if len(fine_periods) < 10:
            continue

        fine_res = bls.power(fine_periods, durations, objective="likelihood")
        fine_power = np.asarray(fine_res.power)
        best_j = int(np.nanargmax(fine_power))

        # Evaluate SDE score on fine periodogram peak height
        score = calculate_sde(fine_power, best_j)

        period = float(fine_res.period[best_j])
        depth_ppm = float(fine_res.depth[best_j] * 1e6)
        duration_hours = float(fine_res.duration[best_j] * 24.0)
        t0 = float(fine_res.transit_time[best_j])

        # Estimate expected transits & fold noise SNR using robust MAD
        n_transits_exp = int(baseline / period) if period > 0 else 0

        # Phase fold to compute fold SNR
        phase = ((t - t0) / period) % 1.0
        phase = np.where(phase > 0.5, phase - 1.0, phase)
        in_transit_mask = np.abs(phase) < (duration_hours / (24.0 * 2.0 * period))
        n_in_transit = np.sum(in_transit_mask)

        # Robust out-of-transit MAD scatter
        out_transit_f = f[~in_transit_mask]
        scatter_ppm = float(np.nanmedian(np.abs(out_transit_f - np.nanmedian(out_transit_f))) * 1.4826 * 1e6) if len(out_transit_f) > 50 else float(np.nanstd(f) * 1e6)
        snr = (depth_ppm / (scatter_ppm / np.sqrt(n_in_transit))) if (n_in_transit >= 5 and scatter_ppm > 0) else 0.0

        # Evaluate harmonic alias powers
        r_half, r_double, r_triple = evaluate_period_alias_powers(bls, period, durations)

        candidate = {
            "period": period,
            "depth_ppm": depth_ppm,
            "duration_hours": duration_hours,
            "t0": t0,
            "sde": score,
            "snr": float(snr),
            "n_transits_expected": n_transits_exp,
            "n_in_transit_points": int(n_in_transit),
            "alias_ratio_half": r_half,
            "alias_ratio_double": r_double,
            "alias_ratio_triple": r_triple
        }

        # Select candidate with highest fine SDE score & valid positive transit depth
        if (best_candidate is None or score > best_candidate["sde"]) and depth_ppm > 0:
            best_candidate = candidate

    return best_candidate or {
        "period": np.nan, "depth_ppm": np.nan, "duration_hours": np.nan,
        "t0": np.nan, "sde": 0.0, "snr": 0.0, "n_transits_expected": 0, "n_in_transit_points": 0,
        "alias_ratio_half": 0.0, "alias_ratio_double": 0.0, "alias_ratio_triple": 0.0
    }
