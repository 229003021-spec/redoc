"""
Detrending module for removing stellar variability and instrumental systematics.
Supports 3-pass iterative in-transit masked Savitzky-Golay detrending and running median fallback.
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from .config import DETREND_WINDOW_DAYS, SAVGOL_WINDOW_DAYS, SAVGOL_POLYORDER


def detrend_running_median(t: np.ndarray, f: np.ndarray, ferr: np.ndarray, q: np.ndarray, window_days: float = DETREND_WINDOW_DAYS):
    """
    Running median detrending maintaining array alignment.
    """
    cadence = np.median(np.diff(t))
    k = max(5, int(window_days / cadence) | 1)  # Force odd window length
    trend = pd.Series(f).rolling(k, center=True, min_periods=k // 3).median().values

    ok = np.isfinite(trend) & (trend > 0)
    t_clean, f_clean, ferr_clean, q_clean, trend_clean = t[ok], f[ok] / trend[ok], ferr[ok] / trend[ok], q[ok], trend[ok]
    assert len(t_clean) == len(f_clean) == len(ferr_clean) == len(q_clean) == len(trend_clean), "Median detrending alignment mismatch!"
    return t_clean, f_clean, ferr_clean, q_clean, trend_clean


def detrend_biweight(t: np.ndarray, f: np.ndarray, ferr: np.ndarray, q: np.ndarray, window_days: float = 2.0, c: float = 6.0):
    """
    Tukey's biweight robust running filter detrending.
    Provides robust, outlier-resistant detrending for non-stationary stellar variability.
    """
    cadence = np.median(np.diff(t))
    half_k = max(2, int((window_days / cadence) / 2))

    trend = np.zeros_like(f)
    n = len(f)

    for i in range(n):
        i_min = max(0, i - half_k)
        i_max = min(n, i + half_k + 1)
        sub_f = f[i_min:i_max]

        med = np.median(sub_f)
        mad = np.median(np.abs(sub_f - med))
        s = 1.4826 * mad

        if s <= 0:
            trend[i] = med
            continue

        u = (sub_f - med) / (c * s)
        mask = np.abs(u) < 1.0
        w = np.zeros_like(u)
        w[mask] = (1.0 - u[mask]**2)**2

        sum_w = np.sum(w)
        trend[i] = np.sum(w * sub_f) / sum_w if sum_w > 0 else med

    ok = np.isfinite(trend) & (trend > 0)
    t_clean, f_clean, ferr_clean, q_clean, trend_clean = t[ok], f[ok] / trend[ok], ferr[ok] / trend[ok], q[ok], trend[ok]
    assert len(t_clean) == len(f_clean) == len(ferr_clean) == len(q_clean) == len(trend_clean), "Biweight detrending alignment mismatch!"
    return t_clean, f_clean, ferr_clean, q_clean, trend_clean


def detrend_savgol(t: np.ndarray, f: np.ndarray, ferr: np.ndarray, q: np.ndarray, window_days: float = SAVGOL_WINDOW_DAYS, polyorder: int = SAVGOL_POLYORDER, n_iter: int = 4, adaptive: bool = True):
    """
    Multi-Scale 2-Pass Iterative Savitzky-Golay detrending with robust edge protection & in-transit masking.
    - Adaptive window scaling allows up to 5.5 days for long lightcurve baselines.
    - 1.8-sigma in-transit dip detection with 15% max mask fraction safety limit to preserve shallow transits (<= 300 ppm).
    - Second pass smoother fit (1.75x window) over masked flux to preserve long-duration transit profiles without erosion.
    - Boundary padding to mitigate endpoint distortions.
    """
    cadence = np.median(np.diff(t))
    t_span = t[-1] - t[0] if len(t) > 1 else 1.0

    # Aggressive adaptive window scaling up to 5.5 days for wide baselines
    effective_window_days = window_days
    if adaptive and t_span > 100.0:
        effective_window_days = max(window_days, min(5.5, t_span / 60.0))

    window_length_p1 = max(polyorder + 2, int(effective_window_days / cadence) | 1)

    f_fit = f.copy()
    trend = savgol_filter(f_fit, window_length_p1, polyorder, mode="nearest")

    # Pass 1: Iterative In-Transit Masking at 1.8-sigma threshold with 15% max mask fraction bound
    max_masked = int(0.15 * len(f))
    for iter_idx in range(n_iter):
        norm_f = f_fit / trend
        res = norm_f - 1.0
        mad = np.nanmedian(np.abs(res))
        sig = 1.4826 * mad

        # Detect candidate transit dips below 1.8 sigma
        in_transit = res < -1.8 * sig
        n_in = np.sum(in_transit)

        if n_in > 0 and n_in <= max_masked and (len(f) - n_in) > 50:
            f_fit[in_transit] = np.interp(t[in_transit], t[~in_transit], f[~in_transit])
            trend = savgol_filter(f_fit, window_length_p1, polyorder, mode="nearest")
        elif n_in > max_masked:
            deepest_indices = np.argsort(res)[:max_masked]
            f_fit[deepest_indices] = np.interp(t[deepest_indices], t[~in_transit], f[~in_transit])
            trend = savgol_filter(f_fit, window_length_p1, polyorder, mode="nearest")
            break
        else:
            break

    # Pass 2: Re-fit smooth trend using longer window (1.75x) over masked flux to preserve wide transit profiles
    window_length_p2 = max(polyorder + 2, int((1.75 * effective_window_days) / cadence) | 1)
    trend = savgol_filter(f_fit, window_length_p2, polyorder, mode="nearest")

    ok = np.isfinite(trend) & (trend > 0)
    t_clean = t[ok]
    f_clean = f[ok] / trend[ok]
    ferr_clean = ferr[ok] / trend[ok]
    q_clean = q[ok]
    trend_clean = trend[ok]

    assert len(t_clean) == len(f_clean) == len(ferr_clean) == len(q_clean) == len(trend_clean), "Savitzky-Golay detrending alignment mismatch!"
    return t_clean, f_clean, ferr_clean, q_clean, trend_clean


def detrend_lightcurve(t: np.ndarray, f: np.ndarray, ferr: np.ndarray = None, q: np.ndarray = None, method: str = "savgol", window_days: float = 1.0):
    """
    Master detrending function ensuring array alignment across all inputs.
    """
    if ferr is None:
        ferr = np.zeros_like(f)
    if q is None:
        q = np.zeros(len(f), dtype=int)

    if method == "savgol":
        return detrend_savgol(t, f, ferr, q, window_days=window_days)
    elif method == "biweight":
        return detrend_biweight(t, f, ferr, q, window_days=window_days)
    elif method == "median":
        return detrend_running_median(t, f, ferr, q, window_days=window_days)
    else:
        return detrend_savgol(t, f, ferr, q, window_days=window_days)
