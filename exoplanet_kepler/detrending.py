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


def detrend_savgol(t: np.ndarray, f: np.ndarray, ferr: np.ndarray, q: np.ndarray, window_days: float = SAVGOL_WINDOW_DAYS, polyorder: int = SAVGOL_POLYORDER, n_iter: int = 3):
    """
    3-Pass Iterative Savitzky-Golay detrending with in-transit masking.
    Iteratively detects dips below 2.5 sigma, interpolates over candidate transit regions,
    and refits the trend to prevent flattening of genuine transits.
    """
    cadence = np.median(np.diff(t))
    window_length = max(polyorder + 2, int(window_days / cadence) | 1)

    f_fit = f.copy()
    trend = savgol_filter(f_fit, window_length, polyorder)

    # 3-Pass Iterative In-Transit Masking
    for iter_idx in range(n_iter):
        norm_f = f_fit / trend
        res = norm_f - 1.0
        mad = np.nanmedian(np.abs(res))
        sig = 1.4826 * mad

        # Identify candidate transit dips below 2.5 sigma
        in_transit = res < -2.5 * sig
        n_in = np.sum(in_transit)

        if n_in > 0 and (len(f) - n_in) > 50:
            f_fit[in_transit] = np.interp(t[in_transit], t[~in_transit], f[~in_transit])
            trend = savgol_filter(f_fit, window_length, polyorder)
        else:
            break

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
    elif method == "median":
        return detrend_running_median(t, f, ferr, q, window_days=window_days)
    else:
        return detrend_savgol(t, f, ferr, q, window_days=window_days)
