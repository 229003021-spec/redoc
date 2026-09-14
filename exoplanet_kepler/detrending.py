"""
Detrending module for removing stellar variability and instrumental systematics.
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
from .config import DETREND_WINDOW_DAYS, SAVGOL_WINDOW_DAYS, SAVGOL_POLYORDER


def detrend_running_median(t: np.ndarray, f: np.ndarray, window_days: float = DETREND_WINDOW_DAYS):
    """
    Quarter-wise running median detrending.
    """
    cadence = np.median(np.diff(t))
    k = max(5, int(window_days / cadence) | 1) # Force odd window length
    trend = pd.Series(f).rolling(k, center=True, min_periods=k // 3).median().values

    ok = np.isfinite(trend) & (trend > 0)
    return t[ok], f[ok] / trend[ok], trend[ok]


def detrend_savgol(t: np.ndarray, f: np.ndarray, window_days: float = SAVGOL_WINDOW_DAYS, polyorder: int = SAVGOL_POLYORDER, mask_transits: bool = True):
    """
    Savitzky-Golay detrending with optional iterative in-transit masking.
    Masking in-transit points prevents Savitzky-Golay from pulling down the trend line into transits.
    """
    cadence = np.median(np.diff(t))
    window_length = max(polyorder + 2, int(window_days / cadence) | 1)

    # Initial Savitzky-Golay fit
    trend = savgol_filter(f, window_length, polyorder)

    if mask_transits:
        # Identify negative dips below 2.5 sigma and temporarily mask them
        norm_f = f / trend
        res = norm_f - 1.0
        mad = np.median(np.abs(res - np.median(res)))
        sig = 1.4826 * mad
        in_transit = res < -2.5 * sig

        if np.sum(in_transit) > 0 and np.sum(~in_transit) > 50:
            # Interpolate trend over masked points
            f_masked = f.copy()
            f_masked[in_transit] = np.interp(t[in_transit], t[~in_transit], f[~in_transit])
            trend = savgol_filter(f_masked, window_length, polyorder)

    ok = np.isfinite(trend) & (trend > 0)
    return t[ok], f[ok] / trend[ok], trend[ok]


def detrend_lightcurve(t: np.ndarray, f: np.ndarray, method: str = "savgol", window_days: float = 1.0):
    """
    Master function to apply light curve detrending.

    Parameters
    ----------
    t : np.ndarray
        Time array.
    f : np.ndarray
        Quarter-normalized flux array.
    method : str
        'savgol' or 'median'.
    window_days : float
        Filter window size in days.

    Returns
    -------
    t_clean, f_detrended, trend
    """
    if method == "savgol":
        return detrend_savgol(t, f, window_days=window_days)
    elif method == "median":
        return detrend_running_median(t, f, window_days=window_days)
    else:
        return detrend_savgol(t, f, window_days=window_days)
