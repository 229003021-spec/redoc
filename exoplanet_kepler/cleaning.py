"""
Data cleaning and quality filtering module for Kepler photometry.
"""

import numpy as np
import pandas as pd
from .config import MIN_CADENCES, SIGMA_CLIP_OUTLIERS


def clean_lightcurve(df: pd.DataFrame, drop_quality_nonzero: bool = True, sigma_clip: float = SIGMA_CLIP_OUTLIERS):
    """
    Cleans raw Kepler photometry parquet DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing columns: time, flux, flux_err, quality, quarter.
    drop_quality_nonzero : bool
        If True, retains only cadences where quality == 0.
    sigma_clip : float
        Sigma threshold for clipping extreme positive flux outliers (cosmic rays).

    Returns
    -------
    tuple (t, f, ferr, q) or (None, None, None, None)
        Cleaned time, normalized flux, normalized flux_err, and quarter arrays.
    """
    if df is None or len(df) == 0:
        return None, None, None, None

    # Quality mask & finite value check
    if drop_quality_nonzero and "quality" in df.columns:
        mask = (df["quality"].values == 0) & np.isfinite(df["flux"].values) & np.isfinite(df["time"].values)
    else:
        mask = np.isfinite(df["flux"].values) & np.isfinite(df["time"].values)

    t = df["time"].values[mask].astype(np.float64)
    f = df["flux"].values[mask].astype(np.float64)
    ferr = df["flux_err"].values[mask].astype(np.float64) if "flux_err" in df.columns else np.zeros_like(f)
    q = df["quarter"].values[mask] if "quarter" in df.columns else np.zeros(len(f), dtype=int)

    if len(t) < MIN_CADENCES:
        return None, None, None, None

    # Normalise each quarter to its own median flux
    for qq in np.unique(q):
        s = (q == qq)
        med = np.nanmedian(f[s])
        if med > 0:
            f[s] = f[s] / med
            ferr[s] = ferr[s] / med
        else:
            f[s] = 1.0

    # Sigma clip extreme positive outliers (e.g. cosmic rays spiking > 6 sigma)
    # Note: We do NOT clip extreme negative points because transits are negative dips!
    std_est = np.nanmedian(np.abs(f - 1.0)) * 1.4826
    if std_est > 0:
        valid_outlier = (f - 1.0) < (sigma_clip * std_est)
        t, f, ferr, q = t[valid_outlier], f[valid_outlier], ferr[valid_outlier], q[valid_outlier]

    return t, f, ferr, q
