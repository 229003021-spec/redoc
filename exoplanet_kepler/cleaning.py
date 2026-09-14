"""
Data cleaning and quality filtering module for Kepler photometry.
Preserves strict array alignment across time, flux, flux_err, quality, and quarter arrays.
"""

import numpy as np
import pandas as pd
from .config import MIN_CADENCES, SIGMA_CLIP_OUTLIERS


def clean_lightcurve(df: pd.DataFrame, drop_quality_nonzero: bool = True, sigma_clip: float = SIGMA_CLIP_OUTLIERS):
    """
    Cleans raw Kepler photometry DataFrame.

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
        Cleaned and aligned time, normalized flux, normalized flux_err, and quarter arrays.
    """
    if df is None or len(df) == 0:
        return None, None, None, None

    # Step 1: Combined quality & finite value boolean index mask
    if drop_quality_nonzero and "quality" in df.columns:
        mask = (df["quality"].values == 0) & np.isfinite(df["flux"].values) & np.isfinite(df["time"].values)
    else:
        mask = np.isfinite(df["flux"].values) & np.isfinite(df["time"].values)

    t = df["time"].values[mask].astype(np.float64)
    f = df["flux"].values[mask].astype(np.float64)
    ferr = df["flux_err"].values[mask].astype(np.float64) if "flux_err" in df.columns else np.zeros_like(f)
    q = df["quarter"].values[mask].astype(int) if "quarter" in df.columns else np.zeros(len(f), dtype=int)

    # Array alignment assertions
    assert len(t) == len(f) == len(ferr) == len(q), "Initial array length mismatch after quality filtering!"

    if len(t) < MIN_CADENCES:
        return None, None, None, None

    # Step 2: Normalise each quarter to its own median flux
    for qq in np.unique(q):
        s = (q == qq)
        med = np.nanmedian(f[s])
        if med > 0:
            f[s] = f[s] / med
            ferr[s] = ferr[s] / med
        else:
            f[s] = 1.0

    # Step 3: Positive outlier clipping (cosmic rays / flares)
    # Note: Preserve negative dips (transits) by evaluating positive outliers only!
    res = f - 1.0
    mad = np.nanmedian(np.abs(res))
    std_est = 1.4826 * mad

    if std_est > 0:
        valid_outlier = res < (sigma_clip * std_est)
        t = t[valid_outlier]
        f = f[valid_outlier]
        ferr = ferr[valid_outlier]
        q = q[valid_outlier]

    # Final alignment assertions
    assert len(t) == len(f) == len(ferr) == len(q), "Array length mismatch after outlier clipping!"

    if len(t) < MIN_CADENCES:
        return None, None, None, None

    return t, f, ferr, q
