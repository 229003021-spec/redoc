"""
Machine Learning classification and confidence score calibration module.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from .config import SDE_BASELINE_THRESHOLD


FEATURE_COLS = [
    "sde", "snr", "period", "depth_ppm", "duration_hours",
    "n_transits_expected", "n_in_transit_points",
    "odd_even_ratio", "odd_even_diff_sig", "sec_depth_ratio",
    "quarter_recurrence", "scatter_ppm", "depth_to_scatter", "radius_ratio"
]


def heuristic_confidence_from_sde(sde: float, midpoint: float = SDE_BASELINE_THRESHOLD, steepness: float = 0.4) -> float:
    """
    Sigmoidal squash mapping raw SDE to (0, 1) probability range.
    """
    if np.isnan(sde):
        return 0.0
    return float(1.0 / (1.0 + np.exp(-steepness * (sde - midpoint))))


class ExoplanetCandidateClassifier:
    """
    Machine learning classifier for exoplanet candidate vetting.
    """

    def __init__(self, model_type: str = "hist_gb"):
        self.model_type = model_type
        if model_type == "hist_gb":
            base_model = HistGradientBoostingClassifier(random_state=42, max_iter=100, max_depth=4)
        else:
            base_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=6)

        # Apply CalibratedClassifierCV for probability calibration
        self.model = CalibratedClassifierCV(estimator=base_model, cv=3)
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """
        Trains calibrated candidate classifier.
        """
        X_feats = X[FEATURE_COLS].copy().fillna(0.0)
        self.model.fit(X_feats, y)
        self.is_fitted = True

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts calibrated probability P(transiting planet).
        """
        if not self.is_fitted:
            # Fallback to heuristic sigmoid SDE mapping if ML not fitted
            return np.array([heuristic_confidence_from_sde(s) for s in X["sde"]])

        X_feats = X[FEATURE_COLS].copy().fillna(0.0)
        probas = self.model.predict_proba(X_feats)[:, 1]
        return probas
