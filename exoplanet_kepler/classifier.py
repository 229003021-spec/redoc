"""
Machine Learning classification and confidence score calibration module.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
from .config import SDE_BASELINE_THRESHOLD


FEATURE_COLS = [
    "sde", "snr", "period", "depth_ppm", "duration_hours",
    "n_transits_expected", "n_in_transit_points",
    "odd_even_ratio", "odd_even_diff_sig", "sec_depth_ratio", "sec_depth_sig",
    "quarter_recurrence", "v_shape_metric", "local_snr", "residual_sde",
    "scatter_ppm", "scatter_mad_ppm",
    "depth_to_scatter", "radius_ratio",
    "alias_ratio_half", "alias_ratio_double", "alias_ratio_triple"
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

    def __init__(self, model_type: str = "hist_gb", random_state: int = 42):
        self.model_type = model_type
        self.random_state = random_state

        if model_type == "hist_gb":
            base_model = HistGradientBoostingClassifier(random_state=random_state, max_iter=100, max_depth=4)
        elif model_type == "rf":
            base_model = RandomForestClassifier(n_estimators=100, random_state=random_state, max_depth=6, class_weight="balanced")
        elif model_type == "extra_trees":
            base_model = ExtraTreesClassifier(n_estimators=100, random_state=random_state, max_depth=6, class_weight="balanced")
        else:
            base_model = HistGradientBoostingClassifier(random_state=random_state, max_iter=100, max_depth=4)

        # Apply CalibratedClassifierCV explicitly using Platt scaling (method="sigmoid")
        self.model = CalibratedClassifierCV(estimator=base_model, method="sigmoid", cv=5)
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """
        Trains calibrated candidate classifier.
        """
        # Select existing feature columns available in X
        avail_cols = [col for col in FEATURE_COLS if col in X.columns]
        X_feats = X[avail_cols].copy().fillna(0.0)
        self.model.fit(X_feats, y)
        self.is_fitted = True

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts calibrated probability P(transiting planet).
        Raises RuntimeError if model is not fitted (zero silent fallbacks).
        """
        if not self.is_fitted:
            raise RuntimeError("ExoplanetCandidateClassifier is not fitted! Call fit() before predict_proba().")

        avail_cols = [col for col in FEATURE_COLS if col in X.columns]
        X_feats = X[avail_cols].copy().fillna(0.0)
        probas = self.model.predict_proba(X_feats)[:, 1]
        return probas
