"""
Automated Unit Test Suite for Kepler Exoplanet Detection Pipeline.
Run with: python -m unittest tests/test_pipeline.py
"""

import unittest
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exoplanet_kepler.cleaning import clean_lightcurve
from exoplanet_kepler.detrending import detrend_lightcurve
from exoplanet_kepler.vetting import check_secondary_eclipse, check_odd_even_consistency, check_quarter_recurrence
from exoplanet_kepler.classifier import ExoplanetCandidateClassifier, heuristic_confidence_from_sde


class TestExoplanetPipeline(unittest.TestCase):

    def test_clean_lightcurve_array_alignment(self):
        """Tests that clean_lightcurve returns perfectly aligned arrays."""
        n_points = 2000
        t_raw = np.linspace(0, 100, n_points)
        f_raw = np.ones(n_points) + np.random.normal(0, 0.001, n_points)
        f_raw[50] = 10.0  # Cosmic ray spike outlier
        ferr_raw = np.full(n_points, 0.001)
        q_raw = np.zeros(n_points, dtype=int)
        quarter_raw = np.full(n_points, 1, dtype=int)

        df = pd.DataFrame({
            "time": t_raw,
            "flux": f_raw,
            "flux_err": ferr_raw,
            "quality": q_raw,
            "quarter": quarter_raw
        })

        t, f, ferr, q = clean_lightcurve(df)
        self.assertIsNotNone(t)
        self.assertEqual(len(t), len(f))
        self.assertEqual(len(t), len(ferr))
        self.assertEqual(len(t), len(q))
        # Cosmic ray spike should be clipped
        self.assertTrue(np.max(f) < 1.05)

    def test_detrend_array_alignment(self):
        """Tests that detrend_lightcurve maintains exact array lengths across all outputs."""
        n_points = 2000
        t = np.linspace(0, 100, n_points)
        f = np.ones(n_points)
        ferr = np.full(n_points, 0.001)
        q = np.full(n_points, 1, dtype=int)

        t_c, f_c, ferr_c, q_c, trend = detrend_lightcurve(t, f, ferr, q, method="savgol")
        self.assertEqual(len(t_c), len(f_c))
        self.assertEqual(len(t_c), len(ferr_c))
        self.assertEqual(len(t_c), len(q_c))
        self.assertEqual(len(t_c), len(trend))

    def test_secondary_eclipse_phase_wrapping(self):
        """Tests that check_secondary_eclipse correctly detects phase 0.5 eclipses without boundary issues."""
        t = np.linspace(0, 100, 5000)
        period = 10.0
        t0 = 2.0
        duration_hours = 4.0

        # Create flux with primary eclipse at phase 0.0 (depth 2000 ppm) and secondary at phase 0.5 (depth 1000 ppm)
        f = np.ones_like(t)
        phase = ((t - t0) / period) % 1.0

        # Primary dip around phase 0.0
        pri_mask = (phase < 0.01) | (phase > 0.99)
        f[pri_mask] -= 0.002

        # Secondary dip around phase 0.5
        sec_mask = np.abs(phase - 0.5) < 0.01
        f[sec_mask] -= 0.001

        sec_ratio = check_secondary_eclipse(t, f, period, t0, duration_hours)
        self.assertGreater(sec_ratio, 0.3)
        self.assertLess(sec_ratio, 0.7)

    def test_odd_even_consistency(self):
        """Tests odd/even transit depth ratio calculation."""
        t = np.linspace(0, 100, 5000)
        period = 10.0
        t0 = 2.0
        duration_hours = 4.0

        f = np.ones_like(t)
        phase = ((t - t0) / period) % 1.0
        in_transit = (phase < 0.01) | (phase > 0.99)

        # Equal transit depths for odd and even
        f[in_transit] -= 0.002

        ratio, diff_sig = check_odd_even_consistency(t, f, period, t0, duration_hours)
        self.assertAlmostEqual(ratio, 1.0, delta=0.2)
        self.assertLess(diff_sig, 3.0)

    def test_quarter_recurrence(self):
        """Tests that quarter recurrence requires valid in-transit points."""
        t = np.linspace(0, 359, 5000)
        q = (t // 90).astype(int)
        period = 10.0
        t0 = 2.0
        duration_hours = 4.0

        f = np.ones_like(t)
        phase = ((t - t0) / period) % 1.0
        in_transit = (phase < 0.01) | (phase > 0.99)
        f[in_transit] -= 0.002

        rec = check_quarter_recurrence(t, f, q, period, t0, duration_hours)
        self.assertEqual(rec, 1.0)

    def test_unfitted_classifier_raises_error(self):
        """Tests that calling predict_proba on unfitted classifier raises RuntimeError (zero silent fallbacks)."""
        clf = ExoplanetCandidateClassifier()
        df_dummy = pd.DataFrame({"sde": [12.0, 5.0]})
        with self.assertRaises(RuntimeError):
            clf.predict_proba(df_dummy)


if __name__ == "__main__":
    unittest.main()
