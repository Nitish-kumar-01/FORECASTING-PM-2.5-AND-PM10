"""
Unit Tests for Phase 3: Baseline Forecasters and Evaluation Framework.
"""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.evaluate import (
    calculate_skill_scores,
    compute_metrics,
    evaluate_forecast,
    run_benchmark_comparison,
)
from src.models.baselines import (
    PersistenceForecaster,
    SeasonalNaiveForecaster,
    TabularGBDTBaseline,
)


class TestPhase3Baselines(unittest.TestCase):
    """Tests baseline model forecasting logic."""

    def setUp(self):
        # Create synthetic window batch: 10 samples, 360 hours, 3 features (PM2.5, PM10, Temp)
        np.random.seed(42)
        self.N = 10
        self.window_len = 360
        self.horizon = 72
        self.n_features = 3
        self.n_targets = 2

        self.X = np.random.uniform(20.0, 150.0, (self.N, self.window_len, self.n_features)).astype(np.float32)
        self.Y = np.random.uniform(20.0, 150.0, (self.N, self.horizon, self.n_targets)).astype(np.float32)

    def test_persistence_forecaster(self):
        forecaster = PersistenceForecaster(horizon=72, past_repeat_window=24)
        pred = forecaster.predict(self.X, target_indices=[0, 1])

        self.assertEqual(pred.shape, (self.N, 72, 2))
        # Verify repetition across Day+1, Day+2, Day+3
        day1 = pred[:, 0:24, :]
        day2 = pred[:, 24:48, :]
        day3 = pred[:, 48:72, :]
        np.testing.assert_allclose(day1, day2)
        np.testing.assert_allclose(day1, day3)

        # Verify it matches the actual last 24 hours of input
        expected_last_24h = self.X[:, -24:, :2]
        np.testing.assert_allclose(day1, expected_last_24h)

    def test_seasonal_naive_forecaster(self):
        forecaster = SeasonalNaiveForecaster(horizon=72, day_len=24)
        pred = forecaster.predict(self.X, target_indices=[0, 1])

        self.assertEqual(pred.shape, (self.N, 72, 2))
        # Verify diurnal repetition across days
        day1 = pred[:, 0:24, :]
        day2 = pred[:, 24:48, :]
        np.testing.assert_allclose(day1, day2)

    def test_tabular_gbdt_baseline(self):
        gbdt = TabularGBDTBaseline(horizon=72, target_names=["PM2.5", "PM10"], n_estimators=10)
        gbdt.fit(self.X, self.Y)
        pred = gbdt.predict(self.X)

        self.assertEqual(pred.shape, (self.N, 72, 2))
        self.assertFalse(np.isnan(pred).any())


class TestPhase3Evaluation(unittest.TestCase):
    """Tests evaluation metrics and benchmark calculation."""

    def test_metric_calculations_known_truth(self):
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        # Perfect prediction
        metrics_perf = compute_metrics(y_true, y_true)
        self.assertEqual(metrics_perf["mae"], 0.0)
        self.assertEqual(metrics_perf["rmse"], 0.0)
        self.assertEqual(metrics_perf["r2"], 1.0)
        self.assertEqual(metrics_perf["bias"], 0.0)

        # Constant offset +2
        y_pred_off = y_true + 2.0
        metrics_off = compute_metrics(y_true, y_pred_off)
        self.assertEqual(metrics_off["mae"], 2.0)
        self.assertEqual(metrics_off["rmse"], 2.0)
        self.assertEqual(metrics_off["bias"], 2.0)

    def test_forecast_evaluation_and_skill_score(self):
        N = 5
        y_true = np.random.uniform(50.0, 100.0, (N, 72, 2))
        y_pers = y_true + np.random.normal(0, 15.0, (N, 72, 2))  # high error
        y_good = y_true + np.random.normal(0, 5.0, (N, 72, 2))   # low error

        eval_pers = evaluate_forecast(y_true, y_pers)
        eval_good = evaluate_forecast(y_true, y_good)

        skills = calculate_skill_scores(eval_good, eval_pers)
        # Good model should have positive skill score over noisy persistence
        for target in ["PM2.5", "PM10"]:
            self.assertGreater(skills[target]["Day+1"], 0.0)
            self.assertGreater(skills[target]["Overall (72h)"], 0.0)

    def test_run_benchmark_comparison_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "baseline_metrics.json"

            X = np.random.uniform(30.0, 100.0, (15, 360, 2)).astype(np.float32)
            Y = np.random.uniform(30.0, 100.0, (15, 72, 2)).astype(np.float32)

            bundle, report_md = run_benchmark_comparison(
                X_test=X,
                Y_test=Y,
                target_names=["PM2.5", "PM10"],
                output_path=out_file,
            )

            self.assertTrue(out_file.exists())
            self.assertIn("evaluations", bundle)
            self.assertIn("Persistence", bundle["evaluations"])
            self.assertIn("LightGBM (GBDT)", bundle["evaluations"])
            self.assertIn("Forecast Benchmark Comparison Table", report_md)


if __name__ == "__main__":
    unittest.main()
