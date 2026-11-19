"""
Unit Tests for Phase 7: Generalization (LOSO), Latency Benchmarking (NFR-2),
Residual Diagnostics, and System Robustness Stress Testing.
"""

from pathlib import Path
import tempfile
import time
import unittest

import numpy as np
import pandas as pd
import torch

from src.data.synthetic import generate_station_data, save_synthetic_cpcb_file
from src.data.windows import StationScaler
from src.models.diagnostics import (
    compute_residual_autocorrelation,
    extract_worst_periods,
    generate_error_analysis_report,
)
from src.models.loso import run_leave_one_station_out
from src.models.lstm import MultiStationLSTMForecaster
from src.predict import run_forecast_pipeline
from src.train import save_model_package


class TestPhase7RobustnessAndGeneralization(unittest.TestCase):
    """Stress tests and benchmarks the forecasting pipeline."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)
        self.pkg_path = self.tmp_path / "test_model_pkg.pt"

        # Setup standard feature schema
        self.feature_names = [
            "PM2.5", "PM10", "NO2", "AT", "RH", "WS",
            "hour_sin", "hour_cos", "dayofweek_sin", "dayofweek_cos",
            "dayofyear_sin", "dayofyear_cos", "WD_sin", "WD_cos",
            "PM2.5_roll_mean_6h", "PM2.5_roll_mean_24h", "PM2.5_roll_mean_72h",
            "PM10_roll_mean_6h", "PM10_roll_mean_24h", "PM10_roll_mean_72h",
            "pm_ratio", "is_winter_inversion", "is_crop_burning", "is_monsoon",
        ]
        self.model = MultiStationLSTMForecaster(
            in_features=len(self.feature_names),
            num_stations=4,
            conv_filters=16,
            bilstm_hidden=16,
            lstm_hidden=16,
        )

        self.scaler = StationScaler(target_cols=["PM2.5", "PM10"])
        dummy_df = pd.DataFrame(
            np.random.uniform(20.0, 100.0, (100, len(self.feature_names))),
            columns=self.feature_names,
        )
        self.scaler.fit(dummy_df)

        save_model_package(
            model=self.model,
            scalers={"Rampur, Korba - CECB": self.scaler},
            station_mapping={"Rampur, Korba - CECB": 0},
            feature_names=self.feature_names,
            target_names=["PM2.5", "PM10"],
            conformal_alpha=1.0,
            metrics={},
            config={
                "data": {
                    "cleaning": {"pm25_max": 1000.0, "pm10_max": 2000.0, "flatline_hours": 6},
                    "window_length_hours": 360,
                    "forecast_horizon_hours": 72,
                }
            },
            output_path=self.pkg_path,
        )

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_nfr2_inference_latency_under_5_seconds(self):
        """NFR-2: Pipeline inference must complete in < 5.0 seconds on standard CPU."""
        df_15d = generate_station_data(
            "Rampur, Korba - CECB",
            start_date="2026-11-01 00:00",
            end_date="2026-11-17 00:00",
            seed=42,
        )
        data_file = self.tmp_path / "latency_test.xlsx"
        save_synthetic_cpcb_file(df_15d, "Rampur, Korba - CECB", data_file)

        from src.data.fetch import load_cpcb_file
        df_loaded, _ = load_cpcb_file(data_file)

        t_start = time.perf_counter()
        payload = run_forecast_pipeline(
            df_raw=df_loaded,
            package_path=self.pkg_path,
            station_name="Rampur, Korba - CECB",
        )
        t_elapsed = time.perf_counter() - t_start

        self.assertLess(t_elapsed, 5.0, f"Inference took {t_elapsed:.2f}s (exceeded 5.0s NFR-2 threshold)!")
        self.assertEqual(payload.y_quantiles.shape, (72, 2, 3))

    def test_robustness_missing_optional_columns(self):
        """Verifies pipeline still works when non-core columns are completely absent."""
        df_synth = generate_station_data(
            "Rampur, Korba - CECB",
            start_date="2026-11-01 00:00",
            end_date="2026-11-17 00:00",
            seed=42,
        )
        # Drop weather and gas columns completely
        cols_to_drop = [c for c in ["Ozone (ug/m3)", "NH3 (ug/m3)", "RF (mm)", "SR (W/m2)"] if c in df_synth.columns]
        df_stripped = df_synth.drop(columns=cols_to_drop)

        data_file = self.tmp_path / "missing_cols.xlsx"
        save_synthetic_cpcb_file(df_stripped, "Rampur, Korba - CECB", data_file)

        from src.data.fetch import load_cpcb_file
        df_loaded, _ = load_cpcb_file(data_file)

        # Must execute without crash
        payload = run_forecast_pipeline(
            df_raw=df_loaded,
            package_path=self.pkg_path,
            station_name="Rampur, Korba - CECB",
        )
        self.assertEqual(payload.y_quantiles.shape, (72, 2, 3))

    def test_robustness_unknown_station_fallback(self):
        """Verifies graceful warning and fallback embedding when station was not seen during training."""
        df_synth = generate_station_data(
            "Unknown Plant, Korba - CECB",
            start_date="2026-11-01 00:00",
            end_date="2026-11-17 00:00",
            seed=42,
        )
        data_file = self.tmp_path / "unknown_stn.xlsx"
        save_synthetic_cpcb_file(df_synth, "Unknown Plant, Korba - CECB", data_file)

        from src.data.fetch import load_cpcb_file
        df_loaded, _ = load_cpcb_file(data_file)

        payload = run_forecast_pipeline(
            df_raw=df_loaded,
            package_path=self.pkg_path,
            station_name="Unknown Plant, Korba - CECB",
        )
        self.assertTrue(any("was not in the model training set" in w for w in payload.warnings))
        self.assertEqual(payload.y_quantiles.shape, (72, 2, 3))

    def test_diagnostics_and_error_report(self):
        """Tests residual autocorrelation and worst period extraction."""
        N = 20
        y_true = np.random.uniform(50.0, 150.0, (N, 72, 2))
        y_pred = y_true + np.random.normal(0, 10.0, (N, 72, 2))
        # Inject one large failure
        y_pred[5, :, :] += 80.0

        worst = extract_worst_periods(y_true, y_pred, n_worst=5)
        self.assertEqual(len(worst), 5)
        # Sample #5 should be rank 1
        self.assertIn("5", worst[0]["period"])

        residuals = y_pred - y_true
        r_lag1 = compute_residual_autocorrelation(residuals, lag=1)
        r_lag24 = compute_residual_autocorrelation(residuals, lag=24)

        report_file = self.tmp_path / "test_report.md"
        report_text = generate_error_analysis_report(worst, r_lag1, r_lag24, report_file)
        self.assertTrue(report_file.exists())
        self.assertIn("Failure Mode Analysis", report_text)
        self.assertIn("Top 10 Worst Prediction Periods", report_text)


if __name__ == "__main__":
    unittest.main()
