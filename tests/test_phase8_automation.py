"""
Unit Tests for Phase 8: Automation, Daily Scheduled Inference,
and Headless Output Export (FR-15).
"""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.data.synthetic import generate_station_data, save_synthetic_cpcb_file
from src.data.windows import StationScaler
from src.models.lstm import MultiStationLSTMForecaster
from src.run_daily import execute_daily_forecast
from src.train import save_model_package


class TestPhase8Automation(unittest.TestCase):
    """Tests automated daily batch execution and output exports."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)
        self.pkg_path = self.tmp_path / "model_package.pt"
        self.data_path = self.tmp_path / "station_15d.xlsx"
        self.out_base = self.tmp_path / "test_outputs"

        feature_names = [
            "PM2.5", "PM10", "NO2", "AT", "RH", "WS",
            "hour_sin", "hour_cos", "dayofweek_sin", "dayofweek_cos",
            "dayofyear_sin", "dayofyear_cos", "WD_sin", "WD_cos",
            "PM2.5_roll_mean_6h", "PM2.5_roll_mean_24h", "PM2.5_roll_mean_72h",
            "PM10_roll_mean_6h", "PM10_roll_mean_24h", "PM10_roll_mean_72h",
            "pm_ratio", "is_winter_inversion", "is_crop_burning", "is_monsoon",
        ]

        # Model and Scaler
        model = MultiStationLSTMForecaster(
            in_features=len(feature_names),
            num_stations=4,
            conv_filters=16,
            bilstm_hidden=16,
            lstm_hidden=16,
        )
        scaler = StationScaler(target_cols=["PM2.5", "PM10"])
        dummy_df = pd.DataFrame(
            np.random.uniform(20.0, 100.0, (100, len(feature_names))),
            columns=feature_names,
        )
        scaler.fit(dummy_df)

        save_model_package(
            model=model,
            scalers={"Rampur, Korba - CECB": scaler},
            station_mapping={"Rampur, Korba - CECB": 0},
            feature_names=feature_names,
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

        # Generate 16 days of 15-minute readings
        df_16d = generate_station_data(
            "Rampur, Korba - CECB",
            start_date="2026-11-01 00:00",
            end_date="2026-11-17 00:00",
            seed=42,
        )
        save_synthetic_cpcb_file(df_16d, "Rampur, Korba - CECB", self.data_path)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_automated_daily_forecast_execution(self):
        """Verifies full execution of headless daily forecast script."""
        result = execute_daily_forecast(
            data_path=self.data_path,
            package_path=self.pkg_path,
            station_name="Rampur, Korba - CECB",
            output_base_dir=self.out_base,
            lang="en",
        )

        csv_path = Path(result["csv_path"])
        json_path = Path(result["json_path"])
        pdf_path = Path(result["pdf_path"])

        # Check all artifacts were generated
        self.assertTrue(csv_path.exists())
        self.assertTrue(json_path.exists())
        self.assertTrue(pdf_path.exists())

        # Inspect CSV format
        df_csv = pd.read_csv(csv_path)
        self.assertEqual(len(df_csv), 72)
        self.assertIn("Timestamp_IST", df_csv.columns)
        self.assertIn("PM2.5_median_ug_m3", df_csv.columns)
        self.assertIn("PM10_median_ug_m3", df_csv.columns)

        # Inspect PDF
        pdf_bytes = pdf_path.read_bytes()
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))


if __name__ == "__main__":
    unittest.main()
