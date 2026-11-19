"""
Unit Tests for Phase 6: Streamlit Visualization Components, PDF Bulletin Export,
and End-to-End Operational Forecast Pipeline.
"""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from app.components.charts import create_72h_forecast_plot, create_training_loss_plot
from app.components.pdf_export import generate_forecast_pdf
from src.aqi import DailyAQIResult
from src.data.synthetic import generate_station_data, save_synthetic_cpcb_file
from src.data.windows import StationScaler
from src.models.lstm import MultiStationLSTMForecaster
from src.precautions import DailyAdvisory, HourlySpikeAlert
from src.predict import run_forecast_pipeline, validate_input_data
from src.train import save_model_package


class TestPhase6AppComponents(unittest.TestCase):
    """Tests visualization and export flow."""

    def test_plotly_forecast_chart_generation(self):
        timestamps = pd.date_range("2026-11-20 00:00", periods=72, freq="1h")
        # Quantiles: (72, 2, 3)
        y_q = np.ones((72, 2, 3)) * 50.0
        y_q[:, :, 0] = 30.0  # q10
        y_q[:, :, 1] = 50.0  # q50
        y_q[:, :, 2] = 80.0  # q90

        fig = create_72h_forecast_plot(
            future_timestamps=timestamps,
            y_quantiles=y_q,
            target_name="PM2.5",
            target_idx=0,
            station_name="Rampur, Korba - CECB",
        )
        self.assertIsNotNone(fig)
        self.assertGreater(len(fig.data), 0)

        # Test loss plot
        history = {"train_loss": [0.5, 0.4, 0.3], "val_loss": [0.6, 0.5, 0.35]}
        loss_fig = create_training_loss_plot(history)
        self.assertIsNotNone(loss_fig)
        self.assertEqual(len(loss_fig.data), 2)

    def test_pdf_export_generation(self):
        daily_res = [
            DailyAQIResult(
                day_name="Day+1",
                hour_range=(0, 24),
                pm25_mean=45.0,
                pm10_mean=90.0,
                pm25_sub_index=75,
                pm10_sub_index=90,
                overall_aqi=90,
                category="Satisfactory",
                color="#82E0AA",
                dominant_pollutant="PM10",
            )
        ]
        advisories = [
            DailyAdvisory(
                day_name="Day+1",
                aqi=90,
                category="Satisfactory",
                color="#82E0AA",
                pm25_mean=45.0,
                pm10_mean=90.0,
                dominant_pollutant="PM10",
                general_public_advice="Normal outdoor activity is acceptable.",
                sensitive_groups_advice="Unusually sensitive individuals should monitor.",
                disclaimer="This is general guidance, not medical advice.",
                language="en",
            )
        ]
        spike = HourlySpikeAlert(
            has_spike=False,
            peak_hour=0,
            peak_pollutant="None",
            peak_concentration=0.0,
            peak_category="Normal",
            alert_title="",
            alert_message="",
            affected_hours_count=0,
        )

        pdf_bytes = generate_forecast_pdf(
            station_name="Rampur, Korba - CECB",
            daily_results=daily_res,
            advisories=advisories,
            spike_alert=spike,
        )

        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 1000)
        # Verify valid PDF header
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))

    def test_end_to_end_forecast_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            pkg_path = Path(tmp_dir) / "test_pkg.pt"
            data_path = Path(tmp_dir) / "test_15d.xlsx"

            # 1. Create a dummy model package with realistic schema
            feature_names = [
                "PM2.5", "PM10", "NO2", "AT", "RH", "WS",
                "hour_sin", "hour_cos", "dayofweek_sin", "dayofweek_cos",
                "dayofyear_sin", "dayofyear_cos", "WD_sin", "WD_cos",
                "PM2.5_roll_mean_6h", "PM2.5_roll_mean_24h", "PM2.5_roll_mean_72h",
                "PM10_roll_mean_6h", "PM10_roll_mean_24h", "PM10_roll_mean_72h",
                "pm_ratio", "is_winter_inversion", "is_crop_burning", "is_monsoon",
            ]
            model = MultiStationLSTMForecaster(
                in_features=len(feature_names),
                num_stations=4,
                conv_filters=16,
                bilstm_hidden=16,
                lstm_hidden=16,
            )

            # Fit a dummy scaler
            scaler = StationScaler(target_cols=["PM2.5", "PM10"])
            dummy_train_df = pd.DataFrame(
                np.random.uniform(20.0, 100.0, (100, len(feature_names))),
                columns=feature_names,
            )
            scaler.fit(dummy_train_df)

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
                output_path=pkg_path,
            )

            # 2. Generate 16 days of 15-minute readings (>= 360 hours)
            df_synth = generate_station_data(
                "Rampur, Korba - CECB",
                start_date="2026-11-01 00:00",
                end_date="2026-11-17 00:00",
                seed=42,
            )
            save_synthetic_cpcb_file(df_synth, "Rampur, Korba - CECB", data_path)

            from src.data.fetch import load_cpcb_file
            df_loaded, _ = load_cpcb_file(data_path)

            # 3. Run full operational forecast
            payload = run_forecast_pipeline(
                df_raw=df_loaded,
                package_path=pkg_path,
                station_name="Rampur, Korba - CECB",
                lang="en",
            )

            self.assertEqual(payload.y_quantiles.shape, (72, 2, 3))
            self.assertEqual(len(payload.future_timestamps), 72)
            self.assertEqual(len(payload.daily_results), 3)
            self.assertEqual(len(payload.advisories), 3)
            self.assertIsNotNone(payload.spike_alert)


if __name__ == "__main__":
    unittest.main()
