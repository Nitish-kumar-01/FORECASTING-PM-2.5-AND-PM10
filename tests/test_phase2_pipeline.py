"""
Unit Tests for Phase 2: Data Cleaning, Imputation, Feature Engineering,
Scaling, and Window Construction Pipeline.
"""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.data.clean import (
    clean_raw_station_data,
    compute_circular_mean_wind,
    convert_to_hourly,
    flag_physical_inconsistencies,
    impute_missing_values,
)
from src.data.features import (
    add_cyclical_time_features,
    add_particulate_ratio,
    add_rolling_features,
    engineer_station_features,
    get_model_feature_columns,
)
from src.data.windows import (
    StationScaler,
    chronological_split,
    create_sliding_windows,
)


class TestPhase2Cleaning(unittest.TestCase):
    """Tests DP-3, DP-4, DP-5, and DP-6 data cleaning logic."""

    def test_negative_and_extreme_value_removal(self):
        dates = pd.date_range("2026-01-01 00:00", periods=5, freq="15min")
        df = pd.DataFrame({
            "PM2.5": [50.0, -10.0, 1500.0, 60.0, 80.0],
            "PM10": [100.0, -5.0, 2500.0, 120.0, 150.0],
        }, index=dates)

        cleaned = clean_raw_station_data(df)
        self.assertTrue(np.isnan(cleaned.iloc[1]["PM2.5"]))  # negative removed
        self.assertTrue(np.isnan(cleaned.iloc[2]["PM2.5"]))  # >1000 removed
        self.assertTrue(np.isnan(cleaned.iloc[1]["PM10"]))   # negative removed
        self.assertTrue(np.isnan(cleaned.iloc[2]["PM10"]))   # >2000 removed
        self.assertEqual(cleaned.iloc[0]["PM2.5"], 50.0)

    def test_circular_mean_wind_direction(self):
        # 350 deg and 10 deg: arithmetic mean = 180 deg (South - incorrect!)
        # Circular mean = 0 deg (North - correct!)
        wd_series = pd.Series([350.0, 350.0, 10.0, 10.0])
        circ_mean = compute_circular_mean_wind(wd_series, min_valid=3)
        self.assertTrue(circ_mean < 5.0 or circ_mean > 355.0, f"Expected near 0/360, got {circ_mean}")

    def test_hourly_conversion_and_min_valid(self):
        dates = pd.date_range("2026-01-01 00:00", periods=8, freq="15min")
        # Hour 0 has 4 valid readings (>=3): should have an hourly mean
        # Hour 1 has only 1 valid reading (<3): should be NaN
        df = pd.DataFrame({
            "PM2.5": [10.0, 20.0, 30.0, 40.0, 50.0, np.nan, np.nan, np.nan],
            "WD": [0.0, 90.0, 180.0, 270.0, 45.0, np.nan, np.nan, np.nan],
        }, index=dates)

        hourly = convert_to_hourly(df, min_intervals=3)
        self.assertEqual(len(hourly), 2)
        self.assertFalse(np.isnan(hourly.iloc[0]["PM2.5"]))
        self.assertEqual(hourly.iloc[0]["PM2.5"], 25.0)
        self.assertTrue(np.isnan(hourly.iloc[1]["PM2.5"]))

    def test_imputation_and_tracking(self):
        dates = pd.date_range("2026-01-01 00:00", periods=10, freq="1h")
        # 2-hour gap (short <=3h: linear), then 6-hour gap (medium 4-24h: diurnal)
        vals = [10.0, np.nan, np.nan, 40.0, np.nan, np.nan, np.nan, np.nan, np.nan, 100.0]
        df_hourly = pd.DataFrame({"PM2.5": vals}, index=dates)

        imputed, was_imputed = impute_missing_values(df_hourly)
        self.assertTrue(was_imputed.iloc[1]["PM2.5"])
        self.assertTrue(was_imputed.iloc[2]["PM2.5"])
        self.assertFalse(np.isnan(imputed.iloc[1]["PM2.5"]))
        self.assertAlmostEqual(imputed.iloc[1]["PM2.5"], 20.0, places=1)

    def test_physical_inconsistency_flag(self):
        dates = pd.date_range("2026-01-01 00:00", periods=3, freq="1h")
        df = pd.DataFrame({
            "PM2.5": [50.0, 150.0, 80.0],
            "PM10": [100.0, 100.0, 120.0],
        }, index=dates)
        flagged = flag_physical_inconsistencies(df, ratio_threshold=1.10)
        self.assertFalse(flagged.iloc[0]["is_physically_inconsistent"])
        self.assertTrue(flagged.iloc[1]["is_physically_inconsistent"])  # 150 > 1.10 * 100


class TestPhase2FeaturesAndWindows(unittest.TestCase):
    """Tests DP-7, DP-8, and DP-9 feature engineering and windowing."""

    def test_feature_engineering_pipeline(self):
        dates = pd.date_range("2026-01-01 00:00", periods=100, freq="1h")
        df = pd.DataFrame({
            "PM2.5": np.random.uniform(20.0, 100.0, 100),
            "PM10": np.random.uniform(50.0, 200.0, 100),
            "WD": np.random.uniform(0.0, 360.0, 100),
            "AT": np.random.uniform(15.0, 35.0, 100),
        }, index=dates)

        feat_df = engineer_station_features(df, station_id=1)
        self.assertIn("hour_sin", feat_df.columns)
        self.assertIn("hour_cos", feat_df.columns)
        self.assertIn("WD_sin", feat_df.columns)
        self.assertIn("WD_cos", feat_df.columns)
        self.assertIn("PM2.5_roll_mean_6h", feat_df.columns)
        self.assertIn("PM2.5_roll_mean_24h", feat_df.columns)
        self.assertIn("pm_ratio", feat_df.columns)
        self.assertIn("is_winter_inversion", feat_df.columns)
        self.assertEqual(feat_df["station_id"].iloc[0], 1)

        cols = get_model_feature_columns(feat_df)
        self.assertNotIn("station_id", cols)
        self.assertIn("PM2.5", cols)
        self.assertIn("hour_sin", cols)

    def test_station_scaler_inversion(self):
        dates = pd.date_range("2026-01-01 00:00", periods=50, freq="1h")
        df = pd.DataFrame({
            "PM2.5": np.random.uniform(25.0, 200.0, 50),
            "PM10": np.random.uniform(60.0, 400.0, 50),
            "AT": np.random.uniform(15.0, 35.0, 50),
        }, index=dates)

        scaler = StationScaler(target_cols=["PM2.5", "PM10"])
        df_scaled = scaler.fit_transform(df)

        # Scaled mean should be near 0 and std near 1
        self.assertAlmostEqual(df_scaled["PM2.5"].mean(), 0.0, places=1)
        self.assertAlmostEqual(df_scaled["PM2.5"].std(), 1.0, places=1)

        # Test inverse transform on targets
        dummy_targets = df_scaled[["PM2.5", "PM10"]].to_numpy().reshape(1, 50, 2)
        recovered = scaler.inverse_transform_targets(dummy_targets)
        self.assertEqual(recovered.shape, (1, 50, 2))
        np.testing.assert_allclose(recovered[0, :, 0], df["PM2.5"].to_numpy(), rtol=1e-4)

    def test_sliding_window_construction(self):
        # 500 hours: window 360, horizon 72, stride 1 -> 500 - (360 + 72) + 1 = 69 windows
        dates = pd.date_range("2026-01-01 00:00", periods=500, freq="1h")
        df = pd.DataFrame({
            "PM2.5": np.random.uniform(30.0, 150.0, 500),
            "PM10": np.random.uniform(80.0, 300.0, 500),
            "AT": np.random.uniform(15.0, 35.0, 500),
        }, index=dates)

        features = ["PM2.5", "PM10", "AT"]
        X, stn, Y = create_sliding_windows(
            df,
            feature_cols=features,
            target_cols=["PM2.5", "PM10"],
            window_len=360,
            horizon=72,
            stride=1,
            station_id=2,
        )

        expected_n = 500 - (360 + 72) + 1
        self.assertEqual(X.shape, (expected_n, 360, 3))
        self.assertEqual(Y.shape, (expected_n, 72, 2))
        self.assertEqual(len(stn), expected_n)
        self.assertEqual(stn[0], 2)

    def test_chronological_split_buffers(self):
        dates = pd.date_range("2026-01-01 00:00", periods=1000, freq="1h")
        df = pd.DataFrame({"PM2.5": np.ones(1000)}, index=dates)

        train, val, test = chronological_split(df, train_ratio=0.70, val_ratio=0.15, buffer_hours=72)
        # Verify no overlapping indices
        train_max = train.index.max()
        val_min = val.index.min()
        val_max = val.index.max()
        test_min = test.index.min()

        self.assertGreater((val_min - train_max).total_seconds() / 3600.0, 71.0)
        self.assertGreater((test_min - val_max).total_seconds() / 3600.0, 71.0)


if __name__ == "__main__":
    unittest.main()
