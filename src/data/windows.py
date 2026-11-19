"""
Chronological Splitting, Scaling, and Sliding Window Generator.
Implements DP-8 (Scaling) and DP-9 (Window Construction).
Guarantees strict zero-leakage by fitting scalers strictly on training data
and inserting buffer gaps between train, validation, and test splits.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

from src.utils.config import load_config

logger = logging.getLogger(__name__)


class StationScaler:
    """
    Per-station scaler applying log1p transformation to skewed concentrations
    followed by standard scaling (zero mean, unit variance) fit strictly on training data.
    Provides inverse transformation for multi-quantile target forecasts.
    """

    def __init__(self, target_cols: List[str] = ["PM2.5", "PM10"]):
        self.target_cols = list(target_cols)
        self.mean_: Dict[str, float] = {}
        self.std_: Dict[str, float] = {}
        self.log_cols_: List[str] = []
        self.feature_names_: List[str] = []
        self.is_fitted: bool = False

    def _get_skewed_columns(self, columns: List[str]) -> List[str]:
        """Identifies concentration columns that require log1p stabilization."""
        skewed_patterns = ["pm2.5", "pm10", "no", "no2", "so2", "co", "ozone", "nh3", "roll_mean"]
        log_cols = []
        for col in columns:
            lowered = col.lower()
            if any(p in lowered for p in skewed_patterns) and not ("sin" in lowered or "cos" in lowered):
                log_cols.append(col)
        return log_cols

    def fit(self, df_train: pd.DataFrame) -> "StationScaler":
        """Compute training statistics (mean and std) per column."""
        self.feature_names_ = [c for c in df_train.columns if c != "station_id"]
        self.log_cols_ = self._get_skewed_columns(self.feature_names_)

        df_transformed = df_train.copy()
        for col in self.log_cols_:
            if col in df_transformed.columns:
                df_transformed[col] = np.log1p(np.maximum(0.0, df_transformed[col].fillna(0.0)))

        for col in self.feature_names_:
            series = df_transformed[col].dropna()
            m = float(series.mean()) if len(series) > 0 else 0.0
            s = float(series.std()) if len(series) > 1 else 1.0
            if s < 1e-6 or np.isnan(s):
                s = 1.0
            self.mean_[col] = m
            self.std_[col] = s

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms input DataFrame using pre-computed training parameters."""
        if not self.is_fitted:
            raise RuntimeError("StationScaler must be fitted before transform.")

        df_scaled = df.copy()
        # Apply log1p on skewed features
        for col in self.log_cols_:
            if col in df_scaled.columns:
                df_scaled[col] = np.log1p(np.maximum(0.0, df_scaled[col]))

        # Standardize features
        for col in self.feature_names_:
            if col in df_scaled.columns:
                m = self.mean_[col]
                s = self.std_[col]
                df_scaled[col] = (df_scaled[col] - m) / s

        return df_scaled

    def fit_transform(self, df_train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df_train).transform(df_train)

    def inverse_transform_targets(self, y_scaled: np.ndarray) -> np.ndarray:
        """
        Inverts scaled target forecasts back to physical concentration units (ug/m3).
        Supports shapes:
          - (H, n_targets, n_quantiles)
          - (N, H, n_targets)
          - (N, H, n_targets, n_quantiles)
          - (N, n_targets) or (H, n_targets)
        """
        if not self.is_fitted:
            raise RuntimeError("StationScaler must be fitted before inverse transform.")

        y_out = np.copy(y_scaled)
        n_targets = len(self.target_cols)

        for i, target in enumerate(self.target_cols):
            m = self.mean_.get(target, 0.0)
            s = self.std_.get(target, 1.0)
            
            # Select target slice based on shape
            if y_out.ndim == 3 and y_out.shape[1] == n_targets:
                # Shape: (horizon, n_targets, n_quantiles)
                y_out[:, i, :] = y_out[:, i, :] * s + m
                if target in self.log_cols_:
                    y_out[:, i, :] = np.expm1(y_out[:, i, :])
            elif y_out.ndim == 3 and y_out.shape[2] == n_targets:
                # Shape: (N, horizon, n_targets)
                y_out[:, :, i] = y_out[:, :, i] * s + m
                if target in self.log_cols_:
                    y_out[:, :, i] = np.expm1(y_out[:, :, i])
            elif y_out.ndim == 4 and y_out.shape[2] == n_targets:
                # Shape: (N, horizon, n_targets, n_quantiles)
                y_out[:, :, i, :] = y_out[:, :, i, :] * s + m
                if target in self.log_cols_:
                    y_out[:, :, i, :] = np.expm1(y_out[:, :, i, :])
            elif y_out.ndim == 2 and y_out.shape[1] == n_targets:
                # Shape: (N, n_targets)
                y_out[:, i] = y_out[:, i] * s + m
                if target in self.log_cols_:
                    y_out[:, i] = np.expm1(y_out[:, i])
            else:
                raise ValueError(f"Unexpected shape {y_out.shape} for inverse_transform_targets with {n_targets} targets")

        # Physical concentrations must be strictly non-negative
        return np.maximum(0.0, y_out)

    def save(self, filepath: Union[str, Path]) -> None:
        """Serializes scaler to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "StationScaler":
        """Loads serialized scaler from disk."""
        return joblib.load(filepath)


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    buffer_hours: int = 72,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits time series into Train (70%), Validation (15%), and Test (15%) partitions.
    Inserts `buffer_hours` (72h) blackout margin between splits to prevent overlapping
    sliding windows from leaking future target information into training.
    """
    n = len(df)
    train_end_idx = int(n * train_ratio)
    val_start_idx = train_end_idx + buffer_hours

    val_end_idx = train_end_idx + int(n * val_ratio)
    test_start_idx = val_end_idx + buffer_hours

    if val_start_idx >= val_end_idx or test_start_idx >= n:
        # Fallback for short dataset / pipeline test mode: reduce buffer proportionally
        buffer_hours = max(1, int(n * 0.02))
        val_start_idx = train_end_idx + buffer_hours
        val_end_idx = train_end_idx + int(n * val_ratio)
        test_start_idx = val_end_idx + buffer_hours

    df_train = df.iloc[:train_end_idx].copy()
    df_val = df.iloc[val_start_idx:val_end_idx].copy()
    df_test = df.iloc[test_start_idx:].copy()

    logger.info(
        f"Chronological split with {buffer_hours}h blackout margins: "
        f"Train={len(df_train)}h, Val={len(df_val)}h, Test={len(df_test)}h"
    )
    return df_train, df_val, df_test


def create_sliding_windows(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_cols: List[str] = ["PM2.5", "PM10"],
    window_len: int = 360,
    horizon: int = 72,
    stride: int = 1,
    station_id: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Constructs sliding window tensors:
        X: (N, window_len, n_features)
        station_ids: (N,)
        Y: (N, horizon, n_targets)
    Excludes any window spanning missing values or discontinuous time jumps.
    """
    total_len = window_len + horizon
    n_rows = len(df)
    if n_rows < total_len:
        return np.empty((0, window_len, len(feature_cols))), np.empty((0,), dtype=int), np.empty((0, horizon, len(target_cols)))

    feat_vals = df[feature_cols].to_numpy(dtype=np.float32)
    targ_vals = df[target_cols].to_numpy(dtype=np.float32)
    timestamps = df.index

    x_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    stn_list: List[int] = []

    for i in range(0, n_rows - total_len + 1, stride):
        # Check temporal continuity: exactly 1-hour interval across the window
        dt = (timestamps[i + total_len - 1] - timestamps[i]).total_seconds() / 3600.0
        expected_dt = total_len - 1
        if abs(dt - expected_dt) > 0.1:
            # Time gap detected; skip window
            continue

        x_slice = feat_vals[i : i + window_len]
        y_slice = targ_vals[i + window_len : i + total_len]

        # Ensure no NaNs in window
        if np.isnan(x_slice).any() or np.isnan(y_slice).any():
            continue

        x_list.append(x_slice)
        y_list.append(y_slice)
        stn_list.append(station_id)

    if not x_list:
        return np.empty((0, window_len, len(feature_cols))), np.empty((0,), dtype=int), np.empty((0, horizon, len(target_cols)))

    X = np.stack(x_list, axis=0)
    Y = np.stack(y_list, axis=0)
    stn_arr = np.array(stn_list, dtype=np.int64)

    return X, stn_arr, Y
