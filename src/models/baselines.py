"""
Baseline Forecasters for PM2.5 & PM10 Air Quality.
Implements PRD Section 7.3:
1. Persistence Forecaster (repeat last 24 hours across 72h horizon)
2. Seasonal Naive Forecaster (same hour of previous day / 7-day diurnal cycle)
3. Tabular GBDT Baseline (HistGradientBoosting / LightGBM on lag and rolling features)
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


class PersistenceForecaster:
    """
    Persistence Forecaster:
    Takes the observed concentration readings from the last 24 hours of the input window
    (timesteps 336 to 360) and tiles/repeats them 3 times across the 72-hour forecast horizon
    (Day+1, Day+2, Day+3).
    """

    def __init__(self, horizon: int = 72, past_repeat_window: int = 24):
        self.horizon = horizon
        self.past_repeat_window = past_repeat_window

    def predict(
        self,
        X: np.ndarray,
        target_indices: List[int] = [0, 1],
    ) -> np.ndarray:
        """
        Args:
            X: Input tensor of shape (N, window_len, n_features)
            target_indices: Feature indices in X corresponding to targets [PM2.5, PM10]
        Returns:
            Y_pred: Forecast array of shape (N, horizon, len(target_indices))
        """
        N, window_len, _ = X.shape
        # Extract last 24 hours of targets: shape (N, 24, n_targets)
        last_24h = X[:, window_len - self.past_repeat_window : window_len, :][:, :, target_indices]
        
        # Tile across 72 hours (repeat 3 times)
        repeats = int(np.ceil(self.horizon / self.past_repeat_window))
        tiled = np.tile(last_24h, (1, repeats, 1))
        return tiled[:, : self.horizon, :]


class SeasonalNaiveForecaster:
    """
    Seasonal Naive Forecaster:
    Averages the hourly concentration profile for each hour of the day across
    the historical past window (15 days / 360 hours) and projects that diurnal
    curve forward across the 72-hour forecast horizon.
    """

    def __init__(self, horizon: int = 72, day_len: int = 24):
        self.horizon = horizon
        self.day_len = day_len

    def predict(
        self,
        X: np.ndarray,
        target_indices: List[int] = [0, 1],
    ) -> np.ndarray:
        """
        Args:
            X: Input tensor of shape (N, window_len, n_features)
            target_indices: Feature indices in X corresponding to targets [PM2.5, PM10]
        Returns:
            Y_pred: Forecast array of shape (N, horizon, len(target_indices))
        """
        N, window_len, _ = X.shape
        target_history = X[:, :, target_indices]  # (N, window_len, n_targets)
        
        # Calculate mean across 24-hour periods for each station sequence
        num_full_days = window_len // self.day_len
        usable_history = target_history[:, window_len - num_full_days * self.day_len :, :]
        # Reshape to (N, num_full_days, 24, n_targets)
        reshaped = usable_history.reshape(N, num_full_days, self.day_len, len(target_indices))
        # Diurnal profile: shape (N, 24, n_targets)
        diurnal_profile = np.mean(reshaped, axis=1)

        # Tile diurnal profile across 72 hours
        repeats = int(np.ceil(self.horizon / self.day_len))
        tiled = np.tile(diurnal_profile, (1, repeats, 1))
        return tiled[:, : self.horizon, :]


class TabularGBDTBaseline:
    """
    Multi-Horizon Tabular Gradient Boosted Decision Tree (LightGBM) Baseline.
    Extracts summary statistics from past window and trains distinct regressors
    for Day+1, Day+2, and Day+3 forecast blocks for PM2.5 and PM10.
    """

    def __init__(
        self,
        horizon: int = 72,
        target_names: List[str] = ["PM2.5", "PM10"],
        n_estimators: int = 100,
        random_state: int = 42,
    ):
        self.horizon = horizon
        self.target_names = list(target_names)
        self.n_estimators = n_estimators
        self.random_state = random_state
        # Models dictionary: {(target_idx, horizon_group): lgb.LGBMRegressor}
        # Horizon groups: 0 (Day+1: 1-24h), 1 (Day+2: 25-48h), 2 (Day+3: 49-72h)
        self.models: Dict[Tuple[int, int], Any] = {}

    def _extract_tabular_features(self, X: np.ndarray) -> np.ndarray:
        """
        Extracts summary features from historical sequences:
        - Latest observation (t-1)
        - Mean, min, max, std over last 6 hours
        - Mean, min, max, std over last 24 hours
        - Mean, min, max, std over full 360 hours
        """
        N, window_len, n_features = X.shape
        feat_list = []

        # Latest values
        feat_list.append(X[:, -1, :])  # (N, n_features)

        # Last 6 hours stats
        x_6h = X[:, -6:, :]
        feat_list.append(np.mean(x_6h, axis=1))
        feat_list.append(np.std(x_6h, axis=1))
        feat_list.append(np.max(x_6h, axis=1))
        feat_list.append(np.min(x_6h, axis=1))

        # Last 24 hours stats
        x_24h = X[:, -24:, :]
        feat_list.append(np.mean(x_24h, axis=1))
        feat_list.append(np.std(x_24h, axis=1))

        # Full 360h stats
        feat_list.append(np.mean(X, axis=1))
        feat_list.append(np.std(X, axis=1))

        return np.concatenate(feat_list, axis=1)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "TabularGBDTBaseline":
        """
        Trains LightGBM regressors for Day+1, Day+2, and Day+3 target averages.
        Args:
            X: (N, window_len, n_features)
            Y: (N, horizon, n_targets)
        """
        X_tab = self._extract_tabular_features(X)
        N, horizon, n_targets = Y.shape

        day_slices = [
            (0, 24),   # Day+1
            (24, 48),  # Day+2
            (48, 72),  # Day+3
        ]

        for t_idx in range(n_targets):
            for d_idx, (start_h, end_h) in enumerate(day_slices):
                # Target is daily average for that block
                y_day = np.mean(Y[:, start_h:end_h, t_idx], axis=1)

                model = HistGradientBoostingRegressor(
                    max_iter=self.n_estimators,
                    random_state=self.random_state,
                )
                model.fit(X_tab, y_day)
                self.models[(t_idx, d_idx)] = model

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generates 72-hour forecast by projecting Day+1, Day+2, Day+3 predictions.
        Returns:
            Y_pred: (N, horizon, n_targets)
        """
        X_tab = self._extract_tabular_features(X)
        N = len(X)
        n_targets = len(self.target_names)
        Y_pred = np.zeros((N, self.horizon, n_targets), dtype=np.float32)

        day_slices = [
            (0, 24),
            (24, 48),
            (48, 72),
        ]

        for t_idx in range(n_targets):
            for d_idx, (start_h, end_h) in enumerate(day_slices):
                model = self.models.get((t_idx, d_idx))
                if model is not None:
                    pred_day = model.predict(X_tab)  # (N,)
                    # Broadcast daily mean across the 24 hours of that day
                    Y_pred[:, start_h:end_h, t_idx] = np.repeat(pred_day[:, np.newaxis], end_h - start_h, axis=1)

        return Y_pred
