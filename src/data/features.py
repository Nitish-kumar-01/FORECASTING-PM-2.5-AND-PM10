"""
Feature Engineering Module for PM2.5 & PM10 Forecasting.
Implements DP-7:
- Cyclical temporal transforms (hour, weekday, day of year)
- Wind direction circular decomposition (sin, cos)
- Rolling aggregations (6h, 24h, 72h) of PM2.5 and PM10
- PM2.5 / PM10 ratio
- Station ID categorical encoding
- Environmental and seasonal indicator flags
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from src.utils.config import load_config


def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes continuous sine and cosine encodings for time cycles."""
    df_feat = df.copy()
    idx = df_feat.index

    # Hour of day (period = 24)
    hour = idx.hour
    df_feat["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    df_feat["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)

    # Day of week (period = 7)
    dayofweek = getattr(idx, "day_of_week", getattr(idx, "dayofweek", None))
    df_feat["dayofweek_sin"] = np.sin(2.0 * np.pi * dayofweek / 7.0)
    df_feat["dayofweek_cos"] = np.cos(2.0 * np.pi * dayofweek / 7.0)

    # Day of year (period = 365.25)
    doy = getattr(idx, "day_of_year", getattr(idx, "dayofyear", None))
    df_feat["dayofyear_sin"] = np.sin(2.0 * np.pi * doy / 365.25)
    df_feat["dayofyear_cos"] = np.cos(2.0 * np.pi * doy / 365.25)

    return df_feat


def add_wind_components(df: pd.DataFrame) -> pd.DataFrame:
    """Converts circular Wind Direction (degrees) to Cartesian vector components."""
    df_feat = df.copy()
    if "WD" in df_feat.columns:
        rads = np.deg2rad(df_feat["WD"].fillna(0.0))
        df_feat["WD_sin"] = np.sin(rads)
        df_feat["WD_cos"] = np.cos(rads)
    else:
        df_feat["WD_sin"] = 0.0
        df_feat["WD_cos"] = 0.0
    return df_feat


def add_rolling_features(df: pd.DataFrame, windows: List[int] = [6, 24, 72]) -> pd.DataFrame:
    """
    Computes backward-looking rolling means for PM2.5 and PM10.
    Strictly past-oriented (no future data leakage).
    """
    df_feat = df.copy()
    for target in ["PM2.5", "PM10"]:
        if target in df_feat.columns:
            for w in windows:
                col_name = f"{target}_roll_mean_{w}h"
                df_feat[col_name] = df_feat[target].rolling(window=w, min_periods=1).mean()
    return df_feat


def add_particulate_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Computes PM2.5 / PM10 concentration ratio with numerical stability."""
    df_feat = df.copy()
    if "PM2.5" in df_feat.columns and "PM10" in df_feat.columns:
        df_feat["pm_ratio"] = df_feat["PM2.5"] / (df_feat["PM10"] + 1e-4)
        # Clip ratio to physical bounds [0.0, 1.5]
        df_feat["pm_ratio"] = df_feat["pm_ratio"].clip(0.0, 1.5)
    else:
        df_feat["pm_ratio"] = 0.5
    return df_feat


def add_seasonal_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds binary flags for distinct regional emission/meteorology regimes:
    - Winter Inversion (Nov - Feb): Highest stagnation and PM peaks
    - Crop Burning / Post-Monsoon (Oct 15 - Nov 30): Stubble burning spike
    - Monsoon Washout (Jun 15 - Sep 15): Heavy wet deposition
    """
    df_feat = df.copy()
    idx = df_feat.index
    month = idx.month
    day = idx.day

    # Winter Inversion
    df_feat["is_winter_inversion"] = ((month >= 11) | (month <= 2)).astype(float)

    # Crop burning season
    is_oct_late = (month == 10) & (day >= 15)
    is_nov = (month == 11)
    df_feat["is_crop_burning"] = (is_oct_late | is_nov).astype(float)

    # Monsoon washout
    is_monsoon = (month >= 6) & (month <= 9)
    df_feat["is_monsoon"] = is_monsoon.astype(float)

    return df_feat


def engineer_station_features(
    df_hourly: pd.DataFrame,
    station_id: int = 0,
    config: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Orchestrates full feature engineering pipeline for a single station's hourly data.
    """
    if config is None:
        config = load_config()

    df_feat = df_hourly.copy()

    # 1. Temporal cyclical features
    df_feat = add_cyclical_time_features(df_feat)

    # 2. Wind direction vector components
    df_feat = add_wind_components(df_feat)

    # 3. Rolling past aggregates (6h, 24h, 72h)
    df_feat = add_rolling_features(df_feat, windows=[6, 24, 72])

    # 4. PM2.5 / PM10 ratio
    df_feat = add_particulate_ratio(df_feat)

    # 5. Seasonal indicators
    df_feat = add_seasonal_flags(df_feat)

    # 6. Station embedding ID
    df_feat["station_id"] = int(station_id)

    # Drop raw Wind Direction (WD) if replaced by vector components
    if "WD" in df_feat.columns:
        df_feat = df_feat.drop(columns=["WD"])

    # Drop boolean flag columns from clean stage before model feature selection
    cols_to_drop = [c for c in ["is_physically_inconsistent", "was_imputed"] if c in df_feat.columns]
    if cols_to_drop:
        df_feat = df_feat.drop(columns=cols_to_drop)

    return df_feat


def get_model_feature_columns(df_engineered: pd.DataFrame) -> List[str]:
    """
    Returns the ordered list of input feature names excluding station_id and targets
    for the temporal sequence encoder.
    """
    excluded = {"station_id"}
    return [col for col in df_engineered.columns if col not in excluded]
