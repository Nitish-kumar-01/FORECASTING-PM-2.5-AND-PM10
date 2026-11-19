"""
Data Cleaning and Aggregation Module for CPCB / CECB Air Quality Data.
Implements DP-3 (Cleaning), DP-4 (Hourly conversion), DP-5 (Missing values), and DP-6 (Consistency).
"""

from dataclasses import dataclass
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.config import load_config

logger = logging.getLogger(__name__)


def clean_raw_station_data(
    df: pd.DataFrame,
    config: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Implements DP-3:
    - Filters negative values to NaN.
    - Filters impossible values (PM2.5 > 1000, PM10 > 2000) to NaN.
    - Identifies stuck sensors (flatlines: >= 24 consecutive 15-min intervals or 6 hours)
      and marks them as NaN.
    """
    if config is None:
        config = load_config()

    clean_cfg = config["data"]["cleaning"]
    pm25_max = clean_cfg.get("pm25_max", 1000.0)
    pm10_max = clean_cfg.get("pm10_max", 2000.0)
    flatline_intervals = clean_cfg.get("flatline_hours", 6) * 4  # 4 intervals per hour

    df_cleaned = df.copy()

    # 1. Negative value removal
    numeric_cols = [c for c in df_cleaned.columns if np.issubdtype(df_cleaned[c].dtype, np.number)]
    for col in numeric_cols:
        neg_mask = df_cleaned[col] < 0.0
        if neg_mask.any():
            df_cleaned.loc[neg_mask, col] = np.nan

    # 2. Impossible value clipping for PM
    if "PM2.5" in df_cleaned.columns:
        df_cleaned.loc[df_cleaned["PM2.5"] > pm25_max, "PM2.5"] = np.nan
    if "PM10" in df_cleaned.columns:
        df_cleaned.loc[df_cleaned["PM10"] > pm10_max, "PM10"] = np.nan

    # 3. Flatline / Stuck Sensor Detection
    for target in ["PM2.5", "PM10"]:
        if target in df_cleaned.columns:
            series = df_cleaned[target]
            # Identify consecutive identical values
            is_same = series == series.shift(1)
            # Group consecutive runs
            run_id = (~is_same).cumsum()
            run_lengths = series.groupby(run_id).transform("count")
            
            # Mask runs exceeding flatline threshold (ignoring NaN)
            flatline_mask = (run_lengths >= flatline_intervals) & series.notna()
            if flatline_mask.any():
                logger.info(f"Detected {flatline_mask.sum()} flatlined {target} readings; setting to NaN.")
                df_cleaned.loc[flatline_mask, target] = np.nan

    return df_cleaned


def compute_circular_mean_wind(wd_series: pd.Series, min_valid: int = 3) -> float:
    """
    Calculates circular mean for wind direction in degrees [0, 360).
    Requires at least `min_valid` readings.
    """
    valid = wd_series.dropna()
    if len(valid) < min_valid:
        return np.nan
    rads = np.deg2rad(valid)
    sin_sum = np.sum(np.sin(rads))
    cos_sum = np.sum(np.cos(rads))
    mean_rad = np.arctan2(sin_sum, cos_sum)
    mean_deg = np.rad2deg(mean_rad) % 360.0
    return float(round(mean_deg, 1))


def convert_to_hourly(
    df: pd.DataFrame,
    min_intervals: int = 3,
) -> pd.DataFrame:
    """
    Implements DP-4:
    - Averages four 15-minute readings per hour.
    - Requires at least `min_intervals` (3 of 4) non-null values per hour.
    - Uses circular mean (sine and cosine vector components) for Wind Direction (WD).
    """
    numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    regular_cols = [c for c in numeric_cols if c != "WD"]

    # 1. Regular columns: mean with count threshold
    grouped = df[regular_cols].resample("1h")
    counts = grouped.count()
    means = grouped.mean()
    hourly_df = means.where(counts >= min_intervals)

    # 2. Wind Direction: circular mean
    if "WD" in df.columns:
        wd_resampled = df["WD"].resample("1h").apply(
            lambda s: compute_circular_mean_wind(s, min_valid=min_intervals)
        )
        hourly_df["WD"] = wd_resampled

    return hourly_df


def impute_missing_values(
    df_hourly: pd.DataFrame,
    config: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Implements DP-5:
    - Linear interpolation for small gaps (<= 3 hours).
    - Time-of-day seasonal (diurnal) interpolation for medium gaps (4 to 24 hours).
    - Large gaps (> 24 hours) remain NaN.
    - Generates a matching boolean DataFrame `was_imputed`.
    """
    if config is None:
        config = load_config()

    clean_cfg = config["data"]["cleaning"]
    short_gap = clean_cfg.get("short_gap_hours", 3)
    medium_gap = clean_cfg.get("medium_gap_hours", 24)

    df_imputed = df_hourly.copy()
    was_imputed = pd.DataFrame(False, index=df_hourly.index, columns=df_hourly.columns)

    for col in df_imputed.columns:
        series = df_imputed[col].copy()
        is_na = series.isna()

        if not is_na.any():
            continue

        # Find consecutive NaN blocks
        na_groups = (~is_na).cumsum()
        gap_sizes = is_na.groupby(na_groups).transform("sum")

        # 1. Short gaps (<= 3h): linear interpolation
        short_mask = is_na & (gap_sizes <= short_gap)
        if short_mask.any():
            interpolated = series.interpolate(method="time", limit=short_gap)
            series.loc[short_mask] = interpolated.loc[short_mask]
            was_imputed.loc[short_mask, col] = True

        # 2. Medium gaps (4 to 24h): diurnal (time-of-day) interpolation
        medium_mask = series.isna() & (gap_sizes <= medium_gap) & (gap_sizes > short_gap)
        if medium_mask.any():
            # Calculate diurnal profile (mean per hour of day)
            hourly_profile = series.groupby(series.index.hour).transform("mean")
            # If still missing (e.g. all values for that hour were missing), fallback to overall mean
            hourly_profile = hourly_profile.fillna(series.mean())
            series.loc[medium_mask] = hourly_profile.loc[medium_mask]
            was_imputed.loc[medium_mask, col] = True

        df_imputed[col] = series

    return df_imputed, was_imputed


def flag_physical_inconsistencies(
    df: pd.DataFrame,
    ratio_threshold: float = 1.10,
) -> pd.DataFrame:
    """
    Implements DP-6:
    Flags hours where PM2.5 > 1.10 * PM10 (physically suspicious).
    Adds a boolean column 'is_physically_inconsistent'.
    """
    df_flagged = df.copy()
    if "PM2.5" in df_flagged.columns and "PM10" in df_flagged.columns:
        both_valid = df_flagged["PM2.5"].notna() & df_flagged["PM10"].notna()
        condition = both_valid & (df_flagged["PM2.5"] > (ratio_threshold * df_flagged["PM10"]))
        df_flagged["is_physically_inconsistent"] = condition
    else:
        df_flagged["is_physically_inconsistent"] = False
    return df_flagged


def run_cleaning_pipeline(
    df_raw: pd.DataFrame,
    config: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes the complete cleaning pipeline from 15-minute raw to clean hourly data.
    Returns:
        df_clean_hourly: Cleaned, imputed, hourly DataFrame.
        was_imputed: Boolean DataFrame indicating imputed cells.
    """
    df_cleaned_raw = clean_raw_station_data(df_raw, config=config)
    df_hourly = convert_to_hourly(df_cleaned_raw)
    df_imputed, was_imputed = impute_missing_values(df_hourly, config=config)
    df_final = flag_physical_inconsistencies(df_imputed)
    return df_final, was_imputed
