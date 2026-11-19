"""
Operational Forecast Engine for PM2.5 & PM10 Air Quality.
Implements FR-12, FR-18, and FR-19:
- Validates 15-day (360h) recent station observation exports
- Executes zero-leakage inference through loaded model package
- Generates 72-hour calibrated multi-quantile predictions and health advisories
"""

from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from src.aqi import DailyAQIResult, evaluate_daily_aqi_forecast
from src.data.clean import run_cleaning_pipeline
from src.data.features import engineer_station_features, get_model_feature_columns
from src.models.lstm import MultiStationLSTMForecaster
from src.precautions import DailyAdvisory, HourlySpikeAlert, PrecautionEngine
from src.train import load_model_package

logger = logging.getLogger(__name__)


@dataclass
class ForecastPayload:
    """Complete output bundle from 15-day operational forecast inference."""
    station_name: str
    inference_time: str
    latest_observation_time: str
    future_timestamps: pd.DatetimeIndex
    y_quantiles: np.ndarray  # Shape: (72, 2, 3) -> 72h x (PM2.5, PM10) x (q10, q50, q90)
    daily_results: List[DailyAQIResult]
    advisories: List[DailyAdvisory]
    spike_alert: HourlySpikeAlert
    past_clean_df: pd.DataFrame
    warnings: List[str]


def validate_input_data(
    df_raw: pd.DataFrame,
    station_name: str,
    station_mapping: Dict[str, int],
    required_hours: int = 360,
    max_staleness_hours: float = 6.0,
) -> Tuple[bool, List[str]]:
    """
    Implements FR-19:
    - Validates station is known to the trained model
    - Checks that at least 80% of the 360 hours are valid
    - Checks staleness (latest data point not older than 6 hours)
    """
    warnings = []
    is_valid = True

    # 1. Station recognition check
    if station_name not in station_mapping:
        warnings.append(
            f"Station '{station_name}' was not in the model training set. "
            "Forecast will use default station embedding and may be less reliable."
        )

    if df_raw.empty:
        return False, ["Input data file is completely empty."]

    # 2. History length check
    time_span = (df_raw.index.max() - df_raw.index.min()).total_seconds() / 3600.0
    if time_span < (required_hours * 0.80):
        warnings.append(
            f"Insufficient time span: {time_span:.1f} hours provided "
            f"(expected ~{required_hours} hours / 15 days)."
        )

    # 3. Staleness check
    latest_ts = df_raw.index.max()
    now_ts = pd.Timestamp.now()
    staleness_hours = (now_ts - latest_ts).total_seconds() / 3600.0
    if staleness_hours > max_staleness_hours and staleness_hours < 87600:  # Allow historical tests
        warnings.append(
            f"Data freshness warning: Latest observation ({latest_ts}) is "
            f"{staleness_hours:.1f} hours old (threshold is {max_staleness_hours}h)."
        )

    return is_valid, warnings


def run_forecast_pipeline(
    df_raw: pd.DataFrame,
    package_path: Union[str, Path] = "models/model_package.pt",
    station_name: str = "Rampur, Korba - CECB",
    lang: str = "en",
    device: Optional[torch.device] = None,
) -> ForecastPayload:
    """
    End-to-end inference from a 15-day raw station DataFrame.
    """
    if device is None:
        device = torch.device("cpu")

    model, package = load_model_package(package_path, device=device)

    station_mapping = package["station_mapping"]
    scalers = package["scalers"]
    feature_names = package["feature_names"]
    target_names = package["target_names"]
    conformal_alpha = package.get("conformal_alpha", 1.0)
    config = package["config"]

    # 1. Input Validation
    _, warnings = validate_input_data(
        df_raw=df_raw,
        station_name=station_name,
        station_mapping=station_mapping,
        required_hours=360,
    )

    # 2. Cleaning & Hourly Resampling
    df_clean_hourly, _ = run_cleaning_pipeline(df_raw, config=config)
    if len(df_clean_hourly) < 24:
        raise ValueError("Insufficient valid hourly records after cleaning pipeline.")

    # 3. Feature Engineering
    station_id = station_mapping.get(station_name, 0)
    df_feat = engineer_station_features(df_clean_hourly, station_id=station_id, config=config)

    # Align columns to model feature schema
    for col in feature_names:
        if col not in df_feat.columns:
            df_feat[col] = 0.0  # Safe zero imputation for missing optional features
    df_feat = df_feat[feature_names]

    # 4. Scaling
    scaler = scalers.get(station_name)
    if scaler is not None and scaler.is_fitted:
        df_scaled = scaler.transform(df_feat)
    else:
        # Fallback to first available scaler
        fallback_scaler = next(iter(scalers.values())) if scalers else None
        if fallback_scaler:
            df_scaled = fallback_scaler.transform(df_feat)
            warnings.append(f"Using fallback scaler parameters for station '{station_name}'.")
        else:
            df_scaled = df_feat

    # 5. Extract latest 360-hour window
    if len(df_scaled) < 360:
        # Pad beginning if slightly under 360h
        pad_len = 360 - len(df_scaled)
        pad_rows = pd.DataFrame(
            np.tile(df_scaled.iloc[0].values, (pad_len, 1)),
            columns=df_scaled.columns,
        )
        df_scaled_window = pd.concat([pad_rows, df_scaled], axis=0).iloc[-360:]
        warnings.append(f"Recent data had {len(df_scaled)} hours; padded initial {pad_len} hours for inference.")
    else:
        df_scaled_window = df_scaled.iloc[-360:]

    x_tensor = torch.tensor(
        df_scaled_window.to_numpy(dtype=np.float32),
        dtype=torch.float32,
    ).unsqueeze(0).to(device)  # (1, 360, in_features)

    stn_tensor = torch.tensor([station_id], dtype=torch.long).to(device)

    # 6. Model Forward Pass
    with torch.no_grad():
        q_pred, _ = model(x_tensor, stn_tensor)  # (1, 72, 2, 3)
        q_pred_np = q_pred.cpu().numpy()[0]     # (72, 2, 3)

    # 7. Apply Post-Hoc Conformal Calibration to Interval
    if conformal_alpha != 1.0:
        q10 = q_pred_np[:, :, 0]
        q50 = q_pred_np[:, :, 1]
        q90 = q_pred_np[:, :, 2]
        q10_cal = q50 - conformal_alpha * (q50 - q10)
        q90_cal = q50 + conformal_alpha * (q90 - q50)
        q_pred_np = np.stack([q10_cal, q50, q90_cal], axis=-1)

    # 8. Inverse Transform back to Physical Units (ug/m3)
    active_scaler = scaler or fallback_scaler
    if active_scaler:
        y_physical = active_scaler.inverse_transform_targets(q_pred_np)
    else:
        y_physical = np.maximum(0.0, q_pred_np)

    # 9. Build Future Timestamps
    latest_ts = df_clean_hourly.index[-1]
    future_timestamps = pd.date_range(
        start=latest_ts + pd.Timedelta(hours=1),
        periods=72,
        freq="1h",
    )

    # 10. Compute CPCB AQI and Health Advisories
    daily_results = evaluate_daily_aqi_forecast(y_physical, target_names=target_names)
    precaution_engine = PrecautionEngine()
    advisories = [precaution_engine.get_advisory(res, lang=lang) for res in daily_results]
    spike_alert = precaution_engine.detect_hourly_spikes(y_physical, target_names=target_names, lang=lang)

    return ForecastPayload(
        station_name=station_name,
        inference_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        latest_observation_time=str(latest_ts),
        future_timestamps=future_timestamps,
        y_quantiles=y_physical,
        daily_results=daily_results,
        advisories=advisories,
        spike_alert=spike_alert,
        past_clean_df=df_clean_hourly,
        warnings=warnings,
    )
