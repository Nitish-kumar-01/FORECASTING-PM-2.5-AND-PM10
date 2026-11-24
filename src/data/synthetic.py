"""
Synthetic Air Quality Data Generator for Pipeline Testing.
Implements PRD Section 5.4:
"A synthetic data generator may be used only to test the pipeline,
must be clearly labelled, and its output is never shown as a real forecast."
"""

from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_station_data(
    station_name: str,
    start_date: str = "2024-01-01 00:00",
    end_date: str = "2025-12-31 23:45",
    freq: str = "15min",
    seed: int = 42,
    missing_rate: float = 0.02,
) -> pd.DataFrame:
    """
    Generates realistic synthetic 15-minute multi-pollutant and weather data
    simulating Korba industrial atmospheric conditions.
    """
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start=start_date, end=end_date, freq=freq)
    n = len(timestamps)

    # Time factors
    hours = timestamps.hour + timestamps.minute / 60.0
    day_of_year = timestamps.day_of_year if hasattr(timestamps, "day_of_year") else timestamps.dayofyear
    
    # 1. Weather Simulation (Korba: hot summers, wet monsoons, dry/cool winters)
    # Ambient Temperature: Annual range (15 to 42 C), Diurnal range (8 to 15 C)
    seasonal_temp = 28.0 - 8.0 * np.cos(2 * np.pi * (day_of_year - 140) / 365.25)
    diurnal_temp = 5.0 * np.sin(2 * np.pi * (hours - 9) / 24.0)
    at = seasonal_temp + diurnal_temp + rng.normal(0, 1.5, n)
    at = np.clip(at, 5.0, 48.0)

    # Relative Humidity: Inverse to temperature, high during monsoon (July-Sept: doy 180-260)
    monsoon_boost = 30.0 * np.exp(-((day_of_year - 220) / 40.0) ** 2)
    diurnal_rh = -15.0 * np.sin(2 * np.pi * (hours - 9) / 24.0)
    rh = 50.0 + monsoon_boost + diurnal_rh + rng.normal(0, 3.0, n)
    rh = np.clip(rh, 15.0, 98.0)

    # Wind Speed: 0.5 to 6 m/s, higher in afternoon
    ws = 1.8 + 1.2 * np.sin(2 * np.pi * (hours - 8) / 24.0) + rng.exponential(0.8, n)
    ws = np.clip(ws, 0.2, 12.0)

    # Wind Direction: 0 to 360 deg, prevailing seasonal winds
    base_wd = 45.0 + 180.0 * (day_of_year > 150) * (day_of_year < 270)
    wd = (base_wd + rng.normal(0, 35.0, n)) % 360.0

    # Solar Radiation: Daylight only (6 AM to 6 PM)
    daylight = np.sin(np.pi * np.clip(hours - 6, 0, 12) / 12.0)
    cloud_cover = (1.0 - monsoon_boost / 60.0)
    sr = 800.0 * daylight * cloud_cover + rng.normal(0, 20.0, n)
    sr = np.clip(sr, 0.0, 1100.0)

    # 2. Particulate Matter Simulation (PM2.5 & PM10)
    # Seasonal inversion factor (winter high Nov-Feb: doy > 300 or doy < 60)
    winter_factor = 1.0 + 0.8 * (np.cos(2 * np.pi * (day_of_year - 15) / 365.25) > 0.5)
    # Diurnal twin peaks (morning 7-9 AM, evening 8-10 PM)
    diurnal_pm = (
        0.35 * np.exp(-((hours - 8.0) / 2.0) ** 2) +
        0.50 * np.exp(-((hours - 21.0) / 2.5) ** 2) +
        0.20
    )
    # Industrial base pollution + episodic coal/industrial spikes
    industrial_base = 55.0 * winter_factor * (1.0 + diurnal_pm)
    spikes = rng.choice([0.0, 1.0], size=n, p=[0.985, 0.015]) * rng.uniform(40.0, 180.0, n)
    
    pm25 = industrial_base + spikes + rng.normal(0, 8.0, n)
    pm25 = np.clip(pm25, 5.0, 600.0)

    # PM10 is correlated with PM2.5 (ratio typically PM2.5/PM10 ~ 0.40 - 0.65 in mining/dust areas)
    pm10 = pm25 * rng.uniform(1.6, 2.3, n) + rng.exponential(25.0, n)
    pm10 = np.clip(pm10, pm25 * 1.05, 1200.0)

    # 3. Gaseous Pollutants
    no2 = 25.0 * winter_factor * (1.0 + 0.3 * np.sin(2 * np.pi * (hours - 8) / 24.0)) + rng.normal(0, 4.0, n)
    no2 = np.clip(no2, 2.0, 180.0)

    no = no2 * 0.4 + rng.exponential(5.0, n)
    no = np.clip(no, 1.0, 100.0)

    so2 = 18.0 + 12.0 * winter_factor + rng.normal(0, 5.0, n)
    so2 = np.clip(so2, 1.0, 120.0)

    co = 0.8 + 0.5 * winter_factor + 0.3 * diurnal_pm + rng.normal(0, 0.15, n)
    co = np.clip(co, 0.1, 8.0)

    ozone = 20.0 + 35.0 * (sr / 800.0) + rng.normal(0, 5.0, n)
    ozone = np.clip(ozone, 2.0, 160.0)

    nh3 = rng.uniform(6.0, 9.0, n)
    rf = np.zeros(n)
    # Rain in monsoon
    rain_mask = (day_of_year > 165) & (day_of_year < 260) & (rng.uniform(0, 1, n) < 0.04)
    rf[rain_mask] = rng.exponential(4.5, np.sum(rain_mask))

    # Construct DataFrame
    from_dates = timestamps.strftime("%d-%m-%Y %H:%M")
    to_dates = (timestamps + pd.Timedelta(minutes=15)).strftime("%d-%m-%Y %H:%M")

    df = pd.DataFrame({
        "From Date": from_dates,
        "To Date": to_dates,
        "PM2.5 (ug/m3)": np.round(pm25, 2),
        "PM10 (ug/m3)": np.round(pm10, 2),
        "NO (ug/m3)": np.round(no, 2),
        "NO2 (ug/m3)": np.round(no2, 2),
        "SO2 (ug/m3)": np.round(so2, 2),
        "CO (mg/m3)": np.round(co, 2),
        "Ozone (ug/m3)": np.round(ozone, 2),
        "RH (%)": np.round(rh, 1),
        "AT (C)": np.round(at, 1),
        "WS (m/s)": np.round(ws, 2),
        "WD (deg)": np.round(wd, 1),
        "SR (W/m2)": np.round(sr, 1),
        "NH3 (ug/m3)": np.round(nh3, 1),
        "RF (mm)": np.round(rf, 1),
    })

    # Inject realistic missing values and sensor flatlines
    if missing_rate > 0:
        val_cols = [c for c in df.columns if c not in ["From Date", "To Date"]]
        # Random missing values
        for col in val_cols:
            mask = rng.uniform(0, 1, n) < missing_rate
            df.loc[mask, col] = np.nan

    return df


def save_synthetic_cpcb_file(
    df: pd.DataFrame,
    station_name: str,
    output_path: Union[str, Path],
    format_type: str = "xlsx",
) -> Path:
    """
    Saves DataFrame wrapped with standard CPCB export header block.
    Clearly marks the file as synthetic test data.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    header_lines = [
        ["State: Chhattisgarh", "", "", "", "", ""],
        ["City: Korba", "", "", "", "", ""],
        [f"Station: {station_name}", "", "", "", "", ""],
        ["AvgPeriod: 15 Minute", "", "", "", "", ""],
        ["NOTE: [SYNTHETIC TEST DATASET - FOR PIPELINE TESTING ONLY - NOT REAL FORECAST]", "", "", "", "", ""],
        ["", "", "", "", "", ""],  # Blank spacer
    ]

    header_df = pd.DataFrame(header_lines)

    if format_type == "csv":
        with open(path, "w", encoding="utf-8") as f:
            for row in header_lines:
                f.write(",".join(row) + "\n")
            df.to_csv(f, index=False)
    else:
        # Write Excel with openpyxl
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            header_df.to_excel(writer, index=False, header=False, startrow=0)
            df.to_excel(writer, index=False, header=True, startrow=len(header_lines))

    logger.info(f"Saved synthetic dataset for '{station_name}' to {path}")
    return path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic CPCB test datasets for Korba stations.")
    parser.add_argument("--stations", nargs="+", default=["Rampur, Korba - CECB", "Banki Colliery, Korba - CECB", "SECL Gevra, Korba - CECB"])
    parser.add_argument("--start", type=str, default="2024-01-01 00:00")
    parser.add_argument("--end", type=str, default="2025-12-31 23:45")
    parser.add_argument("--outdir", type=str, default="data/raw")
    parser.add_argument("--format", choices=["xlsx", "csv"], default="xlsx")
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, stn in enumerate(args.stations):
        fname = f"{stn.split(',')[0].strip().lower().replace(' ', '_')}_synthetic.{args.format}"
        target_path = out_dir / fname
        print(f"Generating synthetic pipeline test data for '{stn}' -> {target_path}...")
        df_stn = generate_station_data(stn, start_date=args.start, end_date=args.end, seed=42 + idx * 100)
        save_synthetic_cpcb_file(df_stn, stn, target_path, format_type=args.format)
    print("Done! All synthetic test datasets generated in", out_dir)

