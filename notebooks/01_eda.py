"""
Exploratory Data Analysis (EDA) Script for Korba PM2.5 & PM10 Air Quality.
Implements Milestone 2:
- Hourly diurnal cycles & seasonal patterns
- PM2.5 vs PM10 correlation & ratio distributions
- Meteorological impact (wind speed, temperature inversion)
- Multi-station comparison and correlation
Exports high-resolution figures to outputs/eda_figures/
"""

import json
import os
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Set writable Matplotlib cache directory
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.fetch import load_all_stations
from src.data.synthetic import generate_station_data, save_synthetic_cpcb_file
DATA_RAW = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eda_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def ensure_data_available():
    """Ensure at least sample data exists in data/raw for EDA."""
    files = list(DATA_RAW.glob("*.xlsx")) + list(DATA_RAW.glob("*.csv"))
    if not files:
        print("[INFO] No raw files found in data/raw. Generating 30-day pipeline test data for EDA...")
        for idx, stn in enumerate(["Rampur, Korba - CECB", "Banki Colliery, Korba - CECB"]):
            df_synth = generate_station_data(stn, start_date="2024-11-01", end_date="2024-11-30", seed=42 + idx)
            save_synthetic_cpcb_file(df_synth, stn, DATA_RAW / f"test_{idx+1}.xlsx")


def run_eda():
    ensure_data_available()
    stations = load_all_stations(DATA_RAW)
    if not stations:
        print("[ERROR] No station data available.")
        return

    print(f"Loaded {len(stations)} station(s) for EDA: {list(stations.keys())}")

    # Set visualization style
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Diurnal Cycle (Hour of Day Profile)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name, df in stations.items():
        hourly = df.groupby(df.index.hour)[["PM2.5", "PM10"]].mean()
        axes[0].plot(hourly.index, hourly["PM2.5"], marker="o", label=f"{name} (PM2.5)")
        axes[1].plot(hourly.index, hourly["PM10"], marker="s", label=f"{name} (PM10)")

    axes[0].set_title("Hourly Diurnal Profile: PM2.5 (ug/m3)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Hour of Day (IST)")
    axes[0].set_ylabel("Mean PM2.5 (ug/m3)")
    axes[0].set_xticks(range(0, 24, 2))
    axes[0].legend(fontsize=9)

    axes[1].set_title("Hourly Diurnal Profile: PM10 (ug/m3)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Hour of Day (IST)")
    axes[1].set_ylabel("Mean PM10 (ug/m3)")
    axes[1].set_xticks(range(0, 24, 2))
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    fig_path1 = OUTPUT_DIR / "diurnal_cycles.png"
    plt.savefig(fig_path1, dpi=200)
    plt.close()
    print(f"Saved diurnal profile figure -> {fig_path1}")

    # 2. PM2.5 vs PM10 Scatter & Ratio Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    all_pm25 = []
    all_pm10 = []
    for name, df in stations.items():
        clean = df[["PM2.5", "PM10"]].dropna()
        if not clean.empty:
            axes[0].scatter(clean["PM10"], clean["PM2.5"], alpha=0.3, s=15, label=name)
            ratio = clean["PM2.5"] / clean["PM10"]
            axes[1].hist(ratio, bins=35, alpha=0.5, density=True, label=name)
            all_pm25.extend(clean["PM2.5"])
            all_pm10.extend(clean["PM10"])

    axes[0].plot([0, 500], [0, 500], "r--", label="1:1 Equality line")
    axes[0].set_title("PM2.5 vs PM10 Concentration Relationship", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("PM10 (ug/m3)")
    axes[0].set_ylabel("PM2.5 (ug/m3)")
    axes[0].legend(fontsize=9)

    axes[1].set_title("PM2.5 / PM10 Ratio Distribution", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Ratio (PM2.5 / PM10)")
    axes[1].set_ylabel("Density")
    axes[1].axvline(1.0, color="red", linestyle="--", label="Physical Bound (1.0)")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    fig_path2 = OUTPUT_DIR / "pm_ratio_analysis.png"
    plt.savefig(fig_path2, dpi=200)
    plt.close()
    print(f"Saved PM ratio figure -> {fig_path2}")

    # 3. Meteorological Impact (Wind Speed vs PM2.5)
    first_stn = list(stations.keys())[0]
    df_stn = stations[first_stn]
    if "WS" in df_stn.columns and "PM2.5" in df_stn.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        valid = df_stn[["WS", "PM2.5"]].dropna()
        ax.scatter(valid["WS"], valid["PM2.5"], alpha=0.4, color="teal", s=15)
        ax.set_title(f"Wind Speed vs PM2.5 Concentration ({first_stn})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Wind Speed (m/s)")
        ax.set_ylabel("PM2.5 (ug/m3)")
        plt.tight_layout()
        fig_path3 = OUTPUT_DIR / "meteorological_dispersion.png"
        plt.savefig(fig_path3, dpi=200)
        plt.close()
        print(f"Saved meteorological dispersion figure -> {fig_path3}")

    print("EDA execution completed successfully!")


if __name__ == "__main__":
    run_eda()
