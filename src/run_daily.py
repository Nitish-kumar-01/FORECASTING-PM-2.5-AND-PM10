"""
Automated Daily Refresh and Scheduled Forecasting Script.
Implements FR-15 and NFR-1:
- Can be scheduled via cron or Windows Task Scheduler to run daily
- Ingests recent 15-day station observations
- Generates 72-hour multi-quantile predictions, CPCB AQI classifications, and PDF bulletins
- Emits emergency alerts if dangerous hourly pollution spikes are detected
"""

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from app.components.pdf_export import generate_forecast_pdf
from src.data.fetch import load_cpcb_file
from src.predict import run_forecast_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_daily")


def execute_daily_forecast(
    data_path: Path,
    package_path: Path,
    station_name: Optional[str] = None,
    output_base_dir: Path = PROJECT_ROOT / "outputs",
    lang: str = "en",
) -> Dict[str, Any]:
    """
    Executes automated daily forecasting batch job:
    1. Loads latest 15-day station data
    2. Runs neural quantile inference
    3. Exports CSV predictions, JSON payload, and printable PDF bulletin
    4. Logs emergency alert if extreme pollution spike is detected
    """
    logger.info(f"Starting daily forecast job | Data: {data_path.name} | Package: {package_path.name}")

    if not data_path.exists():
        raise FileNotFoundError(f"Input data file does not exist: {data_path}")
    if not package_path.exists():
        raise FileNotFoundError(f"Trained model package does not exist: {package_path}")

    # 1. Load Data
    df_raw, meta = load_cpcb_file(data_path)
    active_station = station_name or meta.station_name
    logger.info(f"Target Station: '{active_station}' | Observations: {len(df_raw)} records")

    # 2. Run Inference Pipeline
    payload = run_forecast_pipeline(
        df_raw=df_raw,
        package_path=package_path,
        station_name=active_station,
        lang=lang,
    )

    # 3. Create Partitioned Output Directories
    date_str = datetime.now().strftime("%Y%m%d_%H%M")
    station_slug = active_station.split(",")[0].strip().lower().replace(" ", "_")

    forecasts_dir = output_base_dir / "forecasts"
    reports_dir = output_base_dir / "reports"
    alerts_dir = output_base_dir / "alerts"

    forecasts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    alerts_dir.mkdir(parents=True, exist_ok=True)

    # 4. Save CSV Tabular Forecast
    csv_file = forecasts_dir / f"forecast_{station_slug}_{date_str}.csv"
    csv_df = pd.DataFrame({
        "Timestamp_IST": payload.future_timestamps,
        "PM2.5_q10_ug_m3": np.round(payload.y_quantiles[:, 0, 0], 2),
        "PM2.5_median_ug_m3": np.round(payload.y_quantiles[:, 0, 1], 2),
        "PM2.5_q90_ug_m3": np.round(payload.y_quantiles[:, 0, 2], 2),
        "PM10_q10_ug_m3": np.round(payload.y_quantiles[:, 1, 0], 2),
        "PM10_median_ug_m3": np.round(payload.y_quantiles[:, 1, 1], 2),
        "PM10_q90_ug_m3": np.round(payload.y_quantiles[:, 1, 2], 2),
    })
    csv_df.to_csv(csv_file, index=False)
    logger.info(f"Saved CSV hourly predictions -> {csv_file}")

    # 5. Save JSON Forecast Summary
    json_file = forecasts_dir / f"summary_{station_slug}_{date_str}.json"
    summary_data = {
        "station": active_station,
        "inference_timestamp": payload.inference_time,
        "daily_results": [
            {
                "day": res.day_name,
                "pm25_mean": res.pm25_mean,
                "pm10_mean": res.pm10_mean,
                "overall_aqi": res.overall_aqi,
                "category": res.category,
                "dominant_pollutant": res.dominant_pollutant,
            }
            for res in payload.daily_results
        ],
        "spike_alert": {
            "has_spike": payload.spike_alert.has_spike,
            "peak_hour": payload.spike_alert.peak_hour,
            "peak_category": payload.spike_alert.peak_category,
            "peak_pollutant": payload.spike_alert.peak_pollutant,
            "alert_message": payload.spike_alert.alert_message,
        },
        "warnings": payload.warnings,
    }
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    logger.info(f"Saved JSON summary -> {json_file}")

    # 6. Save Official PDF Bulletin
    pdf_file = reports_dir / f"bulletin_{station_slug}_{date_str}.pdf"
    pdf_bytes = generate_forecast_pdf(
        station_name=active_station,
        daily_results=payload.daily_results,
        advisories=payload.advisories,
        spike_alert=payload.spike_alert,
        issue_date=payload.inference_time,
    )
    with open(pdf_file, "wb") as f:
        f.write(pdf_bytes)
    logger.info(f"Saved official PDF bulletin -> {pdf_file}")

    # 7. Handle Extreme Spike Alerts (FR-8)
    if payload.spike_alert.has_spike:
        alert_file = alerts_dir / f"ALERT_{station_slug}_{date_str}.txt"
        alert_msg = (
            f"EMERGENCY PEAK POLLUTION ALERT\n"
            f"Station: {active_station}\n"
            f"Generated: {payload.inference_time}\n"
            f"Title: {payload.spike_alert.alert_title}\n"
            f"Details: {payload.spike_alert.alert_message}\n"
            f"Peak Hour: Hour {payload.spike_alert.peak_hour} ({payload.spike_alert.peak_pollutant})\n"
            f"Category: {payload.spike_alert.peak_category}\n"
        )
        with open(alert_file, "w", encoding="utf-8") as f:
            f.write(alert_msg)
        logger.warning(f"🚨 EXTREME PEAK ALERT LOGGED -> {alert_file}")

    # Log Terminal Summary Bulletin
    logger.info("=" * 60)
    logger.info(f"FORECAST BULLETIN FOR: {active_station}")
    for res in payload.daily_results:
        logger.info(f"  • {res.day_name}: {res.category} (AQI {res.overall_aqi}) | PM2.5: {res.pm25_mean:.1f} ug/m3 | PM10: {res.pm10_mean:.1f} ug/m3")
    logger.info("=" * 60)

    return {
        "csv_path": str(csv_file),
        "json_path": str(json_file),
        "pdf_path": str(pdf_file),
        "payload": payload,
    }


def main():
    parser = argparse.ArgumentParser(description="Automated Daily Korba Air Quality Forecaster.")
    parser.add_argument(
        "--data-file",
        type=str,
        default=None,
        help="Path to recent 15-day CPCB station export file. If omitted, uses latest file in data/raw/.",
    )
    parser.add_argument(
        "--package-path",
        type=str,
        default="models/model_package.pt",
        help="Path to trained model package.",
    )
    parser.add_argument(
        "--station",
        type=str,
        default=None,
        help="Monitoring station name (overrides file header).",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="outputs",
        help="Base directory for output forecasts and reports.",
    )
    parser.add_argument(
        "--lang",
        choices=["en", "hi"],
        default="en",
        help="Language for health advisories.",
    )
    args = parser.parse_args()

    # Determine input data file
    if args.data_file:
        data_path = Path(args.data_file)
    else:
        raw_files = sorted(
            list((PROJECT_ROOT / "data" / "raw").glob("*.xlsx")) +
            list((PROJECT_ROOT / "data" / "raw").glob("*.csv")),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not raw_files:
            logger.error("No raw data files found in data/raw/. Please provide --data-file.")
            sys.exit(1)
        data_path = raw_files[0]
        logger.info(f"Auto-selected latest file from data/raw/: {data_path.name}")

    package_path = Path(args.package_path)
    output_dir = Path(args.outdir)

    try:
        execute_daily_forecast(
            data_path=data_path,
            package_path=package_path,
            station_name=args.station,
            output_base_dir=output_dir,
            lang=args.lang,
        )
    except Exception as e:
        logger.error(f"Daily forecast execution failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
