"""
Data Quality Audit and Station Health Check Module.
Implements DP-5, DP-6, and Section 5.3 data validation requirements.
Produces a detailed diagnostic report for each station before pipeline training.
"""

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.utils.config import load_config

logger = logging.getLogger(__name__)


@dataclass
class StationAuditResult:
    """Statistical summary of data health for a single monitoring station."""
    station_name: str
    start_date: str
    end_date: str
    total_days: float
    total_records: int
    expected_records: int
    data_completeness_pct: float
    missing_pct_per_column: Dict[str, float] = field(default_factory=dict)
    constant_columns: List[str] = field(default_factory=list)
    pm25_exceeds_pm10_count: int = 0
    pm25_exceeds_pm10_pct: float = 0.0
    valid_hours_pct: float = 0.0
    status: str = "PENDING"  # "APPROVED", "WARNING", "REJECTED"
    reasons: List[str] = field(default_factory=list)


def audit_station_data(
    df: pd.DataFrame,
    station_name: str,
    config: Optional[Dict] = None,
) -> StationAuditResult:
    """
    Performs comprehensive data quality audit on a 15-minute station DataFrame.
    """
    if config is None:
        config = load_config()

    cleaning_cfg = config["data"]["cleaning"]
    min_days = config["data"]["min_days_per_station"]
    target_days = config["data"]["target_days_per_station"]
    min_valid_pct = config["data"]["min_valid_hours_pct"]
    ratio_threshold = cleaning_cfg.get("ratio_anomaly_threshold", 1.10)

    if df.empty:
        return StationAuditResult(
            station_name=station_name,
            start_date="N/A",
            end_date="N/A",
            total_days=0.0,
            total_records=0,
            expected_records=0,
            data_completeness_pct=0.0,
            status="REJECTED",
            reasons=["Station dataset is completely empty."],
        )

    start_ts = df.index.min()
    end_ts = df.index.max()
    total_days = round((end_ts - start_ts).total_seconds() / 86400.0, 1)

    # Expected 15-minute readings
    expected_records = int(((end_ts - start_ts).total_seconds() / 900.0) + 1)
    completeness = round((len(df) / max(1, expected_records)) * 100.0, 2)

    # Missing percentage per column
    missing_pct: Dict[str, float] = {}
    constant_cols: List[str] = []
    for col in df.columns:
        series = df[col]
        pct = round(float(series.isna().mean() * 100.0), 2)
        missing_pct[col] = pct
        
        # Check for near-constant columns (std < 1e-4 or identical values)
        valid_vals = series.dropna()
        if len(valid_vals) > 50 and valid_vals.nunique() <= 2:
            constant_cols.append(col)

    # Check physical consistency: PM2.5 > 1.10 * PM10
    pm25_exceeds = 0
    pm25_exceeds_pct = 0.0
    if "PM2.5" in df.columns and "PM10" in df.columns:
        both_present = df[["PM2.5", "PM10"]].dropna()
        if not both_present.empty:
            anomalous = both_present["PM2.5"] > (ratio_threshold * both_present["PM10"])
            pm25_exceeds = int(anomalous.sum())
            pm25_exceeds_pct = round(float(anomalous.mean() * 100.0), 2)

    # Check hourly completeness
    hourly_counts = df.resample("1h").count()
    if "PM2.5" in hourly_counts.columns:
        valid_hours = (hourly_counts["PM2.5"] >= 3).sum()
        total_hours = max(1, len(hourly_counts))
        valid_hours_pct = round(float((valid_hours / total_hours) * 100.0), 2)
    else:
        valid_hours_pct = 0.0

    # Determine status and suitability
    reasons: List[str] = []
    status = "APPROVED"

    if total_days < min_days:
        status = "REJECTED"
        reasons.append(f"Insufficient history: {total_days} days (minimum {min_days} days required for model training).")
    elif total_days < target_days:
        status = "WARNING"
        reasons.append(f"Sub-optimal history: {total_days} days (recommended {target_days} days / 2 years).")

    if valid_hours_pct < min_valid_pct:
        status = "REJECTED"
        reasons.append(f"Valid hours percentage {valid_hours_pct}% is below required threshold ({min_valid_pct}%).")

    if "PM2.5" not in df.columns or "PM10" not in df.columns:
        status = "REJECTED"
        reasons.append("Missing core target columns (PM2.5 and/or PM10).")
    elif missing_pct.get("PM2.5", 100.0) > 30.0 or missing_pct.get("PM10", 100.0) > 30.0:
        status = "WARNING"
        reasons.append(f"High missingness in targets (PM2.5: {missing_pct.get('PM2.5')}%, PM10: {missing_pct.get('PM10')}%).")

    if pm25_exceeds_pct > 5.0:
        reasons.append(f"High rate ({pm25_exceeds_pct}%) of PM2.5 > PM10 physical anomalies detected.")

    return StationAuditResult(
        station_name=station_name,
        start_date=str(start_ts),
        end_date=str(end_ts),
        total_days=total_days,
        total_records=len(df),
        expected_records=expected_records,
        data_completeness_pct=completeness,
        missing_pct_per_column=missing_pct,
        constant_columns=constant_cols,
        pm25_exceeds_pm10_count=pm25_exceeds,
        pm25_exceeds_pm10_pct=pm25_exceeds_pct,
        valid_hours_pct=valid_hours_pct,
        status=status,
        reasons=reasons,
    )


def generate_quality_report_markdown(results: List[StationAuditResult]) -> str:
    """Generates a formatted markdown table report from station audit results."""
    lines = [
        "# CECB / CPCB Station Data Quality Audit Report",
        "",
        f"**Stations Evaluated:** {len(results)}",
        "",
        "| Station | Status | Days | Range | Valid Hours % | PM2.5 Miss % | PM10 Miss % | Issues / Notes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for res in results:
        status_badge = {
            "APPROVED": "✅ Approved",
            "WARNING": "⚠️ Warning",
            "REJECTED": "❌ Rejected",
        }.get(res.status, res.status)

        pm25_miss = res.missing_pct_per_column.get("PM2.5", "N/A")
        pm10_miss = res.missing_pct_per_column.get("PM10", "N/A")
        date_range_str = f"{res.start_date[:10]} to {res.end_date[:10]}"
        notes = "; ".join(res.reasons) if res.reasons else "Clean data"

        lines.append(
            f"| {res.station_name} | {status_badge} | {res.total_days} | {date_range_str} | "
            f"{res.valid_hours_pct}% | {pm25_miss}% | {pm10_miss}% | {notes} |"
        )

    lines.append("")
    lines.append("### Station Readiness Decision")
    approved_count = sum(1 for r in results if r.status in ["APPROVED", "WARNING"])
    if approved_count >= 3:
        lines.append(f"**Verdict:** System ready for joint training ({approved_count} stations meet training criteria).")
    else:
        lines.append(f"**Verdict:** Training blocked. Found {approved_count} eligible station(s), but at least 3 stations are required by PRD Section 5.3.")
        lines.append("*(Note: You may use Pipeline Test Mode with synthetic datasets to verify the entire pipeline while acquiring real station files).*")

    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    from src.data.fetch import load_all_stations

    parser = argparse.ArgumentParser(description="Audit data quality for all stations in data/raw.")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Directory containing raw CPCB/CECB exports.")
    parser.add_argument("--out-report", type=str, default="outputs/station_audit_report.md", help="Path to save markdown report.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    print(f"Loading and auditing files from {data_dir}...")
    stations = load_all_stations(data_dir)

    if not stations:
        print(f"No station files found in '{data_dir}'. Place CPCB/CECB .xlsx or .csv files there.")
    else:
        results = []
        for name, df in stations.items():
            print(f"Auditing '{name}' ({len(df)} readings)...")
            res = audit_station_data(df, name)
            results.append(res)

        report_md = generate_quality_report_markdown(results)
        out_path = Path(args.out_report)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report_md)

        print("\n" + "=" * 60)
        print(report_md)
        print("=" * 60)
        print(f"Audit report saved to: {out_path}")

