"""
CPCB National Air Quality Index (NAQI) Calculation Engine.
Implements PRD Section 8:
- Piecewise linear sub-index interpolation for 24-hour average concentrations
- Category mapping: Good, Satisfactory, Moderate, Poor, Very Poor, Severe
- Overall AQI determination: max(I_PM2.5, I_PM10)
- Day+1, Day+2, and Day+3 daily mean evaluation
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

# CPCB Breakpoint Definitions (24-hour averages in ug/m3)
CPCB_BREAKPOINTS = {
    "PM2.5": [
        # (BPLo, BPHi, ILo, IHi)
        (0.0, 30.0, 0, 50),
        (30.0, 60.0, 51, 100),
        (60.0, 90.0, 101, 200),
        (90.0, 120.0, 201, 300),
        (120.0, 250.0, 301, 400),
        (250.0, 500.0, 401, 500),
    ],
    "PM10": [
        # (BPLo, BPHi, ILo, IHi)
        (0.0, 50.0, 0, 50),
        (50.0, 100.0, 51, 100),
        (100.0, 250.0, 101, 200),
        (250.0, 350.0, 201, 300),
        (350.0, 430.0, 301, 400),
        (430.0, 600.0, 401, 500),
    ],
}

# CPCB Category Ranges and Colors
AQI_CATEGORIES = [
    {"name": "Good", "min": 0, "max": 50, "color": "#2ECC71"},
    {"name": "Satisfactory", "min": 51, "max": 100, "color": "#82E0AA"},
    {"name": "Moderate", "min": 101, "max": 200, "color": "#F4D03F"},
    {"name": "Poor", "min": 201, "max": 300, "color": "#E67E22"},
    {"name": "Very Poor", "min": 301, "max": 400, "color": "#E74C3C"},
    {"name": "Severe", "min": 401, "max": 500, "color": "#922B21"},
]


@dataclass
class DailyAQIResult:
    """Summary of 24-hour forecast and regulatory AQI status."""
    day_name: str
    hour_range: Tuple[int, int]
    pm25_mean: float
    pm10_mean: float
    pm25_sub_index: int
    pm10_sub_index: int
    overall_aqi: int
    category: str
    color: str
    dominant_pollutant: str
    pm25_q10_mean: Optional[float] = None
    pm25_q90_mean: Optional[float] = None
    pm10_q10_mean: Optional[float] = None
    pm10_q90_mean: Optional[float] = None


def calculate_sub_index(concentration: float, pollutant: str) -> int:
    """
    Computes sub-index for a given pollutant concentration using CPCB formula:
        Ip = ((IHi - ILo) / (BPHi - BPLo)) * (Cp - BPLo) + ILo
    """
    c = max(0.0, float(concentration))
    bands = CPCB_BREAKPOINTS.get(pollutant.upper())
    if not bands:
        raise ValueError(f"Unsupported pollutant: {pollutant}. Must be PM2.5 or PM10.")

    # Values exceeding highest defined breakpoint cap at 500
    if c >= bands[-1][1]:
        return 500

    for bp_lo, bp_hi, i_lo, i_hi in bands:
        if bp_lo <= c <= bp_hi:
            if bp_hi == bp_lo:
                return i_lo
            sub_index = ((i_hi - i_lo) / (bp_hi - bp_lo)) * (c - bp_lo) + i_lo
            return int(round(sub_index))

    return 500


def get_aqi_category_and_color(aqi_val: int) -> Tuple[str, str]:
    """Returns official CPCB category name and hexadecimal color badge for an AQI value."""
    val = max(0, min(500, int(round(aqi_val))))
    for cat in AQI_CATEGORIES:
        if cat["min"] <= val <= cat["max"]:
            return cat["name"], cat["color"]
    return "Severe", "#922B21"


def calculate_aqi(pm25: float, pm10: float) -> Tuple[int, str, str, Dict[str, int]]:
    """
    Computes overall CPCB AQI from 24-hour mean concentrations of PM2.5 and PM10.
    Returns:
        overall_aqi: int
        category: str
        color: str
        sub_indices: {"PM2.5": int, "PM10": int}
    """
    sub_pm25 = calculate_sub_index(pm25, "PM2.5")
    sub_pm10 = calculate_sub_index(pm10, "PM10")

    overall = max(sub_pm25, sub_pm10)
    category, color = get_aqi_category_and_color(overall)

    return overall, category, color, {"PM2.5": sub_pm25, "PM10": sub_pm10}


def evaluate_daily_aqi_forecast(
    y_pred: np.ndarray,
    target_names: List[str] = ["PM2.5", "PM10"],
) -> List[DailyAQIResult]:
    """
    Evaluates Day+1, Day+2, Day+3 24-hour averages from 72-hour forecast array.
    Supports either:
        Point forecast: shape (72, 2)
        Quantile forecast: shape (72, 2, 3) where [:, :, 0] is q10, [:, :, 1] is q50, [:, :, 2] is q90
    """
    day_slices = [
        ("Day+1", (0, 24)),
        ("Day+2", (24, 48)),
        ("Day+3", (48, 72)),
    ]

    has_quantiles = (y_pred.ndim == 3 and y_pred.shape[-1] == 3)
    results: List[DailyAQIResult] = []

    pm25_idx = target_names.index("PM2.5") if "PM2.5" in target_names else 0
    pm10_idx = target_names.index("PM10") if "PM10" in target_names else 1

    for day_name, (start_h, end_h) in day_slices:
        if has_quantiles:
            # Use q50 (median) as primary forecast
            slice_pm25_med = y_pred[start_h:end_h, pm25_idx, 1]
            slice_pm10_med = y_pred[start_h:end_h, pm10_idx, 1]

            pm25_mean = float(np.round(np.mean(slice_pm25_med), 1))
            pm10_mean = float(np.round(np.mean(slice_pm10_med), 1))

            pm25_q10 = float(np.round(np.mean(y_pred[start_h:end_h, pm25_idx, 0]), 1))
            pm25_q90 = float(np.round(np.mean(y_pred[start_h:end_h, pm25_idx, 2]), 1))
            pm10_q10 = float(np.round(np.mean(y_pred[start_h:end_h, pm10_idx, 0]), 1))
            pm10_q90 = float(np.round(np.mean(y_pred[start_h:end_h, pm10_idx, 2]), 1))
        else:
            slice_pm25 = y_pred[start_h:end_h, pm25_idx]
            slice_pm10 = y_pred[start_h:end_h, pm10_idx]

            pm25_mean = float(np.round(np.mean(slice_pm25), 1))
            pm10_mean = float(np.round(np.mean(slice_pm10), 1))
            pm25_q10 = pm25_q90 = pm10_q10 = pm10_q90 = None

        overall_aqi, cat, color, subs = calculate_aqi(pm25_mean, pm10_mean)
        dominant = "PM2.5" if subs["PM2.5"] >= subs["PM10"] else "PM10"

        res = DailyAQIResult(
            day_name=day_name,
            hour_range=(start_h, end_h),
            pm25_mean=pm25_mean,
            pm10_mean=pm10_mean,
            pm25_sub_index=subs["PM2.5"],
            pm10_sub_index=subs["PM10"],
            overall_aqi=overall_aqi,
            category=cat,
            color=color,
            dominant_pollutant=dominant,
            pm25_q10_mean=pm25_q10,
            pm25_q90_mean=pm25_q90,
            pm10_q10_mean=pm10_q10,
            pm10_q90_mean=pm10_q90,
        )
        results.append(res)

    return results
