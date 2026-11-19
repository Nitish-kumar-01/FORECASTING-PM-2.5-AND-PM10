"""
Precaution Advisory Engine & Extreme Hourly Spike Alert System.
Implements PRD Section 9, FR-7, FR-8, and FR-14:
- Translates daily AQI forecasts into actionable demographic health advisories
- Extreme hourly spike detector (flags alert if any individual hour reaches Very Poor or Severe)
- Bilingual guidance support (English / Hindi)
- Enforces mandatory medical disclaimer
"""

from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.aqi import DailyAQIResult, calculate_sub_index
from src.utils.config import load_precautions

logger = logging.getLogger(__name__)


@dataclass
class DailyAdvisory:
    """Formatted health advisory card for a forecast day."""
    day_name: str
    aqi: int
    category: str
    color: str
    pm25_mean: float
    pm10_mean: float
    dominant_pollutant: str
    general_public_advice: str
    sensitive_groups_advice: str
    disclaimer: str
    language: str
    pm25_range: str = ""
    pm10_range: str = ""


@dataclass
class HourlySpikeAlert:
    """Alert banner data for episodic hazardous pollution spikes."""
    has_spike: bool
    peak_hour: int
    peak_pollutant: str
    peak_concentration: float
    peak_category: str
    alert_title: str
    alert_message: str
    affected_hours_count: int


class PrecautionEngine:
    """
    Manages demographic advisory generation and extreme pollution spike detection.
    """

    def __init__(self, precautions_config: Optional[Dict[str, Any]] = None):
        if precautions_config is None:
            precautions_config = load_precautions()
        self.config = precautions_config

    def get_advisory(self, daily_result: DailyAQIResult, lang: str = "en") -> DailyAdvisory:
        """
        Generates day-wise advisory card (FR-7, FR-14) for a given language ('en' or 'hi').
        """
        category = daily_result.category
        precautions_dict = self.config.get("precautions", {})
        cat_info = precautions_dict.get(category, precautions_dict.get("Moderate", {}))

        if lang.lower() == "hi":
            gen_advice = cat_info.get("general_public_hi", "")
            sens_advice = cat_info.get("sensitive_groups_hi", "")
            disclaimer = self.config.get(
                "disclaimer_hi",
                "यह सामान्य मार्गदर्शन है, चिकित्सा सलाह नहीं।"
            )
        else:
            gen_advice = cat_info.get("general_public_en", "")
            sens_advice = cat_info.get("sensitive_groups_en", "")
            disclaimer = self.config.get(
                "disclaimer_en",
                "This is general guidance, not medical advice. People with heart or lung conditions should follow their doctor's advice."
            )

        # Range strings (q10 to q90 if available)
        pm25_range = ""
        if daily_result.pm25_q10_mean is not None and daily_result.pm25_q90_mean is not None:
            pm25_range = f"[{daily_result.pm25_q10_mean:.1f} - {daily_result.pm25_q90_mean:.1f}]"

        pm10_range = ""
        if daily_result.pm10_q10_mean is not None and daily_result.pm10_q90_mean is not None:
            pm10_range = f"[{daily_result.pm10_q10_mean:.1f} - {daily_result.pm10_q90_mean:.1f}]"

        return DailyAdvisory(
            day_name=daily_result.day_name,
            aqi=daily_result.overall_aqi,
            category=category,
            color=daily_result.color,
            pm25_mean=daily_result.pm25_mean,
            pm10_mean=daily_result.pm10_mean,
            dominant_pollutant=daily_result.dominant_pollutant,
            general_public_advice=gen_advice,
            sensitive_groups_advice=sens_advice,
            disclaimer=disclaimer,
            language=lang,
            pm25_range=pm25_range,
            pm10_range=pm10_range,
        )

    def detect_hourly_spikes(
        self,
        y_72h: np.ndarray,
        target_names: List[str] = ["PM2.5", "PM10"],
        lang: str = "en",
    ) -> HourlySpikeAlert:
        """
        Implements FR-8:
        Inspects all 72 hourly forecast points. Triggers an alert banner if any single hour
        enters 'Very Poor' (PM2.5 > 120 or PM10 > 350) or 'Severe' (PM2.5 > 250 or PM10 > 430),
        even when the 24-hour daily average is in a lower category.
        """
        # y_72h shape: (72, 2) or (72, 2, 3)
        if y_72h.ndim == 3 and y_72h.shape[-1] == 3:
            # Inspect median predictions
            hourly_pm25 = y_72h[:, 0, 1]
            hourly_pm10 = y_72h[:, 1, 1]
        else:
            hourly_pm25 = y_72h[:, 0]
            hourly_pm10 = y_72h[:, 1]

        spike_hours = []
        peak_conc = 0.0
        peak_hour = 0
        peak_pollutant = "PM2.5"
        peak_category = "Moderate"

        for h in range(len(hourly_pm25)):
            c25 = float(hourly_pm25[h])
            c10 = float(hourly_pm10[h])

            sub25 = calculate_sub_index(c25, "PM2.5")
            sub10 = calculate_sub_index(c10, "PM10")

            max_sub = max(sub25, sub10)
            if max_sub > 300:  # 301+ is Very Poor or Severe
                spike_hours.append(h)
                if max_sub > peak_conc:
                    peak_conc = max_sub
                    peak_hour = h + 1  # 1-indexed hour (1 to 72)
                    if sub25 >= sub10:
                        peak_pollutant = "PM2.5"
                    else:
                        peak_pollutant = "PM10"
                    peak_category = "Severe" if max_sub > 400 else "Very Poor"

        spike_alert_cfg = self.config.get("hourly_spike_alert", {})

        if spike_hours:
            if lang.lower() == "hi":
                title = spike_alert_cfg.get("title_hi", "उच्च प्रदूषण चेतावनी (पीक ऑवर)")
                msg = spike_alert_cfg.get(
                    "message_hi",
                    f"पूर्वानुमान के दौरान {len(spike_hours)} घंटों में प्रदूषण '{peak_category}' स्तर तक पहुँचने की संभावना है (अधिकतम घंटा {peak_hour})।"
                )
            else:
                title = spike_alert_cfg.get("title_en", "HIGH POLLUTION PEAK WARNING")
                msg = spike_alert_cfg.get(
                    "message_en",
                    f"Warning: {len(spike_hours)} individual hour(s) in the 72-hour forecast reach the {peak_category} category (Peak at Hour {peak_hour}: {peak_pollutant})."
                )

            return HourlySpikeAlert(
                has_spike=True,
                peak_hour=peak_hour,
                peak_pollutant=peak_pollutant,
                peak_concentration=peak_conc,
                peak_category=peak_category,
                alert_title=title,
                alert_message=msg,
                affected_hours_count=len(spike_hours),
            )

        return HourlySpikeAlert(
            has_spike=False,
            peak_hour=0,
            peak_pollutant="None",
            peak_concentration=0.0,
            peak_category="Normal",
            alert_title="",
            alert_message="",
            affected_hours_count=0,
        )
