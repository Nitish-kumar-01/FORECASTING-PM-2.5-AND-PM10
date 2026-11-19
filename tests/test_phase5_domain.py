"""
Unit Tests for Phase 5: CPCB AQI Calculation Engine and Precaution Advisory System.
Validates all breakpoint edges, sub-index formulas, bilingual advisories, and spike alerts.
"""

import unittest
import numpy as np

from src.aqi import (
    calculate_aqi,
    calculate_sub_index,
    evaluate_daily_aqi_forecast,
    get_aqi_category_and_color,
)
from src.precautions import PrecautionEngine


class TestPhase5AQIEngine(unittest.TestCase):
    """Tests CPCB breakpoint calculations and sub-index formulas."""

    def test_pm25_breakpoint_edges(self):
        # Good: 0-30 -> 0-50
        self.assertEqual(calculate_sub_index(0.0, "PM2.5"), 0)
        self.assertEqual(calculate_sub_index(15.0, "PM2.5"), 25)
        self.assertEqual(calculate_sub_index(30.0, "PM2.5"), 50)

        # Satisfactory: 31-60 -> 51-100
        sub_60 = calculate_sub_index(60.0, "PM2.5")
        self.assertEqual(sub_60, 100)

        # Moderate: 61-90 -> 101-200
        sub_90 = calculate_sub_index(90.0, "PM2.5")
        self.assertEqual(sub_90, 200)

        # Poor: 91-120 -> 201-300
        sub_120 = calculate_sub_index(120.0, "PM2.5")
        self.assertEqual(sub_120, 300)

        # Very Poor: 121-250 -> 301-400
        sub_250 = calculate_sub_index(250.0, "PM2.5")
        self.assertEqual(sub_250, 400)

        # Severe: 251-500 -> 401-500
        sub_500 = calculate_sub_index(500.0, "PM2.5")
        self.assertEqual(sub_500, 500)
        # Cap above 500
        self.assertEqual(calculate_sub_index(600.0, "PM2.5"), 500)

    def test_pm10_breakpoint_edges(self):
        # Good: 0-50 -> 0-50
        self.assertEqual(calculate_sub_index(0.0, "PM10"), 0)
        self.assertEqual(calculate_sub_index(50.0, "PM10"), 50)

        # Satisfactory: 51-100 -> 51-100
        self.assertEqual(calculate_sub_index(100.0, "PM10"), 100)

        # Moderate: 101-250 -> 101-200
        self.assertEqual(calculate_sub_index(250.0, "PM10"), 200)

        # Poor: 251-350 -> 201-300
        self.assertEqual(calculate_sub_index(350.0, "PM10"), 300)

        # Very Poor: 351-430 -> 301-400
        self.assertEqual(calculate_sub_index(430.0, "PM10"), 400)

        # Severe: 431-600 -> 401-500
        self.assertEqual(calculate_sub_index(600.0, "PM10"), 500)

    def test_overall_aqi_and_dominant_pollutant(self):
        # Case 1: PM2.5 is Poor (105 ug/m3 -> sub-index 250), PM10 is Good (40 ug/m3 -> sub-index 40)
        aqi_val, cat, color, subs = calculate_aqi(pm25=105.0, pm10=40.0)
        self.assertEqual(cat, "Poor")
        self.assertEqual(aqi_val, subs["PM2.5"])
        self.assertGreater(subs["PM2.5"], subs["PM10"])

        # Case 2: PM10 is Severe (500 ug/m3 -> sub-index ~442), PM2.5 is Moderate (75 ug/m3 -> sub-index 150)
        aqi_val, cat, color, subs = calculate_aqi(pm25=75.0, pm10=500.0)
        self.assertEqual(cat, "Severe")
        self.assertEqual(aqi_val, subs["PM10"])

    def test_daily_forecast_evaluation(self):
        # Create 72-hour forecast: Day+1 Good (PM2.5=20, PM10=40), Day+2 Moderate (PM2.5=75, PM10=180), Day+3 Poor (PM2.5=110, PM10=300)
        y_72h = np.zeros((72, 2, 3), dtype=np.float32)

        # Day+1 (0 to 24)
        y_72h[0:24, 0, 1] = 20.0
        y_72h[0:24, 1, 1] = 40.0

        # Day+2 (24 to 48)
        y_72h[24:48, 0, 1] = 75.0
        y_72h[24:48, 1, 1] = 180.0

        # Day+3 (48 to 72)
        y_72h[48:72, 0, 1] = 110.0
        y_72h[48:72, 1, 1] = 300.0

        results = evaluate_daily_aqi_forecast(y_72h)
        self.assertEqual(len(results), 3)

        self.assertEqual(results[0].day_name, "Day+1")
        self.assertEqual(results[0].category, "Good")

        self.assertEqual(results[1].day_name, "Day+2")
        self.assertEqual(results[1].category, "Moderate")

        self.assertEqual(results[2].day_name, "Day+3")
        self.assertEqual(results[2].category, "Poor")


class TestPhase5PrecautionEngine(unittest.TestCase):
    """Tests health advisories, bilingual text, and spike alerts."""

    def setUp(self):
        self.engine = PrecautionEngine()

    def test_bilingual_advisory_generation(self):
        from src.aqi import DailyAQIResult

        dummy_result = DailyAQIResult(
            day_name="Day+1",
            hour_range=(0, 24),
            pm25_mean=110.0,
            pm10_mean=280.0,
            pm25_sub_index=267,
            pm10_sub_index=230,
            overall_aqi=267,
            category="Poor",
            color="#E67E22",
            dominant_pollutant="PM2.5",
        )

        # English advisory
        adv_en = self.engine.get_advisory(dummy_result, lang="en")
        self.assertIn("N95", adv_en.general_public_advice)
        self.assertIn("medical advice", adv_en.disclaimer)

        # Hindi advisory
        adv_hi = self.engine.get_advisory(dummy_result, lang="hi")
        self.assertIn("मास्क", adv_hi.general_public_advice)
        self.assertIn("चिकित्सा सलाह", adv_hi.disclaimer)

    def test_hourly_spike_alert_trigger(self):
        # 72 hours of generally Moderate pollution (PM2.5=70, PM10=150)
        y_72h = np.ones((72, 2)) * 70.0
        y_72h[:, 1] = 150.0

        # No spike initially
        alert_clean = self.engine.detect_hourly_spikes(y_72h)
        self.assertFalse(alert_clean.has_spike)

        # Inject extreme spike at hour 15 (1-indexed: hour 16)
        y_72h[15, 0] = 300.0  # Severe PM2.5 spike!
        alert_spike = self.engine.detect_hourly_spikes(y_72h)
        self.assertTrue(alert_spike.has_spike)
        self.assertEqual(alert_spike.peak_hour, 16)
        self.assertEqual(alert_spike.peak_category, "Severe")
        self.assertIn("WARNING", alert_spike.alert_title)


if __name__ == "__main__":
    unittest.main()
