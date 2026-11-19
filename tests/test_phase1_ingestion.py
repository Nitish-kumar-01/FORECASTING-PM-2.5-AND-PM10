"""
Unit Tests for Phase 1: Data Ingestion, Parsing, Synthetic Generation, and Quality Audit.
"""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.data.audit import audit_station_data, generate_quality_report_markdown
from src.data.fetch import load_cpcb_file, normalize_column_name, parse_cpcb_header
from src.data.synthetic import generate_station_data, save_synthetic_cpcb_file


class TestPhase1DataIngestion(unittest.TestCase):
    """Tests the CPCB file parser and column normalizer."""

    def test_normalize_column_name(self):
        cases = {
            "PM2.5 (ug/m3)": "PM2.5",
            "pm25": "PM2.5",
            "PM 2.5": "PM2.5",
            "PM10 (ug/m3)": "PM10",
            "pm10": "PM10",
            "NO2 (ug/m3)": "NO2",
            "NO (ug/m3)": "NO",
            "SO2 (ug/m3)": "SO2",
            "CO (mg/m3)": "CO",
            "Ozone (ug/m3)": "OZONE",
            "RH (%)": "RH",
            "Relative Humidity (%)": "RH",
            "AT (degree C)": "AT",
            "Ambient Temperature": "AT",
            "WS (m/s)": "WS",
            "WD (deg)": "WD",
            "SR (W/m2)": "SR",
        }
        for raw, expected in cases.items():
            self.assertEqual(normalize_column_name(raw), expected, f"Failed on raw name: '{raw}'")

    def test_synthetic_generation_and_cpcb_parsing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "test_rampur.xlsx"

            # 1. Generate 3 days of synthetic 15-minute readings
            df_gen = generate_station_data(
                station_name="Rampur, Korba - CECB",
                start_date="2026-11-01 00:00",
                end_date="2026-11-03 23:45",
                freq="15min",
                seed=42,
            )
            self.assertEqual(len(df_gen), 3 * 96)  # 3 days * 96 15-min intervals
            self.assertIn("PM2.5 (ug/m3)", df_gen.columns)
            self.assertIn("PM10 (ug/m3)", df_gen.columns)

            # 2. Save with realistic CPCB header
            saved_file = save_synthetic_cpcb_file(
                df=df_gen,
                station_name="Rampur, Korba - CECB",
                output_path=tmp_path,
                format_type="xlsx",
            )
            self.assertTrue(saved_file.exists())

            # 3. Test Header Parsing
            meta, header_row = parse_cpcb_header(saved_file)
            self.assertIn("Rampur", meta.station_name)
            self.assertEqual(meta.city, "Korba")
            self.assertEqual(meta.state, "Chhattisgarh")
            self.assertGreater(header_row, 0)

            # 4. Test Complete Loading & Parsing
            df_loaded, loaded_meta = load_cpcb_file(saved_file)
            self.assertIsInstance(df_loaded.index, pd.DatetimeIndex)
            self.assertEqual(len(df_loaded), 3 * 96)
            self.assertIn("PM2.5", df_loaded.columns)
            self.assertIn("PM10", df_loaded.columns)
            self.assertIn("NO2", df_loaded.columns)
            self.assertIn("AT", df_loaded.columns)
            self.assertIn("RH", df_loaded.columns)

            # Verify all numeric
            for col in ["PM2.5", "PM10", "NO2", "AT"]:
                self.assertTrue(np.issubdtype(df_loaded[col].dtype, np.number))

            # 5. Test Audit Engine
            audit_res = audit_station_data(df_loaded, station_name="Rampur, Korba - CECB")
            self.assertEqual(audit_res.station_name, "Rampur, Korba - CECB")
            self.assertAlmostEqual(audit_res.total_days, 3.0, delta=0.1)
            self.assertGreater(audit_res.valid_hours_pct, 90.0)
            self.assertEqual(audit_res.status, "REJECTED")  # Rejected because 3 days < 365 days minimum requirement!

            report_md = generate_quality_report_markdown([audit_res])
            self.assertIn("CECB / CPCB Station Data Quality Audit Report", report_md)
            self.assertIn("Rampur, Korba - CECB", report_md)
            self.assertIn("Training blocked", report_md)


if __name__ == "__main__":
    unittest.main()
