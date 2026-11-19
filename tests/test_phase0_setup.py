"""
Test Suite for Phase 0: Project Setup, Scaffolding, and Configuration Baseline.
Uses standard library unittest so it runs in any standard Python environment.
"""

import json
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestPhase0Scaffolding(unittest.TestCase):
    """Verifies that all required folders and baseline files exist."""

    def test_required_directories_exist(self):
        required_dirs = [
            "config",
            "data/raw",
            "data/processed",
            "models",
            "notebooks",
            "outputs",
            "src",
            "src/data",
            "src/models",
            "src/utils",
            "app",
            "tests",
        ]
        for rel_path in required_dirs:
            full_path = PROJECT_ROOT / rel_path
            self.assertTrue(
                full_path.is_dir(),
                f"Required directory missing: {rel_path}"
            )

    def test_package_init_files_exist(self):
        init_files = [
            "src/__init__.py",
            "src/data/__init__.py",
            "src/models/__init__.py",
            "src/utils/__init__.py",
            "app/__init__.py",
            "tests/__init__.py",
        ]
        for rel_path in init_files:
            full_path = PROJECT_ROOT / rel_path
            self.assertTrue(
                full_path.is_file(),
                f"Missing package init file: {rel_path}"
            )

    def test_git_and_environment_files_exist(self):
        required_files = [
            ".gitignore",
            "requirements.txt",
            "pytest.ini",
            "config/config.yaml",
            "config/precautions.json",
            "PROJECT_PHASES_ROADMAP.md",
        ]
        for rel_path in required_files:
            full_path = PROJECT_ROOT / rel_path
            self.assertTrue(
                full_path.is_file(),
                f"Required project file missing: {rel_path}"
            )


class TestPhase0Configuration(unittest.TestCase):
    """Validates structure and content of config.yaml and precautions.json."""

    def test_precautions_json_schema(self):
        precautions_file = PROJECT_ROOT / "config" / "precautions.json"
        self.assertTrue(precautions_file.exists())

        with open(precautions_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("disclaimer_en", data)
        self.assertIn("disclaimer_hi", data)
        self.assertIn("precautions", data)
        self.assertIn("hourly_spike_alert", data)

        required_categories = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
        for cat in required_categories:
            self.assertIn(cat, data["precautions"], f"Missing AQI category: {cat}")
            cat_data = data["precautions"][cat]
            self.assertIn("aqi_range", cat_data)
            self.assertIn("color", cat_data)
            self.assertIn("general_public_en", cat_data)
            self.assertIn("general_public_hi", cat_data)
            self.assertIn("sensitive_groups_en", cat_data)
            self.assertIn("sensitive_groups_hi", cat_data)

    def test_config_loader_utility(self):
        from src.utils.config import load_config, load_precautions

        config = load_config()
        self.assertIsInstance(config, dict)
        self.assertIn("project", config)
        self.assertIn("data", config)
        self.assertIn("model", config)
        self.assertIn("aqi", config)

        # Check key numeric constants
        data_cfg = config["data"]
        self.assertEqual(int(data_cfg["window_length_hours"]), 360)
        self.assertEqual(int(data_cfg["forecast_horizon_hours"]), 72)

        precautions = load_precautions()
        self.assertIsInstance(precautions, dict)
        self.assertIn("precautions", precautions)


if __name__ == "__main__":
    unittest.main()
