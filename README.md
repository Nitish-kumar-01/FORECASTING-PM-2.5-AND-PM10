# Multi-Station LSTM Forecasting of PM2.5 & PM10, Korba (CECB/CPCB Data)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.15+-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, production-grade deep learning air quality forecasting system for **Korba, Chhattisgarh** (an intensive coal-mining and thermal power generation hub). Ingesting historical continuous ambient air quality exports from the Chhattisgarh Environment Conservation Board (**CECB**) and Central Pollution Control Board (**CPCB**), the system trains a joint multi-station temporal neural network and generates **72-hour forward-looking forecasts** (Day+1, Day+2, Day+3) with calibrated uncertainty envelopes, CPCB National AQI categories, and actionable health advisories.

---

## 🌟 Key System Capabilities

1. **Dual Operational Workflow:**
   - **Train Once (Offline / Periodic):** Learns spatio-temporal particulate dynamics jointly across $\ge 2$ years of data from at least 3 stations using learned **Station Entity Embeddings**.
   - **Forecast Daily (Instant Inference):** Takes only the **last 15 days (360 hours)** of continuous readings for any station to generate the next 72 hours of hourly predictions in **$< 1$ second**.
2. **Deep Learning Architecture:**
   - **Conv1D Noise Smoother:** 1D convolution layer filtering high-frequency hourly sensor noise.
   - **Recurrent Encoder:** Stacked Bidirectional LSTM (128 units) and Unidirectional LSTM (64 units) with dropout regularization.
   - **Temporal Bahdanau Attention:** Learns to focus on critical diurnal and seasonal periods in the 15-day lookback window.
   - **Non-Crossing Quantile Head:** Projecting 10th percentile ($q_{10}$), median ($q_{50}$), and 90th percentile ($q_{90}$) bands using positive softplus offsets. Guaranteed monotonic ($q_{10} \le q_{50} \le q_{90}$) to eliminate quantile crossing artifacts.
3. **Loss & Calibration:**
   - **Weighted Pinball Loss:** Double penalty weighting on hours with "Poor" or worse air quality ($\ge 201$ AQI) to avoid under-predicting dangerous pollution spikes.
   - **Conformal Interval Calibration:** Empirically calibrated on validation splits to guarantee $\sim 80\%$ observed coverage.
4. **Regulatory Domain Logic:**
   - Official **CPCB National AQI** sub-index calculations for 24-hour daily averages.
   - **Extreme Hourly Spike Detector:** Raises an emergency alert banner if any single hour in the 72-hour curve reaches "Very Poor" or "Severe", even if the daily average is lower.
   - **Bilingual Guidance:** English and Hindi health advisories tailored for the general public and sensitive demographics (children, elderly, respiratory/cardiac patients).

---

## 📂 Project Structure

```
pm_forecast_korba/
├── PRD_PM25_PM10_Forecasting_Korba (1).md   # Original Product Requirements Document
├── PROJECT_PHASES_ROADMAP.md                 # Full 9-phase execution blueprint
├── requirements.txt                          # Pinned dependencies
├── pytest.ini                                # Test runner configuration
├── .gitignore                                # Git ignore specification
│
├── config/
│   ├── config.yaml                           # System hyperparameters, columns, and thresholds
│   └── precautions.json                      # Bilingual CPCB health advisories & disclaimers
│
├── data/
│   ├── raw/                                  # CPCB/CECB 15-minute .xlsx exports
│   └── processed/                            # Cleaned and engineered intermediate tensors
│
├── models/
│   └── model_package.pt                      # Serialized weights, scalers, schema & metadata
│
├── notebooks/
│   ├── 01_eda.py                             # Exploratory Data Analysis execution script
│   └── 01_eda.ipynb                          # Interactive Jupyter EDA notebook
│
├── outputs/
│   ├── eda_figures/                          # Diurnal, correlation & meteorological plots
│   ├── forecasts/                            # Headless CSV and JSON forecast archives
│   ├── reports/                              # Generated PDF forecast bulletins
│   └── station_audit_report.md               # Data completeness and quality audit report
│
├── src/
│   ├── data/
│   │   ├── fetch.py                          # CPCB Excel header parsing and ingestion (DP-1, DP-2)
│   │   ├── clean.py                          # Cleaning, circular wind, and tiered imputation (DP-3 to DP-6)
│   │   ├── features.py                       # Cyclical time, rolling means & ratio features (DP-7)
│   │   ├── windows.py                        # Station scaler and sliding window tensors (DP-8, DP-9)
│   │   ├── synthetic.py                      # Realistic CPCB synthetic generator (PRD 5.4)
│   │   └── audit.py                          # Station health audit engine (PRD 5.3)
│   │
│   ├── models/
│   │   ├── lstm.py                           # Conv1D + BiLSTM + Attention + Quantile Head (PRD 7.2)
│   │   ├── loss.py                           # Weighted Pinball Loss with penalty multipliers
│   │   ├── baselines.py                      # Persistence, Seasonal-Naive & GBDT baselines (PRD 7.3)
│   │   ├── loso.py                           # Leave-One-Station-Out cross-validation (PRD 7.4)
│   │   └── diagnostics.py                    # Residual autocorrelation & worst period extraction (PRD 7.9)
│   │
│   ├── aqi.py                                # CPCB National AQI sub-index calculator (PRD 8)
│   ├── precautions.py                        # Precaution mapping & extreme spike alerts (PRD 9)
│   ├── train.py                              # PyTorch training, conformal calibration & packaging
│   ├── predict.py                            # Operational 15-day inference pipeline (FR-12, FR-18)
│   ├── evaluate.py                           # Multi-horizon metrics & Persistence Skill Score
│   ├── run_daily.py                          # Automated headless daily scheduler script (FR-15)
│   └── utils/
│       └── config.py                         # Safe configuration and JSON advisory loader
│
├── app/
│   ├── dashboard.py                          # 5-Screen Streamlit interactive web application
│   └── components/
│       ├── charts.py                         # Plotly 72h forecast curves with CPCB color bands
│       └── pdf_export.py                     # ReportLab PDF forecast bulletin generator (FR-13)
│
└── tests/                                    # Automated pytest test suite (41+ tests)
    ├── test_phase0_setup.py
    ├── test_phase1_ingestion.py
    ├── test_phase2_pipeline.py
    ├── test_phase3_baselines.py
    ├── test_phase4_model.py
    ├── test_phase5_domain.py
    ├── test_phase6_app.py
    ├── test_phase7_robustness.py
    └── test_phase8_automation.py
```

---

## 🚀 Quickstart & Installation

### 1. Set Up Environment
```bash
# Clone or navigate to the workspace
cd "/Users/nitishkumar/Desktop/Forcasting pm 2.5 and pm 10"

# Create and activate Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Automated Verification Test Suite
```bash
.venv/bin/pytest -v tests/
```
All **41+ test cases** validate data cleaning, circular wind math, non-crossing quantiles, CPCB breakpoints, and sub-second CPU latency.

---

## 🖥️ Interactive Web Dashboard (Streamlit)

Launch the 5-screen interactive application:
```bash
.venv/bin/streamlit run app/dashboard.py
```

### Dashboard Screens:
* **🔮 Screen 4 (Operational Forecast - The Core Screen):** Upload the last 15 days (360 hours) of continuous data for any Korba station. Instantly renders Day+1, Day+2, and Day+3 summary cards, official CPCB colored badges, demographic precautions, **Emergency Hourly Spike Banners**, and interactive Plotly curves.
* **📊 Screen 3 (Results & Status Board):** Inspect model training metadata, the persistent per-station approval status board, and comparative benchmarks against Persistence, Seasonal-Naive, and GBDT baselines.
* **⚙️ Screen 2 (Train Model):** Train the joint neural network using Quick Mode (CPU $<5$ min) or Full Mode (50 epochs) with live loss convergence charts.
* **📤 Screen 1 (Upload Training Data):** Upload multi-station exports into `data/raw/`, run quality health audits, or enable Pipeline Test Mode for synthetic data.
* **📑 Screen 5 (Report & Share):** Download the official **PDF Forecast Bulletin**, export hourly predictions as **CSV**, or copy one-click **WhatsApp / SMS** text briefing snippets in English or Hindi.

---

## ⏱️ Automated Scheduled Execution (CLI / Cron)

For headless daily batch execution (compatible with Linux/macOS `cron` or Windows Task Scheduler):
```bash
# Runs daily forecast on latest file in data/raw/
.venv/bin/python src/run_daily.py --station "Rampur, Korba - CECB" --lang en

# Explicitly specify input file and output directory:
.venv/bin/python src/run_daily.py --data-file "data/raw/rampur_recent.xlsx" --outdir "outputs"
```

Outputs generated:
* `outputs/forecasts/forecast_<station>_<date>.csv`: 72 hourly predictions (q10, median, q90).
* `outputs/reports/bulletin_<station>_<date>.pdf`: Official printable PDF advisory bulletin.
* `outputs/forecasts/summary_<station>_<date>.json`: Machine-readable forecast summary.
* `outputs/alerts/ALERT_<station>_<date>.txt`: Emergency notification if any single hour enters Very Poor or Severe.

---

## 📋 PRD Acceptance Criteria Checklist

| PRD Section 15 Requirement | Status | Verification Reference |
|---|---|---|
| Complete 5-screen Streamlit application | ✅ Implemented | `app/dashboard.py` (Screens 1 to 5) |
| Single training command trains on all stations | ✅ Implemented | `src/train.py`, Screen 2 Train button |
| 72-hour forecast (median + 10-90 band) & 3 daily means | ✅ Implemented | `src/predict.py`, `app/components/charts.py` |
| Model beats persistence on MAE for all 3 days | ✅ Verified | `src/models/baselines.py`, `src/evaluate.py` |
| 10th to 90th percentile coverage between 70% and 90% | ✅ Verified | `src/train.py` (`calibrate_conformal_multiplier`) |
| Precaution engine with demographic advice and disclaimers | ✅ Implemented | `src/precautions.py`, `config/precautions.json` |
| Closing and reopening app loads saved model without retrain | ✅ Verified | `src/train.py` (`load_model_package`), `models/model_package.pt` |
| Per-station status table persistent across app restarts | ✅ Verified | `status_table` serialized inside `model_package.pt` |
| Leave-one-station-out generalization and error report | ✅ Implemented | `src/models/loso.py`, `outputs/error_analysis_report.md` |

---

## ⚖️ Legal & Medical Guidance Disclaimer
*This forecasting software and its generated advisories are intended strictly for environmental preparedness, operational planning, and public informational purposes. They do not constitute formal medical diagnosis or clinical advice. Individuals with pre-existing cardiopulmonary conditions, asthma, or other respiratory vulnerabilities must follow their personal physician's directives.*  
**Data Attribution:** Continuous Ambient Air Quality Monitoring (CECB / CPCB).
