# PRD: Multi-Station LSTM Forecasting of PM2.5 & PM10, Korba (CECB/CPCB Data)

Version 2.0 (no code in this document; code is written in the build phase) | Owner: Sanjeet | Audience: developer or AI coding assistant who will build the project from this document

## 1. How to Use This PRD (Instructions for the AI Builder)

Read the whole document before writing code. Build in the order of Section 14 (Milestones). Every requirement has an ID (FR-x, NFR-x, DP-x). Where something is marked **[VERIFY]**, check it against the real data or website first and tell the user what you found instead of guessing. Do not invent data. If the live data source is not reachable, use the fallback in Section 5.4. Keep code modular, commented, and runnable end to end with one command per stage.

## 2. Problem Statement

Korba (Chhattisgarh) is a coal mining and thermal power hub with high particulate pollution. CECB and CPCB publish only past and current readings, so the user cannot see PM2.5 and PM10 for the next 3 days and cannot plan precautions in advance. The user's sample file (Rampur, Korba - CECB, 2 to 17 Nov 2026, 15-minute data, about 15.5 days) is enough to build and test the pipeline but not to train the final model. This PRD therefore requires about 2 years of data from at least 3 stations (the user calls them "mines"; this PRD treats them as monitoring stations).

## 3. Goal and Success Criteria

**Goal:** From the past 15 days (360 hours) of data at a station, forecast hourly PM2.5 and PM10 for the next 3 days (72 hours), then show the Day+1, Day+2, Day+3 averages with AQI category and precautions. One model is trained jointly on at least 3 stations, each with about 2 years of data.

**Success criteria:**

- Forecast for Day+1, Day+2, Day+3 (24-hour average) and an hourly 72-hour curve, for PM2.5 and PM10, for every trained station.
- On the untouched test period, the model beats both the persistence and the seasonal-naive baselines on MAE and RMSE for all 3 days.
- AQI category for Day+1 is correct within one category on at least 85% of test days, and exactly correct on at least 65% [targets; adjust after seeing the data].
- The 10th to 90th percentile band contains the actual value about 80% of the time (between 70% and 90% is acceptable).
- A leave-one-station-out test is reported to show how well the model works on a station it never saw.
- The dashboard opens and shows the forecast in under 5 seconds.

## 4. Scope

**In scope:** collecting about 2 years of data for at least 3 stations, cleaning, feature engineering, multi-station LSTM training, baselines, hyperparameter tuning, evaluation, 72-hour forecast with uncertainty band, AQI categories, precautions, Streamlit dashboard, and an app with five screens (upload training data, train, results, forecast from the last 15 days of data, report), plus an optional daily refresh job.

**Out of scope (v1):** other states, mobile app, health diagnosis, source apportionment, own sensor hardware.

## 5. Data Requirements

### 5.1 Source

- Primary: the CECB/CPCB continuous ambient air quality export (.xlsx) for each station, 15-minute average. Format = the user's sample file: a header block (State, City, Station, Parameter list, AvgPeriod), then rows with From Date and To Date in dd-mm-yyyy HH:MM.
- Stations: at least 3, each with about 2 years of history. Rampur, Korba - CECB is the first; the user picks the other two. **[VERIFY]** how many months of history each station really has before committing to it.
- If the portal limits the date range or times out, download each station in several chunks. The portal fills the requested range with empty rows, so the loader must trim empty rows at the start and end.
- Optional (v2): a free weather forecast API such as Open-Meteo for Korba coordinates, used as known future inputs.

### 5.2 Required columns (hourly)

| Column | Role | Note from the user's sample file |
| --- | --- | --- |
| PM2.5, PM10 (ug/m3) | Targets and inputs | Present, about 1.6% missing |
| NO, NO2, SO2, CO, OZONE | Inputs | Present; OZONE about 6% missing |
| RH, AT, WS, WD, SR | Inputs (weather) | Present; WD only spans 11 to 40 degrees, check the sensor |
| NH3 | Input only if it varies | Almost constant (6.1 to 7.2) in the sample |
| RF (rain) | Input only if it varies | All zero in the sample (dry November); keep if rain appears in the 2-year data |
| BP, MH | Drop unless another station fills them | Completely empty in the sample |

Stations can differ in which columns they report. The pipeline keeps the columns common to all chosen stations, lists what it dropped and why, and must still work in minimum mode (PM2.5, PM10 and time features only).

### 5.3 Data volume

Minimum about 730 days per station and at least 3 stations, so each season (winter peak, summer, monsoon, post-monsoon) appears twice. A station with under 365 days is not used for training. After cleaning, each station should keep at least 70% valid hours. With a 360-hour input and 72-hour output sliding one hour at a time, 2 years gives roughly 17,000 overlapping samples per station. They overlap, so the real amount of independent information is much smaller, which is why simple baselines and strong regularisation are required.

### 5.4 Fallback

If automatic download fails, the user downloads files by hand into `data/raw/`. The station name is read from each file's header, so files of different stations may sit in one folder. A synthetic data generator may be used only to test the pipeline, must be clearly labelled, and its output is never shown as a real forecast.

## 6. Data Pipeline Requirements

- **DP-1 Ingestion:** read every export file of every station; keep raw files untouched in `data/raw/`.
- **DP-2 Parsing:** station name from the header, timestamps from From Date (IST), remove duplicates, sort by time.
- **DP-3 Cleaning:** negative values, impossible values (PM2.5 above 1000, PM10 above 2000) and identical PM values repeated for 6 hours or more become missing. Trim empty padding rows.
- **DP-4 Hourly conversion:** average the four 15-minute readings per hour and require at least 3 of 4; wind direction uses a circular mean (sine and cosine).
- **DP-5 Missing values:** linear interpolation for gaps up to 3 hours; time-of-day-aware interpolation for 4 to 24 hours; longer gaps stay missing and any training window touching them is excluded. Add a `was_imputed` flag. Print a per-station quality report (date range, percent missing, constant columns).
- **DP-6 Consistency checks:** flag hours where PM2.5 is more than 10% above PM10 (physically suspicious). Keep real pollution spikes; only remove values that fail DP-3.
- **DP-7 Features:** sine and cosine of hour, weekday and day of year; wind direction as sine and cosine; rolling means (6h, 24h, 72h) of PM2.5 and PM10; PM2.5/PM10 ratio; station ID; optional season flags (Diwali period, October to November crop-burning season, winter inversion months).
- **DP-8 Scaling:** log1p on PM and gas values (right-skewed), then standardise using training-period statistics only, computed per station. Save the scalers.
- **DP-9 Windows:** for each station, input 360 hours, output 72 hours, stride 1 for training and stride 24 for evaluation. A window never crosses a data gap or a station boundary.

## 7. Model Requirements

### 7.1 Task formulation

Supervised multi-step forecasting. **Input window:** the past 15 days = 360 hours of all features, per station. **Output:** the next 72 hourly values for PM2.5 and PM10, shape (72, 2). Day+1, Day+2, Day+3 values are the mean of hours 1 to 24, 25 to 48 and 49 to 72. The station ID is an extra input so one model learns from all stations and still knows which one it is predicting.

### 7.2 Architecture (baseline to build first)

```
Inputs
  A: past window, 360 hours x n_features
  B: station ID -> embedding (8 dims), repeated over time and joined to A
Encoder
  Conv1D(64, kernel 3)               light local smoothing of noisy hourly data
  Bidirectional LSTM(128, sequences) -> Dropout 0.2   (sees only the past window)
  LSTM(64)                           -> Dropout 0.2
  Attention over the 360 steps       (optional; keep only if validation improves)
Head
  Dense(128, relu)
  Dense(72 x 2 x 3) -> reshape       72 hours x 2 pollutants x 3 quantiles (10, 50, 90)
Output: median forecast + 10% to 90% band for PM2.5 and PM10
```

Loss: pinball (quantile) loss for the 10th, 50th and 90th percentiles, with about double weight on hours where the actual value is in the Poor category or worse, so spikes are not under-predicted. Optimizer Adam, learning rate 1e-3 with ReduceLROnPlateau, batch size 64 to 128, up to 100 epochs with EarlyStopping (patience 10, restore best weights), gradient clipping 1.0. Every setting lives in `config.yaml`.

### 7.3 Comparison models (required)

1. Persistence (repeat the last 24 hours).
2. Seasonal naive (same hour of the previous day) and the hourly average of the last 7 days.
3. A tree model (XGBoost or LightGBM) on lag and rolling features, one model per forecast horizon group.
4. Ablations of the proposed model: plain LSTM, GRU, LSTM without station embedding, with and without attention.
5. Stretch: a Transformer-style forecaster (for example Temporal Fusion Transformer or N-HiTS) for comparison only.

The final model is always the LSTM (one design for all stations). The baselines above are used only to check that the LSTM is really better; they are never the final forecast. If the LSTM does not beat them, improve the LSTM (tuning, features, more data) and report the result honestly. The LSTM may be trained with 5 seeds, and the median of the 5 is used as the final forecast only if it beats a single seed on validation.

### 7.4 Splitting

Chronological split inside every station: first about 70% train, next 15% validation, last 15% test, with the test period kept to at least one full season. **No random shuffling across time and no leakage**: windows are built after the split, scalers fit on the training part only, and a 3-day gap is left between splits so windows do not overlap. Extra experiment: leave-one-station-out (train on 2 stations, test on the third) to prove the model generalises.

### 7.5 Uncertainty

The quantile outputs give the forecast range. After training, measure on validation how often the actual value falls inside the 10 to 90 band (target about 80%). If it is off by more than 10 points, widen or narrow the band with a simple conformal adjustment fitted on validation data. The dashboard always shows the median line plus this band.

### 7.6 Evaluation metrics

- MAE, RMSE, R-squared, sMAPE and bias (mean error), separately for PM2.5 and PM10, for each of Day+1, Day+2, Day+3, for each station and for each season.
- Skill score against persistence: 1 minus (model MAE / persistence MAE). It must be positive for all 3 days.
- AQI category accuracy (exact and within one category) on daily means.
- High-pollution detection: precision and recall for days at Poor category or worse.
- Interval quality: coverage and average width of the 10 to 90 band.
- Plots: actual vs predicted over the test period, error vs forecast hour, error by AQI category.
- Accuracy is expected to fall with horizon; report it honestly.

### 7.7 Future weather problem

When forecasting 72 hours ahead, future weather is unknown. v1 uses only past features. v2 adds the weather forecast for the next 72 hours as known future inputs (wind, rain, humidity, temperature); a v1 vs v2 comparison is a planned experiment because rain and wind are the biggest missing drivers of day 2 and day 3 error.

### 7.8 Training strategy for the best model

- Train one joint model on all stations with the station embedding.
- Train 5 random seeds, report mean and standard deviation, and use the 5-seed ensemble (median of medians) as the final forecast if it beats a single seed on validation.
- Hyperparameter search (Optuna or a small grid, about 30 trials) over window length (240, 360, 480 hours), LSTM units (64, 128, 256), dropout (0.1 to 0.4), learning rate, and attention on or off. Choose using validation only; open the test set once, at the end.
- Regularisation against overfitting: dropout, early stopping, weight decay, and no more capacity than validation supports.
- Retrain every month as new data arrives, and keep the old model if the new one is not better on the latest month.

### 7.9 Sanity checks and explainability

- Confirm the model does more than copy the last value: compare with persistence at every forecast hour.
- Permutation importance of each feature group (weather, gases, time) and a station-embedding check.
- Error by season, by hour of day and by AQI category; check that residuals have no strong autocorrelation.
- Inspect the 10 worst test days and write down why the model failed (fire event, festival, plant shutdown, missing weather).

## 8. AQI Categorisation Logic

Use the CPCB National AQI breakpoints on the **24-hour average** concentration (ug/m3). Compute a sub-index for each pollutant by linear interpolation inside the band; the AQI is the maximum of the PM2.5 and PM10 sub-indices. **[VERIFY]** the breakpoints against the current CPCB document before finalising.

| Category | AQI range | PM2.5 (24h) | PM10 (24h) | Colour |
| --- | --- | --- | --- | --- |
| Good | 0-50 | 0-30 | 0-50 | Green |
| Satisfactory | 51-100 | 31-60 | 51-100 | Light green |
| Moderate | 101-200 | 61-90 | 101-250 | Yellow |
| Poor | 201-300 | 91-120 | 251-350 | Orange |
| Very Poor | 301-400 | 121-250 | 351-430 | Red |
| Severe | 401-500 | 250+ | 430+ | Dark red |

Formula: `Ip = ((IHi - ILo) / (BPHi - BPLo)) * (Cp - BPLo) + ILo`. Implement in `src/aqi.py` with unit tests for every band edge.

## 9. Precaution Engine

The app maps the predicted category (worst of PM2.5 and PM10 for that day) to advice. Store text in `config/precautions.json` so it can be edited without code changes. Always show the line: *This is general guidance, not medical advice. People with heart or lung conditions should follow their doctor's advice.*

| Category | General public | Sensitive groups (children, elderly, pregnant, asthma/heart/lung patients) |
| --- | --- | --- |
| Good | Normal activity. Open windows for ventilation. | No restriction. |
| Satisfactory | Normal activity. | Unusually sensitive people should watch for cough or breathlessness during long outdoor exertion. |
| Moderate | Reduce prolonged heavy outdoor exercise. Keep windows closed in dusty hours. | Limit outdoor exertion; keep rescue medicines handy; consider a mask outdoors. |
| Poor | Avoid heavy outdoor exercise. Wear a well-fitted N95/FFP2 mask outdoors. Use indoor air purifier if available. | Stay indoors as much as possible. Postpone outdoor sports and school activities. |
| Very Poor | Avoid outdoor activity, especially morning and late evening. N95 mask for any necessary trip. Keep windows and doors closed. Do not burn waste or use dust-raising activities. | Stay indoors. Contact a doctor if cough, wheeze, chest tightness, or breathlessness appears. |
| Severe | Stay indoors, no outdoor exercise. N95 mask if you must go out. Run a purifier in one clean room. | Avoid all outdoor exposure; seek medical help early for symptoms. |

FR-7: The dashboard shows a day-wise advisory card for Day+1, Day+2, Day+3, with colour, category, numeric range, and the precaution text.

FR-8: Show an extra banner if any single hour in the next 72 hours is predicted in Very Poor or Severe, even when the daily mean is lower.

## 10. Application Requirements

The user's roadmap: the user gives only this PRD to an AI tool, the AI writes the code, the user runs it, the app shows an upload option, the user uploads data, the model is trained, and after training the user gives the last 15 days of data and gets the complete forecast and precautions. The app (Streamlit) therefore has five screens, used in this order.

- **FR-9 Screen 1, Upload training data:** upload many CPCB/CECB .xlsx exports at once (several stations, about 2 years each). The app reads the station name from each file and shows the quality report (days per station, percent missing, columns used, gaps). Training is blocked with a clear message if there are fewer than 3 stations or a station has under 365 days. A clearly labelled "pipeline test mode" may allow training on less data, but its results are marked "not a real forecast".
- **FR-10 Screen 2, Train:** a Train button with two modes. Quick mode: one training run with default settings, must finish in under 30 minutes on a normal laptop CPU. Full mode: tuning and 5 seeds (section 7.8), run outside the browser as a script or on Google Colab; the app can load its saved model. Training runs in the background so the page never freezes, and shows progress, elapsed time and the loss curve. It saves the model, scalers, settings and metrics, and keeps the test period untouched.
- **FR-11 Screen 3, Results:** after training, show test-period metrics against the baselines (persistence, seasonal naive, tree model), plots, and the 10 to 90 band coverage. The model is marked "approved" only if it meets the acceptance criteria of section 15; otherwise a warning says what failed.
- **FR-12 Screen 4, Forecast:** the user uploads the last 15 days (360 hours) of data for one station in the same export format, or picks a station whose data is already loaded. The app checks that the station is known, that at least 80% of the 360 hours are valid, and that the latest data point is not older than 6 hours, and warns if not. It then shows the complete information: 72-hour chart of PM2.5 and PM10 with the 10 to 90 band and AQI colour bands, the Day+1, Day+2 and Day+3 values, the AQI category, the precaution cards (FR-7), the extra banner for Very Poor or Severe hours (FR-8), the model's typical error, and any data warnings.
- **FR-13 Screen 5, Report and share:** download the forecast as PDF and CSV, and copy a short text summary (day-wise values, category, main precaution) to send by message. A public web link needs deployment and is v2.
- **FR-14 Language:** English UI with an optional Hindi toggle for precautions.
- **FR-15 Daily refresh (optional):** a script that fetches or reads new data, cleans it, predicts and saves, run once or twice a day with cron or Windows Task Scheduler.

### 10.1 Two separate jobs: train once, forecast often

The model is trained once on the large data set and saved. After that, forecasting needs only the last 15 days of data.

| | When | What is uploaded | How often |
| --- | --- | --- | --- |
| 1. Training | First, once | About 2 years of data from at least 3 stations | Once, then occasionally again |
| 2. Forecast | Whenever a forecast is needed | Only the last 15 days (360 hours) of one station | Daily or any time |

- **FR-16 Saved model package:** after training, save one package in `models/`: the model file, the scalers, the station list with their IDs, the settings (window 360, horizon 72, features used), the test metrics, and the training date and data range. Forecasting uses only this package.
- **FR-17 App start:** when the app opens it loads the saved package and never trains by itself. If no package exists it shows "No trained model yet" and sends the user to Screen 1 and Screen 2. Every forecast shows the model version, training date and training data range.
- **FR-18 Forecast needs only 15 days:** Screen 4 must work with just the 15-day file. The training data is not needed at forecast time, and a forecast takes only seconds.
- **FR-19 Station and format check:** the station in the 15-day file must be one the model was trained on. If it is not, show a clear warning that the forecast will be less reliable, or block it. The file must be the same export format and contain the columns the model uses; list any missing columns and say what the app will do about them.
- **FR-20 Retraining:** repeat training now and then (about monthly, when new data is available). A new model replaces the old one only if it is not worse on the latest data. Keep the previous package as a backup so the user can go back.
- **FR-21 Model status board:** when training finishes the app shows a clear message "Model trained on <date and time>". A status table with one row per station is shown on the Results screen and on the Forecast screen. Columns: station, training date, training data range (first and last date), number of days used, test MAE for Day+1, Day+2 and Day+3, and status (Approved, Warning, or Not approved). The table is saved inside the model package (FR-16), so it is correct every time the app opens.
- **FR-22 Model age warning:** if the model is older than 30 days, show "Retraining due" beside the status table; the forecast still works. If a station is Warning or Not approved, show that on the Forecast screen next to the numbers.

## 11. Non-Functional Requirements

- **NFR-1 Reproducibility:** fixed random seeds, pinned `requirements.txt`, all hyperparameters in `config.yaml`.
- **NFR-2 Performance:** inference under 5 seconds on a normal laptop CPU; training should finish in under 30 minutes on CPU for 2 years of hourly data, or use Google Colab.
- **NFR-3 Quality:** type hints, docstrings, unit tests for cleaning, windowing, AQI, and precaution mapping; no hard-coded file paths.
- **NFR-4 Honesty:** the app must show model error (for example typical MAE) next to the forecast so users know the uncertainty.
- **NFR-5 Data respect:** use only publicly published data, keep the source credited (CECB / CPCB) in the app footer.

## 12. Tech Stack

Python 3.10+, pandas, numpy, scikit-learn, TensorFlow/Keras (or PyTorch if preferred), matplotlib/plotly, Streamlit, joblib, PyYAML, requests/BeautifulSoup (only if scraping is needed and allowed by the site), pytest.

## 13. Suggested Project Structure

```
pm_forecast_korba/
  config.yaml
  requirements.txt
  README.md
  data/raw/  data/processed/
  config/precautions.json
  notebooks/01_eda.ipynb  02_model_experiments.ipynb
  src/
    data/fetch.py  data/clean.py  data/features.py  data/windows.py
    models/lstm.py  models/baselines.py  train.py  evaluate.py
    aqi.py  precautions.py  predict.py  run_daily.py
  app/dashboard.py
  models/  (saved .keras model, scalers)
  outputs/  (metrics, plots, forecasts)
  tests/
```

## 14. Milestones (Build in This Order)

1. **Data access:** download about 2 years for at least 3 stations, run the quality report, confirm the final station list. Stop and show the report to the user.
2. **EDA:** hourly, daily and seasonal patterns, PM2.5 vs PM10, gaps, differences between stations.
3. **Pipeline:** cleaning, features, scaling, windowing, with unit tests.
4. **Baselines:** persistence, seasonal naive, tree model, with a metrics table.
5. **Model v1:** first working LSTM on Rampur alone, then the joint multi-station model.
6. **Tuning and ablations:** hyperparameter search, 5 seeds, ensemble decision.
7. **Final evaluation:** one run on the test period, leave-one-station-out test, error analysis (section 7.9).
8. **AQI and precautions:** implement and unit-test.
9. **Inference and dashboard:** the five app screens of section 10 (upload, train, results, forecast, report).
10. **Automation and report:** daily run, README, final report with limitations.

## 15. Acceptance Criteria

- Without touching code, the user can open the app, upload training files, see the quality report, train, see the test results, then upload the last 15 days of data and get the full forecast with precautions.
- One training command (or the Train button) trains on all stations and saves the model and scalers.
- For any station, a 72-hour forecast (median plus 10 to 90 band) and the 3 daily means are produced.
- On the test period the model beats persistence on MAE for Day+1, Day+2 and Day+3, for PM2.5 and PM10, on most stations. If it does not, the report says so and explains why.
- The 10 to 90 band coverage is between 70% and 90%.
- The app shows forecast, categories and precautions; all unit tests pass.
- After training once, closing and reopening the app loads the saved model without retraining, and a forecast from only a 15-day file works.
- After training, the app shows "Model trained on <date>" and a per-station status table (training date, data range, test errors, status); the table is still there after the app is closed and reopened.
- A final report contains the metrics tables, plots, leave-one-station-out result and known limitations.

## 16. Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Only the 15-day sample is available | Use it only to build and test the pipeline; do not train the final model until about 2 years per station are collected |
| CECB or CPCB site has no bulk download or changes layout | Manual file download into the raw folder; loader reads the standard export format |
| Stations report different columns | Use the common columns; run the minimum mode (PM only plus time) as a fallback |
| Sensor faults, stuck values, large gaps | Cleaning rules, imputation flag, exclude bad windows, per-station quality report |
| Overfitting because windows overlap | Chronological split with gaps, dropout, early stopping, simple baselines as a check |
| Sudden spikes (fires, festivals, plant events) are unpredictable | Weighted loss on high hours, show uncertainty band, flag high-risk hours |
| Day+3 accuracy is low | Show confidence falling with horizon and the typical error beside each day |
| Users treat advice as medical guidance | Disclaimer on every advisory |

## 17. Open Questions for the Owner (AI builder should ask if still unknown)

1. Which stations will be used (names), and does "mines" mean monitoring stations?
2. How many months of history does each chosen station have?
3. Will training run locally or on Google Colab (a GPU speeds up tuning)?
4. Dashboard only, or also email or WhatsApp alerts (v2)?
5. Hindi precautions in v1?
6. One forecast per station, or also a combined Korba view?

## 18. Future Enhancements (v2+)

Weather forecast as future input, attention-based or Transformer model, multiple Korba stations, alert notifications, model monitoring with automatic retraining each month, and a public mobile-friendly web deployment.
