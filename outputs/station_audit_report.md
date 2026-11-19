# CECB / CPCB Station Data Quality Audit Report

**Stations Evaluated:** 2

| Station | Status | Days | Range | Valid Hours % | PM2.5 Miss % | PM10 Miss % | Issues / Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Rampur, Korba - CECB | ❌ Rejected | 29.0 | 2024-11-01 to 2024-11-30 | 99.57% | 1.94% | 1.97% | Insufficient history: 29.0 days (minimum 365 days required for model training). |
| Banki Colliery, Korba - CECB | ❌ Rejected | 29.0 | 2024-11-01 to 2024-11-30 | 99.28% | 2.41% | 1.72% | Insufficient history: 29.0 days (minimum 365 days required for model training). |

### Station Readiness Decision
**Verdict:** Training blocked. Found 0 eligible station(s), but at least 3 stations are required by PRD Section 5.3.
*(Note: You may use Pipeline Test Mode with synthetic datasets to verify the entire pipeline while acquiring real station files).*