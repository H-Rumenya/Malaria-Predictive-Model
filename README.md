
# Malaria Predictive Model 

This README documents the **Business Understanding** for the malaria early‑warning project. It is intended for stakeholders (MoH/County DoH, facility managers, humanitarian partners), the data team, and field implementers (CHWs).

---

## 1) Executive Summary
Kenya’s health system faces a persistent malaria burden; **10 counties account for ~95% of national cases**. Turkana is among the highest‑burden areas, and **Turkana West (Kakuma Refugee Camp) has managed >65,000 malaria cases annually over the past five years**, with increasingly **unpredictable peaks**.

Climate variability has undermined the reliability of rainy seasons in ASAL counties, making responses **reactive and delayed**, driving lost schooling/productivity, stock‑outs, HCW burnout, and avoidable morbidity/mortality (especially in pregnant women and children).

This project proposes a **climate‑informed Early Warning & Response (EWR) platform** that fuses **recent weather signals** (rainfall, temperature, humidity, wind speed) with **historical malaria trends** to forecast outbreak **timing**, **magnitude**, and **risk** **up to 4 weeks** ahead. 

---

## 2) Context & Need
- **Vector–parasite ecology:** Ambient temperatures **> ~18 °C**, standing water, and bushy surroundings accelerate mosquito breeding; in ASAL contexts, these conditions occur after rains.
- **Uncertain seasons:** Shifting onset and intensity of rains in Kakuma have eroded seasonal predictability, **undermining preparedness** and forcing last‑minute responses that overwhelm staff and supplies.

**Why now?** An operational, climate‑aware forecast gives **lead time** to pre‑position diagnostics (RDTs), treatments (ACTs), nets/IRS, and to schedule CHW outreach before peaks.

---

## 3) Problem Statement
> **How can high‑resolution temporal weather data (rainfall, temperature, windspeed, humidity) combined with historical malaria case trends improve the accuracy of short‑term malaria outbreak forecasts in endemic regions?**

---

## 4) Goals & Objectives
- Deliver **actionable 4‑week forecasts** at facility/catchment level.
- Produce **risk‑tier alerts (Low / Medium / High)** with clear operational playbooks.
- Enable **data‑driven planning** for MoH/County and partners (stock, staffing, outreach).

### SMART Targets (initial)
- **Lead time:** ≥ 4 weeks ahead of observed peaks for ≥ 70% of high‑risk alerts in the pilot period.
- **Forecast quality:** Baseline targets—Regression **MAE ≤ 10–15 cases** per facility‑week; Classification **F1 ≥ 0.70** for high‑risk tier (to be refined after baseline EDA).
- **Operational uptake:** **≥ 80%** of alerts actioned within **7 days** by facilities/CHWs.

---

## 5) Stakeholders & Roles
| Stakeholder | Role in Project | Key Decisions/Actions |
|---|---|---|
| **MoH / County DoH** | Sponsor, governance, integration with surveillance | Approve roll‑out; allocate budget; align with surveillance SOPs |
| **Health Managers (facility/NGO)** | Operational owner | Roster/triage planning; stock pre‑positioning; outreach calendars |
| **CHWs** | Frontline implementers | Door‑to‑door sensitization, net hang‑up, hotspot follow‑up |
| **Humanitarian partners/donors** | Funding & logistics | Targeting, procurement timing, surge staffing |
| **Data team** | Data engineering & modeling | Data pipelines, model training, monitoring & dashboards |

---

## 6) Success Metrics
**Business/Operational**
- **Lead time:** Consistent **4‑week forecasts** at facility/catchment level.
- **Uptake:** % of alerts actioned within **7 days**; **stockout rate** during peaks; **CHW coverage** in high‑risk zones.

**Technical (Forecast Quality)**
- **Regression:** RMSE, MAE, R²
- **Time‑series:** MASE
- **Classification (alerts):** Precision, Recall, **F1**, ROC‑AUC

---

## 7) Scope
**In‑scope**
- Facility‑level malaria case counts & positivity (Kakuma pilot), Open‑Meteo weather (rainfall, temperature, humidity, wind speed).
- Creation of lagged weather features and rolling aggregates.
- Weekly forecasts (option to evaluate daily where data permits).
- Dashboard + weekly SMS/Email briefs.

---

## 8) Data Sources & Access
- **Cases:** Facility registers / DHIS2 extracts from Kakuma sites (counts, RDT positives). Custodian: Facility managers/County DoH.
- **Weather:** **Open‑Meteo** API (rainfall, temperature, humidity, wind speed). Fallback: NASA POWER/NOAA if needed.

> **Access & Security:** Follow MoH/County data‑sharing agreements. No personally identifiable information (PII) is required; use aggregated counts.

---

## 9) Data Dictionary (initial – to be finalized post‑ingestion)
| Field | Type | Example | Notes |
|---|---|---|---|
| `date` | date | `2025-07-14` | ISO‑8601 |
| `facility_id` | string | `KAKUMA-01` | Unique facility code |
| `cases_total` | int | `124` | OPD malaria cases (all ages) |
| `cases_rdt_positive` | int | `78` | Confirmed positives |
| `rain_mm` | float | `12.4` | Daily or aggregated to week |
| `temp_mean_c` | float | `29.1` | °C |
| `rh_mean_pct` | float | `47.0` | Relative humidity |
| `wind_mean_ms` | float | `3.2` | m/s |
| `rain_mm_lag2w` | float | `32.8` | Engineered feature |
| `temp_mean_lag3w` | float | `27.6` | Engineered feature |

---

## 10) Assumptions & Constraints
- Case reporting completeness is ≥ 90% week‑to‑week after basic cleaning/imputation.
- Weather feeds are programmatically retrievable and stable.
- Initial history available: **3–5 years** (longer history improves seasonal modeling).

**Constraints**
- Class imbalance likely (many low weeks, few peaks) — will affect thresholding and evaluation.
- Limited local compute; aim for **lightweight models** initially; scale up only as needed.

---

## 11) Risks & Mitigations
| Risk | Impact | Mitigation |
|---|---|---|
| Missing/incomplete case data | Bias/poor forecasts | Automated checks; imputation; data quality dashboards |
| Weather API downtime | Gaps in features | Redundant sources; caching; retry logic |
| Class imbalance | Over‑prediction of low risk | Use AUPRC monitoring; calibrated thresholds; re‑sampling |
| Black‑box concerns | Low trust/adoption | SHAP/LIME explanations; clear SOPs per risk tier |

---

## 12) Exploratory Data Analysis (EDA) Plan
**Objectives**
1. Characterize **trends, seasonality, and variability** in cases per facility.
2. Quantify **lag relationships** between weather variables and cases.
3. Assess **data quality** (missingness, outliers, reporting gaps) and **class imbalance**.
4. Produce **baseline metrics** to guide modeling targets and thresholds.

**Guiding Questions**
- What are the **peak weeks/months** and how stable are they year‑over‑year?
- Which **weather lags** (e.g., rainfall 2–4 weeks prior) correlate most with cases?


**Analyses & Visuals**
- Line plots: cases vs. time (per facility, aggregated).
- Seasonal decomposition (STL) and autocorrelation (ACF/PACF).
- Cross‑correlation of cases vs. lagged weather features.
- Heatmaps of correlation matrix (raw and lagged).
- Distribution plots: weekly cases; **class imbalance** view.
- Missingness matrix/heatmap; outlier detection summary.

**EDA Deliverables**
- `eda_report.html` (auto‑generated) and a narrative notebook `01_eda.ipynb`.
- Cleaned, versioned datasets in `/data/processed/`.
- Recommendations for feature set and evaluation thresholds.

