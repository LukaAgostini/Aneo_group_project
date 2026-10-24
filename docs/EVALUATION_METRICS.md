# 4-Tier Evaluation Framework & Metric Guide

**Project: Day-Ahead Electricity Consumption Forecasting (Aneo / Retailer)**

This document defines the comprehensive **4-Tier Evaluation Framework** developed in Task 2 & 3. It serves as the common contract aligning:

- **Pair 1 (Tasks 1 & 4):** Problem Definition, Business Case, Financial KPIs, and Strategic Recommendations.
- **Pair 2 (Tasks 2 & 3):** Data Understanding, Feature Engineering, Baseline vs. ML Models (Ridge, LightGBM), and Technical Evaluation.
- **Pair 3 (Tasks 5 & 6):** Deployment, Production Pipeline, SLA Enforcement, and Continuous Monitoring.

---

## Metric Architecture & Cross-Task Alignment

```
┌────────────────────────────────────────────────────────────────────────┐
│ TIER 1: Technical & Statistical Modeling Metrics                       │
│ (Lead: Pair 2 | Target: Assess forecast accuracy & generalization)     │
├────────────────────────────────────────────────────────────────────────┤
│ TIER 2: Business & Financial Value KPIs                                │
│ (Lead: Pairs 2 & 1 | Target: Quantify avoided volume & cost savings)   │
├────────────────────────────────────────────────────────────────────────┤
│ TIER 3: Operational, Latency & MLOps SLA                               │
│ (Lead: Pairs 2 & 3 | Target: Guarantee zero-leakage & timely delivery) │
├────────────────────────────────────────────────────────────────────────┤
│ TIER 4: Client Relationship & Strategic Retention                      │
│ (Lead: Pairs 2 & 1 | Target: Contract ROI, trust band & tail risk)     │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Tier 1: Technical & Statistical Modeling Metrics

| Metric                       | Mathematical Formula                                                                           | Unit | Interpretation & Project Value                                                                                                                                        |
| :--------------------------- | :--------------------------------------------------------------------------------------------- | :--- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **MAE**                      | $\frac{1}{N} \sum_{i=1}^N \|y_i - \hat{y}_i\|$                                                 | MW   | **Average Error Magnitude:** Base operational accuracy metric. Direct proxy for average hourly physical mismatch.                                                     |
| **RMSE**                     | $\sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$                                          | MW   | **Variance & Extreme Error Penalty:** Penalizes large hourly misses heavily (critical during grid stress or extreme cold snaps).                                      |
| **WAPE**                     | $\frac{\sum \|y_i - \hat{y}_i\|}{\sum y_i} \times 100\%$                                       | %    | **Normalized MAE:** Scale-independent. Allows fair comparison across bidding zones with different portfolio sizes (e.g. Oslo vs Tromsø) without zero-division issues. |
| **NMBE**                     | $\frac{\sum (\hat{y}_i - y_i)}{\sum y_i} \times 100\%$                                         | %    | **Systematic Bias:** Positive means systematic overforecasting (buying too much); negative means underforecasting. Ideal value is $0\%$.                              |
| **Benchmark Outperformance** | $\left(1 - \frac{\text{MAE}_{\text{model}}}{\text{MAE}_{\text{baseline}}}\right) \times 100\%$ | %    | **Skill Score vs Naive Baseline:** Quantifies the percentage error reduction gained over the naive 7-day lag ($t-168\text{h}$) benchmark.                             |

_How to use:_ Pair 2 uses these in **Section 3 (Technical Evaluation)** to prove model superiority. Pair 1 cites them in **Section 1 & 4** as evidence of forecast quality.

---

## Tier 2: Business & Financial Value KPIs

| Metric                               | Mathematical Formula                                                   | Unit | Interpretation & Project Value                                                                                    |
| :----------------------------------- | :--------------------------------------------------------------------- | :--- | :---------------------------------------------------------------------------------------------------------------- |
| **Total Imbalance Volume**           | $\sum_{t=1}^N \|y_t - \hat{y}_t\| \times 1\text{h}$                    | MWh  | Total physical imbalance volume the retailer must settle on the balancing market over the evaluation period.      |
| **Avoided Imbalance Volume**         | $\text{Imbalance}_{\text{baseline}} - \text{Imbalance}_{\text{model}}$ | MWh  | Physical volume mismatch prevented by using the ML model instead of the baseline.                                 |
| **Simulated Imbalance Cost**         | $\sum_{t=1}^N \|y_t - \hat{y}_t\| \times P_t$                          | NOK  | Financial penalty paid to the TSO/market. Hourly volume error multiplied by hourly spot/settlement price ($P_t$). |
| **Avoided Imbalance Cost (Savings)** | $\text{Cost}_{\text{baseline}} - \text{Cost}_{\text{model}}$           | NOK  | **Direct Financial Savings:** The monetary value generated for the retailer over the evaluation period.           |

_How to use:_ Core input for **Pair 1 (Section 4: Business Evaluation)**. Demonstrates how data science translates directly into bottom-line operational savings.

---

## Tier 3: Operational, Latency & MLOps SLA

| Metric / Check                | Requirement / Rule                                                  | Status          | Interpretation & Production Role                                                                                                      |
| :---------------------------- | :------------------------------------------------------------------ | :-------------- | :------------------------------------------------------------------------------------------------------------------------------------ |
| **Zero Data Leakage**         | $\text{Min Consumption Lag} \ge 120\text{h}$ (Set to $144\text{h}$) | `PASS` / `FAIL` | Enforces the real-world 5-day metering delay. Guarantees forecasts only use features legally known at gate closure ($D-1$ 10:00 UTC). |
| **Timeliness (Pipeline SLA)** | $\text{Runtime} \le 1800\text{s}$ (30 min)                          | `PASS` / `FAIL` | Data Day-Ahead gate closes at 12:00 CET (10:30 UTC operational target). Pipeline must run well within budget.                         |

_How to use:_ Critical bridge for **Pair 3 (Sections 5 & 6: Deployment & Monitoring)** to define CI/CD automated validation tests and scheduling triggers.

---

## Tier 4: Client Relationship & Strategic Retention

| Metric                | Mathematical Formula                                                                                    | Unit | Interpretation & Strategic Value                                                                                                       |
| :-------------------- | :------------------------------------------------------------------------------------------------------ | :--- | :------------------------------------------------------------------------------------------------------------------------------------- |
| **ROI**               | $\left(\frac{\text{Avoided Cost} - \text{Contract Fee}}{\text{Contract Fee}}\right) \times 100\%$       | %    | Quantifies the return on investment for the retailer. Compares saved imbalance costs against Aneo's service fee.                       |
| **Trust Reliability** | $\frac{1}{N} \sum_{i=1}^N \mathbb{I}\left(\frac{\|y_i - \hat{y}_i\|}{y_i} \le 15\%\right) \times 100\%$ | %    | Percentage of hours where relative forecast error is within an acceptable commercial threshold ($\le 15\%$). Drives trader confidence. |
| **P95 Tail Risk**     | 95th Percentile of $\|y_t - \hat{y}_t\|$                                                                | MW   | Quantifies extreme tail error. Lower P95 reduces catastrophic exposure during price spikes.                                            |

### Pricing Policy & the 1/3 Rule

- **Guiding Principle:** For a long-term win-win relationship, Aneo's fee should never exceed **$\frac{1}{3}$ of the retailer's savings** ($\text{Contract Fee} \le \frac{1}{3} \times \text{Avoided Cost}$).
- **Target Value Ratio:** $\text{ROI} \ge 200\%$, meaning for every **1 NOK** paid to Aneo, the retailer recovers at least **3 NOK** in avoided penalties.
- **Contract Fee Calibrations (over 90-day test set):**
  - Over the 90-day test period, LightGBM saves **~3,320,000 NOK** across the portfolio (~36,800 NOK/day total, or ~6,100 NOK/day per city).
  - Recommended maximum contract fee: **$\le 12,000$ NOK/day** across the full portfolio (~2,000 NOK/day per city), guaranteeing an attractive ROI of $\ge 2.0\times$ (200%).

---

## Summary of Results (90-Day Test Set)

| Tier   | Metric                 | Baseline (Lag 168h) | RidgeCV (Linear) | LightGBM (Champion) |
| :----- | :--------------------- | :-----------------: | :--------------: | :-----------------: |
| **T1** | **MAE [MW]**           |        0.53         |       0.29       |      **0.28**       |
| **T1** | **RMSE [MW]**          |        0.87         |       0.46       |      **0.43**       |
| **T1** | **WAPE [%]**           |        9.48%        |      5.17%       |      **5.01%**      |
| **T1** | **Outperformance**     |        0.0%         |      +45.4%      |     **+47.1%**      |
| **T2** | **Avoided Imbalance**  |        0 MWh        |    3,093 MWh     |    **3,209 MWh**    |
| **T2** | **Avoided Cost**       |        0 NOK        |  3,162,894 NOK   |  **3,319,801 NOK**  |
| **T3** | **Leakage Check**      |       `PASS`        |      `PASS`      |       `PASS`        |
| **T3** | **Pipeline SLA**       |       `PASS`        |      `PASS`      |       `PASS`        |
| **T4** | **Trust (<15% Err)**   |        76.2%        |      96.9%       |      **95.4%**      |
| **T4** | **P95 Imbalance [MW]** |       1.89 MW       |     1.05 MW      |     **0.93 MW**     |
