# Aneo

## Electricity Consumption Forecasting for a retailer

Your goal is to forecast next-day hourly consumption under realistic data-availability constraints and evaluate whether the resulting forecasts reduce the retailer’s imbalance risk and create sufficient business value.

### Dataset
* `consumption_temp.csv` (Download)
* Slides: `Aneo.pdf` (Download)

---

### Context

Aneo is a power production company, but it also provides services to other companies in the energy sector. One such service is **electricity consumption forecasting for energy retailers**.

The dataset contains the **hourly electricity consumption served by one anonymized retailer** across several Nordic price areas: NO1, NO2, NO3, NO4, NO5, and FI.

* The retailer supplies electricity to a subset of customers within each price area. Therefore, the consumption values in this dataset do **not** represent the total electricity consumption of the corresponding city or price area. They represent only the consumption of customers served by this retailer.
* The **Location** column gives the largest city in each price area and is mainly used as a representative location for weather data.

---

### Forecasting Problem

Your task is to build a forecasting solution that predicts the retailer’s **hourly electricity consumption for the following day**.

* Every day, at approximately **10:00**, you must generate forecasts for all 24 hours of the next day.
* *For example:* At 10:00 on 18 September 2023, your model should predict consumption for 19 September 2023, 00:00–23:00.
* This means that each daily forecast consists of **24 hourly consumption predictions**.

#### Data availability constraint
A key part of this problem is that not all historical consumption data is available when you make the forecast.

* Consumption measurements are available with approximately a **5-day delay**.
* Therefore, when forecasting consumption for 19 September, the latest historical consumption values available to you are from **13 September**. You must respect this restriction when building features and evaluating your model.
* Weather forecasts are different. The weather forecast for the following day is available when you generate the consumption forecast. Therefore, when predicting consumption for 19 September, you may use the weather forecast for 19 September.

**In short:**
* **Available at forecasting time:**
  * historical consumption up to approximately 5 days earlier
  * weather forecasts for the following day
  * calendar information such as hour, weekday, month, season, holidays, etc.
* **Not available at forecasting time:**
  * yesterday’s consumption
  * consumption earlier on the day being predicted
  * any information that would only become known after the forecast was submitted

Your solution should reproduce these real forecasting conditions.

---

### Business Problem

The purpose of the forecast is **not** to determine how much electricity Aneo should produce. Instead, the forecast supports the retailer’s participation in the electricity market.

Before electricity is delivered, the retailer must estimate how much electricity its customers are expected to consume. This forecast is used when planning and nominating the retailer’s expected electricity demand.

If actual consumption differs from the forecast, an **imbalance** occurs.
* *Forecast consumption at 14:00:* $25\text{ MW}$
* *Actual consumption at 14:00:* $20\text{ MW}$
* *Imbalance:* $5\text{ MW}$

This imbalance must be financially settled in the balancing/regulating market. The settlement price can differ substantially from the regular spot-market price and can be difficult to predict.

**Therefore:** Larger forecast errors can expose the retailer to greater financial risk.

Both overforecasting and underforecasting can create imbalance. For the scope of this project, assume that there is no preferred direction of error. The goal is to predict consumption as accurately and reliably as possible.

#### Business Value for Aneo
For this project, assume that the retailer pays Aneo a **fixed contract fee** for providing the forecasting service. Aneo therefore does not directly earn or lose money from each imbalance. However, the quality of the forecast determines the value of the service provided to the retailer.

A simplified value chain is:
$$\text{Better forecast} \rightarrow \text{smaller imbalance} \rightarrow \text{lower financial exposure} \rightarrow \text{higher perceived value} \rightarrow \text{greater customer satisfaction}$$

The business question is therefore not only:
* *Which model predicts consumption most accurately?*

but also:
* *Does the forecasting solution provide sufficiently reliable and valuable forecasts for the retailer to continue using the service?*

---

### Evaluation

You should evaluate your solution at both a **forecasting level** and a **business level**.

#### 1. Forecasting KPIs
Possible forecasting metrics include:
* MAE
* RMSE
* Error by hour of day
* Error during high-consumption periods
* Comparison with a simple baseline, such as consumption from the same hour one week earlier

*Note:* Because this is a time-series problem, you should use a **chronological evaluation setup**. Randomly shuffling observations into train and test sets is not appropriate.

#### 2. Business KPIs
Forecast accuracy is only a proxy for business value. You should therefore define and evaluate one or more business-oriented KPIs.
Possible KPIs include:
* Total absolute imbalance, for example in MWh
* Percentage reduction in imbalance compared with a baseline forecast
* Frequency of large forecast errors
* Percentage of forecasts within an acceptable error range
* Reliability across different price areas and time periods

If historical spot and balancing/regulating prices are available, you may also estimate:
* Historical imbalance cost
* Reduction in imbalance cost compared with a baseline forecast

*Note:* You are not required to reproduce the complete electricity-market settlement mechanism. A reasonable approximation is sufficient if assumptions are clearly stated.

From Aneo’s perspective, longer-term service KPIs could also include retailer satisfaction and customer retention. These do not need to be directly calculated from the dataset, but you should explain how your forecasting solution could affect them.

---

### Dataset Columns

* **Time:** Hourly timestamp in UTC.
* **Location:** The largest city in the corresponding price area. This is mainly used as a representative location for weather information.
* **Consumption:** Average hourly electricity consumption served by the retailer, measured in MW.
* **Temperature:** Weather forecast for the corresponding location and hour, in degrees Celsius. The temperature value represents a forecast available at the time the consumption prediction is made, not the actual observed temperature afterward. It may therefore be used as an input feature.

---

### Frequently Asked Questions

* **What other datasets could be interesting?**
  You may use similar electricity-consumption datasets from other countries or regions. However, consumption patterns may differ significantly between locations (e.g., heating demand in winter vs. air-conditioning in summer).
* **Are electricity prices relevant as forecasting features?**
  Probably not as a primary feature for this dataset, as households and industry short-term consumption may not respond strongly to hourly prices. However, they can still be very useful for business evaluation.
* **What other forecasting features could be useful?**
  Strong temporal patterns (hour of day, weekday, weekend indicator, month, season, holidays), weather forecasts, and historical consumption lags (remembering the 5-day data delay!).
* **Is the size of the city relevant?**
  Not necessarily. Consumption depends on how many customers the retailer serves in that price area, not on total city population.
* **Are these forecasts used for Aneo’s production planning?**
  No. These forecasts are intended to support the retailer’s electricity-market operations.
* **Is it better to overestimate or underestimate consumption?**
  Neither. Both positive and negative forecasting errors create imbalance and financial uncertainty. Aim for the best unbiased forecast possible.