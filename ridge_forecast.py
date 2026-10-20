"""Alternative model: one regularized linear model (RidgeCV) per city. Same features/split as lgbm_forecast.py."""
import numpy as np 
import pandas as pd
import holidays
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# variables
DELAY_H, TEST_DAYS, N_FOLDS = 144, 90, 4

# ---------- data + features (identical to the LightGBM script) ----------
df = pd.read_csv("consumption_temp.csv")
df["time"] = pd.to_datetime(df["time"], utc=True)
df = df.drop_duplicates(["time", "location"]).sort_values(["location", "time"])

# add the features 
def add_features(g):
    
    # set the time as index
    g = g.set_index("time").asfreq("h")

    # filling loaction if missing 
    g["location"] = g["location"].ffill().bfill()

    # adding the features with lag
    for lag in (DELAY_H, 168, 192, 336):
        g[f"lag_{lag}"] = g["consumption"].shift(lag)
    g["roll_mean_7d"] = g["consumption"].shift(DELAY_H).rolling(168).mean()
    g["roll_std_7d"] = g["consumption"].shift(DELAY_H).rolling(168).std()
    
    # forecasts are known for the target dayc
    g["temp_day_mean"] = g["temperature"].groupby(g.index.date).transform("mean")
    g["temp_lag_24"] = g["temperature"].shift(24)
    
    # diff between the temp forcasted for the consumption hour with the same hour 6/7 days ago
    for lag in (DELAY_H, 168):
        g[f"temp_diff_{lag}"] = g["temperature"] - g["temperature"].shift(lag)

    return g.reset_index()

# adding the feature for each location
df = pd.concat([add_features(g) for _, g in df.groupby("location")], ignore_index=True)

# adding the holidays, days off, hours and month flags
loc_hol = {l: holidays.country_holidays("FI" if l.upper().startswith(("HEL", "FI")) else "NO")
           for l in df["location"].unique()}
t = df["time"]
df["hour"], df["dow"], df["month"] = t.dt.hour, t.dt.dayofweek, t.dt.month
df["is_weekend"] = (df["dow"] >= 5).astype(int)
df["is_holiday"] = [int(d in loc_hol[l]) for d, l in zip(t.dt.date, df["location"])]
df["temp_sq"] = df["temperature"] ** 2             # heating/cooling is non-linear in temperature
df = df.dropna().sort_values("time").reset_index(drop=True)   # same rows as the LightGBM test set

CAT = ["hour", "dow", "month"]
NUM = [c for c in df.columns if c not in ["time", "location", "consumption", *CAT]]
cut = df["time"].max() - pd.Timedelta(days=TEST_DAYS)


# ---------- one RidgeCV per city (alpha chosen by expanding-window CV on train) ----------
parts = []
for loc, d in df.groupby("location"):
    tr, te = d[d["time"] <= cut], d[d["time"] > cut].copy()
    cv = TimeSeriesSplit(n_splits=N_FOLDS, test_size=len(tr) // (N_FOLDS + 2), gap=24 * 6)
    model = make_pipeline(
        ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
                           ("num", StandardScaler(), NUM)]),
        RidgeCV(alphas=np.logspace(-3, 3, 25), cv=cv, scoring="neg_mean_absolute_error"))
    model.fit(tr[CAT + NUM], tr["consumption"])
    te["pred"] = model.predict(te[CAT + NUM])
    print(f"{loc:12s} alpha={model[-1].alpha_:.3g}")
    parts.append(te)

test = pd.concat(parts)
test["err"] = test["pred"] - test["consumption"]

# ---------- metrics (same as LightGBM) ----------
rmse = lambda d: np.sqrt(mean_squared_error(d["consumption"], d["pred"]))
print(f"\nTEST  MAE={test['err'].abs().mean():.4f}  RMSE={rmse(test):.4f}  bias={test['err'].mean():.4f}")
print("\nBy location:\n", test.groupby("location").apply(
    lambda d: pd.Series({"MAE": d["err"].abs().mean(), "RMSE": rmse(d),
                         "nMAE_%": 100 * d["err"].abs().mean() / d["consumption"].mean(),
                         "bias": d["err"].mean()}), include_groups=False).round(4))
print("\nMAE by hour:\n", test.groupby("hour")["err"].apply(lambda e: e.abs().mean()).round(4).to_string())
test[["time", "location", "consumption", "pred", "err", "hour"]].to_csv("ridge_test_predictions.csv", index=False)
