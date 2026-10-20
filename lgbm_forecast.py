"""LightGBM day-ahead forecast: features (5-day delay safe), Optuna tuning, chronological test."""
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna, holidays
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error


 # forecasting D, last known consumption is D-6 days at same hour
DELAY_H = 144 

# varibales for: num test days, folds, trials, randomseed        
TEST_DAYS, N_FOLDS, N_TRIALS, SEED = 90, 4, 40, 42

# ---------- data ----------
df = pd.read_csv("consumption_temp.csv", parse_dates=["time"])
df["time"] = pd.to_datetime(df["time"], utc=True)
df = df.drop_duplicates(["time", "location"]).sort_values(["location", "time"])

# adding the features
def add_features(g):
    '''
    Add the features to the datset
    '''
    
    # setting the time column as index
    g = g.set_index("time").asfreq("h")
    g["location"] = g["location"].ffill().bfill()
    
    # adding the consumption and temperature features
    #    all >= 144h -> no leakage
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
df["location"] = df["location"].astype("category")


FEATS = [c for c in df.columns if c not in ("time", "consumption")]

# cleaning the data
df = df.dropna(subset=["consumption", "lag_336", "roll_mean_7d"]).sort_values("time").reset_index(drop=True)


# ---------- chronological split ----------
cut = df["time"].max() - pd.Timedelta(days=TEST_DAYS)
train, test = df[df["time"] <= cut], df[df["time"] > cut]
X, y = train[FEATS], train["consumption"]


# ---------- tuning (expanding-window CV on train, early stopping per fold) ----------
tscv = TimeSeriesSplit(n_splits=N_FOLDS, test_size=len(train) // (N_FOLDS + 2), gap=24 * 6 * df["location"].nunique())

# tune the models using various hyperparameters
def objective(trial):
    
    # parameters
    p = dict(
        objective="regression", n_estimators=2000, verbosity=-1, random_state=SEED,
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 127),
        max_depth=trial.suggest_int("max_depth", 4, 12),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 100),
        subsample=trial.suggest_float("subsample", 0.6, 1.0), subsample_freq=1,
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
    )
    maes, iters = [], []
    
    for tr, va in tscv.split(X):
        m = lgb.LGBMRegressor(**p).fit(
            X.iloc[tr], y.iloc[tr], eval_X=X.iloc[va], eval_y=y.iloc[va], eval_metric="l1",
            callbacks=[lgb.early_stopping(100, verbose=False)])
        maes.append(mean_absolute_error(y.iloc[va], m.predict(X.iloc[va])))
        iters.append(m.best_iteration_)
    trial.set_user_attr("n_estimators", int(np.median(iters)))
    return np.mean(maes)

study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=N_TRIALS)
best = {**study.best_params, "n_estimators": study.best_trial.user_attrs["n_estimators"]}
print("CV MAE:", round(study.best_value, 4), "\nBest params:", best)


# ---------- final fit + test ----------
model = lgb.LGBMRegressor(objective="regression", subsample_freq=1, verbosity=-1,
                          random_state=SEED, **best).fit(X, y)
test = test.assign(pred=model.predict(test[FEATS]))
test["err"] = test["pred"] - test["consumption"]
 
rmse = lambda d: np.sqrt(mean_squared_error(d["consumption"], d["pred"]))
print(f"\nTEST  MAE={test['err'].abs().mean():.4f}  RMSE={rmse(test):.4f}  bias={test['err'].mean():.4f}")
print("\nBy location:\n", test.groupby("location", observed=True).apply(
    lambda d: pd.Series({"MAE": d["err"].abs().mean(), "RMSE": rmse(d),
                         "nMAE_%": 100 * d["err"].abs().mean() / d["consumption"].mean(),
                         "bias": d["err"].mean()}), include_groups=False).round(4))
print("\nMAE by hour:\n", test.groupby("hour")["err"].apply(lambda e: e.abs().mean()).round(4).to_string())
 
imp = pd.Series(model.booster_.feature_importance("gain"), FEATS).sort_values(ascending=False)
print("\nFeature importance (gain %):\n", (100 * imp / imp.sum()).round(1).head(10).to_string())

 # reusable for later business KPIs
test.to_csv("lgbm_test_predictions.csv", index=False)  


'''
Best params: 
    - learning_rate: 0.056568631938395224 
    - num_leaves: 69
    - max_depth: 4
    - min_child_samples: 14
    - subsample: 0.7922543545603272
    - colsample_bytree: 0.5202080346852211
    - reg_alpha: 0.07258106225267884
    - reg_lambda: 0.0036156052342859795
    - n_estimators: 414

CV MAE: 0.5554 
TEST  MAE=0.2778  
RMSE=0.4272  
bias=0.0801
'''