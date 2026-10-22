"""Tiny M5-shaped fake data + stand-in predictors, so the pipeline can be tested
without the real CSVs or the real models."""
import numpy as np, pandas as pd

def make(n=120, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2011-01-29", "2016-06-19")           # real M5 calendar span (1969 days)
    cal = pd.DataFrame({"date": dates, "d": [f"d_{i+1}" for i in range(len(dates))]})
    cal["wday"] = (dates.weekday + 2) % 7 + 1
    cal["month"], cal["year"], cal["weekday"] = dates.month, dates.year, dates.day_name()
    for c in ("snap_CA", "snap_TX", "snap_WI"):
        cal[c] = (dates.day <= 10).astype(int)
    cal["event_name_1"] = np.where((dates.month == 12) & (dates.day == 25), "Christmas", None)
    cal["event_type_1"] = np.where(cal.event_name_1.notna(), "National", None)
    stores = ["CA_1", "CA_2", "TX_1", "WI_1"]; depts = ["FOODS_1", "FOODS_2", "HOBBIES_1", "HOUSEHOLD_1"]
    rows = []
    for i in range(n):
        st, dp = stores[i % 4], depts[(i // 4) % 4]
        rows.append({"id": f"{dp}_{i:03d}_{st}_validation", "item_id": f"{dp}_{i:03d}", "dept_id": dp,
                     "cat_id": dp.split("_")[0], "store_id": st, "state_id": st.split("_")[0]})
    sales = pd.DataFrame(rows)
    nd = 1913
    base = rng.gamma(1.5, 1.2, n)[:, None]
    wk = 1 + 0.3 * np.sin(2 * np.pi * np.arange(nd) / 7)
    mat = rng.poisson(base * wk).astype(float)
    mat[rng.random(mat.shape) < 0.01] = np.nan                   # a few nulls like the real data
    sales = pd.concat([sales, pd.DataFrame(mat, columns=[f"d_{i+1}" for i in range(nd)])], axis=1)
    return sales, cal

def stand_in_predictors():
    f = lambda X: 0.5 * X["lag_7"].to_numpy() + 0.5 * X["rolling_mean_28"].to_numpy()
    g = lambda X: 0.9 * X["rolling_mean_7"].to_numpy()
    return {"LightGBM": f, "LSTM": g}, {"skipped": {}, "lightgbm_params": {"n_estimators": 100}}
