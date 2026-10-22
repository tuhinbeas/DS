"""The forecasting engine. ONE place that builds features, so the dashboard
numbers use exactly the transformations the models were trained with."""
import pickle
import numpy as np
import pandas as pd
from . import config as C

SNAP = ["snap_CA", "snap_TX", "snap_WI"]


def calendar_features(calendar: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """wday/month/snap for each date. M5's calendar ends 2016-06-19; later dates are
    derived: wday & month from the date, SNAP copied from the latest known
    same-day-of-month (SNAP days follow a monthly pattern)."""
    cal = calendar.drop_duplicates("date").set_index("date")[C.CAL_FEATURES]
    snap_by_dom = cal[SNAP].groupby(cal.index.day).last()
    rows = []
    for dt in dates:
        if dt in cal.index:
            rows.append(cal.loc[dt].to_dict())
        else:
            row = {"wday": (dt.weekday() + 2) % 7 + 1, "month": dt.month}  # M5: 1=Saturday
            row.update(snap_by_dom.loc[dt.day].to_dict())
            rows.append(row)
    return pd.DataFrame(rows, index=dates)


def seasonal_naive(last_week: np.ndarray, horizon: int = C.HORIZON) -> np.ndarray:
    reps = int(np.ceil(horizon / 7))
    return np.tile(last_week, (1, reps))[:, :horizon]


def recursive_forecast(last28: np.ndarray, cal_feats: pd.DataFrame, predict_fn) -> np.ndarray:
    """Day-by-day forecast. Lags/rolling means that point into the forecast window use
    the model's OWN earlier predictions (never actual future sales).
    last28: (n_series, 28) most recent actual sales before the forecast origin."""
    n, h = last28.shape[0], len(cal_feats)
    buf = np.concatenate([last28.astype(np.float64), np.zeros((n, h))], axis=1)
    off = last28.shape[1]
    for t in range(h):
        X = {f"lag_{k}": buf[:, off + t - k] for k in C.LAGS}
        for w in C.WINDOWS:
            X[f"rolling_mean_{w}"] = buf[:, off + t - w: off + t].mean(axis=1)
        for c in C.CAL_FEATURES:
            X[c] = np.full(n, cal_feats.iloc[t][c])
        buf[:, off + t] = predict_fn(pd.DataFrame(X)[C.FEATURE_COLS])
    return buf[:, off:]


# ---------- model loading ----------
def load_predictors() -> tuple[dict, dict]:
    """Returns ({model name: predict_fn(DataFrame)->array}, info). LSTM is skipped
    (with a note in `info`) if TensorFlow/Keras is not installed."""
    preds, info = {}, {"skipped": {}}
    with open(C.LGB_PATH, "rb") as f:
        lgb = pickle.load(f)
    preds["LightGBM"] = lambda X: lgb.predict(X)
    info["lightgbm_params"] = {k: v for k, v in lgb.get_params().items()
                               if k in ("objective", "n_estimators", "learning_rate", "num_leaves", "random_state")}
    with open(C.SCALER_PATH, "rb") as f:
        sc = pickle.load(f)
    fs, ts = sc["features_scaler"], sc["target_scaler"]
    try:
        from tensorflow import keras
        lstm = keras.models.load_model(C.LSTM_PATH)

        def lstm_fn(X):
            Xs = X.copy()
            Xs[C.NUM_FEATURES] = fs.transform(X[C.NUM_FEATURES])
            p = lstm.predict(np.expand_dims(Xs.to_numpy(dtype="float32"), 1), batch_size=8192, verbose=0)
            return ts.inverse_transform(p).ravel()
        preds["LSTM"] = lstm_fn
        lines = []
        lstm.summary(print_fn=lines.append)
        info["lstm_summary"] = "\n".join(lines)
    except Exception as e:  # pragma: no cover
        info["skipped"]["LSTM"] = f"{type(e).__name__}: {e}"
    return preds, info
