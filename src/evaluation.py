"""Metrics. calculate_metrics is the notebook's function, unchanged in behaviour."""
import numpy as np
import pandas as pd


def calculate_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.clip(np.asarray(y_pred, dtype=float).ravel(), 0, None)
    s = y_true.sum()
    return {
        "MAE": float(np.mean(np.abs(y_true - y_pred))),
        "RMSE": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "WAPE": float(np.abs(y_true - y_pred).sum() / s) if s > 0 else 0.0,
        "Bias": float((y_pred.sum() - s) / s) if s > 0 else 0.0,
    }


def horizon_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Error for forecast day 1, 2, ... 90 (errors usually grow with the horizon)."""
    y_pred = np.clip(y_pred, 0, None)
    err = np.abs(y_true - y_pred).sum(axis=0)
    act = y_true.sum(axis=0)
    safe = np.where(act > 0, act, 1)
    return pd.DataFrame({"horizon_day": np.arange(1, y_true.shape[1] + 1),
                         "MAE": np.abs(y_true - y_pred).mean(axis=0),
                         "WAPE": np.where(act > 0, err / safe, np.nan),
                         "Bias": np.where(act > 0, (y_pred.sum(axis=0) - act) / safe, np.nan)})


def group_metrics(y_true, y_pred, labels: pd.Series, group_type: str) -> pd.DataFrame:
    y_pred = np.clip(y_pred, 0, None)
    rows = []
    for g in sorted(labels.unique()):
        m = (labels == g).to_numpy()
        rows.append({"group_type": group_type, "group": g, **calculate_metrics(y_true[m], y_pred[m])})
    return pd.DataFrame(rows)


def series_wape(y_true, y_pred) -> np.ndarray:
    """One WAPE per product-store series; NaN when the series sold nothing."""
    y_pred = np.clip(y_pred, 0, None)
    act = y_true.sum(axis=1)
    err = np.abs(y_true - y_pred).sum(axis=1)
    return np.where(act > 0, err / np.where(act > 0, act, 1), np.nan)
