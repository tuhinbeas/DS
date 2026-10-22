"""Loads the files written by build_artifacts.py into one Bundle object."""
import json
from dataclasses import dataclass
import numpy as np
import pandas as pd
from . import config as C


@dataclass
class Bundle:
    meta: pd.DataFrame            # one row per product-store series
    history: np.ndarray           # (n_series, HISTORY_DAYS) int16, most recent actual sales
    hist_dates: pd.DatetimeIndex
    fdates: pd.DatetimeIndex      # the 90 forecast dates
    forecasts: dict               # model name -> (n_series, 90) float32
    metrics: pd.DataFrame         # overall metrics, both windows
    by_horizon: pd.DataFrame
    by_group: pd.DataFrame
    daily_agg: pd.DataFrame
    risk: pd.DataFrame
    store_dept_history: pd.DataFrame
    calendar: pd.DataFrame
    manifest: dict


def artifacts_ready(out_dir=C.OUT_DIR) -> bool:
    return (out_dir / "manifest.json").exists()


def load_bundle(out_dir=C.OUT_DIR) -> Bundle:
    if not artifacts_ready(out_dir):
        raise FileNotFoundError("outputs/manifest.json not found - run `python build_artifacts.py` first.")
    man = json.loads((out_dir / "manifest.json").read_text())
    npz = np.load(out_dir / "forecasts_future.npz")
    inv = {v: k for k, v in C.MODEL_KEYS.items()}
    fc = {inv[k]: npz[k] for k in npz.files}
    hist = np.load(out_dir / "history_recent.npy")
    end = pd.Timestamp(man["data_end"])
    meta = pd.read_csv(out_dir / "series_meta.csv")
    return Bundle(
        meta=meta, history=hist,
        hist_dates=pd.date_range(end=end, periods=hist.shape[1]),
        fdates=pd.date_range(man["forecast"][0], periods=man["horizon"]),
        forecasts=fc,
        metrics=pd.read_csv(out_dir / "validation_metrics.csv"),
        by_horizon=pd.read_csv(out_dir / "metrics_by_horizon.csv"),
        by_group=pd.read_csv(out_dir / "metrics_by_group.csv"),
        daily_agg=pd.read_csv(out_dir / "daily_aggregates.csv", parse_dates=["date"]),
        risk=pd.read_csv(out_dir / "series_risk.csv").set_index("id"),
        store_dept_history=pd.read_csv(out_dir / "history_daily_store_dept.csv", parse_dates=["date"]),
        calendar=pd.read_csv(out_dir / "calendar_slim.csv", parse_dates=["date"]),
        manifest=man,
    )
