"""Run ONCE (or whenever models/data change):   python build_artifacts.py

Reads data/ + models/, produces every number the dashboard shows in outputs/.
The Streamlit app never trains or forecasts live; it only reads outputs/.
Windows (all recursive 90-day forecasts, no actual future sales used as inputs):
  validation : origin 2015-10-27  -> picks the champion model
  test       : origin 2016-01-25  -> final, untouched evaluation
  future     : origin 2016-04-24  -> the dashboard's 90-day outlook
"""
import argparse, hashlib, json, platform, time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from src import config as C
from src import evaluation as E
from src import forecasting as F


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12] if path.exists() else None


def _lib_versions():
    out = {"python": platform.python_version()}
    for lib in ("numpy", "pandas", "sklearn", "lightgbm", "tensorflow", "keras"):
        try:
            out[lib] = __import__(lib).__version__
        except Exception:
            pass
    return out


def build(sales, calendar, predictors, info, out_dir, log=print):
    out_dir.mkdir(parents=True, exist_ok=True)
    scols = [c for c in sales.columns if c.startswith("d_")]
    sales[scols] = sales[scols].fillna(0)                      # same cleaning as the notebook
    day_date = dict(zip(calendar["d"], calendar["date"]))
    dates = pd.DatetimeIndex([day_date[c] for c in scols])
    M = sales[scols].to_numpy(dtype=np.float32)
    meta = sales[["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]].reset_index(drop=True)
    pos = lambda ts: dates.get_loc(pd.Timestamp(ts))
    models = [m for m in C.MODELS if m == "Seasonal Naive" or m in predictors]

    def run(origin):
        end = pos(origin)
        last28 = M[:, end - 27: end + 1]
        fdates = pd.date_range(origin + pd.Timedelta(days=1), periods=C.HORIZON)
        cal = F.calendar_features(calendar, fdates)
        res = {}
        for m in models:
            t0 = time.time()
            if m == "Seasonal Naive":
                p = F.seasonal_naive(last28[:, -7:])
            else:
                p = F.recursive_forecast(last28, cal, predictors[m])
            res[m] = (np.clip(p, 0, None).astype(np.float32), time.time() - t0)
            log(f"  {m:15s} origin {origin.date()}  {res[m][1]:.1f}s")
        actual = M[:, end + 1: end + 1 + C.HORIZON] if end + C.HORIZON < len(dates) else None
        return fdates, res, actual

    overall, horizon_rows, group_rows, daily_rows = [], [], [], []
    windows = {"validation": C.TRAIN_END, "test": C.VAL_END}
    results = {}
    for wname, origin in windows.items():
        log(f"[{wname}]")
        fdates, res, actual = run(origin)
        assert actual is not None and actual.shape[1] == C.HORIZON, "need 90 actual days after origin"
        results[wname] = (fdates, res, actual)
        for m, (p, secs) in res.items():
            overall.append({"window": wname, "model": m, **E.calculate_metrics(actual, p),
                            "forecast_seconds": round(secs, 2),
                            "train_seconds": C.NOTEBOOK_TRAIN_SECONDS.get(m)})
            h = E.horizon_metrics(actual, p); h.insert(0, "model", m); h.insert(0, "window", wname)
            horizon_rows.append(h)
            for gt in ("cat_id", "dept_id", "store_id"):
                g = E.group_metrics(actual, p, meta[gt], gt); g.insert(0, "model", m); g.insert(0, "window", wname)
                group_rows.append(g)
        for gt in ("total", "cat_id", "dept_id", "store_id"):
            labels = pd.Series("All", index=meta.index) if gt == "total" else meta[gt]
            series = {"Actual": actual, **{m: res[m][0] for m in res}}
            for name, mat in series.items():
                agg = pd.DataFrame(mat).groupby(labels.to_numpy()).sum()
                for g, row in agg.iterrows():
                    daily_rows.append(pd.DataFrame({"window": wname, "group_type": gt, "group": g,
                                                    "series": name, "date": fdates, "units": row.to_numpy()}))
    overall = pd.DataFrame(overall)
    val = overall[overall.window == "validation"].set_index("model")
    champion = val["WAPE"].idxmin()
    log(f"Champion (lowest validation WAPE): {champion}")

    # per-series historical error label (relative: terciles of the champion's TEST error)
    tf, tres, tact = results["test"]
    sw = E.series_wape(tact, tres[champion][0])
    lo, hi = np.nanquantile(sw, [1 / 3, 2 / 3])
    label = np.where(np.isnan(sw), "No sales in test window",
             np.where(sw <= lo, "Lower historical error", np.where(sw <= hi, "Medium historical error", "Higher historical error")))
    pd.DataFrame({"id": meta["id"], "test_wape": sw, "risk_label": label}).to_csv(out_dir / "series_risk.csv", index=False)

    # ---- future forecast (the dashboard's 90-day outlook) ----
    log("[future]")
    origin = dates[-1]
    ffdates, fres, _ = run(origin)
    np.savez_compressed(out_dir / "forecasts_future.npz", **{C.MODEL_KEYS[m]: fres[m][0] for m in fres})

    # ---- compact history + small tables for the app ----
    np.save(out_dir / "history_recent.npy", M[:, -C.HISTORY_DAYS:].astype(np.int16))
    meta.to_csv(out_dir / "series_meta.csv", index=False)
    pd.concat(horizon_rows).to_csv(out_dir / "metrics_by_horizon.csv", index=False)
    pd.concat(group_rows).to_csv(out_dir / "metrics_by_group.csv", index=False)
    pd.concat(daily_rows).to_csv(out_dir / "daily_aggregates.csv", index=False)
    overall.to_csv(out_dir / "validation_metrics.csv", index=False)    # both windows, column 'window'
    parts = []
    for (st, dp), idx in meta.groupby(["store_id", "dept_id"]).groups.items():
        parts.append(pd.DataFrame({"date": dates, "store_id": st, "dept_id": dp, "units": M[idx].sum(axis=0)}))
    agg = pd.concat(parts, ignore_index=True)
    agg.to_csv(out_dir / "history_daily_store_dept.csv", index=False)
    cal_cols = [c for c in ["date", "wday", "month", "year", "weekday", "event_name_1", "event_type_1"] if c in calendar.columns]
    calendar[cal_cols].to_csv(out_dir / "calendar_slim.csv", index=False)

    manifest = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_series": int(len(meta)), "n_days": int(len(dates)),
        "data_start": str(dates[0].date()), "data_end": str(dates[-1].date()),
        "train": [str(dates[0].date()), str(C.TRAIN_END.date())],
        "validation": [str(C.VAL_START.date()), str(C.VAL_END.date())],
        "test": [str(C.TEST_START.date()), str(C.TEST_END.date())],
        "forecast": [str(ffdates[0].date()), str(ffdates[-1].date())],
        "history_days_kept": C.HISTORY_DAYS, "horizon": C.HORIZON,
        "models_available": models, "champion": champion, "skipped_models": info.get("skipped", {}),
        "features": C.FEATURE_COLS, "lightgbm_params": info.get("lightgbm_params", {}),
        "lstm_summary": info.get("lstm_summary", ""),
        "model_file_hashes": {p.name: _hash(p) for p in (C.LGB_PATH, C.LSTM_PATH, C.SCALER_PATH)},
        "library_versions": _lib_versions(),
        "null_values_filled_with_zero": True,
        "snap_beyond_calendar": "copied from latest known same day-of-month" if ffdates[-1] > calendar["date"].max() else "from calendar",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log(f"Done. Artifacts in {out_dir}")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="use only N random series (quick dry run)")
    a = ap.parse_args()
    for p in (C.SALES_CSV, C.CALENDAR_CSV):
        if not p.exists():
            raise SystemExit(f"Missing {p}. Copy the M5 CSVs into data/ first.")
    sales = pd.read_csv(C.SALES_CSV)
    if a.limit:
        sales = sales.sample(a.limit, random_state=42).reset_index(drop=True)
    calendar = pd.read_csv(C.CALENDAR_CSV, parse_dates=["date"])
    predictors, info = F.load_predictors()
    if info["skipped"]:
        print("WARNING skipped:", info["skipped"])
    build(sales, calendar, predictors, info, C.OUT_DIR)


if __name__ == "__main__":
    main()
