import sys, tempfile, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from tests.make_synthetic import make, stand_in_predictors
from build_artifacts import build
from src import config as C, forecasting as F, data_io, insights

def test_recursive_matches_manual_notebook_logic():
    """Vectorised recursion must equal the notebook's step-by-step loop."""
    rng = np.random.default_rng(1); last28 = rng.poisson(2, (5, 28)).astype(float)
    cal = pd.DataFrame({"wday": [1,2,3], "month": [1,1,1], "snap_CA": [0,1,0], "snap_TX": [0,0,0], "snap_WI": [1,1,1]})
    fn = lambda X: 0.4 * X["lag_1"].to_numpy() + 0.3 * X["lag_7"].to_numpy() + 0.3 * X["rolling_mean_28"].to_numpy()
    got = F.recursive_forecast(last28, cal, fn)
    preds = np.zeros((5, 3))
    for d in range(3):
        def pick(ptr): return preds[:, ptr] if ptr >= 0 else last28[:, ptr]
        lag = {k: pick(d - k) for k in (1, 7, 14, 28)}
        r28 = np.mean([pick(d - o) for o in range(1, 29)], axis=0)
        preds[:, d] = 0.4 * lag[1] + 0.3 * lag[7] + 0.3 * r28
    assert np.allclose(got, preds)

def test_calendar_beyond_known_dates():
    _, cal = make(4)
    f = F.calendar_features(cal, pd.date_range("2016-06-20", periods=7))   # past calendar end
    assert list(f.columns) == C.CAL_FEATURES and f.wday.between(1, 7).all()
    assert f.loc["2016-06-25", "wday"] == 1          # a Saturday -> 1 in M5

def test_full_build_and_app_logic():
    sales, cal = make(120); preds, info = stand_in_predictors()
    with tempfile.TemporaryDirectory() as td:
        out = pathlib.Path(td)
        man = build(sales, cal, preds, info, out, log=lambda *a: None)
        assert man["forecast"] == ["2016-04-25", "2016-07-23"] and man["n_series"] == 120
        b = data_io.load_bundle(out)
        assert set(b.forecasts) == set(C.MODELS) and b.forecasts["LightGBM"].shape == (120, 90)
        assert (b.forecasts["LightGBM"] >= 0).all()
        mask = insights.series_mask(b.meta, [], ["FOODS"], [])
        tot = insights.top_items(b, "LightGBM", mask, 90).head(10)
        assert len(tot) == 10 and tot.predicted_units.is_monotonic_decreasing
        s = insights.aggregate(b, "LightGBM", mask)
        assert len(s["forecast"]) == 90 and len(s["history"]) == C.HISTORY_DAYS
        w = insights.to_period(s["forecast"], "W", drop_partial=True)
        assert (w > 0).all() and len(w) == 12
        assert b.metrics.query("window=='test'").WAPE.between(0, 5).all()
        assert b.manifest["champion"] in C.MODELS
        print("champion:", b.manifest["champion"]); print(b.metrics.round(3).to_string())

if __name__ == "__main__":
    test_recursive_matches_manual_notebook_logic(); test_calendar_beyond_known_dates()
    test_full_build_and_app_logic(); print("ALL PASSED")
