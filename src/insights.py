"""Pure functions that turn the Bundle + sidebar filters into tables/series.
No Streamlit in here, so everything is easy to test."""
import numpy as np
import pandas as pd


def series_mask(meta: pd.DataFrame, stores, cats, depts) -> np.ndarray:
    m = np.ones(len(meta), dtype=bool)
    if stores: m &= meta.store_id.isin(stores).to_numpy()
    if cats:   m &= meta.cat_id.isin(cats).to_numpy()
    if depts:  m &= meta.dept_id.isin(depts).to_numpy()
    return m


def aggregate(b, model: str, mask: np.ndarray, days: int | None = None) -> dict:
    """Total units per day for the selected series: history (actual) and forecast."""
    days = days or b.fdates.size
    return {"history": pd.Series(b.history[mask].sum(axis=0), index=b.hist_dates),
            "forecast": pd.Series(b.forecasts[model][mask, :days].sum(axis=0), index=b.fdates[:days])}


def to_period(s: pd.Series, freq: str, drop_partial: bool = False) -> pd.Series:
    """freq: 'D' daily, 'W' weekly (Mon-Sun), 'M' monthly. Optionally hide incomplete periods."""
    if freq == "D":
        return s
    rule = {"W": "W-SUN", "M": "MS"}[freq]
    g = s.resample(rule)
    out, cnt = g.sum(), g.count()
    if drop_partial:
        need = 7 if freq == "W" else out.index.days_in_month
        out = out[cnt == need]
    return out


def top_items(b, model: str, mask: np.ndarray, days: int) -> pd.DataFrame:
    """Item-level ranking. 'previous' = the same number of days of ACTUAL sales just before the forecast."""
    days = min(days, b.history.shape[1])
    df = b.meta[mask].assign(pred=b.forecasts[model][mask, :days].sum(axis=1),
                             prev=b.history[mask, -days:].sum(axis=1))
    g = df.groupby(["item_id", "dept_id", "cat_id"], as_index=False).agg(
        predicted_units=("pred", "sum"), previous_units=("prev", "sum"))
    g["change_units"] = g.predicted_units - g.previous_units
    g["change_pct"] = np.where(g.previous_units > 0, g.change_units / g.previous_units.replace(0, np.nan), np.nan)
    g[["predicted_units", "previous_units", "change_units"]] = g[["predicted_units", "previous_units", "change_units"]].round(1)
    return g.sort_values("predicted_units", ascending=False).reset_index(drop=True)


def largest_increases(items: pd.DataFrame, n=10, min_previous=50) -> pd.DataFrame:
    """Biggest % rise, ignoring tiny bases (a jump from 2 to 6 units is noise, not insight)."""
    x = items[items.previous_units >= min_previous].dropna(subset=["change_pct"])
    return x.sort_values("change_pct", ascending=False).head(n)


def group_totals(b, model: str, mask: np.ndarray, by: str, days: int) -> pd.DataFrame:
    df = b.meta[mask].assign(pred=b.forecasts[model][mask, :days].sum(axis=1),
                             prev=b.history[mask, -min(days, b.history.shape[1]):].sum(axis=1))
    return df.groupby(by, as_index=False)[["pred", "prev"]].sum().rename(
        columns={"pred": "predicted_units", "prev": "previous_units"}).sort_values("predicted_units", ascending=False)


def item_by_store(b, model: str, item_id: str, days: int) -> pd.DataFrame:
    m = (b.meta.item_id == item_id).to_numpy()
    return b.meta[m].assign(predicted_units=b.forecasts[model][m, :days].sum(axis=1).round(1),
                            previous_units=b.history[m, -min(days, b.history.shape[1]):].sum(axis=1))[
        ["store_id", "predicted_units", "previous_units"]].sort_values("predicted_units", ascending=False)


def weekday_profile(b, model: str, mask: np.ndarray) -> pd.DataFrame:
    s = pd.Series(b.forecasts[model][mask].sum(axis=0), index=b.fdates)
    df = s.groupby(s.index.day_name()).mean().rename("avg_daily_units").rename_axis("weekday").reset_index()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    df["weekday"] = pd.Categorical(df.weekday, order, ordered=True)
    return df.sort_values("weekday")


def series_frame(b, model: str, row_mask: np.ndarray) -> pd.DataFrame:
    """Long frame (date, units, kind) for one or more series summed together."""
    a = aggregate(b, model, row_mask)
    return pd.concat([a["history"].rename("units").to_frame().assign(kind="Actual"),
                      a["forecast"].rename("units").to_frame().assign(kind=f"Forecast ({model})")]
                     ).rename_axis("date").reset_index()


def champion_row(b, window="test") -> pd.Series:
    m = b.metrics[b.metrics.window == window].set_index("model")
    return m.loc[b.manifest["champion"]]


def unusual_days(totals: pd.Series, n=8) -> pd.DataFrame:
    """Days far above/below the trailing 28-day typical level (robust z-score)."""
    med = totals.rolling(28, min_periods=14).median()
    mad = (totals - med).abs().rolling(28, min_periods=14).median().replace(0, np.nan)
    z = ((totals - med) / (1.4826 * mad)).dropna()
    df = pd.DataFrame({"units": totals.loc[z.index], "typical": med.loc[z.index].round(0), "score": z.round(1)})
    return pd.concat([df.nlargest(n, "score").assign(kind="Unusually high"),
                      df.nsmallest(n, "score").assign(kind="Unusually low")]).rename_axis("date").reset_index()
