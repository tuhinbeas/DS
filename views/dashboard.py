import pandas as pd
import plotly.express as px
import streamlit as st
from src import insights, ui

b, f = ui.get_bundle(), ui.filters()
mask = ui.mask(b, f)
st.title("Demand outlook")
if not mask.any():
    st.warning("No products match these filters. Change or clear a filter in the sidebar."); st.stop()

agg = insights.aggregate(b, f.model, mask, f.days)
total_pred = agg["forecast"].sum()
prev = agg["history"].iloc[-f.days:].sum()
champ = insights.champion_row(b, "test")

c1, c2, c3 = st.columns(3)
c1.metric(f"Expected sales over {f.days} days", ui.fmt_int(total_pred),
          delta=ui.fmt_pct((total_pred - prev) / prev, True) + " vs prior period" if prev > 0 else None)
c2.metric("Product/store pairs included", ui.fmt_int(mask.sum()))
c3.metric("Past test forecast error", ui.fmt_pct(champ["WAPE"]),
          help="Typical forecast error as a share of actual sales. Lower is better; this is a historical test, not a guarantee.")

st.subheader("Expected sales over time")
wk_h = insights.to_period(agg["history"], "W", drop_partial=True).tail(26)
wk_f = insights.to_period(agg["forecast"], "W", drop_partial=True)
if len(wk_f) >= 2:
    d = pd.concat([wk_h.rename("units").to_frame().assign(kind="Past sales"),
                   wk_f.rename("units").to_frame().assign(kind="Expected sales")]).rename_axis("week").reset_index()
    fig = px.line(d, x="week", y="units", color="kind", markers=True)
else:
    d = pd.concat([agg["history"].tail(60).rename("units").to_frame().assign(kind="Past sales"),
                   agg["forecast"].rename("units").to_frame().assign(kind="Expected sales")]).rename_axis("week").reset_index()
    fig = px.line(d, x="week", y="units", color="kind")
fig.update_layout(title="Recent sales and expected sales", legend_title_text="", yaxis_title="Units sold",
                  xaxis_title="Week", margin=dict(t=45))
st.plotly_chart(fig, width="stretch")
st.caption("Each point is the total number of units sold or expected in that week. The line changes from past sales to the forecast at the dataset's last sales date.")

items = insights.top_items(b, f.model, mask, f.days)
left, right = st.columns(2)
with left:
    st.subheader("Products with the most expected sales")
    top = items.head(10)
    bar = px.bar(top.iloc[::-1], x="predicted_units", y="item_id", orientation="h",
                 labels={"predicted_units": "Expected units", "item_id": "M5 product code"},
                 title=f"Top products over {f.days} days")
    bar.update_layout(yaxis_title=None, xaxis_title="Expected units sold", margin=dict(t=45))
    st.plotly_chart(bar, width="stretch")
    st.caption("Bars rank product codes by expected sales in the selected period. These are dataset codes, not consumer-facing product names.")
with right:
    st.subheader("Biggest changes from the prior period")
    inc = insights.largest_increases(items, 10, min_previous=50)
    if inc.empty:
        st.info("No product has enough prior-period sales for a useful comparison.")
    else:
        t = inc[["item_id", "previous_units", "predicted_units", "change_pct"]].copy()
        t["change_pct"] = (t.change_pct * 100).round(1)
        st.dataframe(t.rename(columns={"change_pct": "change %"}), hide_index=True, width="stretch")
    st.caption("Change compares expected units with actual sales in the same-length period immediately before the forecast. Very low-volume products are excluded to avoid noisy percentages.")
