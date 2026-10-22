import plotly.express as px
import streamlit as st
from src import ui

b = ui.get_bundle()
st.title("Forecast accuracy")
st.write(
    "Before using a forecast, it helps to see how close the same approaches came "
    "when predicting sales that are now in the past."
)

metrics = b.metrics[b.metrics.window == "test"].sort_values("WAPE").copy()
if metrics.empty:
    st.warning("Past accuracy results are not available.")
    st.stop()

display_names = {
    "Seasonal Naive": "Repeat last week's pattern",
    "LightGBM": "Tree-based forecast",
    "LSTM": "Neural-network forecast",
}
champion = b.manifest["champion"]
chosen = metrics.set_index("model").loc[champion]
st.metric("Selected approach", display_names.get(champion, champion))
st.metric("Difference from actual sales in the past test", ui.fmt_pct(chosen["WAPE"]))
st.caption(
    f"The test compares predictions with recorded sales from {b.manifest['test'][0]} "
    f"to {b.manifest['test'][1]}. In plain terms, the selected forecast's total "
    f"miss was about {chosen['WAPE'] * 100:.0f} units for every 100 units actually sold. "
    "That is a substantial miss; treat the forecast as a demonstration, not a precise plan."
)

metrics["Approach"] = metrics["model"].map(display_names).fillna(metrics["model"])
metrics["Difference (%)"] = metrics["WAPE"] * 100
fig = px.bar(
    metrics.sort_values("Difference (%)"),
    x="Approach",
    y="Difference (%)",
    title="How far each approach was from actual sales",
    labels={"Difference (%)": "Difference from actual sales (%)"},
    text_auto=".1f",
)
fig.update_layout(xaxis_title=None, yaxis_title="Lower is closer", margin=dict(t=45))
st.plotly_chart(fig, width="stretch")
st.caption(
    "Each bar summarizes one approach's total error across products in the same past 90-day period. "
    "The selected approach is the one that performed best in the earlier comparison."
)
