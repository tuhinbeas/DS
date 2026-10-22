import pandas as pd
import plotly.express as px
import streamlit as st
from src import insights, ui

b = ui.get_bundle()
filters = ui.filters()
st.title("Product forecast")
st.write("Choose a product and store to see recent sales and the expected sales trend.")

categories = sorted(b.meta.cat_id.unique())
category = st.selectbox("Category", categories)
departments = sorted(b.meta.loc[b.meta.cat_id == category, "dept_id"].unique())
department = st.selectbox("Department", departments)
products = sorted(b.meta.loc[b.meta.dept_id == department, "item_id"].unique())
item_id = st.selectbox("Product", products)
stores = sorted(b.meta.loc[b.meta.item_id == item_id, "store_id"].unique())
store_id = st.selectbox("Store", stores)

row_mask = ((b.meta.item_id == item_id) & (b.meta.store_id == store_id)).to_numpy()
history = insights.aggregate(b, filters.model, row_mask)["history"].tail(90)
forecast = insights.aggregate(b, filters.model, row_mask)["forecast"].iloc[:filters.days]
daily = pd.concat(
    [
        history.rename("Units").to_frame().assign(Period="Past sales"),
        forecast.rename("Units").to_frame().assign(Period="Expected sales"),
    ]
).rename_axis("Date").reset_index()

fig = px.line(
    daily,
    x="Date",
    y="Units",
    color="Period",
    title=f"Daily sales: {item_id} at {store_id}",
    labels={"Units": "Units sold", "Period": " "},
)
fig.update_layout(legend_title_text="", margin=dict(t=45))
st.plotly_chart(fig, width="stretch")
st.caption(
    "The first line is recorded sales; the second is the forecast. "
    "The forecast covers the selected number of days from the sidebar."
)

st.metric(f"Expected units over {filters.days} days", ui.fmt_int(forecast.sum()))

download = pd.DataFrame({"Date": forecast.index, "Expected units": forecast.to_numpy()})
st.download_button(
    "Download forecast",
    download.to_csv(index=False),
    file_name=f"forecast_{item_id}_{store_id}.csv",
    mime="text/csv",
)
