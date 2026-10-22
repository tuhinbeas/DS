import plotly.express as px
import streamlit as st
from src import insights, ui

b, filters = ui.get_bundle(), ui.filters()
mask = ui.mask(b, filters)
st.title("Where demand is")
st.write("Compare expected sales across categories, departments, or stores.")
if not mask.any():
    st.warning("No sales match these filters. Clear or change a filter in the sidebar.")
    st.stop()

group_by = st.radio(
    "Compare",
    ["cat_id", "dept_id", "store_id"],
    horizontal=True,
    format_func=lambda value: {
        "cat_id": "Categories",
        "dept_id": "Departments",
        "store_id": "Stores",
    }[value],
)
groups = insights.group_totals(b, filters.model, mask, group_by, filters.days)
groups = groups.rename(columns={"previous_units": "Prior-period sales", "predicted_units": "Expected sales"})
groups = groups.melt(
    id_vars=group_by,
    value_vars=["Prior-period sales", "Expected sales"],
    var_name="Period",
    value_name="Units",
)
fig = px.bar(
    groups,
    x=group_by,
    y="Units",
    color="Period",
    barmode="group",
    title="Recorded and expected sales",
    labels={
        group_by: {
            "cat_id": "Category",
            "dept_id": "Department",
            "store_id": "Store",
        }[group_by],
        "Units": "Units sold",
    },
)
fig.update_layout(legend_title_text="", margin=dict(t=45))
st.plotly_chart(fig, width="stretch")
st.caption(
    f"Each group compares expected units with actual sales in the previous {filters.days}-day period. "
    "Use this to see where sales are concentrated and how expected totals compare with the recent past."
)

st.subheader("Products with the most expected sales")
products = insights.top_items(b, filters.model, mask, filters.days).head(10)
products = products[["item_id", "dept_id", "previous_units", "predicted_units"]].rename(
    columns={
        "item_id": "Product code",
        "dept_id": "Department",
        "previous_units": "Prior-period units",
        "predicted_units": "Expected units",
    }
)
st.dataframe(products, hide_index=True, width="stretch")
st.caption("Product identifiers are the codes used in the M5 dataset.")
