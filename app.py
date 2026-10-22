"""Entry point: streamlit run app.py."""
import streamlit as st
from src import data_io, ui

st.set_page_config(page_title="M5 Retail Demand Outlook", page_icon=":material/insights:", layout="wide")

if not data_io.artifacts_ready():
    st.title("M5 Retail Demand Outlook")
    st.error("The dashboard has not been prepared yet.")
    st.markdown(
        "To prepare this historical demonstration:\n"
        "1. Add the M5 sales and calendar files to `data/`.\n"
        "2. Add the saved model files to `models/`.\n"
        "3. Run `python build_artifacts.py`, then restart the dashboard."
    )
    st.stop()

b = ui.get_bundle()
pages = [
    st.Page("views/dashboard.py", title="Overview", icon=":material/dashboard:", default=True),
    st.Page("views/forecast_explorer.py", title="Product forecast", icon=":material/query_stats:"),
    st.Page("views/product_store_insights.py", title="Where demand is", icon=":material/storefront:"),
    st.Page("views/model_comparison.py", title="Forecast accuracy", icon=":material/assessment:"),
]
nav = st.navigation(pages)
ui.sidebar_filters(b)
nav.run()
