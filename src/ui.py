"""Streamlit glue: cached loading, the shared sidebar filters, small formatters."""
from dataclasses import dataclass, field
import numpy as np
import streamlit as st
from . import data_io, insights


@st.cache_resource(show_spinner="Loading forecast artifacts...")
def get_bundle():
    return data_io.load_bundle()


@dataclass
class Filters:
    stores: list = field(default_factory=list)
    cats: list = field(default_factory=list)
    depts: list = field(default_factory=list)
    model: str = "LightGBM"
    days: int = 90


def sidebar_filters(b) -> Filters:
    """Render shared scope filters; use the validation-selected model by default."""
    st.sidebar.markdown("### Choose what to view")
    st.sidebar.caption("Filters apply across pages. Leave a filter empty to include all.")
    stores = st.sidebar.multiselect("Store", sorted(b.meta.store_id.unique()), key="f_stores")
    cats = st.sidebar.multiselect("Category", sorted(b.meta.cat_id.unique()), key="f_cats")
    dept_opts = sorted(b.meta[b.meta.cat_id.isin(cats)].dept_id.unique() if cats else b.meta.dept_id.unique())
    depts = st.sidebar.multiselect("Department", dept_opts, key="f_depts")
    model = b.manifest["champion"]
    days = st.sidebar.slider("Forecast period (days)", 7, int(b.manifest["horizon"]), int(b.manifest["horizon"]), key="f_days")
    f = Filters(stores, cats, depts, model, days)
    st.session_state["filters"] = f
    return f


def filters() -> Filters:
    return st.session_state["filters"]


def mask(b, f: Filters) -> np.ndarray:
    return insights.series_mask(b.meta, f.stores, f.cats, f.depts)


def fmt_int(x) -> str:
    return f"{x:,.0f}"


def fmt_pct(x, signed=False) -> str:
    return f"{x:+.1%}" if signed else f"{x:.1%}"

