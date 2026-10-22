"""Runs every page with a FAKE streamlit/plotly to catch Python errors (typos, bad
column names, empty-data crashes). It cannot catch Streamlit API misuse: launch the real app too."""
import sys, types, pathlib, tempfile, runpy
from unittest.mock import MagicMock
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tests.make_synthetic import make, stand_in_predictors

st = MagicMock(name="streamlit")
st.session_state = {}
st.cache_resource = lambda *a, **k: (lambda fn: fn)
class Box:  # a column/tab: forwards widget calls to the fake st, works in `with`
    def __getattr__(self, n): return getattr(st, n)
    def __enter__(self): return self
    def __exit__(self, *a): return False
st.columns = lambda spec, **k: [Box() for _ in range(spec if isinstance(spec, int) else len(spec))]
st.tabs = lambda labels: [Box() for _ in labels]
st.selectbox = lambda label, options, index=0, **k: list(options)[index]
st.radio = lambda label, options, **k: list(options)[0]
st.multiselect = lambda label, options, default=None, **k: list(default or [])
st.select_slider = lambda label, options=None, value=None, **k: value
st.slider = lambda label, lo, hi, value=None, **k: value if value is not None else lo
sys.modules["streamlit"] = st
px = MagicMock(); sys.modules["plotly"] = MagicMock(); sys.modules["plotly.express"] = px

from build_artifacts import build
from src import data_io, ui
sales, cal = make(120); preds, info = stand_in_predictors()
with tempfile.TemporaryDirectory() as td:
    out = pathlib.Path(td); build(sales, cal, preds, info, out, log=lambda *a: None)
    bundle = data_io.load_bundle(out)
ui.get_bundle = lambda: bundle

def run_all(filters):
    st.session_state["filters"] = filters
    for p in sorted((ROOT / "views").glob("*.py")):
        runpy.run_path(str(p)); print("ok", p.name, filters.cats or "(all)")

run_all(ui.Filters(model=bundle.manifest["champion"]))
run_all(ui.Filters(stores=["CA_1"], cats=["FOODS"], depts=["FOODS_1"], model="LSTM", days=14))
print("SMOKE PASSED")
