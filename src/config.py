"""Central settings. Every path, date and feature name lives here so the
app and the build script can never disagree with each other (or the notebook)."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR, MODEL_DIR, OUT_DIR = ROOT / "data", ROOT / "models", ROOT / "outputs"

SALES_CSV = DATA_DIR / "sales_train_validation.csv"
CALENDAR_CSV = DATA_DIR / "calendar.csv"
LGB_PATH = MODEL_DIR / "lightgbm_champion_model.pkl"
LSTM_PATH = MODEL_DIR / "lstm_neural_network_model.keras"
SCALER_PATH = MODEL_DIR / "minmax_scalers.pkl"

HORIZON = 90
HISTORY_DAYS = 365          # per-series history kept for the app (keeps files small)

# Windows copied from the notebook (Part 2)
TRAIN_END = pd.Timestamp("2015-10-27")
VAL_START, VAL_END = pd.Timestamp("2015-10-28"), pd.Timestamp("2016-01-25")
TEST_START, TEST_END = pd.Timestamp("2016-01-26"), pd.Timestamp("2016-04-24")

# Feature order MUST match the notebook (the models were trained on this order)
NUM_FEATURES = ["lag_1", "lag_7", "lag_14", "lag_28", "rolling_mean_7", "rolling_mean_28"]
CAL_FEATURES = ["wday", "month", "snap_CA", "snap_TX", "snap_WI"]
FEATURE_COLS = NUM_FEATURES + CAL_FEATURES
LAGS, WINDOWS = [1, 7, 14, 28], [7, 28]

MODELS = ["Seasonal Naive", "LightGBM", "LSTM"]
MODEL_KEYS = {"Seasonal Naive": "snaive", "LightGBM": "lightgbm", "LSTM": "lstm"}

# Training times printed in the notebook post-mortem (not stored in the model files)
NOTEBOOK_TRAIN_SECONDS = {"Seasonal Naive": 0.0, "LightGBM": 10.6, "LSTM": 6.5}
