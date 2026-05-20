"""
config/settings.py
Central configuration for the PrimeTrade Sentiment Analysis project.
"""
from pathlib import Path

# ─── Project Root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

# ─── Data Paths ────────────────────────────────────────────────────────────────
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "reports" / "figures"

TRADES_FILE = RAW_DIR / "historical_data.csv"
FEAR_GREED_FILE = RAW_DIR / "fear_greed_index.csv"
MERGED_FILE = PROCESSED_DIR / "merged_dataset.csv"

# ─── Sentiment Config ──────────────────────────────────────────────────────────
SENTIMENT_ORDER = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]

SENTIMENT_COLORS = {
    "Extreme Fear": "#d62728",
    "Fear":         "#ff7f0e",
    "Neutral":      "#7f7f7f",
    "Greed":        "#2ca02c",
    "Extreme Greed":"#1f77b4",
}

SENTIMENT_BINS = {
    (0,  20):  "Extreme Fear",
    (21, 40):  "Fear",
    (41, 60):  "Neutral",
    (61, 80):  "Greed",
    (81, 100): "Extreme Greed",
}

# ─── Analysis Config ───────────────────────────────────────────────────────────
ROLLING_WINDOW_DAYS = 7
LAG_DAYS = [0, 1, 2, 3, 5, 7, 14]
MIN_TRADES_FOR_PROFILING = 20       # Min trades to include a trader in profiling
LEVERAGE_BINS = [0, 2, 5, 10, 25, 100]
LEVERAGE_LABELS = ["1-2x", "3-5x", "6-10x", "11-25x", "26x+"]

# ─── Model Config ─────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 200
MAX_DEPTH = 6

# ─── Plot Style ────────────────────────────────────────────────────────────────
FIGURE_DPI = 150
FIGURE_SIZE_DEFAULT = (14, 7)
FIGURE_SIZE_WIDE = (18, 8)
FIGURE_SIZE_SQUARE = (10, 10)
PLOT_STYLE = "dark_background"
ACCENT_COLOR = "#00d4ff"
POSITIVE_COLOR = "#00c853"
NEGATIVE_COLOR = "#ff1744"
