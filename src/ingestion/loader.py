"""
src/ingestion/loader.py

Handles all data loading, validation, and preprocessing.
Produces a clean merged DataFrame ready for analysis.
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running standalone
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import (
    FEAR_GREED_FILE,
    MERGED_FILE,
    PROCESSED_DIR,
    SENTIMENT_ORDER,
    TRADES_FILE,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ─── Loaders ──────────────────────────────────────────────────────────────────

def load_fear_greed(path: Path = FEAR_GREED_FILE) -> pd.DataFrame:
    """Load and validate the Fear & Greed Index dataset."""
    log.info(f"Loading Fear/Greed data from {path}")
    df = pd.read_csv(path)

    # Normalise column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    required = {"date", "classification"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Fear/Greed CSV missing columns: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    df["classification"] = (
        df["classification"].str.strip().str.title()
    )

    # Map to canonical categories
    alias = {
        "Extreme Fear": "Extreme Fear",
        "Fear":         "Fear",
        "Neutral":      "Neutral",
        "Greed":        "Greed",
        "Extreme Greed":"Extreme Greed",
    }
    df["classification"] = df["classification"].map(alias).fillna("Neutral")
    df["classification"] = pd.Categorical(
        df["classification"], categories=SENTIMENT_ORDER, ordered=True
    )

    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    log.info(f"  → {len(df)} daily rows | range: {df['date'].min().date()} – {df['date'].max().date()}")
    return df


def load_trades(path: Path = TRADES_FILE) -> pd.DataFrame:
    """Load and validate the Hyperliquid historical trader dataset."""
    log.info(f"Loading Trades data from {path}")
    df = pd.read_csv(path)

    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Rename common variants
    rename_map = {
        "closedpnl":       "closedPnL",
        "closed_pnl":      "closedPnL",
        "pnl":             "closedPnL",
        "execprice":       "execution_price",
        "exec_price":      "execution_price",
        "startposition":   "start_position",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    # Time parsing
    df["time"] = pd.to_datetime(df["time"])
    df["date"] = df["time"].dt.normalize()

    # Numeric coercion
    for col in ["execution_price", "size", "closedPnL", "leverage"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Derived flags
    df["is_close"] = df["event"].str.upper() == "CLOSE"
    df["is_long"] = df["side"].str.upper().isin(["BUY", "LONG"])
    df["is_profitable"] = df["closedPnL"] > 0

    # Leverage bucket
    df["leverage_bucket"] = pd.cut(
        df["leverage"],
        bins=[0, 2, 5, 10, 25, 1000],
        labels=["1-2x", "3-5x", "6-10x", "11-25x", "26x+"],
        right=True,
    )

    df = df.sort_values("time").reset_index(drop=True)
    log.info(
        f"  → {len(df):,} trades | {df['account'].nunique()} accounts "
        f"| range: {df['date'].min().date()} – {df['date'].max().date()}"
    )
    return df


def merge_datasets(
    trades: pd.DataFrame,
    fear_greed: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge trade records with the daily Fear & Greed Index on date.
    Trades outside the sentiment date range are dropped with a warning.
    """
    log.info("Merging trades with Fear/Greed index on date …")

    fg = fear_greed[["date", "classification"]].copy()
    if "value" in fear_greed.columns:
        fg["fg_value"] = fear_greed["value"]

    merged = trades.merge(fg, on="date", how="left")

    n_missing = merged["classification"].isna().sum()
    if n_missing:
        log.warning(f"  {n_missing} trades had no matching Fear/Greed date → dropped")
        merged = merged.dropna(subset=["classification"])

    log.info(f"  → Merged dataset: {len(merged):,} rows")
    return merged.reset_index(drop=True)


def load_and_merge(save: bool = True) -> pd.DataFrame:
    """
    One-shot entry point: load both raw files, merge, optionally save.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    fg = load_fear_greed()
    trades = load_trades()
    merged = merge_datasets(trades, fg)

    if save:
        merged.to_csv(MERGED_FILE, index=False)
        log.info(f"  Saved merged dataset → {MERGED_FILE}")

    return merged


def load_merged(force_rebuild: bool = False) -> pd.DataFrame:
    """
    Load cached merged dataset, rebuilding if needed.
    """
    if not MERGED_FILE.exists() or force_rebuild:
        log.info("Merged cache not found — building from scratch …")
        return load_and_merge(save=True)

    log.info(f"Loading cached merged dataset from {MERGED_FILE}")
    df = pd.read_csv(MERGED_FILE, parse_dates=["time", "date"])
    df["classification"] = pd.Categorical(
        df["classification"], categories=SENTIMENT_ORDER, ordered=True
    )
    return df


# ─── Quick validation ─────────────────────────────────────────────────────────

def validate_merged(df: pd.DataFrame) -> dict:
    """Return a dict of data-quality stats."""
    close_df = df[df["is_close"]]
    return {
        "total_rows":         len(df),
        "close_events":       int(close_df["is_close"].sum()),
        "unique_accounts":    df["account"].nunique(),
        "unique_symbols":     df["symbol"].nunique(),
        "date_range":         (df["date"].min().date(), df["date"].max().date()),
        "null_pnl":           int(df["closedPnL"].isna().sum()),
        "sentiment_coverage": df["classification"].value_counts().to_dict(),
    }


if __name__ == "__main__":
    merged = load_and_merge()
    stats = validate_merged(merged)
    for k, v in stats.items():
        print(f"  {k}: {v}")
