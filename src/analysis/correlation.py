"""
src/analysis/correlation.py

Lag-correlation analysis between sentiment and trader PnL,
contrarian strategy backtest, and rolling sentiment windows.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import LAG_DAYS, SENTIMENT_ORDER


# ─── Sentiment ↔ PnL Lag Correlation ─────────────────────────────────────────

def compute_lag_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each lag N (in LAG_DAYS), compute Pearson & Spearman correlation
    between the Fear/Greed numeric value and daily average PnL shifted N days forward.
    """
    if "fg_value" not in df.columns:
        # Encode classification ordinally
        enc = {s: i for i, s in enumerate(SENTIMENT_ORDER)}
        df = df.copy()
        df["fg_value"] = df["classification"].map(enc)

    closes = df[df["is_close"] & df["closedPnL"].notna()].copy()
    daily = (
        closes.groupby("date")
        .agg(avg_pnl=("closedPnL", "mean"), fg_value=("fg_value", "first"))
        .reset_index()
        .sort_values("date")
    )

    records = []
    for lag in LAG_DAYS:
        shifted_pnl = daily["avg_pnl"].shift(-lag)  # PnL N days after sentiment
        valid = daily["fg_value"].notna() & shifted_pnl.notna()
        x, y = daily["fg_value"][valid].values, shifted_pnl[valid].values
        if len(x) < 10:
            continue
        pearson_r, pearson_p = stats.pearsonr(x, y)
        spearman_r, spearman_p = stats.spearmanr(x, y)
        records.append({
            "lag_days": lag,
            "pearson_r": round(pearson_r, 4),
            "pearson_p": round(pearson_p, 4),
            "spearman_r": round(spearman_r, 4),
            "spearman_p": round(spearman_p, 4),
            "n": int(valid.sum()),
        })

    return pd.DataFrame(records)


# ─── Contrarian Strategy Backtest ─────────────────────────────────────────────

def contrarian_backtest(df: pd.DataFrame) -> dict:
    """
    Simulates a contrarian strategy:
      - Enter LONG during 'Extreme Fear' days
      - Exit LONG / Enter SHORT during 'Extreme Greed' days
      - Neutral/Fear/Greed → hold

    Compares to a random (benchmark) strategy over the same period.
    Returns daily equity curve and summary statistics.
    """
    closes = df[df["is_close"] & df["closedPnL"].notna()].copy()

    # Daily sentiment + PnL
    daily = (
        closes.groupby("date")
        .agg(
            avg_pnl=("closedPnL", "mean"),
            classification=("classification", "first"),
            long_pnl=(
                "closedPnL",
                lambda x: x[closes.loc[x.index, "is_long"]].mean() if closes.loc[x.index, "is_long"].any() else 0,
            ),
            short_pnl=(
                "closedPnL",
                lambda x: x[~closes.loc[x.index, "is_long"]].mean() if (~closes.loc[x.index, "is_long"]).any() else 0,
            ),
        )
        .reset_index()
        .sort_values("date")
    )

    # Strategy signal
    def signal(cls):
        if cls == "Extreme Fear":
            return 1    # Go LONG
        elif cls == "Extreme Greed":
            return -1   # Go SHORT (contrarian)
        return 0        # Flat

    daily["signal"] = daily["classification"].apply(signal)

    # Strategy returns: long_pnl when signal=1, short_pnl when signal=-1, 0 when flat
    daily["strategy_return"] = np.where(
        daily["signal"] == 1,
        daily["long_pnl"].fillna(0),
        np.where(daily["signal"] == -1, -daily["short_pnl"].fillna(0), 0),
    )
    daily["benchmark_return"] = daily["avg_pnl"].fillna(0)

    # Cumulative equity curves
    daily["strategy_equity"] = daily["strategy_return"].cumsum()
    daily["benchmark_equity"] = daily["benchmark_return"].cumsum()

    # Summary stats
    active_days = daily[daily["signal"] != 0]
    summary = {
        "total_strategy_pnl": daily["strategy_return"].sum(),
        "total_benchmark_pnl": daily["benchmark_return"].sum(),
        "strategy_win_days": int((daily["strategy_return"] > 0).sum()),
        "total_active_days": int((daily["signal"] != 0).sum()),
        "alpha": daily["strategy_return"].sum() - daily["benchmark_return"].sum(),
        "max_drawdown_strategy": _max_drawdown(daily["strategy_equity"]),
        "max_drawdown_benchmark": _max_drawdown(daily["benchmark_equity"]),
    }
    return {"daily": daily, "summary": summary}


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity - peak).min()
    return round(float(dd), 2)


# ─── Rolling Sentiment Momentum ───────────────────────────────────────────────

def rolling_sentiment_momentum(df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """
    Compute rolling average sentiment value and correlate with next-N-day PnL.
    Returns a daily DataFrame with rolling metrics.
    """
    enc = {s: i for i, s in enumerate(SENTIMENT_ORDER)}
    df = df.copy()
    df["sentiment_num"] = df["classification"].map(enc)

    closes = df[df["is_close"] & df["closedPnL"].notna()].copy()
    daily = (
        closes.groupby("date")
        .agg(avg_pnl=("closedPnL", "mean"), sentiment_num=("sentiment_num", "first"))
        .reset_index()
        .sort_values("date")
    )
    daily[f"rolling_sentiment_{window}d"] = (
        daily["sentiment_num"].rolling(window, min_periods=1).mean()
    )
    daily["rolling_pnl_7d"] = daily["avg_pnl"].rolling(7, min_periods=1).mean()
    return daily


# ─── Sentiment Transition Matrix ──────────────────────────────────────────────

def sentiment_transition_matrix(fear_greed: pd.DataFrame) -> pd.DataFrame:
    """
    Compute day-to-day sentiment transition probabilities.
    """
    fg = fear_greed.sort_values("date").copy()
    fg["next_cls"] = fg["classification"].shift(-1)
    fg = fg.dropna(subset=["next_cls"])

    matrix = (
        fg.groupby(["classification", "next_cls"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    # Row-normalise to probabilities
    matrix = matrix.div(matrix.sum(axis=1), axis=0)
    return matrix.reindex(index=SENTIMENT_ORDER, columns=SENTIMENT_ORDER, fill_value=0)
