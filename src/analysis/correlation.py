"""Lag-correlation analysis, contrarian backtest, and sentiment momentum."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import LAG_DAYS, SENTIMENT_ORDER


def _encode_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Add ordinal sentiment encoding if fg_value is missing."""
    if "fg_value" not in df.columns:
        enc = {s: i for i, s in enumerate(SENTIMENT_ORDER)}
        df = df.copy()
        df["fg_value"] = df["classification"].map(enc)
    return df


def compute_lag_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson & Spearman correlation between sentiment and PnL at various lags."""
    df = _encode_sentiment(df)
    closes = df[df["is_close"] & df["closedPnL"].notna()].copy()

    daily = (
        closes.groupby("date")
        .agg(avg_pnl=("closedPnL", "mean"), fg_value=("fg_value", "first"))
        .reset_index()
        .sort_values("date")
    )

    records = []
    for lag in LAG_DAYS:
        shifted_pnl = daily["avg_pnl"].shift(-lag)
        valid = daily["fg_value"].notna() & shifted_pnl.notna()
        x, y = daily["fg_value"][valid].values, shifted_pnl[valid].values
        if len(x) < 10:
            continue
        pearson_r, pearson_p = stats.pearsonr(x, y)
        spearman_r, spearman_p = stats.spearmanr(x, y)
        records.append({
            "lag_days":   lag,
            "pearson_r":  round(pearson_r, 4),
            "pearson_p":  round(pearson_p, 4),
            "spearman_r": round(spearman_r, 4),
            "spearman_p": round(spearman_p, 4),
            "n":          int(valid.sum()),
        })

    return pd.DataFrame(records)


def contrarian_backtest(df: pd.DataFrame) -> dict:
    """
    Contrarian strategy: long during Extreme Fear, short during Extreme Greed.
    Returns daily equity curve and summary statistics.
    """
    closes = df[df["is_close"] & df["closedPnL"].notna()].copy()

    daily = (
        closes.groupby("date")
        .agg(
            avg_pnl=("closedPnL", "mean"),
            classification=("classification", "first"),
            long_pnl=(
                "closedPnL",
                lambda x: x[closes.loc[x.index, "is_long"]].mean()
                if closes.loc[x.index, "is_long"].any() else 0,
            ),
            short_pnl=(
                "closedPnL",
                lambda x: x[~closes.loc[x.index, "is_long"]].mean()
                if (~closes.loc[x.index, "is_long"]).any() else 0,
            ),
        )
        .reset_index()
        .sort_values("date")
    )

    def _signal(cls):
        if cls == "Extreme Fear":
            return 1
        elif cls == "Extreme Greed":
            return -1
        return 0

    daily["signal"] = daily["classification"].apply(_signal)
    daily["strategy_return"] = np.where(
        daily["signal"] == 1,
        daily["long_pnl"].fillna(0),
        np.where(daily["signal"] == -1, -daily["short_pnl"].fillna(0), 0),
    )
    daily["benchmark_return"] = daily["avg_pnl"].fillna(0)
    daily["strategy_equity"] = daily["strategy_return"].cumsum()
    daily["benchmark_equity"] = daily["benchmark_return"].cumsum()

    summary = {
        "total_strategy_pnl":    daily["strategy_return"].sum(),
        "total_benchmark_pnl":   daily["benchmark_return"].sum(),
        "strategy_win_days":     int((daily["strategy_return"] > 0).sum()),
        "total_active_days":     int((daily["signal"] != 0).sum()),
        "alpha":                 daily["strategy_return"].sum() - daily["benchmark_return"].sum(),
        "max_drawdown_strategy": _max_drawdown(daily["strategy_equity"]),
        "max_drawdown_benchmark": _max_drawdown(daily["benchmark_equity"]),
    }
    return {"daily": daily, "summary": summary}


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    return round(float((equity - peak).min()), 2)


def rolling_sentiment_momentum(df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """Rolling average sentiment value correlated with next-N-day PnL."""
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


def sentiment_transition_matrix(fear_greed: pd.DataFrame) -> pd.DataFrame:
    """Day-to-day sentiment transition probability matrix."""
    fg = fear_greed.sort_values("date").copy()
    fg["next_cls"] = fg["classification"].shift(-1)
    fg = fg.dropna(subset=["next_cls"])

    matrix = (
        fg.groupby(["classification", "next_cls"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    matrix = matrix.div(matrix.sum(axis=1), axis=0)
    return matrix.reindex(index=SENTIMENT_ORDER, columns=SENTIMENT_ORDER, fill_value=0)
