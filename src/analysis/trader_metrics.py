"""Per-trader and per-sentiment performance metrics."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import MIN_TRADES_FOR_PROFILING, SENTIMENT_ORDER


def compute_close_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to CLOSE events with valid PnL."""
    return df[df["is_close"] & df["closedPnL"].notna()].copy()


def pnl_by_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate PnL metrics by sentiment regime."""
    closes = compute_close_metrics(df)

    agg = (
        closes.groupby("classification", observed=True)["closedPnL"]
        .agg(
            mean_pnl="mean",
            median_pnl="median",
            total_pnl="sum",
            trade_count="count",
            std_pnl="std",
        )
        .reset_index()
    )
    win_rate = (
        closes.groupby("classification", observed=True)["is_profitable"]
        .mean()
        .rename("win_rate")
        .reset_index()
    )

    result = agg.merge(win_rate, on="classification")
    result["classification"] = pd.Categorical(
        result["classification"], categories=SENTIMENT_ORDER, ordered=True
    )
    result["sharpe_like"] = result["mean_pnl"] / result["std_pnl"].replace(0, np.nan)
    return result.sort_values("classification").reset_index(drop=True)


def leverage_by_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Average leverage per sentiment class."""
    agg = (
        df.groupby("classification", observed=True)["leverage"]
        .agg(mean_leverage="mean", median_leverage="median", count="count")
        .reset_index()
    )
    agg["classification"] = pd.Categorical(
        agg["classification"], categories=SENTIMENT_ORDER, ordered=True
    )
    return agg.sort_values("classification")


def long_short_ratio_by_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Long vs short trade proportion under each sentiment."""
    g = (
        df.groupby(["classification", "is_long"], observed=True)
        .size()
        .unstack(fill_value=0)
        .rename(columns={True: "longs", False: "shorts"})
        .reset_index()
    )
    g["total"] = g["longs"] + g["shorts"]
    g["long_ratio"] = g["longs"] / g["total"]
    g["classification"] = pd.Categorical(
        g["classification"], categories=SENTIMENT_ORDER, ordered=True
    )
    return g.sort_values("classification")


def top_traders(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Rank traders by total PnL with win rate and Sharpe-like ratio."""
    closes = compute_close_metrics(df)
    grouped = closes.groupby("account")

    metrics = grouped["closedPnL"].agg(
        total_pnl="sum",
        mean_pnl="mean",
        std_pnl="std",
        trade_count="count",
    )
    metrics["win_rate"] = grouped["is_profitable"].mean()
    metrics = metrics[metrics["trade_count"] >= MIN_TRADES_FOR_PROFILING]
    metrics["sharpe"] = metrics["mean_pnl"] / metrics["std_pnl"].replace(0, np.nan)
    metrics = metrics.sort_values("total_pnl", ascending=False).head(top_n)
    metrics["rank"] = range(1, len(metrics) + 1)
    return metrics.reset_index()


def bottom_traders(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Worst traders by total PnL."""
    closes = compute_close_metrics(df)
    grouped = closes.groupby("account")
    metrics = grouped["closedPnL"].agg(total_pnl="sum", trade_count="count")
    metrics["win_rate"] = grouped["is_profitable"].mean()
    metrics = metrics[metrics["trade_count"] >= MIN_TRADES_FOR_PROFILING]
    return metrics.sort_values("total_pnl").head(top_n).reset_index()


def trader_sentiment_preference(df: pd.DataFrame) -> pd.DataFrame:
    """Per-trader PnL breakdown across sentiment regimes."""
    closes = compute_close_metrics(df)
    return (
        closes.groupby(["account", "classification"], observed=True)["closedPnL"]
        .agg(["sum", "mean", "count"])
        .rename(columns={"sum": "total_pnl", "mean": "avg_pnl", "count": "trades"})
        .reset_index()
    )


def daily_pnl_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """Daily aggregate PnL with 7-day rolling mean."""
    closes = compute_close_metrics(df)
    daily = (
        closes.groupby("date")["closedPnL"]
        .agg(total_pnl="sum", trade_count="count", win_rate=lambda x: (x > 0).mean())
        .reset_index()
    )
    daily["rolling_pnl_7d"] = daily["total_pnl"].rolling(7, min_periods=1).mean()
    return daily


def pnl_by_symbol_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Mean PnL per symbol × sentiment — used for heatmap."""
    closes = compute_close_metrics(df)
    return (
        closes.groupby(["symbol", "classification"], observed=True)["closedPnL"]
        .mean()
        .unstack("classification")
        .reindex(columns=SENTIMENT_ORDER)
    )


def statistical_tests(df: pd.DataFrame) -> dict:
    """Kruskal-Wallis and pairwise Mann-Whitney U tests on PnL by sentiment."""
    closes = compute_close_metrics(df)
    groups = {
        label: closes[closes["classification"] == label]["closedPnL"].dropna().values
        for label in SENTIMENT_ORDER
        if label in closes["classification"].values
    }

    kw_stat, kw_p = stats.kruskal(*[v for v in groups.values() if len(v) > 1])

    records = []
    labels = list(groups.keys())
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = groups[labels[i]], groups[labels[j]]
            if len(a) > 1 and len(b) > 1:
                u_stat, p_val = stats.mannwhitneyu(a, b, alternative="two-sided")
                records.append({
                    "group_a":     labels[i],
                    "group_b":     labels[j],
                    "u_stat":      round(u_stat, 2),
                    "p_value":     round(p_val, 4),
                    "significant": p_val < 0.05,
                })

    return {
        "kruskal_wallis": {"stat": kw_stat, "p": kw_p},
        "pairwise": pd.DataFrame(records),
    }
