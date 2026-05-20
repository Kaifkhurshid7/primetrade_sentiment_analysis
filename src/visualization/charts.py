"""Plotting functions for sentiment analysis. Each function returns and saves a Figure."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import (
    ACCENT_COLOR,
    FIGURES_DIR,
    FIGURE_DPI,
    FIGURE_SIZE_DEFAULT,
    FIGURE_SIZE_SQUARE,
    FIGURE_SIZE_WIDE,
    NEGATIVE_COLOR,
    POSITIVE_COLOR,
    SENTIMENT_COLORS,
    SENTIMENT_ORDER,
)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

_DARK_BG = "#0d0d1a"


def _style():
    plt.style.use("dark_background")
    plt.rcParams.update({
        "axes.facecolor":    _DARK_BG,
        "figure.facecolor":  _DARK_BG,
        "axes.edgecolor":    "#2a2a3e",
        "grid.color":        "#1e1e2e",
        "text.color":        "#e0e0f0",
        "axes.labelcolor":   "#e0e0f0",
        "xtick.color":       "#a0a0c0",
        "ytick.color":       "#a0a0c0",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "font.family":       "monospace",
    })


def _save(fig, name: str):
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  ✓ Saved → {path}")
    return fig


def plot_sentiment_distribution(fear_greed: pd.DataFrame) -> plt.Figure:
    _style()
    counts = (
        fear_greed["classification"]
        .value_counts()
        .reindex(SENTIMENT_ORDER)
        .fillna(0)
    )
    colors = [SENTIMENT_COLORS[c] for c in counts.index]

    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE_DEFAULT)
    fig.suptitle(
        "Fear & Greed Index — Sentiment Distribution",
        fontsize=15, y=1.02, color=ACCENT_COLOR,
    )

    ax = axes[0]
    bars = ax.barh(counts.index, counts.values, color=colors, edgecolor="none", height=0.6)
    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            f" {int(val)} days", va="center", fontsize=9, color="#c0c0d0",
        )
    ax.set_xlabel("Number of Days")
    ax.set_title("Days Per Sentiment Class")
    ax.invert_yaxis()

    ax2 = axes[1]
    wedges, texts, autotexts = ax2.pie(
        counts.values, labels=counts.index, colors=colors,
        autopct="%1.1f%%", startangle=90,
        wedgeprops={"edgecolor": _DARK_BG, "linewidth": 2},
    )
    for t in autotexts:
        t.set_color("white")
        t.set_fontsize(9)
    ax2.set_title("Distribution (%)")

    fig.tight_layout()
    return _save(fig, "01_sentiment_distribution")


def plot_pnl_by_sentiment(pnl_df: pd.DataFrame) -> plt.Figure:
    _style()
    fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZE_WIDE)
    fig.suptitle("Trader P&L Performance by Market Sentiment", fontsize=15, color=ACCENT_COLOR)
    colors = [SENTIMENT_COLORS[c] for c in pnl_df["classification"]]

    ax = axes[0]
    bars = ax.bar(pnl_df["classification"], pnl_df["mean_pnl"], color=colors, edgecolor="none")
    ax.axhline(0, color="#aaa", lw=0.8, linestyle="--")
    ax.set_title("Mean PnL per Trade")
    ax.set_ylabel("USD")
    ax.set_xticklabels(pnl_df["classification"], rotation=30, ha="right")
    for b in bars:
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2, h + (0.5 if h >= 0 else -1.5),
            f"${h:.1f}", ha="center", fontsize=8,
        )

    ax2 = axes[1]
    ax2.bar(pnl_df["classification"], pnl_df["win_rate"] * 100, color=colors, edgecolor="none")
    ax2.axhline(50, color="#aaa", lw=0.8, linestyle="--", label="50% line")
    ax2.set_title("Win Rate (%)")
    ax2.set_ylabel("%")
    ax2.set_ylim(0, 100)
    ax2.set_xticklabels(pnl_df["classification"], rotation=30, ha="right")

    ax3 = axes[2]
    ax3.bar(pnl_df["classification"], pnl_df["trade_count"], color=colors, edgecolor="none", alpha=0.85)
    ax3.set_title("Trade Volume")
    ax3.set_ylabel("# Trades")
    ax3.set_xticklabels(pnl_df["classification"], rotation=30, ha="right")

    fig.tight_layout()
    return _save(fig, "02_pnl_by_sentiment")


def plot_leverage_by_sentiment(lev_df: pd.DataFrame) -> plt.Figure:
    _style()
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [SENTIMENT_COLORS[c] for c in lev_df["classification"]]
    bars = ax.bar(lev_df["classification"], lev_df["mean_leverage"], color=colors, edgecolor="none", width=0.5)
    ax.errorbar(
        lev_df["classification"], lev_df["mean_leverage"],
        yerr=lev_df["median_leverage"] * 0.1,
        fmt="none", color="white", capsize=4, lw=1.2,
    )
    for b, row in zip(bars, lev_df.itertuples()):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.2,
                f"{row.mean_leverage:.1f}x", ha="center", fontsize=9)
    ax.set_title("Average Leverage Used by Sentiment Regime", fontsize=13, color=ACCENT_COLOR)
    ax.set_ylabel("Mean Leverage (×)")
    ax.set_xticklabels(lev_df["classification"], rotation=20, ha="right")
    fig.tight_layout()
    return _save(fig, "03_leverage_by_sentiment")


def plot_long_short_ratio(ls_df: pd.DataFrame) -> plt.Figure:
    _style()
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(ls_df))
    w = 0.35
    ax.bar(x - w / 2, ls_df["longs"], width=w, label="Long", color=POSITIVE_COLOR, alpha=0.85)
    ax.bar(x + w / 2, ls_df["shorts"], width=w, label="Short", color=NEGATIVE_COLOR, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(ls_df["classification"], rotation=20, ha="right")
    ax.set_ylabel("Trade Count")
    ax.set_title("Long vs Short Trade Distribution by Sentiment", fontsize=13, color=ACCENT_COLOR)
    ax.legend()

    ax2 = ax.twinx()
    ax2.plot(x, ls_df["long_ratio"] * 100, color=ACCENT_COLOR, marker="o", lw=2, label="Long Ratio %", zorder=5)
    ax2.set_ylabel("Long %", color=ACCENT_COLOR)
    ax2.tick_params(axis="y", colors=ACCENT_COLOR)
    ax2.set_ylim(0, 100)

    fig.tight_layout()
    return _save(fig, "04_long_short_ratio")


def plot_top_traders(top_df: pd.DataFrame) -> plt.Figure:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE_WIDE)
    fig.suptitle("Top 10 Trader Performance Profile", fontsize=15, color=ACCENT_COLOR)

    top_df = top_df.copy()
    top_df["short_addr"] = top_df["account"].apply(lambda x: x[:6] + "…" + x[-4:])

    ax = axes[0]
    colors = [POSITIVE_COLOR if p > 0 else NEGATIVE_COLOR for p in top_df["total_pnl"]]
    ax.barh(top_df["short_addr"][::-1], top_df["total_pnl"][::-1], color=colors[::-1])
    ax.axvline(0, color="white", lw=0.7)
    ax.set_xlabel("Total PnL (USD)")
    ax.set_title("Total Realized PnL")

    ax2 = axes[1]
    scatter = ax2.scatter(
        top_df["win_rate"] * 100, top_df["total_pnl"],
        c=top_df["trade_count"], cmap="plasma", s=120,
        edgecolors="white", linewidths=0.5, zorder=5,
    )
    for _, row in top_df.iterrows():
        ax2.annotate(
            row["short_addr"], (row["win_rate"] * 100, row["total_pnl"]),
            fontsize=7, color="#c0c0d0", xytext=(4, 4), textcoords="offset points",
        )
    plt.colorbar(scatter, ax=ax2, label="Trade Count")
    ax2.axhline(0, color="#aaa", lw=0.7, linestyle="--")
    ax2.axvline(50, color="#aaa", lw=0.7, linestyle="--")
    ax2.set_xlabel("Win Rate (%)")
    ax2.set_ylabel("Total PnL (USD)")
    ax2.set_title("Win Rate vs Total PnL")

    fig.tight_layout()
    return _save(fig, "05_top_traders")


def plot_symbol_sentiment_heatmap(pivot: pd.DataFrame) -> plt.Figure:
    _style()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        pivot, ax=ax, annot=True, fmt=".1f", cmap="RdYlGn", center=0,
        linewidths=0.5, linecolor=_DARK_BG, cbar_kws={"label": "Mean PnL (USD)"},
    )
    ax.set_title("Mean PnL per Symbol × Sentiment Regime", fontsize=13, color=ACCENT_COLOR)
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Symbol")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return _save(fig, "06_symbol_sentiment_heatmap")


def plot_lag_correlation(lag_df: pd.DataFrame) -> plt.Figure:
    _style()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(lag_df["lag_days"], lag_df["pearson_r"], marker="o", color=ACCENT_COLOR, lw=2, label="Pearson r")
    ax.plot(lag_df["lag_days"], lag_df["spearman_r"], marker="s", color="#ff9f43", lw=2, linestyle="--", label="Spearman r")
    ax.fill_between(lag_df["lag_days"], lag_df["pearson_r"], 0, alpha=0.15, color=ACCENT_COLOR)
    ax.axhline(0, color="#aaa", lw=0.7)
    ax.axhline(0.1, color="#555", lw=0.7, linestyle=":")
    ax.axhline(-0.1, color="#555", lw=0.7, linestyle=":")
    ax.set_xlabel("Lag (days) — PnL measured N days after sentiment")
    ax.set_ylabel("Correlation Coefficient")
    ax.set_title("Sentiment → PnL Lag Correlation", fontsize=13, color=ACCENT_COLOR)
    ax.legend()
    ax.set_xticks(lag_df["lag_days"])
    fig.tight_layout()
    return _save(fig, "07_lag_correlation")


def plot_contrarian_strategy(backtest: dict) -> plt.Figure:
    _style()
    daily = backtest["daily"]
    summary = backtest["summary"]

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(
        "Contrarian Strategy: Buy Extreme Fear / Sell Extreme Greed",
        fontsize=14, color=ACCENT_COLOR,
    )

    ax = axes[0]
    ax.plot(daily["date"], daily["strategy_equity"], lw=2, color=POSITIVE_COLOR,
            label=f"Contrarian  α={summary['alpha']:.1f}")
    ax.plot(daily["date"], daily["benchmark_equity"], lw=1.5, color="#aaa",
            linestyle="--", label="Benchmark (random)")
    ax.fill_between(
        daily["date"], daily["strategy_equity"], daily["benchmark_equity"],
        where=daily["strategy_equity"] > daily["benchmark_equity"],
        alpha=0.2, color=POSITIVE_COLOR,
    )
    ax.fill_between(
        daily["date"], daily["strategy_equity"], daily["benchmark_equity"],
        where=daily["strategy_equity"] < daily["benchmark_equity"],
        alpha=0.2, color=NEGATIVE_COLOR,
    )
    ax.axhline(0, color="#555", lw=0.7)
    ax.set_ylabel("Cumulative PnL (USD)")
    ax.legend()

    ax2 = axes[1]
    colors_signal = daily["signal"].map({1: POSITIVE_COLOR, -1: NEGATIVE_COLOR, 0: "#555"})
    ax2.bar(daily["date"], daily["strategy_return"], color=colors_signal, width=1)
    ax2.axhline(0, color="#aaa", lw=0.5)
    ax2.set_ylabel("Daily Return (USD)")
    ax2.set_xlabel("Date")

    fig.tight_layout()
    return _save(fig, "08_contrarian_strategy")


def plot_sentiment_timeline(fear_greed: pd.DataFrame, df: pd.DataFrame) -> plt.Figure:
    _style()
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    fig.suptitle("Market Sentiment & Trader Performance Over Time", fontsize=14, color=ACCENT_COLOR)

    fg = fear_greed.sort_values("date")
    if "value" not in fg.columns:
        enc = {s: i * 25 for i, s in enumerate(SENTIMENT_ORDER)}
        fg["value"] = fg["classification"].map(enc)

    ax = axes[0]
    ax.fill_between(fg["date"], fg["value"], where=fg["value"] <= 40, color="#d62728", alpha=0.3, label="Fear zone")
    ax.fill_between(fg["date"], fg["value"], where=fg["value"] >= 60, color="#2ca02c", alpha=0.3, label="Greed zone")
    ax.plot(fg["date"], fg["value"], color=ACCENT_COLOR, lw=1, alpha=0.8)
    ax.axhline(50, color="#666", lw=0.8, linestyle="--")
    ax.set_ylabel("Fear/Greed Score")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", fontsize=8)

    closes = df[df["is_close"] & df["closedPnL"].notna()].copy()
    daily_pnl = closes.groupby("date")["closedPnL"].mean().reset_index()
    daily_pnl["rolling_7d"] = daily_pnl["closedPnL"].rolling(7, min_periods=1).mean()

    ax2 = axes[1]
    ax2.bar(
        daily_pnl["date"], daily_pnl["closedPnL"],
        color=[POSITIVE_COLOR if v > 0 else NEGATIVE_COLOR for v in daily_pnl["closedPnL"]],
        width=1, alpha=0.5,
    )
    ax2.plot(daily_pnl["date"], daily_pnl["rolling_7d"], color=ACCENT_COLOR, lw=2, label="7-day rolling mean PnL")
    ax2.axhline(0, color="#aaa", lw=0.7)
    ax2.set_ylabel("Avg Daily PnL (USD)")
    ax2.set_xlabel("Date")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    return _save(fig, "09_sentiment_timeline")


def plot_transition_matrix(matrix: pd.DataFrame) -> plt.Figure:
    _style()
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_SQUARE)
    sns.heatmap(
        matrix * 100, ax=ax, annot=True, fmt=".1f", cmap="Blues",
        linewidths=0.5, linecolor=_DARK_BG,
        cbar_kws={"label": "Transition Probability (%)"}, vmin=0, vmax=100,
    )
    ax.set_title("Sentiment Regime Transition Probability (%)", fontsize=13, color=ACCENT_COLOR)
    ax.set_xlabel("Next Day Sentiment →")
    ax.set_ylabel("Current Sentiment")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return _save(fig, "10_transition_matrix")


def plot_pnl_violin(df: pd.DataFrame) -> plt.Figure:
    _style()
    closes = df[df["is_close"] & df["closedPnL"].notna()].copy()
    q01, q99 = closes["closedPnL"].quantile(0.01), closes["closedPnL"].quantile(0.99)
    closes = closes[(closes["closedPnL"] >= q01) & (closes["closedPnL"] <= q99)]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_DEFAULT)
    present_classes = [c for c in SENTIMENT_ORDER if c in closes["classification"].values]
    data = [closes[closes["classification"] == c]["closedPnL"].values for c in present_classes]
    positions = list(range(len(present_classes)))

    parts = ax.violinplot(data, positions=positions, showmedians=True, showextrema=False)
    for pc, color in zip(parts["bodies"], [SENTIMENT_COLORS[c] for c in present_classes]):
        pc.set_facecolor(color)
        pc.set_alpha(0.75)
    parts["cmedians"].set_color("white")
    parts["cmedians"].set_linewidth(2)

    ax.set_xticks(positions)
    ax.set_xticklabels(present_classes, rotation=20, ha="right")
    ax.axhline(0, color="#aaa", lw=0.8, linestyle="--")
    ax.set_ylabel("Closed PnL (USD)")
    ax.set_title("PnL Distribution by Sentiment (1st–99th percentile)", fontsize=13, color=ACCENT_COLOR)
    fig.tight_layout()
    return _save(fig, "11_pnl_violin")
