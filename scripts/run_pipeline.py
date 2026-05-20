"""
scripts/run_pipeline.py

One-click runner: loads data → runs all analyses → generates all charts → prints insights.
Run from project root:   python scripts/run_pipeline.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config.settings import FIGURES_DIR, PROCESSED_DIR, SENTIMENT_ORDER
from src.analysis.correlation import (
    compute_lag_correlations,
    contrarian_backtest,
    rolling_sentiment_momentum,
    sentiment_transition_matrix,
)
from src.analysis.trader_metrics import (
    bottom_traders,
    leverage_by_sentiment,
    long_short_ratio_by_sentiment,
    pnl_by_sentiment,
    pnl_by_symbol_sentiment,
    statistical_tests,
    top_traders,
)
from src.ingestion.loader import load_and_merge, load_fear_greed, validate_merged
from src.models.sentiment_predictor import SentimentTradePredictor, plot_model_results
from src.visualization.charts import (
    plot_contrarian_strategy,
    plot_lag_correlation,
    plot_long_short_ratio,
    plot_leverage_by_sentiment,
    plot_pnl_by_sentiment,
    plot_pnl_violin,
    plot_sentiment_distribution,
    plot_sentiment_timeline,
    plot_symbol_sentiment_heatmap,
    plot_top_traders,
    plot_transition_matrix,
)

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def section(title: str):
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}")


def run():
    t0 = time.time()

    # ── 1. Load Data ──────────────────────────────────────────────────────
    section("1 / LOADING DATA")
    df = load_and_merge(save=True)
    fg = load_fear_greed()
    stats = validate_merged(df)
    print(f"\n  Total rows      : {stats['total_rows']:,}")
    print(f"  CLOSE events    : {stats['close_events']:,}")
    print(f"  Unique accounts : {stats['unique_accounts']}")
    print(f"  Unique symbols  : {stats['unique_symbols']}")
    print(f"  Date range      : {stats['date_range'][0]} → {stats['date_range'][1]}")
    print(f"  Sentiment dist  :")
    for k, v in stats["sentiment_coverage"].items():
        print(f"    {k:<15}: {v} days")

    # ── 2. Analysis ───────────────────────────────────────────────────────
    section("2 / RUNNING ANALYSES")

    pnl_df = pnl_by_sentiment(df)
    print("\n  PnL by Sentiment:")
    print(pnl_df[["classification", "mean_pnl", "win_rate", "trade_count"]].to_string(index=False))

    lev_df = leverage_by_sentiment(df)
    ls_df = long_short_ratio_by_sentiment(df)
    top_df = top_traders(df, top_n=10)
    bottom_df = bottom_traders(df, top_n=5)
    symbol_pivot = pnl_by_symbol_sentiment(df)
    lag_df = compute_lag_correlations(df)
    backtest = contrarian_backtest(df)
    transition = sentiment_transition_matrix(fg)
    stat_tests = statistical_tests(df)
    momentum = rolling_sentiment_momentum(df)

    print(f"\n  Contrarian strategy alpha: ${backtest['summary']['alpha']:.2f}")
    print(f"  Kruskal-Wallis p-value: {stat_tests['kruskal_wallis']['p']:.4f}")
    print(f"\n  Lag Correlations (Pearson r):")
    for _, row in lag_df.iterrows():
        print(f"    Lag {int(row['lag_days']):2d}d: r={row['pearson_r']:+.4f}  (p={row['pearson_p']:.3f})")

    # ── 3. Charts ─────────────────────────────────────────────────────────
    section("3 / GENERATING CHARTS")

    plot_sentiment_distribution(fg)
    plot_pnl_by_sentiment(pnl_df)
    plot_leverage_by_sentiment(lev_df)
    plot_long_short_ratio(ls_df)
    plot_top_traders(top_df)
    plot_symbol_sentiment_heatmap(symbol_pivot)
    plot_lag_correlation(lag_df)
    plot_contrarian_strategy(backtest)
    plot_sentiment_timeline(fg, df)
    plot_transition_matrix(transition)
    plot_pnl_violin(df)

    # ── 4. ML Model ───────────────────────────────────────────────────────
    section("4 / ML MODEL — Trade Profitability Predictor")

    predictor = SentimentTradePredictor()
    predictor.fit(df)
    results = predictor.evaluate()
    plot_model_results(results, FIGURES_DIR)

    print(f"\n  ROC-AUC (test):      {results['roc_auc']:.4f}")
    print(f"  ROC-AUC (5-fold CV): {results['cv_auc_mean']:.4f} ± {results['cv_auc_std']:.4f}")
    print("\n  Top Features:")
    fi = predictor.feature_importance_df()
    for _, row in fi.head(5).iterrows():
        bar = "█" * int(row["importance"] * 200)
        print(f"    {row['feature']:<25} {bar}  {row['importance']:.4f}")

    # ── 5. Key Insights ───────────────────────────────────────────────────
    section("5 / KEY INSIGHTS")

    # Best sentiment for trading
    best_row = pnl_df.loc[pnl_df["mean_pnl"].idxmax()]
    worst_row = pnl_df.loc[pnl_df["mean_pnl"].idxmin()]
    most_active = pnl_df.loc[pnl_df["trade_count"].idxmax()]
    highest_lev = lev_df.loc[lev_df["mean_leverage"].idxmax()]

    print(f"""
  ┌─────────────────────────────────────────────────────────┐
  │           EXECUTIVE INSIGHTS SUMMARY                    │
  ├─────────────────────────────────────────────────────────┤
  │ 🏆 Best PnL regime  : {best_row['classification']:<15} (avg ${best_row['mean_pnl']:+.2f})   │
  │ 📉 Worst PnL regime : {worst_row['classification']:<15} (avg ${worst_row['mean_pnl']:+.2f})   │
  │ 📊 Most active sent.: {most_active['classification']:<15} ({int(most_active['trade_count'])} trades)      │
  │ ⚡ Peak leverage     : {highest_lev['classification']:<15} ({highest_lev['mean_leverage']:.1f}× avg)       │
  │ 🎯 Contrarian alpha : ${backtest['summary']['alpha']:+.2f}                            │
  │ 🤖 ML AUC score     : {results['roc_auc']:.4f} (CV: {results['cv_auc_mean']:.4f})              │
  │ 📐 KW significance  : p={stat_tests['kruskal_wallis']['p']:.4f}                           │
  └─────────────────────────────────────────────────────────┘
""")

    elapsed = time.time() - t0
    print(f"  ✅ Pipeline complete in {elapsed:.1f}s")
    print(f"  📂 All charts saved to: {FIGURES_DIR}")
    print(f"  📂 Processed data at:   {PROCESSED_DIR}\n")


if __name__ == "__main__":
    run()
