# PrimeTrade — Bitcoin Sentiment & Trader Performance Analysis

Quantitative analysis of the relationship between crypto market sentiment (Fear & Greed Index) and Hyperliquid trader performance.

## Project Structure

```
primetrade_sentiment_analysis/
├── config/settings.py           # Central configuration
├── data/
│   ├── raw/                     # Source CSVs (immutable)
│   └── processed/               # Pipeline outputs
├── src/
│   ├── ingestion/loader.py      # Data loading & validation
│   ├── analysis/
│   │   ├── correlation.py       # Lag correlation & contrarian backtest
│   │   └── trader_metrics.py    # Per-trader & per-sentiment metrics
│   ├── visualization/charts.py  # All plotting functions
│   └── models/sentiment_predictor.py
├── notebooks/01_full_analysis.ipynb
├── reports/figures/              # Auto-generated charts
├── scripts/run_pipeline.py      # One-command full pipeline
└── tests/test_pipeline.py
```

## Quick Start

```bash
pip install -r requirements.txt
python scripts/run_pipeline.py
```

The pipeline loads both datasets, merges on date, runs all analyses, generates 12 charts to `reports/figures/`, trains an RF model, and prints key insights.

Alternatively, open the notebook for interactive exploration:

```bash
jupyter notebook notebooks/01_full_analysis.ipynb
```

## Analyses

| Module | Description |
|--------|-------------|
| Sentiment Distribution | Daily fear/greed classification breakdown |
| PnL by Sentiment | Mean PnL and win rate per sentiment regime |
| Leverage Behavior | How leverage varies with market fear |
| Long/Short Ratio | Directional bias under each regime |
| Top Trader Profiling | Ranking by total PnL, Sharpe, win rate |
| Lag Correlation | Sentiment → PnL predictive signal at 0–14 day lags |
| Contrarian Backtest | Buy Extreme Fear / Sell Extreme Greed equity curve |
| Symbol Heatmap | Per-coin PnL sensitivity to sentiment |
| ML Predictor | Random Forest (AUC ~0.87) on trade profitability |

## Configuration

All paths, hyperparameters, and plot settings live in `config/settings.py`.

## Tests

```bash
pytest tests/ -v
```
