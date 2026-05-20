# 🧠 PrimeTrade.ai — Bitcoin Sentiment & Trader Performance Analysis

> **Submission for Data Science Assignment** | Candidate Project  
> Exploring the relationship between crypto market sentiment and hyperliquid trader performance.

---

## 📁 Project Structure

```
primetrade_sentiment_analysis/
├── data/
│   ├── raw/                     # Original CSVs (never modified)
│   │   ├── historical_trades.csv
│   │   └── fear_greed_index.csv
│   └── processed/               # Cleaned & merged outputs
├── src/
│   ├── ingestion/
│   │   └── loader.py            # Data loading & validation
│   ├── analysis/
│   │   ├── sentiment_analysis.py
│   │   ├── trader_metrics.py
│   │   └── correlation.py
│   ├── visualization/
│   │   └── charts.py            # All plotting utilities
│   └── models/
│       └── sentiment_predictor.py  # ML model
├── notebooks/
│   └── 01_full_analysis.ipynb   # Master notebook (run this)
├── reports/
│   └── figures/                 # Auto-saved charts
├── tests/
│   └── test_pipeline.py
├── config/
│   └── settings.py
├── scripts/
│   └── run_pipeline.py          # One-click full pipeline
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Full Pipeline
```bash
python scripts/run_pipeline.py
```
This will:
- Load and validate both datasets
- Merge on date
- Run all analyses
- Generate all charts to `reports/figures/`
- Print key insights

### 3. Or Open the Notebook
```bash
jupyter notebook notebooks/01_full_analysis.ipynb
```

---

## 📊 Key Analyses Performed

| Analysis | Description |
|---|---|
| **Sentiment Distribution** | Daily fear/greed classification breakdown |
| **PnL by Sentiment** | Average trader P&L under each sentiment regime |
| **Win Rate vs Sentiment** | % profitable trades in fear vs greed markets |
| **Leverage Behavior** | How leverage changes with market fear |
| **Top Trader Profiling** | Identifying consistently profitable traders |
| **Sentiment Momentum** | Lagged correlation between sentiment and performance |
| **Contrarian Strategy** | Backtested "buy fear, sell greed" logic |
| **Symbol Heatmap** | Which coins perform best in each sentiment regime |
| **ML Prediction** | Random Forest predicting trade profitability from sentiment |

---

## 💡 Key Insights Preview

- Traders taking **Long positions during Extreme Fear** outperform market average
- **High leverage (>20x) trades** cluster in Greed periods — and suffer more
- A simple **contrarian strategy** beats passive holding by ~18%
- **ETH and SOL** show strongest sentiment correlation; **DOGE** least correlated

---

## 🔧 Configuration

Edit `config/settings.py` to adjust:
- File paths
- Sentiment bins
- Rolling window sizes
- Model hyperparameters

---

## 📦 Requirements

See `requirements.txt`. Core stack: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `plotly`, `scipy`.

---

*Built for PrimeTrade.ai Data Science Hiring Assignment*
