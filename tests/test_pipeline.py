"""Unit tests for ingestion, analysis, and model modules."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import SENTIMENT_ORDER
from src.analysis.correlation import compute_lag_correlations, contrarian_backtest
from src.analysis.trader_metrics import leverage_by_sentiment, pnl_by_sentiment, top_traders
from src.ingestion.loader import merge_datasets, validate_merged


@pytest.fixture
def sample_fg():
    dates = pd.date_range("2023-01-01", periods=100)
    classes = (SENTIMENT_ORDER * 20)[:100]
    return pd.DataFrame({
        "date": dates,
        "classification": pd.Categorical(classes, categories=SENTIMENT_ORDER, ordered=True),
        "value": np.linspace(0, 100, 100).astype(int),
    })


@pytest.fixture
def sample_trades(sample_fg):
    n = 500
    rng = np.random.default_rng(42)
    dates = pd.to_datetime(rng.choice(sample_fg["date"].values, size=n))
    df = pd.DataFrame({
        "account":         rng.choice([f"0x{'a' * 40}"], size=n),
        "symbol":          rng.choice(["BTC", "ETH", "SOL"], size=n),
        "execution_price": rng.uniform(100, 50000, n),
        "size":            rng.uniform(0.01, 1.0, n),
        "side":            rng.choice(["BUY", "SELL"], size=n),
        "time":            dates,
        "date":            pd.to_datetime(dates).normalize(),
        "start_position":  rng.uniform(-1, 1, n),
        "event":           rng.choice(["OPEN", "CLOSE"], size=n),
        "closedPnL":       rng.normal(0, 50, n),
        "leverage":        rng.choice([1, 5, 10, 20], size=n),
    })
    df["is_close"] = df["event"] == "CLOSE"
    df["is_long"] = df["side"] == "BUY"
    df["is_profitable"] = df["closedPnL"] > 0
    df.loc[~df["is_close"], "closedPnL"] = 0
    return df


@pytest.fixture
def merged(sample_trades, sample_fg):
    return merge_datasets(sample_trades, sample_fg)


class TestIngestion:
    def test_merge_shape(self, merged, sample_trades):
        assert len(merged) <= len(sample_trades)
        assert "classification" in merged.columns

    def test_no_null_classification(self, merged):
        assert merged["classification"].isna().sum() == 0

    def test_validate_returns_dict(self, merged):
        stats = validate_merged(merged)
        assert isinstance(stats, dict)
        assert stats["total_rows"] == len(merged)


class TestAnalysis:
    def test_pnl_by_sentiment_shape(self, merged):
        result = pnl_by_sentiment(merged)
        assert "mean_pnl" in result.columns
        assert "win_rate" in result.columns
        assert len(result) >= 1

    def test_win_rate_bounds(self, merged):
        result = pnl_by_sentiment(merged)
        assert (result["win_rate"] >= 0).all()
        assert (result["win_rate"] <= 1).all()

    def test_leverage_by_sentiment(self, merged):
        result = leverage_by_sentiment(merged)
        assert "mean_leverage" in result.columns
        assert (result["mean_leverage"] > 0).all()

    def test_top_traders_non_empty(self, merged):
        import config.settings as cfg
        original = cfg.MIN_TRADES_FOR_PROFILING
        cfg.MIN_TRADES_FOR_PROFILING = 1
        result = top_traders(merged, top_n=3)
        cfg.MIN_TRADES_FOR_PROFILING = original
        assert len(result) >= 1

    def test_lag_correlations(self, merged):
        result = compute_lag_correlations(merged)
        assert "pearson_r" in result.columns
        assert ((result["pearson_r"] >= -1) & (result["pearson_r"] <= 1)).all()

    def test_contrarian_backtest_returns_dict(self, merged):
        result = contrarian_backtest(merged)
        assert "daily" in result
        assert "summary" in result
        assert "alpha" in result["summary"]


class TestSmoke:
    def test_full_analysis_runs(self, merged, sample_fg):
        """Ensure all analysis functions complete without exceptions."""
        from src.analysis.correlation import rolling_sentiment_momentum, sentiment_transition_matrix
        from src.analysis.trader_metrics import (
            long_short_ratio_by_sentiment,
            pnl_by_symbol_sentiment,
            statistical_tests,
        )

        pnl_by_sentiment(merged)
        leverage_by_sentiment(merged)
        long_short_ratio_by_sentiment(merged)
        pnl_by_symbol_sentiment(merged)
        compute_lag_correlations(merged)
        contrarian_backtest(merged)
        sentiment_transition_matrix(sample_fg)
        statistical_tests(merged)
        rolling_sentiment_momentum(merged)
