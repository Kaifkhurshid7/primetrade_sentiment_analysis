"""
src/models/sentiment_predictor.py

Random Forest classifier to predict whether a trade will be profitable
given market sentiment, leverage, symbol, direction, and time features.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import (
    MAX_DEPTH,
    N_ESTIMATORS,
    RANDOM_STATE,
    SENTIMENT_ORDER,
    TEST_SIZE,
)


class SentimentTradePredictor:
    """
    Predicts binary trade outcome (profitable / not) from:
    - Sentiment classification (ordinal encoded)
    - Leverage
    - Symbol (label encoded)
    - Direction (long/short)
    - Day of week, hour of day
    - Rolling 7-day average sentiment
    """

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        )
        self.label_encoders: dict = {}
        self.feature_names: list = []
        self.is_fitted = False

    # ── Feature Engineering ────────────────────────────────────────────────

    def _encode_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        closes = df[df["is_close"] & df["closedPnL"].notna()].copy()

        # Ordinal sentiment
        enc_map = {s: i for i, s in enumerate(SENTIMENT_ORDER)}
        closes["sentiment_num"] = closes["classification"].map(enc_map).fillna(2)

        # Rolling sentiment
        daily_sent = (
            closes.groupby("date")["sentiment_num"].first().reset_index()
        )
        daily_sent["rolling_sentiment_7d"] = (
            daily_sent["sentiment_num"].rolling(7, min_periods=1).mean()
        )
        closes = closes.merge(
            daily_sent[["date", "rolling_sentiment_7d"]], on="date", how="left"
        )

        # Time features
        closes["hour"] = closes["time"].dt.hour
        closes["day_of_week"] = closes["time"].dt.dayofweek
        closes["month"] = closes["time"].dt.month

        # Leverage (log scale)
        closes["log_leverage"] = np.log1p(closes["leverage"].fillna(1))

        # Label encode symbol
        if fit:
            le = LabelEncoder()
            closes["symbol_enc"] = le.fit_transform(closes["symbol"].fillna("UNKNOWN"))
            self.label_encoders["symbol"] = le
        else:
            le = self.label_encoders["symbol"]
            sym = closes["symbol"].fillna("UNKNOWN")
            sym = sym.where(sym.isin(le.classes_), other=le.classes_[0])
            closes["symbol_enc"] = le.transform(sym)

        # Direction
        closes["is_long_int"] = closes["is_long"].astype(int)

        # Size (log)
        closes["log_size"] = np.log1p(closes["size"].fillna(0))

        features = [
            "sentiment_num",
            "rolling_sentiment_7d",
            "log_leverage",
            "symbol_enc",
            "is_long_int",
            "hour",
            "day_of_week",
            "month",
            "log_size",
        ]
        return closes[features + ["is_profitable"]].dropna()

    # ── Train / Evaluate ──────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "SentimentTradePredictor":
        data = self._encode_features(df, fit=True)
        self.feature_names = [c for c in data.columns if c != "is_profitable"]

        X = data[self.feature_names].values
        y = data["is_profitable"].astype(int).values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

        self.model.fit(X_train, y_train)
        self.is_fitted = True

        # Store test data for evaluation
        self._X_test = X_test
        self._y_test = y_test
        self._X_train = X_train
        self._y_train = y_train

        return self

    def evaluate(self) -> dict:
        assert self.is_fitted, "Call .fit() first"
        y_pred = self.model.predict(self._X_test)
        y_prob = self.model.predict_proba(self._X_test)[:, 1]

        report = classification_report(self._y_test, y_pred, output_dict=True)
        fpr, tpr, _ = roc_curve(self._y_test, y_prob)
        auc = roc_auc_score(self._y_test, y_prob)

        # 5-fold CV
        all_X = np.vstack([self._X_train, self._X_test])
        all_y = np.concatenate([self._y_train, self._y_test])
        cv_scores = cross_val_score(
            self.model, all_X, all_y, cv=5, scoring="roc_auc", n_jobs=-1
        )

        return {
            "classification_report": report,
            "roc_auc": round(auc, 4),
            "cv_auc_mean": round(cv_scores.mean(), 4),
            "cv_auc_std": round(cv_scores.std(), 4),
            "confusion_matrix": confusion_matrix(self._y_test, y_pred),
            "roc_curve": {"fpr": fpr, "tpr": tpr},
            "feature_importances": dict(
                zip(self.feature_names, self.model.feature_importances_)
            ),
        }

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        data = self._encode_features(df, fit=False)
        X = data[self.feature_names].values
        return self.model.predict_proba(X)[:, 1]

    def feature_importance_df(self) -> pd.DataFrame:
        assert self.is_fitted
        fi = pd.DataFrame({
            "feature": self.feature_names,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False)
        return fi


def plot_model_results(results: dict, figures_dir: Path) -> None:
    """Generate model evaluation charts."""
    import matplotlib.pyplot as plt

    plt.style.use("dark_background")
    bg = "#0d0d1a"
    accent = "#00d4ff"

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=bg)
    fig.suptitle("Random Forest — Trade Profitability Predictor", fontsize=14, color=accent)

    # ROC Curve
    ax = axes[0]
    ax.set_facecolor(bg)
    fpr = results["roc_curve"]["fpr"]
    tpr = results["roc_curve"]["tpr"]
    ax.plot(fpr, tpr, color=accent, lw=2,
            label=f"AUC = {results['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="#555")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()

    # Feature Importance
    ax2 = axes[1]
    ax2.set_facecolor(bg)
    fi = pd.DataFrame(results["feature_importances"].items(),
                      columns=["feature", "importance"]).sort_values("importance")
    colors = [accent if "sentiment" in f else "#ff9f43" for f in fi["feature"]]
    ax2.barh(fi["feature"], fi["importance"], color=colors)
    ax2.set_title("Feature Importance")
    ax2.set_xlabel("Importance")

    # Confusion Matrix
    ax3 = axes[2]
    ax3.set_facecolor(bg)
    cm = results["confusion_matrix"]
    im = ax3.imshow(cm, cmap="Blues")
    ax3.set_xticks([0, 1])
    ax3.set_yticks([0, 1])
    ax3.set_xticklabels(["Not Profit", "Profit"])
    ax3.set_yticklabels(["Not Profit", "Profit"])
    for i in range(2):
        for j in range(2):
            ax3.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)
    ax3.set_title("Confusion Matrix")
    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("Actual")

    path = figures_dir / "12_model_evaluation.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=bg)
    print(f"  ✓ Saved → {path}")
    plt.close(fig)
