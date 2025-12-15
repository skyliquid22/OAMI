"""Compute feature importances for StockFeatureBuilder outputs using local cache."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from oami.features import StockFeatureBuilder


def load_cached_market(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    for ticker in tickers:
        path = Path(f"data/cache/stocks/{ticker}.parquet")
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        rename_map = {"ticker": "tic", "window_start": "date"}
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        df["date"] = pd.to_datetime(df["date"])
        mask = (df["date"] >= start_ts) & (df["date"] <= end_ts)
        df = df.loc[mask]
        if df.empty:
            continue
        df["tic"] = ticker
        frames.append(df[["date", "tic", "open", "high", "low", "close", "volume"]])
    if not frames:
        raise RuntimeError("No cached market data found. Run tester.py first to populate caches.")
    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values(["tic", "date"]).reset_index(drop=True)


def main() -> None:
    TICKERS = ["AAPL"]
    START = "2020-01-01"
    END = "2023-12-31"
    TRAIN_END = pd.Timestamp("2022-12-31")

    df_market = load_cached_market(TICKERS, START, END)

    builder = StockFeatureBuilder()
    df_feat = builder.transform(df_market)
    df_feat["date"] = pd.to_datetime(df_feat["date"])
    df_feat = df_feat.sort_values(["tic", "date"]).reset_index(drop=True)

    df_feat["target"] = df_feat.groupby("tic")["close"].shift(-1) / df_feat["close"] - 1.0
    df_feat = df_feat.dropna(subset=["target"])

    train = df_feat[df_feat["date"] <= TRAIN_END]
    base_cols = ["date", "tic", "open", "high", "low", "close", "volume", "target"]
    feature_cols = [
        col for col in df_feat.columns if col not in base_cols and pd.api.types.is_numeric_dtype(df_feat[col])
    ]

    if not feature_cols:
        raise RuntimeError("No numeric feature columns available.")

    X_train = train[feature_cols]
    y_train = train["target"]

    model = RandomForestRegressor(n_estimators=400, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("Top 20 feature importances (predicting next interval return):")
    for feature, score in importances.head(20).items():
        print(f"{feature:30s} {score:.4f}")


if __name__ == "__main__":
    main()
