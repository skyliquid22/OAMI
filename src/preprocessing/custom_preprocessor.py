"""Custom preprocessing pipeline for FinRL using option-derived features."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .option_feature_builder import OptionFeatureBuilder


class CustomFinRLPreprocessor:
    """Prepare market and option data for FinRL training workflows."""

    def __init__(
        self,
        tenor_buckets: Optional[list[tuple[int, int]]] = None,
        risk_free_rate: float = 0.05,
        use_open_interest: bool = False,
        cache_dir: str | Path = "./data/cache",
    ) -> None:
        self.tenor_buckets = tenor_buckets or [(0, 7), (8, 30), (31, 90), (91, 9999)]
        self.risk_free_rate = risk_free_rate
        self.use_open_interest = use_open_interest
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.feature_builder = OptionFeatureBuilder(
            tenor_buckets=self.tenor_buckets,
            risk_free_rate=self.risk_free_rate,
            use_open_interest=self.use_open_interest,
        )

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def fit_transform(self, df_market: pd.DataFrame, df_options: pd.DataFrame) -> pd.DataFrame:
        """Return enriched DataFrame ready for FinRL environments."""
        print("✅ Starting preprocessing: market shape", df_market.shape, "options shape", df_options.shape)

        features = self.feature_builder.fit_transform(df_market, df_options)
        print("✅ Option features computed:", [col for col in features.columns if col.startswith(("option_", "iv_", "gamma_", "oi_"))])

        merged = features.copy()
        merged = merged.sort_values(["tic", "date"]).reset_index(drop=True)
        merged["date"] = pd.to_datetime(merged["date"])

        numeric_cols = merged.select_dtypes(include=[np.number]).columns
        merged[numeric_cols] = merged.groupby("tic")[numeric_cols].apply(lambda frame: frame.ffill().bfill())

        merged = merged.dropna(subset=["open", "high", "low", "close"])

        expected_prefix = ["date", "tic", "open", "high", "low", "close", "volume"]
        cols = expected_prefix + [col for col in merged.columns if col not in expected_prefix]
        merged = merged[cols]

        print("✅ Final preprocessed shape:", merged.shape)
        return merged

    def save_cache(self, df: pd.DataFrame, name: str) -> Path:
        path = self.cache_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        print(f"✅ Saved cache to {path}")
        return path

    def load_cache(self, name: str) -> Optional[pd.DataFrame]:
        path = self.cache_dir / f"{name}.parquet"
        if not path.exists():
            print(f"⚠️ Cache not found: {path}")
            return None
        print(f"✅ Loaded cache from {path}")
        return pd.read_parquet(path)

    def prepare_train_test_split(self, df: pd.DataFrame, train_end_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        cutoff = pd.to_datetime(train_end_date)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["tic", "date"])  # ensure ordering
        df_train = df[df["date"] <= cutoff].copy()
        df_test = df[df["date"] > cutoff].copy()
        print("✅ Train DataFrame shape:", df_train.shape)
        print("✅ Test DataFrame shape:", df_test.shape)
        return df_train, df_test
