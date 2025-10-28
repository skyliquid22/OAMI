"""Advanced feature pipelines built atop :class:`OptionFeatureBuilder`."""

from __future__ import annotations

from typing import Iterable, Mapping

import pandas as pd

from .features import OptionFeatureBuilder
from .features_crossasset import CrossAssetFeatures
from .features_select import FeatureSelector


class AdvancedFeatureBuilder:
    """Compose cross-asset and option-derived features into a unified frame."""

    def __init__(
        self,
        market_df: pd.DataFrame,
        options_df: pd.DataFrame | None = None,
        *,
        option_builder_kwargs: Mapping[str, object] | None = None,
    ) -> None:
        self.market_df = market_df.copy()
        self.options_df = options_df.copy() if options_df is not None else None
        self.option_builder = OptionFeatureBuilder(**(option_builder_kwargs or {}))

    def add_crossasset(self, benchmarks: Iterable[str] | None = None) -> "AdvancedFeatureBuilder":
        cross_asset = CrossAssetFeatures(self.market_df, list(benchmarks) if benchmarks else None)
        self.market_df = cross_asset.build()
        return self

    def add_options_implied(self) -> "AdvancedFeatureBuilder":
        if self.options_df is None:
            return self

        features = self.option_builder.fit_transform(self.market_df, self.options_df)
        options_columns = [col for col in features.columns if col not in {"date", "tic"}]

        merged = self.market_df.copy()
        merged["date"] = pd.to_datetime(merged["date"] if "date" in merged.columns else merged["window_start"]).dt.normalize()
        merged["tic"] = merged.get("tic", merged.get("ticker", None))
        merged["tic"] = merged["tic"].astype(str).str.upper()

        enriched = features[["date", "tic", *options_columns]]
        merged = merged.merge(enriched, on=["date", "tic"], how="left")
        self.market_df = merged
        return self

    def apply_selection(self, n_components: int = 5) -> "AdvancedFeatureBuilder":
        selector = FeatureSelector(self.market_df)
        self.market_df = selector.apply_pca(n_components)
        return self

    def finalize(self) -> pd.DataFrame:
        return self.market_df.dropna(axis=0, how="any").reset_index(drop=True)
