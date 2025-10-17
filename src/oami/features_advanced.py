import pandas as pd
from .features_crossasset import CrossAssetFeatures
from .options_features import build_options_sentiment
from .features_select import FeatureSelector

class AdvancedFeatureBuilder:
    def __init__(self, market_df: pd.DataFrame, options_df: pd.DataFrame | None = None):
        self.market_df = market_df.copy()
        self.options_df = options_df.copy() if options_df is not None else None

    def add_crossasset(self, benchmarks: list[str] | None = None) -> 'AdvancedFeatureBuilder':
        caf = CrossAssetFeatures(self.market_df, benchmarks)
        self.market_df = caf.build()
        return self

    def add_options_implied(self) -> 'AdvancedFeatureBuilder':
        if self.options_df is not None:
            enhanced = build_options_sentiment(self.options_df)
            self.market_df = pd.merge(self.market_df, enhanced, on='Date', how='left')
        return self

    def apply_selection(self, n_components: int = 5) -> 'AdvancedFeatureBuilder':
        selector = FeatureSelector(self.market_df)
        self.market_df = selector.apply_pca(n_components)
        return self

    def finalize(self) -> pd.DataFrame:
        return self.market_df.dropna(axis=0, how='any')
