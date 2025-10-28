"""Dataset assembly utilities built atop :class:`OptionFeatureBuilder`."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from .features import OptionFeatureBuilder

DEFAULT_TENOR_BUCKETS = [(0, 7), (8, 30), (31, 90), (91, 9999)]


def build_unified_dataset(
    df_market: pd.DataFrame,
    df_options: pd.DataFrame,
    *,
    tenor_buckets: list[tuple[int, int]] | None = None,
    risk_free_rate: float = 0.05,
    use_open_interest: bool = False,
    eps: float = 1e-6,
    builder_kwargs: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Return a tenor-aware option feature matrix merged with market data."""
    kwargs = dict(builder_kwargs or {})
    kwargs.setdefault("tenor_buckets", tenor_buckets or DEFAULT_TENOR_BUCKETS)
    kwargs.setdefault("risk_free_rate", risk_free_rate)
    kwargs.setdefault("use_open_interest", use_open_interest)
    kwargs.setdefault("eps", eps)

    builder = OptionFeatureBuilder(**kwargs)
    return builder.fit_transform(df_market, df_options)
