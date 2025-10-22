import numpy as np
import pandas as pd
import pytest

pytest.importorskip("ta")

from oami.dataset import build_unified_dataset


def _make_ohlcv(periods: int = 40) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=periods, freq="D")
    return pd.DataFrame(
        {
            "t": idx,
            "o": np.linspace(1.0, 1.5, periods),
            "h": np.linspace(1.1, 1.6, periods),
            "l": np.linspace(0.9, 1.4, periods),
            "c": np.linspace(1.05, 1.55, periods),
            "v": np.linspace(100, 140, periods),
        }
    )


def test_unified_dataset_contains_target():
    mkt = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=60),
        "Open": np.linspace(100, 120, 60),
        "High": np.linspace(101, 121, 60),
        "Low": np.linspace(99, 119, 60),
        "Close": np.linspace(100.5, 120.5, 60),
        "Volume": np.linspace(1_000_000, 1_200_000, 60),
    })

    ohlcv_df = _make_ohlcv(60)
    opt = pd.DataFrame(
        {
            "ticker": ["O:TEST240301C00100000", "O:TEST240301P00090000"],
            "contract_type": ["CALL", "PUT"],
            "strike_price": [100, 90],
            "expiration_date": [pd.Timestamp("2024-03-01")] * 2,
            "underlying_price": [100, 100],
            "ohlcv": [ohlcv_df, ohlcv_df.copy()],
        }
    )

    cfg = {"sma_windows": [5, 10], "lags": [1, 2], "rolling_windows": [5]}
    ds = build_unified_dataset(mkt, opt, cfg)
    assert "next_return" in ds.columns
    assert ds["next_return"].notna().any()
