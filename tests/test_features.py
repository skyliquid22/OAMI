from datetime import datetime, timedelta
from pathlib import Path
import sys

import numpy as np
import pandas as pd

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from oami.features import OptionFeatureBuilder  # noqa: E402


def _make_market_frame(start: str = "2024-01-01", periods: int = 5) -> pd.DataFrame:
    dates = pd.date_range(start, periods=periods, freq="D")
    records = []
    for idx, dt in enumerate(dates):
        records.append(
            {
                "ticker": "SPY",
                "volume": 1_000_000 + idx * 10_000,
                "open": 100 + idx,
                "close": 101 + idx,
                "high": 102 + idx,
                "low": 99 + idx,
                "window_start": dt,
                "transactions": 5_000 + idx * 100,
            }
        )
    return pd.DataFrame(records)


def _make_options_frame() -> pd.DataFrame:
    base = datetime(2024, 1, 1)
    rows = []
    expiries = [base + timedelta(days=5), base + timedelta(days=15), base + timedelta(days=60)]
    for day_offset in range(3):
        window = base + timedelta(days=day_offset)
        for exp in expiries:
            rows.append(
                {
                    "ticker": f"O:SPY{exp:%y%m%d}C000{40000 + day_offset}",
                    "volume": 100 + day_offset * 10,
                    "open": 5.0 + day_offset,
                    "close": 5.2 + day_offset,
                    "high": 5.3 + day_offset,
                    "low": 4.9 + day_offset,
                    "window_start": window,
                    "transactions": 50 + day_offset,
                    "underlying_symbol": "SPY",
                    "strike_price": 100 + day_offset,
                    "contract_type": "call",
                    "expiration_date": exp,
                    "open_interest": 150 + day_offset * 5,
                }
            )
            rows.append(
                {
                    "ticker": f"O:SPY{exp:%y%m%d}P000{40000 + day_offset}",
                    "volume": 80 + day_offset * 5,
                    "open": 4.8 + day_offset,
                    "close": 4.9 + day_offset,
                    "high": 5.0 + day_offset,
                    "low": 4.6 + day_offset,
                    "window_start": window,
                    "transactions": 40 + day_offset,
                    "underlying_symbol": "SPY",
                    "strike_price": 100 + day_offset,
                    "contract_type": "put",
                    "expiration_date": exp,
                    "open_interest": 120 + day_offset * 3,
                }
            )
    return pd.DataFrame(rows)


def test_option_feature_builder_generates_expected_columns():
    builder = OptionFeatureBuilder(use_open_interest=True)
    market = _make_market_frame()
    options = _make_options_frame()

    result = builder.fit_transform(market, options)
    expected_cols = {
        "date",
        "tic",
        "option_sentiment_t1_0_7",
        "option_sentiment_t2_8_30",
        "option_sentiment_t3_31_90",
        "option_sentiment_t4_90_plus",
        "iv_mean_t1_0_7",
        "iv_mean_t2_8_30",
        "iv_mean_t3_31_90",
        "iv_mean_t4_90_plus",
        "iv_skew_t1_0_7",
        "iv_skew_t2_8_30",
        "iv_skew_t3_31_90",
        "iv_skew_t4_90_plus",
        "gamma_total_t1_0_7",
        "gamma_total_t2_8_30",
        "oi_call_put_short",
        "oi_call_put_long",
    }
    assert expected_cols.issubset(result.columns)
    assert not result.empty
    assert (result["tic"] == "SPY").all()


def test_option_feature_builder_handles_missing_options():
    builder = OptionFeatureBuilder()
    market = _make_market_frame()
    options = _make_options_frame().iloc[0:0]

    result = builder.fit_transform(market, options)
    assert result.shape[0] == len(market)
    all_feature_cols = [col for col in result.columns if col.startswith(("option_", "iv_", "gamma_", "oi_"))]
    assert result[all_feature_cols].isna().all().all()


def test_option_feature_builder_fill_forward_backward():
    builder = OptionFeatureBuilder()
    market = _make_market_frame(periods=7)
    options = _make_options_frame().query("window_start <= '2024-01-02'")

    result = builder.fit_transform(market, options)
    sentiment_cols = [col for col in result.columns if col.startswith("option_sentiment")]
    assert not result[sentiment_cols].isna().all(axis=None)
    # Ensure forward/back-fill applied
    for col in sentiment_cols:
        if result[col].notna().any():
            assert result.groupby("tic")[col].apply(lambda s: s.isna().sum()).max() == 0
