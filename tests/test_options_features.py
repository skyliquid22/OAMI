import numpy as np
import pandas as pd

from oami.options_features import build_options_sentiment, compute_sentiment_scores


def _make_ohlcv(start: str, volumes: list[float], closes: list[float], highs: list[float], lows: list[float]) -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(volumes), freq="D")
    return pd.DataFrame({
        "t": idx,
        "o": closes,
        "h": highs,
        "l": lows,
        "c": closes,
        "v": volumes,
    })


def test_build_options_sentiment_produces_expected_columns():
    call_frame = _make_ohlcv(
        start="2024-01-01",
        volumes=[100, 110],
        closes=[5.0, 5.5],
        highs=[5.5, 6.0],
        lows=[4.8, 5.2],
    )
    put_frame = _make_ohlcv(
        start="2024-01-01",
        volumes=[90, 85],
        closes=[4.0, 4.2],
        highs=[4.3, 4.4],
        lows=[3.7, 3.8],
    )

    df_options = pd.DataFrame(
        {
            "ticker": ["O:TEST240119C00100000", "O:TEST240119P00090000"],
            "contract_type": ["CALL", "PUT"],
            "strike_price": [100, 90],
            "expiration_date": [pd.Timestamp("2024-01-19")]*2,
            "ohlcv": [call_frame, put_frame],
            "underlying_price": [102, 102],
        }
    )

    features = build_options_sentiment(df_options)
    expected_columns = {
        "Date",
        "put_call_volume_ratio",
        "weighted_average_moneyness",
        "implied_direction_bias",
        "skewed_volume_distribution",
        "near_far_expiry_vol_ratio",
        "realized_vol_proxy",
        "high_low_spread_ratio",
        "term_structure_slope",
        "total_volume",
        "call_volume",
        "put_volume",
        "avg_option_return",
        "call_range_mean",
        "put_range_mean",
    }

    assert expected_columns.issubset(set(features.columns))
    first_row = features.iloc[0]
    assert np.isclose(first_row["put_call_volume_ratio"], 90 / 100)
    assert first_row["total_volume"] == 190


def test_compute_sentiment_scores_returns_expected_metrics():
    call_frame = _make_ohlcv(
        start="2024-01-01",
        volumes=[50, 60],
        closes=[6.0, 6.5],
        highs=[6.6, 6.9],
        lows=[5.8, 6.1],
    )
    put_frame = _make_ohlcv(
        start="2024-01-01",
        volumes=[80, 75],
        closes=[3.5, 3.7],
        highs=[3.8, 3.9],
        lows=[3.3, 3.5],
    )

    df_options = pd.DataFrame(
        {
            "ticker": ["O:TEST240126C00105000", "O:TEST240126P00095000"],
            "contract_type": ["CALL", "PUT"],
            "strike_price": [105, 95],
            "expiration_date": [pd.Timestamp("2024-01-26")]*2,
            "ohlcv": [call_frame, put_frame],
            "underlying_price": [100, 100],
        }
    )

    features = build_options_sentiment(df_options)
    scores = compute_sentiment_scores(features)

    assert {"Date", "option_sentiment_score", "fear_greed_skew"} == set(scores.columns)
    assert len(scores) == len(features)
