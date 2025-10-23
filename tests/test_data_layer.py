import os
import sys
from pathlib import Path

import pandas as pd

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

if "polygon" not in sys.modules:
    import types

    polygon_stub = types.ModuleType("polygon")
    exceptions_stub = types.ModuleType("polygon.exceptions")

    class _DummyRESTClient:
        def __init__(self, *args, **kwargs):
            pass

    class _DummyBadResponse(Exception):
        pass

    polygon_stub.RESTClient = _DummyRESTClient
    exceptions_stub.BadResponse = _DummyBadResponse
    polygon_stub.exceptions = exceptions_stub
    sys.modules["polygon"] = polygon_stub
    sys.modules["polygon.exceptions"] = exceptions_stub

from oami import data_layer  # noqa: E402

def test_cache_path_and_io(tmp_path):
    symbol = "TEST"
    df = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=3),
        "Close": [100,101,102]
    })
    out_dir = tmp_path / "day"
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{symbol}.csv"
    df.to_csv(p, index=False)
    loaded = pd.read_csv(p)
    assert not loaded.empty
    assert "Close" in loaded.columns


def test_get_options_data_structured_output(monkeypatch):
    ticker = "O:XYZ250101C00050000"

    def fake_get_market_data(*args, **kwargs):
        return pd.DataFrame(
            {
                "Open": [100.0, 101.0],
                "High": [102.0, 103.0],
                "Low": [99.0, 100.0],
                "Close": [101.0, 102.0],
            }
        )

    def fake_load_or_fetch_contracts(
        symbol,
        timeframe,
        from_,
        to,
        strike_min,
        strike_max,
        bucket_cache,
        bucket_removals,
    ):
        bucket_key = data_layer.make_option_key(ticker, timeframe=timeframe)
        bucket_df = pd.DataFrame(
            {
                "ticker": [ticker],
                "t": [pd.Timestamp("2025-01-02 09:30")],
                "o": [1.0],
                "h": [1.0],
                "l": [1.0],
                "c": [1.0],
                "v": [100],
            }
        )
        contracts = pd.DataFrame(
            {
                "ticker": [ticker],
                "expiration_date": [pd.Timestamp("2025-01-05")],
                "strike_price": [50.0],
                "contract_type": ["call"],
            }
        )
        return contracts, {bucket_key: bucket_df}, {}

    monkeypatch.setattr(data_layer, "get_market_data", fake_get_market_data)
    monkeypatch.setattr(data_layer, "_load_or_fetch_contracts", fake_load_or_fetch_contracts)
    monkeypatch.setattr(data_layer, "_persist_option_buckets", lambda *args, **kwargs: None)

    def _iter_passthrough(iterable, **_kwargs):
        for item in iterable:
            yield item

    monkeypatch.setattr(data_layer, "_progress_iter", _iter_passthrough)

    result = data_layer.get_options_data("SPY", "2025-01-01", "2025-01-05", interval="1D", look_forward=0)

    assert list(result.columns) == ["expiration_date", "strike_price", "contract_type", "ohlcv"]
    assert list(result.index) == [ticker]
    assert result.loc[ticker, "contract_type"] == "call"
    assert result.loc[ticker, "strike_price"] == 50.0
    assert isinstance(result.loc[ticker, "ohlcv"], pd.DataFrame)
