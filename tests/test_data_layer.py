import os
import logging
import sys
from pathlib import Path

import pandas as pd
import pytest

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


def test_load_or_fetch_contracts_skips_coverage_on_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(data_layer.cache_manager, "H5_PATH", tmp_path / "missing_store.h5")
    monkeypatch.setattr(data_layer, "read_hdf", lambda key: pd.DataFrame())
    monkeypatch.setattr(data_layer, "fetch_option_contracts", lambda **kwargs: [])
    monkeypatch.setattr(data_layer, "read_option_coverage", lambda symbol, timeframe: (None, None))

    calls: list[tuple] = []

    def fake_update(symbol, timeframe, start, end):
        calls.append((symbol, timeframe, start, end))

    monkeypatch.setattr(data_layer, "update_option_coverage", fake_update)

    meta, cache, removals = data_layer._load_or_fetch_contracts(
        symbol="SPY",
        timeframe="1D",
        from_="2025-08-24",
        to="2025-09-22",
        strike_min=600.0,
        strike_max=650.0,
        bucket_cache={},
        bucket_removals={},
    )

    assert meta.empty
    assert calls == []


def test_polygon_flatfile_client_logs_and_returns(monkeypatch, caplog):
    class Result:
        stdout = "                           PRE global_crypto/\n                           PRE global_forex/\n"

    captured_cmd = {}

    def fake_run(cmd, check, capture_output, text):
        captured_cmd["cmd"] = cmd
        return Result()

    monkeypatch.setattr(data_layer.subprocess, "run", fake_run)

    client = data_layer.PolygonFlatfileClient()
    with caplog.at_level(logging.INFO):
        output = client.list_root_directories()

    assert captured_cmd["cmd"][:4] == ["aws", "s3", "ls", "s3://flatfiles/"]
    assert output == ["global_crypto/", "global_forex/"]
    assert any("global_crypto" in message for message in caplog.messages)


def test_polygon_flatfile_client_download_single_file(monkeypatch, tmp_path, caplog):
    class Result:
        stdout = "downloaded"
        stderr = ""

    captured_cmd = {}

    def fake_run(cmd, check, capture_output, text):
        captured_cmd["cmd"] = cmd
        return Result()

    monkeypatch.setattr(data_layer.subprocess, "run", fake_run)
    monkeypatch.chdir(tmp_path)

    client = data_layer.PolygonFlatfileClient()
    with caplog.at_level(logging.INFO):
        dest = client.download_flatfile("stocks", "trades_v1", 2024, 3, 7)

    expected_cmd = [
        "aws",
        "s3",
        "cp",
        "s3://flatfiles/us_stocks_sip/trades_v1/2024/03/2024-03-07.csv.gz",
        str(tmp_path / "data/flatfiles/us_stocks_sip/trades_v1/2024/03/2024-03-07.csv.gz"),
        "--endpoint-url",
        "https://files.polygon.io",
    ]
    assert captured_cmd["cmd"] == expected_cmd
    assert dest == tmp_path / "data/flatfiles/us_stocks_sip/trades_v1/2024/03/2024-03-07.csv.gz"
    assert dest.parent.exists()
    assert any("download complete" in message for message in caplog.messages)


def test_polygon_flatfile_client_download_directory(monkeypatch, tmp_path):
    class Result:
        stdout = "completed"
        stderr = ""

    captured_cmd = {}

    def fake_run(cmd, check, capture_output, text):
        captured_cmd["cmd"] = cmd
        return Result()

    monkeypatch.setattr(data_layer.subprocess, "run", fake_run)
    monkeypatch.chdir(tmp_path)

    client = data_layer.PolygonFlatfileClient()
    dest = client.download_flatfile("options", "trades_v1", 2023, 12)

    expected_cmd = [
        "aws",
        "s3",
        "cp",
        "s3://flatfiles/us_options_opra/trades_v1/2023/12/",
        str(tmp_path / "data/flatfiles/us_options_opra/trades_v1/2023/12"),
        "--endpoint-url",
        "https://files.polygon.io",
        "--recursive",
    ]
    assert captured_cmd["cmd"] == expected_cmd
    assert dest == tmp_path / "data/flatfiles/us_options_opra/trades_v1/2023/12"
    assert dest.exists()
