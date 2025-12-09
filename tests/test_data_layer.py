import os
import logging
import sqlite3
import sys
from pathlib import Path
import builtins
from datetime import date

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


def test_build_and_load_ticker_cache(tmp_path):
    raw_root = tmp_path / "raw" / "us_stocks_sip" / "day_aggs_v1"
    month_dir = raw_root / "2024" / "01"
    month_dir.mkdir(parents=True, exist_ok=True)

    day_one = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "open": 100.0,
                "high": 101.0,
                "low": 99.5,
                "close": 100.5,
                "volume": 1_000,
                "window_start": "2024-01-01T09:30:00Z",
            },
            {
                "ticker": "MSFT",
                "open": 200.0,
                "high": 205.0,
                "low": 198.0,
                "close": 204.0,
                "volume": 2_000,
                "window_start": "2024-01-01T09:30:00Z",
            },
        ]
    )
    day_one.to_csv(month_dir / "2024-01-01.csv.gz", index=False, compression="gzip")

    day_two = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "open": 101.0,
                "high": 102.0,
                "low": 100.0,
                "close": 101.5,
                "volume": 1_100,
                "window_start": "2024-01-02T09:30:00Z",
            },
            {
                "ticker": "MSFT",
                "open": 204.0,
                "high": 206.0,
                "low": 203.0,
                "close": 205.5,
                "volume": 2_100,
                "window_start": "2024-01-02T09:30:00Z",
            },
        ]
    )
    day_two.to_csv(month_dir / "2024-01-02.csv.gz", index=False, compression="gzip")

    cache_root = tmp_path / "cache" / "stocks"
    meta_path = tmp_path / "meta" / "stocks_index.db"

    results = data_layer.build_ticker_cache(
        ["AAPL", "MSFT"],
        raw_root=raw_root,
        cache_root=cache_root,
        meta_path=meta_path,
    )

    assert set(results) == {"AAPL", "MSFT"}
    for path in results.values():
        assert Path(path).exists()

    conn = sqlite3.connect(meta_path)
    meta_rows = conn.execute(
        "SELECT ticker, start_date, end_date, rows FROM stocks_cache_index ORDER BY ticker"
    ).fetchall()
    conn.close()

    assert meta_rows == [
        ("AAPL", "2024-01-01", "2024-01-02", 2),
        ("MSFT", "2024-01-01", "2024-01-02", 2),
    ]

    loaded = data_layer.load_ticker_data(
        ["AAPL", "MSFT"],
        "2024-01-01",
        "2024-01-02",
        cache_root=cache_root,
        meta_path=meta_path,
    )

    assert len(loaded) == 4
    assert set(loaded["ticker"]) == {"AAPL", "MSFT"}
    assert loaded["window_start"].min() == pd.Timestamp("2024-01-01 09:30:00")
    assert loaded["window_start"].max() == pd.Timestamp("2024-01-02 09:30:00")

    data_layer.build_ticker_cache(
        ["AAPL", "MSFT"],
        raw_root=raw_root,
        cache_root=cache_root,
        meta_path=meta_path,
    )

    loaded_again = data_layer.load_ticker_data(
        ["AAPL", "MSFT"],
        "2024-01-01",
        "2024-01-02",
        cache_root=cache_root,
        meta_path=meta_path,
    )

    assert len(loaded_again) == len(loaded)
    pd.testing.assert_frame_equal(loaded.sort_index(axis=1), loaded_again.sort_index(axis=1))


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
        stderr = ""

    commands = []

    def fake_run(cmd, check, capture_output, text):
        commands.append(cmd)
        if len(cmd) >= 3 and cmd[1] == "configure":
            return Result()
        return Result()

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA_TEST")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "SECRET_TEST")
    monkeypatch.setattr(data_layer.subprocess, "run", fake_run)

    client = data_layer.PolygonFlatfileClient()
    with caplog.at_level(logging.INFO):
        output = client.list_root_directories()

    assert commands[2][:4] == ["aws", "s3", "ls", "s3://flatfiles/"]
    assert commands[0][:3] == ["aws", "configure", "set"]
    assert output == ["global_crypto/", "global_forex/"]
    assert any("global_crypto" in message for message in caplog.messages)


def test_polygon_flatfile_client_list_files_for_month(monkeypatch):
    class Result:
        stdout = (
            "2024-03-07 00:00:00          0 2024-03-07.csv.gz\n"
            "2024-03-08 00:00:00          0 2024-03-08.csv.gz\n"
        )
        stderr = ""

    commands = []

    def fake_run(cmd, check, capture_output, text):
        commands.append(cmd)
        return Result()

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA_TEST")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "SECRET_TEST")
    monkeypatch.setattr(data_layer.subprocess, "run", fake_run)

    client = data_layer.PolygonFlatfileClient()
    files = client.list_files_for_month("options", 2024, 3)

    list_cmd = commands[2]
    assert list_cmd[:4] == ["aws", "s3", "ls", "s3://flatfiles/us_options_opra/day_aggs_v1/2024/03/"]
    assert files == {"2024-03-07.csv.gz", "2024-03-08.csv.gz"}


def test_polygon_flatfile_client_download_single_file(monkeypatch, tmp_path, caplog):
    class Result:
        stdout = "downloaded"
        stderr = ""

    commands = []

    def fake_run(cmd, check, capture_output, text):
        commands.append(cmd)
        if len(cmd) >= 3 and cmd[1] == "configure":
            return Result()
        return Result()

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA_TEST")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "SECRET_TEST")
    monkeypatch.setattr(data_layer.subprocess, "run", fake_run)
    monkeypatch.chdir(tmp_path)

    client = data_layer.PolygonFlatfileClient()
    with caplog.at_level(logging.INFO):
        dest = client.download_flatfile("stocks", 2024, 3, 7, timeframe="day")

    expected_cmd = [
        "aws",
        "s3",
        "cp",
        "s3://flatfiles/us_stocks_sip/day_aggs_v1/2024/03/2024-03-07.csv.gz",
        str(tmp_path / "data/flatfiles/us_stocks_sip/day_aggs_v1/2024/03/2024-03-07.csv.gz"),
        "--endpoint-url",
        "https://files.polygon.io",
    ]
    assert commands[-1] == expected_cmd
    assert dest == tmp_path / "data/flatfiles/us_stocks_sip/day_aggs_v1/2024/03/2024-03-07.csv.gz"
    assert dest.parent.exists()
    assert any("download complete" in message for message in caplog.messages)


def test_polygon_flatfile_client_download_directory(monkeypatch, tmp_path):
    class ConfigureResult:
        stdout = ""
        stderr = ""

    class ListResult:
        stdout = (
            "2023-12-01 00:00:00          0 2023-12-01.csv.gz\n"
            "2023-12-02 00:00:00          0 2023-12-02.csv.gz\n"
        )
        stderr = ""

    class CopyResult:
        stdout = "downloaded"
        stderr = ""

    commands = []

    def fake_run(cmd, check, capture_output, text):
        commands.append(cmd)
        if len(cmd) >= 3 and cmd[1] == "configure":
            return ConfigureResult()
        if cmd[:3] == ["aws", "s3", "ls"]:
            return ListResult()
        return CopyResult()

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA_TEST")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "SECRET_TEST")
    monkeypatch.setattr(data_layer.subprocess, "run", fake_run)
    monkeypatch.chdir(tmp_path)

    client = data_layer.PolygonFlatfileClient()
    dest = client.download_flatfile("options", 2023, 12)

    expected_cmd = [
        "aws",
        "s3",
        "cp",
        "s3://flatfiles/us_options_opra/day_aggs_v1/2023/12/2023-12-02.csv.gz",
        str(tmp_path / "data/flatfiles/us_options_opra/day_aggs_v1/2023/12/2023-12-02.csv.gz"),
        "--endpoint-url",
        "https://files.polygon.io",
    ]
    assert commands[-1] == expected_cmd
    assert dest == tmp_path / "data/flatfiles/us_options_opra/day_aggs_v1/2023/12"


def test_get_option_flatfile_data_filters(tmp_path, monkeypatch):
    base = tmp_path / "flatfiles_root"
    target_dir = base / "us_options_opra" / "day_aggs_v1" / "2024" / "03"
    target_dir.mkdir(parents=True, exist_ok=True)
    data = pd.DataFrame(
        {
            "ticker": ["O:XYZ240307C00050000", "O:XYZ240307P00040000", "O:ABC240307C00010000"],
            "volume": [100, 80, 60],
            "open": [1.0, 2.0, 3.0],
            "close": [1.5, 2.5, 3.5],
            "high": [1.6, 2.6, 3.6],
            "low": [0.9, 1.9, 2.9],
            "window_start": [
                "2024-03-07T09:30:00.123456789Z",
                "2024-03-07T09:30:00.123456789Z",
                "2024-03-07T09:30:00.123456789Z",
            ],
            "transactions": [10, 12, 4],
        }
    )
    path = target_dir / "2024-03-07.csv.gz"
    data.to_csv(path, index=False, compression="gzip")

    result = data_layer.get_option_flatfile_data(
        underlying_ticker="XYZ",
        as_of="2024-03-07",
        expiration_start="2024-03-07",
        expiration_end="2024-03-07",
        strike_lower=45,
        strike_upper=55,
        contract_type="call",
        base_dir=base,
    )

    assert len(result) == 1
    assert result.iloc[0]["ticker"] == "O:XYZ240307C00050000"
    assert "strike_price" in result.columns
    assert float(result.iloc[0]["strike_price"]) == 50.0
    assert result.iloc[0]["window_start"] == date(2024, 3, 7)


def test_get_option_flatfile_data_missing_file(tmp_path):
    base = tmp_path / "flatfiles_root"
    base.mkdir(parents=True, exist_ok=True)

    result = data_layer.get_option_flatfile_data(
        underlying_ticker="XYZ",
        as_of="2024-03-07",
        as_to="2024-03-07",
        base_dir=base,
    )
    assert result.empty


def test_get_option_flatfile_data_triggers_download(tmp_path, monkeypatch):
    base = tmp_path / "flatfiles_root"
    base.mkdir(parents=True, exist_ok=True)

    class FakeClient:
        def __init__(self):
            self.calls: list[tuple] = []

        def list_files_for_month(self, data_type, year, month, timeframe="day"):
            self.calls.append(("list", data_type, year, month, timeframe))
            return {f"{year:04d}-{month:02d}-07.csv.gz"}

        def download_flatfile(self, data_type, year, month, day=None, timeframe="day", available_files=None):
            self.calls.append((data_type, year, month, day, timeframe))
            expiration = pd.Timestamp(year=year, month=month, day=day)
            path = data_layer._option_flatfile_path(expiration, base)
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "ticker": ["O:XYZ240307C00050000"],
                    "volume": [10],
                    "open": [1.0],
                    "close": [1.5],
                    "high": [1.6],
                    "low": [0.9],
                    "window_start": ["2024-03-07T09:30:00.123456789Z"],
                    "transactions": [5],
                }
            ).to_csv(path, index=False, compression="gzip")
            return path

    monkeypatch.setattr(data_layer, "PolygonFlatfileClient", FakeClient)

    result = data_layer.get_option_flatfile_data(
        underlying_ticker="XYZ",
        as_of="2024-03-07",
        as_to="2024-03-07",
        base_dir=base,
    )
    assert len(result) == 1
    assert result.iloc[0]["ticker"] == "O:XYZ240307C00050000"


def test_build_option_cache_filters_and_persists(tmp_path):
    flatfile_root = tmp_path / "flatfiles"
    cache_root = tmp_path / "cache"
    day_dir = flatfile_root / "us_options_opra" / "day_aggs_v1" / "2025" / "03"
    day_dir.mkdir(parents=True, exist_ok=True)

    ts_1 = pd.Timestamp("2025-03-10 09:30:00")
    ts_2 = pd.Timestamp("2025-03-10 10:30:00")

    df_initial = pd.DataFrame(
        [
            {
                "ticker": "O:A250321C00125000",
                "volume": 10,
                "open": 1.0,
                "close": 1.2,
                "high": 1.3,
                "low": 0.9,
                "window_start": ts_1.value,
                "transactions": 5,
            },
            {
                "ticker": "O:A250321P00125000",
                "volume": 15,
                "open": 2.0,
                "close": 2.2,
                "high": 2.4,
                "low": 1.8,
                "window_start": ts_2.value,
                "transactions": 6,
            },
            {
                # Different underlying, should be excluded.
                "ticker": "O:B250321C00125000",
                "volume": 50,
                "open": 5.0,
                "close": 5.5,
                "high": 5.7,
                "low": 4.9,
                "window_start": ts_1.value,
                "transactions": 10,
            },
        ]
    )
    path_initial = day_dir / "2025-03-10.csv.gz"
    df_initial.to_csv(path_initial, index=False, compression="gzip")

    result_first = data_layer.build_option_cache(
        ["A"],
        "2025-03-10",
        "2025-03-10",
        flatfile_root=flatfile_root,
        cache_root=cache_root,
    )

    expected_columns = [
        "ticker",
        "volume",
        "open",
        "close",
        "high",
        "low",
        "window_start",
        "underlying_symbol",
        "strike_price",
        "contract_type",
        "expiration_date",
    ]

    assert not result_first.empty
    assert list(result_first.columns) == expected_columns
    assert set(result_first["underlying_symbol"]) == {"A"}
    assert set(result_first["contract_type"]) == {"call", "put"}
    assert pd.api.types.is_datetime64_ns_dtype(result_first["window_start"])
    cache_path = cache_root / "A.parquet"
    assert cache_path.exists()
    loaded_first = pd.read_parquet(cache_path)
    pd.testing.assert_frame_equal(result_first, loaded_first)

    ts_next = pd.Timestamp("2025-03-11 09:30:00")
    df_follow_up = pd.DataFrame(
        [
            {
                "ticker": "O:A250321C00125000",
                "volume": 8,
                "open": 1.1,
                "close": 1.0,
                "high": 1.2,
                "low": 0.8,
                "window_start": ts_next.value,
                "transactions": 7,
            }
        ]
    )
    path_follow_up = day_dir / "2025-03-11.csv.gz"
    df_follow_up.to_csv(path_follow_up, index=False, compression="gzip")

    result_second = data_layer.build_option_cache(
        ["A"],
        "2025-03-11",
        "2025-03-11",
        flatfile_root=flatfile_root,
        cache_root=cache_root,
    )

    assert len(result_second) == len(result_first) + 1
    assert (result_second["window_start"] == ts_next.normalize()).any()

    multi_result = data_layer.build_option_cache(
        ["A", "B"],
        "2025-03-10",
        "2025-03-11",
        flatfile_root=flatfile_root,
        cache_root=cache_root,
    )

    assert set(multi_result) == {"A", "B"}
    assert not multi_result["A"].empty
    assert set(multi_result["B"]["underlying_symbol"]) == {"B"}

    loaded = data_layer.load_option_data(
        ["A"],
        "2025-03-10",
        "2025-03-11",
        flatfile_root=flatfile_root,
        cache_root=cache_root,
    )

    assert len(loaded) == len(result_second)
    assert set(loaded["contract_type"]) == {"call", "put"}
    assert loaded["window_start"].min() == pd.Timestamp("2025-03-10")
    assert loaded["window_start"].max() == pd.Timestamp("2025-03-11")

    empty = data_layer.load_option_data(
        ["C"],
        "2025-03-10",
        "2025-03-11",
        flatfile_root=flatfile_root,
        cache_root=cache_root,
        rebuild_missing=False,
    )
    assert empty.empty
