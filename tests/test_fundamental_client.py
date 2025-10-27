import types
import sys
import pandas as pd

sys.path.append("src")

# Provide dummy polygon module to satisfy imports during tests.
polygon_stub = types.ModuleType("polygon")


class _DummyRESTClient:
    def __init__(self, *args, **kwargs):
        pass


polygon_stub.RESTClient = _DummyRESTClient
exceptions_stub = types.ModuleType("polygon.exceptions")


class _DummyBadResponse(Exception):
    pass


exceptions_stub.BadResponse = _DummyBadResponse
polygon_stub.exceptions = exceptions_stub
sys.modules["polygon"] = polygon_stub
sys.modules["polygon.exceptions"] = exceptions_stub

from oami.data_layer.restapi import FundamentalDataClient


def test_mask_by_window_handles_multiple_date_formats(monkeypatch):
    client = FundamentalDataClient("AAPL", "2020-01-01", "2025-01-01", use_cache=False)

    frame = pd.DataFrame(
        {
            "calendar_date": ["2021-01-01", "2023-02-15", "garbage"],
            "metric": [1, 2, 3],
        }
    )

    masked = client._mask_by_window(frame)
    assert len(masked) == 2
    assert masked.iloc[0]["calendar_date"] == pd.to_datetime("2021-01-01")
