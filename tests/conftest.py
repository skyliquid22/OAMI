import pandas as pd, pytest, pathlib
from unittest.mock import patch

@pytest.fixture
def sample_market_df():
    path = pathlib.Path(__file__).parent / "resources" / "market_sample.csv"
    return pd.read_csv(path, parse_dates=["Date"])

@pytest.fixture
def sample_options_df():
    path = pathlib.Path(__file__).parent / "resources" / "options_sample.csv"
    return pd.read_csv(path, parse_dates=["Date"])

@pytest.fixture
def mock_polygon(monkeypatch):
    with patch("oami.data_layer.RESTClient") as MockClient:
        instance = MockClient.return_value
        instance.list_aggs.return_value = []
        instance.get_aggs.return_value = type("obj",(object,),{"results": []})()
        yield instance
