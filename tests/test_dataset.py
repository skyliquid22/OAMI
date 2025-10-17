import pandas as pd
from oami.dataset import build_unified_dataset

def test_unified_dataset_contains_target():
    mkt = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=40),
        "Open": range(40), "High": range(1,41),
        "Low": range(40), "Close": range(1,41),
        "Volume": [100]*40
    })
    opt = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=40),
        "PutVol": [1]*40, "CallVol": [2]*40,
        "PutOI": [3]*40, "CallOI": [4]*40,
        "PutCallVolRatio": [0.5]*40, "SentimentIndex": [0.5]*40
    })
    cfg = {"sma_windows":[5,10], "lags":[1,2], "rolling_windows":[5]}
    ds = build_unified_dataset(mkt, opt, cfg)
    assert "next_return" in ds.columns
    assert ds["next_return"].notna().any()
