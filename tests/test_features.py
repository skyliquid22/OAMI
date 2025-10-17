import pandas as pd
from oami.features import FeatureBuilder

def test_featurebuilder_adds_indicators_and_lags():
    df = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=60),
        "Open": range(60), "High": range(1,61),
        "Low": range(60), "Close": range(1,61),
        "Volume": [100]*60
    })
    out = FeatureBuilder(df).add_indicators().add_lags().add_rolling().finalize()
    for c in ["sma_10","ema_20","rsi","macd","bb_width","atr"]:
        assert any(col.startswith(c) for col in out.columns)
    assert "Close_rollmean_5" in out.columns
