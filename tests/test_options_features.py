import pandas as pd
from oami.options_features import build_options_sentiment

def test_options_sentiment_features_created():
    df = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=20),
        "PutVol": range(20,40), "CallVol": range(40,60),
        "PutOI": range(10,30), "CallOI": range(12,32),
        "PutCallVolRatio": [0.5]*20, "SentimentIndex": [0.5]*20
    })
    out = build_options_sentiment(df)
    assert "put_call_oi_ratio" in out.columns
    assert any(c.startswith("sentiment_rollmean") for c in out.columns)
