import pandas as pd, os

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
