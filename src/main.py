from oami.logging_config import setup_json_logging
from oami.data_layer import get_market_data, get_options_data
from oami.dataset import build_unified_dataset
from oami.__version__ import __version__
import logging, pandas as pd

if __name__ == "__main__":
    setup_json_logging()
    logging.info("Starting OAMI pipeline", extra={"version": __version__})
    # Prefer offline sample data if available
    try:
        df_m = pd.read_csv("data/csv/day/SPY.csv", parse_dates=["Date"])
        df_o = pd.read_csv("data/csv/day/SPY_options.csv", parse_dates=["Date"])
    except Exception:
        df_m = get_market_data("SPY", "2024-01-01", "2025-10-01")
        df_o = get_options_data("SPY", "2024-01-01", "2025-10-01")
    feature_cfg = {
        "sma_windows":[10,20,50],"ema_windows":[10,20,50],"rsi_window":14,
        "macd_fast":12,"macd_slow":26,"macd_signal":9,"bb_window":20,"bb_std":2,
        "lags":[1,2,3,5],"rolling_windows":[5,10,20]
    }
    df = build_unified_dataset(df_m, df_o, feature_cfg, target="next_return")
    logging.info("Dataset ready", extra={"rows": len(df), "cols": len(df.columns)})
