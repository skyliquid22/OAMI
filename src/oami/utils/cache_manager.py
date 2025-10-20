"""
cache_manager.py (HDF5 Version 2)
---------------------------------
Hierarchical HDF5 cache for OAMI data.

Options data key format:
    /options/{expiry_year}/{expiry_month}/{underlying}/{timeframe}/{ticker}

Stocks and features keep a simpler pattern:
    /stocks/{symbol}/{timeframe}
    /features/{symbol}/{timeframe}
"""

import re
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd

H5_PATH = Path("./data/cache/oami_store.h5")

# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------
MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


def parse_occ_ticker(ticker: str) -> dict:
    """Parse OCC-style option ticker to extract expiry and underlying.

    Parameters
    ----------
    ticker : str
        OCC option ticker, e.g., "O:SPY250117C00470000"

    Returns
    -------
    dict
        {
          'underlying': 'SPY',
          'year': 2025,
          'month': 'JAN',
          'day': 17
        }
    """
    # Match OCC pattern: O:SYMYYMMDDT########
    m = re.match(r"O:([A-Z]{1,6})(\d{2})(\d{2})(\d{2})([CP])(\d{8})", ticker)
    if not m:
        raise ValueError(f"Invalid OCC ticker: {ticker}")
    symbol, yy, mm, dd, opt_type, strike = m.groups()
    year = 2000 + int(yy)
    month = MONTHS[int(mm) - 1]
    return {"underlying": symbol, "year": year, "month": month, "day": int(dd)}


def get_timeframe_str(multiplier: int, timespan: str) -> str:
    """Return normalized timeframe code like '1D', '4H', etc."""
    return f"{multiplier}{timespan[0].upper()}"


def _sanitize(s: str) -> str:
    """Make safe for HDF5 key."""
    return s.replace(":", "_").replace("/", "_")


# -------------------------------------------------------------------------
# Key builders
# -------------------------------------------------------------------------
def make_option_key(ticker: str, timeframe: str) -> str:
    """Build the HDF5 key for an options contract."""
    info = parse_occ_ticker(ticker)
    return f"/options/{info['year']}/{info['month']}/{info['underlying']}/{timeframe}/{_sanitize(ticker)}"


def make_stock_key(symbol: str, timeframe: str) -> str:
    """Build key for stock OHLCV data."""
    return f"/stocks/{symbol}/{timeframe}"


def make_feature_key(symbol: str, timeframe: str) -> str:
    """Build key for features."""
    return f"/features/{symbol}/{timeframe}"


# -------------------------------------------------------------------------
# IO wrappers
# -------------------------------------------------------------------------
def write_hdf(df: pd.DataFrame, key: str):
    """Write DataFrame to HDF5 store."""
    if df.empty:
        logging.warning("Skipping empty write for %s", key)
        return
    H5_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_hdf(
        H5_PATH,
        key=key,
        mode="a",
        format="table",
        complevel=9,
        complib="blosc"
    )
    logging.info("Saved %d rows → %s", len(df), key)


def read_hdf(key: str) -> pd.DataFrame:
    """Read DataFrame from store by key."""
    if not H5_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_hdf(H5_PATH, key=key)
    except (KeyError, FileNotFoundError):
        return pd.DataFrame()
