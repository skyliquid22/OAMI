"""
cache_manager.py (HDF5 Version 2)
---------------------------------
Hierarchical HDF5 cache for OAMI data.

Options data key format:
    /options/{expiry_year}/{expiry_month}/{underlying}/{timeframe}
Each bucket stores all contracts for the month/timeframe with a `ticker`
column identifying individual OCC symbols.

Stocks and features keep a simpler pattern:
    /stocks/{symbol}/{timeframe}
    /features/{symbol}/{timeframe}
"""

import re
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

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
    cleaned = s.replace(":", "_").replace("/", "_")
    if cleaned and cleaned[0].isdigit():
        cleaned = f"t_{cleaned}"
    return cleaned


def _desanitize_component(component: str) -> str:
    """Revert `_sanitize` prefix logic for compatibility with legacy keys."""
    if component.startswith("t_"):
        return component[2:]
    return component


def _desanitize_key(key: str) -> str:
    """Return a legacy-compatible version of the provided HDF5 key."""
    parts = key.split("/")
    desanitized = [_desanitize_component(part) for part in parts]
    # Preserve leading slash if present
    if key.startswith("/") and not desanitized[0]:
        return "/" + "/".join(desanitized[1:])
    return "/".join(desanitized)


def _unique(seq: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


# -------------------------------------------------------------------------
# Key builders
# -------------------------------------------------------------------------
def make_option_key(ticker: str, timeframe: str) -> str:
    """Build the HDF5 bucket key for an options contract timeframe."""
    info = parse_occ_ticker(ticker)
    return (
        f"/options/"
        f"{_sanitize(str(info['year']))}/"
        f"{_sanitize(info['month'])}/"
        f"{_sanitize(str(info['underlying']))}/"
        f"{_sanitize(timeframe)}"
    )


def make_option_contract_key(ticker: str, timeframe: str) -> str:
    """Build the legacy per-contract key (used for cleanup/compat)."""
    bucket = make_option_key(ticker, timeframe)
    return f"{bucket}/{_sanitize(ticker)}"


def make_stock_key(symbol: str, timeframe: str) -> str:
    """Build key for stock OHLCV data."""
    return f"/stocks/{_sanitize(symbol)}/{_sanitize(timeframe)}"


def make_feature_key(symbol: str, timeframe: str) -> str:
    """Build key for features."""
    return f"/features/{_sanitize(symbol)}/{_sanitize(timeframe)}"


# -------------------------------------------------------------------------
# IO wrappers
# -------------------------------------------------------------------------
def write_hdf(df: pd.DataFrame, key: str, extra_remove: list[str] | None = None):
    """Write DataFrame to HDF5 store."""
    if df.empty:
        logging.warning("Skipping empty write for %s", key)
        return
    H5_PATH.parent.mkdir(parents=True, exist_ok=True)
    extra_remove = extra_remove or []
    with pd.HDFStore(H5_PATH, mode="a", complevel=9, complib="blosc") as store:
        targets = [key, *_unique(extra_remove)]
        for target in targets:
            if target in store:
                store.remove(target)
            legacy_target = _desanitize_key(target)
            if legacy_target != target and legacy_target in store:
                store.remove(legacy_target)
        store.put(key, df, format="table")
    logging.info("Saved %d rows → %s", len(df), key)


def read_hdf(key: str) -> pd.DataFrame:
    """Read DataFrame from store by key."""
    if not H5_PATH.exists():
        return pd.DataFrame()
    try:
        with pd.HDFStore(H5_PATH, mode="r") as store:
            if key not in store:
                fallback = _desanitize_key(key)
                if fallback != key and fallback in store:
                    logging.info("Reading legacy HDF key", extra={"sanitized": key, "legacy": fallback})
                    return store.select(fallback)
                return pd.DataFrame()
            return store.select(key)
    except (TypeError, ValueError) as exc:
        logging.warning("Failed to read HDF key %s (%s). Removing corrupted entry.", key, exc)
        with pd.HDFStore(H5_PATH, mode="a") as store:
            if key in store:
                store.remove(key)
            fallback = _desanitize_key(key)
            if fallback != key and fallback in store:
                store.remove(fallback)
        return pd.DataFrame()
