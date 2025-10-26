"""Data layer utilities exposing flatfile and REST helpers."""

from __future__ import annotations

from oami.utils import cache_manager
from oami.utils.cache_manager import (
    get_timeframe_str,
    make_option_contract_key,
    make_option_key,
    make_stock_key,
    parse_occ_ticker,
    read_hdf,
    read_option_coverage,
    update_option_coverage,
    write_hdf,
)

from . import flatfiles, restapi
from .flatfiles import (
    PolygonFlatfileClient,
    _download_stocks_flatfile,
    _filter_flatfile_frame,
    _filter_stock_flatfile_frame,
    _load_option_flatfile_for_date,
    _option_flatfile_path,
    _parse_option_ticker_metadata,
    _parse_stock_flatfile_date,
    _progress_iter,
    _subtract_years,
    get_option_flatfile_data,
    get_stock_flatfile_data,
)
from .restapi import (
    INTERVAL_CONFIG,
    SET,
    _cache_market_bounds,
    _ensure_market_datetime,
    _interval_delta,
    _load_or_fetch_contract_ohlcv,
    _load_or_fetch_contracts,
    _normalize_bucket_frame,
    _normalize_interval,
    _persist_option_buckets,
    _resolve_window,
    _timespan_delta,
    fetch_market_data,
    fetch_option_contracts,
    get_market_data,
    get_options_data,
)

# Re-export subprocess for compatibility with historical monkeypatches.
subprocess = flatfiles.subprocess

__all__ = [
    "PolygonFlatfileClient",
    "SET",
    "INTERVAL_CONFIG",
    "_cache_market_bounds",
    "_download_stocks_flatfile",
    "_ensure_market_datetime",
    "_filter_flatfile_frame",
    "_filter_stock_flatfile_frame",
    "_interval_delta",
    "_load_option_flatfile_for_date",
    "_load_or_fetch_contract_ohlcv",
    "_load_or_fetch_contracts",
    "_normalize_bucket_frame",
    "_normalize_interval",
    "_option_flatfile_path",
    "_parse_option_ticker_metadata",
    "_parse_stock_flatfile_date",
    "_persist_option_buckets",
    "_progress_iter",
    "_resolve_window",
    "_subtract_years",
    "_timespan_delta",
    "cache_manager",
    "fetch_market_data",
    "fetch_option_contracts",
    "get_market_data",
    "get_option_flatfile_data",
    "get_options_data",
    "get_stock_flatfile_data",
    "make_option_contract_key",
    "make_option_key",
    "make_stock_key",
    "parse_occ_ticker",
    "read_hdf",
    "read_option_coverage",
    "update_option_coverage",
    "write_hdf",
    "subprocess",
]

