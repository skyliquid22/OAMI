"""REST-based data acquisition for Polygon services.

This module collects helper functions that talk directly to Polygon's REST
APIs.  It includes resilient fetchers for market data, options metadata, and
fundamental filings, along with caching helpers that persist results to HDF5.
"""

import logging
import os
import re
import sys
from datetime import date, datetime, timedelta
from time import perf_counter
from typing import Dict, List, Tuple, Iterable, Sequence

import pandas as pd
import requests
from polygon import RESTClient

from oami.utils import cache_manager
from oami.utils.decorators import retry_request
from oami.utils.cache_manager import (
    parse_occ_ticker,
    read_hdf,
    write_hdf,
    make_option_key,
    make_option_contract_key,
    get_timeframe_str,
    make_stock_key,
    read_option_coverage,
    update_option_coverage,
)

from ..config import Settings
from ..logging_config import setup_json_logging
SET = Settings()

INTERVAL_CONFIG: dict[str, dict[str, object]] = {
    "1M": {"timespan": "minute", "multiplier": 1, "delta": pd.Timedelta(minutes=1)},
    "5M": {"timespan": "minute", "multiplier": 5, "delta": pd.Timedelta(minutes=5)},
    "15M": {"timespan": "minute", "multiplier": 15, "delta": pd.Timedelta(minutes=15)},
    "1H": {"timespan": "hour", "multiplier": 1, "delta": pd.Timedelta(hours=1)},
    "4H": {"timespan": "hour", "multiplier": 4, "delta": pd.Timedelta(hours=4)},
    "1D": {"timespan": "day", "multiplier": 1, "delta": pd.Timedelta(days=1)},
}


def _progress(iterable, *, desc: str, total: int | None = None):
    """Yield progress updates using whichever iterator the package exposes.

    The test-suite frequently monkeypatches ``oami.data_layer._progress_iter``.
    Rather than importing that symbol directly, this helper asks the package
    for the active implementation at runtime and falls back to the flatfile
    iterator when no override is registered.
    """

    progress_fn = None
    dl = sys.modules.get("oami.data_layer")
    if dl is not None:
        progress_fn = getattr(dl, "_progress_iter", None)
    if progress_fn is None:
        from .flatfiles import _progress_iter as progress_fn  # pylint: disable=import-outside-toplevel
    yield from progress_fn(iterable, desc=desc, total=total)


def _ensure_logging() -> None:
    """Ensure structured logging is configured before issuing HTTP requests."""

    root = logging.getLogger()
    has_file_handler = any(isinstance(handler, logging.FileHandler) for handler in root.handlers)
    if not has_file_handler:
        try:
            setup_json_logging()
        except Exception:  # pragma: no cover - fallback for unexpected initialization issues
            logging.basicConfig(level=logging.DEBUG)

def _load_or_fetch_contracts(
    symbol: str,
    timeframe: str,
    from_: str,
    to: str,
    strike_min: float,
    strike_max: float,
    bucket_cache: dict[str, pd.DataFrame] | None = None,
    bucket_removals: dict[str, set[str]] | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, set[str]]]:
    """Return cached option contracts and backfill cache gaps on demand.

    Parameters
    ----------
    symbol : str
        Underlying ticker symbol associated with the options contracts.
    timeframe : str
        Timeframe code (e.g. ``"1D"``) used to determine which cache partition
        to inspect.
    from_ : str
        Inclusive ISO date string used as the lower bound for contract expiry
        selection.
    to : str
        Inclusive ISO date string used as the upper bound for contract expiry
        selection.
    strike_min : float
        Minimum strike price (inclusive) to accept when scanning the cache.
    strike_max : float
        Maximum strike price (inclusive) to accept when scanning the cache.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing OCC tickers and parsed metadata columns. The frame
        is empty when no contracts meet the supplied filters.

    Raises
    ------
    ValueError
        Raised when supplied dates or strike bounds are invalid.
    """

    try:
        start_date = pd.to_datetime(from_).date()
        end_date = pd.to_datetime(to).date()
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive guard
        raise ValueError("Invalid ISO date supplied to _load_or_fetch_contracts") from exc

    if start_date > end_date:
        raise ValueError("Parameter 'from_' must be on or before 'to'")

    strike_floor = float(strike_min)
    strike_ceiling = float(strike_max)
    if strike_floor > strike_ceiling:
        raise ValueError("Parameter 'strike_min' must be <= 'strike_max'")

    interval_key = _normalize_interval(timeframe)
    interval_cfg = INTERVAL_CONFIG[interval_key]
    timeframe_normalized = cache_manager._sanitize(interval_key)
    underlying = cache_manager._sanitize(symbol.upper())

    if bucket_cache is None:
        bucket_cache = {}
    if bucket_removals is None:
        bucket_removals = {}
    if bucket_cache is None:
        bucket_cache = {}
    if bucket_removals is None:
        bucket_removals = {}

    meta_columns = [
        "ticker",
        "expiration_date",
        "strike_price",
        "contract_type",
    ]

    ticker_pattern = re.compile(r"O:([A-Z]{1,6})(\d{2})(\d{2})(\d{2})([CP])(\d{8})")

    def _month_prefixes() -> list[str]:
        """Construct sanitized cache prefixes for each month in [start, end]."""
        prefixes: list[str] = []
        current_year = start_date.year
        current_month = start_date.month
        while (current_year, current_month) <= (end_date.year, end_date.month):
            try:
                month_token = cache_manager.MONTHS[current_month - 1]
            except IndexError as exc:  # pragma: no cover - defensive guard
                raise ValueError("Invalid month generated for cache prefix") from exc
            year_part = cache_manager._sanitize(f"{current_year:04d}")
            month_part = cache_manager._sanitize(month_token)
            prefixes.append(f"/options/{year_part}/{month_part}/{underlying}/{timeframe_normalized}")
            if current_month == 12:
                current_year += 1
                current_month = 1
            else:
                current_month += 1
        return prefixes

    def _list_option_keys() -> list[str]:
        """List cached HDF keys for the derived monthly prefixes."""
        prefixes = _month_prefixes()
        if not prefixes:
            return []
        legacy_map = {prefix: cache_manager._desanitize_key(prefix) for prefix in prefixes}
        prefix_candidates = prefixes + [
            legacy for prefix, legacy in legacy_map.items() if legacy != prefix
        ]

        keys: list[str] = []
        if cache_manager.H5_PATH.exists():
            with pd.HDFStore(cache_manager.H5_PATH, mode="r") as store:
                for key in store.keys():
                    for candidate in prefix_candidates:
                        if key == candidate or key.startswith(f"{candidate}/"):
                            keys.append(key)
                            break

        if bucket_cache:
            for candidate in prefix_candidates:
                for mem_key in bucket_cache.keys():
                    if mem_key == candidate or mem_key.startswith(f"{candidate}/"):
                        keys.append(mem_key)

        return cache_manager._unique(keys)

    def _unsanitize_ticker(token: str) -> str:
        """Rehydrate an OCC ticker from the sanitized cache token."""
        if token.startswith("O_"):
            return "O:" + token[2:]
        return token

    def _build_metadata(keys: list[str]) -> pd.DataFrame:
        """Parse OCC metadata from cached option buckets."""
        records: list[dict[str, object]] = []
        seen: set[str] = set()

        for key in keys:
            if bucket_cache and key in bucket_cache:
                df_bucket = bucket_cache[key]
            else:
                df_bucket = read_hdf(key)
            if not isinstance(df_bucket, pd.DataFrame) or df_bucket.empty:
                legacy_key = cache_manager._desanitize_key(key)
                if legacy_key != key:
                    if bucket_cache and legacy_key in bucket_cache:
                        df_bucket = bucket_cache[legacy_key]
                    else:
                        df_bucket = read_hdf(legacy_key)
            if not isinstance(df_bucket, pd.DataFrame) or df_bucket.empty:
                continue

            tickers: set[str] = set()
            if isinstance(df_bucket, pd.DataFrame) and not df_bucket.empty and "ticker" in df_bucket.columns:
                tickers.update(
                    pd.Series(df_bucket["ticker"], dtype="string").dropna().astype(str).unique().tolist()
                )

            token = key.rsplit("/", 1)[-1]
            if not tickers and token.startswith("O"):
                tickers.add(_unsanitize_ticker(token))

            for ticker in tickers:
                if ticker in seen:
                    continue
                seen.add(ticker)
                try:
                    occ_meta = parse_occ_ticker(ticker)
                except ValueError as exc:
                    logging.warning("Skipping unparsable OCC ticker", extra={"key": key, "ticker": ticker, "error": str(exc)})
                    continue

                match = ticker_pattern.match(ticker)
                if not match:
                    logging.warning("Failed to regex-parse OCC ticker", extra={"ticker": ticker})
                    continue

                strike_price = int(match.group(6)) / 1000.0
                if not (strike_floor <= strike_price <= strike_ceiling):
                    continue

                contract_type_token = match.group(5)
                contract_type = "call" if contract_type_token.upper() == "C" else "put"
                month_name = occ_meta["month"]
                try:
                    month_index = cache_manager.MONTHS.index(month_name) + 1
                except ValueError:
                    logging.warning("Unknown month in OCC metadata", extra={"month": month_name, "ticker": ticker})
                    continue

                expiration = pd.Timestamp(year=occ_meta["year"], month=month_index, day=occ_meta["day"])
                expiration_date = expiration.date()
                if expiration_date < start_date or expiration_date > end_date:
                    continue

                records.append(
                    {
                        "ticker": ticker,
                        "expiration_date": expiration,
                        "strike_price": strike_price,
                        "contract_type": contract_type,
                    }
                )

        if not records:
            return pd.DataFrame(columns=meta_columns)

        frame = pd.DataFrame(records)
        frame = frame.drop_duplicates(subset="ticker", keep="last")
        if "expiration_date" in frame.columns:
            frame = frame.sort_values(["expiration_date", "ticker"]).reset_index(drop=True)
        for column in meta_columns:
            if column not in frame.columns:
                frame[column] = pd.NA
        return frame.reset_index(drop=True)[meta_columns]

    option_keys = _list_option_keys()
    cached_meta = _build_metadata(option_keys)

    coverage_start, coverage_end = read_option_coverage(symbol.upper(), timeframe)
    if cached_meta.empty:
        coverage_start = coverage_end = None
    request_start_date = pd.to_datetime(from_).date()
    request_end_date = pd.to_datetime(to).date()

    missing_ranges: list[tuple[date, date]] = []
    if coverage_start is None or coverage_end is None:
        missing_ranges.append((request_start_date, request_end_date))
    else:
        if request_start_date < coverage_start:
            missing_start = request_start_date
            missing_end = min(request_end_date, coverage_start - timedelta(days=1))
            if missing_start <= missing_end:
                missing_ranges.append((missing_start, missing_end))
        if request_end_date > coverage_end:
            missing_start = max(request_start_date, coverage_end + timedelta(days=1))
            if missing_start <= request_end_date:
                missing_ranges.append((missing_start, request_end_date))

    if missing_ranges:
        multiplier = int(interval_cfg["multiplier"])  # type: ignore[call-overload]
        timespan = str(interval_cfg["timespan"])  # type: ignore[call-overload]

        for range_start, range_end in _progress(
            missing_ranges, desc=f"Backfill ranges {symbol}", total=len(missing_ranges)
        ):
            new_from_str = range_start.isoformat()
            new_to_str = range_end.isoformat()
            logging.info(
                "Backfilling option contracts",
                extra={
                    "symbol": symbol,
                    "from": new_from_str,
                    "to": new_to_str,
                    "timeframe": timeframe_normalized,
                },
            )

            fetch_timer = perf_counter()
            tickers = fetch_option_contracts(
                underlying=underlying,
                exp_date_min=new_from_str,
                exp_date_max=new_to_str,
                strike_min=strike_floor,
                strike_max=strike_ceiling,
            )
            logging.info(
                "fetch_option_contracts duration",
                extra={
                    "symbol": symbol,
                    "from": new_from_str,
                    "to": new_to_str,
                    "seconds": round(perf_counter() - fetch_timer, 3),
                    "tickers": len(tickers),
                },
            )
            if not tickers:
                continue

            from_pd = pd.Timestamp(new_from_str).tz_localize(None)
            to_pd = pd.Timestamp(new_to_str).tz_localize(None)

            for ticker in _progress(list(map(str, tickers)), desc=f"Option OHLCV {symbol}", total=len(tickers)):
                try:
                    _load_or_fetch_contract_ohlcv(
                        ticker,
                        multiplier=multiplier,
                        timespan=timespan,
                        window_override=(from_pd, to_pd),
                        bucket_cache=bucket_cache,
                        bucket_removals=bucket_removals,
                    )
                except Exception as exc:  # pragma: no cover - network/IO failure guard
                    logging.error(
                        "Failed to backfill contract window",
                        extra={"ticker": ticker, "from": new_from_str, "to": new_to_str, "error": str(exc)},
                    )
                    continue
    else:
        logging.info(
            "Options coverage hit",
            extra={
                "symbol": symbol,
                "timeframe": timeframe,
                "from": request_start_date.isoformat(),
                "to": request_end_date.isoformat(),
            },
        )

    updated_keys = _list_option_keys()
    updated_meta = _build_metadata(updated_keys)

    if updated_meta.empty:
        return cached_meta, bucket_cache, bucket_removals

    exp_dates = pd.to_datetime(updated_meta["expiration_date"], errors="coerce").dropna()
    if not exp_dates.empty:
        cov_start = exp_dates.min().date()
        cov_end = exp_dates.max().date()
        if coverage_start is not None:
            cov_start = min(cov_start, coverage_start)
        if coverage_end is not None:
            cov_end = max(cov_end, coverage_end)
        update_option_coverage(symbol.upper(), timeframe, cov_start, cov_end)

    return updated_meta, bucket_cache, bucket_removals


def _load_or_fetch_contract_ohlcv(
    contract_ticker: str,
    multiplier: int = 1,
    timespan: str = "day",
    window_override: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    bucket_cache: dict[str, pd.DataFrame] | None = None,
    bucket_removals: dict[str, set[str]] | None = None,
) -> pd.DataFrame:
    """Retrieve OHLCV aggregates for a single OCC contract."""

    if bucket_cache is None:
        bucket_cache = {}
    if bucket_removals is None:
        bucket_removals = {}

    timeframe = get_timeframe_str(multiplier, timespan)
    bucket_key = make_option_key(contract_ticker, timeframe)
    legacy_bucket_key = cache_manager._desanitize_key(bucket_key)
    contract_key = make_option_contract_key(contract_ticker, timeframe)

    price_columns = ["o", "h", "l", "c", "v"]

    if window_override is not None:
        start_ts, end_ts = window_override
    else:
        occ_info = parse_occ_ticker(contract_ticker)
        try:
            month_index = cache_manager.MONTHS.index(occ_info["month"]) + 1
        except ValueError as exc:  # pragma: no cover - defensive guard
            raise ValueError(f"Unknown OCC month code in ticker {contract_ticker}") from exc

        expiration = pd.Timestamp(year=occ_info["year"], month=month_index, day=occ_info["day"]).tz_localize(None)

        span = _timespan_delta(timespan, multiplier)
        start_ts = expiration - span
        end_ts = expiration

    start_ts = pd.Timestamp(start_ts).tz_localize(None)
    end_ts = pd.Timestamp(end_ts).tz_localize(None)

    if bucket_key in bucket_cache:
        df_bucket = bucket_cache[bucket_key]
    else:
        df_bucket = read_hdf(bucket_key)
        if not isinstance(df_bucket, pd.DataFrame) or df_bucket.empty:
            legacy_bucket = cache_manager._desanitize_key(bucket_key)
            if legacy_bucket != bucket_key:
                df_bucket = read_hdf(legacy_bucket)
        if not isinstance(df_bucket, pd.DataFrame) or df_bucket.empty:
            df_bucket = pd.DataFrame(columns=["ticker", "t", *price_columns])
        df_bucket = _normalize_bucket_frame(df_bucket, default_ticker=None)
        bucket_cache[bucket_key] = df_bucket

    subset = bucket_cache[bucket_key][bucket_cache[bucket_key]["ticker"] == contract_ticker]
    subset_window = subset.set_index("t")[price_columns]
    subset_window = subset_window.sort_index()

    if not subset_window.empty:
        min_cached = subset_window.index.min()
        max_cached = subset_window.index.max()
        if min_cached <= start_ts and max_cached >= end_ts:
            return subset_window.loc[(subset_window.index >= start_ts) & (subset_window.index <= end_ts)]

    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{contract_ticker}/range/"
        f"{multiplier}/{timespan}/{start_ts.date().isoformat()}/{end_ts.date().isoformat()}"
    )
    params = {
        "adjusted": "true",
        "include_open_interest": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": SET.api_key,
    }

    try:
        resp = requests.get(url, params=params)
    except requests.exceptions.RequestException as exc:
        logging.error(
            "Option contract aggregate request failed",
            extra={"contract": contract_ticker, "error": str(exc)},
        )
        return subset_window.loc[(subset_window.index >= start_ts) & (subset_window.index <= end_ts)]

    if resp.status_code != 200:
        logging.error(
            "Option contract aggregate HTTP error",
            extra={"contract": contract_ticker, "status": resp.status_code, "body": resp.text[:500]},
        )
        return subset_window.loc[(subset_window.index >= start_ts) & (subset_window.index <= end_ts)]

    results = resp.json().get("results", [])
    if results:
        df_new = pd.DataFrame(results)
        if "t" not in df_new.columns:
            logging.error("Polygon agg response missing timestamp", extra={"contract": contract_ticker})
            return subset_window.loc[(subset_window.index >= start_ts) & (subset_window.index <= end_ts)]

        df_new["t"] = pd.to_datetime(df_new["t"], unit="ms", errors="coerce").dt.tz_localize(None)
        df_new = df_new.dropna(subset=["t"])
        for col in price_columns:
            if col not in df_new.columns:
                df_new[col] = pd.NA
        df_new["ticker"] = contract_ticker
        df_new = df_new[["ticker", "t", *price_columns]]

        existing = bucket_cache.get(bucket_key)
        if not isinstance(existing, pd.DataFrame) or existing.empty:
            combined = _normalize_bucket_frame(df_new)
        else:
            combined = _normalize_bucket_frame(pd.concat([existing, df_new], ignore_index=True))
        bucket_cache[bucket_key] = combined
        bucket_removals.setdefault(bucket_key, set()).add(contract_key)

        subset_window = combined[combined["ticker"] == contract_ticker].set_index("t")[price_columns].sort_index()

    return subset_window.loc[(subset_window.index >= start_ts) & (subset_window.index <= end_ts)]


@retry_request(max_retries=3)
def fetch_option_contracts(
    underlying: str,
    exp_date_min: str,
    exp_date_max: str,
    strike_min: float,
    strike_max: float,
    limit: int = 1000,
) -> list[str]:
    """Return OCC tickers from Polygon within the expiration and strike window.

    Parameters
    ----------
    underlying : str
        Ticker symbol for the contract's underlying security.
    exp_date_min : str
        Inclusive ISO date string representing the minimum expiration.
    exp_date_max : str
        Inclusive ISO date string representing the maximum expiration.
    strike_min : float
        Minimum strike price (inclusive) to request from Polygon.
    strike_max : float
        Maximum strike price (inclusive) to request from Polygon.
    limit : int, default=1000
        Page size to request from the Polygon API.

    Returns
    -------
    list of str
        OCC contract tickers satisfying the supplied filters. The list is empty
        if no contracts are returned.

    Raises
    ------
    ValueError
        Raised when expiration inputs or strike bounds are invalid.
    """

    try:
        start_date = pd.to_datetime(exp_date_min).date()
        end_date = pd.to_datetime(exp_date_max).date()
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive guard
        raise ValueError("Invalid ISO expiration date supplied to fetch_option_contracts") from exc

    if start_date > end_date:
        raise ValueError("Parameter 'exp_date_min' must be on or before 'exp_date_max'")

    strike_floor = float(strike_min)
    strike_ceiling = float(strike_max)
    if strike_floor > strike_ceiling:
        raise ValueError("Parameter 'strike_min' must be <= 'strike_max'")

    base_url = "https://api.polygon.io/v3/reference/options/contracts"
    params: Dict[str, object] = {
        "underlying_ticker": underlying.upper(),
        "as_of": start_date.isoformat(),
        "expiration_date.gte": start_date.isoformat(),
        "expiration_date.lte": end_date.isoformat(),
        "strike_price.gte": strike_floor,
        "strike_price.lte": strike_ceiling,
        "limit": limit,
        "apiKey": SET.api_key,
    }

    tickers: list[str] = []
    seen: set[str] = set()
    url = base_url
    first_page = True

    while url:
        # Walk Polygon's paginated endpoint, following ``next_url`` links when present.
        logging.info(
            "Fetching option contracts",
            extra={
                "underlying": underlying,
                "from": start_date.isoformat(),
                "to": end_date.isoformat(),
                "url": url if first_page else "next_url",
            },
        )
        try:
            resp = requests.get(url, params=params if first_page else None)
        except requests.exceptions.RequestException as exc:
            logging.error("Option contracts request failed", extra={"underlying": underlying, "error": str(exc)})
            break

        if resp.status_code != 200:
            logging.error(
                "Option contracts HTTP error",
                extra={"underlying": underlying, "status": resp.status_code, "body": resp.text[:500]},
            )
            break

        payload = resp.json()
        results = payload.get("results", [])
        for result in results:
            ticker = result.get("ticker")
            if not ticker or ticker in seen:
                continue
            seen.add(ticker)
            tickers.append(str(ticker))

        next_url = payload.get("next_url")
        if not next_url:
            break

        url = next_url
        params = None
        if "apiKey" not in url:
            connector = "&" if "?" in url else "?"
            url = f"{url}{connector}apiKey={SET.api_key}"
        first_page = False

    return tickers


def _normalize_interval(interval: str | None) -> str:
    """Normalize interval strings to the canonical cache key representation.

    Parameters
    ----------
    interval : str or None
        Interval string supplied by the caller. ``None`` defaults to the value
        configured in settings.

    Returns
    -------
    str
        Uppercase interval key understood by downstream helpers.

    Raises
    ------
    ValueError
        Raised when the interval is not in ``INTERVAL_CONFIG``.
    """
    default_interval = (SET.default_interval or "1D").upper()
    interval_key = (interval or default_interval).upper()
    if interval_key not in INTERVAL_CONFIG:
        raise ValueError(
            f"Unsupported interval '{interval_key}'. Expected one of {list(INTERVAL_CONFIG.keys())}"
        )
    return interval_key


def _interval_delta(interval_key: str) -> pd.Timedelta:
    """Return the pandas ``Timedelta`` configured for the given interval.

    Parameters
    ----------
    interval_key : str
        Canonical interval key such as ``"1D"`` or ``"15M"``.

    Returns
    -------
    pandas.Timedelta
        Timedelta representing the spacing between adjacent bars.
    """
    return INTERVAL_CONFIG[interval_key]["delta"]  # type: ignore[index]


def _timespan_delta(timespan: str, multiplier: int) -> pd.Timedelta:
    base = timespan.lower()
    if base.startswith("day"):
        unit = pd.Timedelta(days=1)
    elif base.startswith("hour"):
        unit = pd.Timedelta(hours=1)
    elif base.startswith("minute"):
        unit = pd.Timedelta(minutes=1)
    else:
        unit = pd.Timedelta(days=1)
    return unit * int(multiplier) * 90


def _normalize_bucket_frame(df: pd.DataFrame, default_ticker: str | None = None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["ticker", "t", "o", "h", "l", "c", "v"])
    tmp = df.copy()
    if "t" not in tmp.columns:
        if isinstance(tmp.index, pd.DatetimeIndex):
            tmp = tmp.reset_index().rename(columns={tmp.index.name or "index": "t"})
        else:
            tmp["t"] = pd.to_datetime(tmp.get("t"), errors="coerce")
    tmp["t"] = pd.to_datetime(tmp["t"], errors="coerce").dt.tz_localize(None)
    if "ticker" not in tmp.columns:
        if default_ticker is not None:
            tmp["ticker"] = default_ticker
        else:
            raise ValueError("Bucket frame missing ticker column")
    for col in ["o", "h", "l", "c", "v"]:
        if col not in tmp.columns:
            tmp[col] = pd.NA
    tmp = tmp.dropna(subset=["t", "ticker"])
    tmp = tmp.sort_values(["ticker", "t"]).drop_duplicates(subset=["ticker", "t"], keep="last").reset_index(drop=True)
    return tmp[["ticker", "t", "o", "h", "l", "c", "v"]]


def _ensure_market_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize timestamp column for market dataframes."""
    if df is None or df.empty:
        return pd.DataFrame()
    frame = df.copy()
    if "Timestamp" in frame.columns:
        frame["Timestamp"] = pd.to_datetime(frame["Timestamp"])
    elif "Date" in frame.columns:
        frame["Timestamp"] = pd.to_datetime(frame["Date"])
    else:
        raise ValueError("Market dataframe is missing both 'Timestamp' and 'Date' columns.")
    frame["Timestamp"] = frame["Timestamp"].dt.tz_localize(None)
    frame = frame.drop(columns=["Date"], errors="ignore")
    frame = frame.drop_duplicates(subset="Timestamp").sort_values("Timestamp").reset_index(drop=True)
    return frame


def _resolve_window(start: str | pd.Timestamp, end: str | pd.Timestamp, interval_key: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Normalize start/end bounds for a given interval.

    Parameters
    ----------
    start : str or pandas.Timestamp
        Inclusive start boundary of the requested window.
    end : str or pandas.Timestamp
        Inclusive end boundary of the requested window.
    interval_key : str
        Canonical interval key used to adjust day-level boundaries.

    Returns
    -------
    tuple of pandas.Timestamp
        Normalized ``(start, end)`` timestamps free of timezone information.

    Raises
    ------
    ValueError
        Raised when ``start`` occurs after ``end``.
    """
    start_ts = pd.to_datetime(start).tz_localize(None)
    end_ts = pd.to_datetime(end).tz_localize(None)
    if interval_key == "1D":
        start_ts = start_ts.normalize()
        end_ts = end_ts.normalize()
    if start_ts > end_ts:
        raise ValueError("start_date must be before end_date")
    return start_ts, end_ts


def _filter_market_window(df: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp, interval_key: str) -> pd.DataFrame:
    """Return the slice of market data between start/end for the provided interval."""
    if df is None or df.empty:
        return pd.DataFrame()
    frame = _ensure_market_datetime(df)
    ts = frame["Timestamp"]
    if interval_key == "1D":
        mask = (ts.dt.normalize() >= start_ts.normalize()) & (ts.dt.normalize() <= end_ts.normalize())
    else:
        mask = (ts >= start_ts) & (ts <= end_ts)
    return frame.loc[mask].reset_index(drop=True)


def _cache_market_bounds(df: pd.DataFrame, interval_key: str) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """Return the minimum and maximum timestamp stored for market data."""
    if df is None or df.empty:
        return None, None
    frame = _ensure_market_datetime(df)
    ts = frame["Timestamp"]
    if interval_key == "1D":
        ts = ts.dt.normalize()
    return ts.min(), ts.max()


@retry_request(max_retries=3)
def fetch_market_data(symbol: str, start: str, end: str, interval: str | None = None) -> pd.DataFrame:
    """
    Fetch market data from the Polygon API for a given interval.

    Parameters
    ----------
    symbol : str
        The ticker symbol, e.g., "SPY".
    start : str
        Start timestamp in ISO format.
    end : str
        End timestamp in ISO format.
    interval : str, optional
        Interval key (1M, 5M, 15M, 1H, 4H, 1D). Defaults to settings default.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns Timestamp, Date, Open, High, Low, Close, Volume.
    """
    interval_key = _normalize_interval(interval)
    config = INTERVAL_CONFIG[interval_key]

    try:
        client = RESTClient(SET.api_key)
        bars = client.list_aggs(
            ticker=symbol,
            multiplier=config["multiplier"],  # type: ignore[arg-type]
            timespan=config["timespan"],       # type: ignore[arg-type]
            from_=start,
            to=end,
            limit=50000,
        )

        data = []
        for bar in bars:
            ts = getattr(bar, "timestamp", None)
            if ts is None:
                continue
            try:
                timestamp = datetime.fromtimestamp(ts / 1000)
            except Exception as inner_e:
                logging.warning("Skipping bar due to timestamp parse error", extra={"symbol": symbol, "error": str(inner_e)})
                continue

            data.append({
                "Timestamp": timestamp,
                "Open": bar.open,
                "High": bar.high,
                "Low": bar.low,
                "Close": bar.close,
                "Volume": bar.volume,
            })

        if not data:
            logging.warning("No bars returned from Polygon API", extra={"symbol": symbol, "interval": interval_key})
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df = _ensure_market_datetime(df)
        logging.info(
            "Fetched %s bars for %s",
            len(df),
            symbol,
            extra={"interval": interval_key, "from": start, "to": end},
        )
        return df

    except Exception as e:
        logging.error(f"Error in fetch_market_data for {symbol}: {e}", exc_info=True)
        return pd.DataFrame()


def get_market_data(symbol: str, start: str, end: str, interval: str | None = None, use_cache: bool = True) -> pd.DataFrame:
    """Retrieve market data for a symbol, optionally using cached values.

    Parameters
    ----------
    symbol : str
        Underlying ticker to query.
    start : str
        Inclusive ISO timestamp (or date) marking the beginning of the window.
    end : str
        Inclusive ISO timestamp (or date) marking the end of the window.
    interval : str, optional
        Interval key (``"1M"``, ``"5M"``, ``"1D"``, etc.). ``None`` falls back to
        the default interval in settings.
    use_cache : bool, default=True
        When ``True`` the function will attempt to reuse and extend locally
        cached data.

    Returns
    -------
    pandas.DataFrame
        Windowed OHLCV data satisfying the request. An empty frame is returned
        if data cannot be retrieved.
    """
    try:
        interval_key = _normalize_interval(interval)
    except ValueError as exc:
        logging.error(str(exc))
        return pd.DataFrame()

    start_ts, end_ts = _resolve_window(start, end, interval_key)

    market_key = make_stock_key(symbol, interval_key)
    cached_full = read_hdf(market_key) if use_cache else None
    if isinstance(cached_full, pd.DataFrame) and not cached_full.empty:
        cached_min, cached_max = _cache_market_bounds(cached_full, interval_key)
        if cached_min is not None and cached_max is not None:
            if start_ts >= cached_min and end_ts <= cached_max:
                return _filter_market_window(cached_full, start_ts, end_ts, interval_key)

    # Cache miss or partial coverage—request the missing portion directly from Polygon.
    print(f"Fetching market data {symbol}...", end="\r")
    df = fetch_market_data(
        symbol,
        start_ts.strftime("%Y-%m-%d") if interval_key == "1D" else start_ts.isoformat(),
        end_ts.strftime("%Y-%m-%d") if interval_key == "1D" else end_ts.isoformat(),
        interval=interval_key,
    )
    print(f"Fetching market data {symbol}...done")
    if df.empty:
        logging.warning(f"No market data retrieved for {symbol}", extra={"interval": interval_key})
        return pd.DataFrame()

    if isinstance(cached_full, pd.DataFrame) and not cached_full.empty:
        combined = pd.concat([cached_full, df], ignore_index=True)
    else:
        combined = df
    combined = _ensure_market_datetime(combined)
    write_hdf(combined, market_key)
    window = _filter_market_window(combined, start_ts, end_ts, interval_key)
    return window


def get_options_data(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str | None = None,
    use_cache: bool = True,
    look_forward: int = 30,
    look_backward: int = 30,
    strike_bounds: float = 0.05,
) -> pd.DataFrame:
    """Return OCC contract metadata for the requested window.

    Parameters
    ----------
    symbol : str
        Underlying ticker used to locate both market data and option contracts.
    start_date : str
        Inclusive ISO date representing the beginning of the analysis window.
    end_date : str
        Inclusive ISO date representing the end of the analysis window.
    interval : str, optional
        Interval key passed through to market-data helpers. ``None`` defaults to
        the configured interval.
    use_cache : bool, default=True
        When ``True`` both the market data lookup and contract loader will
        consult cached data before calling external services.
    look_forward : int, default=30
        Number of interval units to project forward when choosing option
        expirations. Helps ensure contracts expiring shortly after the window
        remain available downstream.
    look_backward : int, default=30
        Number of interval units to project backward from ``start_date`` when
        deriving contract filters. Ensures contracts expiring shortly before
        the window stay available for feature engineering.
    strike_bounds : float, default=0.05
        Buffer applied to the observed OHLC price range, widening the minimum
        and maximum strike thresholds by the supplied percentage.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by OCC ticker with columns ``expiration_date``,
        ``strike_price``, ``contract_type``, and ``ohlcv`` (per-contract OHLCV
        DataFrame). Empty when no contracts are located or prerequisites fail.
    """

    try:
        interval_key = _normalize_interval(interval)
    except ValueError as exc:
        logging.error(str(exc))
        return pd.DataFrame()

    dl = sys.modules.get("oami.data_layer")
    market_fn = getattr(dl, "get_market_data", get_market_data) if dl else get_market_data
    contracts_fn = getattr(dl, "_load_or_fetch_contracts", _load_or_fetch_contracts) if dl else _load_or_fetch_contracts
    persist_fn = getattr(dl, "_persist_option_buckets", _persist_option_buckets) if dl else _persist_option_buckets

    if strike_bounds < 0:
        logging.error(
            "strike_bounds must be non-negative",
            extra={"symbol": symbol, "strike_bounds": strike_bounds},
        )
        return pd.DataFrame()

    result_columns = ["expiration_date", "strike_price", "contract_type", "ohlcv"]

    start_ts, end_ts = _resolve_window(start_date, end_date, interval_key)
    step = _interval_delta(interval_key)
    forward_multiplier = max(0, look_forward)
    backward_multiplier = max(0, look_backward)
    forward_delta = step * forward_multiplier
    backward_delta = step * backward_multiplier

    market_start_ts = start_ts - backward_delta
    market_end_ts = end_ts + forward_delta

    def _format_market_input(ts: pd.Timestamp) -> str:
        return ts.date().isoformat() if interval_key == "1D" else ts.isoformat()

    market_timer = perf_counter()
    market_df = market_fn(
        symbol,
        _format_market_input(market_start_ts),
        _format_market_input(market_end_ts),
        interval=interval_key,
        use_cache=use_cache,
    )
    logging.info(
        "get_market_data duration",
        extra={
            "symbol": symbol,
            "seconds": round(perf_counter() - market_timer, 3),
        },
    )
    if market_df.empty:
        logging.warning(
            "Unable to compute strike bounds due to missing market data",
            extra={"symbol": symbol, "start": start_date, "end": end_date},
        )
        return pd.DataFrame()

    # Remove raw volume columns—only price levels contribute to strike bounds.
    drop_cols = [col for col in ("v", "Volume") if col in market_df.columns]
    if drop_cols:
        market_df = market_df.drop(columns=drop_cols)

    # Accept typical OHLC naming variants produced by different data sources.
    price_columns = [
        col
        for col in ("Open", "High", "Low", "Close", "open", "high", "low", "close", "o", "h", "l", "c")
        if col in market_df.columns
    ]
    if not price_columns:
        logging.error(
            "Market data lacks OHLC columns required for strike bounds",
            extra={"symbol": symbol, "columns": list(market_df.columns)},
        )
        return pd.DataFrame()

    prices = market_df[price_columns].apply(pd.to_numeric, errors="coerce")
    strike_min_val = prices.min().min()
    strike_max_val = prices.max().max()

    if pd.isna(strike_min_val) or pd.isna(strike_max_val):
        logging.warning(
            "Failed to derive strike bounds from market data",
            extra={"symbol": symbol, "start": start_date, "end": end_date},
        )
        return pd.DataFrame()

    strike_min = float(max(0.0, strike_min_val * (1 - strike_bounds)))
    strike_max = float(strike_max_val * (1 + strike_bounds))

    if strike_min > strike_max:
        logging.warning(
            "Computed strike bounds are invalid",
            extra={
                "symbol": symbol,
                "strike_min": strike_min,
                "strike_max": strike_max,
                "strike_bounds": strike_bounds,
            },
        )
        return pd.DataFrame()

    from_ts = start_ts - backward_delta
    to_ts = end_ts + forward_delta

    from_str = from_ts.date().isoformat()
    to_str = to_ts.date().isoformat()

    bucket_cache: dict[str, pd.DataFrame] = {}
    bucket_removals: dict[str, set[str]] = {}

    try:
        contracts_timer = perf_counter()
        contracts_df, bucket_cache, bucket_removals = contracts_fn(
            symbol=symbol,
            timeframe=interval_key,
            from_=from_str,
            to=to_str,
            strike_min=strike_min,
            strike_max=strike_max,
            bucket_cache=bucket_cache,
            bucket_removals=bucket_removals,
        )
        logging.info(
            "_load_or_fetch_contracts duration",
            extra={
                "symbol": symbol,
                "from": from_str,
                "to": to_str,
                "seconds": round(perf_counter() - contracts_timer, 3),
                "contracts": len(contracts_df),
            },
        )
    except ValueError as exc:
        logging.error(
            "Failed to load option contracts",
            extra={"symbol": symbol, "from": from_str, "to": to_str, "error": str(exc)},
        )
        persist_fn(bucket_cache, bucket_removals)
        return pd.DataFrame()

    if contracts_df.empty:
        logging.info(
            "No option contracts returned",
            extra={
                "symbol": symbol,
                "from": from_str,
                "to": to_str,
                "look_forward": forward_multiplier,
            },
        )
        persist_fn(bucket_cache, bucket_removals)
        return pd.DataFrame(columns=result_columns)

    multiplier = int(INTERVAL_CONFIG[interval_key]["multiplier"])  # type: ignore[call-overload]
    timespan = str(INTERVAL_CONFIG[interval_key]["timespan"])  # type: ignore[call-overload]

    grouped: dict[str, list[str]] = {}
    for ticker in contracts_df["ticker"].astype(str):
        try:
            info = parse_occ_ticker(ticker)
        except ValueError:
            continue
        bucket = make_option_key(ticker, timeframe=interval_key)
        grouped.setdefault(bucket, []).append(ticker)

    ohlcv_map: dict[str, pd.DataFrame] = {}
    enrich_total = 0.0

    for bucket_key, tickers_in_bucket in grouped.items():
        legacy_bucket_key = cache_manager._desanitize_key(bucket_key)
        if bucket_key in bucket_cache:
            df_bucket = bucket_cache[bucket_key]
        else:
            df_bucket = read_hdf(bucket_key)
            if not isinstance(df_bucket, pd.DataFrame) or df_bucket.empty:
                if legacy_bucket_key != bucket_key:
                    df_bucket = read_hdf(legacy_bucket_key)
        df_bucket = _normalize_bucket_frame(df_bucket)
        bucket_cache[bucket_key] = df_bucket

        for ticker in _progress(tickers_in_bucket, desc=f"Option OHLCV {symbol} {bucket_key}", total=len(tickers_in_bucket)):
            subset = df_bucket[df_bucket["ticker"] == ticker]
            if subset.empty:
                single_start = perf_counter()
                from_pd = pd.Timestamp(from_str).tz_localize(None)
                to_pd = pd.Timestamp(to_str).tz_localize(None)
                ohlcv_df = _load_or_fetch_contract_ohlcv(
                    contract_ticker=ticker,
                    multiplier=multiplier,
                    timespan=timespan,
                    window_override=(from_pd, to_pd),
                    bucket_cache=bucket_cache,
                    bucket_removals=bucket_removals,
                )
                enrich_total += perf_counter() - single_start
                df_bucket = bucket_cache[bucket_key]
                subset = df_bucket[df_bucket["ticker"] == ticker]
            else:
                ohlcv_df = subset.set_index("t")[["o", "h", "l", "c", "v"]].sort_index()
            ohlcv_map[ticker] = ohlcv_df

    if not ohlcv_map:
        logging.warning(
            "No contract OHLCV data available after enrichment",
            extra={"symbol": symbol, "from": from_str, "to": to_str},
        )
        persist_fn(bucket_cache, bucket_removals)
        return pd.DataFrame(columns=result_columns)

    logging.info(
        "_load_or_fetch_contract_ohlcv total duration",
        extra={
            "symbol": symbol,
            "contracts": len(ohlcv_map),
            "seconds": round(enrich_total, 3),
        },
    )

    persist_fn(bucket_cache, bucket_removals)

    ohlcv_series = pd.Series(ohlcv_map, name="ohlcv")
    metadata = contracts_df.set_index("ticker")
    for meta_col in ("expiration_date", "strike_price", "contract_type"):
        if meta_col not in metadata.columns:
            metadata[meta_col] = pd.NA
    desired_meta = metadata.reindex(ohlcv_series.index)[["expiration_date", "strike_price", "contract_type"]]
    result = desired_meta.copy()
    result["ohlcv"] = ohlcv_series
    result.index.name = "ticker"
    return result


class FundamentalDataClient:
    """Retrieve Polygon fundamentals over a filing-date window.

    Parameters
    ----------
    ticker : str
        The instrument symbol (equity ticker) to request.
    start, end : str
        Inclusive ISO-8601 dates used to bound ``filing_date``.
    timeframe : str, optional
        Polygon reporting cadence. Valid options are ``\"annual\"`` (default),
        ``\"quarterly\"``, and ``\"ttm\"``.
    use_cache : bool, default False
        When ``True`` the client attempts to read/write cached responses
        under ``/fundamentals/{ticker}/{timeframe}`` in the shared HDF store.
    """

    VALID_TIMEFRAMES = {"annual", "quarterly", "ttm"}
    DEFAULT_TIMEFRAME = "annual"
    DATE_COLUMNS = (
        "filing_date",
        "calendar_date",
        "financials.filing_date",
        "financials.calendar_date",
        "period_end_date",
        "fiscal_period",
        "fiscal_end_date",
        "date",
    )

    def __init__(
        self,
        ticker: str,
        start: str,
        end: str,
        timeframe: str | None = None,
        use_cache: bool = False,
    ) -> None:
        try:
            start_dt = pd.to_datetime(start).date()
            end_dt = pd.to_datetime(end).date()
        except (TypeError, ValueError) as exc:
            raise ValueError("start and end must be ISO-8601 compatible dates") from exc
        if start_dt > end_dt:
            raise ValueError("start must be on or before end")

        timeframe_normalized = (timeframe or self.DEFAULT_TIMEFRAME).lower()
        if timeframe_normalized not in self.VALID_TIMEFRAMES:
            raise ValueError(f"timeframe must be one of {sorted(self.VALID_TIMEFRAMES)}")

        self.ticker = ticker.upper().strip()
        self.start_date = start_dt
        self.end_date = end_dt
        self.timeframe = timeframe_normalized
        self._key = f"/fundamentals/{self.ticker}/{self.timeframe}"
        self.use_cache = bool(use_cache)

    def load(self, json: bool = True):
        """Return fundamentals for the configured window.

        Parameters
        ----------
        json : bool, default True
            When ``True`` return a list of JSON-compatible dictionaries.
            When ``False`` return a pandas DataFrame.

        Returns
        -------
        list[dict] | pandas.DataFrame
            Fundamental filings bounded by ``start``/``end``.
        """

        frame: pd.DataFrame | None = None
        if self.use_cache:
            frame = self._load_from_cache()
        if frame is None or frame.empty:
            frame = self._fetch_from_polygon(store=self.use_cache)
        if frame is None or frame.empty:
            return [] if json else pd.DataFrame()
        masked = self._mask_by_window(frame)
        return masked.to_dict(orient="records") if json else masked

    def _load_from_cache(self) -> pd.DataFrame | None:
        try:
            frame = read_hdf(self._key)
        except (FileNotFoundError, KeyError, OSError, ValueError):
            return None
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return None
        return frame

    def _fetch_from_polygon(self, *, store: bool) -> pd.DataFrame | None:
        """Request fundamentals from Polygon, optionally persisting the result."""
        api_key = os.getenv("POLYGON_API_KEY") or SET.api_key
        if not api_key or api_key == "YOUR_KEY_HERE":
            logging.warning(
                "Skipping fundamentals fetch; Polygon API key missing.",
                extra={"ticker": self.ticker},
            )
            return None

        base_url = "https://api.polygon.io/vX/reference/financials"
        params = {
            "ticker": self.ticker,
            "filing_date.gte": self.start_date.isoformat(),
            "filing_date.lte": self.end_date.isoformat(),
            "timeframe": self.timeframe,
            "order": "asc",
            "sort": "filing_date",
            "limit": 100,
            "apiKey": api_key,
        }
        logging.info(
            "Fundamentals request prepared",
            extra={"url": base_url, "params": params},
        )

        frames: list[pd.DataFrame] = []
        url = base_url
        first_page = True

        while url:
            _ensure_logging()
            logging.info(
                "Fetching fundamentals",
                extra={
                    "ticker": self.ticker,
                    "timeframe": self.timeframe,
                    "url": url if first_page else "next_url",
                },
            )
            try:
                resp = requests.get(url, params=params if first_page else None)
            except requests.exceptions.RequestException as exc:
                logging.error(
                    "Fundamentals request failed",
                    extra={"ticker": self.ticker, "error": str(exc)},
                )
                break

            logging.debug(
                "Fundamentals response",
                extra={
                    "ticker": self.ticker,
                    "status": resp.status_code,
                    "body": resp.text,
                    "headers": dict(resp.headers),
                },
            )

            if resp.status_code != 200:
                logging.error(
                    "Fundamentals HTTP error",
                    extra={"ticker": self.ticker, "status": resp.status_code, "body": resp.text[:500]},
                )
                break

            try:
                payload = resp.json()
            except ValueError as exc:
                logging.error(
                    "Failed to decode fundamentals JSON",
                    extra={"ticker": self.ticker, "error": str(exc)},
                )
                break
            results = payload.get("results", [])
            if results:
                frames.append(pd.json_normalize(results))

            next_url = payload.get("next_url")
            if not next_url:
                break

            url = next_url
            params = None
            if "apiKey" not in url:
                connector = "&" if "?" in url else "?"
                url = f"{url}{connector}apiKey={SET.api_key}"
            first_page = False

        if not frames:
            return None

        combined = pd.concat(frames, ignore_index=True)
        if store:
            try:
                write_hdf(combined, self._key)
            except Exception as exc:  # pragma: no cover - cache write guard
                logging.error(
                    "Failed to cache fundamentals",
                    extra={"ticker": self.ticker, "timeframe": self.timeframe, "error": str(exc)},
                )

        return combined

    def _mask_by_window(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Filter the response to rows whose filing dates fall inside the window."""
        date_col = self._detect_date_column(frame)
        if date_col is None:
            return frame
        filtered = frame.copy()
        filtered[date_col] = pd.to_datetime(filtered[date_col], errors="coerce", format="ISO8601")
        filtered = filtered.dropna(subset=[date_col])
        mask = (filtered[date_col].dt.date >= self.start_date) & (filtered[date_col].dt.date <= self.end_date)
        return filtered.loc[mask].reset_index(drop=True)

    def _detect_date_column(self, frame: pd.DataFrame) -> str | None:
        for candidate in self.DATE_COLUMNS:
            if candidate in frame.columns:
                return candidate
        return None


def _persist_option_buckets(
    bucket_cache: dict[str, pd.DataFrame],
    bucket_removals: dict[str, set[str]],
) -> None:
    if not bucket_cache:
        return
    for key, df in bucket_cache.items():
        try:
            remove_keys = list(bucket_removals.get(key, set()))
            write_hdf(df, key, extra_remove=remove_keys)
            logging.info(
                "Persisted option bucket",
                extra={"key": key, "rows": len(df)},
            )
        except Exception as exc:
            logging.error("Failed to persist option bucket", extra={"key": key, "error": str(exc)})
def fetch_snapshot_tickers(
    *,
    limit: int | None = None,
    sort_by: str = "usd_volume",
    descending: bool = True,
) -> pd.DataFrame:
    """Fetch Polygon's full market snapshot and return summary metrics per ticker."""
    url = "https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers"
    params = {"apiKey": SET.api_key}
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:  # pragma: no cover
        logging.error("Failed to fetch snapshot tickers", extra={"error": str(exc)})
        raise RuntimeError(f"Snapshot request failed: {exc}") from exc

    tickers = payload.get("tickers") or payload.get("results") or []
    rows: list[dict[str, object]] = []
    for entry in tickers:
        last_trade = entry.get("lastTrade") or entry.get("last_trade") or {}
        last_quote = entry.get("lastQuote") or entry.get("last_quote") or {}
        day = entry.get("day") or entry.get("prevDay") or {}

        close_price = (
            last_trade.get("p")
            or last_quote.get("p")
            or last_trade.get("price")
            or last_quote.get("price")
            or day.get("close")
            or day.get("c")
        )
        volume_value = (
            day.get("volume")
            or day.get("v")
            or last_trade.get("s")
            or last_quote.get("s")
        )
        rows.append(
            {
                "ticker": entry.get("ticker"),
                "close": close_price,
                "volume": volume_value,
                "change_percent": day.get("changePercent") or day.get("change_percent"),
                "open": day.get("open") or day.get("o"),
                "high": day.get("high") or day.get("h"),
                "low": day.get("low") or day.get("l"),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df["usd_volume"] = df["close"] * df["volume"]
    df = df.dropna(subset=["usd_volume", "ticker"])
    df = df.sort_values(sort_by, ascending=not descending)
    if limit is not None:
        df = df.head(limit)
    return df.reset_index(drop=True)


def is_common_stock_ticker(ticker: str) -> bool:
    """Return True when Polygon classifies the ticker as common stock."""
    if not ticker:
        return False
    url = f"https://api.polygon.io/v3/reference/tickers/{ticker}"
    params = {"apiKey": SET.api_key}
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:  # pragma: no cover
        logging.error("Failed to fetch ticker reference", extra={"ticker": ticker, "error": str(exc)})
        raise RuntimeError(f"Ticker reference request failed for {ticker}: {exc}") from exc

    results = payload.get("results") or {}
    ticker_type = results.get("type")
    if isinstance(ticker_type, str):
        ticker_type = ticker_type.strip().lower()
        return ticker_type == "cs"
    return False


def filter_common_stock_tickers(tickers: Sequence[str]) -> list[str]:
    """Return tickers classified as common stock via Polygon reference data."""
    equities: list[str] = []
    for ticker in tickers:
        if not ticker:
            continue
        try:
            if is_common_stock_ticker(ticker):
                equities.append(ticker)
        except Exception as exc:  # pragma: no cover
            logging.warning("Skipping ticker due to reference lookup failure", extra={"ticker": ticker, "error": str(exc)})
    return equities
