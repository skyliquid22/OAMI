import re, logging, pandas as pd, requests
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple
from time import perf_counter
from tqdm import tqdm  # type: ignore
from polygon import RESTClient

from oami.utils.decorators import retry_request
from oami.utils import cache_manager
from oami.utils.cache_manager import (
    parse_occ_ticker,
    read_hdf,
    write_hdf,
    make_option_key,
    make_option_contract_key,
    get_timeframe_str,
    make_stock_key,
)

from .config import Settings
SET = Settings()

INTERVAL_CONFIG: dict[str, dict[str, object]] = {
    "1M": {"timespan": "minute", "multiplier": 1, "delta": pd.Timedelta(minutes=1)},
    "5M": {"timespan": "minute", "multiplier": 5, "delta": pd.Timedelta(minutes=5)},
    "15M": {"timespan": "minute", "multiplier": 15, "delta": pd.Timedelta(minutes=15)},
    "1H": {"timespan": "hour", "multiplier": 1, "delta": pd.Timedelta(hours=1)},
    "4H": {"timespan": "hour", "multiplier": 4, "delta": pd.Timedelta(hours=4)},
    "1D": {"timespan": "day", "multiplier": 1, "delta": pd.Timedelta(days=1)},
}


def _load_or_fetch_contracts(
    symbol: str,
    timeframe: str,
    from_: str,
    to: str,
    strike_min: float,
    strike_max: float,
) -> pd.DataFrame:
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

    end_year_raw = f"{end_date.year:04d}"
    try:
        end_month_raw = cache_manager.MONTHS[end_date.month - 1]
    except IndexError as exc:  # pragma: no cover - defensive guard
        raise ValueError("Invalid month extracted from 'to' parameter") from exc

    end_year = cache_manager._sanitize(end_year_raw)
    end_month = cache_manager._sanitize(end_month_raw)

    base_prefix = f"/options/{end_year}/{end_month}/{underlying}/{timeframe_normalized}"

    meta_columns = [
        "ticker",
        "underlying",
        "contract_type",
        "strike_price",
        "date",
        "expiration_date",
    ]

    ticker_pattern = re.compile(r"O:([A-Z]{1,6})(\d{2})(\d{2})(\d{2})([CP])(\d{8})")

    def _list_option_keys() -> list[str]:
        """List cached HDF keys for the derived prefix."""
        if not cache_manager.H5_PATH.exists():
            return []
        legacy_prefix = cache_manager._desanitize_key(base_prefix)
        with pd.HDFStore(cache_manager.H5_PATH, mode="r") as store:
            keys: list[str] = []
            for key in store.keys():
                if key == base_prefix or key.startswith(f"{base_prefix}/"):
                    keys.append(key)
                elif legacy_prefix != base_prefix and (key == legacy_prefix or key.startswith(f"{legacy_prefix}/")):
                    keys.append(key)
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
            df_bucket = read_hdf(key)
            if df_bucket.empty:
                legacy_key = cache_manager._desanitize_key(key)
                if legacy_key != key:
                    df_bucket = read_hdf(legacy_key)

            tickers: set[str] = set()
            if not df_bucket.empty and "ticker" in df_bucket.columns:
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

                contract_type = match.group(5)
                month_name = occ_meta["month"]
                try:
                    month_index = cache_manager.MONTHS.index(month_name) + 1
                except ValueError:
                    logging.warning("Unknown month in OCC metadata", extra={"month": month_name, "ticker": ticker})
                    continue

                expiration = date(occ_meta["year"], month_index, occ_meta["day"])
                if expiration < start_date or expiration > end_date:
                    continue

                records.append(
                    {
                        "ticker": ticker,
                        "underlying": occ_meta["underlying"],
                        "contract_type": "CALL" if contract_type == "C" else "PUT",
                        "strike_price": strike_price,
                        "date": expiration,
                        "expiration_date": expiration,
                    }
                )

        if not records:
            return pd.DataFrame(columns=meta_columns)

        frame = pd.DataFrame(records)
        frame = frame.drop_duplicates(subset="ticker", keep="last")
        frame = frame.sort_values(["date", "ticker"]).reset_index(drop=True)
        return frame

    def _compute_missing_dates(meta_df: pd.DataFrame) -> list[date]:
        """Determine which expiry dates are absent from the cached metadata."""
        requested_dates = [d.date() for d in pd.date_range(start_date, end_date, freq="D")]
        if meta_df.empty:
            return requested_dates
        available_dates = set(meta_df["date"].tolist())
        return [d for d in requested_dates if d not in available_dates]

    option_keys = _list_option_keys()
    cached_meta = _build_metadata(option_keys)

    if not cached_meta.empty:
        min_cached = cached_meta["date"].min()
        max_cached = cached_meta["date"].max()
    else:
        min_cached = max_cached = None

    missing_dates = _compute_missing_dates(cached_meta)
    if (
        not missing_dates
        and min_cached is not None
        and max_cached is not None
        and min_cached <= start_date
        and max_cached >= end_date
    ):
        return cached_meta

    missing_dates.sort()
    missing_ranges: list[tuple[date, date]] = []
    if missing_dates:
        range_start = missing_dates[0]
        range_end = missing_dates[0]
        for current in missing_dates[1:]:
            if current == range_end + timedelta(days=1):
                range_end = current
            else:
                missing_ranges.append((range_start, range_end))
                range_start = range_end = current
        missing_ranges.append((range_start, range_end))

    multiplier = int(interval_cfg["multiplier"])  # type: ignore[call-overload]
    timespan = str(interval_cfg["timespan"])  # type: ignore[call-overload]

    for range_start, range_end in missing_ranges:
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

        for ticker in map(str, tickers):
            try:
                # Ensure OHLCV history exists for the contract and date range; the
                # helper manages persistence once data is downloaded from Polygon.
                _load_or_fetch_contract_ohlcv(
                    ticker,
                    multiplier=multiplier,
                    timespan=timespan,
                    window_override=(from_pd, to_pd),
                )
            except Exception as exc:  # pragma: no cover - network/IO failure guard
                logging.error(
                    "Failed to backfill contract window",
                    extra={"ticker": ticker, "from": new_from_str, "to": new_to_str, "error": str(exc)},
                )
                continue

    updated_keys = _list_option_keys()
    updated_meta = _build_metadata(updated_keys)

    if updated_meta.empty:
        return cached_meta

    return updated_meta


def _load_or_fetch_contract_ohlcv(
    contract_ticker: str,
    multiplier: int = 1,
    timespan: str = "day",
    window_override: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """Retrieve OHLCV aggregates for a single OCC contract."""

    timeframe = get_timeframe_str(multiplier, timespan)
    bucket_key = make_option_key(contract_ticker, timeframe)
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

        timespan_lower = timespan.lower()
        if timespan_lower.startswith("day"):
            span = pd.Timedelta(days=1)
        elif timespan_lower.startswith("hour"):
            span = pd.Timedelta(hours=1)
        elif timespan_lower.startswith("minute"):
            span = pd.Timedelta(minutes=1)
        else:
            span = pd.Timedelta(days=1)

        start_ts = expiration - span * int(multiplier) * 90
        end_ts = expiration

    start_ts = pd.Timestamp(start_ts).tz_localize(None)
    end_ts = pd.Timestamp(end_ts).tz_localize(None)

    def _normalize_bucket(df: pd.DataFrame, default_ticker: str | None = None) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["ticker", "t", *price_columns])
        tmp = df.copy()
        if "t" not in tmp.columns:
            if isinstance(tmp.index, pd.DatetimeIndex):
                tmp = tmp.reset_index().rename(columns={tmp.index.name or "index": "t"})
            else:
                tmp["t"] = pd.to_datetime(tmp.get("t"), errors="coerce")
        tmp["t"] = pd.to_datetime(tmp["t"], errors="coerce").dt.tz_localize(None)
        if "ticker" not in tmp.columns:
            if default_ticker is None:
                raise ValueError("Option bucket missing ticker column")
            tmp["ticker"] = default_ticker
        for col in price_columns:
            if col not in tmp.columns:
                tmp[col] = pd.NA
        tmp = tmp.dropna(subset=["t", "ticker"])
        tmp = tmp.sort_values(["ticker", "t"]).drop_duplicates(subset=["ticker", "t"], keep="last").reset_index(drop=True)
        return tmp[["ticker", "t", *price_columns]]

    df_bucket = read_hdf(bucket_key)
    if df_bucket.empty:
        legacy_bucket = cache_manager._desanitize_key(bucket_key)
        if legacy_bucket != bucket_key:
            df_bucket = read_hdf(legacy_bucket)

    df_bucket = _normalize_bucket(df_bucket, default_ticker=contract_ticker)
    subset = df_bucket[df_bucket["ticker"] == contract_ticker]
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

        combined = pd.concat([df_bucket, df_new], ignore_index=True)
        combined = _normalize_bucket(combined)

        try:
            write_hdf(combined, bucket_key, extra_remove=[contract_key])
            logging.info("Cached contract OHLCV", extra={"contract": contract_ticker, "key": bucket_key})
        except Exception as exc:
            logging.error("Failed to cache contract OHLCV", extra={"key": bucket_key, "error": str(exc)})

        df_bucket = combined
        subset_window = df_bucket[df_bucket["ticker"] == contract_ticker].set_index("t")[price_columns].sort_index()

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
    cached_full = read_hdf(market_key) if use_cache else pd.DataFrame()
    if not cached_full.empty:
        cached_min, cached_max = _cache_market_bounds(cached_full, interval_key)
        if cached_min is not None and cached_max is not None:
            if start_ts >= cached_min and end_ts <= cached_max:
                return _filter_market_window(cached_full, start_ts, end_ts, interval_key)

    # Cache miss or partial coverage—request the missing portion directly from Polygon.
    with _progress([0], desc=f"Market data {symbol}", leave=False) as progress:
        df = fetch_market_data(
            symbol,
            start_ts.strftime("%Y-%m-%d") if interval_key == "1D" else start_ts.isoformat(),
            end_ts.strftime("%Y-%m-%d") if interval_key == "1D" else end_ts.isoformat(),
            interval=interval_key,
        )
        progress.update(1)
    if df.empty:
        logging.warning(f"No market data retrieved for {symbol}", extra={"interval": interval_key})
        return pd.DataFrame()

    combined = pd.concat([cached_full, df], ignore_index=True) if not cached_full.empty else df
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
    strike_bounds : float, default=0.05
        Buffer applied to the observed OHLC price range, widening the minimum
        and maximum strike thresholds by the supplied percentage.

    Returns
    -------
    pandas.DataFrame
        DataFrame of OCC contract metadata meeting the derived filters. Empty
        when no contracts are located or prerequisites fail.
    """

    try:
        interval_key = _normalize_interval(interval)
    except ValueError as exc:
        logging.error(str(exc))
        return pd.DataFrame()

    if strike_bounds < 0:
        logging.error(
            "strike_bounds must be non-negative",
            extra={"symbol": symbol, "strike_bounds": strike_bounds},
        )
        return pd.DataFrame()

    market_timer = perf_counter()
    market_df = get_market_data(symbol, start_date, end_date, interval=interval_key, use_cache=use_cache)
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

    start_ts, end_ts = _resolve_window(start_date, end_date, interval_key)
    step = _interval_delta(interval_key)
    forward_multiplier = max(0, look_forward)
    forward_delta = step * forward_multiplier

    from_ts = start_ts
    to_ts = end_ts + forward_delta

    from_str = from_ts.date().isoformat()
    to_str = to_ts.date().isoformat()

    try:
        contracts_timer = perf_counter()
        contracts_df = _load_or_fetch_contracts(
            symbol=symbol,
            timeframe=interval_key,
            from_=from_str,
            to=to_str,
            strike_min=strike_min,
            strike_max=strike_max,
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
        return contracts_df

    multiplier = int(INTERVAL_CONFIG[interval_key]["multiplier"])  # type: ignore[call-overload]
    timespan = str(INTERVAL_CONFIG[interval_key]["timespan"])  # type: ignore[call-overload]

    enriched_rows: list[pd.Series] = []
    records = contracts_df.to_dict("records")
    total_contracts = len(records)
    progress_iter = _progress(records, desc=f"Option OHLCV {symbol}", total=total_contracts, leave=False)
    enrich_total = 0.0
    for idx, record in enumerate(progress_iter, start=1):
        row = pd.Series(record)
        ticker = row.get("ticker")
        if not isinstance(ticker, str):
            progress_iter.set_postfix_str(f"{idx}/{total_contracts}")
            continue

        single_start = perf_counter()
        try:
            ohlcv_df = _load_or_fetch_contract_ohlcv(
                contract_ticker=ticker,
                multiplier=multiplier,
                timespan=timespan,
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            logging.error(
                "Failed to load contract OHLCV",
                extra={"ticker": ticker, "from": from_str, "to": to_str, "error": str(exc)},
            )
            progress_iter.set_postfix_str(f"{idx}/{total_contracts}")
            continue

        enrich_total += perf_counter() - single_start

        enriched = row.copy()
        enriched["ohlcv"] = ohlcv_df
        enriched_rows.append(enriched)
        progress_iter.set_postfix_str(f"{idx}/{total_contracts}")

    if not enriched_rows:
        logging.warning(
            "No contract OHLCV data available after enrichment",
            extra={"symbol": symbol, "from": from_str, "to": to_str},
        )
        return pd.DataFrame()

    logging.info(
        "_load_or_fetch_contract_ohlcv total duration",
        extra={
            "symbol": symbol,
            "contracts": len(enriched_rows),
            "seconds": round(enrich_total, 3),
        },
    )

    result = pd.DataFrame(enriched_rows)
    drop_cols = ["underlying", "contract_type", "strike_price", "date", "expiration_date"]
    if not result.empty:
        result = result.drop(columns=drop_cols, errors="ignore")
    return result


def _progress(iterable, *, desc: str, total: int | None = None, leave: bool = False):
    """Return a tqdm iterator configured for console/Jupyter friendly output."""
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        leave=leave,
        dynamic_ncols=True,
        miniters=1,
        ascii=True,
    )
