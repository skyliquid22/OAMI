import re, logging, pandas as pd, requests
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple
try:
    from tqdm.auto import tqdm  # type: ignore
except ImportError:  # pragma: no cover
    from tqdm import tqdm  # type: ignore
from polygon import RESTClient

from oami.utils.decorators import retry_request
from oami.utils import cache_manager
from oami.utils.cache_manager import (
    parse_occ_ticker,
    read_hdf,
    write_hdf,
    make_option_key,
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
    timeframe_normalized = interval_key
    underlying = symbol.upper()

    end_year = f"{end_date.year:04d}"
    try:
        end_month = cache_manager.MONTHS[end_date.month - 1]
    except IndexError as exc:  # pragma: no cover - defensive guard
        raise ValueError("Invalid month extracted from 'to' parameter") from exc

    base_prefix = f"/options/{end_year}/{end_month}/{underlying}/{timeframe_normalized}/"

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
        with pd.HDFStore(cache_manager.H5_PATH, mode="r") as store:
            return [key for key in store.keys() if key.startswith(base_prefix)]

    def _unsanitize_ticker(token: str) -> str:
        """Rehydrate an OCC ticker from the sanitized cache token."""
        if token.startswith("O_"):
            return "O:" + token[2:]
        return token

    def _build_metadata(keys: list[str]) -> pd.DataFrame:
        """Parse OCC metadata from cached key names without reading payloads."""
        records: list[dict[str, object]] = []
        for key in keys:
            token = key.rsplit("/", 1)[-1]
            ticker = _unsanitize_ticker(token)

            try:
                occ_meta = parse_occ_ticker(ticker)
            except ValueError as exc:
                logging.warning("Skipping unparsable OCC ticker from key", extra={"key": key, "error": str(exc)})
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
                    "contract_type": contract_type,
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

        tickers = fetch_option_contracts(
            underlying=underlying,
            exp_date_min=new_from_str,
            exp_date_max=new_to_str,
            strike_min=strike_floor,
            strike_max=strike_ceiling,
        )
        if not tickers:
            continue

        for ticker in map(str, tickers):
            start_ts = pd.Timestamp(new_from_str)
            end_ts = pd.Timestamp(new_to_str)

            try:
                # Ensure OHLCV history exists for the contract and date range; the
                # helper manages persistence once data is downloaded from Polygon.
                _load_or_fetch_contract_ohlcv(
                    ticker,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    multiplier=multiplier,
                    timespan=timespan,
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
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    multiplier: int = 1,
    timespan: str = "day",
) -> pd.DataFrame:
    """Retrieve OHLCV aggregates for a single OCC contract.

    Parameters
    ----------
    contract_ticker : str
        OCC-formatted option contract ticker (e.g. ``"O:SPY250117C00470000"``).
    start_ts : pandas.Timestamp
        Inclusive lower bound for the aggregate time window.
    end_ts : pandas.Timestamp
        Inclusive upper bound for the aggregate time window.
    multiplier : int, default=1
        Multiplier for the Polygon aggregate query.
    timespan : str, default="day"
        Timespan string understood by Polygon (e.g. ``"day"`` or ``"minute"``).

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by timestamp ``t`` with open, high, low, close, and
        volume columns. Empty when neither cache nor API returned data.
    """
    timeframe = get_timeframe_str(multiplier, timespan)
    key = make_option_key(contract_ticker, timeframe)

    price_columns = ["o", "h", "l", "c", "v"]

    df_cached = read_hdf(key)
    if not df_cached.empty:
        df_cached = df_cached.copy()
        if "t" in df_cached.columns and not isinstance(df_cached.index, pd.DatetimeIndex):
            df_cached.index = pd.to_datetime(df_cached.pop("t"), errors="coerce")
        elif not isinstance(df_cached.index, pd.DatetimeIndex):
            df_cached.index = pd.to_datetime(df_cached.index, errors="coerce")

        df_cached = df_cached[~df_cached.index.isna()]
        df_cached.index = df_cached.index.tz_localize(None)
        df_cached.index.name = "t"
        for col in price_columns:
            if col not in df_cached.columns:
                df_cached[col] = pd.NA
        df_cached = df_cached[price_columns]

        if not df_cached.empty:
            min_cached = df_cached.index.min()
            max_cached = df_cached.index.max()
            if min_cached <= start_ts and max_cached >= end_ts:
                return df_cached.loc[(df_cached.index >= start_ts) & (df_cached.index <= end_ts)]
    else:
        df_cached = pd.DataFrame(columns=price_columns)

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
        return df_cached.loc[(df_cached.index >= start_ts) & (df_cached.index <= end_ts)]

    if resp.status_code != 200:
        logging.error(
            "Option contract aggregate HTTP error",
            extra={"contract": contract_ticker, "status": resp.status_code, "body": resp.text[:500]},
        )
        return df_cached.loc[(df_cached.index >= start_ts) & (df_cached.index <= end_ts)]

    payload = resp.json()
    results = payload.get("results", [])
    if results:
        df_new = pd.DataFrame(results)
        if "t" not in df_new.columns:
            logging.error("Polygon agg response missing timestamp", extra={"contract": contract_ticker})
            return df_cached.loc[(df_cached.index >= start_ts) & (df_cached.index <= end_ts)]

        df_new["t"] = pd.to_datetime(df_new["t"], unit="ms", errors="coerce")
        df_new = df_new.dropna(subset=["t"])
        df_new["t"] = df_new["t"].dt.tz_localize(None)
        df_new = df_new.set_index("t")
        df_new.index.name = "t"

        # Ensure the resulting frame contains the expected OHLCV schema.
        for col in price_columns:
            if col not in df_new.columns:
                df_new[col] = pd.NA
        df_new = df_new[price_columns]

        combined = pd.concat([df_cached, df_new])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index()

        try:
            write_hdf(combined, key)
            logging.info("Cached contract OHLCV", extra={"contract": contract_ticker, "key": key})
        except Exception as exc:
            logging.error("Failed to cache contract OHLCV", extra={"key": key, "error": str(exc)})

        df_cached = combined

    window = df_cached.loc[(df_cached.index >= start_ts) & (df_cached.index <= end_ts)]
    return window


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
    """Standardize timestamp and date columns for market dataframes.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw market data that may contain ``Timestamp`` or ``Date`` columns.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with normalized ``Timestamp`` and ``Date`` columns. An
        empty frame is returned when the input is empty.

    Raises
    ------
    ValueError
        Raised when neither ``Timestamp`` nor ``Date`` columns are present.
    """
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
    if "Date" in frame.columns:
        frame["Date"] = pd.to_datetime(frame["Date"]).dt.date
    else:
        frame["Date"] = frame["Timestamp"].dt.date
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
    """Return the slice of market data between start/end for the provided interval.

    Parameters
    ----------
    df : pandas.DataFrame
        Market data frame to filter.
    start_ts : pandas.Timestamp
        Inclusive lower bound.
    end_ts : pandas.Timestamp
        Inclusive upper bound.
    interval_key : str
        Interval key controlling whether comparisons use dates or timestamps.

    Returns
    -------
    pandas.DataFrame
        Filtered frame containing only rows within the requested window.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    frame = _ensure_market_datetime(df)
    if interval_key == "1D":
        dates = pd.to_datetime(frame["Date"]).dt.normalize()
        mask = (dates >= start_ts) & (dates <= end_ts)
    else:
        mask = (frame["Timestamp"] >= start_ts) & (frame["Timestamp"] <= end_ts)
    return frame.loc[mask].reset_index(drop=True)


def _cache_market_bounds(df: pd.DataFrame, interval_key: str) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """Return the minimum and maximum timestamp/date stored for market data.

    Parameters
    ----------
    df : pandas.DataFrame
        Cached market data frame.
    interval_key : str
        Interval key used to determine whether to inspect ``Timestamp`` or ``Date``.

    Returns
    -------
    tuple of pandas.Timestamp or None
        Tuple ``(min_timestamp, max_timestamp)`` or ``(None, None)`` when the
        frame is empty.
    """
    if df is None or df.empty:
        return None, None
    frame = _ensure_market_datetime(df)
    if interval_key == "1D":
        dates = pd.to_datetime(frame["Date"]).dt.normalize()
        return dates.min(), dates.max()
    return frame["Timestamp"].min(), frame["Timestamp"].max()


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
                "Date": timestamp.date(),
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
    df = fetch_market_data(
        symbol,
        start_ts.strftime("%Y-%m-%d") if interval_key == "1D" else start_ts.isoformat(),
        end_ts.strftime("%Y-%m-%d") if interval_key == "1D" else end_ts.isoformat(),
        interval=interval_key,
    )
    if df.empty:
        logging.warning(f"No market data retrieved for {symbol}", extra={"interval": interval_key})
        return pd.DataFrame()

    combined = pd.concat([cached_full, df], ignore_index=True) if not cached_full.empty else df
    combined = _ensure_market_datetime(combined)
    write_hdf(combined, market_key)
    return _filter_market_window(combined, start_ts, end_ts, interval_key)


def get_options_data(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str | None = None,
    use_cache: bool = True,
    look_forward: int = 7,
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
    look_forward : int, default=7
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

    market_df = get_market_data(symbol, start_date, end_date, interval=interval_key, use_cache=use_cache)
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

    from_ts = start_ts + forward_delta
    to_ts = end_ts + forward_delta

    from_str = from_ts.date().isoformat()
    to_str = to_ts.date().isoformat()

    try:
        # Use the derived window and strike filters to retrieve cached contracts,
        # backfilling the HDF store when required.
        contracts_df = _load_or_fetch_contracts(
            symbol=symbol,
            timeframe=interval_key,
            from_=from_str,
            to=to_str,
            strike_min=strike_min,
            strike_max=strike_max,
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
