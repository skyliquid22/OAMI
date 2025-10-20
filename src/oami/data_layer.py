import time, random, logging, pandas as pd, requests
from datetime import datetime
from typing import Dict, List, Tuple
try:
    from tqdm.auto import tqdm  # type: ignore
except ImportError:  # pragma: no cover
    from tqdm import tqdm  # type: ignore
from polygon import RESTClient
from polygon.exceptions import BadResponse

from oami.utils.cache_manager import (
    load_cache,
    save_cache,
    read_hdf,
    write_hdf,
    make_option_key,
    get_timeframe_str,
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

_FIRST_OPTIONS_REQUEST_LOGGED: set[str] = set()


def _contracts_key(symbol: str) -> str:
    return f"/options/contracts/{symbol.upper()}"


def _load_or_fetch_contracts(symbol: str, as_of: str | None = None) -> pd.DataFrame:
    """Load contract metadata for a date, fetching and persisting when missing."""
    columns = ["ticker", "contract_type", "expiration_date", "strike_price"]
    key = _contracts_key(symbol)

    df_cached = read_hdf(key)
    if df_cached.empty:
        df_cached = pd.DataFrame(columns=columns)
    else:
        df_cached = df_cached.reindex(columns=columns, fill_value=pd.NA)

    if as_of:
        df_raw = fetch_option_contracts(symbol, as_of)
    else:
        df_raw = pd.DataFrame()

    if not df_raw.empty:
        df_new = df_raw.reindex(columns=columns, fill_value=pd.NA)
        if df_cached.empty:
            df_cached = df_new
        else:
            df_cached = pd.concat([df_cached, df_new], ignore_index=True)
            df_cached = df_cached.drop_duplicates(subset=["ticker"], keep="last").reset_index(drop=True)
        try:
            write_hdf(df_cached, key)
            logging.info("Cached option contracts", extra={"symbol": symbol, "key": key})
        except Exception as exc:
            logging.error("Failed to cache contracts", extra={"key": key, "error": str(exc)})

    for col in columns:
        if col not in df_cached.columns:
            df_cached[col] = pd.NA
    return df_cached[columns].copy()



def retry_request(max_retries: int = 3, base_delay: float = 1.0, jitter: float = 0.3, backoff: float = 2.0):
    def decorator(func):
        def wrapper(*args, **kwargs):
            delay = base_delay
            for attempt in range(1, max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    logging.info("Polygon call succeeded", extra={"function": func.__name__, "attempt": attempt})
                    return result
                except (BadResponse, requests.exceptions.RequestException) as e:
                    logging.warning("Polygon API/network error", extra={"function": func.__name__, "attempt": attempt, "error": str(e)})
                except Exception as e:
                    logging.error("Unexpected error", extra={"function": func.__name__, "attempt": attempt, "error": str(e)})
                if attempt < max_retries:
                    sleep_time = delay + random.uniform(0, jitter)
                    logging.info("Retrying Polygon API call", extra={"function": func.__name__, "attempt": attempt, "next_delay": sleep_time})
                    time.sleep(sleep_time); delay *= backoff
            logging.error("Polygon call failed after retries", extra={"function": func.__name__}); return None
        return wrapper
    return decorator

@retry_request(max_retries=3)
def fetch_option_contracts(symbol: str, as_of: str, limit: int = 1000) -> pd.DataFrame:
    """
    Fetch contract metadata for the given underlying and as_of date using Polygon's reference endpoint.
    Handles pagination via `next_url`.
    """
    base_url = "https://api.polygon.io/v3/reference/options/contracts"
    params = {
        "underlying_ticker": symbol,
        "as_of": as_of,
        "limit": limit,
        "apiKey": SET.api_key,
    }
    url = base_url
    all_results: List[Dict] = []
    first_page = True

    while url:
        logging.info(
            "Fetching option contracts",
            extra={"symbol": symbol, "as_of": as_of, "url": url if first_page else "next_url"},
        )
        try:
            resp = requests.get(url, params=params if first_page else None)
        except requests.exceptions.RequestException as exc:
            logging.error("Option contracts request failed", extra={"symbol": symbol, "error": str(exc)})
            return pd.DataFrame()

        if resp.status_code != 200:
            logging.error(
                "Option contracts HTTP error",
                extra={"symbol": symbol, "status": resp.status_code, "body": resp.text[:500]},
            )
            return pd.DataFrame()

        payload = resp.json()
        if first_page and symbol not in _FIRST_OPTIONS_REQUEST_LOGGED:
            _FIRST_OPTIONS_REQUEST_LOGGED.add(symbol)
        results = payload.get("results", [])
        all_results.extend(results)

        next_url = payload.get("next_url")
        if not next_url:
            break

        url = next_url
        params = None
        if "apiKey" not in url:
            connector = "&" if "?" in url else "?"
            url = f"{url}{connector}apiKey={SET.api_key}"
        first_page = False

    if not all_results:
        return pd.DataFrame()

    df = pd.DataFrame(all_results)
    df["as_of"] = pd.to_datetime(as_of)
    return df


@retry_request(max_retries=3)
def fetch_option_contract_agg(
    symbol: str,
    contract_ticker: str,
    start_date: str,
    end_date: str,
    multiplier: int = 1,
    timespan: str = "day",
) -> pd.DataFrame:
    """Fetch aggregated bars for an option contract and persist them."""
    timeframe = get_timeframe_str(multiplier, timespan)
    key = make_option_key(contract_ticker, timeframe)

    df_cached = read_hdf(key)
    if not df_cached.empty:
        df_cached = df_cached.copy()
        df_cached["date"] = pd.to_datetime(df_cached.get("date"), errors="coerce")
        df_cached = df_cached.dropna(subset=["date"])
        df_cached["date"] = df_cached["date"].dt.date
    else:
        df_cached = pd.DataFrame(columns=["timestamp", "date", "volume", "open_interest", "ticker"])

    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    if pd.isna(start_ts) or pd.isna(end_ts) or end_ts < start_ts:
        logging.warning("Invalid date range for contract agg", extra={"contract": contract_ticker, "start": start_date, "end": end_date})
        return pd.DataFrame()

    if not df_cached.empty and not df_cached[df_cached.get("ticker") == contract_ticker].empty:
        existing = df_cached[df_cached.get("ticker") == contract_ticker]
        min_cached = existing["date"].min()
        max_cached = existing["date"].max()
        if min_cached <= start_ts.date() and max_cached >= end_ts.date():
            return existing.reset_index(drop=True)

    url = f"https://api.polygon.io/v2/aggs/ticker/{contract_ticker}/range/{multiplier}/{timespan}/{start_ts.date().isoformat()}/{end_ts.date().isoformat()}"
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
        return pd.DataFrame()

    if resp.status_code != 200:
        logging.error(
            "Option contract aggregate HTTP error",
            extra={"contract": contract_ticker, "status": resp.status_code, "body": resp.text[:500]},
        )
        return pd.DataFrame()

    payload = resp.json()
    results = payload.get("results", [])
    if not results:
        logging.info("No aggregate data returned for contract", extra={"contract": contract_ticker})
        df_new = pd.DataFrame(columns=["timestamp", "date", "volume", "open_interest", "ticker"])
    else:
        df_new = pd.DataFrame(results)
        rename_map = {"t": "timestamp", "v": "volume", "oi": "open_interest"}
        df_new = df_new.rename(columns=rename_map)
        if "timestamp" in df_new.columns:
            df_new["timestamp"] = pd.to_datetime(df_new["timestamp"], unit="ms", errors="coerce")
            df_new["date"] = df_new["timestamp"].dt.date
        for col in ["volume", "open_interest"]:
            if col not in df_new.columns:
                df_new[col] = pd.NA
        if "date" in df_new.columns:
            df_new["date"] = pd.to_datetime(df_new["date"], errors="coerce").dt.date
        df_new["ticker"] = contract_ticker

    try:
        combined = pd.concat([df_cached, df_new], ignore_index=True)
        combined["date"] = pd.to_datetime(combined.get("date"), errors="coerce")
        combined = combined.dropna(subset=["date"])
        combined["date"] = combined["date"].dt.date
        combined = combined.drop_duplicates(subset=["ticker", "date"], keep="last").reset_index(drop=True)
        write_hdf(combined, key)
        logging.info("Cached contract aggregates", extra={"symbol": symbol, "ticker": contract_ticker, "key": key})
    except Exception as exc:
        logging.error("Failed to cache contract aggregates", extra={"key": key, "error": str(exc)})

    combined_contract = combined[combined.get("ticker") == contract_ticker].copy()
    combined_contract["date"] = pd.to_datetime(combined_contract.get("date"), errors="coerce").dt.date
    return combined_contract[(combined_contract["date"] >= start_ts.date()) & (combined_contract["date"] <= end_ts.date())].reset_index(drop=True)


def _normalize_interval(interval: str | None) -> str:
    """Return a canonical interval key used across the data layer."""
    default_interval = (SET.default_interval or "1D").upper()
    interval_key = (interval or default_interval).upper()
    if interval_key not in INTERVAL_CONFIG:
        raise ValueError(
            f"Unsupported interval '{interval_key}'. Expected one of {list(INTERVAL_CONFIG.keys())}"
        )
    return interval_key

def _interval_delta(interval_key: str) -> pd.Timedelta:
    return INTERVAL_CONFIG[interval_key]["delta"]  # type: ignore[index]

def _ensure_market_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize timestamp/date columns for market dataframes."""
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
    """Normalize start/end bounds for a given interval."""
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
    if interval_key == "1D":
        dates = pd.to_datetime(frame["Date"]).dt.normalize()
        mask = (dates >= start_ts) & (dates <= end_ts)
    else:
        mask = (frame["Timestamp"] >= start_ts) & (frame["Timestamp"] <= end_ts)
    return frame.loc[mask].reset_index(drop=True)

def _filter_options_window(df: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    """Slice options sentiment data between the requested day bounds."""
    if df is None or df.empty:
        return pd.DataFrame()
    frame = df.copy()
    frame["Date"] = pd.to_datetime(frame["Date"]).dt.normalize()
    mask = (frame["Date"] >= start_ts) & (frame["Date"] <= end_ts)
    return frame.loc[mask].reset_index(drop=True)

def _cache_market_bounds(df: pd.DataFrame, interval_key: str) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """Return the minimum and maximum timestamp/date stored for market data."""
    if df is None or df.empty:
        return None, None
    frame = _ensure_market_datetime(df)
    if interval_key == "1D":
        dates = pd.to_datetime(frame["Date"]).dt.normalize()
        return dates.min(), dates.max()
    return frame["Timestamp"].min(), frame["Timestamp"].max()

def _cache_options_bounds(df: pd.DataFrame) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """Return the min/max available dates for options sentiment cache."""
    if df is None or df.empty:
        return None, None
    dates = pd.to_datetime(df["Date"]).dt.normalize()
    return dates.min(), dates.max()


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


def _fetch_options_range(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    interval: str = "1D",
    show_progress: bool = True,
) -> pd.DataFrame:
    """Fetch aggregated options sentiment for the provided inclusive date range."""
    interval_key = _normalize_interval(interval)
    if interval_key != "1D":
        raise ValueError("Options sentiment currently supports only the 1D interval.")

    start_ts, end_ts = _resolve_window(start, end, interval_key)
    dates = pd.date_range(start_ts, end_ts, freq="D")
    iterator = tqdm(dates, desc=f"Options data {symbol}", unit="day", leave=False, position=0) if show_progress else dates
    agg_cache: Dict[str, pd.DataFrame] = {}

    summary = []

    for d in iterator:
        as_of = d.strftime("%Y-%m-%d")
        contracts_master = _load_or_fetch_contracts(symbol, as_of)
        contracts_master = contracts_master.dropna(subset=["ticker"]).copy()
        contracts_master["contract_type"] = contracts_master.get("contract_type", "").astype(str).str.upper()
        contracts_master["strike_price"] = pd.to_numeric(contracts_master.get("strike_price"), errors="coerce")
        contracts_master["expiration_date"] = pd.to_datetime(contracts_master.get("expiration_date"), errors="coerce")
        contracts_in_range = contracts_master[
            (contracts_master["expiration_date"] >= start_ts)
            & (contracts_master["expiration_date"] <= end_ts)
        ].dropna(subset=["expiration_date"]).reset_index(drop=True)
        contracts_list = contracts_in_range.to_dict("records")
        total_contracts = len(contracts_list)
        row = {
            "Date": as_of,
            "CallVol": None,
            "PutVol": None,
            "CallOI": None,
            "PutOI": None,
            "PutCallVolRatio": None,
            "PutCallOIRatio": None,
            "SentimentIndex": None,
            "CallContractCount": 0,
            "PutContractCount": 0,
            "AvgCallStrike": None,
            "AvgPutStrike": None,
            "EarliestExpiration": None,
            "LatestExpiration": None,
        }

        if total_contracts == 0:
            logging.info("No contracts in expiration window", extra={"symbol": symbol})
            summary.append(row)
            continue

        row["EarliestExpiration"] = contracts_in_range["expiration_date"].min()
        row["LatestExpiration"] = contracts_in_range["expiration_date"].max()

        total_call_vol = total_put_vol = 0.0
        total_call_oi = total_put_oi = 0.0
        call_count = put_count = 0
        call_strikes: List[float] = []
        put_strikes: List[float] = []

        for idx, contract in enumerate(contracts_list, start=1):
            ticker = contract.get("ticker")
            if not ticker or pd.isna(ticker):
                continue
            contract_type = str(contract.get("contract_type") or "").upper()
            strike_price = pd.to_numeric(contract.get("strike_price"), errors="coerce")

            if show_progress and total_contracts:
                iterator.set_postfix_str(f"{as_of} {idx}/{total_contracts} {ticker}")

            agg_df = agg_cache.get(ticker)
            if agg_df is None:
                agg_df = fetch_option_contract_agg(
                    symbol=symbol,
                    contract_ticker=ticker,
                    start_date=start_ts.date().isoformat(),
                    end_date=end_ts.date().isoformat(),
                )
                if "date" in agg_df.columns:
                    agg_df["date"] = pd.to_datetime(agg_df["date"], errors="coerce").dt.date
                agg_cache[ticker] = agg_df

            daily_rows = pd.DataFrame()
            if not agg_df.empty and "date" in agg_df.columns:
                daily_rows = agg_df[agg_df["date"] == d.date()]

            volume = pd.to_numeric(daily_rows.get("volume"), errors="coerce").fillna(0.0).sum() if "volume" in daily_rows.columns else 0.0
            open_interest = pd.to_numeric(daily_rows.get("open_interest"), errors="coerce").fillna(0.0).sum() if "open_interest" in daily_rows.columns else 0.0

            if contract_type == "CALL":
                call_count += 1
                if pd.notna(strike_price):
                    call_strikes.append(float(strike_price))
                total_call_vol += volume
                total_call_oi += open_interest
            elif contract_type == "PUT":
                put_count += 1
                if pd.notna(strike_price):
                    put_strikes.append(float(strike_price))
                total_put_vol += volume
                total_put_oi += open_interest

        if show_progress and total_contracts:
            iterator.set_postfix_str("")

        total_vol = total_call_vol + total_put_vol

        row.update({
            "CallVol": total_call_vol if total_call_vol else None,
            "PutVol": total_put_vol if total_put_vol else None,
            "CallOI": total_call_oi if total_call_oi else None,
            "PutOI": total_put_oi if total_put_oi else None,
            "PutCallVolRatio": (total_put_vol / total_call_vol) if total_call_vol else None,
            "PutCallOIRatio": (total_put_oi / total_call_oi) if total_call_oi else None,
            "SentimentIndex": ((total_call_vol - total_put_vol) / total_vol) if total_vol else None,
            "CallContractCount": call_count,
            "PutContractCount": put_count,
            "AvgCallStrike": (sum(call_strikes) / len(call_strikes)) if call_strikes else None,
            "AvgPutStrike": (sum(put_strikes) / len(put_strikes)) if put_strikes else None,
        })

        if row["EarliestExpiration"] is not None:
            row["EarliestExpiration"] = pd.to_datetime(row["EarliestExpiration"]).date().isoformat()
        if row["LatestExpiration"] is not None:
            row["LatestExpiration"] = pd.to_datetime(row["LatestExpiration"]).date().isoformat()

        if row["EarliestExpiration"] is not None:
            row["EarliestExpiration"] = pd.to_datetime(row["EarliestExpiration"]).date().isoformat()
        if row["LatestExpiration"] is not None:
            row["LatestExpiration"] = pd.to_datetime(row["LatestExpiration"]).date().isoformat()

        summary.append(row)

    df = pd.DataFrame(summary)
    if df.empty:
        return df
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    if "EarliestExpiration" in df.columns:
        df["EarliestExpiration"] = pd.to_datetime(df["EarliestExpiration"], errors="coerce").dt.strftime("%Y-%m-%d")
    if "LatestExpiration" in df.columns:
        df["LatestExpiration"] = pd.to_datetime(df["LatestExpiration"], errors="coerce").dt.strftime("%Y-%m-%d")
    return df.sort_values("Date").reset_index(drop=True)

def get_market_data(symbol: str, start: str, end: str, interval: str | None = None, use_cache: bool = True) -> pd.DataFrame:
    """
    Retrieve market data for a symbol, optionally using cached values.

    Parameters
    ----------
    symbol : str
        Ticker symbol.
    start : str
        Start timestamp in ISO format.
    end : str
        End timestamp in ISO format.
    interval : str, optional
        Interval key (1M, 5M, 15M, 1H, 4H, 1D). Defaults to settings default.
    use_cache : bool
        Whether to read/write cache.
    """
    try:
        interval_key = _normalize_interval(interval)
    except ValueError as exc:
        logging.error(str(exc))
        return pd.DataFrame()

    start_ts, end_ts = _resolve_window(start, end, interval_key)

    cached_full = load_cache("market", symbol, interval=interval_key) if use_cache else pd.DataFrame()
    if not cached_full.empty:
        cached_min, cached_max = _cache_market_bounds(cached_full, interval_key)
        if cached_min is not None and cached_max is not None:
            if start_ts >= cached_min and end_ts <= cached_max:
                return _filter_market_window(cached_full, start_ts, end_ts, interval_key)

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
    save_cache(combined, "market", symbol, interval=interval_key)
    return _filter_market_window(combined, start_ts, end_ts, interval_key)

def get_options_data(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetches daily aggregated options sentiment data (volume, OI ratios) for a given symbol.
    Uses Polygon's /v2/aggs/ticker endpoint to retrieve historical ranges.

    Parameters
    ----------
    symbol : str
        Underlying symbol (e.g. 'SPY')
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date (YYYY-MM-DD)
    interval : str, optional
        Supported interval keys (currently only 1D).
    use_cache : bool
        Whether to read/write from local cache

    Returns
    -------
    pandas.DataFrame
        Daily sentiment summary (Call/Put ratios, OI metrics)
    """
    try:
        interval_key = _normalize_interval(interval)
    except ValueError as exc:
        logging.error(str(exc))
        return pd.DataFrame()

    if interval_key != "1D":
        logging.error("Options data currently supports only the 1D interval.")
        return pd.DataFrame()

    start_ts, end_ts = _resolve_window(start_date, end_date, interval_key)

    cached_full = load_cache("options", symbol, interval=interval_key) if use_cache else pd.DataFrame()
    if not cached_full.empty:
        cached_min, cached_max = _cache_options_bounds(cached_full)
        if cached_min is not None and cached_max is not None:
            if start_ts >= cached_min and end_ts <= cached_max:
                return _filter_options_window(cached_full, start_ts, end_ts)

    df = _fetch_options_range(
        symbol,
        start_ts,
        end_ts,
        interval=interval_key,
        show_progress=True,
    )
    if df.empty:
        logging.info(f"Fetched 0 days of options sentiment for {symbol}")
        return df

    combined_full = pd.concat([cached_full, df], ignore_index=True) if not cached_full.empty else df
    combined_full = combined_full.copy()
    combined_full["Date"] = pd.to_datetime(combined_full["Date"]).dt.normalize()
    combined_full = combined_full.drop_duplicates(subset="Date").sort_values("Date").reset_index(drop=True)

    window = _filter_options_window(combined_full, start_ts, end_ts)
    if window.empty:
        logging.info(f"Fetched 0 days of options sentiment for {symbol}")
        return window

    save_cache(combined_full, "options", symbol, interval=interval_key)
    logging.info(f"Fetched {len(window)} days of options sentiment for {symbol}")
    return window

def load_or_fetch_market(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Load cached market data for the requested window, fetching and backfilling gaps as needed.

    Data are stored under `data/csv/market/<interval>/<symbol>.csv`. Only missing segments
    are retrieved from Polygon to minimize API usage.
    """
    try:
        interval_key = _normalize_interval(interval)
    except ValueError as exc:
        logging.error(str(exc))
        return pd.DataFrame()

    requested_start, requested_end = _resolve_window(start_date, end_date, interval_key)

    cached = load_cache("market", symbol, interval=interval_key) if not force_refresh else pd.DataFrame()
    if not cached.empty:
        cached = _ensure_market_datetime(cached)

    cached_min, cached_max = _cache_market_bounds(cached, interval_key)

    if (
        not force_refresh
        and cached_min is not None
        and cached_max is not None
        and requested_start >= cached_min
        and requested_end <= cached_max
    ):
        return _filter_market_window(cached, requested_start, requested_end, interval_key)

    missing_ranges: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    if force_refresh or cached.empty or cached_min is None or cached_max is None:
        missing_ranges = [(requested_start, requested_end)]
    else:
        step = _interval_delta(interval_key)
        if requested_start < cached_min:
            fetch_end = min(requested_end, cached_min - step)
            if fetch_end >= requested_start:
                missing_ranges.append((requested_start, fetch_end))
        if requested_end > cached_max:
            fetch_start = max(requested_start, cached_max + step)
            if requested_end >= fetch_start:
                missing_ranges.append((fetch_start, requested_end))

        range_iterator = tqdm(missing_ranges, desc=f"{symbol}: market gaps", unit="range", leave=False, position=0) if missing_ranges else missing_ranges
    for range_start, range_end in range_iterator:
        if range_start > range_end:
            continue
        start_str = range_start.strftime("%Y-%m-%d") if interval_key == "1D" else range_start.isoformat()
        end_str = range_end.strftime("%Y-%m-%d") if interval_key == "1D" else range_end.isoformat()
        logging.info(
            "Fetching market data via Polygon",
            extra={"symbol": symbol, "interval": interval_key, "from": start_str, "to": end_str},
        )
        chunk = fetch_market_data(symbol, start_str, end_str, interval=interval_key)
        if chunk.empty:
            continue
        fetched = _ensure_market_datetime(chunk)
        cached = pd.concat([cached, fetched], ignore_index=True) if not cached.empty else fetched
        cached = _ensure_market_datetime(cached)
        save_cache(cached, "market", symbol, interval=interval_key)

    if force_refresh and cached.empty:
        save_cache(cached, "market", symbol, interval=interval_key)

    if cached.empty:
        return cached

    return _filter_market_window(cached, requested_start, requested_end, interval_key)

def load_or_fetch_options(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Load cached options sentiment if present, otherwise fetch from Polygon and persist to
    `data/csv/options/<interval>/<symbol>.csv`. Only the 1D interval is currently supported.
    """
    try:
        interval_key = _normalize_interval(interval)
    except ValueError as exc:
        logging.error(str(exc))
        return pd.DataFrame()

    if interval_key != "1D":
        logging.error("Options data currently supports only the 1D interval.")
        return pd.DataFrame()

    requested_start, requested_end = _resolve_window(start_date, end_date, interval_key)

    cached = load_cache("options", symbol, interval=interval_key) if not force_refresh else pd.DataFrame()
    if not cached.empty:
        cached = cached.copy()
        cached["Date"] = pd.to_datetime(cached["Date"]).dt.normalize()
        cached = cached.drop_duplicates(subset="Date").sort_values("Date").reset_index(drop=True)

    cached_min, cached_max = _cache_options_bounds(cached)

    if (
        not force_refresh
        and cached_min is not None
        and cached_max is not None
        and requested_start >= cached_min
        and requested_end <= cached_max
    ):
        return _filter_options_window(cached, requested_start, requested_end)

    missing_ranges: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    if force_refresh or cached.empty or cached_min is None or cached_max is None:
        missing_ranges = [(requested_start, requested_end)]
    else:
        step = pd.Timedelta(days=1)
        if requested_start < cached_min:
            fetch_end = min(requested_end, cached_min - step)
            if fetch_end >= requested_start:
                missing_ranges.append((requested_start, fetch_end))
        if requested_end > cached_max:
            fetch_start = max(requested_start, cached_max + step)
            if requested_end >= fetch_start:
                missing_ranges.append((fetch_start, requested_end))

    range_iterator = tqdm(missing_ranges, desc=f"{symbol}: option gaps", unit="range", leave=False, position=0) if missing_ranges else missing_ranges
    for range_start, range_end in range_iterator:
        logging.info(
            "Fetching options data via Polygon",
            extra={"symbol": symbol, "from": range_start.strftime("%Y-%m-%d"), "to": range_end.strftime("%Y-%m-%d")},
        )
        chunk = _fetch_options_range(symbol, range_start, range_end, interval=interval_key, show_progress=True)
        if chunk.empty:
            continue
        fetched = chunk.copy()
        fetched["Date"] = pd.to_datetime(fetched["Date"]).dt.normalize()
        cached = pd.concat([cached, fetched], ignore_index=True) if not cached.empty else fetched
        cached = cached.drop_duplicates(subset="Date").sort_values("Date").reset_index(drop=True)
        save_cache(cached, "options", symbol, interval=interval_key)

    if force_refresh and cached.empty:
        save_cache(cached, "options", symbol, interval=interval_key)

    if cached.empty:
        return cached

    return _filter_options_window(cached, requested_start, requested_end)
