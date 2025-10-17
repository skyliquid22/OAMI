import os, time, random, logging, pandas as pd, numpy as np, requests
from datetime import datetime
from tqdm import tqdm
from polygon import RESTClient
from polygon.exceptions import BadResponse
from .config import Settings
SET = Settings()

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

def _cache_path(symbol: str, suffix: str = "", timeframe: str = None) -> str:
    tf = timeframe or SET.timeframe
    folder = os.path.join(SET.cache_root, tf); os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"{symbol}{suffix}.csv")

def _load_from_cache(symbol: str, suffix: str = "", timeframe: str = None) -> pd.DataFrame:
    path = _cache_path(symbol, suffix, timeframe)
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, parse_dates=["Date"])
            logging.info("Loaded cached data", extra={"path": path, "rows": len(df)}); return df
        except Exception as e:
            logging.warning("Failed to load cache", extra={"path": path, "error": str(e)})
    return pd.DataFrame()

def _save_to_cache(df: pd.DataFrame, symbol: str, suffix: str = "", timeframe: str = None) -> None:
    path = _cache_path(symbol, suffix, timeframe)
    try:
        df.to_csv(path, index=False); logging.info("Saved data to cache", extra={"path": path, "rows": len(df)})
    except Exception as e:
        logging.error("Cache write failed", extra={"path": path, "error": str(e)})

@retry_request(max_retries=3)
def fetch_market_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    client = RESTClient(SET.api_key)
    bars = client.list_aggs(ticker=symbol, multiplier=1, timespan="day", from_=start, to=end, limit=50000)
    data = [{"Date": bar.timestamp.date(), "Open": bar.open, "High": bar.high, "Low": bar.low, "Close": bar.close, "Volume": bar.volume} for bar in bars]
    return pd.DataFrame(data).sort_values("Date")

@retry_request(max_retries=3)
def fetch_options_data(symbol: str, date: str):
    client = RESTClient(SET.api_key)
    return client.get_aggs(f"O:{symbol}", multiplier=1, timespan="day", from_=date, to=date, limit=50000)

def get_market_data(symbol: str, start: str, end: str, use_cache: bool = True) -> pd.DataFrame:
    if use_cache:
        cached = _load_from_cache(symbol)
        if not cached.empty: return cached
    df = fetch_market_data(symbol, start, end)
    if df is None or df.empty:
        logging.warning("No market data retrieved", extra={"symbol": symbol}); return pd.DataFrame()
    _save_to_cache(df, symbol); return df

def get_options_data(symbol: str, start_date: str, end_date: str, use_cache: bool = True) -> pd.DataFrame:
    if use_cache:
        cached = _load_from_cache(symbol, suffix="_options")
        if not cached.empty: return cached
    dates = pd.date_range(datetime.fromisoformat(start_date), datetime.fromisoformat(end_date)); summary = []
    for d in tqdm(dates, desc=f"Fetching Options Data for {symbol}"):
        aggs = fetch_options_data(symbol, d.strftime("%Y-%m-%d"))
        if not aggs or not hasattr(aggs, "results") or len(aggs.results) == 0: continue
        puts = [a for a in aggs.results if "P" in a.get("T","")]; calls = [a for a in aggs.results if "C" in a.get("T","")]
        if not puts or not calls: continue
        total_call_vol = sum(a.get("v",0) for a in calls); total_put_vol  = sum(a.get("v",0) for a in puts)
        total_call_oi  = np.mean([a.get("o",0) for a in calls]) if calls else 0.0
        total_put_oi   = np.mean([a.get("o",0) for a in puts]) if puts else 0.0
        ratio = (total_put_vol / total_call_vol) if total_call_vol else np.nan
        summary.append({"Date": d.date(),"PutVol": total_put_vol,"CallVol": total_call_vol,"PutOI": total_put_oi,"CallOI": total_call_oi,"PutCallVolRatio": ratio})
    df_opt = pd.DataFrame(summary)
    if df_opt.empty:
        logging.warning("No options data collected", extra={"symbol": symbol}); return df_opt
    df_opt.sort_values("Date", inplace=True); df_opt["SentimentIndex"] = 1 - df_opt["PutCallVolRatio"]
    _save_to_cache(df_opt, symbol, suffix="_options"); return df_opt
