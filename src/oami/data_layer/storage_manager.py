"""Local cache helpers for option flatfiles and stock flatfile caches."""

from __future__ import annotations

import logging
import re
import sqlite3
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from tqdm import tqdm

from oami.utils import cache_manager
from oami.utils.cache_manager import parse_occ_ticker

from .flatfiles import _load_option_flatfile_for_date, _option_flatfile_path

LOGGER = logging.getLogger(__name__)

RAW_STOCK_ROOT = Path("data/raw/us_stocks_sip/day_aggs_v1")
STOCK_CACHE_ROOT = Path("data/cache/stocks")
STOCK_META_PATH = Path("data/meta/stocks_index.db")

_OCC_TICKER_RE = re.compile(r"O:([A-Z]{1,6})(\d{2})(\d{2})(\d{2})([CP])(\d{8})")

OPTION_COLUMNS = [
    "ticker",
    "volume",
    "open",
    "close",
    "high",
    "low",
    "window_start",
    "underlying_symbol",
    "strike_price",
    "contract_type",
    "expiration_date",
]


def _normalize_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    try:
        return pd.to_datetime(value).date()
    except (TypeError, ValueError) as exc:
        raise ValueError("Dates must be parseable as YYYY-MM-DD") from exc


def _convert_window_start(series: pd.Series) -> pd.Series:
    """Return normalized timestamps from Polygon nanosecond epoch values."""
    if series.empty:
        return series
    if pd.api.types.is_numeric_dtype(series):
        converted = pd.to_datetime(series, unit="ns", errors="coerce")
        return converted.dt.tz_localize(None)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        converted = pd.to_datetime(numeric, unit="ns", errors="coerce")
        missing = converted.isna()
        if missing.any():
            converted.loc[missing] = pd.to_datetime(series.loc[missing], errors="coerce")
        return converted.dt.tz_localize(None)
    converted = pd.to_datetime(series, errors="coerce")
    try:
        return converted.dt.tz_localize(None)
    except (TypeError, AttributeError):
        return converted


def _build_ticker_metadata(tickers: Iterable[str], target_symbol: str) -> dict[str, dict[str, object]]:
    """Parse OCC metadata for tickers matching the requested underlying."""
    meta: dict[str, dict[str, object]] = {}
    for ticker in tickers:
        if not ticker or not isinstance(ticker, str):
            continue
        try:
            info = parse_occ_ticker(ticker)
        except ValueError:
            LOGGER.debug("Skipping unparsable OCC ticker", extra={"ticker": ticker})
            continue

        if info.get("underlying", "").upper() != target_symbol:
            continue

        match = _OCC_TICKER_RE.match(ticker)
        if not match:
            LOGGER.debug("Ticker failed OCC regex after parse", extra={"ticker": ticker})
            continue

        strike_price = int(match.group(6)) / 1000.0
        contract_type = "call" if match.group(5).upper() == "C" else "put"
        month_token = info["month"]
        try:
            month_index = cache_manager.MONTHS.index(month_token) + 1
        except ValueError:
            LOGGER.debug("Unknown OCC month token", extra={"ticker": ticker, "month": month_token})
            continue

        expiration_ts = pd.Timestamp(year=info["year"], month=month_index, day=info["day"])

        meta[ticker] = {
            "underlying_symbol": info["underlying"].upper(),
            "strike_price": strike_price,
            "contract_type": contract_type,
            "expiration_date": expiration_ts,
        }
    return meta


def _extract_symbol_rows(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Filter and enrich raw option aggregates for a specific underlying."""
    if frame.empty:
        return pd.DataFrame(columns=OPTION_COLUMNS)

    base_columns = ["ticker", "volume", "open", "close", "high", "low", "window_start"]
    missing = [col for col in base_columns if col not in frame.columns]
    if missing:
        LOGGER.warning("Flatfile frame missing required columns", extra={"missing_columns": missing})
        return pd.DataFrame(columns=OPTION_COLUMNS)

    working = frame[base_columns].copy()
    working["ticker"] = working["ticker"].astype("string").str.strip()
    unique_tickers = working["ticker"].dropna().unique().tolist()
    if not unique_tickers:
        return pd.DataFrame(columns=OPTION_COLUMNS)

    metadata_lookup = _build_ticker_metadata(unique_tickers, symbol)
    mask = working["ticker"].map(metadata_lookup.__contains__)
    if not mask.any():
        return pd.DataFrame(columns=OPTION_COLUMNS)

    filtered = working.loc[mask].copy()
    meta_frame = pd.DataFrame([metadata_lookup[t] for t in filtered["ticker"]], index=filtered.index)
    filtered = pd.concat([filtered, meta_frame], axis=1)
    filtered["window_start"] = _convert_window_start(filtered["window_start"])
    filtered = filtered.dropna(subset=["window_start"])
    filtered = filtered[OPTION_COLUMNS]
    return filtered.reset_index(drop=True)


# -------------------------------------------------------------------------
# Stock cache helpers
# -------------------------------------------------------------------------
STOCK_REQUIRED_COLUMNS = ["ticker", "open", "high", "low", "close", "volume", "window_start"]


def _normalize_tickers(tickers: Sequence[str]) -> list[str]:
    unique = {ticker.strip().upper() for ticker in tickers if ticker and isinstance(ticker, str)}
    return sorted(unique)


def _connect_stock_metadata(meta_path: Path) -> sqlite3.Connection:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(meta_path)
    _ensure_stock_metadata_schema(conn)
    return conn


def _ensure_stock_metadata_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS stocks_cache_index (
            ticker TEXT NOT NULL,
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            path TEXT NOT NULL,
            rows INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (ticker, start_date, end_date)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_stocks_cache_range
        ON stocks_cache_index (ticker, start_date, end_date)
        """
    )


def _load_existing_stock_ranges(conn: sqlite3.Connection, tickers: list[str]) -> dict[str, list[tuple[date, date]]]:
    ranges: dict[str, list[tuple[date, date]]] = {ticker: [] for ticker in tickers}
    if not tickers:
        return ranges
    placeholders = ",".join("?" for _ in tickers)
    query = f"""
        SELECT ticker, start_date, end_date
        FROM stocks_cache_index
        WHERE ticker IN ({placeholders})
    """
    for ticker, start_str, end_str in conn.execute(query, tickers):
        try:
            start_dt = pd.to_datetime(start_str).date()
            end_dt = pd.to_datetime(end_str).date()
        except (TypeError, ValueError):
            continue
        ranges.setdefault(ticker, []).append((start_dt, end_dt))
    return ranges


def _is_range_covered(ranges: list[tuple[date, date]], start: date, end: date) -> bool:
    for existing_start, existing_end in ranges:
        if existing_start <= start and existing_end >= end:
            return True
    return False


def _upsert_stock_metadata(
    conn: sqlite3.Connection,
    *,
    ticker: str,
    start: date,
    end: date,
    path: Path,
    rows: int,
) -> None:
    conn.execute(
        """
        INSERT INTO stocks_cache_index (ticker, start_date, end_date, path, rows, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker, start_date, end_date) DO UPDATE SET
            path=excluded.path,
            rows=excluded.rows,
            created_at=excluded.created_at
        """,
        (
            ticker,
            start.isoformat(),
            end.isoformat(),
            str(path),
            int(rows),
            datetime.utcnow().isoformat(),
        ),
    )


def build_ticker_cache(
    tickers: Sequence[str],
    *,
    raw_root: str | Path | None = None,
    cache_root: str | Path | None = None,
    meta_path: str | Path | None = None,
) -> dict[str, Path]:
    """Construct per-ticker parquet caches from raw daily stock flatfiles."""

    normalized_tickers = _normalize_tickers(tickers)
    if not normalized_tickers:
        LOGGER.info("No tickers supplied to build_ticker_cache")
        return {}

    raw_root_path = Path(raw_root) if raw_root else RAW_STOCK_ROOT
    cache_root_path = Path(cache_root) if cache_root else STOCK_CACHE_ROOT
    meta_path_obj = Path(meta_path) if meta_path else STOCK_META_PATH

    if not raw_root_path.exists():
        LOGGER.warning("Raw stock flatfile directory missing", extra={"path": str(raw_root_path)})
        return {}

    cache_root_path.mkdir(parents=True, exist_ok=True)

    conn = _connect_stock_metadata(meta_path_obj)
    try:
        existing_ranges = _load_existing_stock_ranges(conn, normalized_tickers)

        month_dirs: list[Path] = []
        for year_dir in sorted(raw_root_path.iterdir()):
            if not year_dir.is_dir():
                continue
            for month_dir in sorted(year_dir.iterdir()):
                if month_dir.is_dir():
                    month_dirs.append(month_dir)

        results: dict[str, Path] = {}
        for month_dir in month_dirs:
            csv_paths = sorted(month_dir.glob("*.csv.gz"))
            if not csv_paths:
                continue

            parsed_dates: list[date] = []
            for csv_path in csv_paths:
                name = csv_path.name.split(".")[0]
                try:
                    file_date = pd.to_datetime(name).date()
                except (TypeError, ValueError):
                    continue
                parsed_dates.append(file_date)

            if not parsed_dates:
                continue

            month_start = min(parsed_dates)
            month_end = max(parsed_dates)

            pending_tickers = [
                ticker
                for ticker in normalized_tickers
                if not _is_range_covered(existing_ranges.get(ticker, []), month_start, month_end)
            ]
            if not pending_tickers:
                continue

            month_frames: dict[str, list[pd.DataFrame]] = defaultdict(list)
            progress_iter = tqdm(
                csv_paths,
                desc=f"Processing {month_dir.parent.name}-{month_dir.name}",
                leave=False,
                disable=not LOGGER.isEnabledFor(logging.INFO),
            )
            for csv_path in progress_iter:
                try:
                    df = pd.read_csv(csv_path)
                except Exception as exc:  # pragma: no cover - IO errors
                    LOGGER.warning("Failed to read stock flatfile", extra={"path": str(csv_path), "error": str(exc)})
                    continue

                if df.empty:
                    continue
                if "ticker" not in df.columns:
                    LOGGER.warning("Stock flatfile missing ticker column", extra={"path": str(csv_path)})
                    continue
                missing_cols = [col for col in STOCK_REQUIRED_COLUMNS if col not in df.columns]
                if missing_cols:
                    LOGGER.warning(
                        "Stock flatfile missing required columns",
                        extra={"path": str(csv_path), "missing": missing_cols},
                    )
                    continue

                df = df.copy()
                df["ticker"] = df["ticker"].astype(str).str.upper()
                mask = df["ticker"].isin(pending_tickers)
                if not mask.any():
                    continue

                filtered = df.loc[mask].copy()
                filtered["window_start"] = _convert_window_start(filtered["window_start"])
                filtered = filtered.dropna(subset=["window_start"])
                if filtered.empty:
                    continue

                filtered = filtered.sort_values("window_start").reset_index(drop=True)
                for ticker, group in filtered.groupby("ticker", sort=False):
                    month_frames[ticker].append(group.reset_index(drop=True))

            for ticker, frames in month_frames.items():
                if not frames:
                    continue
                ticker_month = pd.concat(frames, ignore_index=True)
                ticker_month = ticker_month.sort_values("window_start").reset_index(drop=True)
                ticker_month = ticker_month.drop_duplicates(subset=["window_start"], keep="last")

                cache_path = cache_root_path / f"{ticker}.parquet"
                if cache_path.exists():
                    try:
                        existing = pd.read_parquet(cache_path)
                    except Exception as exc:  # pragma: no cover - parquet corruption
                        LOGGER.warning(
                            "Failed to read existing ticker cache; overwriting",
                            extra={"ticker": ticker, "path": str(cache_path), "error": str(exc)},
                        )
                        existing = pd.DataFrame()
                    if not existing.empty:
                        if "window_start" in existing.columns:
                            existing = existing.copy()
                            existing["window_start"] = pd.to_datetime(existing["window_start"], errors="coerce")
                            existing = existing.dropna(subset=["window_start"])
                        ticker_month = (
                            pd.concat([existing, ticker_month], ignore_index=True)
                            .drop_duplicates(subset=["window_start"], keep="last")
                            .sort_values("window_start")
                            .reset_index(drop=True)
                        )

                ticker_month.to_parquet(cache_path, index=False, engine="pyarrow", compression="zstd")
                results[ticker] = cache_path

                rows_written = sum(len(frame) for frame in frames)
                _upsert_stock_metadata(
                    conn,
                    ticker=ticker,
                    start=month_start,
                    end=month_end,
                    path=cache_path,
                    rows=rows_written,
                )
                existing_ranges.setdefault(ticker, []).append((month_start, month_end))

            conn.commit()

        return results
    finally:
        conn.close()


def load_ticker_data(
    tickers: Sequence[str],
    start_date: str,
    end_date: str,
    *,
    cache_root: str | Path | None = None,
    meta_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load cached parquet slices for selected tickers within a date window."""

    normalized_tickers = _normalize_tickers(tickers)
    if not normalized_tickers:
        return pd.DataFrame(columns=STOCK_REQUIRED_COLUMNS)

    try:
        start_bound = pd.to_datetime(start_date)
        end_bound = pd.to_datetime(end_date)
    except (TypeError, ValueError) as exc:
        raise ValueError("start_date/end_date must be parseable dates") from exc
    if start_bound > end_bound:
        raise ValueError("start_date must be on or before end_date")

    start_date_str = start_bound.date().isoformat()
    end_date_str = end_bound.date().isoformat()
    end_exclusive = (end_bound + pd.Timedelta(days=1))

    meta_path_obj = Path(meta_path) if meta_path else STOCK_META_PATH
    cache_root_path = Path(cache_root) if cache_root else STOCK_CACHE_ROOT

    if not meta_path_obj.exists():
        LOGGER.info("Stock metadata database missing", extra={"path": str(meta_path_obj)})
        return pd.DataFrame(columns=STOCK_REQUIRED_COLUMNS)

    conn = _connect_stock_metadata(meta_path_obj)
    try:
        placeholders = ",".join("?" for _ in normalized_tickers)
        query = f"""
            SELECT ticker, start_date, end_date, path
            FROM stocks_cache_index
            WHERE ticker IN ({placeholders})
              AND start_date <= ?
              AND end_date >= ?
        """
        params = [*normalized_tickers, end_date_str, start_date_str]
        rows = conn.execute(query, params).fetchall()
    finally:
        conn.close()

    frames: list[pd.DataFrame] = []
    for ticker, start_str, end_str, path_str in rows:
        cache_path = Path(path_str)
        if not cache_path.is_absolute():
            cache_path = (cache_root_path / cache_path).resolve()
        if not cache_path.exists():
            LOGGER.warning("Ticker cache parquet missing on disk", extra={"ticker": ticker, "path": str(cache_path)})
            continue

        try:
            df = pd.read_parquet(cache_path)
        except Exception as exc:  # pragma: no cover - parquet read errors
            LOGGER.warning(
                "Failed to read ticker cache parquet",
                extra={"ticker": ticker, "path": str(cache_path), "error": str(exc)},
            )
            continue

        if df.empty:
            continue
        if "window_start" not in df.columns:
            LOGGER.warning(
                "Ticker cache missing window_start column",
                extra={"ticker": ticker, "path": str(cache_path)},
            )
            continue

        df = df.copy()
        df["ticker"] = df["ticker"].astype(str).str.upper()
        df = df[df["ticker"] == ticker.upper()]
        if df.empty:
            continue

        windows = pd.to_datetime(df["window_start"], errors="coerce")
        windows = windows.dt.tz_localize(None)
        mask = windows.notna()
        mask &= windows >= start_bound
        mask &= windows < end_exclusive
        filtered = df.loc[mask].copy()
        if filtered.empty:
            continue
        filtered["window_start"] = windows.loc[filtered.index]
        frames.append(filtered)

    if not frames:
        return pd.DataFrame(columns=STOCK_REQUIRED_COLUMNS)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["window_start"])
    combined = combined.sort_values(["window_start", "ticker"]).reset_index(drop=True)
    combined = combined.drop_duplicates(subset=["ticker", "window_start"], keep="last")
    return combined




def _build_single_option_cache(
    symbol_upper: str,
    start_date: date,
    end_date: date,
    storage_root: Path,
    cache_dir: Path,
) -> pd.DataFrame:
    date_index = pd.date_range(start=start_date, end=end_date, freq="D")
    frames: list[pd.DataFrame] = []

    for expiration in date_index:
        path = _option_flatfile_path(expiration, storage_root)
        if not path.exists():
            LOGGER.debug("Flatfile missing for date", extra={"path": str(path)})
            continue
        df_raw = _load_option_flatfile_for_date(expiration, storage_root)
        if df_raw.empty:
            continue
        filtered = _extract_symbol_rows(df_raw, symbol_upper)
        if not filtered.empty:
            frames.append(filtered)

    if frames:
        merged = pd.concat(frames, ignore_index=True)
        merged = merged.drop_duplicates(subset=["ticker", "window_start"], keep="last")
    else:
        merged = pd.DataFrame(columns=OPTION_COLUMNS)

    cache_path = cache_dir / f"{symbol_upper}.parquet"

    if cache_path.exists():
        try:
            existing = pd.read_parquet(cache_path)
            merged = pd.concat([existing, merged], ignore_index=True)
            merged = merged.drop_duplicates(subset=["ticker", "window_start"], keep="last")
            merged = merged.sort_values(["window_start", "ticker"]).reset_index(drop=True)
            merged = merged[OPTION_COLUMNS]
        except Exception as exc:  # pragma: no cover - unexpected parquet corruption
            LOGGER.warning("Failed to read existing option cache; overwriting", extra={"path": str(cache_path)})
            LOGGER.debug("Option cache read error", exc_info=exc)

    if merged.empty:
        merged = pd.DataFrame(columns=OPTION_COLUMNS)

    merged = merged[OPTION_COLUMNS]
    merged.to_parquet(cache_path, index=False)
    LOGGER.info(
        "Persisted option cache",
        extra={"symbol": symbol_upper, "rows": len(merged), "path": str(cache_path)},
    )
    return merged


def build_option_cache(
    symbols: Sequence[str] | str,
    start: str | date,
    end: str | date,
    *,
    flatfile_root: str | Path | None = None,
    cache_root: str | Path | None = None,
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """
    Build or update parquet caches of option aggregates for one or more symbols.

    Parameters
    ----------
    symbols : Sequence[str] or str
        Underlying ticker symbol(s) (e.g., ``"AAPL"``).
    start, end : str or datetime.date
        Inclusive date range selecting Polygon option flatfiles.
    flatfile_root : str or Path, optional
        Override root directory containing ``us_options_opra/day_aggs_v1``.
    cache_root : str or Path, optional
        Directory where parquet caches should be written. Defaults to ``data/cache/options``.

    Returns
    -------
    pandas.DataFrame
        For a single symbol input the cached frame is returned. When multiple
        symbols are supplied, a mapping of ``symbol -> DataFrame`` is returned.
    """

    single_symbol_input = isinstance(symbols, str)
    if single_symbol_input:
        symbol_list = [symbols]  # type: ignore[list-item]
    else:
        try:
            symbol_list = list(symbols)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError("symbols must be an iterable of non-empty strings") from exc

    normalized_symbols: list[str] = []
    for sym in symbol_list:
        if not isinstance(sym, str) or not sym.strip():
            raise ValueError("symbols must contain non-empty strings")
        normalized_symbols.append(sym.strip().upper())

    if not normalized_symbols:
        raise ValueError("symbols must contain at least one non-empty value")

    start_date = _normalize_date(start)
    end_date = _normalize_date(end)
    if start_date > end_date:
        raise ValueError("start must be on or before end")

    storage_root = Path(flatfile_root) if flatfile_root else Path("data/flatfiles")
    cache_dir = Path(cache_root) if cache_root else Path("data/cache/options")
    cache_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, pd.DataFrame] = {}
    for symbol_upper in normalized_symbols:
        results[symbol_upper] = _build_single_option_cache(
            symbol_upper=symbol_upper,
            start_date=start_date,
            end_date=end_date,
            storage_root=storage_root,
            cache_dir=cache_dir,
        )

    if single_symbol_input or len(normalized_symbols) == 1:
        return results[normalized_symbols[0]]
    return results


def load_option_data(
    symbols: Sequence[str],
    start: str | date,
    end: str | date,
    *,
    flatfile_root: str | Path | None = None,
    cache_root: str | Path | None = None,
    rebuild_missing: bool = True,
) -> pd.DataFrame:
    """
    Load cached option aggregates for one or more symbols.

    Parameters
    ----------
    symbols : Sequence[str]
        Collection of underlying tickers.
    start, end : str or datetime.date
        Inclusive date range for filtering on ``window_start``.
    flatfile_root : str or Path, optional
        Override for the local Polygon flatfile directory. Required when
        ``rebuild_missing`` is ``True`` and caches are absent.
    cache_root : str or Path, optional
        Directory containing per-symbol parquet caches. Defaults to ``data/cache/options``.
    rebuild_missing : bool, default True
        When ``True`` the cache is rebuilt for symbols lacking parquet files.

    Returns
    -------
    pandas.DataFrame
        Concatenated option rows within ``[start, end]`` across the requested symbols.
        Empty when no cache entries exist or the filter excludes all rows.
    """

    if not symbols:
        return pd.DataFrame(columns=OPTION_COLUMNS)

    start_date = _normalize_date(start)
    end_date = _normalize_date(end)
    if start_date > end_date:
        raise ValueError("start must be on or before end")

    storage_root = Path(flatfile_root) if flatfile_root else Path("data/flatfiles")
    cache_dir = Path(cache_root) if cache_root else Path("data/cache/options")
    cache_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    seen: set[str] = set()

    for symbol in symbols:
        if not symbol:
            continue
        symbol_upper = symbol.strip().upper()
        if not symbol_upper or symbol_upper in seen:
            continue
        seen.add(symbol_upper)

        cache_path = cache_dir / f"{symbol_upper}.parquet"
        if rebuild_missing and not cache_path.exists():
            build_option_cache(
                symbol_upper,
                start_date,
                end_date,
                flatfile_root=storage_root,
                cache_root=cache_dir,
            )

        if not cache_path.exists():
            LOGGER.info(
                "Option cache missing; returning empty frame",
                extra={"symbol": symbol_upper, "path": str(cache_path)},
            )
            continue

        try:
            df_cached = pd.read_parquet(cache_path)
        except Exception as exc:  # pragma: no cover - IO/parquet errors
            LOGGER.warning("Failed to read option cache", extra={"symbol": symbol_upper, "error": str(exc)})
            continue

        if df_cached.empty:
            continue

        if "window_start" not in df_cached.columns:
            LOGGER.warning(
                "Option cache missing window_start column",
                extra={"symbol": symbol_upper, "columns": list(df_cached.columns)},
            )
            continue

        windows = pd.to_datetime(df_cached["window_start"], errors="coerce")
        mask = windows.notna()
        mask &= windows >= pd.Timestamp(start_date)
        mask &= windows <= pd.Timestamp(end_date)
        filtered = df_cached.loc[mask].copy()
        if filtered.empty:
            continue

        filtered["window_start"] = windows.loc[filtered.index]
        filtered = filtered.sort_values(["window_start", "ticker"]).reset_index(drop=True)
        frames.append(filtered[OPTION_COLUMNS])

    if not frames:
        return pd.DataFrame(columns=OPTION_COLUMNS)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["ticker", "window_start"], keep="last")
    return combined.sort_values(["window_start", "ticker"]).reset_index(drop=True)


__all__ = [
    "build_ticker_cache",
    "load_ticker_data",
    "build_option_cache",
    "load_option_data",
]
