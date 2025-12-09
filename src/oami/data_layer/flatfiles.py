"""Flatfile ingestion utilities for Polygon datasets.

This module centralises all interactions with Polygon's S3-hosted flatfiles.
It is responsible for downloading stock and options aggregates, hydrating local
cache directories, and exposing helpers that higher-level data components rely
on.  The functions here are intentionally lightweight wrappers around the AWS
CLI so they can operate in restricted environments where a full S3 SDK may not
be available.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path
from time import perf_counter
from typing import Iterable

import pandas as pd

from oami.utils import cache_manager
from oami.utils.cache_manager import parse_occ_ticker

LOGGER = logging.getLogger(__name__)

OCC_TICKER_RE = re.compile(r"O:([A-Z]{1,6})(\d{2})(\d{2})(\d{2})([CP])(\d{8})")


def _resolve_client_class() -> type["PolygonFlatfileClient"]:
    """Return the active ``PolygonFlatfileClient`` implementation.

    Tests sometimes monkeypatch the class exported by ``oami.data_layer`` so we
    consult the package first, falling back to the local definition if the
    package has not been manipulated.
    """
    dl = sys.modules.get("oami.data_layer")
    if dl is not None:
        client_cls = getattr(dl, "PolygonFlatfileClient", None)
        if client_cls is not None:
            return client_cls  # type: ignore[return-value]
    return PolygonFlatfileClient


def _progress_iter(iterable: Iterable, *, desc: str, total: int | None = None):
    """Yield items from ``iterable`` while emitting a simple terminal progress indicator."""

    items = list(iterable)
    total = total if total is not None else len(items)
    start = perf_counter()
    if total == 0:
        sys.stdout.write(f"{desc}: 0/0 [0.00s]\n")
        sys.stdout.flush()
        return

    for idx, item in enumerate(items, start=1):
        if total:
            pct = 100.0 * idx / total
            sys.stdout.write(f"\r{desc}: {idx}/{total} ({pct:5.1f}%)")
        else:
            sys.stdout.write(f"\r{desc}: {idx}")
        sys.stdout.flush()
        yield item

    elapsed = perf_counter() - start
    if total:
        sys.stdout.write(f"\r{desc}: {total}/{total} (100.0%) [{elapsed:.2f}s]\n")
    else:
        sys.stdout.write(f"\r{desc}: {idx} [{elapsed:.2f}s]\n")
    sys.stdout.flush()


def _download_stocks_flatfile(
    start: str = "2020-01-01", end: str | None = None
) -> dict[str, list[str]]:
    """Download Polygon SIP trade flatfiles for the supplied date window."""

    def _parse_date(value: str | date) -> date:
        if isinstance(value, date):
            return value
        try:
            return datetime.strptime(str(value), "%Y-%m-%d").date()
        except (TypeError, ValueError) as exc:
            raise ValueError("Dates must be provided in YYYY-MM-DD format") from exc

    start_date = _parse_date(start)
    end_date = _parse_date(end) if end else date.today()
    if start_date > end_date:
        raise ValueError("Parameter 'start' must be on or before 'end'")

    month_tokens: list[tuple[int, int]] = []
    cursor = date(start_date.year, start_date.month, 1)
    end_anchor = date(end_date.year, end_date.month, 1)
    while cursor <= end_anchor:
        month_tokens.append((cursor.year, cursor.month))
        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)

    local_root = Path("data/flatfiles/us_stocks_sip/day_aggs_v1")
    downloaded: dict[str, list[str]] = {}
    flatfile_client: PolygonFlatfileClient | None = None

    if not month_tokens:
        return downloaded

    for year, month in _progress_iter(
        month_tokens,
        desc="Downloading stock flatfiles",
        total=len(month_tokens),
    ):
        month_str = f"{month:02d}"
        month_key = f"{year}-{month_str}"
        expected_dir = (local_root / f"{year:04d}" / month_str).resolve()
        expected_dir.mkdir(parents=True, exist_ok=True)

        if flatfile_client is None:
            try:
                flatfile_client = _resolve_client_class()()
            except RuntimeError as exc:
                LOGGER.error("Failed to initialize PolygonFlatfileClient", extra={"error": str(exc)})
                downloaded[month_key] = []
                continue

        try:
            destination = flatfile_client.download_flatfile(
                data_type="stocks",
                timeframe="day",
                year=year,
                month=month,
            )
        except RuntimeError as exc:
            LOGGER.warning(
                "Polygon stock flatfile download failed",
                extra={"year": year, "month": month, "error": str(exc)},
            )
            downloaded[month_key] = []
            continue

        target_dir = destination if destination is not None else expected_dir
        target_dir = target_dir.resolve()
        if target_dir.is_file():
            target_dir = target_dir.parent
        if not target_dir.exists():
            LOGGER.warning(
                "Polygon stock flatfile directory missing after download",
                extra={"expected": str(target_dir)},
            )
            downloaded[month_key] = []
            continue

        filenames: list[str] = []
        for file_path in sorted(target_dir.iterdir()):
            if not file_path.is_file():
                continue
            trade_date = _parse_stock_flatfile_date(file_path.name)
            if trade_date is None:
                LOGGER.debug(
                    "Skipping unrecognized Polygon flatfile name",
                    extra={"path": str(file_path)},
                )
                continue
            if trade_date < start_date or trade_date > end_date:
                continue
            filenames.append(file_path.name)

        downloaded[month_key] = filenames

    return downloaded


def _parse_stock_flatfile_date(filename: str) -> date | None:
    """Extract the trading date embedded in the Polygon stock flatfile name."""

    token = Path(filename).name
    match = re.match(r"(\d{4}-\d{2}-\d{2})", token)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y-%m-%d").date()
    except ValueError:
        return None


def _subtract_years(anchor: date, years: int) -> date:
    """Return the calendar date ``years`` prior to ``anchor."""

    try:
        return anchor.replace(year=anchor.year - years)
    except ValueError:
        return anchor.replace(year=anchor.year - years, month=2, day=28)


def _load_stock_flatfile(path: Path) -> pd.DataFrame:
    """Load a Polygon stock flatfile into memory as a DataFrame."""

    if not path.exists():
        LOGGER.warning("Stock flatfile missing locally", extra={"path": str(path)})
        return pd.DataFrame()

    try:
        suffixes = path.suffixes
        if suffixes[-2:] == [".csv", ".gz"] or suffixes[-1] == ".csv":
            frame = pd.read_csv(path)
        elif suffixes and suffixes[-1] in {".parquet", ".pq"}:
            frame = pd.read_parquet(path)
        else:
            LOGGER.warning(
                "Unsupported stock flatfile extension",
                extra={"path": str(path), "suffixes": suffixes},
            )
            return pd.DataFrame()
    except Exception as exc:  # pragma: no cover - IO errors
        LOGGER.error(
            "Failed to read stock flatfile",
            extra={"path": str(path), "error": str(exc)},
        )
        return pd.DataFrame()
    return frame


def _filter_stock_flatfile_frame(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Filter a stock flatfile frame down to the requested ticker symbol."""

    candidate_cols = [col for col in ("ticker", "symbol", "T") if col in df.columns]
    if not candidate_cols:
        LOGGER.warning(
            "Stock flatfile missing ticker column",
            extra={"columns": list(df.columns)},
        )
        return pd.DataFrame()

    column = candidate_cols[0]
    ticker_upper = ticker.upper()
    filtered = df[df[column].astype(str).str.upper() == ticker_upper]
    if filtered.empty:
        return pd.DataFrame()
    return filtered.copy()


class PolygonFlatfileClient:
    """Simple client for Polygon flat files stored on S3-compatible storage."""

    def __init__(self) -> None:
        self._logger = logging.getLogger(__name__)
        self._access_key = os.getenv("AWS_ACCESS_KEY_ID")
        self._secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        if not self._access_key or not self._secret_key:
            raise RuntimeError("AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set in the environment")
        self._configure_cli()

    def _configure_cli(self) -> None:
        commands = [
            (["aws", "configure", "set", "aws_access_key_id", self._access_key], "aws_access_key_id"),
            (["aws", "configure", "set", "aws_secret_access_key", self._secret_key], "aws_secret_access_key"),
        ]
        for cmd, key_name in commands:
            try:
                subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                self._logger.info("Configured AWS CLI credential for %s", key_name)
            except FileNotFoundError as exc:
                raise RuntimeError("AWS CLI is not installed or not available in PATH") from exc
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(f"AWS CLI command failed: {exc.stderr.strip()}") from exc

    def list_root_directories(self) -> list[str]:
        """Return the directories present at the Polygon flatfiles root."""

        try:
            completed = subprocess.run(
                [
                    "aws",
                    "s3",
                    "ls",
                    "s3://flatfiles/",
                    "--endpoint-url",
                    "https://files.polygon.io",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("AWS CLI is not installed or not available in PATH") from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"AWS CLI command failed: {exc.stderr.strip()}") from exc

        directories: list[str] = []
        for line in completed.stdout.splitlines():
            entry = line.strip()
            if entry.endswith("/"):
                directories.append(entry.split()[-1])
        self._logger.info("Polygon flatfiles root entries: %s", directories)
        return directories

    def download_flatfile(
        self,
        data_type: str,
        year: int,
        month: int,
        day: int | None = None,
        timeframe: str = "day",
        available_files: set[str] | None = None,
    ) -> Path | None:
        """Download flatfile data to ``data/flatfiles`` for the requested path."""

        type_map = {
            "stocks": "us_stocks_sip",
            "options": "us_options_opra",
        }
        timeframe_map = {
            "stocks": {"day": "day_aggs_v1"},
            "options": {"day": "day_aggs_v1"},
        }
        try:
            type_key = data_type.lower()
            root_dir = type_map[type_key]
        except KeyError as exc:
            raise ValueError(f"Unsupported data_type '{data_type}'. Expected one of {list(type_map)}") from exc

        timeframe_key = timeframe.lower()
        try:
            timeframe_dir = timeframe_map[type_key][timeframe_key]
        except KeyError as exc:
            expected = list(timeframe_map.get(type_key, {}))
            raise ValueError(f"Unsupported timeframe '{timeframe}'. Expected one of {expected}") from exc

        year_str = f"{int(year):04d}"
        month_str = f"{int(month):02d}"
        base_s3 = f"s3://flatfiles/{root_dir}/{timeframe_dir}/{year_str}/{month_str}"
        local_root_path = Path("data/flatfiles") / root_dir / timeframe_dir / year_str / month_str
        local_root_path.mkdir(parents=True, exist_ok=True)
        local_root = local_root_path.resolve()

        if available_files is not None:
            try:
                file_candidates = sorted(str(name) for name in available_files)
            except TypeError:
                file_candidates = sorted(available_files)
        else:
            file_candidates = sorted(
                self.list_files_for_month(
                    data_type=data_type,
                    year=year,
                    month=month,
                    timeframe=timeframe,
                )
            )

        if day is not None:
            day_str = f"{int(day):02d}"
            expected_name = f"{year_str}-{month_str}-{day_str}.csv.gz"
            file_candidates = [expected_name]

        if not file_candidates:
            self._logger.warning(
                "No Polygon flatfiles available for download",
                extra={"data_type": data_type, "year": year, "month": month, "timeframe": timeframe},
            )
            return None

        downloaded_targets: list[Path] = []
        last_destination: Path | None = None
        for filename in _progress_iter(
            file_candidates,
            desc=f"Downloading {data_type.lower()} {year_str}-{month_str}",
            total=len(file_candidates),
        ):
            source = f"{base_s3}/{filename}"
            destination = local_root / filename
            if available_files is not None and filename not in available_files:
                continue

            if destination.exists():
                self._logger.info(
                    "Polygon flatfile already present locally; skipping download",
                    extra={"flatfile_name": filename},
                )
                downloaded_targets.append(destination)
                last_destination = destination
                continue

            cmd = [
                "aws",
                "s3",
                "cp",
                source,
                str(destination),
                "--endpoint-url",
                "https://files.polygon.io",
            ]

            self._logger.debug("Executing AWS CLI command", extra={"aws_cli_cmd": cmd})
            try:
                completed = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except FileNotFoundError as exc:
                raise RuntimeError("AWS CLI is not installed or not available in PATH") from exc
            except subprocess.CalledProcessError as exc:
                self._logger.warning(
                    "AWS CLI command failed",
                    extra={
                        "flatfile_name": filename,
                        "error": exc.stderr.strip() if exc.stderr else str(exc),
                    },
                )
                continue

            self._logger.info("Polygon flatfiles download complete", extra={"flatfile_name": filename})
            if completed.stdout.strip():
                self._logger.debug("AWS CLI output:\n%s", completed.stdout.strip())
            if completed.stderr.strip():
                self._logger.warning("AWS CLI stderr:\n%s", completed.stderr.strip())
            downloaded_targets.append(destination)
            last_destination = destination

        if not downloaded_targets:
            self._logger.warning(
                "No Polygon flatfiles downloaded for month",
                extra={"year": year, "month": month, "data_type": data_type, "timeframe": timeframe},
            )
            return local_root if day is None else None

        if day is None:
            return local_root
        return last_destination

    def list_files_for_month(
        self,
        data_type: str,
        year: int,
        month: int,
        timeframe: str = "day",
    ) -> set[str]:
        """List available files for the specified month on Polygon flatfiles."""

        type_map = {
            "stocks": "us_stocks_sip",
            "options": "us_options_opra",
        }
        timeframe_map = {
            "stocks": {"day": "day_aggs_v1"},
            "options": {"day": "day_aggs_v1"},
        }

        try:
            type_key = data_type.lower()
            root_dir = type_map[type_key]
        except KeyError as exc:
            raise ValueError(f"Unsupported data_type '{data_type}'. Expected one of {list(type_map)}") from exc
        timeframe_key = timeframe.lower()
        try:
            timeframe_dir = timeframe_map[type_key][timeframe_key]
        except KeyError as exc:
            expected = list(timeframe_map.get(type_key, {}))
            raise ValueError(f"Unsupported timeframe '{timeframe}'. Expected one of {expected}") from exc

        year_str = f"{int(year):04d}"
        month_str = f"{int(month):02d}"
        s3_path = f"s3://flatfiles/{root_dir}/{timeframe_dir}/{year_str}/{month_str}/"

        try:
            completed = subprocess.run(
                [
                    "aws",
                    "s3",
                    "ls",
                    s3_path,
                    "--endpoint-url",
                    "https://files.polygon.io",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("AWS CLI is not installed or not available in PATH") from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"AWS CLI command failed: {exc.stderr.strip()}") from exc

        files: set[str] = set()
        for line in completed.stdout.splitlines():
            entry = line.strip()
            if not entry or entry.endswith("/"):
                continue
            token = entry.split()[-1]
            files.add(token)
        self._logger.info(
            "Polygon flatfiles month listing",
            extra={"path": s3_path, "files": sorted(files)},
        )
        return files


def get_option_flatfile_data(
    underlying_ticker: str,
    as_of: str | None = None,
    as_to: str | None = None,
    expiration_start: str | None = None,
    expiration_end: str | None = None,
    strike_lower: float | None = None,
    strike_upper: float | None = None,
    contract_type: str | None = None,
    base_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Collect Polygon option flatfiles and filter contracts by strikes, expiries, and type.

    Parameters
    ----------
    underlying_ticker : str
        Underlying ticker symbol (e.g. "SPY").
    as_of, as_to : str or None, optional
        Inclusive flatfile download window in ISO format. Defaults to the last two years when omitted.
    expiration_start, expiration_end : str or None, optional
        Optional inclusive expiration-date filter.
    strike_lower, strike_upper : float or None, optional
        Optional strike-price bounds.
    contract_type : str or None, optional
        Contract type filter ("CALL" or "PUT").
    base_dir : pathlib.Path or None, optional
        Override for the local flatfile cache root. Defaults to ``data/flatfiles``.

    Returns
    -------
    pandas.DataFrame
        Aggregated option rows satisfying the supplied filters. Empty when no contracts match."""
    """Collect Polygon option flatfiles and filter contracts by strikes, expiries, and type."""

    today = date.today()
    default_start = _subtract_years(today, 2)

    try:
        file_start = pd.to_datetime(as_of).date() if as_of else default_start
        file_end = pd.to_datetime(as_to).date() if as_to else today
    except (TypeError, ValueError) as exc:
        raise ValueError("as_of/as_to must be parseable as YYYY-MM-DD dates") from exc

    if file_start > file_end:
        raise ValueError("as_of must be on or before as_to")

    exp_start_dt = pd.to_datetime(expiration_start).date() if expiration_start else None
    exp_end_dt = pd.to_datetime(expiration_end).date() if expiration_end else None
    if exp_start_dt and exp_end_dt and exp_start_dt > exp_end_dt:
        raise ValueError("expiration_start must be on or before expiration_end")

    dates = pd.date_range(start=file_start, end=file_end, freq="D")
    frames: list[pd.DataFrame] = []
    storage_root = Path(base_dir) if base_dir else Path("data/flatfiles")

    flatfile_client: PolygonFlatfileClient | None = None
    remote_listing_cache: dict[tuple[int, int], set[str]] = {}

    date_list = list(dates)
    if not date_list:
        return pd.DataFrame()

    existing_frames: dict[pd.Timestamp, pd.DataFrame] = {}
    missing_dates: list[pd.Timestamp] = []

    for expiration in date_list:
        local_path = _option_flatfile_path(expiration, storage_root)
        if not local_path.exists():
            missing_dates.append(expiration)
            continue
        df_cached = _load_option_flatfile_for_date(expiration, storage_root)
        if df_cached.empty:
            missing_dates.append(expiration)
            continue
        existing_frames[expiration] = df_cached

    if missing_dates:
        if flatfile_client is None:
            try:
                flatfile_client = _resolve_client_class()()
            except RuntimeError as exc:
                LOGGER.warning("Unable to initialize PolygonFlatfileClient", extra={"error": str(exc)})
                missing_dates = []
        if missing_dates:
            grouped_missing: dict[tuple[int, int], list[pd.Timestamp]] = {}
            for expiration in missing_dates:
                grouped_missing.setdefault((expiration.year, expiration.month), []).append(expiration)

            month_order = sorted(grouped_missing.keys())
            for year_month in _progress_iter(
                month_order,
                desc=f"Downloading option flatfiles {underlying_ticker.upper()}",
                total=len(month_order),
            ):
                year, month = year_month
                month_dates = grouped_missing[year_month]
                month_key = (year, month)
                if month_key not in remote_listing_cache:
                    try:
                        remote_listing_cache[month_key] = flatfile_client.list_files_for_month(
                            "options",
                            year,
                            month,
                        )
                    except RuntimeError as exc:
                        LOGGER.warning(
                            "Failed to list Polygon flatfiles",
                            extra={
                                "year": year,
                                "month": month,
                                "error": str(exc),
                            },
                        )
                        continue

                available_files = remote_listing_cache.get(month_key, set())
                for expiration in month_dates:
                    path = _option_flatfile_path(expiration, storage_root)
                    filename = path.name
                    if filename not in available_files:
                        LOGGER.info(
                            "Skipping download; file missing from Polygon flatfiles",
                            extra={
                                "flatfile_name": filename,
                                "year": expiration.year,
                                "month": expiration.month,
                            },
                        )
                        continue
                    try:
                        flatfile_client.download_flatfile(
                            "options",
                            expiration.year,
                            expiration.month,
                            day=expiration.day,
                            available_files=available_files,
                        )
                    except RuntimeError as exc:
                        LOGGER.warning(
                            "Polygon flatfile download failed",
                            extra={"date": expiration.date().isoformat(), "error": str(exc)},
                        )
                        continue
                    df_downloaded = _load_option_flatfile_for_date(expiration, storage_root)
                    if not df_downloaded.empty:
                        existing_frames[expiration] = df_downloaded

    for expiration in date_list:
        df = existing_frames.get(expiration)
        if df is None or df.empty:
            continue
        filtered = _filter_flatfile_frame(
            df,
            underlying_ticker=underlying_ticker,
            strike_lower=strike_lower,
            strike_upper=strike_upper,
            contract_type=contract_type,
            expiration_start=exp_start_dt,
            expiration_end=exp_end_dt,
        )
        if filtered.empty:
            continue
        frames.append(filtered)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def get_stock_flatfile_data(
    ticker: str, start: str = "2020-01-01", end: str | None = None
) -> pd.DataFrame:
    """Download and aggregate Polygon daily stock flatfiles for a ticker.

    Parameters
    ----------
    ticker : str
        Underlying ticker symbol (e.g. "SPY").
    start : str, default "2020-01-01"
        Inclusive start date in ISO format. Clamped to Polygon's five-year mirror if older.
    end : str or None, default None
        Inclusive end date in ISO format. Defaults to today when omitted.

    Returns
    -------
    pandas.DataFrame
        Concatenated daily stock aggregates (OHLCV plus metadata). The frame is empty when
        no data is available within the requested window..
    """

    try:
        start_date = pd.to_datetime(start).date()
        end_date = pd.to_datetime(end).date() if end else date.today()
    except (TypeError, ValueError) as exc:
        raise ValueError("start/end must be parseable as YYYY-MM-DD dates") from exc

    if start_date > end_date:
        raise ValueError("Parameter 'start' must be on or before 'end'")

    lookback_floor = _subtract_years(date.today(), 5)
    if start_date < lookback_floor:
        LOGGER.info(
            "Clamping stock flatfile start date to %s",
            lookback_floor.isoformat(),
        )
        start_date = lookback_floor
        if start_date > end_date:
            raise ValueError("Adjusted 'start' exceeds 'end'; narrow the requested window")

    download_map = _download_stocks_flatfile(
        start=start_date.isoformat(),
        end=end_date.isoformat(),
    )
    if not download_map:
        return pd.DataFrame()

    local_root = Path("data/flatfiles/us_stocks_sip/day_aggs_v1")
    frames: list[pd.DataFrame] = []

    month_keys = sorted(download_map.keys())
    for month_key in month_keys:
        try:
            year_str, month_str = month_key.split("-", maxsplit=1)
        except ValueError:
            LOGGER.warning(
                "Unexpected month key returned from download map",
                extra={"key": month_key},
            )
            continue

        month_path = local_root / year_str / month_str
        if not month_path.exists():
            LOGGER.warning(
                "Month directory missing after download",
                extra={"path": str(month_path)},
            )
            continue

        files = sorted(path for path in month_path.iterdir() if path.is_file())
        if not files:
            LOGGER.warning(
                "No stock flatfiles present in month directory",
                extra={"path": str(month_path)},
            )
            continue

        for file_path in files:
            trade_date = _parse_stock_flatfile_date(file_path.name)
            if trade_date is None or trade_date < start_date or trade_date > end_date:
                continue
            frame = _load_stock_flatfile(file_path)
            if frame.empty:
                continue
            filtered = _filter_stock_flatfile_frame(frame, ticker)
            if filtered.empty:
                continue
            frames.append(filtered)

    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    if "window_start" in result.columns:
        result["window_start"] = (
            pd.to_datetime(result["window_start"], errors="coerce", utc=True)
            .dt.tz_convert(None)
            .dt.date
        )
    return result


def _option_flatfile_path(expiration: pd.Timestamp, base_dir: Path) -> Path:
    year_str = f"{expiration.year:04d}"
    month_str = f"{expiration.month:02d}"
    filename = f"{expiration.year:04d}-{expiration.month:02d}-{expiration.day:02d}.csv.gz"
    return base_dir / "us_options_opra" / "day_aggs_v1" / year_str / month_str / filename


def _load_option_flatfile_for_date(expiration: pd.Timestamp, base_dir: Path) -> pd.DataFrame:
    path = _option_flatfile_path(expiration, base_dir)
    if not path.exists():
        logging.warning("Option flatfile not found", extra={"path": str(path)})
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - IO errors
        logging.error("Failed to read option flatfile", extra={"path": str(path), "error": str(exc)})
        return pd.DataFrame()
    if "ticker" in df.columns:
        metadata = df["ticker"].astype(str).apply(_parse_option_ticker_metadata)
        df["underlying_symbol"] = metadata.apply(lambda meta: meta["underlying"] if meta else None)
        df["strike_price"] = metadata.apply(lambda meta: meta["strike"] if meta else None)
        df["contract_type"] = metadata.apply(lambda meta: meta["contract_type"] if meta else None)
        df["expiration_date"] = metadata.apply(
            lambda meta: meta["expiration"] if meta and meta["expiration"] is not None else expiration.date()
        )
    else:  # pragma: no cover - safety for unexpected schema
        df["underlying_symbol"] = None
        df["strike_price"] = None
        df["contract_type"] = None
        df["expiration_date"] = expiration.date()
    if "window_start" in df.columns:
        df["window_start"] = (
            pd.to_datetime(df["window_start"], errors="coerce", utc=True)
            .dt.tz_convert(None)
            .dt.floor("min")
            .dt.date
        )
    return df


def _filter_flatfile_frame(
    df: pd.DataFrame,
    *,
    underlying_ticker: str,
    strike_lower: float | None,
    strike_upper: float | None,
    contract_type: str | None,
    expiration_start: date | None,
    expiration_end: date | None,
) -> pd.DataFrame:
    frame = df.copy()
    target_symbol = underlying_ticker.upper()

    if "underlying_symbol" not in frame.columns:
        logging.warning("Flatfile frame missing derived underlying column", extra={"columns": list(frame.columns)})
        return pd.DataFrame()

    frame = frame[frame["underlying_symbol"].astype(str).str.upper() == target_symbol]

    if expiration_start is not None or expiration_end is not None:
        if "expiration_date" in frame.columns:
            exp_dates = pd.to_datetime(frame["expiration_date"], errors="coerce").dt.date
            frame = frame.assign(_exp_date=exp_dates)
            if expiration_start is not None:
                frame = frame[frame["_exp_date"] >= expiration_start]
            if expiration_end is not None:
                frame = frame[frame["_exp_date"] <= expiration_end]
            frame = frame.drop(columns="_exp_date")
        else:
            logging.warning("Flatfile frame missing expiration_date column", extra={"columns": list(frame.columns)})
            frame = pd.DataFrame()

    if "strike_price" in frame.columns:
        frame = frame.assign(strike_price=pd.to_numeric(frame["strike_price"], errors="coerce"))
        if strike_lower is not None:
            frame = frame[frame["strike_price"] >= strike_lower]
        if strike_upper is not None:
            frame = frame[frame["strike_price"] <= strike_upper]

    if contract_type:
        normalized = contract_type.lower()
        if "contract_type" in frame.columns:
            frame = frame[frame["contract_type"].astype(str).str.lower() == normalized]

    return frame.reset_index(drop=True)


def _parse_option_ticker_metadata(ticker: str) -> dict[str, object] | None:
    match = OCC_TICKER_RE.match(ticker)
    if not match:
        return None
    try:
        info = parse_occ_ticker(ticker)
        expiration = date(info["year"], cache_manager.MONTHS.index(info["month"]) + 1, info["day"])
    except Exception:
        expiration = None
        info = {"underlying": match.group(1)}
    strike = int(match.group(6)) / 1000.0
    contract_type = "call" if match.group(5).upper() == "C" else "put"
    return {
        "underlying": info["underlying"].upper(),
        "strike": strike,
        "contract_type": contract_type,
        "expiration": expiration,
    }


__all__ = [
    "PolygonFlatfileClient",
    "_download_stocks_flatfile",
    "_filter_flatfile_frame",
    "_filter_stock_flatfile_frame",
    "_option_flatfile_path",
    "_parse_option_ticker_metadata",
    "_parse_stock_flatfile_date",
    "_progress_iter",
    "_subtract_years",
    "get_option_flatfile_data",
    "get_stock_flatfile_data",
]
