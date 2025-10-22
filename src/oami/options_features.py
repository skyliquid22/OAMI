"""
Options Sentiment Feature Engineering
=====================================

Utilities for converting raw per-contract option OHLCV data into aggregated
sentiment signals that can be merged with market features.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency at runtime
    from numpy.typing import ArrayLike
except ImportError:  # pragma: no cover
    ArrayLike = Iterable[float]  # type: ignore

from oami.utils.cache_manager import parse_occ_ticker


# --------------------------------------------------------------------------- #
# Data structures and helpers
# --------------------------------------------------------------------------- #

@dataclass
class _OptionRowMetadata:
    """Container holding per-contract metadata needed for feature engineering."""

    ticker: str | None
    contract_type: str | None
    strike: float | None
    expiration: pd.Timestamp | None
    underlying: str | None
    spot_price: float | None


def _normalize_contract_type(value: str | None, ticker: str | None) -> str | None:
    """Return canonical ``CALL`` / ``PUT`` labels."""
    if isinstance(value, str) and value:
        upper = value.upper()
        if upper.startswith("C"):
            return "CALL"
        if upper.startswith("P"):
            return "PUT"

    if isinstance(ticker, str):
        match = re.match(r"O:[A-Z]{1,6}\d{6}([CP])\d{8}", ticker)
        if match:
            return "CALL" if match.group(1) == "C" else "PUT"
    return None


def _extract_metadata(row: pd.Series) -> _OptionRowMetadata:
    """Parse metadata for a single options row."""
    ticker = row.get("ticker")
    contract_type = _normalize_contract_type(row.get("contract_type"), ticker)
    strike = row.get("strike_price") or row.get("strike")
    expiration = row.get("expiration_date") or row.get("expiration")
    expiration_ts = pd.to_datetime(expiration).tz_localize(None) if pd.notna(expiration) else None

    underlying = row.get("underlying")
    spot_price = row.get("underlying_price") or row.get("spot_price") or row.get("spot")

    if ticker and contract_type is None:
        try:
            info = parse_occ_ticker(ticker)
            match_type = re.match(r"O:[A-Z]{1,6}\d{6}([CP])\d{8}", ticker)
            if match_type:
                contract_type = "CALL" if match_type.group(1) == "C" else "PUT"
            underlying = underlying or info.get("underlying")
            if expiration_ts is None:
                year = info.get("year")
                month = info.get("month")
                day = info.get("day")
                try:
                    expiration_ts = pd.to_datetime(f"{day} {month} {year}", errors="coerce")
                except Exception:  # pragma: no cover
                    expiration_ts = pd.NaT
        except Exception:  # pragma: no cover - defensive
            contract_type = contract_type

    return _OptionRowMetadata(
        ticker=str(ticker) if ticker else None,
        contract_type=contract_type,
        strike=float(strike) if pd.notna(strike) else None,
        expiration=expiration_ts,
        underlying=underlying,
        spot_price=float(spot_price) if pd.notna(spot_price) else None,
    )


def _weighted_average(values: ArrayLike, weights: ArrayLike) -> float:
    """Return the weighted mean ignoring NaNs."""
    v = np.asarray(values, dtype="float64")
    w = np.asarray(weights, dtype="float64")
    mask = ~np.isnan(v) & ~np.isnan(w) & (w > 0)
    if not mask.any():
        return float("nan")
    return float(np.average(v[mask], weights=w[mask]))


def _zscore(series: pd.Series) -> pd.Series:
    """Standardise a series while ignoring NaNs."""
    mean = series.mean(skipna=True)
    std = series.std(skipna=True, ddof=0)
    if std is None or math.isclose(std, 0.0, abs_tol=1e-12) or np.isnan(std):
        return pd.Series(np.zeros(len(series)), index=series.index, dtype="float64")
    return (series - mean) / std


# --------------------------------------------------------------------------- #
# Core feature construction
# --------------------------------------------------------------------------- #

def build_options_sentiment(
    df_options: pd.DataFrame,
    *,
    near_term_days: int = 7,
    far_term_days: int = 30,
) -> pd.DataFrame:
    """Aggregate per-contract OHLCV data into daily option sentiment features.

    Parameters
    ----------
    df_options : pandas.DataFrame
        DataFrame where each row represents an options contract containing an
        ``ohlcv`` column with a per-contract DataFrame of Polygon aggregates.
        Additional columns such as ``ticker``, ``contract_type``, ``strike_price``,
        and ``expiration_date`` are used when present.
    near_term_days : int, default=7
        Threshold (in calendar days) distinguishing near-expiry contracts.
    far_term_days : int, default=30
        Threshold (in calendar days) distinguishing far-expiry contracts.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by ``Date`` containing option sentiment features such
        as Put/Call Volume Ratio, Weighted Average Moneyness, and Term Structure
        Slope. The frame is empty if no usable option data is provided.
    """
    if df_options is None or df_options.empty:
        return pd.DataFrame(
            columns=[
                "Date",
                "put_call_volume_ratio",
                "weighted_average_moneyness",
                "implied_direction_bias",
                "skewed_volume_distribution",
                "near_far_expiry_vol_ratio",
                "realized_vol_proxy",
                "high_low_spread_ratio",
                "term_structure_slope",
                "total_volume",
                "call_volume",
                "put_volume",
                "avg_option_return",
                "call_range_mean",
                "put_range_mean",
            ]
        )

    expanded_rows: List[pd.DataFrame] = []
    for _, row in df_options.iterrows():
        ohlcv = row.get("ohlcv")
        if not isinstance(ohlcv, pd.DataFrame) or ohlcv.empty:
            continue

        meta = _extract_metadata(row)
        local = ohlcv.copy()

        # Normalise time index.
        if isinstance(local.index, pd.DatetimeIndex):
            local_index = local.index
        elif "t" in local.columns:
            local_index = pd.to_datetime(local["t"], errors="coerce")
            local = local.drop(columns=["t"])
        else:
            local_index = pd.to_datetime(local.index, errors="coerce")

        local_index = pd.DatetimeIndex(local_index).tz_localize(None)
        local = local.set_index(local_index)
        local = local[~local.index.isna()]

        # Normalise OHLCV schema.
        rename_map = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
        local = local.rename(columns=rename_map)
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in local.columns:
                local[col] = np.nan

        local = local[["open", "high", "low", "close", "volume"]]
        local["Date"] = local.index.normalize()
        local["ticker"] = meta.ticker
        local["contract_type"] = meta.contract_type
        local["strike"] = meta.strike
        local["expiration"] = meta.expiration
        local["underlying"] = meta.underlying
        local["spot_price"] = meta.spot_price
        expanded_rows.append(local.reset_index(drop=True))

    if not expanded_rows:
        return pd.DataFrame(
            columns=[
                "Date",
                "put_call_volume_ratio",
                "weighted_average_moneyness",
                "implied_direction_bias",
                "skewed_volume_distribution",
                "near_far_expiry_vol_ratio",
                "realized_vol_proxy",
                "high_low_spread_ratio",
                "term_structure_slope",
                "total_volume",
                "call_volume",
                "put_volume",
                "avg_option_return",
                "call_range_mean",
                "put_range_mean",
            ]
        )

    expanded = pd.concat(expanded_rows, ignore_index=True)
    if expanded.empty:
        return pd.DataFrame()

    expanded["contract_type"] = expanded.apply(
        lambda r: _normalize_contract_type(r.get("contract_type"), r.get("ticker")),
        axis=1,
    )

    expanded["range"] = expanded["high"] - expanded["low"]
    expanded["mid_price"] = expanded[["open", "high", "low", "close"]].mean(axis=1)
    expanded["Date"] = pd.to_datetime(expanded["Date"]).dt.normalize()
    expanded["expiration"] = pd.to_datetime(expanded["expiration"]).dt.normalize()
    expanded = expanded.sort_values(["ticker", "Date"])
    expanded["option_return"] = expanded.groupby("ticker")["close"].pct_change()
    expanded["days_to_expiry"] = (
        (expanded["expiration"] - expanded["Date"]).dt.days
        if expanded["expiration"].notna().any()
        else np.nan
    )

    feature_rows: List[dict[str, float | pd.Timestamp]] = []

    grouped = expanded.groupby("Date")
    for date, group in grouped:
        call_mask = group["contract_type"] == "CALL"
        put_mask = group["contract_type"] == "PUT"

        call_vol = group.loc[call_mask, "volume"].sum()
        put_vol = group.loc[put_mask, "volume"].sum()
        total_vol = call_vol + put_vol

        pcr = (put_vol / call_vol) if call_vol > 0 else np.nan
        implied_direction_bias = (
            (call_vol - put_vol) / total_vol if total_vol > 0 else np.nan
        )

        spot_series = group["spot_price"].dropna()
        if not spot_series.empty:
            spot_value = spot_series.mean()
            strike_ref = spot_value
            moneyness_values = (spot_value - group["strike"]) / spot_value
        else:
            strike_ref = group["strike"].median()
            moneyness_values = (group["mid_price"] - group["strike"]) / group["strike"]
        weighted_avg_moneyness = (
            _weighted_average(moneyness_values, group["volume"])
            if total_vol > 0
            else np.nan
        )

        strike_reference = strike_ref if not np.isnan(strike_ref) else group["strike"].median()
        call_otm = group.loc[call_mask & (group["strike"] > strike_reference), "volume"].sum()
        put_otm = group.loc[put_mask & (group["strike"] < strike_reference), "volume"].sum()
        call_itm = call_vol - call_otm
        put_itm = put_vol - put_otm
        skewed_volume_distribution = (
            ((call_otm + put_otm) - (call_itm + put_itm)) / total_vol
            if total_vol > 0
            else np.nan
        )

        if group["days_to_expiry"].notna().any():
            near_volume = group.loc[group["days_to_expiry"] <= near_term_days, "volume"].sum()
            far_volume = group.loc[group["days_to_expiry"] >= far_term_days, "volume"].sum()
            near_far_expiry_ratio = (
                near_volume / far_volume if far_volume > 0 else np.nan
            )
        else:
            near_volume = far_volume = np.nan
            near_far_expiry_ratio = np.nan

        realized_vol_proxy = _weighted_average(
            np.log(group["high"] / group["low"]).replace([np.inf, -np.inf], np.nan),
            group["volume"],
        )
        high_low_spread_ratio = _weighted_average(
            (group["high"] - group["low"]) / group["close"],
            group["volume"],
        )

        near_close = _weighted_average(
            group.loc[group["days_to_expiry"] <= near_term_days, "close"],
            group.loc[group["days_to_expiry"] <= near_term_days, "volume"],
        )
        far_close = _weighted_average(
            group.loc[group["days_to_expiry"] >= far_term_days, "close"],
            group.loc[group["days_to_expiry"] >= far_term_days, "volume"],
        )
        term_structure_slope = (
            far_close - near_close if not np.isnan(far_close) and not np.isnan(near_close) else np.nan
        )

        avg_option_return = _weighted_average(group["option_return"], group["volume"])
        call_range_mean = group.loc[call_mask, "range"].mean()
        put_range_mean = group.loc[put_mask, "range"].mean()

        feature_rows.append(
            {
                "Date": date,
                "put_call_volume_ratio": pcr,
                "weighted_average_moneyness": weighted_avg_moneyness,
                "implied_direction_bias": implied_direction_bias,
                "skewed_volume_distribution": skewed_volume_distribution,
                "near_far_expiry_vol_ratio": near_far_expiry_ratio,
                "realized_vol_proxy": realized_vol_proxy,
                "high_low_spread_ratio": high_low_spread_ratio,
                "term_structure_slope": term_structure_slope,
                "total_volume": total_vol,
                "call_volume": call_vol,
                "put_volume": put_vol,
                "avg_option_return": avg_option_return,
                "call_range_mean": call_range_mean,
                "put_range_mean": put_range_mean,
            }
        )

    features = pd.DataFrame(feature_rows).sort_values("Date").reset_index(drop=True)
    return features


def compute_sentiment_scores(features: pd.DataFrame) -> pd.DataFrame:
    """Calculate higher-level sentiment scores from aggregated option features.

    Parameters
    ----------
    features : pandas.DataFrame
        Output from :func:`build_options_sentiment` containing, at a minimum,
        ``put_call_volume_ratio``, ``total_volume``, ``avg_option_return``,
        ``call_range_mean``, and ``put_range_mean``.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``Date``, ``option_sentiment_score`` (OSS), and
        ``fear_greed_skew``. Empty if prerequisites are not met.
    """
    required = {
        "Date",
        "put_call_volume_ratio",
        "total_volume",
        "avg_option_return",
        "call_range_mean",
        "put_range_mean",
    }
    if not required.issubset(set(features.columns)):
        missing = ", ".join(sorted(required - set(features.columns)))
        raise ValueError(f"Features missing required columns: {missing}")

    df = features.copy().sort_values("Date").reset_index(drop=True)
    volume_z = _zscore(df["total_volume"])
    return_z = _zscore(df["avg_option_return"])

    oss = (1 - df["put_call_volume_ratio"]) * volume_z * return_z
    fear_greed_skew = df["put_range_mean"] / df["call_range_mean"]

    result = pd.DataFrame(
        {
            "Date": df["Date"],
            "option_sentiment_score": oss,
            "fear_greed_skew": fear_greed_skew,
        }
    )
    return result
