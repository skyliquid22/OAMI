"""Option feature engineering utilities with tenor-aware aggregation.

This module exposes an :class:`OptionFeatureBuilder` capable of transforming raw
market and option contract data into daily, per-underlying metrics that capture
sentiment, implied volatility term-structure, and gamma exposure across tenor
buckets.  The resulting feature set is suitable for downstream modelling
pipelines such as reinforcement learning agents or machine learning workflows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency at runtime
    from py_vollib.black_scholes.implied_volatility import (
        implied_volatility as bs_implied_volatility,
    )

    _HAVE_PY_VOLLIB = True
except ImportError:  # pragma: no cover - fallback to internal implementation
    _HAVE_PY_VOLLIB = False

__all__ = ["OptionFeatureBuilder"]


@dataclass(frozen=True)
class _TenorDefinition:
    """Container describing a tenor bucket."""

    start: int
    end: int
    label: str

    @property
    def display(self) -> str:
        return self.label


class OptionFeatureBuilder:
    """Engineer option-derived features with tenor bucketing.

    Parameters
    ----------
    tenor_buckets : Sequence[tuple[int, int]], optional
        Inclusive lower and upper bounds (in days) for each tenor bucket.
        Defaults to ``[(0, 7), (8, 30), (31, 90), (91, 9999)]``.
    bands : Sequence[float] | None, optional
        Optional strike-distance percentage thresholds used to calculate
        bracketed ratios. When omitted no band-specific metrics are produced.
    risk_free_rate : float, default=0.05
        Annualised continuously-compounded risk-free rate used in implied-vol
        estimation.
    use_open_interest : bool, default=False
        When ``True`` the builder expects an ``open_interest`` column in the
        options dataframe and prefers it over volume when computing sentiment
        balances and call/put ratios. When absent the implementation falls back
        to using traded volume.
    eps : float, default=1e-6
        Numerical stability constant to prevent divide-by-zero.
    """

    def __init__(
        self,
        tenor_buckets: Sequence[tuple[int, int]] | None = None,
        *,
        bands: Sequence[float] | None = None,
        risk_free_rate: float = 0.05,
        use_open_interest: bool = False,
        eps: float = 1e-6,
    ) -> None:
        self.tenor_buckets = tenor_buckets or [(0, 7), (8, 30), (31, 90), (91, 9999)]
        self.bands = tuple(sorted(bands)) if bands else None
        self.risk_free_rate = float(risk_free_rate)
        self.use_open_interest = bool(use_open_interest)
        self.eps = float(eps)
        self._tenor_defs = self._build_tenor_definitions(self.tenor_buckets)

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def fit_transform(
        self,
        df_market: pd.DataFrame,
        df_options: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return tenor-aware option features merged with market data."""
        market = self._prepare_market(df_market)
        options = self._prepare_options(df_options)

        if options.empty:
            return self._finalize_output(market, pd.DataFrame(), pd.DataFrame())

        options = self._attach_market_reference(options, market)
        options = self._enrich_contract_metrics(options)

        aggregated = self._aggregate_contract_metrics(options)
        tenor_features = self._compute_tenor_features(aggregated)
        coarse_ratios = self._compute_coarse_ratios(aggregated)

        return self._finalize_output(market, tenor_features, coarse_ratios)

    # --------------------------------------------------------------------- #
    # Preparation helpers
    # --------------------------------------------------------------------- #

    @staticmethod
    def _prepare_market(df: pd.DataFrame) -> pd.DataFrame:
        required = {
            "ticker",
            "volume",
            "open",
            "close",
            "high",
            "low",
            "window_start",
            "transactions",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"df_market is missing required columns: {', '.join(sorted(missing))}")

        market = df.copy()
        market["window_start"] = pd.to_datetime(market["window_start"], utc=False, errors="coerce")
        market["date"] = market["window_start"].dt.normalize()
        market["ticker"] = market["ticker"].astype(str).str.upper()

        aggregations: Mapping[str, str] = {
            "open": "first",
            "close": "last",
            "high": "max",
            "low": "min",
            "volume": "sum",
            "transactions": "sum",
        }
        columns = ["date", "ticker"] + list(aggregations.keys())
        market = (
            market[columns]
            .groupby(["date", "ticker"], as_index=False)
            .agg(aggregations)
        )
        return market.sort_values(["ticker", "date"]).reset_index(drop=True)

    def _prepare_options(self, df: pd.DataFrame) -> pd.DataFrame:
        required = {
            "ticker",
            "volume",
            "open",
            "close",
            "high",
            "low",
            "window_start",
            "transactions",
            "underlying_symbol",
            "strike_price",
            "contract_type",
            "expiration_date",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"df_options is missing required columns: {', '.join(sorted(missing))}")

        options = df.copy()
        options["window_start"] = pd.to_datetime(options["window_start"], utc=False, errors="coerce")
        options["expiration_date"] = pd.to_datetime(options["expiration_date"], utc=False, errors="coerce")
        options["date"] = options["window_start"].dt.normalize()
        options["contract_type"] = options["contract_type"].astype(str).str.upper()
        options["contract_type"] = options["contract_type"].replace({"CALL": "CALL", "PUT": "PUT", "C": "CALL", "P": "PUT"})
        options["underlying_symbol"] = options["underlying_symbol"].astype(str).str.upper()
        options["strike_price"] = pd.to_numeric(options["strike_price"], errors="coerce")
        options["volume"] = pd.to_numeric(options["volume"], errors="coerce").fillna(0.0)
        options["transactions"] = pd.to_numeric(options["transactions"], errors="coerce").fillna(0.0)

        options["days_to_expiry"] = (options["expiration_date"] - options["window_start"]).dt.days
        options = options[options["days_to_expiry"] >= 0]
        if options.empty:
            return options

        options["tenor_bucket"] = self._assign_tenor_labels(options["days_to_expiry"].to_numpy())
        options = options[options["tenor_bucket"].notna()]
        return options.reset_index(drop=True)

    @staticmethod
    def _build_tenor_definitions(buckets: Sequence[tuple[int, int]]) -> List[_TenorDefinition]:
        if not buckets:
            raise ValueError("At least one tenor bucket must be provided.")
        definitions: List[_TenorDefinition] = []
        for idx, (start, end) in enumerate(buckets, start=1):
            if start > end:
                raise ValueError(f"Invalid tenor bucket ({start}, {end}) – start must be <= end.")
            if idx == len(buckets) and end >= 90:
                label = f"t{idx}_90_plus"
            else:
                label = f"t{idx}_{start}_{end}"
            definitions.append(_TenorDefinition(start=start, end=end, label=label.replace(" ", "")))
        return definitions

    def _assign_tenor_labels(self, days_to_expiry: np.ndarray) -> np.ndarray:
        labels = np.full(days_to_expiry.shape, None, dtype=object)
        for definition in self._tenor_defs:
            mask = (days_to_expiry >= definition.start) & (days_to_expiry <= definition.end)
            labels = np.where(mask, definition.label, labels)
        return labels

    @staticmethod
    def _attach_market_reference(options: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
        if options.empty:
            return options

        ref = market.rename(columns={"ticker": "underlying_symbol", "close": "spot"})
        merged = options.merge(
            ref[["date", "underlying_symbol", "spot"]],
            on=["date", "underlying_symbol"],
            how="left",
        )
        merged["spot"] = merged.groupby("underlying_symbol")["spot"].transform(lambda s: s.ffill().bfill())
        return merged

    # --------------------------------------------------------------------- #
    # Per-contract metrics
    # --------------------------------------------------------------------- #

    def _enrich_contract_metrics(self, options: pd.DataFrame) -> pd.DataFrame:
        if options.empty:
            return options

        options = options.copy()
        options["time_to_expiry"] = options["days_to_expiry"] / 365.0

        price_cols = ["open", "high", "low", "close"]
        present = [col for col in price_cols if col in options.columns]
        options["mid_price"] = options[present].mean(axis=1, skipna=True)

        options["flag"] = np.where(options["contract_type"] == "CALL", "c", "p")
        options["implied_volatility"] = self._compute_implied_volatility(options)

        spot = options["spot"].replace({0: np.nan})
        moneyness = options["strike_price"] / spot
        synthetic_delta = np.clip(1.0 - np.abs(moneyness - 1.0), 0.0, 1.0)
        synthetic_delta = synthetic_delta.fillna(0.0)
        options["synthetic_delta"] = np.where(
            options["contract_type"] == "CALL",
            synthetic_delta,
            -synthetic_delta,
        )
        options["gamma"] = np.abs(options["synthetic_delta"]) * (1.0 - np.abs(options["synthetic_delta"]))

        if self.use_open_interest and "open_interest" in options.columns:
            options["oi"] = pd.to_numeric(options["open_interest"], errors="coerce").fillna(0.0)
        else:
            options["oi"] = options["volume"]

        if self.bands:
            options = self._compute_band_indicators(options, moneyness)

        return options

    def _compute_implied_volatility(self, options: pd.DataFrame) -> pd.Series:
        price = options["mid_price"].to_numpy(dtype=float)
        spot = options["spot"].to_numpy(dtype=float)
        strike = options["strike_price"].to_numpy(dtype=float)
        time_to_expiry = options["time_to_expiry"].to_numpy(dtype=float)
        flags = options["flag"].to_numpy()

        iv = np.full(price.shape, np.nan, dtype=float)

        valid_mask = (
            np.isfinite(price)
            & np.isfinite(spot)
            & np.isfinite(strike)
            & np.isfinite(time_to_expiry)
            & (price > 0)
            & (spot > 0)
            & (strike > 0)
            & (time_to_expiry > 1e-6)
        )

        if not valid_mask.any():
            return pd.Series(iv, index=options.index)

        try:
            if _HAVE_PY_VOLLIB:
                vectorised = np.vectorize(
                    lambda p, s, k, t, f: self._safe_implied_volatility(p, s, k, t, f),
                    otypes=[float],
                )
                iv[valid_mask] = vectorised(
                    price[valid_mask],
                    spot[valid_mask],
                    strike[valid_mask],
                    time_to_expiry[valid_mask],
                    flags[valid_mask],
                )
            else:
                vectorised = np.vectorize(
                    lambda p, s, k, t, f: _implied_volatility_newton(
                        price=p,
                        spot=s,
                        strike=k,
                        time_to_expiry=t,
                        rate=self.risk_free_rate,
                        flag=f,
                    ),
                    otypes=[float],
                )
                iv[valid_mask] = vectorised(
                    price[valid_mask],
                    spot[valid_mask],
                    strike[valid_mask],
                    time_to_expiry[valid_mask],
                    flags[valid_mask],
                )
        except Exception:  # pragma: no cover - defensive
            iv[valid_mask] = [
                self._safe_implied_volatility(p, s, k, t, f)
                for p, s, k, t, f in zip(
                    price[valid_mask],
                    spot[valid_mask],
                    strike[valid_mask],
                    time_to_expiry[valid_mask],
                    flags[valid_mask],
                    strict=False,
                )
            ]

        return pd.Series(iv, index=options.index)

    def _safe_implied_volatility(
        self,
        price: float,
        spot: float,
        strike: float,
        time_to_expiry: float,
        flag: str,
    ) -> float:
        try:
            if _HAVE_PY_VOLLIB:
                return float(bs_implied_volatility(price, spot, strike, time_to_expiry, self.risk_free_rate, flag))
            return _implied_volatility_newton(
                price=price,
                spot=spot,
                strike=strike,
                time_to_expiry=time_to_expiry,
                rate=self.risk_free_rate,
                flag=flag,
            )
        except Exception:
            return np.nan

    def _compute_band_indicators(self, options: pd.DataFrame, moneyness: pd.Series) -> pd.DataFrame:
        abs_distance = np.abs(moneyness - 1.0)
        for band in self.bands or ():
            options[f"in_band_{band:.2f}"] = abs_distance <= band
        return options

    # --------------------------------------------------------------------- #
    # Aggregation helpers
    # --------------------------------------------------------------------- #

    def _aggregate_contract_metrics(self, options: pd.DataFrame) -> pd.DataFrame:
        group_keys = ["date", "underlying_symbol", "tenor_bucket", "contract_type"]
        agg = (
            options.groupby(group_keys, as_index=False)
            .agg(
                volume_sum=("volume", "sum"),
                oi_sum=("oi", "sum"),
                iv_mean=("implied_volatility", "mean"),
                gamma_sum=("gamma", "sum"),
            )
        )
        return agg

    def _compute_tenor_features(self, agg: pd.DataFrame) -> pd.DataFrame:
        index_cols = ["date", "underlying_symbol", "tenor_bucket"]

        vol = self._pivot_contract_metric(agg, "volume_sum", index_cols)
        oi = self._pivot_contract_metric(agg, "oi_sum", index_cols)
        iv = self._pivot_contract_metric(agg, "iv_mean", index_cols)
        gamma = self._pivot_contract_metric(agg, "gamma_sum", index_cols)

        # Option sentiment per tenor
        num = (vol["CALL"] + oi["CALL"]) - (vol["PUT"] + oi["PUT"])
        den = (vol["CALL"] + oi["CALL"] + vol["PUT"] + oi["PUT"]) + self.eps
        option_sentiment = num / den

        # IV statistics
        iv_all = iv[["CALL", "PUT"]].mean(axis=1)
        iv_skew = (iv["CALL"] - iv["PUT"]) / (iv_all + self.eps)

        tenor_df = pd.DataFrame(
            {
                "option_sentiment": option_sentiment,
                "iv_mean": iv_all,
                "iv_skew": iv_skew,
                "gamma_total": gamma.sum(axis=1),
            }
        )
        tenor_df = tenor_df.reset_index()

        # Pivot to wide format
        wide = {}
        tenor_labels = [definition.label for definition in self._tenor_defs]
        for metric in ["option_sentiment", "iv_mean", "iv_skew", "gamma_total"]:
            pivot = tenor_df.pivot_table(
                index=["date", "underlying_symbol"],
                columns="tenor_bucket",
                values=metric,
            )
            pivot = pivot.reindex(columns=tenor_labels)
            pivot = pivot.rename(
                columns={
                    tenor: f"{metric}_{tenor}"
                    for tenor in pivot.columns
                }
            )
            wide[metric] = pivot

        merged = pd.concat(wide.values(), axis=1)
        expected_columns: list[str] = []
        for definition in self._tenor_defs:
            label = definition.label
            expected_columns.extend(
                [
                    f"option_sentiment_{label}",
                    f"iv_mean_{label}",
                    f"iv_skew_{label}",
                    f"gamma_total_{label}",
                ]
            )
        for column in expected_columns:
            if column not in merged.columns:
                merged[column] = np.nan
        merged = merged[expected_columns]
        return merged

    def _compute_coarse_ratios(self, agg: pd.DataFrame) -> pd.DataFrame:
        short_labels = [tenor.label for tenor in self._tenor_defs if tenor.end <= 30]
        long_labels = [tenor.label for tenor in self._tenor_defs if tenor.start > 30]

        def ratio_for(labels: Iterable[str]) -> pd.Series:
            if not labels:
                return pd.Series(dtype=float)
            subset = agg[agg["tenor_bucket"].isin(labels)]
            if subset.empty:
                return pd.Series(dtype=float)
            grouped = subset.groupby(["date", "underlying_symbol", "contract_type"])["oi_sum"].sum()
            pivot = grouped.unstack("contract_type").fillna(0.0)
            call = pivot.get("CALL", pd.Series(0.0, index=pivot.index))
            put = pivot.get("PUT", pd.Series(0.0, index=pivot.index)).replace(0.0, np.nan)
            return call.divide(put)

        short_ratio = ratio_for(short_labels)
        long_ratio = ratio_for(long_labels)

        return pd.DataFrame(
            {
                "oi_call_put_short": short_ratio,
                "oi_call_put_long": long_ratio,
            }
        )

    @staticmethod
    def _pivot_contract_metric(
        agg: pd.DataFrame,
        metric: str,
        index_cols: list[str],
    ) -> pd.DataFrame:
        pivot = agg.pivot_table(
            index=index_cols,
            columns="contract_type",
            values=metric,
            fill_value=0.0,
        )
        if isinstance(pivot.columns, pd.MultiIndex):
            pivot.columns = pivot.columns.get_level_values(-1)
        return pivot

    # --------------------------------------------------------------------- #
    # Final assembly
    # --------------------------------------------------------------------- #

    def _finalize_output(
        self,
        market: pd.DataFrame,
        tenor_features: pd.DataFrame,
        coarse_ratios: pd.DataFrame,
    ) -> pd.DataFrame:
        base = market.rename(columns={"ticker": "tic"}).copy()

        feature_df = base.set_index(["date", "tic"])
        if not tenor_features.empty:
            tenor_features = tenor_features.reindex(feature_df.index, fill_value=np.nan)
            feature_df = feature_df.join(tenor_features, how="left")
        if not coarse_ratios.empty:
            coarse_ratios = coarse_ratios.reindex(feature_df.index, fill_value=np.nan)
            feature_df = feature_df.join(coarse_ratios, how="left")

        feature_df = (
            feature_df.groupby(level="tic", group_keys=False)
            .apply(lambda frame: frame.ffill().bfill())
        )

        result = feature_df.reset_index().sort_values(["tic", "date"]).reset_index(drop=True)
        return result


# --------------------------------------------------------------------------- #
# Black-Scholes helpers (fallback when py_vollib is unavailable)
# --------------------------------------------------------------------------- #


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)


def _bs_price(flag: str, spot: float, strike: float, time_to_expiry: float, rate: float, sigma: float) -> float:
    if sigma <= 0.0 or time_to_expiry <= 0.0 or spot <= 0.0 or strike <= 0.0:
        return float("nan")
    sqrt_t = math.sqrt(time_to_expiry)
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma ** 2) * time_to_expiry) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    if flag.lower().startswith("c"):
        return spot * _norm_cdf(d1) - strike * math.exp(-rate * time_to_expiry) * _norm_cdf(d2)
    return strike * math.exp(-rate * time_to_expiry) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


def _bs_vega(spot: float, strike: float, time_to_expiry: float, rate: float, sigma: float) -> float:
    if sigma <= 0.0 or time_to_expiry <= 0.0 or spot <= 0.0 or strike <= 0.0:
        return float("nan")
    sqrt_t = math.sqrt(time_to_expiry)
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma ** 2) * time_to_expiry) / (sigma * sqrt_t)
    return spot * _norm_pdf(d1) * sqrt_t


def _implied_volatility_newton(
    *,
    price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    flag: str,
) -> float:
    if price <= 0.0 or spot <= 0.0 or strike <= 0.0 or time_to_expiry <= 0.0:
        return np.nan

    sigma = 0.2
    for _ in range(100):
        theoretical = _bs_price(flag, spot, strike, time_to_expiry, rate, sigma)
        if not np.isfinite(theoretical):
            return np.nan
        vega = _bs_vega(spot, strike, time_to_expiry, rate, sigma)
        if not np.isfinite(vega) or abs(vega) < 1e-8:
            break
        increment = (theoretical - price) / vega
        sigma -= increment
        if sigma <= 0.0:
            sigma = 1e-6
        if abs(increment) < 1e-6:
            return sigma
    final_price = _bs_price(flag, spot, strike, time_to_expiry, rate, sigma)
    if np.isfinite(final_price) and abs(final_price - price) < 1e-3:
        return sigma
    return np.nan
