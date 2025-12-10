"""Feature engineering utilities for options and equities.

This module exposes two primary builders:

- :class:`OptionFeatureBuilder` transforms raw market and option contract data
  into tenor-aware sentiment, volatility, and gamma metrics suited for modelling
  pipelines (e.g., RL agents or ML workflows).
- :class:`StockFeatureBuilder` derives leakage-safe technical indicators from
  OHLCV market data, including trend, momentum, volatility, and liquidity
  proxies with optional lagging, normalization, and calendar features.
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

try:  # pragma: no cover - optional imports
    import ta  # type: ignore
    from ta.momentum import RSIIndicator as TaRSIIndicator
    from ta.trend import MACD as TaMACD
    from ta.volume import OnBalanceVolumeIndicator as TaOBV
    from ta.volatility import AverageTrueRange as TaATR
except Exception:  # pragma: no cover
    ta = None
    TaRSIIndicator = None
    TaMACD = None
    TaOBV = None
    TaATR = None

try:  # pragma: no cover - optional TA-Lib
    import talib  # type: ignore
except Exception:  # pragma: no cover
    talib = None

_TA_AVAILABLE = ta is not None
_TALIB_AVAILABLE = talib is not None

__all__ = ["OptionFeatureBuilder", "StockFeatureBuilder"]


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


class StockFeatureBuilder:
    """Engineer leakage-safe technical indicators from OHLCV market data."""

    def __init__(
        self,
        windows: Sequence[int] = (5, 10, 20, 60),
        *,
        norm_window: int = 60,
        add_calendar_features: bool = True,
        add_normalized_features: bool = True,
        use_ta: bool = True,
        use_talib: bool = False,
        lag_features: tuple[int, ...] = (1,),
        rsi_window: int = 14,
    ) -> None:
        cleaned_windows = tuple(int(w) for w in windows if int(w) > 0)
        if not cleaned_windows:
            raise ValueError("windows must contain at least one positive integer")
        self.windows = cleaned_windows
        self.norm_window = max(1, int(norm_window))
        self.add_calendar_features = add_calendar_features
        self.add_normalized_features = add_normalized_features
        self.use_ta = bool(use_ta) and _TA_AVAILABLE
        self.use_talib = bool(use_talib) and _TALIB_AVAILABLE
        lag_set = {int(lag) for lag in lag_features if int(lag) > 0}
        self.lag_features = tuple(sorted(lag_set))
        self.rsi_window = max(1, int(rsi_window))

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return df augmented with leakage-safe technical indicators."""
        required = {"date", "tic", "open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Input DataFrame is missing required columns: {', '.join(sorted(missing))}")

        data = df.copy()
        data["date"] = pd.to_datetime(data["date"], utc=False, errors="coerce")
        data = data.sort_values(["tic", "date"]).reset_index(drop=True)

        original_columns = list(data.columns)
        enriched = data.groupby("tic", group_keys=False).apply(self._compute_group_features).reset_index(drop=True)
        feature_cols = [col for col in enriched.columns if col not in original_columns]

        if feature_cols:
            enriched = self._apply_leakage_controls(enriched, feature_cols)

        if self.add_calendar_features:
            enriched = self._add_calendar_features(enriched)

        return enriched.sort_values(["tic", "date"]).reset_index(drop=True)

    def _compute_group_features(self, group: pd.DataFrame) -> pd.DataFrame:
        g = group.sort_values("date").copy()
        for column in ["open", "high", "low", "close", "volume"]:
            g[column] = pd.to_numeric(g[column], errors="coerce")

        close = g["close"]
        high = g["high"]
        low = g["low"]
        volume = g["volume"]

        close_safe = close.replace(0, np.nan)
        g["ret_1"] = close_safe.pct_change()
        g["log_ret_1"] = np.log(close_safe).diff()

        prev_close = close.shift(1)
        tr_components = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        )
        true_range = tr_components.max(axis=1)

        for window in self.windows:
            g[f"rv_{window}"] = g["log_ret_1"].rolling(window).std()
            if self.use_talib:
                try:
                    g[f"atr_{window}"] = talib.ATR(
                        high.to_numpy(dtype=float),
                        low.to_numpy(dtype=float),
                        close.to_numpy(dtype=float),
                        timeperiod=window,
                    )
                except Exception:
                    g[f"atr_{window}"] = true_range.rolling(window).mean()
            elif self.use_ta and TaATR is not None:
                atr = TaATR(high=high, low=low, close=close, window=window, fillna=False)
                g[f"atr_{window}"] = atr.average_true_range()
            else:
                g[f"atr_{window}"] = true_range.rolling(window).mean()

            g[f"sma_{window}"] = close.rolling(window).mean()
            g[f"ema_{window}"] = close.ewm(span=window, adjust=False).mean()
            g[f"roc_{window}"] = close.pct_change(window)

            vol_mean = volume.rolling(window).mean()
            vol_std = volume.rolling(window).std()
            g[f"vol_z_{window}"] = (volume - vol_mean) / (vol_std + 1e-9)

        mid_window = self.windows[min(1, len(self.windows) - 1)]
        ma_mid = close.rolling(mid_window).mean()
        sd_mid = close.rolling(mid_window).std()
        upper = ma_mid + 2.0 * sd_mid
        lower = ma_mid - 2.0 * sd_mid
        band_width = upper - lower
        safe_band_width = band_width.replace(0, np.nan)
        g[f"bb_pctb_{mid_window}"] = (close - lower) / safe_band_width
        g[f"bb_bw_{mid_window}"] = band_width / ma_mid.replace(0, np.nan)

        if self.use_ta and TaOBV is not None:
            obv_indicator = TaOBV(close=close, volume=volume, fillna=False)
            g["obv"] = obv_indicator.on_balance_volume()
        else:
            direction = np.sign(g["ret_1"].fillna(0.0))
            g["obv"] = (direction * volume.fillna(0.0)).cumsum()

        g["hl_spread"] = (high - low) / close_safe
        volume_safe = volume.replace(0, np.nan)
        g["amihud"] = g["ret_1"].abs() / volume_safe

        roll = self.windows[-1]
        roll_max = close.rolling(roll).max()
        g[f"mdd_{roll}"] = (close / roll_max) - 1.0

        # RSI
        rsi_col = f"rsi_{self.rsi_window}"
        if self.use_talib:
            try:
                g[rsi_col] = talib.RSI(close.to_numpy(dtype=float), timeperiod=self.rsi_window)
            except Exception:
                g[rsi_col] = self._compute_rsi_manual(close)
        elif self.use_ta and TaRSIIndicator is not None:
            rsi_indicator = TaRSIIndicator(close=close, window=self.rsi_window, fillna=False)
            g[rsi_col] = rsi_indicator.rsi()
        else:
            g[rsi_col] = self._compute_rsi_manual(close)

        # MACD
        if self.use_talib:
            try:
                macd, macd_signal, macd_hist = talib.MACD(
                    close.to_numpy(dtype=float), fastperiod=12, slowperiod=26, signalperiod=9
                )
                g["macd"] = macd
                g["macd_signal"] = macd_signal
                g["macd_hist"] = macd_hist
            except Exception:
                self._compute_macd_manual(g, close)
        elif self.use_ta and TaMACD is not None:
            macd_indicator = TaMACD(close=close, window_fast=12, window_slow=26, window_sign=9, fillna=False)
            g["macd"] = macd_indicator.macd()
            g["macd_signal"] = macd_indicator.macd_signal()
            g["macd_hist"] = macd_indicator.macd_diff()
        else:
            self._compute_macd_manual(g, close)

        return g

    def _compute_rsi_manual(self, close: pd.Series) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1 / self.rsi_window, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / self.rsi_window, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        return 100 - (100 / (1 + rs))

    def _compute_macd_manual(self, frame: pd.DataFrame, close: pd.Series) -> None:
        ema_fast = close.ewm(span=12, adjust=False).mean()
        ema_slow = close.ewm(span=26, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal = macd_line.ewm(span=9, adjust=False).mean()
        frame["macd"] = macd_line
        frame["macd_signal"] = signal
        frame["macd_hist"] = macd_line - signal

    def _apply_leakage_controls(self, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        if not feature_cols:
            return df

        lag_values = self.lag_features
        norm_window = self.norm_window
        add_norm = self.add_normalized_features

        def _per_group(group: pd.DataFrame) -> pd.DataFrame:
            g = group.sort_values("date").copy()
            g[feature_cols] = g[feature_cols].shift(1)
            if add_norm:
                for col in feature_cols:
                    roll = g[col].rolling(norm_window)
                    mean = roll.mean()
                    std = roll.std()
                    g[f"{col}_nz"] = (g[col] - mean) / (std + 1e-9)
            for lag in lag_values:
                shifted = g[feature_cols].shift(lag)
                shifted = shifted.add_suffix(f"_lag{lag}")
                g = pd.concat([g, shifted], axis=1)
            return g

        return df.groupby("tic", group_keys=False).apply(_per_group).reset_index(drop=True)

    def _add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        enriched = df.copy()
        enriched["date"] = pd.to_datetime(enriched["date"], utc=False, errors="coerce")
        dow = enriched["date"].dt.weekday
        dummies = pd.get_dummies(dow, prefix="dow", dtype=float)
        for idx in range(7):
            column = f"dow_{idx}"
            if column not in dummies:
                dummies[column] = 0.0
        dummies = dummies[[f"dow_{i}" for i in range(7)]]
        return pd.concat([enriched, dummies], axis=1)


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


if __name__ == "__main__":  # pragma: no cover - basic sanity example
    import pandas as pd

    sample = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=5, freq="D").tolist()
            + pd.date_range("2024-01-01", periods=5, freq="D").tolist(),
            "tic": ["AAPL"] * 5 + ["MSFT"] * 5,
            "open": np.linspace(100, 104, 5).tolist() + np.linspace(200, 204, 5).tolist(),
            "high": np.linspace(101, 105, 5).tolist() + np.linspace(201, 205, 5).tolist(),
            "low": np.linspace(99, 103, 5).tolist() + np.linspace(199, 203, 5).tolist(),
            "close": np.linspace(100.5, 104.5, 5).tolist() + np.linspace(200.5, 204.5, 5).tolist(),
            "volume": [1_000_000 + i * 10_000 for i in range(5)]
            + [2_000_000 + i * 20_000 for i in range(5)],
        }
    )

    builder = StockFeatureBuilder()
    sample_features = builder.transform(sample)
    print(sample_features.head())
