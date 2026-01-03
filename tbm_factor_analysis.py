"""Evaluate feature predictive power against TBM labels using MI, XGBoost, SHAP."""

from __future__ import annotations

import math
import re
import argparse
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import classification_report
try:  # pragma: no cover - optional dependency in sandboxes
    from xgboost import DMatrix, XGBClassifier
except Exception:  # pragma: no cover
    DMatrix = None
    XGBClassifier = None

from oami.features import OptionFeatureBuilder, StockFeatureBuilder

STOCK_DIR = Path("data/flatfiles/us_stocks_sip/day_aggs_v1")
OPTION_DIR = Path("data/flatfiles/us_options_opra/day_aggs_v1")
DEFAULT_TICKERS = [
    "AAPL",
    "MSFT",
    "TSLA",
    "NVDA",
    "META",
    "AMZN",
    "GOOGL",
    "JPM",
    "UNH",
    "JNJ",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TBM feature predictability.")
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=None,
        help="List of tickers to include (space-separated). Defaults to liquid mega-cap basket.",
    )
    parser.add_argument(
        "--tickers-file",
        type=str,
        default=None,
        help="Optional text file with one ticker per line (overrides --tickers).",
    )
    parser.add_argument("--start", type=str, default="2025-03-01", help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", type=str, default="2025-05-31", help="End date (YYYY-MM-DD).")
    parser.add_argument("--tbm-horizon", type=int, default=10, help="TBM vertical barrier horizon in bars.")
    parser.add_argument(
        "--tbm-up-mult",
        type=float,
        default=1.5,
        help="Upper barrier multiplier applied to volatility proxy.",
    )
    parser.add_argument(
        "--tbm-dn-mult",
        type=float,
        default=1.0,
        help="Lower barrier multiplier applied to volatility proxy.",
    )
    return parser.parse_args()


def _resolve_tickers(args: argparse.Namespace) -> list[str]:
    if args.tickers_file:
        path = Path(args.tickers_file).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Ticker file not found: {path}")
        tickers = [
            line.strip().upper()
            for line in path.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        if not tickers:
            raise ValueError(f"No tickers found in {path}")
        return tickers
    if args.tickers:
        return [ticker.upper() for ticker in args.tickers]
    return DEFAULT_TICKERS


def iter_daily_files(base_dir: Path, start: pd.Timestamp, end: pd.Timestamp) -> Iterable[Path]:
    for day in pd.date_range(start, end, freq="D"):
        path = base_dir / f"{day.year:04d}" / f"{day.month:02d}" / f"{day:%Y-%m-%d}.csv.gz"
        if path.exists():
            yield path


def load_stock_data(tickers: Sequence[str], start: str, end: str) -> pd.DataFrame:
    start_ts = pd.to_datetime(start).normalize()
    end_ts = pd.to_datetime(end).normalize()
    tickers = [t.upper() for t in tickers]
    frames: list[pd.DataFrame] = []
    for path in iter_daily_files(STOCK_DIR, start_ts, end_ts):
        df = pd.read_csv(path)
        df = df[df["ticker"].isin(tickers)]
        if df.empty:
            continue
        df["window_start"] = pd.to_datetime(df["window_start"], unit="ns")
        frames.append(df)
    if not frames:
        raise RuntimeError("No stock flatfiles found for requested range.")
    data = pd.concat(frames, ignore_index=True)
    data["date"] = data["window_start"].dt.normalize()
    data["ticker"] = data["ticker"].str.upper()
    return data.sort_values(["ticker", "date"]).reset_index(drop=True)


def _parse_occ_frame(df: pd.DataFrame) -> pd.DataFrame:
    pattern = re.compile(r"O:([A-Z0-9]{1,6})(\d{2})(\d{2})(\d{2})([CP])(\d{8})")
    extracted = df["ticker"].str.extract(pattern)
    extracted.columns = ["underlying", "yy", "mm", "dd", "cp", "strike"]
    df = df[~extracted.isna().any(axis=1)].copy()
    extracted = extracted.loc[df.index]

    df["underlying_symbol"] = extracted["underlying"].str.upper()
    years = 2000 + extracted["yy"].astype(int)
    months = extracted["mm"].astype(int)
    days = extracted["dd"].astype(int)
    df["expiration_date"] = pd.to_datetime(
        pd.DataFrame({"year": years, "month": months, "day": days}),
        errors="coerce",
    )
    df["contract_type"] = extracted["cp"].map({"C": "CALL", "P": "PUT"})
    df["strike_price"] = extracted["strike"].astype(int) / 1000.0
    return df


def load_option_data(tickers: Sequence[str], start: str, end: str) -> pd.DataFrame:
    start_ts = pd.to_datetime(start).normalize()
    end_ts = pd.to_datetime(end).normalize()
    tickers = [t.upper() for t in tickers]
    frames: list[pd.DataFrame] = []
    for path in iter_daily_files(OPTION_DIR, start_ts, end_ts):
        df = pd.read_csv(path)
        df = _parse_occ_frame(df)
        df = df[df["underlying_symbol"].isin(tickers)]
        if df.empty:
            continue
        df["window_start"] = pd.to_datetime(df["window_start"], unit="ns")
        frames.append(df)
    if not frames:
        raise RuntimeError("No option flatfiles found for requested range.")
    data = pd.concat(frames, ignore_index=True)
    data["date"] = data["window_start"].dt.normalize()
    data["ticker"] = data["ticker"].str.upper()
    return data.sort_values(["underlying_symbol", "date"]).reset_index(drop=True)


def build_feature_matrix(
    tickers: Sequence[str],
    start: str,
    end: str,
    *,
    tbm_horizon: int = 10,
    tbm_up_mult: float = 1.5,
    tbm_dn_mult: float = 1.0,
) -> pd.DataFrame:
    stocks = load_stock_data(tickers, start, end)
    options = load_option_data(tickers, start, end)

    stock_builder = StockFeatureBuilder()
    stock_base = stocks.rename(columns={"ticker": "tic"})[
        ["date", "tic", "open", "high", "low", "close", "volume"]
    ]
    stock_features = stock_builder.transform(stock_base)

    option_builder = OptionFeatureBuilder()
    option_features = option_builder.fit_transform(stocks, options)

    base_cols = ["open", "high", "low", "close", "volume", "transactions"]
    option_payload = option_features.drop(columns=[c for c in base_cols if c in option_features.columns], errors="ignore")

    merged = stock_features.merge(option_payload, on=["date", "tic"], how="left")
    merged = merged.sort_values(["tic", "date"]).reset_index(drop=True)
    merged["tbm_label"] = compute_tbm_labels(
        merged,
        horizon=tbm_horizon,
        up_mult=tbm_up_mult,
        dn_mult=tbm_dn_mult,
    )
    return merged


def compute_tbm_labels(
    df: pd.DataFrame,
    horizon: int = 10,
    up_mult: float = 1.5,
    dn_mult: float = 1.0,
) -> pd.Series:
    labels = np.zeros(len(df), dtype=float)
    for tic, group in df.groupby("tic"):
        idxs = group.index.to_numpy()
        close = group["close"].to_numpy(dtype=float)
        high = group["high"].to_numpy(dtype=float)
        low = group["low"].to_numpy(dtype=float)
        returns = pd.Series(close).pct_change()
        vol = returns.rolling(window=20).std().bfill().fillna(returns.std())
        vol = vol.replace(0.0, np.nan).fillna(0.01)
        for local_idx, global_idx in enumerate(idxs):
            price = close[local_idx]
            sigma = float(vol.iloc[local_idx]) if not np.isnan(vol.iloc[local_idx]) else 0.01
            upper = price * (1 + up_mult * sigma)
            lower = price * (1 - dn_mult * sigma)
            label = 0
            max_idx = min(len(close) - 1, local_idx + horizon)
            for lookahead in range(local_idx + 1, max_idx + 1):
                if high[lookahead] >= upper:
                    label = 1
                    break
                if low[lookahead] <= lower:
                    label = -1
                    break
            labels[global_idx] = label
    return pd.Series(labels, index=df.index)


def prepare_ml_inputs(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, list[str], pd.DataFrame]:
    df = df.dropna(subset=["tbm_label"]).copy()
    feature_cols = [
        col
        for col in df.columns
        if col
        not in {
            "date",
            "tic",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "tbm_label",
        }
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    feature_frame = df[feature_cols].ffill().bfill()
    feature_frame = feature_frame.fillna(feature_frame.median(numeric_only=True)).fillna(0.0)
    labels = df["tbm_label"].astype(int)
    mapping = {-1: 0, 0: 1, 1: 2}
    encoded = labels.map(mapping)
    aligned = df.loc[feature_frame.index, ["date", "tic", "tbm_label"]].reset_index(drop=True)
    return feature_frame.reset_index(drop=True), encoded.reset_index(drop=True), feature_cols, aligned


def compute_mutual_information(
    features: pd.DataFrame,
    labels: pd.Series,
    feature_names: list[str],
) -> pd.DataFrame:
    mi_scores = mutual_info_classif(features, labels, discrete_features=False, random_state=42)
    return (
        pd.DataFrame({"feature": feature_names, "mi": mi_scores})
        .sort_values("mi", ascending=False)
        .reset_index(drop=True)
    )


def train_xgb_and_shap(
    features: pd.DataFrame,
    labels: pd.Series,
    feature_names: list[str],
) -> tuple[XGBClassifier, pd.DataFrame, dict]:
    if XGBClassifier is None or DMatrix is None:
        raise ImportError("xgboost is required for train_xgb_and_shap. Install xgboost>=2.0.0.")
    split_idx = int(len(features) * 0.8)
    X_train = features.iloc[:split_idx]
    X_test = features.iloc[split_idx:]
    y_train = labels.iloc[:split_idx]
    y_test = labels.iloc[split_idx:]

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        max_depth=5,
        learning_rate=0.05,
        n_estimators=400,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_lambda=1.0,
        tree_method="hist",
        random_state=42,
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    inv_map = {0: -1, 1: 0, 2: 1}
    y_test_raw = y_test.map(inv_map)
    pred_raw = pd.Series(pred).map(inv_map)
    report = classification_report(y_test_raw, pred_raw, output_dict=True, zero_division=0)

    booster = model.get_booster()
    dmatrix = DMatrix(features, feature_names=feature_names)
    shap_values = booster.predict(dmatrix, pred_contribs=True)
    if shap_values.ndim == 3:
        shap_contrib = shap_values[:, :, :-1]
        shap_importance = np.mean(np.abs(shap_contrib), axis=(0, 1)).ravel()
        n_shap_features = shap_contrib.shape[-1]
    else:
        shap_contrib = shap_values[:, :-1]
        shap_importance = np.mean(np.abs(shap_contrib), axis=0).ravel()
        n_shap_features = shap_contrib.shape[1]
    if n_shap_features != len(feature_names):
        raise ValueError(
            f"SHAP dimension mismatch: {n_shap_features} contributions vs {len(feature_names)} features"
        )
    shap_df = (
        pd.DataFrame({"feature": feature_names, "shap": shap_importance})
        .sort_values("shap", ascending=False)
        .reset_index(drop=True)
    )

    metrics = {
        "classification_report": report,
        "accuracy": float((pred_raw.values == y_test_raw.values).mean()),
    }
    return model, shap_df, metrics


def quantile_monotonicity(
    df: pd.DataFrame,
    features: Sequence[str],
    label_col: str = "tbm_label",
    bins: int = 5,
) -> pd.DataFrame:
    records = []
    for feature in features:
        series = df[feature]
        temp = pd.DataFrame({"feature": series, "label": df[label_col]}).dropna()
        if len(temp) < bins:
            continue
        try:
            quantiles = pd.qcut(temp["feature"], q=bins, duplicates="drop", labels=False)
        except ValueError:
            continue
        grouped = temp.groupby(quantiles)["label"].mean()
        order = np.arange(len(grouped))
        monotonicity = grouped.corr(pd.Series(order, index=grouped.index), method="spearman")
        record = {
            "feature": feature,
            "monotonicity": float(monotonicity) if not math.isnan(monotonicity) else 0.0,
        }
        for idx, value in grouped.items():
            record[f"q{idx}"] = value
        records.append(record)
    if not records:
        return pd.DataFrame()
    return (
        pd.DataFrame(records)
        .sort_values("monotonicity", key=lambda s: s.abs(), ascending=False)
        .reset_index(drop=True)
    )


def build_multi_factor_alpha(
    df: pd.DataFrame,
    feature_weights: dict[str, float],
    label_col: str = "tbm_label",
) -> tuple[pd.DataFrame, dict]:
    valid_features = [f for f, w in feature_weights.items() if f in df.columns and not math.isclose(w, 0.0)]
    if not valid_features:
        raise ValueError("No valid features supplied for alpha construction.")

    result = df.copy()
    z_map = {}
    for feature in valid_features:
        series = result[feature]
        z = (series - series.mean()) / (series.std(ddof=0) + 1e-6)
        z_map[feature] = z.fillna(0.0)
        result[f"{feature}_z"] = z_map[feature]

    weight_sum = sum(abs(feature_weights[f]) for f in valid_features)
    score = np.zeros(len(result))
    for feature in valid_features:
        weight = feature_weights[feature] / weight_sum
        score += weight * z_map[feature].to_numpy()
    result["alpha_score"] = score

    corr = result[["alpha_score", label_col]].corr(method="spearman").iloc[0, 1]
    try:
        quantiles = pd.qcut(result["alpha_score"], q=5, labels=False, duplicates="drop")
    except ValueError:
        quantiles = pd.Series(np.zeros(len(result), dtype=int), index=result.index)
    bucket_means = result.groupby(quantiles)[label_col].mean()
    metrics = {
        "spearman_alpha_vs_label": float(corr),
        "bucket_means": bucket_means.to_dict(),
    }
    return result, metrics


def main() -> None:
    args = _parse_args()
    tickers = _resolve_tickers(args)
    start = args.start
    end = args.end

    print(f"Loading flatfiles for {tickers} between {start} and {end}...")
    feature_table = build_feature_matrix(
        tickers,
        start,
        end,
        tbm_horizon=args.tbm_horizon,
        tbm_up_mult=args.tbm_up_mult,
        tbm_dn_mult=args.tbm_dn_mult,
    )
    features, labels, feature_cols, meta = prepare_ml_inputs(feature_table)
    analysis_df = pd.concat([meta, features], axis=1)

    print(f"Feature matrix shape: {features.shape}, labels distribution:\n{meta['tbm_label'].value_counts()}")
    label_by_ticker = meta.groupby("tic")["tbm_label"].value_counts().unstack(fill_value=0)
    print("\nLabel distribution by ticker (rows=ticker, cols=TBM labels):")
    print(label_by_ticker.to_string())

    mi_df = compute_mutual_information(features, labels, feature_cols)
    print("\nTop 15 Mutual Information Scores:")
    print(mi_df.head(15).to_string(index=False))

    model, shap_df, model_metrics = train_xgb_and_shap(features, labels, feature_cols)
    print("\nXGBoost Accuracy:", model_metrics["accuracy"])
    report = model_metrics["classification_report"]
    print("Classification report (TBM labels -1/0/1):")
    for label in ["-1", "0", "1"]:
        if label in report:
            stats = report[label]
            print(
                f"Label {label}: precision={stats['precision']:.3f}, recall={stats['recall']:.3f}, f1={stats['f1-score']:.3f}"
            )

    print("\nTop 15 SHAP importances:")
    print(shap_df.head(15).to_string(index=False))

    top_features = shap_df.head(10)["feature"].tolist()
    quantile_df = quantile_monotonicity(analysis_df, top_features)
    if not quantile_df.empty:
        print("\nQuantile monotonicity (top 10 SHAP features):")
        print(quantile_df.head(10).to_string(index=False))
    else:
        print("Quantile analysis produced no valid features.")

    shap_map = dict(zip(shap_df["feature"], shap_df["shap"]))
    monotonic_candidates = quantile_df[quantile_df["monotonicity"].abs() >= 0.5]["feature"].tolist()
    selected = monotonic_candidates[:5]
    if not selected:
        selected = shap_df.head(3)["feature"].tolist()
    weight_map = {feature: shap_map[feature] for feature in selected if feature in shap_map}
    print("\nAlpha factor weights (SHAP scaled):")
    for feature, weight in weight_map.items():
        print(f"  {feature}: {weight:.4f}")
    enriched_df, alpha_metrics = build_multi_factor_alpha(analysis_df, weight_map)
    print("\nMulti-factor alpha metrics:")
    print(f"Spearman(alpha, TBM label) = {alpha_metrics['spearman_alpha_vs_label']:.3f}")
    for bucket, value in alpha_metrics["bucket_means"].items():
        print(f"Alpha bucket {bucket}: mean TBM label {value:.3f}")

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    enriched_df.to_parquet(results_dir / "tbm_feature_alpha.parquet", index=False)
    shap_df.to_csv(results_dir / "tbm_feature_shap.csv", index=False)
    mi_df.to_csv(results_dir / "tbm_feature_mi.csv", index=False)
    quantile_df.to_csv(results_dir / "tbm_quantile_monotonicity.csv", index=False)
    print("\nArtifacts saved under results/ directory.")


if __name__ == "__main__":
    main()
