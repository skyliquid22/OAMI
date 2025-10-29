"""Training pipeline for FinRL using custom option-derived features."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:  # pragma: no cover - heavy dependency
    from finrl.env.env_stocktrading import StockTradingEnv
    from finrl.agents.stablebaselines3.models import DRLAgent
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "This pipeline requires the FinRL package. Install via `pip install finrl` before running."
    ) from exc

from src.preprocessing.custom_preprocessor import CustomFinRLPreprocessor

RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_frame(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data path not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _collect_indicators(df: pd.DataFrame) -> list[str]:
    prefixes = ("option_sentiment", "iv_mean", "iv_skew", "gamma_total", "oi_call_put")
    return [col for col in df.columns if col.startswith(prefixes)]


def run_pipeline(market_path: str | Path, options_path: str | Path, train_end: str, risk_free_rate: float) -> None:
    df_market = _load_frame(market_path)
    df_options = _load_frame(options_path)

    preprocessor = CustomFinRLPreprocessor(risk_free_rate=risk_free_rate)
    df_features = preprocessor.fit_transform(df_market, df_options)

    df_train, df_test = preprocessor.prepare_train_test_split(df_features, train_end)

    tech_indicator_list = _collect_indicators(df_features)
    print("✅ Technical indicators used:", tech_indicator_list)

    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1_000_000,
        "buy_cost_pct": 0.001,
        "sell_cost_pct": 0.001,
        "state_space": len(tech_indicator_list) + 2,
        "stock_dim": df_features["tic"].nunique(),
        "tech_indicator_list": tech_indicator_list,
        "action_space": df_features["tic"].nunique(),
        "reward_scaling": 1e-4,
    }

    env_train = StockTradingEnv(df=df_train, **env_kwargs)
    env_trade = StockTradingEnv(df=df_test, **env_kwargs)

    agent = DRLAgent(env=env_train)
    model_kwargs = {"n_steps": 2048, "ent_coef": 0.005, "learning_rate": 1e-4, "batch_size": 128}
    print("✅ Training PPO agent...")
    model = agent.get_model("ppo", model_kwargs=model_kwargs)
    trained_model = agent.train_model(model, total_timesteps=100_000)

    print("✅ Running simulation...")
    df_account_value, df_actions = DRLAgent.DRL_prediction(model=trained_model, environment=env_trade)

    account_path = RESULTS_DIR / "account_value.parquet"
    actions_path = RESULTS_DIR / "actions.parquet"
    df_account_value.to_parquet(account_path, index=False)
    df_actions.to_parquet(actions_path, index=False)
    print(f"✅ Saved account values to {account_path}")
    print(f"✅ Saved actions to {actions_path}")

    start_value = df_account_value.iloc[0]["account_value"]
    end_value = df_account_value.iloc[-1]["account_value"]
    total_return = (end_value / start_value) - 1.0
    print(f"✅ Simulation complete, total return: {total_return:.2%}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FinRL training pipeline with custom option features")
    parser.add_argument("--market_path", type=str, required=True)
    parser.add_argument("--options_path", type=str, required=True)
    parser.add_argument("--train_end", type=str, required=True)
    parser.add_argument("--risk_free_rate", type=float, default=0.05)
    return parser.parse_args()


if __name__ == "__main__":  # pragma: no cover
    args = parse_args()
    run_pipeline(
        market_path=args.market_path,
        options_path=args.options_path,
        train_end=args.train_end,
        risk_free_rate=args.risk_free_rate,
    )
