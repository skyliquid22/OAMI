"""Custom FinRL trading environment with advanced order mechanics.

The environment extends :class:`finrl.env.env_stocktrading.StockTradingEnv`
by introducing limit orders with configurable take-profit (TP), stop-loss (SL),
and trailing-stop behaviour.  Actions are vectorised per ticker and encode the
parameters of a single order which may or may not be filled depending on the
subsequent price path.

The implementation intentionally keeps the observation space identical to the
parent environment so that existing agents (e.g. PPO from Stable Baselines)
remain compatible.  Only the action interpretation and reward shaping are
modified to support realistic order execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import gymnasium as gym
try:  # pragma: no cover - official FinRL module path
    from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "OrderTradingEnv requires FinRL (>=0.3). Install via `pip install finrl`."
    ) from exc


@dataclass
class _Order:
    """Container representing a single limit order for one ticker."""

    ticker_id: int
    shares: int
    limit_price: float
    take_profit_pct: float
    stop_loss_pct: float
    trailing_pct: float
    age: int = 0
    max_age: int = 1  # expire after next bar if unfilled


@dataclass
class _Position:
    """Details about an open position used for TP/SL tracking."""

    shares: int
    entry_price: float
    take_profit_pct: float
    stop_loss_pct: float
    trailing_pct: float
    trailing_reference: float


class OrderTradingEnv(StockTradingEnv):
    """FinRL trading environment with limit/stop order mechanics.

    Each action represents one limit order per ticker.  Orders can remain
    pending for at most one bar.  When triggered, the position is tracked with
    deterministic take-profit, stop-loss, and trailing-stop thresholds that are
    evaluated against the current bar's high/low range.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Action space: one 5-dimensional vector per ticker
        self.action_space = gym.spaces.Box(
            low=np.array([[-1.0, -0.05, 0.0, 0.0, 0.0]] * self.stock_dim, dtype=np.float32),
            high=np.array([[1.0, 0.05, 0.20, 0.10, 0.05]] * self.stock_dim, dtype=np.float32),
            dtype=np.float32,
        )

        self.active_orders: List[_Order] = []
        self.positions: Dict[int, _Position] = {}
        self.last_realized_pnl: float = 0.0
        self.last_position_change: float = 0.0
        self._skip_default_take_action: bool = False
        if not hasattr(self, "stocks"):
            self.stocks = [0.0] * self.stock_dim

    # ------------------------------------------------------------------
    # Environment Hooks
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):  # type: ignore[override]
        observation = super().reset(seed=seed, options=options)
        self.active_orders = []
        self.positions = {}
        self.last_realized_pnl = 0.0
        self.last_position_change = 0.0
        return observation

    # FinRL calls _take_action inside step(); we intercept to interpret orders.
    def _take_action(self, actions: np.ndarray) -> None:  # type: ignore[override]
        if self._skip_default_take_action:
            return
        actions = np.asarray(actions, dtype=float)
        if actions.ndim == 1 and actions.shape[0] == self.stock_dim:
            return
        self._handle_order_actions(actions.reshape(self.stock_dim, 5))

    def _handle_order_actions(self, actions: np.ndarray) -> None:
        self.last_realized_pnl = 0.0
        self.last_position_change = 0.0

        current_data = self._current_price_frame()
        closes = current_data["close"].to_numpy(dtype=float)

        new_orders: List[_Order] = []
        for ticker_id in range(self.stock_dim):
            qty_scaled, limit_offset, take_profit, stop_loss, trailing = actions[ticker_id]
            shares = int(np.round(qty_scaled * self.hmax))
            if shares == 0:
                continue

            limit_price = closes[ticker_id] * (1.0 + np.clip(limit_offset, -0.05, 0.05))
            new_orders.append(
                _Order(
                    ticker_id=ticker_id,
                    shares=shares,
                    limit_price=float(limit_price),
                    take_profit_pct=float(np.clip(take_profit, 0.0, 0.20)),
                    stop_loss_pct=float(np.clip(stop_loss, 0.0, 0.10)),
                    trailing_pct=float(np.clip(trailing, 0.0, 0.05)),
                )
            )

        if new_orders:
            self.active_orders.extend(new_orders)

        realized, position_change = self._execute_orders(current_data)
        self.last_realized_pnl += realized
        self.last_position_change += position_change

    def step(self, actions: np.ndarray):
        actions_array = np.asarray(actions, dtype=float)
        if actions_array.size == self.stock_dim * 5:
            self._handle_order_actions(actions_array.reshape(self.stock_dim, 5))
            self._skip_default_take_action = True
            base_actions = np.zeros(self.stock_dim, dtype=float)
            result = super().step(base_actions)
            self._skip_default_take_action = False
            return result
        return super().step(actions)

    def _execute_orders(self, price_frame: pd.DataFrame) -> tuple[float, float]:  # type: ignore[override]
        """Handle order filling and protective stop execution.

        Parameters
        ----------
        price_frame : pd.DataFrame
            View containing the OHLC data for the current step with length equal
            to ``self.stock_dim``.
        """

        realized_pnl = 0.0
        position_change = 0.0

        highs = price_frame.get("high", price_frame["close"]).to_numpy(dtype=float)
        lows = price_frame.get("low", price_frame["close"]).to_numpy(dtype=float)
        closes = price_frame["close"].to_numpy(dtype=float)

        remaining_orders: List[_Order] = []
        for order in self.active_orders:
            idx = order.ticker_id
            limit_price = order.limit_price
            filled = False
            if order.shares > 0:
                if lows[idx] <= limit_price <= highs[idx]:
                    filled = True
            else:
                if lows[idx] <= limit_price <= highs[idx]:
                    filled = True

            if filled:
                pnl_delta, change = self._process_fill(order, limit_price)
                realized_pnl += pnl_delta
                position_change += change
            else:
                order.age += 1
                if order.age <= order.max_age:
                    remaining_orders.append(order)

        self.active_orders = remaining_orders

        pnl_from_stops, change_from_stops = self._evaluate_positions(highs, lows, closes)
        realized_pnl += pnl_from_stops
        position_change += change_from_stops

        return realized_pnl, position_change

    def _calculate_reward(self) -> float:  # type: ignore[override]
        current_data = self._current_price_frame()
        closes = current_data["close"].to_numpy(dtype=float)

        unrealized = 0.0
        for idx, position in self.positions.items():
            unrealized += (closes[idx] - position.entry_price) * position.shares

        reward = (
            self.last_realized_pnl
            + unrealized
            - 0.001 * len(self.active_orders)
            - 0.0005 * self.last_position_change
        )
        return reward

    # ------------------------------------------------------------------
    # Order / Position helpers
    # ------------------------------------------------------------------

    def _current_price_frame(self) -> pd.DataFrame:
        data = getattr(self, "data", None)
        if data is None:
            data = self.df.loc[self.current_step]
        if isinstance(data, pd.Series):
            data = data.to_frame().T
        return data.reset_index(drop=True)

    def _process_fill(self, order: _Order, fill_price: float) -> tuple[float, float]:
        idx = order.ticker_id
        shares = order.shares

        trade_value = fill_price * shares
        if shares > 0:
            self.state[0] -= trade_value * (1.0 + self.buy_cost_pct)
        else:
            self.state[0] -= trade_value * (1.0 - self.sell_cost_pct)

        self.stocks[idx] += shares
        self.state[1 + self.stock_dim + idx] = self.stocks[idx]

        realized_pnl = 0.0
        position_change = float(abs(shares))

        prev_position = self.positions.get(idx)
        if prev_position is None or prev_position.shares == 0:
            self.positions[idx] = _Position(
                shares=shares,
                entry_price=fill_price,
                take_profit_pct=order.take_profit_pct,
                stop_loss_pct=order.stop_loss_pct,
                trailing_pct=order.trailing_pct,
                trailing_reference=fill_price,
            )
            return realized_pnl, position_change

        prev_shares = prev_position.shares
        new_shares = prev_shares + shares

        if np.sign(prev_shares) == np.sign(new_shares) and new_shares != 0:
            avg_price = (prev_position.entry_price * prev_shares + fill_price * shares) / new_shares
            prev_position.shares = new_shares
            prev_position.entry_price = avg_price
            prev_position.take_profit_pct = order.take_profit_pct
            prev_position.stop_loss_pct = order.stop_loss_pct
            prev_position.trailing_pct = order.trailing_pct
            prev_position.trailing_reference = fill_price
            return realized_pnl, position_change

        # Position reduced or flipped; realise PnL on overlapping shares
        closing_shares = -prev_shares if np.sign(prev_shares) != np.sign(new_shares) else shares
        realized_pnl += (fill_price - prev_position.entry_price) * closing_shares

        if new_shares == 0:
            self.positions[idx] = _Position(
                shares=0,
                entry_price=0.0,
                take_profit_pct=0.0,
                stop_loss_pct=0.0,
                trailing_pct=0.0,
                trailing_reference=0.0,
            )
        else:
            self.positions[idx] = _Position(
                shares=new_shares,
                entry_price=fill_price,
                take_profit_pct=order.take_profit_pct,
                stop_loss_pct=order.stop_loss_pct,
                trailing_pct=order.trailing_pct,
                trailing_reference=fill_price,
            )

        return realized_pnl, position_change

    def _evaluate_positions(
        self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray
    ) -> tuple[float, float]:
        realized_pnl = 0.0
        position_change = 0.0

        for idx, position in list(self.positions.items()):
            shares = position.shares
            if shares == 0:
                continue

            entry_price = position.entry_price
            take_profit_price, stop_loss_price = self._tp_sl_prices(position)
            exit_price: Optional[float] = None

            if shares > 0:
                position.trailing_reference = max(position.trailing_reference, highs[idx])
                trailing_price = position.trailing_reference * (1.0 - position.trailing_pct)

                if position.stop_loss_pct > 0 and lows[idx] <= stop_loss_price:
                    exit_price = stop_loss_price
                elif position.take_profit_pct > 0 and highs[idx] >= take_profit_price:
                    exit_price = take_profit_price
                elif position.trailing_pct > 0 and lows[idx] <= trailing_price:
                    exit_price = trailing_price
            else:
                position.trailing_reference = min(position.trailing_reference, lows[idx])
                trailing_price = position.trailing_reference * (1.0 + position.trailing_pct)

                if position.stop_loss_pct > 0 and highs[idx] >= stop_loss_price:
                    exit_price = stop_loss_price
                elif position.take_profit_pct > 0 and lows[idx] <= take_profit_price:
                    exit_price = take_profit_price
                elif position.trailing_pct > 0 and highs[idx] >= trailing_price:
                    exit_price = trailing_price

            if exit_price is None:
                continue

            trade_value = exit_price * (-shares)
            if shares > 0:
                self.state[0] -= trade_value * (1.0 - self.sell_cost_pct)
            else:
                self.state[0] -= trade_value * (1.0 + self.buy_cost_pct)

            self.stocks[idx] -= shares
            self.state[1 + self.stock_dim + idx] = self.stocks[idx]

            realized_pnl += (exit_price - entry_price) * shares
            position_change += abs(shares)
            self.positions[idx] = _Position(
                shares=0,
                entry_price=0.0,
                take_profit_pct=0.0,
                stop_loss_pct=0.0,
                trailing_pct=0.0,
                trailing_reference=0.0,
            )

        return realized_pnl, position_change

    def _tp_sl_prices(self, position: _Position) -> tuple[float, float]:
        if position.shares > 0:
            tp = position.entry_price * (1.0 + position.take_profit_pct) if position.take_profit_pct > 0 else np.inf
            sl = position.entry_price * (1.0 - position.stop_loss_pct) if position.stop_loss_pct > 0 else -np.inf
        else:
            tp = position.entry_price * (1.0 - position.take_profit_pct) if position.take_profit_pct > 0 else -np.inf
            sl = position.entry_price * (1.0 + position.stop_loss_pct) if position.stop_loss_pct > 0 else np.inf
        return tp, sl

    def render(self, mode: str = "human"):  # type: ignore[override]
        account_value = self.state[0] + np.sum(self.stocks * self._current_price_frame()["close"].to_numpy())
        print(f"Current account value: {account_value:,.2f}")
        for order in self.active_orders:
            print(
                f"Open order -> ticker {order.ticker_id} shares {order.shares} limit {order.limit_price:.2f}"
            )
        for idx, pos in self.positions.items():
            if pos.shares != 0:
                print(
                    f"Position -> ticker {idx} shares {pos.shares} entry {pos.entry_price:.2f}"
                )
