"""Institutional-grade FinRL environment with realistic execution mechanics."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

import numpy as np
import pandas as pd

try:  # pragma: no cover - external dependency
    from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "InstitutionalTradingEnv requires FinRL (>=0.3). Install via `pip install finrl`."
    ) from exc

from .order_trading_env import OrderTradingEnv, _Order, _Position


@dataclass
class _QueuedOrder:
    """Wrapper holding an order scheduled for execution after latency."""

    due_step: int
    order: _Order


class InstitutionalTradingEnv(OrderTradingEnv):
    """Extended trading environment with institutional execution features.

    The environment augments :class:`OrderTradingEnv` with:
    - Slippage and probabilistic partial fills
    - Latency queues for deferred execution
    - Leverage and margin constraints with forced liquidation
    - Reward shaped using risk-adjusted metrics (Sharpe-style)
    - Drawdown and volatility penalties to stabilise RL training
    """

    def __init__(self, *args, **kwargs) -> None:
        if 'num_stock_shares' not in kwargs:
            kwargs['num_stock_shares'] = [0] * kwargs.get('stock_dim', 1)
        super().__init__(*args, **kwargs)

        self.slippage_coef: float = 0.0005
        self.partial_fill_prob: float = 0.6
        self.leverage_limit: float = 3.0
        self.margin_req: float = 0.25
        self.latency_steps: int = 1
        self.active_queue: Deque[_QueuedOrder] = deque()
        self.pnl_history: List[float] = []
        self.drawdown_window: int = 50
        self.reward_weights: Dict[str, float] = {
            "pnl": 1.0,
            "vol_penalty": 0.1,
            "drawdown_penalty": 0.2,
            "unfilled_penalty": 0.05,
            "leverage_penalty": 0.1,
        }
        self.reward_buffer: Deque[float] = deque(maxlen=200)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):  # type: ignore[override]
        observation = super().reset(seed=seed, options=options)
        self.active_queue.clear()
        self.pnl_history.clear()
        self.reward_buffer.clear()
        return observation

    # ------------------------------------------------------------------
    # Order Handling
    # ------------------------------------------------------------------

    def _take_action(self, actions: np.ndarray) -> None:  # type: ignore[override]
        """Interpret actions as latency-delayed limit orders.

        The method converts each ticker's 5-element action vector into a
        :class:`_Order`, applies leverage-aware scaling, and stores the order
        in the latency queue. Orders whose latency has elapsed are promoted to
        ``self.active_orders`` for execution on the current bar.
        """

        actions = np.asarray(actions, dtype=float).reshape(self.stock_dim, 5)
        current_data = self._current_price_frame()
        closes = current_data["close"].to_numpy(dtype=float)

        self._check_leverage_and_margin(current_data)

        new_orders: List[_Order] = []

        for ticker_id in range(self.stock_dim):
            qty_scaled, limit_offset, tp, sl, trailing = actions[ticker_id]
            shares = int(np.round(qty_scaled * self.hmax))
            if shares == 0:
                continue
            limit_price = closes[ticker_id] * (1.0 + np.clip(limit_offset, -0.05, 0.05))
            order = _Order(
                ticker_id=ticker_id,
                shares=shares,
                limit_price=float(limit_price),
                take_profit_pct=float(np.clip(tp, 0.0, 0.20)),
                stop_loss_pct=float(np.clip(sl, 0.0, 0.10)),
                trailing_pct=float(np.clip(trailing, 0.0, 0.05)),
            )
            new_orders.append(order)

        if new_orders:
            scaled_orders = self._scale_orders_for_leverage(new_orders, closes)
            for order in scaled_orders:
                if order.shares == 0:
                    continue
                due_step = self.current_step + self.latency_steps
                self.active_queue.append(_QueuedOrder(due_step=due_step, order=order))

        # Execute matured orders
        matured_orders: List[_Order] = []
        while self.active_queue and self.active_queue[0].due_step <= self.current_step:
            matured_orders.append(self.active_queue.popleft().order)

        if matured_orders:
            self.active_orders.extend(matured_orders)

        realized, position_change = self._execute_orders(current_data)
        self.last_realized_pnl = realized
        self.last_position_change = position_change

    def _apply_slippage_and_partial_fill(self, order: _Order) -> tuple[int, float]:
        """Return the filled quantity and price after slippage adjustments.

        Slippage is modelled as a fixed basis-point adjustment in the direction
        of the trade. Partial fills are sampled, with ``partial_fill_prob``
        controlling the likelihood of a complete execution once the order is
        triggered.
        """

        if np.random.rand() <= self.partial_fill_prob:
            fill_fraction = 1.0
        else:
            fill_fraction = np.random.rand()
        fill_qty = int(np.round(order.shares * fill_fraction))
        if fill_qty == 0:
            return 0, order.limit_price

        direction = np.sign(fill_qty)
        fill_price = order.limit_price * (1.0 + self.slippage_coef * direction)
        return fill_qty, fill_price

    def _execute_orders(self, price_frame: pd.DataFrame) -> tuple[float, float]:  # type: ignore[override]
        realized_pnl = 0.0
        position_change = 0.0

        highs = price_frame.get("high", price_frame["close"]).to_numpy(dtype=float)
        lows = price_frame.get("low", price_frame["close"]).to_numpy(dtype=float)

        remaining_orders: List[_Order] = []
        for order in self.active_orders:
            idx = order.ticker_id
            limit_price = order.limit_price
            triggered = lows[idx] <= limit_price <= highs[idx]
            if triggered:
                fill_qty, fill_price = self._apply_slippage_and_partial_fill(order)
                if fill_qty == 0:
                    remaining_orders.append(order)
                    continue
                partial_order = _Order(
                    ticker_id=order.ticker_id,
                    shares=fill_qty,
                    limit_price=fill_price,
                    take_profit_pct=order.take_profit_pct,
                    stop_loss_pct=order.stop_loss_pct,
                    trailing_pct=order.trailing_pct,
                )
                pnl_delta, change = self._process_fill(partial_order, fill_price)
                realized_pnl += pnl_delta
                position_change += change
                remaining_shares = order.shares - fill_qty
                if remaining_shares != 0:
                    order.shares = remaining_shares
                    remaining_orders.append(order)
            else:
                order.age += 1
                if order.age <= order.max_age:
                    remaining_orders.append(order)

        self.active_orders = remaining_orders

        closes = price_frame["close"].to_numpy(dtype=float)
        pnl_stops, change_stops = self._evaluate_positions(highs, lows, closes)
        realized_pnl += pnl_stops
        position_change += change_stops

        self._check_leverage_and_margin(price_frame)

        return realized_pnl, position_change

    # ------------------------------------------------------------------
    # Risk Controls
    # ------------------------------------------------------------------

    def _check_leverage_and_margin(self, price_frame: pd.DataFrame) -> None:
        """Enforce leverage limits and perform margin-driven liquidations."""

        closes = price_frame["close"].to_numpy(dtype=float)
        position_exposure = float(np.sum(np.abs(self.stocks) * closes))
        portfolio_value = np.dot(self.stocks, closes)
        account_equity = self.state[0] + portfolio_value
        leverage = position_exposure / (account_equity + 1e-8)

        if leverage > self.leverage_limit:
            scale = (self.leverage_limit * account_equity) / max(
                position_exposure, 1e-8
            )
            scale = np.clip(scale, 0.0, 1.0)
            for order in self.active_orders:
                order.shares = int(np.round(order.shares * scale))
            self.active_orders = [
                order for order in self.active_orders if order.shares != 0
            ]
            for queued in list(self.active_queue):
                queued.order.shares = int(np.round(queued.order.shares * scale))
            self.active_queue = deque(
                [q for q in self.active_queue if q.order.shares != 0]
            )

        if (
            account_equity < self.margin_req * position_exposure
            and position_exposure > 0
        ):
            for idx, shares in enumerate(self.stocks):
                if shares == 0:
                    continue
                price = closes[idx]
                liquidation_order = _Order(
                    ticker_id=idx,
                    shares=-shares,
                    limit_price=price,
                    take_profit_pct=0.0,
                    stop_loss_pct=0.0,
                    trailing_pct=0.0,
                )
                self._process_fill(liquidation_order, price)
                self.positions[idx] = _Position(
                    shares=0,
                    entry_price=0.0,
                    take_profit_pct=0.0,
                    stop_loss_pct=0.0,
                    trailing_pct=0.0,
                    trailing_reference=0.0,
                )
            self.active_orders.clear()
            self.active_queue.clear()

    def _scale_orders_for_leverage(
        self, orders: List[_Order], closes: np.ndarray
    ) -> List[_Order]:
        """Scale a batch of new orders so projected leverage stays within limits."""

        position_exposure = float(np.sum(np.abs(self.stocks) * closes))
        projected_exposure = position_exposure + sum(
            abs(order.shares * order.limit_price) for order in orders
        )
        portfolio_value = np.dot(self.stocks, closes)
        account_equity = self.state[0] + portfolio_value
        if account_equity <= 0:
            return []
        projected_leverage = projected_exposure / max(account_equity, 1e-8)
        if projected_leverage <= self.leverage_limit:
            return orders

        target_exposure = self.leverage_limit * account_equity
        scale = target_exposure / max(projected_exposure, 1e-8)
        scaled_orders: List[_Order] = []
        for order in orders:
            scaled_shares = int(np.round(order.shares * scale))
            if scaled_shares == 0:
                continue
            order.shares = scaled_shares
            scaled_orders.append(order)
        return scaled_orders

    # ------------------------------------------------------------------
    # Reward Calculation
    # ------------------------------------------------------------------

    def _calculate_reward(self) -> float:  # type: ignore[override]
        current_data = self._current_price_frame()
        closes = current_data["close"].to_numpy(dtype=float)

        exposure = float(np.sum(np.abs(self.stocks) * closes))
        portfolio_value = np.dot(self.stocks, closes)
        account_equity = self.state[0] + portfolio_value
        self.pnl_history.append(account_equity)

        # Base PnL components
        unrealized = 0.0
        for idx, position in self.positions.items():
            unrealized += (closes[idx] - position.entry_price) * position.shares

        pnl_component = self.reward_weights["pnl"] * (
            self.last_realized_pnl + unrealized
        )

        # Volatility penalty via rolling std of equity changes
        vol_penalty = 0.0
        if len(self.pnl_history) > 1:
            equity_series = pd.Series(self.pnl_history)
            returns = equity_series.diff().fillna(0.0)
            rolling_vol = (
                returns.rolling(window=min(len(returns), self.drawdown_window))
                .std()
                .fillna(0.0)
                .iloc[-1]
            )
            vol_penalty = self.reward_weights["vol_penalty"] * rolling_vol

        # Drawdown penalty
        dd_penalty = 0.0
        dd = 0.0
        if self.pnl_history:
            window_vals = self.pnl_history[-self.drawdown_window :]
            rolling_max = max(window_vals)
            if rolling_max != 0:
                dd = (account_equity - rolling_max) / rolling_max
            dd_penalty = self.reward_weights["drawdown_penalty"] * abs(dd)

        # Order and leverage penalties
        unfilled_penalty = self.reward_weights["unfilled_penalty"] * len(
            self.active_orders
        )
        leverage = exposure / (account_equity + 1e-8)
        excess_leverage = max(0.0, leverage - self.leverage_limit)
        leverage_penalty = self.reward_weights["leverage_penalty"] * excess_leverage

        raw_reward = (
            pnl_component
            - vol_penalty
            - dd_penalty
            - unfilled_penalty
            - leverage_penalty
        )

        self.reward_buffer.append(raw_reward)
        buffer_array = np.array(self.reward_buffer, dtype=float)
        mean_reward = buffer_array.mean()
        std_reward = buffer_array.std(ddof=0) + 1e-6
        sharpe_like_reward = mean_reward / std_reward

        if self.current_step % max(1, self.latency_steps) == 0:
            print(
                f"Step {self.current_step} | Equity {account_equity:.2f} | Leverage {leverage:.2f} | DD {dd:.2%}"
            )

        return sharpe_like_reward

    # ------------------------------------------------------------------
    # Metrics & Rendering
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, float]:
        if not self.pnl_history:
            return {"sharpe": 0.0, "max_drawdown": 0.0, "mean_reward": 0.0}
        equity_series = pd.Series(self.pnl_history)
        returns = equity_series.diff().fillna(0.0)
        sharpe = returns.mean() / (returns.std(ddof=0) + 1e-6)
        rolling_max = equity_series.cummax()
        drawdowns = (equity_series - rolling_max) / (rolling_max + 1e-6)
        max_drawdown = drawdowns.min()
        mean_reward = float(np.mean(self.reward_buffer)) if self.reward_buffer else 0.0
        return {
            "sharpe": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "mean_reward": mean_reward,
        }

    def render(self, mode: str = "human"):  # type: ignore[override]
        current_data = self._current_price_frame()
        closes = current_data["close"].to_numpy(dtype=float)
        exposure = float(np.sum(np.abs(self.stocks) * closes))
        equity = self.state[0] + np.dot(self.stocks, closes)
        leverage = exposure / (equity + 1e-8)
        metrics = self.get_metrics()
        print(
            f"Equity: {equity:,.2f} | Leverage: {leverage:.2f} | Sharpe: {metrics['sharpe']:.2f} | Max DD: {metrics['max_drawdown']:.2%}"
        )
        super().render(mode=mode)
