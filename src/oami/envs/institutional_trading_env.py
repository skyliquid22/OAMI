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


@dataclass
class _TBMState:
    """Metadata describing triple-barrier configuration for a position."""

    ticker_id: int
    entry_step: int
    entry_price: float
    tbm_upper: float
    tbm_lower: float
    tbm_expiry_step: int
    shares: int
    tbm_label: Optional[int] = None
    resolved: bool = False


class InstitutionalTradingEnv(OrderTradingEnv):
    """Extended trading environment with institutional execution features.

    The environment augments :class:`OrderTradingEnv` with:
    - Slippage and probabilistic partial fills
    - Latency queues for deferred execution
    - Leverage and margin constraints with forced liquidation
    - Reward shaped using risk-adjusted metrics (Sharpe-style)
    - Drawdown and volatility penalties to stabilise RL training
    """

    def __init__(self, *args, exposure_penalty: float = 0.0, **kwargs) -> None:
        if "num_stock_shares" not in kwargs:
            kwargs["num_stock_shares"] = [0] * kwargs.get("stock_dim", 1)

        self.tbm_enable: bool = kwargs.pop("tbm_enable", True)
        self.tbm_horizon: int = kwargs.pop("tbm_horizon", 10)
        self.tbm_up_mult: float = kwargs.pop("tbm_up_mult", 1.5)
        self.tbm_dn_mult: float = kwargs.pop("tbm_dn_mult", 1.0)
        self.tbm_weight: float = kwargs.pop("tbm_weight", 0.1)

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
        self.exposure_penalty: float = exposure_penalty
        self.reward_buffer: Deque[float] = deque(maxlen=200)
        self.tbm_states: Dict[int, _TBMState] = {}
        self._tbm_price_history: Dict[int, Deque[float]] = {}
        self._tbm_last_history_step: int = -1
        self.tbm_positive_events: int = 0
        self.tbm_negative_events: int = 0
        self.tbm_neutral_events: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):  # type: ignore[override]
        observation = super().reset(seed=seed, options=options)
        self.active_queue.clear()
        self.pnl_history.clear()
        self.reward_buffer.clear()
        self.tbm_states.clear()
        self._tbm_price_history.clear()
        self._tbm_last_history_step = -1
        self.tbm_positive_events = 0
        self.tbm_negative_events = 0
        self.tbm_neutral_events = 0
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

    def _calculate_reward(
        self, resolved_positions: Optional[List[_TBMState]] = None
    ) -> float:  # type: ignore[override]
        current_data = self._current_price_frame()
        closes = current_data["close"].to_numpy(dtype=float)

        if resolved_positions is None and self.tbm_enable:
            resolved_positions = self._update_tbm_for_positions(current_data)

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
        exposure_penalty = self.exposure_penalty * min(leverage, self.leverage_limit)

        raw_reward = (
            pnl_component
            - vol_penalty
            - dd_penalty
            - unfilled_penalty
            - leverage_penalty
            + exposure_penalty
        )

        if self.tbm_enable and resolved_positions:
            tbm_bonus = 0.0
            for state in resolved_positions:
                if state.tbm_label == 1:
                    tbm_bonus += 1.0
                    self.tbm_positive_events += 1
                elif state.tbm_label == -1:
                    tbm_bonus -= 1.0
                    self.tbm_negative_events += 1
                elif state.tbm_label == 0:
                    self.tbm_neutral_events += 1
            raw_reward += self.tbm_weight * tbm_bonus

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

    # ------------------------------------------------------------------
    # TBM helpers
    # ------------------------------------------------------------------

    def _current_price_frame(self) -> pd.DataFrame:  # type: ignore[override]
        frame = super()._current_price_frame()
        if self.tbm_enable:
            self._update_tbm_price_history(frame)
        return frame

    def _update_tbm_price_history(self, price_frame: pd.DataFrame) -> None:
        if self._tbm_last_history_step == self.current_step:
            return
        self._tbm_last_history_step = self.current_step
        closes = price_frame["close"].to_numpy(dtype=float)
        history_len = max(self.tbm_horizon * 5, 60)
        for idx, price in enumerate(closes):
            history = self._tbm_price_history.get(idx)
            if history is None:
                history = deque(maxlen=history_len)
                self._tbm_price_history[idx] = history
            history.append(float(price))

    def _estimate_tbm_volatility(self, ticker_id: int) -> float:
        history = self._tbm_price_history.get(ticker_id)
        if not history or len(history) < 2:
            return 1e-3
        prices = np.array(history, dtype=float)
        prices = np.clip(prices, 1e-6, None)
        log_returns = np.diff(np.log(prices))
        if log_returns.size == 0:
            return 1e-3
        window = min(len(log_returns), max(self.tbm_horizon, 5))
        vol = float(np.nanstd(log_returns[-window:]))
        return max(vol, 1e-4)

    def _init_tbm_state(self, ticker_id: int, entry_price: float, shares: int) -> None:
        vol_proxy = self._estimate_tbm_volatility(ticker_id)
        up_offset = self.tbm_up_mult * vol_proxy * entry_price
        dn_offset = self.tbm_dn_mult * vol_proxy * entry_price

        if shares >= 0:
            upper = entry_price + up_offset
            lower = entry_price - dn_offset
        else:
            upper = entry_price - up_offset
            lower = entry_price + dn_offset

        self.tbm_states[ticker_id] = _TBMState(
            ticker_id=ticker_id,
            entry_step=self.current_step,
            entry_price=float(entry_price),
            tbm_upper=float(upper),
            tbm_lower=float(lower),
            tbm_expiry_step=self.current_step + self.tbm_horizon,
            shares=shares,
        )

    def _finalize_tbm_state(self, ticker_id: int, label: Optional[int] = 0) -> None:
        state = self.tbm_states.pop(ticker_id, None)
        if state is None:
            return
        state.tbm_label = label
        state.resolved = True

    def _update_tbm_for_positions(self, price_frame: pd.DataFrame) -> List[_TBMState]:
        if not self.tbm_states:
            return []

        closes = price_frame["close"].to_numpy(dtype=float)
        resolved: List[_TBMState] = []
        for ticker_id, state in list(self.tbm_states.items()):
            current_price = closes[ticker_id]
            if state.shares >= 0:
                hit_up = current_price >= state.tbm_upper
                hit_down = current_price <= state.tbm_lower
            else:
                hit_up = current_price <= state.tbm_upper
                hit_down = current_price >= state.tbm_lower
            expired = self.current_step >= state.tbm_expiry_step

            if hit_up:
                state.tbm_label = 1
                state.resolved = True
            elif hit_down:
                state.tbm_label = -1
                state.resolved = True
            elif expired:
                state.tbm_label = 0
                state.resolved = True

            if state.resolved:
                resolved.append(state)
                del self.tbm_states[ticker_id]

        return resolved

    def _process_fill(self, order: _Order, fill_price: float) -> tuple[float, float]:  # type: ignore[override]
        prev_position = self.positions.get(order.ticker_id)
        prev_shares = prev_position.shares if prev_position else 0
        realized_pnl, position_change = super()._process_fill(order, fill_price)

        if self.tbm_enable:
            self._handle_tbm_post_fill(order.ticker_id, prev_shares)

        return realized_pnl, position_change

    def _handle_tbm_post_fill(self, ticker_id: int, prev_shares: int) -> None:
        position = self.positions.get(ticker_id)
        if position is None or position.shares == 0:
            if ticker_id in self.tbm_states:
                self._finalize_tbm_state(ticker_id, label=0)
            return

        new_shares = position.shares
        prev_sign = np.sign(prev_shares)
        new_sign = np.sign(new_shares)
        if ticker_id not in self.tbm_states or prev_shares == 0 or prev_sign != new_sign:
            self._init_tbm_state(ticker_id, float(position.entry_price), new_shares)
