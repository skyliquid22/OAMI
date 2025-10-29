"""Metrics dashboard for FinRL trading environments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("darkgrid")


class MetricsDashboard:
    """Track, analyse, and visualise trading performance metrics."""

    def __init__(
        self,
        title: str = "Trading Performance Dashboard",
        figsize: tuple[int, int] = (14, 8),
    ) -> None:
        self.title = title
        self.figsize = figsize
        self.account_values: List[float] = []
        self.returns: List[float] = []
        self.drawdowns: List[float] = []
        self.actions: List[Dict[str, float]] = []
        self.timestamps: List[pd.Timestamp] = []
        self.metrics: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def update(
        self,
        equity: float,
        action_dict: Dict[str, float],
        timestamp: Optional[pd.Timestamp] = None,
    ) -> None:
        """Append a new observation of equity and actions."""
        if not self.account_values:
            step_return = 0.0
            drawdown = 0.0
        else:
            prev_equity = self.account_values[-1]
            step_return = (equity / prev_equity) - 1.0 if prev_equity != 0 else 0.0
            rolling_max = max(self.account_values)
            drawdown = (equity - rolling_max) / rolling_max if rolling_max != 0 else 0.0

        self.account_values.append(float(equity))
        self.returns.append(float(step_return))
        self.drawdowns.append(float(drawdown))
        self.actions.append(action_dict)
        self.timestamps.append(
            timestamp if timestamp is not None else pd.Timestamp.utcnow()
        )

    def bulk_load(self, df_account_value: pd.DataFrame) -> None:
        """Load equity curve from a DataFrame following FinRL convention."""
        df = df_account_value.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        self.account_values = df["account_value"].astype(float).tolist()
        self.timestamps = df["date"].tolist()
        returns = pd.Series(self.account_values).pct_change().fillna(0.0)
        drawdowns, _ = self._compute_drawdown_curve(pd.Series(self.account_values))
        self.returns = returns.tolist()
        self.drawdowns = drawdowns.tolist()
        self.actions = [{} for _ in self.account_values]

    # ------------------------------------------------------------------
    # Metric Computation
    # ------------------------------------------------------------------

    def compute_metrics(self) -> Dict[str, float]:
        """Compute key performance metrics and store them in ``self.metrics``."""
        equity_series = pd.Series(self.account_values)
        returns = pd.Series(self.returns)
        drawdown_series, max_drawdown = self._compute_drawdown_curve(equity_series)

        total_return = (
            (equity_series.iloc[-1] / equity_series.iloc[0]) - 1.0
            if len(equity_series) > 1
            else 0.0
        )
        ann_return = self._annualize_return(returns)
        sharpe = self._sharpe_ratio(returns)
        sortino = self._sortino_ratio(returns)
        volatility = returns.std(ddof=0) * np.sqrt(252)
        positive_returns = returns[returns > 0]
        win_rate = (
            len(positive_returns) / len(returns[returns != 0])
            if len(returns[returns != 0]) > 0
            else 0.0
        )
        avg_trade_return = returns.mean()
        dd_durations = self._drawdown_durations(drawdown_series)
        avg_dd_duration = float(np.mean(dd_durations)) if dd_durations else 0.0

        self.metrics = {
            "total_return": float(total_return),
            "annualized_return": float(ann_return),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "max_drawdown": float(max_drawdown),
            "volatility": float(volatility),
            "win_rate": float(win_rate),
            "average_trade_return": float(avg_trade_return),
            "average_drawdown_duration": float(avg_dd_duration),
        }

        metrics_path = Path("./results")
        metrics_path.mkdir(parents=True, exist_ok=True)
        with open(metrics_path / "metrics_summary.json", "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=2)

        print(
            f"Total Return: {self.metrics['total_return']:.2%}\n"
            f"Sharpe Ratio: {self.metrics['sharpe_ratio']:.2f}\n"
            f"Max Drawdown: {self.metrics['max_drawdown']:.2%}\n"
            f"Win Rate: {self.metrics['win_rate']:.2%}"
        )

        return self.metrics

    # ------------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------------

    def plot_equity_curve(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        if ax is None:
            _, ax = plt.subplots(figsize=self.figsize)
        times = self.timestamps if self.timestamps else range(len(self.account_values))
        ax.plot(times, self.account_values, label="Equity Curve", color="tab:blue")
        drawdown_series, _ = self._compute_drawdown_curve(
            pd.Series(self.account_values)
        )
        ax.fill_between(
            times,
            np.array(self.account_values) * (1 + drawdown_series),
            self.account_values,
            color="tab:red",
            alpha=0.2,
            label="Drawdown",
        )
        ax.set_title("Equity Curve")
        ax.set_ylabel("Account Value")
        ax.legend()
        return ax

    def plot_drawdown(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        if ax is None:
            _, ax = plt.subplots(figsize=self.figsize)
        times = self.timestamps if self.timestamps else range(len(self.drawdowns))
        ax.plot(times, self.drawdowns, color="tab:red")
        ax.set_title("Drawdown")
        ax.set_ylabel("Drawdown")
        return ax

    def plot_return_histogram(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        if ax is None:
            _, ax = plt.subplots(figsize=self.figsize)
        sns.histplot(self.returns, bins=40, kde=True, ax=ax, color="tab:green")
        ax.set_title("Return Distribution")
        ax.set_xlabel("Step Return")
        return ax

    def plot_action_heatmap(self, ax: Optional[plt.Axes] = None) -> Optional[plt.Axes]:
        if not self.actions or not isinstance(self.actions[0], dict):
            return None
        tickers = sorted({k for action in self.actions for k in action.keys()})
        if not tickers:
            return None
        matrix = np.zeros((len(tickers), len(self.actions)))
        for col, action in enumerate(self.actions):
            for row, ticker in enumerate(tickers):
                matrix[row, col] = action.get(ticker, 0.0)
        if ax is None:
            _, ax = plt.subplots(figsize=self.figsize)
        sns.heatmap(
            matrix, cmap="coolwarm", ax=ax, cbar=True, center=0.0, yticklabels=tickers
        )
        ax.set_title("Action Heatmap")
        ax.set_xlabel("Step")
        ax.set_ylabel("Ticker")
        return ax

    def show_summary_table(self, ax: plt.Axes) -> None:
        if not self.metrics:
            self.compute_metrics()
        metrics_df = pd.DataFrame(self.metrics, index=["Value"]).T
        ax.axis("off")
        table = ax.table(
            cellText=np.round(metrics_df.values, 4),
            rowLabels=metrics_df.index,
            colLabels=metrics_df.columns,
            loc="center",
        )
        table.scale(1, 2)
        ax.set_title("Performance Summary")

    # ------------------------------------------------------------------
    # Composite dashboard
    # ------------------------------------------------------------------

    def render_dashboard(self) -> None:
        fig, axes = plt.subplots(3, 2, figsize=self.figsize, constrained_layout=True)
        fig.suptitle(self.title)

        self.plot_equity_curve(ax=axes[0, 0])
        self.plot_drawdown(ax=axes[1, 0])
        self.plot_return_histogram(ax=axes[0, 1])
        heatmap_ax = self.plot_action_heatmap(ax=axes[1, 1])
        if heatmap_ax is None:
            axes[1, 1].axis("off")
            axes[1, 1].text(
                0.5, 0.5, "Action data unavailable", ha="center", va="center"
            )

        self.compute_metrics()
        self.show_summary_table(ax=axes[2, 1])

        returns_series = pd.Series(self.returns)
        rolling_sharpe = returns_series.rolling(window=50).mean() / (
            returns_series.rolling(window=50).std() + 1e-8
        )
        rolling_vol = returns_series.rolling(window=50).std()
        axes[2, 0].plot(
            self.timestamps if self.timestamps else range(len(rolling_sharpe)),
            rolling_sharpe,
            label="Rolling Sharpe",
            color="tab:purple",
        )
        axes[2, 0].plot(
            self.timestamps if self.timestamps else range(len(rolling_vol)),
            rolling_vol,
            label="Rolling Vol",
            color="tab:orange",
        )
        axes[2, 0].set_title("Rolling Risk Metrics (window=50)")
        axes[2, 0].legend()

        results_dir = Path("./results")
        results_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(results_dir / "dashboard.png", dpi=150)
        plt.show()

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_drawdown_curve(series: pd.Series) -> tuple[pd.Series, float]:
        if series.empty:
            return pd.Series(dtype=float), 0.0
        rolling_max = series.cummax().replace(0, np.nan)
        drawdown = (series - rolling_max) / rolling_max
        drawdown = drawdown.fillna(0.0)
        max_drawdown = drawdown.min()
        return drawdown, float(max_drawdown)

    @staticmethod
    def _annualize_return(daily_returns: pd.Series) -> float:
        compound_growth = (1 + daily_returns).prod()
        n_periods = len(daily_returns)
        if n_periods == 0:
            return 0.0
        return float(compound_growth ** (252 / n_periods) - 1)

    @staticmethod
    def _sharpe_ratio(daily_returns: pd.Series) -> float:
        if daily_returns.std(ddof=0) == 0:
            return 0.0
        return float(daily_returns.mean() / daily_returns.std(ddof=0) * np.sqrt(252))

    @staticmethod
    def _sortino_ratio(daily_returns: pd.Series) -> float:
        downside = daily_returns[daily_returns < 0]
        downside_std = downside.std(ddof=0)
        if downside_std == 0:
            return 0.0
        return float(daily_returns.mean() / downside_std * np.sqrt(252))

    @staticmethod
    def _drawdown_durations(drawdown_series: pd.Series) -> List[int]:
        durations = []
        count = 0
        for value in drawdown_series:
            if value < 0:
                count += 1
            elif count > 0:
                durations.append(count)
                count = 0
        if count > 0:
            durations.append(count)
        return durations
