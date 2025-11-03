"""
Walk-Forward Backtesting System.

Implements walk-forward analysis for strategy validation, which divides
historical data into multiple in-sample (training) and out-of-sample (testing)
periods to simulate realistic trading conditions and avoid overfitting.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.common_logger import DATA_DIR, get_logger

logger = get_logger("walk_forward_backtest")


@dataclass
class WalkForwardWindow:
    """
    Represents a single walk-forward window.

    Attributes:
        window_id: Unique window identifier
        in_sample_start: Start of in-sample period
        in_sample_end: End of in-sample period
        out_sample_start: Start of out-of-sample period
        out_sample_end: End of out-of-sample period
        optimized_params: Parameters optimized on in-sample data
        in_sample_metrics: Performance metrics on in-sample data
        out_sample_metrics: Performance metrics on out-of-sample data
    """

    window_id: int
    in_sample_start: datetime
    in_sample_end: datetime
    out_sample_start: datetime
    out_sample_end: datetime
    optimized_params: Dict[str, Any]
    in_sample_metrics: Dict[str, float]
    out_sample_metrics: Dict[str, float]


@dataclass
class WalkForwardResults:
    """
    Complete walk-forward analysis results.

    Attributes:
        windows: List of walk-forward windows
        aggregate_metrics: Aggregated performance metrics
        efficiency_ratio: Out-of-sample / In-sample performance ratio
        consistency_score: Measure of performance consistency across windows
        total_trades: Total number of trades across all windows
        profitable_windows: Number of profitable out-of-sample windows
        metadata: Additional metadata
    """

    windows: List[WalkForwardWindow]
    aggregate_metrics: Dict[str, float]
    efficiency_ratio: float
    consistency_score: float
    total_trades: int
    profitable_windows: int
    metadata: Dict[str, Any]


class WalkForwardBacktester:
    """
    Walk-forward backtesting engine.

    Features:
    - Configurable window sizes and step sizes
    - Parameter optimization on in-sample data
    - Out-of-sample validation
    - Performance metrics aggregation
    - Visualization of results
    """

    def __init__(
        self,
        strategy_func: Callable,
        data: pd.DataFrame,
        initial_capital: float = 10000.0,
        in_sample_days: int = 90,
        out_sample_days: int = 30,
        step_days: int = 30,
        param_grid: Optional[Dict[str, List[Any]]] = None,
        commission: float = 0.001,
        slippage: float = 0.001,
    ) -> None:
        """
        Initialize the walk-forward backtester.

        Args:
            strategy_func: Trading strategy function to test
            data: Historical market data (OHLCV)
            initial_capital: Starting capital for each window
            in_sample_days: Length of in-sample period in days
            out_sample_days: Length of out-of-sample period in days
            step_days: Step size between windows in days
            param_grid: Parameter grid for optimization
            commission: Commission rate (as decimal)
            slippage: Slippage rate (as decimal)
        """
        self.strategy_func = strategy_func
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.in_sample_days = in_sample_days
        self.out_sample_days = out_sample_days
        self.step_days = step_days
        self.param_grid = param_grid or {}
        self.commission = commission
        self.slippage = slippage
        if "timestamp" in self.data.columns:
            self.data["timestamp"] = pd.to_datetime(self.data["timestamp"])
            self.data = self.data.sort_values("timestamp").reset_index(drop=True)
        elif self.data.index.name == "timestamp" or isinstance(self.data.index, pd.DatetimeIndex):
            self.data = self.data.sort_index()
        else:
            raise ValueError("Data must have 'timestamp' column or DatetimeIndex")
        logger.info(
            "Initialized WalkForwardBacktester with %s data points, in-sample: %sd, out-sample: %sd, step: %sd",
            len(self.data),
            in_sample_days,
            out_sample_days,
            step_days,
        )

    def _generate_windows(self) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """
        Generate walk-forward windows.

        Returns:
            List of (in_sample_start, in_sample_end, out_sample_start, out_sample_end) tuples
        """
        windows = []
        if "timestamp" in self.data.columns:
            start_date = self.data["timestamp"].min()
            end_date = self.data["timestamp"].max()
        else:
            start_date = self.data.index.min()
            end_date = self.data.index.max()
        current_start = start_date
        while True:
            in_sample_start = current_start
            in_sample_end = in_sample_start + timedelta(days=self.in_sample_days)
            out_sample_start = in_sample_end
            out_sample_end = out_sample_start + timedelta(days=self.out_sample_days)
            if out_sample_end > end_date:
                break
            windows.append((in_sample_start, in_sample_end, out_sample_start, out_sample_end))
            current_start += timedelta(days=self.step_days)
        logger.info("Generated %s walk-forward windows", len(windows))
        return windows

    def _get_data_for_period(self, start: datetime, end: datetime) -> pd.DataFrame:
        """
        Extract data for a specific period.

        Args:
            start: Period start
            end: Period end

        Returns:
            DataFrame with data for the period
        """
        if "timestamp" in self.data.columns:
            mask = (self.data["timestamp"] >= start) & (self.data["timestamp"] < end)
            return self.data[mask].copy()
        else:
            return self.data.loc[start:end].copy()

    def _optimize_parameters(self, in_sample_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Optimize strategy parameters on in-sample data.

        Args:
            in_sample_data: In-sample data

        Returns:
            Optimized parameters
        """
        if not self.param_grid:
            return {}
        best_params = {}
        best_sharpe = -np.inf
        param_combinations = self._generate_param_combinations()
        logger.debug("Testing %s parameter combinations", len(param_combinations))
        for params in param_combinations:
            metrics = self._run_backtest(in_sample_data, params)
            sharpe = metrics.get("sharpe_ratio", -np.inf)
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = params.copy()
        logger.debug("Best parameters: %s (Sharpe: %.3f)", best_params, best_sharpe)
        return best_params

    def _generate_param_combinations(self) -> List[Dict[str, Any]]:
        """
        Generate all parameter combinations from param grid.

        Returns:
            List of parameter dictionaries
        """
        if not self.param_grid:
            return [{}]
        import itertools

        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        combinations = []
        for combination in itertools.product(*values):
            param_dict = dict(zip(keys, combination))
            combinations.append(param_dict)
        return combinations

    def _run_backtest(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, float]:
        """
        Run backtest on data with given parameters.

        Args:
            data: Market data
            params: Strategy parameters

        Returns:
            Performance metrics
        """
        signals = self.strategy_func(data, **params)
        trades = self._simulate_trades(data, signals)
        metrics = self._calculate_metrics(trades, data)
        return metrics

    def _simulate_trades(self, data: pd.DataFrame, signals: pd.Series) -> List[Dict[str, Any]]:
        """
        Simulate trades based on signals.

        Args:
            data: Market data
            signals: Trading signals (1=buy, -1=sell, 0=hold)

        Returns:
            List of trade dictionaries
        """
        trades = []
        position = 0
        entry_price = 0.0
        capital = self.initial_capital
        for i in range(len(data)):
            signal = signals.iloc[i] if i < len(signals) else 0
            price = data.iloc[i]["close"]
            if signal > 0 and position == 0:
                execution_price = price * (1 + self.slippage)
                shares = capital / execution_price
                commission_cost = capital * self.commission
                capital -= commission_cost
                position = 1
                entry_price = execution_price
                trades.append(
                    {
                        "timestamp": (
                            data.iloc[i]["timestamp"]
                            if "timestamp" in data.columns
                            else data.index[i]
                        ),
                        "type": "buy",
                        "price": execution_price,
                        "shares": shares,
                        "capital": capital,
                        "commission": commission_cost,
                    }
                )
            elif signal < 0 and position == 1:
                execution_price = price * (1 - self.slippage)
                shares = trades[-1]["shares"]
                proceeds = shares * execution_price
                commission_cost = proceeds * self.commission
                proceeds -= commission_cost
                capital = proceeds
                pnl = proceeds - shares * entry_price
                pnl_pct = (execution_price - entry_price) / entry_price * 100
                position = 0
                trades.append(
                    {
                        "timestamp": (
                            data.iloc[i]["timestamp"]
                            if "timestamp" in data.columns
                            else data.index[i]
                        ),
                        "type": "sell",
                        "price": execution_price,
                        "shares": shares,
                        "capital": capital,
                        "commission": commission_cost,
                        "pnl": pnl,
                        "pnl_pct": pnl_pct,
                    }
                )
        return trades

    def _calculate_metrics(
        self, trades: List[Dict[str, Any]], data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate performance metrics from trades.

        Args:
            trades: List of trades
            data: Market data

        Returns:
            Dictionary of metrics
        """
        if not trades:
            return {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "num_trades": 0,
                "avg_trade_return": 0.0,
            }
        final_capital = trades[-1]["capital"] if trades else self.initial_capital
        total_return = (final_capital - self.initial_capital) / self.initial_capital * 100
        sell_trades = [t for t in trades if t["type"] == "sell"]
        num_trades = len(sell_trades)
        if num_trades > 0:
            returns = [t["pnl_pct"] for t in sell_trades]
            avg_trade_return = np.mean(returns)
            std_trade_return = np.std(returns) if len(returns) > 1 else 0.0
            if std_trade_return > 0:
                sharpe_ratio = avg_trade_return / std_trade_return * np.sqrt(252)
            else:
                sharpe_ratio = 0.0
            winning_trades = sum((1 for t in sell_trades if t["pnl"] > 0))
            win_rate = winning_trades / num_trades * 100
            equity_curve = [self.initial_capital]
            for trade in trades:
                equity_curve.append(trade["capital"])
            max_drawdown = self._calculate_max_drawdown(equity_curve)
        else:
            avg_trade_return = 0.0
            sharpe_ratio = 0.0
            win_rate = 0.0
            max_drawdown = 0.0
        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "num_trades": num_trades,
            "avg_trade_return": avg_trade_return,
        }

    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """
        Calculate maximum drawdown from equity curve.

        Args:
            equity_curve: List of equity values

        Returns:
            Maximum drawdown percentage
        """
        if not equity_curve:
            return 0.0
        peak = equity_curve[0]
        max_dd = 0.0
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def run(self) -> WalkForwardResults:
        """
        Run walk-forward analysis.

        Returns:
            Walk-forward results
        """
        logger.info("Starting walk-forward analysis...")
        window_specs = self._generate_windows()
        if not window_specs:
            raise ValueError("No valid windows generated. Check data range and window sizes.")
        windows = []
        total_trades = 0
        profitable_windows = 0
        for i, (in_start, in_end, out_start, out_end) in enumerate(window_specs):
            logger.info(
                "Processing window %s/%s: IS %s to %s, OOS %s to %s",
                i + 1,
                len(window_specs),
                in_start.date(),
                in_end.date(),
                out_start.date(),
                out_end.date(),
            )
            in_sample_data = self._get_data_for_period(in_start, in_end)
            out_sample_data = self._get_data_for_period(out_start, out_end)
            if len(in_sample_data) == 0 or len(out_sample_data) == 0:
                logger.warning("Skipping window %s: insufficient data", i + 1)
                continue
            optimized_params = self._optimize_parameters(in_sample_data)
            in_sample_metrics = self._run_backtest(in_sample_data, optimized_params)
            out_sample_metrics = self._run_backtest(out_sample_data, optimized_params)
            window = WalkForwardWindow(
                window_id=i + 1,
                in_sample_start=in_start,
                in_sample_end=in_end,
                out_sample_start=out_start,
                out_sample_end=out_end,
                optimized_params=optimized_params,
                in_sample_metrics=in_sample_metrics,
                out_sample_metrics=out_sample_metrics,
            )
            windows.append(window)
            total_trades += out_sample_metrics["num_trades"]
            if out_sample_metrics["total_return"] > 0:
                profitable_windows += 1
        aggregate_metrics = self._calculate_aggregate_metrics(windows)
        efficiency_ratio = self._calculate_efficiency_ratio(windows)
        consistency_score = self._calculate_consistency_score(windows)
        results = WalkForwardResults(
            windows=windows,
            aggregate_metrics=aggregate_metrics,
            efficiency_ratio=efficiency_ratio,
            consistency_score=consistency_score,
            total_trades=total_trades,
            profitable_windows=profitable_windows,
            metadata={
                "total_windows": len(windows),
                "in_sample_days": self.in_sample_days,
                "out_sample_days": self.out_sample_days,
                "step_days": self.step_days,
                "initial_capital": self.initial_capital,
                "commission": self.commission,
                "slippage": self.slippage,
            },
        )
        logger.info(
            "Walk-forward analysis complete. Windows: %s, Profitable: %s, Efficiency: %.2f",
            len(windows),
            profitable_windows,
            efficiency_ratio,
        )
        return results

    def _calculate_aggregate_metrics(self, windows: List[WalkForwardWindow]) -> Dict[str, float]:
        """Calculate aggregate metrics across all windows."""
        if not windows:
            return {}
        oos_returns = [w.out_sample_metrics["total_return"] for w in windows]
        oos_sharpes = [w.out_sample_metrics["sharpe_ratio"] for w in windows]
        oos_drawdowns = [w.out_sample_metrics["max_drawdown"] for w in windows]
        oos_win_rates = [w.out_sample_metrics["win_rate"] for w in windows]
        return {
            "avg_return": np.mean(oos_returns),
            "std_return": np.std(oos_returns),
            "avg_sharpe": np.mean(oos_sharpes),
            "avg_drawdown": np.mean(oos_drawdowns),
            "avg_win_rate": np.mean(oos_win_rates),
            "min_return": np.min(oos_returns),
            "max_return": np.max(oos_returns),
        }

    def _calculate_efficiency_ratio(self, windows: List[WalkForwardWindow]) -> float:
        """
        Calculate efficiency ratio (OOS / IS performance).

        A ratio close to 1.0 indicates minimal overfitting.
        """
        if not windows:
            return 0.0
        is_returns = [w.in_sample_metrics["total_return"] for w in windows]
        oos_returns = [w.out_sample_metrics["total_return"] for w in windows]
        avg_is = np.mean(is_returns)
        avg_oos = np.mean(oos_returns)
        if avg_is == 0:
            return 0.0
        return avg_oos / avg_is

    def _calculate_consistency_score(self, windows: List[WalkForwardWindow]) -> float:
        """
        Calculate consistency score based on return variability.

        Higher score indicates more consistent performance.
        """
        if not windows:
            return 0.0
        oos_returns = [w.out_sample_metrics["total_return"] for w in windows]
        if len(oos_returns) < 2:
            return 0.0
        mean_return = np.mean(oos_returns)
        std_return = np.std(oos_returns)
        if std_return == 0:
            return 100.0
        cv = std_return / abs(mean_return) if mean_return != 0 else float("inf")
        consistency = max(0, 100 - cv * 10)
        return consistency

    def save_results(self, results: WalkForwardResults, output_dir: Optional[Path] = None) -> Path:
        """
        Save walk-forward results to file.

        Args:
            results: Walk-forward results
            output_dir: Output directory

        Returns:
            Path to saved results file
        """
        if output_dir is None:
            output_dir = DATA_DIR / "backtest_results" / "walk_forward"
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        results_file = output_dir / f"wf_results_{timestamp}.json"
        results_dict = {
            "windows": [
                {
                    **asdict(w),
                    "in_sample_start": w.in_sample_start.isoformat(),
                    "in_sample_end": w.in_sample_end.isoformat(),
                    "out_sample_start": w.out_sample_start.isoformat(),
                    "out_sample_end": w.out_sample_end.isoformat(),
                }
                for w in results.windows
            ],
            "aggregate_metrics": results.aggregate_metrics,
            "efficiency_ratio": results.efficiency_ratio,
            "consistency_score": results.consistency_score,
            "total_trades": results.total_trades,
            "profitable_windows": results.profitable_windows,
            "metadata": results.metadata,
        }
        with results_file.open("w", encoding="utf-8") as f:
            json.dump(results_dict, f, indent=2)
        logger.info("Saved walk-forward results to %s", results_file)
        return results_file

    def plot_results(self, results: WalkForwardResults, output_file: Optional[Path] = None) -> None:
        """
        Plot walk-forward results.

        Args:
            results: Walk-forward results
            output_file: Optional output file path
        """
        (fig, axes) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Walk-Forward Analysis Results", fontsize=16)
        window_ids = [w.window_id for w in results.windows]
        is_returns = [w.in_sample_metrics["total_return"] for w in results.windows]
        oos_returns = [w.out_sample_metrics["total_return"] for w in results.windows]
        oos_sharpes = [w.out_sample_metrics["sharpe_ratio"] for w in results.windows]
        oos_drawdowns = [w.out_sample_metrics["max_drawdown"] for w in results.windows]
        axes[0, 0].plot(window_ids, is_returns, "o-", label="In-Sample", alpha=0.7)
        axes[0, 0].plot(window_ids, oos_returns, "s-", label="Out-of-Sample", alpha=0.7)
        axes[0, 0].axhline(y=0, color="r", linestyle="--", alpha=0.3)
        axes[0, 0].set_xlabel("Window")
        axes[0, 0].set_ylabel("Return (%)")
        axes[0, 0].set_title("In-Sample vs Out-of-Sample Returns")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 1].bar(window_ids, oos_sharpes, alpha=0.7)
        axes[0, 1].axhline(y=0, color="r", linestyle="--", alpha=0.3)
        axes[0, 1].set_xlabel("Window")
        axes[0, 1].set_ylabel("Sharpe Ratio")
        axes[0, 1].set_title("Out-of-Sample Sharpe Ratio")
        axes[0, 1].grid(True, alpha=0.3)
        axes[1, 0].bar(window_ids, oos_drawdowns, alpha=0.7, color="red")
        axes[1, 0].set_xlabel("Window")
        axes[1, 0].set_ylabel("Max Drawdown (%)")
        axes[1, 0].set_title("Out-of-Sample Maximum Drawdown")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 1].axis("off")
        summary_text = f"Walk-Forward Analysis Summary\n{'=' * 40}\nTotal Windows: {len(results.windows)}\nProfitable Windows: {results.profitable_windows}\nWin Rate: {results.profitable_windows / len(results.windows) * 100:.1f}%\n\nAverage OOS Return: {results.aggregate_metrics['avg_return']:.2f}%\nAverage OOS Sharpe: {results.aggregate_metrics['avg_sharpe']:.2f}\nAverage OOS Drawdown: {results.aggregate_metrics['avg_drawdown']:.2f}%\n\nEfficiency Ratio: {results.efficiency_ratio:.2f}\nConsistency Score: {results.consistency_score:.1f}/100\n\nTotal Trades: {results.total_trades}"
        axes[1, 1].text(
            0.1, 0.5, summary_text, fontsize=11, family="monospace", verticalalignment="center"
        )
        plt.tight_layout()
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            logger.info("Saved walk-forward plot to %s", output_file)
        else:
            plt.show()
        plt.close()
