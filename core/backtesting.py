"""
Enhanced Backtesting Analytics Module

Provides advanced backtesting analysis including:
- Walk-Forward Analysis
- Strategy Comparison Matrix
- Trade Analysis Dashboard
- Performance Heatmaps
- Correlation Analysis
- Optimization Engine with Genetic Algorithms
"""

import json
import logging
import os
import random
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import product
from typing import Any, Dict, List, Optional

import numpy as np

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Container for backtest results"""

    run_id: str
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    trades: List[Dict[str, Any]]
    portfolio_history: List[float]
    start_date: str
    end_date: str
    duration_days: int


class EnhancedBacktesting:
    """Enhanced backtesting analytics engine"""

    def __init__(self, data_dir: str = None) -> None:
        """
        Initialize enhanced backtesting

        Args:
            data_dir: Directory containing backtest results
        """
        self.data_dir = data_dir or os.path.join(
            os.path.dirname(__file__), "..", "data", "backtest_results"
        )
        self.results_cache: dict = {}

    def load_backtest_results(self, run_ids: List[str] = None) -> List[BacktestResult]:
        """
        Load backtest results from files

        Args:
            run_ids: Specific run IDs to load, or None for all

        Returns:
            List of BacktestResult objects
        """
        results: list = []
        try:
            if not os.path.exists(self.data_dir):
                logger.warning("Backtest data directory not found: %s", self.data_dir)
                return results
            all_runs = [
                d
                for d in os.listdir(self.data_dir)
                if os.path.isdir(os.path.join(self.data_dir, d))
            ]
            if run_ids:
                all_runs = [r for r in all_runs if r in run_ids]
            for run_id in all_runs:
                try:
                    result = self._load_single_result(run_id)
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error("Error loading result %s: %s", run_id, e)
                    continue
        except Exception as e:
            logger.error("Error loading backtest results: %s", e)
        return results

    def _load_single_result(self, run_id: str) -> Optional[BacktestResult]:
        """Load a single backtest result"""
        if run_id in self.results_cache:
            return self.results_cache[run_id]
        run_dir = os.path.join(self.data_dir, run_id)
        metrics_file = os.path.join(run_dir, f"backtest_metrics_{run_id}.json")
        if not os.path.exists(metrics_file):
            return None
        with open(metrics_file, "r") as f:
            metrics_data = json.load(f)
        trades_file = os.path.join(run_dir, f"backtest_trades_{run_id}.json")
        trades: list = []
        if os.path.exists(trades_file):
            with open(trades_file, "r") as f:
                trades = json.load(f)
        portfolio_file = os.path.join(run_dir, f"backtest_portfolio_{run_id}.json")
        portfolio_history: list = []
        if os.path.exists(portfolio_file):
            with open(portfolio_file, "r") as f:
                portfolio_data = json.load(f)
                portfolio_history = portfolio_data.get("portfolio_values", [])
        parameters = metrics_data.get("parameters", {})
        start_date = metrics_data.get("start_date", "")
        end_date = metrics_data.get("end_date", "")
        duration_days = 0
        try:
            if start_date and end_date:
                start = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
                end = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                duration_days = (end - start).days
        except:
            pass
        result = BacktestResult(
            run_id=run_id,
            parameters=parameters,
            metrics=metrics_data.get("metrics", {}),
            trades=trades,
            portfolio_history=portfolio_history,
            start_date=start_date,
            end_date=end_date,
            duration_days=duration_days,
        )
        self.results_cache[run_id] = result
        return result

    def walk_forward_analysis(
        self,
        train_period_days: int = 180,
        test_period_days: int = 60,
        step_days: int = 30,
        start_date: str = None,
        end_date: str = None,
    ) -> Dict[str, Any]:
        """
        Perform walk-forward analysis

        Args:
            train_period_days: Training period length
            test_period_days: Testing period length
            step_days: Step size between windows
            start_date: Analysis start date
            end_date: Analysis end date

        Returns:
            Dict containing walk-forward analysis results
        """
        try:
            results = self.load_backtest_results()
            if not results:
                return {"error": "No backtest results available"}  # type: ignore[dict-item]
            windows: list = []
            window_results = []
            results.sort(key=lambda x: x.start_date)
            if start_date and end_date:
                current_date = datetime.fromisoformat(start_date)
                end_dt = datetime.fromisoformat(end_date)
                while current_date + timedelta(days=train_period_days + test_period_days) <= end_dt:
                    train_start = current_date
                    train_end = current_date + timedelta(days=train_period_days)
                    test_start = train_end
                    test_end = test_start + timedelta(days=test_period_days)
                    windows.append(
                        {
                            "train_start": train_start.isoformat(),
                            "train_end": train_end.isoformat(),
                            "test_start": test_start.isoformat(),
                            "test_end": test_end.isoformat(),
                        }
                    )
                    current_date += timedelta(days=step_days)
            for i, window in enumerate(windows):
                window_train_results: list = []
                window_test_results = []
                for result in results:
                    result_start = datetime.fromisoformat(result.start_date.replace("Z", "+00:00"))
                    result_end = datetime.fromisoformat(result.end_date.replace("Z", "+00:00"))
                    train_start = datetime.fromisoformat(window["train_start"])
                    train_end = datetime.fromisoformat(window["train_end"])
                    test_start = datetime.fromisoformat(window["test_start"])
                    test_end = datetime.fromisoformat(window["test_end"])
                    if result_start >= train_start and result_end <= train_end:
                        window_train_results.append(result)
                    if result_start >= test_start and result_end <= test_end:
                        window_test_results.append(result)
                train_metrics = self._calculate_aggregate_metrics(window_train_results)
                test_metrics = self._calculate_aggregate_metrics(window_test_results)
                window_results.append(
                    {
                        "window_id": i + 1,
                        "window": window,
                        "train_results_count": len(window_train_results),
                        "test_results_count": len(window_test_results),
                        "train_metrics": train_metrics,
                        "test_metrics": test_metrics,
                        "out_of_sample_performance": test_metrics.get("total_return", 0)
                        - train_metrics.get("total_return", 0),
                    }
                )
            oos_returns = [w["test_metrics"].get("total_return", 0) for w in window_results]
            is_returns = [w["train_metrics"].get("total_return", 0) for w in window_results]
            return {
                "analysis_type": "walk_forward",
                "parameters": {
                    "train_period_days": train_period_days,
                    "test_period_days": test_period_days,
                    "step_days": step_days,
                },
                "windows": window_results,
                "summary": {
                    "total_windows": len(window_results),
                    "avg_oos_return": np.mean(oos_returns) if oos_returns else 0,
                    "avg_is_return": np.mean(is_returns) if is_returns else 0,
                    "oos_consistency": (
                        len([r for r in oos_returns if r > 0]) / len(oos_returns)
                        if oos_returns
                        else 0
                    ),
                    "oos_volatility": np.std(oos_returns) if len(oos_returns) > 1 else 0,
                    "degradation": (
                        np.mean(is_returns) - np.mean(oos_returns)
                        if is_returns and oos_returns
                        else 0
                    ),
                },
            }
        except Exception as e:
            logger.error("Error in walk-forward analysis: %s", e)
            return {"error": str(e)}  # type: ignore[dict-item]

    def strategy_comparison_matrix(self, metrics: List[str] = None) -> Dict[str, Any]:
        """
        Create strategy comparison matrix

        Args:
            metrics: List of metrics to compare

        Returns:
            Dict containing comparison matrix
        """
        try:
            results = self.load_backtest_results()
            if not results:
                return {"error": "No backtest results available"}  # type: ignore[dict-item]
            if not metrics:
                metrics = [
                    "total_return",
                    "sharpe_ratio",
                    "max_drawdown",
                    "win_rate",
                    "profit_factor",
                ]
            comparison_data: list = []
            for result in results:
                row = {
                    "run_id": result.run_id,
                    "start_date": result.start_date,
                    "end_date": result.end_date,
                    "duration_days": result.duration_days,
                    "parameters": result.parameters,
                }
                for metric in metrics:
                    row[metric] = result.metrics.get(metric, 0)
                comparison_data.append(row)
            rankings = {}
            for metric in metrics:
                values = [(i, row[metric]) for (i, row) in enumerate(comparison_data)]
                reverse = metric not in ["max_drawdown", "volatility"]
                values.sort(key=lambda x: x[1], reverse=reverse)
                rankings[metric] = {
                    "ranking": [
                        {"index": i, "value": v, "rank": rank + 1}
                        for (rank, (i, v)) in enumerate(values)
                    ],
                    "best_run": comparison_data[values[0][0]]["run_id"],
                    "best_value": values[0][1],
                    "worst_run": comparison_data[values[-1][0]]["run_id"],
                    "worst_value": values[-1][1],
                }
            for i, row in enumerate(comparison_data):
                composite_score = 0
                for metric in metrics:
                    rank = next((r["rank"] for r in rankings[metric]["ranking"] if r["index"] == i))
                    normalized_rank = (len(comparison_data) - rank + 1) / len(comparison_data)
                    composite_score += normalized_rank
                row["composite_score"] = composite_score / len(metrics)
            comparison_data.sort(key=lambda x: x["composite_score"], reverse=True)
            return {
                "comparison_matrix": comparison_data,
                "metrics_analyzed": metrics,
                "rankings": rankings,
                "best_overall": comparison_data[0]["run_id"] if comparison_data else None,
                "total_strategies": len(comparison_data),
            }
        except Exception as e:
            logger.error("Error in strategy comparison: %s", e)
            return {"error": str(e)}  # type: ignore[dict-item]

    def trade_analysis_dashboard(self, run_id: str = None) -> Dict[str, Any]:
        """
        Create comprehensive trade analysis

        Args:
            run_id: Specific run to analyze, or None for aggregate

        Returns:
            Dict containing trade analysis
        """
        try:
            if run_id:
                results = [self._load_single_result(run_id)]
                results = [r for r in results if r is not None]
            else:
                results = self.load_backtest_results()
            if not results:
                return {"error": "No backtest results available"}  # type: ignore[dict-item]
            all_trades: list = []
            for result in results:
                for trade in result.trades:
                    trade_copy = trade.copy()
                    trade_copy["run_id"] = result.run_id
                    all_trades.append(trade_copy)
            if not all_trades:
                return {"error": "No trades found"}  # type: ignore[dict-item]
            winning_trades = [t for t in all_trades if t.get("pnl", 0) > 0]
            losing_trades = [t for t in all_trades if t.get("pnl", 0) < 0]
            total_trades = len(all_trades)
            win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
            total_pnl = sum((t.get("pnl", 0) for t in all_trades))
            avg_win = np.mean([t.get("pnl", 0) for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t.get("pnl", 0) for t in losing_trades]) if losing_trades else 0
            durations: list = []
            for trade in all_trades:
                if "entry_time" in trade and "exit_time" in trade:
                    try:
                        entry = datetime.fromisoformat(trade["entry_time"].replace("Z", "+00:00"))
                        exit = datetime.fromisoformat(trade["exit_time"].replace("Z", "+00:00"))
                        duration = (exit - entry).total_seconds() / 3600
                        durations.append(duration)
                    except:
                        continue
            avg_duration = np.mean(durations) if durations else 0
            asset_stats = {}
            for trade in all_trades:
                asset = trade.get("symbol", "Unknown")
                if asset not in asset_stats:
                    asset_stats[asset] = {"trades": 0, "wins": 0, "total_pnl": 0, "avg_pnl": 0}
                asset_stats[asset]["trades"] += 1
                asset_stats[asset]["total_pnl"] += trade.get("pnl", 0)
                if trade.get("pnl", 0) > 0:
                    asset_stats[asset]["wins"] += 1
            for asset, stats in asset_stats.items():
                stats["win_rate"] = stats["wins"] / stats["trades"] if stats["trades"] > 0 else 0
                stats["avg_pnl"] = (
                    stats["total_pnl"] / stats["trades"] if stats["trades"] > 0 else 0
                )
            hourly_stats = {}
            daily_stats = {}
            monthly_stats = {}
            for trade in all_trades:
                if "entry_time" in trade:
                    try:
                        entry_time = datetime.fromisoformat(
                            trade["entry_time"].replace("Z", "+00:00")
                        )
                        hour = entry_time.hour
                        day = entry_time.strftime("%A")
                        month = entry_time.strftime("%B")
                        if hour not in hourly_stats:
                            hourly_stats[hour] = {"trades": 0, "total_pnl": 0}
                        hourly_stats[hour]["trades"] += 1
                        hourly_stats[hour]["total_pnl"] += trade.get("pnl", 0)
                        if day not in daily_stats:
                            daily_stats[day] = {"trades": 0, "total_pnl": 0}
                        daily_stats[day]["trades"] += 1
                        daily_stats[day]["total_pnl"] += trade.get("pnl", 0)
                        if month not in monthly_stats:
                            monthly_stats[month] = {"trades": 0, "total_pnl": 0}
                        monthly_stats[month]["trades"] += 1
                        monthly_stats[month]["total_pnl"] += trade.get("pnl", 0)
                    except:
                        continue
            return {
                "trade_summary": {
                    "total_trades": total_trades,
                    "winning_trades": len(winning_trades),
                    "losing_trades": len(losing_trades),
                    "win_rate": win_rate,
                    "total_pnl": total_pnl,
                    "avg_win": avg_win,
                    "avg_loss": avg_loss,
                    "profit_factor": abs(avg_win / avg_loss) if avg_loss != 0 else float("inf"),
                    "avg_duration_hours": avg_duration,
                },
                "asset_analysis": asset_stats,
                "time_analysis": {
                    "hourly": hourly_stats,
                    "daily": daily_stats,
                    "monthly": monthly_stats,
                },
                "trade_distribution": {
                    "pnl_histogram": self._create_pnl_histogram(all_trades),
                    "duration_histogram": self._create_duration_histogram(durations),
                },
                "recent_trades": sorted(
                    all_trades, key=lambda x: x.get("entry_time", ""), reverse=True
                )[:20],
            }
        except Exception as e:
            logger.error("Error in trade analysis: %s", e)
            return {"error": str(e)}  # type: ignore[dict-item]

    def performance_heatmaps(self) -> Dict[str, Any]:
        """
        Generate performance heatmaps

        Returns:
            Dict containing heatmap data
        """
        try:
            results = self.load_backtest_results()
            if not results:
                return {"error": "No backtest results available"}  # type: ignore[dict-item]
            monthly_returns = {}
            yearly_returns = {}
            for result in results:
                if not result.portfolio_history:
                    continue
                try:
                    start_date = datetime.fromisoformat(result.start_date.replace("Z", "+00:00"))
                    year = start_date.year
                    month = start_date.month
                    if len(result.portfolio_history) > 1:
                        initial_value = result.portfolio_history[0]
                        final_value = result.portfolio_history[-1]
                        return_pct = (
                            (final_value - initial_value) / initial_value
                            if initial_value != 0
                            else 0
                        )
                    else:
                        return_pct = 0
                    if year not in monthly_returns:
                        monthly_returns[year] = {}
                    if month not in monthly_returns[year]:
                        monthly_returns[year][month] = []
                    monthly_returns[year][month].append(return_pct)
                    if year not in yearly_returns:
                        yearly_returns[year] = []
                    yearly_returns[year].append(return_pct)
                except:
                    continue
            monthly_heatmap = {}
            for year, months in monthly_returns.items():
                monthly_heatmap[year] = {}
                for month, returns in months.items():
                    monthly_heatmap[year][month] = np.mean(returns)
            param_performance = {}
            for result in results:
                param_key = json.dumps(result.parameters, sort_keys=True)
                if param_key not in param_performance:
                    param_performance[param_key] = {
                        "parameters": result.parameters,
                        "returns": [],
                        "sharpe_ratios": [],
                        "max_drawdowns": [],
                    }
                param_performance[param_key]["returns"].append(  # type: ignore[str]
                    result.metrics.get("total_return", 0)
                )
                param_performance[param_key]["sharpe_ratios"].append(  # type: ignore[str]
                    result.metrics.get("sharpe_ratio", 0)
                )
                param_performance[param_key]["max_drawdowns"].append(  # type: ignore[str]
                    result.metrics.get("max_drawdown", 0)
                )
            param_heatmap = {}
            for param_key, data in param_performance.items():
                param_heatmap[param_key] = {
                    "parameters": data["parameters"],
                    "avg_return": np.mean(data["returns"]),
                    "avg_sharpe": np.mean(data["sharpe_ratios"]),
                    "avg_drawdown": np.mean(data["max_drawdowns"]),
                    "consistency": (
                        len([r for r in data["returns"] if r > 0]) / len(data["returns"])
                        if data["returns"]
                        else 0
                    ),
                }
            return {
                "monthly_heatmap": monthly_heatmap,
                "yearly_summary": {
                    year: np.mean(returns) for (year, returns) in yearly_returns.items()
                },
                "parameter_heatmap": param_heatmap,
                "best_month": (
                    max(monthly_heatmap.items(), key=lambda x: max(x[1].values()) if x[1] else 0)
                    if monthly_heatmap
                    else None
                ),
                "worst_month": (
                    min(monthly_heatmap.items(), key=lambda x: min(x[1].values()) if x[1] else 0)
                    if monthly_heatmap
                    else None
                ),
            }
        except Exception as e:
            logger.error("Error generating heatmaps: %s", e)
            return {"error": str(e)}  # type: ignore[dict-item]

    def correlation_analysis(self) -> Dict[str, Any]:
        """
        Perform correlation analysis between strategies

        Returns:
            Dict containing correlation analysis
        """
        try:
            results = self.load_backtest_results()
            if len(results) < 2:
                return {"error": "Need at least 2 backtest results for correlation analysis"}  # type: ignore[dict-item]
            returns_data = {}
            for result in results:
                if len(result.portfolio_history) > 1:
                    portfolio_values = np.array(result.portfolio_history)  # type: ignore[assignment]
                    returns = np.diff(portfolio_values) / portfolio_values[:-1]
                    returns_data[result.run_id] = returns
            if len(returns_data) < 2:
                return {"error": "Insufficient data for correlation analysis"}  # type: ignore[dict-item]
            min_length = min((len(returns) for returns in returns_data.values()))
            aligned_returns = {}
            for run_id, returns in returns_data.items():
                aligned_returns[run_id] = returns[:min_length]
            run_ids = list(aligned_returns.keys())
            n_strategies = len(run_ids)
            correlation_matrix = np.zeros((n_strategies, n_strategies))
            for i, run_id_i in enumerate(run_ids):
                for j, run_id_j in enumerate(run_ids):
                    if i == j:
                        correlation_matrix[i, j] = 1.0
                    else:
                        corr = np.corrcoef(aligned_returns[run_id_i], aligned_returns[run_id_j])[
                            0, 1
                        ]
                        correlation_matrix[i, j] = corr if not np.isnan(corr) else 0.0
            high_correlations: list = []
            for i in range(n_strategies):
                for j in range(i + 1, n_strategies):
                    corr = correlation_matrix[i, j]
                    if abs(corr) > 0.7:
                        high_correlations.append(
                            {
                                "strategy_1": run_ids[i],
                                "strategy_2": run_ids[j],
                                "correlation": corr,
                            }
                        )
            avg_correlation = np.mean(correlation_matrix[np.triu_indices(n_strategies, k=1)])
            equal_weight_returns = np.mean([aligned_returns[run_id] for run_id in run_ids], axis=0)
            portfolio_volatility = np.std(equal_weight_returns) * np.sqrt(252)
            avg_individual_vol = np.mean(
                [np.std(aligned_returns[run_id]) * np.sqrt(252) for run_id in run_ids]
            )
            diversification_ratio = (
                avg_individual_vol / portfolio_volatility if portfolio_volatility != 0 else 1
            )
            return {
                "correlation_matrix": {
                    "strategies": run_ids,
                    "matrix": correlation_matrix.tolist(),
                },
                "high_correlations": high_correlations,
                "diversification_analysis": {
                    "avg_correlation": avg_correlation,
                    "portfolio_volatility": portfolio_volatility,
                    "avg_individual_volatility": avg_individual_vol,
                    "diversification_ratio": diversification_ratio,
                    "diversification_benefit": (diversification_ratio - 1) * 100,
                },
                "clustering_suggestions": self._suggest_strategy_clusters(
                    correlation_matrix, run_ids
                ),
            }
        except Exception as e:
            logger.error("Error in correlation analysis: %s", e)
            return {"error": str(e)}  # type: ignore[dict-item]

    def optimization_engine(
        self,
        parameter_ranges: Dict[str, List],
        objective: str = "sharpe_ratio",
        method: str = "genetic",
        max_iterations: int = 50,
    ) -> Dict[str, Any]:
        """
        Parameter optimization engine

        Args:
            parameter_ranges: Dict of parameter names to ranges
            objective: Optimization objective ('sharpe_ratio', 'total_return', etc.)
            method: Optimization method ('genetic', 'grid', 'random')
            max_iterations: Maximum iterations for genetic algorithm

        Returns:
            Dict containing optimization results
        """
        try:
            if method == "genetic":
                return self._genetic_optimization(parameter_ranges, objective, max_iterations)
            elif method == "grid":
                return self._grid_search_optimization(parameter_ranges, objective)
            elif method == "random":
                return self._random_search_optimization(parameter_ranges, objective, max_iterations)
            else:
                return {"error": f"Unknown optimization method: {method}"}  # type: ignore[dict-item]
        except Exception as e:
            logger.error("Error in optimization: %s", e)
            return {"error": str(e)}  # type: ignore[dict-item]

    def _genetic_optimization(
        self, parameter_ranges: Dict[str, List], objective: str, max_iterations: int
    ) -> Dict[str, Any]:
        """Genetic algorithm optimization"""
        results = self.load_backtest_results()
        if not results:
            return {"error": "No backtest results for optimization"}  # type: ignore[dict-item]
        param_performance = {}
        for result in results:
            param_key = json.dumps(result.parameters, sort_keys=True)
            param_performance[param_key] = result.metrics.get(objective, 0)
        population_size = 20
        mutation_rate = 0.1
        crossover_rate = 0.8
        population: list = []
        for _ in range(population_size):
            individual = {}
            for param, values in parameter_ranges.items():
                individual[param] = random.choice(values)
            population.append(individual)
        best_fitness_history: list = []
        avg_fitness_history = []
        for generation in range(max_iterations):
            fitness_scores: list = []
            for individual in population:
                param_key = json.dumps(individual, sort_keys=True)
                fitness = param_performance.get(param_key, 0)
                fitness_scores.append(fitness)
            best_fitness = max(fitness_scores)
            avg_fitness = np.mean(fitness_scores)
            best_fitness_history.append(best_fitness)
            avg_fitness_history.append(avg_fitness)
            new_population: list = []
            for _ in range(population_size):
                tournament_size = 3
                tournament_indices = random.sample(range(population_size), tournament_size)
                winner_idx = max(tournament_indices, key=lambda i: fitness_scores[i])
                new_population.append(population[winner_idx].copy())
            for i in range(0, population_size - 1, 2):
                if random.random() < crossover_rate:
                    (parent1, parent2) = (new_population[i], new_population[i + 1])
                    params = list(parameter_ranges.keys())
                    crossover_point = random.randint(1, len(params) - 1)
                    (child1, child2) = (parent1.copy(), parent2.copy())
                    for j in range(crossover_point, len(params)):
                        param = params[j]
                        (child1[param], child2[param]) = (child2[param], child1[param])
                    (new_population[i], new_population[i + 1]) = (child1, child2)
            for individual in new_population:
                if random.random() < mutation_rate:
                    param = random.choice(list(parameter_ranges.keys()))
                    individual[param] = random.choice(parameter_ranges[param])
            population = new_population
        final_fitness: list = []
        for individual in population:
            param_key = json.dumps(individual, sort_keys=True)
            fitness = param_performance.get(param_key, 0)
            final_fitness.append((individual, fitness))
        final_fitness.sort(key=lambda x: x[1], reverse=True)
        return {
            "method": "genetic_algorithm",
            "objective": objective,
            "generations": max_iterations,
            "population_size": population_size,
            "best_parameters": final_fitness[0][0],
            "best_fitness": final_fitness[0][1],
            "top_10_solutions": final_fitness[:10],
            "convergence_history": {
                "best_fitness": best_fitness_history,
                "avg_fitness": avg_fitness_history,
            },
            "parameter_ranges": parameter_ranges,
        }

    def _grid_search_optimization(
        self, parameter_ranges: Dict[str, List], objective: str
    ) -> Dict[str, Any]:
        """Grid search optimization"""
        results = self.load_backtest_results()
        if not results:
            return {"error": "No backtest results for optimization"}  # type: ignore[dict-item]
        param_performance = {}
        for result in results:
            param_key = json.dumps(result.parameters, sort_keys=True)
            param_performance[param_key] = result.metrics.get(objective, 0)
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())
        all_combinations = list(product(*param_values))
        results_grid: list = []
        for combination in all_combinations:
            params = dict(zip(param_names, combination))
            param_key = json.dumps(params, sort_keys=True)
            performance = param_performance.get(param_key, 0)
            results_grid.append({"parameters": params, "performance": performance})
        results_grid.sort(key=lambda x: x["performance"], reverse=True)
        return {
            "method": "grid_search",
            "objective": objective,
            "total_combinations": len(all_combinations),
            "evaluated_combinations": len([r for r in results_grid if r["performance"] != 0]),
            "best_parameters": results_grid[0]["parameters"],
            "best_performance": results_grid[0]["performance"],
            "top_10_results": results_grid[:10],
            "parameter_ranges": parameter_ranges,
            "full_results": results_grid,
        }

    def _random_search_optimization(
        self, parameter_ranges: Dict[str, List], objective: str, max_iterations: int
    ) -> Dict[str, Any]:
        """Random search optimization"""
        results = self.load_backtest_results()
        if not results:
            return {"error": "No backtest results for optimization"}  # type: ignore[dict-item]
        param_performance = {}
        for result in results:
            param_key = json.dumps(result.parameters, sort_keys=True)
            param_performance[param_key] = result.metrics.get(objective, 0)
        search_results: list = []
        for _ in range(max_iterations):
            params = {}
            for param, values in parameter_ranges.items():
                params[param] = random.choice(values)
            param_key = json.dumps(params, sort_keys=True)
            performance = param_performance.get(param_key, 0)
            search_results.append({"parameters": params, "performance": performance})
        search_results.sort(key=lambda x: x["performance"], reverse=True)
        return {
            "method": "random_search",
            "objective": objective,
            "iterations": max_iterations,
            "best_parameters": search_results[0]["parameters"],
            "best_performance": search_results[0]["performance"],
            "top_10_results": search_results[:10],
            "parameter_ranges": parameter_ranges,
            "performance_distribution": {
                "mean": np.mean([r["performance"] for r in search_results]),
                "std": np.std([r["performance"] for r in search_results]),
                "min": min((r["performance"] for r in search_results)),
                "max": max((r["performance"] for r in search_results)),
            },
        }

    def _calculate_aggregate_metrics(self, results: List[BacktestResult]) -> Dict[str, float]:
        """Calculate aggregate metrics from multiple results"""
        if not results:
            return {}
        metrics = {}
        for metric in ["total_return", "sharpe_ratio", "max_drawdown", "win_rate", "profit_factor"]:
            values = [r.metrics.get(metric, 0) for r in results]
            metrics[f"avg_{metric}"] = np.mean(values)
            metrics[f"std_{metric}"] = np.std(values)
            metrics[f"min_{metric}"] = np.min(values)
            metrics[f"max_{metric}"] = np.max(values)
        return metrics

    def _create_pnl_histogram(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create PnL histogram data"""
        pnls = [t.get("pnl", 0) for t in trades]
        if not pnls:
            return {}
        bins = np.linspace(min(pnls), max(pnls), 20)
        (hist, bin_edges) = np.histogram(pnls, bins=bins)
        return {"bins": bin_edges.tolist(), "counts": hist.tolist(), "total_trades": len(pnls)}

    def _create_duration_histogram(self, durations: List[float]) -> Dict[str, Any]:
        """Create duration histogram data"""
        if not durations:
            return {}
        bins = np.linspace(0, max(durations), 15)
        (hist, bin_edges) = np.histogram(durations, bins=bins)
        return {"bins": bin_edges.tolist(), "counts": hist.tolist(), "total_trades": len(durations)}

    def _suggest_strategy_clusters(
        self, correlation_matrix: np.ndarray, strategy_names: List[str]
    ) -> List[Dict[str, Any]]:
        """Suggest strategy clusters based on correlation"""
        try:
            from sklearn.cluster import AgglomerativeClustering

            distance_matrix = 1 - np.abs(correlation_matrix)
            n_clusters = min(3, len(strategy_names))
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters, metric="precomputed", linkage="average"
            )
            cluster_labels = clustering.fit_predict(distance_matrix)
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(strategy_names[i])
            suggestions: list = []
            for cluster_id, strategies in clusters.items():
                if len(strategies) > 1:
                    indices = [strategy_names.index(s) for s in strategies]
                    cluster_correlations: list = []
                    for i in range(len(indices)):
                        for j in range(i + 1, len(indices)):
                            cluster_correlations.append(correlation_matrix[indices[i], indices[j]])
                    avg_correlation = np.mean(cluster_correlations) if cluster_correlations else 0
                    suggestions.append(
                        {
                            "cluster_id": int(cluster_id),
                            "strategies": strategies,
                            "avg_correlation": float(avg_correlation),
                            "recommendation": (
                                "High correlation - consider diversifying"
                                if avg_correlation > 0.7
                                else "Good diversification"
                            ),
                        }
                    )
            return suggestions
        except ImportError:
            return [{"note": "Install scikit-learn for advanced clustering analysis"}]
        except Exception as e:
            logger.error("Error in clustering: %s", e)
            return []
