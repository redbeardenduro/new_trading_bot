"""
Advanced Portfolio Analytics Module

Provides comprehensive portfolio analysis including:
- Risk Attribution Analysis
- Performance Attribution
- Sharpe Ratio Optimization
- Value at Risk (VaR) calculations
- Drawdown Analysis
- Monte Carlo Simulations
"""

import logging
import warnings
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import norm

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class PortfolioAnalytics:
    """Advanced portfolio analytics engine"""

    def __init__(
        self, portfolio_data: Dict[str, Any], market_data: Dict[str, pd.DataFrame]
    ) -> None:
        """
        Initialize portfolio analytics

        Args:
            portfolio_data: Portfolio history and positions
            market_data: Market price data for assets
        """
        self.portfolio_data = portfolio_data
        self.market_data = market_data
        self.risk_free_rate = 0.02

    def calculate_returns(self, prices: pd.Series) -> pd.Series:
        """Calculate returns from price series"""
        return prices.pct_change().dropna()

    def calculate_portfolio_returns(self) -> pd.Series:
        """Calculate portfolio returns from historical data"""
        if "portfolio_history" not in self.portfolio_data:
            return pd.Series()
        portfolio_values = pd.Series(self.portfolio_data["portfolio_history"])
        return self.calculate_returns(portfolio_values)

    def risk_attribution_analysis(self) -> Dict[str, Any]:
        """
        Perform factor-based risk decomposition

        Returns:
            Dict containing risk attribution metrics
        """
        try:
            portfolio_returns = self.calculate_portfolio_returns()
            if portfolio_returns.empty:
                return {"error": "No portfolio data available"}  # type: ignore[dict-item]
            portfolio_vol = portfolio_returns.std() * np.sqrt(252)
            factors = {}
            factor_loadings = {}
            for asset, data in self.market_data.items():
                if "close" in data.columns:
                    asset_returns = self.calculate_returns(data["close"])
                    if len(asset_returns) > 0:
                        correlation = portfolio_returns.corr(asset_returns)
                        if not np.isnan(correlation):
                            factors[asset] = {
                                "volatility": asset_returns.std() * np.sqrt(252),
                                "correlation": correlation,
                                "beta": correlation
                                * (portfolio_vol / (asset_returns.std() * np.sqrt(252))),
                            }
                            factor_loadings[asset] = correlation
            total_risk = portfolio_vol**2
            systematic_risk = sum(
                [
                    loading**2 * factors[asset]["volatility"] ** 2
                    for (asset, loading) in factor_loadings.items()
                    if asset in factors
                ]
            )
            idiosyncratic_risk = max(0, total_risk - systematic_risk)
            return {
                "portfolio_volatility": float(portfolio_vol),
                "systematic_risk": float(np.sqrt(systematic_risk)),
                "idiosyncratic_risk": float(np.sqrt(idiosyncratic_risk)),
                "risk_decomposition": {
                    "systematic_pct": float(systematic_risk / total_risk * 100),
                    "idiosyncratic_pct": float(idiosyncratic_risk / total_risk * 100),
                },
                "factor_exposures": factors,
                "factor_loadings": factor_loadings,
            }
        except Exception as e:
            logger.error("Error in risk attribution analysis: %s", e)
            return {"error": str(e)}  # type: ignore[dict-item]

    def performance_attribution(self) -> Dict[str, Any]:
        """
        Calculate alpha vs beta performance attribution

        Returns:
            Dict containing performance attribution metrics
        """
        try:
            portfolio_returns = self.calculate_portfolio_returns()
            if portfolio_returns.empty:
                return {"error": "No portfolio data available"}  # type: ignore[dict-item]
            benchmark_returns = pd.Series()
            benchmark_name = "Market"
            for asset, data in self.market_data.items():
                if "close" in data.columns:
                    benchmark_returns = self.calculate_returns(data["close"])
                    benchmark_name = asset
                    break
            if benchmark_returns.empty:
                return {"error": "No benchmark data available"}  # type: ignore[dict-item]
            aligned_data = pd.concat([portfolio_returns, benchmark_returns], axis=1)  # type: ignore[call-overload].dropna()
            if aligned_data.empty:
                return {"error": "No aligned data available"}  # type: ignore[dict-item]
            port_ret = aligned_data.iloc[:, 0]
            bench_ret = aligned_data.iloc[:, 1]
            portfolio_annual_return = port_ret.mean() * 252
            benchmark_annual_return = bench_ret.mean() * 252
            covariance = np.cov(port_ret, bench_ret)[0, 1]
            benchmark_variance = np.var(bench_ret)
            beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
            alpha = portfolio_annual_return - (
                self.risk_free_rate + beta * (benchmark_annual_return - self.risk_free_rate)
            )
            tracking_error = (port_ret - bench_ret).std() * np.sqrt(252)
            information_ratio = alpha / tracking_error if tracking_error != 0 else 0
            return {
                "alpha": float(alpha),
                "beta": float(beta),
                "portfolio_return": float(portfolio_annual_return),
                "benchmark_return": float(benchmark_annual_return),
                "benchmark_name": benchmark_name,
                "tracking_error": float(tracking_error),
                "information_ratio": float(information_ratio),
                "excess_return": float(portfolio_annual_return - benchmark_annual_return),
                "risk_adjusted_return": float(alpha),
            }
        except Exception as e:
            logger.error("Error in performance attribution: %s", e)
            return {"error": str(e)}  # type: ignore[dict-item]

    def sharpe_ratio_optimization(self, target_return: Optional[float] = None) -> Dict[str, Any]:
        """
        Perform Sharpe ratio optimization

        Args:
            target_return: Target return for optimization (optional)  # type: ignore[unreachable]

        Returns:
            Dict containing optimization results
        """
        try:
            asset_returns = {}
            for asset, data in self.market_data.items():
                if "close" in data.columns:
                    returns = self.calculate_returns(data["close"])
                    if len(returns) > 20:
                        asset_returns[asset] = returns
            if len(asset_returns) < 2:
                return {"error": "Insufficient asset data for optimization"}  # type: ignore[dict-item]
            returns_df = pd.DataFrame(asset_returns).dropna()
            if returns_df.empty:
                return {"error": "No aligned returns data"}  # type: ignore[dict-item]
            assets = returns_df.columns.tolist()
            returns_matrix = returns_df.values
            expected_returns = returns_matrix.mean(axis=0) * 252
            cov_matrix = np.cov(returns_matrix.T) * 252

            def negative_sharpe_ratio(weights):
                portfolio_return = np.sum(weights * expected_returns)
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                return -(portfolio_return - self.risk_free_rate) / portfolio_vol

            constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}  # type: ignore[dict-item]
            bounds = tuple(((0, 1) for _ in range(len(assets))))
            initial_guess = np.array([1 / len(assets)] * len(assets))
            result = optimize.minimize(
                negative_sharpe_ratio,
                initial_guess,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )
            if result.success:
                optimal_weights = result.x
                optimal_return = np.sum(optimal_weights * expected_returns)
                optimal_vol = np.sqrt(
                    np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights))
                )
                optimal_sharpe = (optimal_return - self.risk_free_rate) / optimal_vol
                portfolio_returns = self.calculate_portfolio_returns()
                current_sharpe = 0
                if not portfolio_returns.empty:
                    current_return = portfolio_returns.mean() * 252
                    current_vol = portfolio_returns.std() * np.sqrt(252)
                    current_sharpe = (
                        (current_return - self.risk_free_rate) / current_vol
                        if current_vol != 0
                        else 0
                    )
                return {
                    "optimization_success": True,
                    "optimal_weights": dict(zip(assets, optimal_weights.tolist())),
                    "optimal_return": float(optimal_return),
                    "optimal_volatility": float(optimal_vol),
                    "optimal_sharpe_ratio": float(optimal_sharpe),
                    "current_sharpe_ratio": float(current_sharpe),
                    "improvement": float(optimal_sharpe - current_sharpe),
                    "expected_returns": dict(zip(assets, expected_returns.tolist())),
                    "risk_free_rate": self.risk_free_rate,
                }
            else:
                return {"error": "Optimization failed", "message": result.message}  # type: ignore[dict-item]
        except Exception as e:
            logger.error("Error in Sharpe ratio optimization: %s", e)
            return {"error": str(e)}  # type: ignore[dict-item]

    def calculate_var(
        self, confidence_level: float = 0.05, method: str = "historical"
    ) -> Dict[str, Any]:
        """
        Calculate Value at Risk (VaR)

        Args:
            confidence_level: Confidence level (default 5% for 95% VaR)
            method: 'historical', 'parametric', or 'monte_carlo'

        Returns:
            Dict containing VaR calculations
        """
        try:
            portfolio_returns = self.calculate_portfolio_returns()
            if portfolio_returns.empty:
                return {"error": "No portfolio data available"}  # type: ignore[dict-item]
            results = {}
            if method in ["historical", "all"]:
                historical_var = np.percentile(portfolio_returns, confidence_level * 100)
                results["historical_var"] = {
                    "daily": float(historical_var),
                    "monthly": float(historical_var * np.sqrt(21)),
                    "annual": float(historical_var * np.sqrt(252)),
                }
            if method in ["parametric", "all"]:
                mean_return = portfolio_returns.mean()
                std_return = portfolio_returns.std()
                z_score = norm.ppf(confidence_level)
                parametric_var = mean_return + z_score * std_return
                results["parametric_var"] = {
                    "daily": float(parametric_var),
                    "monthly": float(parametric_var * np.sqrt(21)),
                    "annual": float(parametric_var * np.sqrt(252)),
                }
            if method in ["monte_carlo", "all"]:
                mc_var = self._monte_carlo_var(portfolio_returns, confidence_level)
                results["monte_carlo_var"] = mc_var
            historical_var_daily = np.percentile(portfolio_returns, confidence_level * 100)
            tail_returns = portfolio_returns[portfolio_returns <= historical_var_daily]
            expected_shortfall = (
                tail_returns.mean() if len(tail_returns) > 0 else historical_var_daily
            )
            results["expected_shortfall"] = {
                "daily": float(expected_shortfall),
                "monthly": float(expected_shortfall * np.sqrt(21)),
                "annual": float(expected_shortfall * np.sqrt(252)),
            }
            results["confidence_level"] = confidence_level
            results["method"] = method
            return results
        except Exception as e:
            logger.error("Error in VaR calculation: %s", e)
            return {"error": str(e)}  # type: ignore[dict-item]

    def _monte_carlo_var(
        self, returns: pd.Series, confidence_level: float, simulations: int = 10000
    ) -> Dict[str, float]:
        """Monte Carlo VaR simulation"""
        try:
            mean_return = returns.mean()
            std_return = returns.std()
            random_returns = np.random.normal(mean_return, std_return, simulations)
            mc_var = np.percentile(random_returns, confidence_level * 100)
            return {
                "daily": float(mc_var),
                "monthly": float(mc_var * np.sqrt(21)),
                "annual": float(mc_var * np.sqrt(252)),
                "simulations": simulations,
            }
        except Exception as e:
            logger.error("Error in Monte Carlo VaR: %s", e)
            return {"daily": 0.0, "monthly": 0.0, "annual": 0.0, "simulations": 0}

    def drawdown_analysis(self) -> Dict[str, Any]:
        """
        Perform comprehensive drawdown analysis

        Returns:
            Dict containing drawdown metrics and analysis
        """
        try:
            if "portfolio_history" not in self.portfolio_data:
                return {"error": "No portfolio history available"}  # type: ignore[dict-item]
            portfolio_values = pd.Series(self.portfolio_data["portfolio_history"])
            if portfolio_values.empty:
                return {"error": "Empty portfolio history"}  # type: ignore[dict-item]
            running_max = portfolio_values.expanding().max()
            drawdown = (portfolio_values - running_max) / running_max
            max_drawdown = drawdown.min()
            max_dd_date = drawdown.idxmin()
            drawdown_periods: list = []
            in_drawdown = False
            start_date = None
            for date, dd in drawdown.items():
                if dd < 0 and (not in_drawdown):
                    in_drawdown = True
                    start_date = date
                elif dd >= 0 and in_drawdown:
                    in_drawdown = False
                    if start_date is not None:
                        duration = date - start_date
                        drawdown_periods.append(
                            {
                                "start": start_date,
                                "end": date,
                                "duration_days": (
                                    duration.days if hasattr(duration, "days") else duration
                                ),
                                "max_drawdown": drawdown[start_date:date].min(),
                            }
                        )
            current_drawdown = drawdown.iloc[-1]
            recovery_times: list = []
            for period in drawdown_periods:
                if "duration_days" in period:
                    recovery_times.append(period["duration_days"])
            avg_recovery_time = np.mean(recovery_times) if recovery_times else 0
            portfolio_returns = self.calculate_portfolio_returns()
            annual_return = portfolio_returns.mean() * 252 if not portfolio_returns.empty else 0
            calmar_ratio = abs(annual_return / max_drawdown) if max_drawdown != 0 else 0
            return {
                "max_drawdown": float(max_drawdown),
                "max_drawdown_date": str(max_dd_date),
                "current_drawdown": float(current_drawdown),
                "drawdown_periods": len(drawdown_periods),
                "avg_recovery_time_days": float(avg_recovery_time),
                "longest_drawdown_days": (
                    max([p.get("duration_days", 0) for p in drawdown_periods])
                    if drawdown_periods
                    else 0
                ),
                "calmar_ratio": float(calmar_ratio),
                "drawdown_series": drawdown.to_dict(),
                "underwater_curve": (drawdown * 100).to_dict(),
                "recovery_periods": drawdown_periods,
            }
        except Exception as e:
            logger.error("Error in drawdown analysis: %s", e)
            return {"error": str(e)}  # type: ignore[dict-item]

    def monte_carlo_simulation(self, days: int = 252, simulations: int = 1000) -> Dict[str, Any]:
        """
        Perform Monte Carlo portfolio simulation

        Args:
            days: Number of days to simulate
            simulations: Number of simulation runs

        Returns:
            Dict containing simulation results
        """
        try:
            portfolio_returns = self.calculate_portfolio_returns()
            if portfolio_returns.empty:
                return {"error": "No portfolio data available"}  # type: ignore[dict-item]
            mean_return = portfolio_returns.mean()
            std_return = portfolio_returns.std()
            current_value = self.portfolio_data.get("total_value", 100000)
            simulation_results: list = []
            final_values = []
            for sim in range(simulations):
                random_returns = np.random.normal(mean_return, std_return, days)
                portfolio_path = [current_value]
                for daily_return in random_returns:
                    new_value = portfolio_path[-1] * (1 + daily_return)
                    portfolio_path.append(new_value)
                simulation_results.append(portfolio_path)
                final_values.append(portfolio_path[-1])
            final_values = np.array(final_values)  # type: ignore[assignment]
            percentiles = [5, 10, 25, 50, 75, 90, 95]
            percentile_results = {}
            for p in percentiles:
                percentile_results[f"p{p}"] = float(np.percentile(final_values, p))
            prob_loss = (final_values < current_value).mean()
            expected_value = final_values.mean()  # type: ignore[Any]
            worst_5_pct = np.percentile(final_values, 5)
            best_5_pct = np.percentile(final_values, 95)
            sample_size = min(100, simulations)
            sample_indices = np.random.choice(simulations, sample_size, replace=False)
            sample_paths = [simulation_results[i] for i in sample_indices]
            return {
                "simulations": simulations,
                "days": days,
                "current_value": float(current_value),
                "expected_final_value": float(expected_value),
                "percentiles": percentile_results,
                "probability_of_loss": float(prob_loss),
                "worst_case_5pct": float(worst_5_pct),
                "best_case_5pct": float(best_5_pct),
                "mean_return": float(mean_return),
                "volatility": float(std_return),
                "sample_paths": sample_paths,
                "statistics": {
                    "mean": float(final_values.mean()),  # type: ignore[Any]
                    "std": float(final_values.std()),  # type: ignore[Any]
                    "min": float(final_values.min()),  # type: ignore[Any]
                    "max": float(final_values.max()),  # type: ignore[Any]
                    "skewness": float(self._calculate_skewness(final_values)),
                    "kurtosis": float(self._calculate_kurtosis(final_values)),
                },
            }
        except Exception as e:
            logger.error("Error in Monte Carlo simulation: %s", e)
            return {"error": str(e)}  # type: ignore[dict-item]

    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data"""
        try:
            mean = np.mean(data)
            std = np.std(data)
            return np.mean(((data - mean) / std) ** 3) if std != 0 else 0
        except:
            return 0

    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data"""
        try:
            mean = np.mean(data)
            std = np.std(data)
            return np.mean(((data - mean) / std) ** 4) - 3 if std != 0 else 0
        except:
            return 0

    def comprehensive_analysis(self) -> Dict[str, Any]:
        """
        Perform comprehensive portfolio analysis

        Returns:
            Dict containing all analytics results
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "risk_attribution": self.risk_attribution_analysis(),
            "performance_attribution": self.performance_attribution(),
            "sharpe_optimization": self.sharpe_ratio_optimization(),
            "var_analysis": self.calculate_var(method="all"),
            "drawdown_analysis": self.drawdown_analysis(),
            "monte_carlo_simulation": self.monte_carlo_simulation(),
        }
