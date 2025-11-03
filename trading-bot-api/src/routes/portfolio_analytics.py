"""
Portfolio Analytics API Routes

Provides REST endpoints for advanced portfolio analytics including:
- Risk attribution analysis
- Performance attribution
- Sharpe ratio optimization
- Value at Risk calculations
- Drawdown analysis
- Monte Carlo simulations
"""

import json
import os
import sys
from datetime import datetime
from typing import Any, Dict

import numpy as np
import pandas as pd
from flask import Blueprint, jsonify, request

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

try:
    from production_trading_bot.core.portfolio_analytics import \
        PortfolioAnalytics
except ImportError:
    # Fallback if import fails
    PortfolioAnalytics = None

portfolio_analytics_bp = Blueprint("portfolio_analytics", __name__)


def load_portfolio_data() -> Dict[str, Any]:
    """Load portfolio data from various sources"""
    try:
        # Try to load from backtest results first
        backtest_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "data", "backtest_results"
        )

        if os.path.exists(backtest_dir):
            # Get the most recent backtest
            subdirs = [
                d for d in os.listdir(backtest_dir) if os.path.isdir(os.path.join(backtest_dir, d))
            ]
            if subdirs:
                latest_dir = max(subdirs)
                portfolio_file = os.path.join(
                    backtest_dir, latest_dir, f"backtest_portfolio_{latest_dir}.json"
                )

                if os.path.exists(portfolio_file):
                    with open(portfolio_file, "r") as f:
                        data = json.load(f)
                        return {
                            "portfolio_history": data.get("portfolio_values", []),
                            "total_value": data.get("final_value", 100000),
                            "positions": data.get("positions", {}),
                            "source": "backtest",
                        }

        # Fallback to paper trading data
        paper_trades_file = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "data",
            "trades",
            "paper",
            "paper_trades.json",
        )
        if os.path.exists(paper_trades_file):
            with open(paper_trades_file, "r") as f:
                data = json.load(f)
                return {
                    "portfolio_history": [100000],  # Default starting value
                    "total_value": data.get("total_balance", 100000),
                    "positions": data.get("positions", {}),
                    "source": "paper_trading",
                }

        # Default portfolio data
        return {
            "portfolio_history": [100000, 102000, 98000, 105000, 103000],
            "total_value": 103000,
            "positions": {"BTC": 0.5, "ETH": 0.3, "USD": 0.2},
            "source": "default",
        }

    except Exception as e:
        print(f"Error loading portfolio data: {e}")
        return {
            "portfolio_history": [100000],
            "total_value": 100000,
            "positions": {},
            "source": "error_fallback",
        }


def load_market_data() -> Dict[str, pd.DataFrame]:
    """Load market data for analysis"""
    try:
        market_data = {}
        cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "cache")

        if os.path.exists(cache_dir):
            for filename in os.listdir(cache_dir):
                if filename.endswith("_cache.json"):
                    asset = filename.split("_")[0]
                    filepath = os.path.join(cache_dir, filename)

                    try:
                        with open(filepath, "r") as f:
                            data = json.load(f)
                            if isinstance(data, list) and len(data) > 0:
                                df = pd.DataFrame(data)
                                if "timestamp" in df.columns:
                                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                                    df.set_index("timestamp", inplace=True)
                                market_data[asset] = df
                    except Exception as e:
                        print(f"Error loading {filename}: {e}")
                        continue

        # If no real data, create sample data
        if not market_data:
            dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq="D")
            for asset in ["BTC", "ETH", "LTC"]:
                # Generate sample price data
                np.random.seed(42)  # For reproducibility
                prices = 50000 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, len(dates))))
                market_data[asset] = pd.DataFrame(
                    {
                        "close": prices,
                        "open": prices * (1 + np.random.normal(0, 0.001, len(dates))),
                        "high": prices * (1 + np.abs(np.random.normal(0, 0.005, len(dates)))),
                        "low": prices * (1 - np.abs(np.random.normal(0, 0.005, len(dates)))),
                        "volume": np.random.uniform(1000, 10000, len(dates)),
                    },
                    index=dates,
                )

        return market_data

    except Exception as e:
        print(f"Error loading market data: {e}")
        return {}


@portfolio_analytics_bp.route("/api/portfolio/analytics/risk-attribution", methods=["GET"])
def risk_attribution():
    """Get risk attribution analysis"""
    try:
        if not PortfolioAnalytics:
            return jsonify({"error": "Portfolio analytics module not available"}), 500

        portfolio_data = load_portfolio_data()
        market_data = load_market_data()

        analytics = PortfolioAnalytics(portfolio_data, market_data)
        result = analytics.risk_attribution_analysis()

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@portfolio_analytics_bp.route("/api/portfolio/analytics/performance-attribution", methods=["GET"])
def performance_attribution():
    """Get performance attribution analysis"""
    try:
        if not PortfolioAnalytics:
            return jsonify({"error": "Portfolio analytics module not available"}), 500

        portfolio_data = load_portfolio_data()
        market_data = load_market_data()

        analytics = PortfolioAnalytics(portfolio_data, market_data)
        result = analytics.performance_attribution()

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@portfolio_analytics_bp.route("/api/portfolio/analytics/sharpe-optimization", methods=["GET"])
def sharpe_optimization():
    """Get Sharpe ratio optimization results"""
    try:
        if not PortfolioAnalytics:
            return jsonify({"error": "Portfolio analytics module not available"}), 500

        target_return = request.args.get("target_return", type=float)

        portfolio_data = load_portfolio_data()
        market_data = load_market_data()

        analytics = PortfolioAnalytics(portfolio_data, market_data)
        result = analytics.sharpe_ratio_optimization(target_return)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@portfolio_analytics_bp.route("/api/portfolio/analytics/var", methods=["GET"])
def value_at_risk():
    """Get Value at Risk analysis"""
    try:
        if not PortfolioAnalytics:
            return jsonify({"error": "Portfolio analytics module not available"}), 500

        confidence_level = request.args.get("confidence_level", 0.05, type=float)
        method = request.args.get("method", "historical")

        portfolio_data = load_portfolio_data()
        market_data = load_market_data()

        analytics = PortfolioAnalytics(portfolio_data, market_data)
        result = analytics.calculate_var(confidence_level, method)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@portfolio_analytics_bp.route("/api/portfolio/analytics/drawdown", methods=["GET"])
def drawdown_analysis():
    """Get drawdown analysis"""
    try:
        if not PortfolioAnalytics:
            return jsonify({"error": "Portfolio analytics module not available"}), 500

        portfolio_data = load_portfolio_data()
        market_data = load_market_data()

        analytics = PortfolioAnalytics(portfolio_data, market_data)
        result = analytics.drawdown_analysis()

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@portfolio_analytics_bp.route("/api/portfolio/analytics/monte-carlo", methods=["GET"])
def monte_carlo_simulation():
    """Get Monte Carlo simulation results"""
    try:
        if not PortfolioAnalytics:
            return jsonify({"error": "Portfolio analytics module not available"}), 500

        days = request.args.get("days", 252, type=int)
        simulations = request.args.get("simulations", 1000, type=int)

        portfolio_data = load_portfolio_data()
        market_data = load_market_data()

        analytics = PortfolioAnalytics(portfolio_data, market_data)
        result = analytics.monte_carlo_simulation(days, simulations)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@portfolio_analytics_bp.route("/api/portfolio/analytics/comprehensive", methods=["GET"])
def comprehensive_analysis():
    """Get comprehensive portfolio analysis"""
    try:
        if not PortfolioAnalytics:
            return jsonify({"error": "Portfolio analytics module not available"}), 500

        portfolio_data = load_portfolio_data()
        market_data = load_market_data()

        analytics = PortfolioAnalytics(portfolio_data, market_data)
        result = analytics.comprehensive_analysis()

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@portfolio_analytics_bp.route("/api/portfolio/analytics/status", methods=["GET"])
def analytics_status():
    """Get analytics module status and available data"""
    try:
        portfolio_data = load_portfolio_data()
        market_data = load_market_data()

        return jsonify(
            {
                "status": "available" if PortfolioAnalytics else "unavailable",
                "portfolio_data_source": portfolio_data.get("source", "unknown"),
                "portfolio_value": portfolio_data.get("total_value", 0),
                "market_data_assets": list(market_data.keys()),
                "market_data_count": len(market_data),
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500
