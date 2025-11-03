"""
Enhanced Backtesting Analytics API Routes

Provides REST endpoints for advanced backtesting analysis including:
- Walk-forward analysis
- Strategy comparison matrix
- Trade analysis dashboard
- Performance heatmaps
- Correlation analysis
- Optimization engine
"""

import os
import sys
from datetime import datetime
from typing import Optional

from flask import Blueprint, jsonify, request

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

try:
    from production_trading_bot.core.enhanced_backtesting import \
        EnhancedBacktesting
except ImportError:
    # Fallback if import fails
    EnhancedBacktesting = None

enhanced_backtesting_bp = Blueprint("enhanced_backtesting", __name__)


def get_backtesting_engine() -> Optional[EnhancedBacktesting]:
    """Get enhanced backtesting engine instance"""
    if not EnhancedBacktesting:
        return None

    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "backtest_results")
    return EnhancedBacktesting(data_dir)


@enhanced_backtesting_bp.route("/api/backtesting/enhanced/walk-forward", methods=["GET"])
def walk_forward_analysis():
    """Perform walk-forward analysis"""
    try:
        engine = get_backtesting_engine()
        if not engine:
            return jsonify({"error": "Enhanced backtesting module not available"}), 500

        # Get parameters from request
        train_period_days = request.args.get("train_period_days", 180, type=int)
        test_period_days = request.args.get("test_period_days", 60, type=int)
        step_days = request.args.get("step_days", 30, type=int)
        start_date = request.args.get("start_date")
        end_date = request.args.get("end_date")

        result = engine.walk_forward_analysis(
            train_period_days=train_period_days,
            test_period_days=test_period_days,
            step_days=step_days,
            start_date=start_date,
            end_date=end_date,
        )

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@enhanced_backtesting_bp.route("/api/backtesting/enhanced/strategy-comparison", methods=["GET"])
def strategy_comparison():
    """Get strategy comparison matrix"""
    try:
        engine = get_backtesting_engine()
        if not engine:
            return jsonify({"error": "Enhanced backtesting module not available"}), 500

        # Get metrics to compare
        metrics_param = request.args.get("metrics")
        metrics = metrics_param.split(",") if metrics_param else None

        result = engine.strategy_comparison_matrix(metrics)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@enhanced_backtesting_bp.route("/api/backtesting/enhanced/trade-analysis", methods=["GET"])
def trade_analysis():
    """Get comprehensive trade analysis"""
    try:
        engine = get_backtesting_engine()
        if not engine:
            return jsonify({"error": "Enhanced backtesting module not available"}), 500

        run_id = request.args.get("run_id")

        result = engine.trade_analysis_dashboard(run_id)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@enhanced_backtesting_bp.route("/api/backtesting/enhanced/performance-heatmaps", methods=["GET"])
def performance_heatmaps():
    """Get performance heatmaps"""
    try:
        engine = get_backtesting_engine()
        if not engine:
            return jsonify({"error": "Enhanced backtesting module not available"}), 500

        result = engine.performance_heatmaps()

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@enhanced_backtesting_bp.route("/api/backtesting/enhanced/correlation-analysis", methods=["GET"])
def correlation_analysis():
    """Get correlation analysis between strategies"""
    try:
        engine = get_backtesting_engine()
        if not engine:
            return jsonify({"error": "Enhanced backtesting module not available"}), 500

        result = engine.correlation_analysis()

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@enhanced_backtesting_bp.route("/api/backtesting/enhanced/optimization", methods=["POST"])
def parameter_optimization():
    """Run parameter optimization"""
    try:
        engine = get_backtesting_engine()
        if not engine:
            return jsonify({"error": "Enhanced backtesting module not available"}), 500

        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        parameter_ranges = data.get("parameter_ranges", {})
        objective = data.get("objective", "sharpe_ratio")
        method = data.get("method", "genetic")
        max_iterations = data.get("max_iterations", 50)

        if not parameter_ranges:
            return jsonify({"error": "Parameter ranges required"}), 400

        result = engine.optimization_engine(
            parameter_ranges=parameter_ranges,
            objective=objective,
            method=method,
            max_iterations=max_iterations,
        )

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@enhanced_backtesting_bp.route("/api/backtesting/enhanced/results", methods=["GET"])
def get_backtest_results():
    """Get available backtest results"""
    try:
        engine = get_backtesting_engine()
        if not engine:
            return jsonify({"error": "Enhanced backtesting module not available"}), 500

        run_ids = request.args.get("run_ids")
        if run_ids:
            run_ids = run_ids.split(",")

        results = engine.load_backtest_results(run_ids)

        # Convert results to serializable format
        serialized_results = []
        for result in results:
            serialized_results.append(
                {
                    "run_id": result.run_id,
                    "parameters": result.parameters,
                    "metrics": result.metrics,
                    "start_date": result.start_date,
                    "end_date": result.end_date,
                    "duration_days": result.duration_days,
                    "trade_count": len(result.trades),
                    "portfolio_history_length": len(result.portfolio_history),
                }
            )

        return jsonify({"results": serialized_results, "total_results": len(serialized_results)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@enhanced_backtesting_bp.route("/api/backtesting/enhanced/results/<run_id>", methods=["GET"])
def get_single_result(run_id: str):
    """Get detailed single backtest result"""
    try:
        engine = get_backtesting_engine()
        if not engine:
            return jsonify({"error": "Enhanced backtesting module not available"}), 500

        result = engine._load_single_result(run_id)
        if not result:
            return jsonify({"error": f"Result {run_id} not found"}), 404

        # Include trade details if requested
        include_trades = request.args.get("include_trades", "false").lower() == "true"
        include_portfolio = request.args.get("include_portfolio", "false").lower() == "true"

        response_data = {
            "run_id": result.run_id,
            "parameters": result.parameters,
            "metrics": result.metrics,
            "start_date": result.start_date,
            "end_date": result.end_date,
            "duration_days": result.duration_days,
            "trade_count": len(result.trades),
        }

        if include_trades:
            response_data["trades"] = result.trades

        if include_portfolio:
            response_data["portfolio_history"] = result.portfolio_history

        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@enhanced_backtesting_bp.route("/api/backtesting/enhanced/summary", methods=["GET"])
def get_summary():
    """Get summary of available backtest data"""
    try:
        engine = get_backtesting_engine()
        if not engine:
            return jsonify({"error": "Enhanced backtesting module not available"}), 500

        results = engine.load_backtest_results()

        if not results:
            return jsonify(
                {
                    "total_results": 0,
                    "date_range": None,
                    "available_metrics": [],
                    "parameter_summary": {},
                }
            )

        # Calculate summary statistics
        total_results = len(results)

        # Date range
        start_dates = [r.start_date for r in results if r.start_date]
        end_dates = [r.end_date for r in results if r.end_date]

        date_range = {
            "earliest_start": min(start_dates) if start_dates else None,
            "latest_end": max(end_dates) if end_dates else None,
        }

        # Available metrics
        all_metrics = set()
        for result in results:
            all_metrics.update(result.metrics.keys())

        # Parameter summary
        parameter_summary = {}
        for result in results:
            for param, value in result.parameters.items():
                if param not in parameter_summary:
                    parameter_summary[param] = set()
                parameter_summary[param].add(str(value))

        # Convert sets to lists for JSON serialization
        for param in parameter_summary:
            parameter_summary[param] = list(parameter_summary[param])

        # Performance summary
        performance_summary = {}
        for metric in ["total_return", "sharpe_ratio", "max_drawdown", "win_rate"]:
            values = [r.metrics.get(metric, 0) for r in results]
            if values:
                performance_summary[metric] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len([v for v in values if v != 0]),
                }

        return jsonify(
            {
                "total_results": total_results,
                "date_range": date_range,
                "available_metrics": list(all_metrics),
                "parameter_summary": parameter_summary,
                "performance_summary": performance_summary,
                "data_directory": engine.data_dir,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@enhanced_backtesting_bp.route("/api/backtesting/enhanced/status", methods=["GET"])
def get_status():
    """Get enhanced backtesting module status"""
    try:
        engine = get_backtesting_engine()

        if not engine:
            return jsonify(
                {
                    "status": "unavailable",
                    "error": "Enhanced backtesting module not available",
                }
            )

        # Check data directory
        data_dir_exists = os.path.exists(engine.data_dir)

        # Count available results
        result_count = 0
        if data_dir_exists:
            try:
                subdirs = [
                    d
                    for d in os.listdir(engine.data_dir)
                    if os.path.isdir(os.path.join(engine.data_dir, d))
                ]
                result_count = len(subdirs)
            except:
                pass

        return jsonify(
            {
                "status": "available",
                "data_directory": engine.data_dir,
                "data_directory_exists": data_dir_exists,
                "available_results": result_count,
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500
