import json
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

from flask import Blueprint, jsonify, request
from flask_socketio import emit

backtesting_bp = Blueprint("backtesting", __name__)

# Store active backtest processes
active_backtests = {}


@backtesting_bp.route("/api/backtest/start", methods=["POST"])
def start_backtest():
    """Start a new backtest with the provided configuration."""
    try:
        config = request.json

        # Validate required parameters
        required_fields = [
            "pairs",
            "timeframe",
            "startDate",
            "endDate",
            "initialCapital",
        ]
        for field in required_fields:
            if field not in config:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Generate unique backtest ID
        backtest_id = f"bt_{int(time.time())}"

        # Prepare backtest parameters
        pairs_str = ",".join(config["pairs"])

        # Create backtest command
        project_root = Path(__file__).parent.parent.parent.parent
        backtest_script = project_root / "tests" / "backtesting.py"

        cmd = [
            "python3",
            str(backtest_script),
            "--pairs",
            pairs_str,
            "--timeframe",
            config["timeframe"],
            "--log-level",
            "INFO",
        ]

        # Add optional config file if provided
        if "configPath" in config:
            cmd.extend(["--config", config["configPath"]])

        # Start backtest process in background
        def run_backtest():
            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=str(project_root),
                )

                active_backtests[backtest_id] = {
                    "process": process,
                    "status": "running",
                    "start_time": datetime.now(),
                    "config": config,
                }

                # Monitor process output
                while True:
                    output = process.stdout.readline()
                    if output == "" and process.poll() is not None:
                        break
                    if output:
                        # Emit progress updates via WebSocket
                        emit(
                            "backtest_progress",
                            {
                                "backtest_id": backtest_id,
                                "message": output.strip(),
                                "timestamp": datetime.now().isoformat(),
                            },
                            namespace="/backtest",
                        )

                # Process completed
                return_code = process.poll()
                if return_code == 0:
                    active_backtests[backtest_id]["status"] = "completed"
                    # Try to load results
                    results = load_backtest_results(backtest_id)
                    emit(
                        "backtest_completed",
                        {"backtest_id": backtest_id, "results": results},
                        namespace="/backtest",
                    )
                else:
                    active_backtests[backtest_id]["status"] = "failed"
                    error_output = process.stderr.read()
                    emit(
                        "backtest_failed",
                        {"backtest_id": backtest_id, "error": error_output},
                        namespace="/backtest",
                    )

            except Exception as e:
                active_backtests[backtest_id]["status"] = "failed"
                emit(
                    "backtest_failed",
                    {"backtest_id": backtest_id, "error": str(e)},
                    namespace="/backtest",
                )

        # Start backtest in background thread
        thread = threading.Thread(target=run_backtest)
        thread.daemon = True
        thread.start()

        return jsonify(
            {
                "backtest_id": backtest_id,
                "status": "started",
                "message": "Backtest started successfully",
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@backtesting_bp.route("/api/backtest/stop/<backtest_id>", methods=["POST"])
def stop_backtest(backtest_id):
    """Stop a running backtest."""
    try:
        if backtest_id not in active_backtests:
            return jsonify({"error": "Backtest not found"}), 404

        backtest = active_backtests[backtest_id]
        if backtest["status"] == "running":
            process = backtest["process"]
            process.terminate()
            backtest["status"] = "stopped"

            return jsonify(
                {
                    "backtest_id": backtest_id,
                    "status": "stopped",
                    "message": "Backtest stopped successfully",
                }
            )
        else:
            return jsonify(
                {
                    "backtest_id": backtest_id,
                    "status": backtest["status"],
                    "message": f'Backtest is not running (status: {backtest["status"]})',
                }
            )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@backtesting_bp.route("/api/backtest/status/<backtest_id>", methods=["GET"])
def get_backtest_status(backtest_id):
    """Get the status of a specific backtest."""
    try:
        if backtest_id not in active_backtests:
            return jsonify({"error": "Backtest not found"}), 404

        backtest = active_backtests[backtest_id]
        return jsonify(
            {
                "backtest_id": backtest_id,
                "status": backtest["status"],
                "start_time": backtest["start_time"].isoformat(),
                "config": backtest["config"],
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@backtesting_bp.route("/api/backtest/results/<backtest_id>", methods=["GET"])
def get_backtest_results(backtest_id):
    """Get the results of a completed backtest."""
    try:
        results = load_backtest_results(backtest_id)
        if results:
            return jsonify(results)
        else:
            return (
                jsonify({"error": "Results not found or backtest not completed"}),
                404,
            )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@backtesting_bp.route("/api/backtest/history", methods=["GET"])
def get_backtest_history():
    """Get the history of all backtests."""
    try:
        project_root = Path(__file__).parent.parent.parent.parent
        results_dir = project_root / "data" / "backtest_results"

        if not results_dir.exists():
            return jsonify([])

        history = []
        for result_dir in results_dir.iterdir():
            if result_dir.is_dir():
                # Look for metrics file
                metrics_files = list(result_dir.glob("backtest_metrics_*.json"))
                if metrics_files:
                    try:
                        with open(metrics_files[0], "r") as f:
                            metrics = json.load(f)

                        history.append(
                            {
                                "id": result_dir.name,
                                "timestamp": metrics.get("run_details", {}).get("start_timestamp"),
                                "results": metrics,
                                "duration": calculate_duration(metrics),
                            }
                        )
                    except Exception as e:
                        print(f"Error loading backtest result {result_dir.name}: {e}")
                        continue

        # Sort by timestamp (newest first)
        history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        return jsonify(history)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def load_backtest_results(backtest_id):
    """Load backtest results from the results directory."""
    try:
        project_root = Path(__file__).parent.parent.parent.parent
        results_dir = project_root / "data" / "backtest_results"

        # Find the most recent results directory (since backtest_id might not match exactly)
        result_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
        if not result_dirs:
            return None

        # Get the most recent directory
        latest_dir = max(result_dirs, key=lambda d: d.stat().st_mtime)

        # Look for metrics file
        metrics_files = list(latest_dir.glob("backtest_metrics_*.json"))
        if not metrics_files:
            return None

        with open(metrics_files[0], "r") as f:
            metrics = json.load(f)

        # Also load portfolio and trades data if available
        portfolio_files = list(latest_dir.glob("backtest_portfolio_*.json"))
        trades_files = list(latest_dir.glob("backtest_trades_*.json"))

        result = {"metrics": metrics, "portfolio_data": None, "trades_data": None}

        if portfolio_files:
            try:
                with open(portfolio_files[0], "r") as f:
                    result["portfolio_data"] = json.load(f)
            except Exception as e:
                print(f"Error loading portfolio data: {e}")

        if trades_files:
            try:
                with open(trades_files[0], "r") as f:
                    result["trades_data"] = json.load(f)
            except Exception as e:
                print(f"Error loading trades data: {e}")

        return result

    except Exception as e:
        print(f"Error loading backtest results: {e}")
        return None


def calculate_duration(metrics):
    """Calculate the duration of a backtest from metrics."""
    try:
        start_time = metrics.get("run_details", {}).get("start_timestamp")
        end_time = metrics.get("run_details", {}).get("end_timestamp")

        if start_time and end_time:
            start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
            duration = end_dt - start_dt

            total_seconds = int(duration.total_seconds())
            if total_seconds < 60:
                return f"{total_seconds}s"
            elif total_seconds < 3600:
                return f"{total_seconds // 60}m {total_seconds % 60}s"
            else:
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                return f"{hours}h {minutes}m"

        return "Unknown"

    except Exception:
        return "Unknown"


# WebSocket events for real-time backtest updates
from flask_socketio import Namespace


class BacktestNamespace(Namespace):
    def on_connect(self):
        print("Client connected to backtest namespace")

    def on_disconnect(self):
        print("Client disconnected from backtest namespace")

    def on_subscribe_backtest(self, data):
        backtest_id = data.get("backtest_id")
        if backtest_id:
            # Join room for this specific backtest
            from flask_socketio import join_room

            join_room(f"backtest_{backtest_id}")
