import logging
import os
import sys
import threading
import time
from datetime import datetime
from queue import Empty, Queue

from flask import Blueprint
from flask_socketio import SocketIO, emit

# Add the parent directory to the path to import trading bot modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

console_bp = Blueprint("console", __name__)

# Global variables for console management
console_logs = Queue()
socketio = None
log_handler = None


class ConsoleLogHandler(logging.Handler):
    """Custom log handler that captures bot logs and sends them to the console"""

    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        try:
            # Format the log message
            log_entry = {
                "timestamp": datetime.fromtimestamp(record.created).strftime(
                    "%Y-%m-%d %H:%M:%S.%f"
                )[:-3],
                "level": record.levelname,
                "logger": record.name,
                "message": self.format(record),
                "module": getattr(record, "module", ""),
                "funcName": getattr(record, "funcName", ""),
                "lineno": getattr(record, "lineno", ""),
            }

            # Add to queue for WebSocket transmission
            self.log_queue.put(log_entry)

            # Emit to connected clients if socketio is available
            if socketio:
                socketio.emit("console_log", log_entry, namespace="/console")

        except Exception as e:
            print(f"Error in ConsoleLogHandler: {e}")


def setup_console_logging():
    """Set up console logging to capture bot messages"""
    global log_handler

    if log_handler:
        return  # Already set up

    # Create custom log handler
    log_handler = ConsoleLogHandler(console_logs)
    log_handler.setLevel(logging.DEBUG)

    # Create formatter for log messages
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s in %(name)s: %(message)s")
    log_handler.setFormatter(formatter)

    # Add handler to root logger to capture all bot logs
    root_logger = logging.getLogger()
    root_logger.addHandler(log_handler)
    root_logger.setLevel(logging.DEBUG)

    # Also add to specific bot loggers if they exist
    bot_loggers = [
        "core.bot",
        "core.portfolio_manager",
        "integrations.exchange.kraken",
        "integrations.data.reddit",
        "integrations.data.news",
        "integrations.ai.openai",
    ]

    for logger_name in bot_loggers:
        logger = logging.getLogger(logger_name)
        logger.addHandler(log_handler)
        logger.setLevel(logging.DEBUG)


def init_socketio(app):
    """Initialize SocketIO with the Flask app"""
    global socketio
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

    @socketio.on("connect", namespace="/console")
    def handle_connect():
        print("Console client connected")
        emit("console_status", {"status": "connected"})

        # Send recent logs to newly connected client
        recent_logs = []
        temp_queue = Queue()

        # Get up to 50 recent log entries
        while not console_logs.empty() and len(recent_logs) < 50:
            try:
                log_entry = console_logs.get_nowait()
                recent_logs.append(log_entry)
                temp_queue.put(log_entry)
            except Empty:
                break

        # Put logs back in queue
        while not temp_queue.empty():
            console_logs.put(temp_queue.get())

        # Send recent logs to client
        for log_entry in recent_logs[-20:]:  # Send last 20 logs
            emit("console_log", log_entry)

    @socketio.on("disconnect", namespace="/console")
    def handle_disconnect():
        print("Console client disconnected")

    @socketio.on("clear_console", namespace="/console")
    def handle_clear_console():
        """Clear the console logs"""
        global console_logs
        console_logs = Queue()
        emit("console_cleared", {}, broadcast=True)

    @socketio.on("request_logs", namespace="/console")
    def handle_request_logs(data):
        """Send requested number of recent logs"""
        count = data.get("count", 20)
        recent_logs = []
        temp_queue = Queue()

        # Get recent log entries
        while not console_logs.empty() and len(recent_logs) < count:
            try:
                log_entry = console_logs.get_nowait()
                recent_logs.append(log_entry)
                temp_queue.put(log_entry)
            except Empty:
                break

        # Put logs back in queue
        while not temp_queue.empty():
            console_logs.put(temp_queue.get())

        # Send logs to requesting client
        for log_entry in recent_logs[-count:]:
            emit("console_log", log_entry)

    return socketio


def simulate_bot_logs():
    """Simulate bot logging messages for demonstration"""

    def log_simulator():
        logger = logging.getLogger("trading_bot_simulator")

        sample_messages = [
            ("INFO", "Trading bot initialized successfully"),
            ("INFO", "Loading configuration from user_config.json"),
            ("INFO", "Connecting to Kraken API..."),
            ("INFO", "Kraken API connection established"),
            ("INFO", "Initializing sentiment analysis sources..."),
            ("INFO", "Reddit API connected successfully"),
            ("INFO", "News API connected successfully"),
            ("INFO", "OpenAI API connected successfully"),
            ("INFO", "Starting trading cycle..."),
            ("INFO", "Fetching market data for BTC/USD"),
            ("DEBUG", "RSI: 45.2, MACD: 0.15, BB Position: 0.7"),
            ("INFO", "Analyzing sentiment for BTC..."),
            ("DEBUG", "Reddit sentiment: 0.75 (bullish)"),
            ("DEBUG", "News sentiment: 0.15 (neutral)"),
            ("INFO", "AI analysis: Market showing bullish momentum"),
            ("INFO", "Opportunity score: 0.82 (HIGH confidence)"),
            ("INFO", "Generating BUY signal for BTC/USD"),
            ("INFO", "Executing paper trade: BUY 0.05 BTC at $47,800"),
            ("INFO", "Trade executed successfully - Order ID: T001"),
            ("INFO", "Portfolio rebalancing check..."),
            ("DEBUG", "Current BTC allocation: 35.2%, Target: 35.0%"),
            ("INFO", "No rebalancing required"),
            ("INFO", "Cycle completed. Next cycle in 60 seconds..."),
            ("WARNING", "Reddit API rate limit approaching"),
            ("INFO", "Fetching market data for ETH/USD"),
            ("DEBUG", "RSI: 52.1, MACD: -0.05, BB Position: 0.5"),
            ("INFO", "Generating HOLD signal for ETH/USD"),
            ("INFO", "Risk assessment: VaR 95%: $1,240"),
            ("INFO", "Portfolio performance: +2.1% today"),
        ]

        message_index = 0
        while True:
            if message_index >= len(sample_messages):
                message_index = 0

            level, message = sample_messages[message_index]

            if level == "INFO":
                logger.info(message)
            elif level == "DEBUG":
                logger.debug(message)
            elif level == "WARNING":
                logger.warning(message)
            elif level == "ERROR":
                logger.error(message)

            message_index += 1
            time.sleep(2)  # Send a log message every 2 seconds

    # Start the simulator in a separate thread
    simulator_thread = threading.Thread(target=log_simulator, daemon=True)
    simulator_thread.start()


# Initialize console logging when module is imported
setup_console_logging()
