import os
import sys
from datetime import datetime, timedelta

from flask import Blueprint, jsonify, request
from flask_cors import cross_origin
from flask_jwt_extended import jwt_required

# Add the parent directory to the path to import trading bot modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

try:
    from production_trading_bot.core.bot import MultiCryptoTradingBot
    from production_trading_bot.core.config import BotConfig
    from production_trading_bot.core.portfolio_manager import PortfolioManager
    from production_trading_bot.integrations.ai.openai import OpenAIAnalyzer
    from production_trading_bot.integrations.data.news import \
        NewsSentimentSource
    from production_trading_bot.integrations.data.reddit import \
        RedditSentimentSource
    from production_trading_bot.integrations.exchange.kraken import \
        KrakenClient
except ImportError as e:
    print(f"Warning: Could not import trading bot modules: {e}")

    # Create mock classes for development
    class MockBotConfig:
        def __init__(self):
            self.config = {}

    class MockBot:
        def __init__(self, *args, **kwargs):
            self.running = False

        def start(self):
            self.running = True

        def stop(self):
            self.running = False

    BotConfig = MockBotConfig
    MultiCryptoTradingBot = MockBot

trading_bot_bp = Blueprint("trading_bot", __name__)

# Global bot instance
bot_instance = None
bot_config = None


def initialize_bot():
    """Initialize the trading bot with proper configuration"""
    global bot_instance, bot_config
    try:
        # Load configuration
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "config", "user_config.json"
        )
        bot_config = BotConfig(config_path)

        # Initialize components
        exchange_client = KrakenClient(bot_config)
        reddit_source = RedditSentimentSource(bot_config)
        news_source = NewsSentimentSource(bot_config)
        ai_analyzer = OpenAIAnalyzer(bot_config)
        portfolio_manager = PortfolioManager(bot_config, exchange_client)

        # Create bot instance
        bot_instance = MultiCryptoTradingBot(
            config=bot_config,
            exchange_client=exchange_client,
            sentiment_sources=[reddit_source, news_source],
            ai_analyzer=ai_analyzer,
            portfolio_manager=portfolio_manager,
        )
        return True
    except Exception as e:
        print(f"Failed to initialize bot: {e}")
        return False


@trading_bot_bp.route("/status", methods=["GET"])
@cross_origin()
def get_bot_status():
    """Get current bot status and system health"""
    try:
        # Mock data for demonstration
        status_data = {
            "bot_running": bot_instance.running if bot_instance else False,
            "paper_trading": True,
            "last_update": datetime.now().isoformat(),
            "api_status": {
                "kraken": {
                    "status": "connected",
                    "last_check": datetime.now().isoformat(),
                },
                "openai": {
                    "status": "connected",
                    "last_check": datetime.now().isoformat(),
                },
                "reddit": {
                    "status": "limited",
                    "last_check": datetime.now().isoformat(),
                },
                "news": {
                    "status": "connected",
                    "last_check": datetime.now().isoformat(),
                },
            },
            "system_health": "healthy",
        }
        return jsonify(status_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@trading_bot_bp.route("/portfolio", methods=["GET"])
@cross_origin()
def get_portfolio():
    """Get current portfolio information"""
    try:
        # Mock portfolio data
        portfolio_data = {
            "total_value": 16000.0,
            "daily_pnl": 320.0,
            "daily_pnl_percent": 2.1,
            "total_return": 12.5,
            "holdings": [
                {"symbol": "BTC", "amount": 0.35, "value": 5600.0, "allocation": 35.0},
                {"symbol": "ETH", "amount": 1.15, "value": 4000.0, "allocation": 25.0},
                {
                    "symbol": "XRP",
                    "amount": 2500.0,
                    "value": 2400.0,
                    "allocation": 15.0,
                },
                {"symbol": "LTC", "amount": 13.0, "value": 1600.0, "allocation": 10.0},
                {"symbol": "DOT", "amount": 56.0, "value": 1600.0, "allocation": 10.0},
                {"symbol": "DOGE", "amount": 2500.0, "value": 800.0, "allocation": 5.0},
            ],
            "performance_history": [
                {"date": "2024-01-01", "value": 10000},
                {"date": "2024-02-01", "value": 12000},
                {"date": "2024-03-01", "value": 11500},
                {"date": "2024-04-01", "value": 14000},
                {"date": "2024-05-01", "value": 13500},
                {"date": "2024-06-01", "value": 16000},
            ],
        }
        return jsonify(portfolio_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@trading_bot_bp.route("/trading-pairs", methods=["GET"])
@cross_origin()
def get_trading_pairs():
    """Get current trading pairs with signals and prices"""
    try:
        # Mock trading pairs data
        trading_pairs = [
            {
                "symbol": "BTC/USD",
                "price": 47800.0,
                "change_24h": 2.3,
                "volume": "1.2B",
                "signal": "BUY",
                "confidence": 85,
                "indicators": {"rsi": 45.2, "macd": 0.15, "bb_position": 0.7},
            },
            {
                "symbol": "ETH/USD",
                "price": 3450.0,
                "change_24h": 1.8,
                "volume": "800M",
                "signal": "HOLD",
                "confidence": 72,
                "indicators": {"rsi": 52.1, "macd": -0.05, "bb_position": 0.5},
            },
            {
                "symbol": "XRP/USD",
                "price": 0.95,
                "change_24h": -0.5,
                "volume": "450M",
                "signal": "SELL",
                "confidence": 68,
                "indicators": {"rsi": 65.8, "macd": -0.12, "bb_position": 0.3},
            },
            {
                "symbol": "LTC/USD",
                "price": 185.0,
                "change_24h": 3.2,
                "volume": "200M",
                "signal": "BUY",
                "confidence": 78,
                "indicators": {"rsi": 42.5, "macd": 0.08, "bb_position": 0.8},
            },
            {
                "symbol": "DOT/USD",
                "price": 28.5,
                "change_24h": -1.2,
                "volume": "150M",
                "signal": "HOLD",
                "confidence": 65,
                "indicators": {"rsi": 48.9, "macd": 0.02, "bb_position": 0.4},
            },
            {
                "symbol": "DOGE/USD",
                "price": 0.32,
                "change_24h": 5.8,
                "volume": "300M",
                "signal": "BUY",
                "confidence": 82,
                "indicators": {"rsi": 38.7, "macd": 0.18, "bb_position": 0.9},
            },
        ]
        return jsonify(trading_pairs)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@trading_bot_bp.route("/sentiment", methods=["GET"])
@cross_origin()
def get_sentiment():
    """Get sentiment analysis from multiple sources"""
    try:
        sentiment_data = {
            "overall_sentiment": "bullish",
            "overall_score": 0.65,
            "sources": [
                {
                    "name": "Reddit",
                    "sentiment": "bullish",
                    "score": 0.75,
                    "confidence": 0.82,
                    "posts_analyzed": 1250,
                    "trending_topics": [
                        "bitcoin rally",
                        "ethereum upgrade",
                        "altcoin season",
                    ],
                },
                {
                    "name": "News",
                    "sentiment": "neutral",
                    "score": 0.15,
                    "confidence": 0.68,
                    "posts_analyzed": 89,
                    "trending_topics": [
                        "regulation news",
                        "institutional adoption",
                        "market volatility",
                    ],
                },
                {
                    "name": "AI Analysis",
                    "sentiment": "bullish",
                    "score": 0.68,
                    "confidence": 0.91,
                    "posts_analyzed": 1,
                    "trending_topics": [
                        "technical breakout",
                        "momentum building",
                        "support levels",
                    ],
                },
            ],
            "sentiment_history": [
                {"date": "2024-06-01", "score": 0.45},
                {"date": "2024-06-02", "score": 0.52},
                {"date": "2024-06-03", "score": 0.48},
                {"date": "2024-06-04", "score": 0.61},
                {"date": "2024-06-05", "score": 0.58},
                {"date": "2024-06-06", "score": 0.65},
            ],
        }
        return jsonify(sentiment_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@trading_bot_bp.route("/metrics", methods=["GET"])
@cross_origin()
def get_metrics():
    """Get trading performance metrics"""
    try:
        metrics_data = {
            "total_trades": 47,
            "winning_trades": 32,
            "losing_trades": 15,
            "win_rate": 68.5,
            "profit_factor": 1.85,
            "sharpe_ratio": 1.85,
            "sortino_ratio": 2.12,
            "max_drawdown": 8.5,
            "var_95": 1240.0,
            "average_trade_duration": "4.2 hours",
            "best_trade": 450.0,
            "worst_trade": -180.0,
            "recent_trades": [
                {
                    "id": "T001",
                    "symbol": "BTC/USD",
                    "side": "buy",
                    "amount": 0.05,
                    "price": 47200.0,
                    "timestamp": (datetime.now() - timedelta(hours=2)).isoformat(),
                    "pnl": 30.0,
                    "status": "completed",
                },
                {
                    "id": "T002",
                    "symbol": "ETH/USD",
                    "side": "sell",
                    "amount": 0.2,
                    "price": 3420.0,
                    "timestamp": (datetime.now() - timedelta(hours=4)).isoformat(),
                    "pnl": -15.0,
                    "status": "completed",
                },
            ],
        }
        return jsonify(metrics_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@trading_bot_bp.route("/start", methods=["POST"])
@jwt_required()
@cross_origin()
def start_bot():
    """Start the trading bot"""
    try:
        global bot_instance
        if not bot_instance:
            if not initialize_bot():
                return jsonify({"error": "Failed to initialize bot"}), 500

        if bot_instance:
            bot_instance.start()
            return jsonify({"message": "Bot started successfully", "status": "running"})
        else:
            return jsonify({"error": "Bot not initialized"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@trading_bot_bp.route("/stop", methods=["POST"])
@jwt_required()
@cross_origin()
def stop_bot():
    """Stop the trading bot"""
    try:
        global bot_instance
        if bot_instance:
            bot_instance.stop()
            return jsonify({"message": "Bot stopped successfully", "status": "stopped"})
        else:
            return jsonify({"message": "Bot was not running", "status": "stopped"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@trading_bot_bp.route("/backtest", methods=["POST"])
@jwt_required()
@cross_origin()
def run_backtest():
    """Run a backtest with given parameters"""
    try:
        data = request.get_json()

        # Mock backtest results
        backtest_results = {
            "run_id": f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "parameters": data,
            "results": {
                "total_return": 24.5,
                "sharpe_ratio": 1.85,
                "sortino_ratio": 2.12,
                "max_drawdown": -8.2,
                "win_rate": 68.5,
                "total_trades": 156,
                "profit_factor": 1.92,
                "start_date": "2024-01-01",
                "end_date": "2024-06-01",
                "initial_capital": 10000.0,
                "final_capital": 12450.0,
            },
            "equity_curve": [
                {"date": "2024-01-01", "value": 10000},
                {"date": "2024-01-15", "value": 10200},
                {"date": "2024-02-01", "value": 10800},
                {"date": "2024-02-15", "value": 10600},
                {"date": "2024-03-01", "value": 11200},
                {"date": "2024-03-15", "value": 11800},
                {"date": "2024-04-01", "value": 11400},
                {"date": "2024-04-15", "value": 12000},
                {"date": "2024-05-01", "value": 12200},
                {"date": "2024-05-15", "value": 12100},
                {"date": "2024-06-01", "value": 12450},
            ],
        }

        return jsonify(backtest_results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Initialize bot on module load
initialize_bot()
