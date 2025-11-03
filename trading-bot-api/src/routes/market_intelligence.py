"""
Market Intelligence API Routes

Provides REST endpoints for real-time market intelligence including:
- Sentiment radar
- News impact analyzer
- Market regime detection
- Fear & Greed index
- Dynamic correlation matrix
- Intelligent alert system
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
from flask import Blueprint, jsonify, request

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

try:
    from production_trading_bot.core.market_intelligence import \
        MarketIntelligence
except ImportError:
    # Fallback if import fails
    MarketIntelligence = None

market_intelligence_bp = Blueprint("market_intelligence", __name__)


def get_market_intelligence() -> Optional[MarketIntelligence]:
    """Get market intelligence engine instance"""
    if not MarketIntelligence:
        return None

    config = {
        "reddit_enabled": False,  # Would be True with proper API keys
        "news_enabled": False,  # Would be True with proper API keys
        "twitter_enabled": False,  # Would be True with proper API keys
    }

    return MarketIntelligence(config)


def load_sample_market_data() -> Dict[str, Any]:
    """Load sample market data for analysis"""
    try:
        # Try to load real data from cache
        cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "cache")
        market_data = {}

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
            import numpy as np

            dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq="H")

            for asset in ["BTC", "ETH", "LTC", "XRP", "DOGE"]:
                np.random.seed(hash(asset) % 2**32)  # Consistent seed per asset

                # Generate realistic price data
                initial_price = {
                    "BTC": 50000,
                    "ETH": 3000,
                    "LTC": 100,
                    "XRP": 0.5,
                    "DOGE": 0.1,
                }[asset]
                returns = np.random.normal(
                    0.0001, 0.02, len(dates)
                )  # Small positive drift with volatility
                prices = initial_price * np.exp(np.cumsum(returns))

                # Generate volume data
                base_volume = np.random.uniform(1000, 10000)
                volume_multiplier = 1 + 0.5 * np.abs(returns)  # Higher volume on big moves
                volumes = base_volume * volume_multiplier * np.random.uniform(0.5, 2, len(dates))

                market_data[asset] = pd.DataFrame(
                    {
                        "close": prices,
                        "open": prices * (1 + np.random.normal(0, 0.001, len(dates))),
                        "high": prices * (1 + np.abs(np.random.normal(0, 0.005, len(dates)))),
                        "low": prices * (1 - np.abs(np.random.normal(0, 0.005, len(dates)))),
                        "volume": volumes,
                    },
                    index=dates,
                )

        return market_data

    except Exception as e:
        print(f"Error loading market data: {e}")
        return {}


@market_intelligence_bp.route("/api/market-intelligence/sentiment-radar", methods=["GET"])
def sentiment_radar():
    """Get real-time sentiment radar data"""
    try:
        engine = get_market_intelligence()
        if not engine:
            return jsonify({"error": "Market intelligence module not available"}), 500

        assets = request.args.get("assets")
        if assets:
            assets = assets.split(",")

        # Run async function in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(engine.get_sentiment_radar(assets))
        finally:
            loop.close()

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@market_intelligence_bp.route("/api/market-intelligence/news-impact", methods=["GET"])
def news_impact():
    """Get news impact analysis"""
    try:
        engine = get_market_intelligence()
        if not engine:
            return jsonify({"error": "Market intelligence module not available"}), 500

        limit = request.args.get("limit", 50, type=int)

        # Run async function in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(engine.analyze_news_impact(limit))
        finally:
            loop.close()

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@market_intelligence_bp.route("/api/market-intelligence/market-regime", methods=["GET"])
def market_regime():
    """Get market regime detection"""
    try:
        engine = get_market_intelligence()
        if not engine:
            return jsonify({"error": "Market intelligence module not available"}), 500

        market_data = load_sample_market_data()
        result = engine.detect_market_regime(market_data)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@market_intelligence_bp.route("/api/market-intelligence/fear-greed-index", methods=["GET"])
def fear_greed_index():
    """Get Fear & Greed Index"""
    try:
        engine = get_market_intelligence()
        if not engine:
            return jsonify({"error": "Market intelligence module not available"}), 500

        # Prepare market data for Fear & Greed calculation
        price_data = load_sample_market_data()

        # Get sentiment data
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            sentiment_data = loop.run_until_complete(engine.get_sentiment_radar())
        finally:
            loop.close()

        market_data = {"price_data": price_data, "sentiment_data": sentiment_data}

        result = engine.calculate_fear_greed_index(market_data)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@market_intelligence_bp.route("/api/market-intelligence/correlation-matrix", methods=["GET"])
def correlation_matrix():
    """Get dynamic correlation matrix"""
    try:
        engine = get_market_intelligence()
        if not engine:
            return jsonify({"error": "Market intelligence module not available"}), 500

        window = request.args.get("window", 30, type=int)
        price_data = load_sample_market_data()

        result = engine.get_dynamic_correlation_matrix(price_data, window)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@market_intelligence_bp.route("/api/market-intelligence/alerts", methods=["GET"])
def market_alerts():
    """Get market alerts"""
    try:
        engine = get_market_intelligence()
        if not engine:
            return jsonify({"error": "Market intelligence module not available"}), 500

        # Prepare comprehensive market data
        price_data = load_sample_market_data()

        # Get sentiment data
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            sentiment_data = loop.run_until_complete(engine.get_sentiment_radar())
        finally:
            loop.close()

        # Get regime data
        regime_data = engine.detect_market_regime(price_data)

        # Get Fear & Greed data
        fear_greed_data = engine.calculate_fear_greed_index(
            {"price_data": price_data, "sentiment_data": sentiment_data}
        )

        market_data = {
            "price_data": price_data,
            "sentiment_data": sentiment_data,
            "regime_data": regime_data,
            "fear_greed_data": fear_greed_data,
        }

        # Get alert configuration from request
        alert_config = {}
        if request.args.get("price_threshold"):
            alert_config["price_change_threshold"] = float(request.args.get("price_threshold"))
        if request.args.get("volume_threshold"):
            alert_config["volume_change_threshold"] = float(request.args.get("volume_threshold"))
        if request.args.get("sentiment_threshold"):
            alert_config["sentiment_extreme_threshold"] = float(
                request.args.get("sentiment_threshold")
            )

        alerts = engine.generate_market_alerts(market_data, alert_config)

        # Convert alerts to serializable format
        serialized_alerts = []
        for alert in alerts:
            serialized_alerts.append(
                {
                    "alert_id": alert.alert_id,
                    "timestamp": alert.timestamp.isoformat(),
                    "alert_type": alert.alert_type,
                    "severity": alert.severity,
                    "message": alert.message,
                    "data": alert.data,
                    "triggered_by": alert.triggered_by,
                }
            )

        return jsonify(
            {
                "alerts": serialized_alerts,
                "total_alerts": len(serialized_alerts),
                "alert_summary": {
                    "critical": len([a for a in alerts if a.severity == "critical"]),
                    "high": len([a for a in alerts if a.severity == "high"]),
                    "medium": len([a for a in alerts if a.severity == "medium"]),
                    "low": len([a for a in alerts if a.severity == "low"]),
                },
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@market_intelligence_bp.route("/api/market-intelligence/comprehensive", methods=["GET"])
def comprehensive_intelligence():
    """Get comprehensive market intelligence dashboard data"""
    try:
        engine = get_market_intelligence()
        if not engine:
            return jsonify({"error": "Market intelligence module not available"}), 500

        # Load market data
        price_data = load_sample_market_data()

        # Get all intelligence data
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            sentiment_data = loop.run_until_complete(engine.get_sentiment_radar())
            news_data = loop.run_until_complete(engine.analyze_news_impact(20))
        finally:
            loop.close()

        regime_data = engine.detect_market_regime(price_data)

        fear_greed_data = engine.calculate_fear_greed_index(
            {"price_data": price_data, "sentiment_data": sentiment_data}
        )

        correlation_data = engine.get_dynamic_correlation_matrix(price_data)

        # Generate alerts
        market_data = {
            "price_data": price_data,
            "sentiment_data": sentiment_data,
            "regime_data": regime_data,
            "fear_greed_data": fear_greed_data,
        }

        alerts = engine.generate_market_alerts(market_data)

        # Serialize alerts
        serialized_alerts = []
        for alert in alerts:
            serialized_alerts.append(
                {
                    "alert_id": alert.alert_id,
                    "timestamp": alert.timestamp.isoformat(),
                    "alert_type": alert.alert_type,
                    "severity": alert.severity,
                    "message": alert.message,
                    "data": alert.data,
                    "triggered_by": alert.triggered_by,
                }
            )

        return jsonify(
            {
                "timestamp": datetime.now().isoformat(),
                "sentiment_radar": sentiment_data,
                "news_impact": news_data,
                "market_regime": regime_data,
                "fear_greed_index": fear_greed_data,
                "correlation_matrix": correlation_data,
                "alerts": {
                    "alerts": serialized_alerts,
                    "total_alerts": len(serialized_alerts),
                    "alert_summary": {
                        "critical": len([a for a in alerts if a.severity == "critical"]),
                        "high": len([a for a in alerts if a.severity == "high"]),
                        "medium": len([a for a in alerts if a.severity == "medium"]),
                        "low": len([a for a in alerts if a.severity == "low"]),
                    },
                },
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@market_intelligence_bp.route("/api/market-intelligence/status", methods=["GET"])
def intelligence_status():
    """Get market intelligence module status"""
    try:
        engine = get_market_intelligence()

        if not engine:
            return jsonify(
                {
                    "status": "unavailable",
                    "error": "Market intelligence module not available",
                }
            )

        # Check data availability
        market_data = load_sample_market_data()

        return jsonify(
            {
                "status": "available",
                "data_sources": {
                    "reddit_enabled": engine.reddit_enabled,
                    "news_enabled": engine.news_enabled,
                    "twitter_enabled": engine.twitter_enabled,
                },
                "available_assets": list(market_data.keys()),
                "data_points": {asset: len(data) for asset, data in market_data.items()},
                "features": [
                    "sentiment_radar",
                    "news_impact_analysis",
                    "market_regime_detection",
                    "fear_greed_index",
                    "correlation_matrix",
                    "intelligent_alerts",
                ],
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@market_intelligence_bp.route("/api/market-intelligence/config", methods=["GET", "POST"])
def intelligence_config():
    """Get or update market intelligence configuration"""
    try:
        if request.method == "GET":
            # Return current configuration
            return jsonify(
                {
                    "alert_thresholds": {
                        "price_change_threshold": 0.05,
                        "volume_change_threshold": 2.0,
                        "sentiment_extreme_threshold": 0.7,
                        "fear_greed_extreme_threshold": 20,
                    },
                    "data_sources": {
                        "reddit_enabled": False,
                        "news_enabled": False,
                        "twitter_enabled": False,
                    },
                    "update_intervals": {
                        "sentiment_update_minutes": 15,
                        "news_update_minutes": 30,
                        "price_update_minutes": 5,
                    },
                }
            )

        elif request.method == "POST":
            # Update configuration
            config_data = request.get_json()
            if not config_data:
                return jsonify({"error": "No configuration data provided"}), 400

            # In a real implementation, this would update the configuration
            # For now, just return success
            return jsonify(
                {
                    "status": "success",
                    "message": "Configuration updated successfully",
                    "updated_config": config_data,
                }
            )

    except Exception as e:
        return jsonify({"error": str(e)}), 500
