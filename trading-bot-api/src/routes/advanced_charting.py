"""
Advanced Charting API Routes

Provides REST endpoints for advanced charting features including:
- Technical indicators
- Drawing tools
- Volume profile
- Pattern recognition
- Multi-timeframe analysis
- Interactive annotations
"""

import json
import os
import sys
from datetime import datetime
from typing import Optional

import pandas as pd
from flask import Blueprint, jsonify, request

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

try:
    from core.advanced_charting import (AdvancedCharting, DrawingTool,
                                        DrawingType)
except ImportError:
    # Fallback if import fails
    AdvancedCharting = None
    DrawingTool = None
    DrawingType = None

advanced_charting_bp = Blueprint("advanced_charting", __name__)


def get_charting_engine() -> Optional[AdvancedCharting]:
    """Get advanced charting engine instance"""
    if not AdvancedCharting:
        return None

    config = {
        "default_indicators": ["RSI", "MACD", "BB", "SMA", "EMA"],
        "pattern_detection_enabled": True,
        "drawing_tools_enabled": True,
    }

    return AdvancedCharting(config)


def load_sample_price_data(asset: str = "BTC", timeframe: str = "1h") -> pd.DataFrame:
    """Load sample price data for charting"""
    try:
        # Try to load real data from cache
        cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "cache")
        cache_file = os.path.join(cache_dir, f"{asset}_cache.json")

        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    df = pd.DataFrame(data)
                    if "timestamp" in df.columns:
                        df["timestamp"] = pd.to_datetime(df["timestamp"])
                        df.set_index("timestamp", inplace=True)
                        return df

        # Generate sample data if no real data available
        import numpy as np

        # Generate date range based on timeframe
        if timeframe == "1m":
            periods = 1440  # 1 day of minute data
            freq = "1min"
        elif timeframe == "5m":
            periods = 2016  # 1 week of 5-minute data
            freq = "5min"
        elif timeframe == "1h":
            periods = 720  # 1 month of hourly data
            freq = "1H"
        elif timeframe == "4h":
            periods = 720  # 4 months of 4-hour data
            freq = "4H"
        elif timeframe == "1d":
            periods = 365  # 1 year of daily data
            freq = "1D"
        else:
            periods = 720
            freq = "1H"

        dates = pd.date_range(start="2024-01-01", periods=periods, freq=freq)

        # Generate realistic OHLCV data
        np.random.seed(hash(asset + timeframe) % 2**32)

        initial_price = {
            "BTC": 50000,
            "ETH": 3000,
            "LTC": 100,
            "XRP": 0.5,
            "DOGE": 0.1,
        }.get(asset, 1000)

        # Generate price series with realistic volatility
        returns = np.random.normal(0.0001, 0.02, len(dates))
        prices = initial_price * np.exp(np.cumsum(returns))

        # Generate OHLC from close prices
        opens = np.roll(prices, 1)
        opens[0] = initial_price

        # Add some intraday volatility
        volatility = np.abs(np.random.normal(0, 0.01, len(dates)))
        highs = prices * (1 + volatility)
        lows = prices * (1 - volatility)

        # Ensure OHLC relationships are correct
        for i in range(len(dates)):
            high = max(opens[i], prices[i], highs[i])
            low = min(opens[i], prices[i], lows[i])
            highs[i] = high
            lows[i] = low

        # Generate volume with correlation to price movements
        base_volume = np.random.uniform(1000, 10000)
        volume_multiplier = 1 + 2 * np.abs(returns)  # Higher volume on big moves
        volumes = base_volume * volume_multiplier * np.random.uniform(0.5, 2, len(dates))

        df = pd.DataFrame(
            {
                "open": opens,
                "high": highs,
                "low": lows,
                "close": prices,
                "volume": volumes,
            },
            index=dates,
        )

        return df

    except Exception as e:
        print(f"Error loading price data: {e}")
        return pd.DataFrame()


@advanced_charting_bp.route("/api/charting/indicators", methods=["GET"])
def get_technical_indicators():
    """Get technical indicators for a chart"""
    try:
        engine = get_charting_engine()
        if not engine:
            return jsonify({"error": "Advanced charting module not available"}), 500

        asset = request.args.get("asset", "BTC")
        timeframe = request.args.get("timeframe", "1h")
        indicators = request.args.get("indicators", "RSI,MACD,BB,SMA,EMA").split(",")

        # Load price data
        price_data = load_sample_price_data(asset, timeframe)
        if price_data.empty:
            return jsonify({"error": "No price data available"}), 404

        # Calculate indicators
        indicator_results = engine.calculate_technical_indicators(price_data, indicators)

        # Convert to serializable format
        serialized_results = {}
        for name, indicator in indicator_results.items():
            if hasattr(indicator, "data"):
                if isinstance(indicator.data, pd.DataFrame):
                    data_dict = {}
                    for col in indicator.data.columns:
                        data_dict[col] = indicator.data[col].dropna().to_dict()
                else:
                    data_dict = indicator.data.dropna().to_dict()

                serialized_results[name] = {
                    "name": indicator.name,
                    "type": indicator.type,
                    "parameters": indicator.parameters,
                    "data": data_dict,
                    "metadata": indicator.metadata,
                }

        return jsonify(
            {
                "asset": asset,
                "timeframe": timeframe,
                "indicators": serialized_results,
                "data_points": len(price_data),
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@advanced_charting_bp.route("/api/charting/volume-profile", methods=["GET"])
def get_volume_profile():
    """Get volume profile analysis"""
    try:
        engine = get_charting_engine()
        if not engine:
            return jsonify({"error": "Advanced charting module not available"}), 500

        asset = request.args.get("asset", "BTC")
        timeframe = request.args.get("timeframe", "1h")
        bins = request.args.get("bins", 50, type=int)

        # Load price data
        price_data = load_sample_price_data(asset, timeframe)
        if price_data.empty:
            return jsonify({"error": "No price data available"}), 404

        # Calculate volume profile
        volume_profile = engine.calculate_volume_profile(price_data, bins)

        return jsonify({"asset": asset, "timeframe": timeframe, "volume_profile": volume_profile})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@advanced_charting_bp.route("/api/charting/patterns", methods=["GET"])
def detect_patterns():
    """Detect chart patterns"""
    try:
        engine = get_charting_engine()
        if not engine:
            return jsonify({"error": "Advanced charting module not available"}), 500

        asset = request.args.get("asset", "BTC")
        timeframe = request.args.get("timeframe", "1h")

        # Load price data
        price_data = load_sample_price_data(asset, timeframe)
        if price_data.empty:
            return jsonify({"error": "No price data available"}), 404

        # Detect patterns
        patterns = engine.detect_chart_patterns(price_data)

        # Convert patterns to serializable format
        serialized_patterns = []
        for pattern in patterns:
            serialized_patterns.append(
                {
                    "type": pattern.type.value,
                    "confidence": pattern.confidence,
                    "start_time": pattern.start_time.isoformat(),
                    "end_time": pattern.end_time.isoformat(),
                    "key_points": pattern.key_points,
                    "description": pattern.description,
                    "metadata": pattern.metadata,
                }
            )

        return jsonify(
            {
                "asset": asset,
                "timeframe": timeframe,
                "patterns": serialized_patterns,
                "pattern_count": len(serialized_patterns),
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@advanced_charting_bp.route("/api/charting/multi-timeframe", methods=["GET"])
def multi_timeframe_analysis():
    """Get multi-timeframe analysis"""
    try:
        engine = get_charting_engine()
        if not engine:
            return jsonify({"error": "Advanced charting module not available"}), 500

        asset = request.args.get("asset", "BTC")
        timeframes = request.args.get("timeframes", "1h,4h,1d").split(",")

        # Load data for each timeframe
        price_data = {}
        for tf in timeframes:
            data = load_sample_price_data(asset, tf)
            if not data.empty:
                price_data[tf] = data

        if not price_data:
            return jsonify({"error": "No price data available for any timeframe"}), 404

        # Perform multi-timeframe analysis
        analysis = engine.create_multi_timeframe_analysis(price_data)

        return jsonify(
            {
                "asset": asset,
                "timeframes_analyzed": list(price_data.keys()),
                "analysis": analysis,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@advanced_charting_bp.route("/api/charting/drawings", methods=["GET", "POST"])
def manage_drawings():
    """Manage drawing tools on charts"""
    try:
        engine = get_charting_engine()
        if not engine:
            return jsonify({"error": "Advanced charting module not available"}), 500

        chart_id = request.args.get("chart_id", "default")

        if request.method == "GET":
            # Get existing drawings
            drawings = engine.get_chart_drawings(chart_id)

            # Convert to serializable format
            serialized_drawings = []
            for drawing in drawings:
                serialized_drawings.append(
                    {
                        "id": drawing.id,
                        "type": drawing.type.value,
                        "points": drawing.points,
                        "style": drawing.style,
                        "metadata": drawing.metadata,
                        "created_at": drawing.created_at.isoformat(),
                        "updated_at": drawing.updated_at.isoformat(),
                    }
                )

            return jsonify(
                {
                    "chart_id": chart_id,
                    "drawings": serialized_drawings,
                    "drawing_count": len(serialized_drawings),
                }
            )

        elif request.method == "POST":
            # Add new drawing
            drawing_data = request.get_json()
            if not drawing_data:
                return jsonify({"error": "No drawing data provided"}), 400

            # Validate required fields
            required_fields = ["type", "points"]
            for field in required_fields:
                if field not in drawing_data:
                    return jsonify({"error": f"Missing required field: {field}"}), 400

            # Create drawing tool
            if not DrawingType:
                return jsonify({"error": "Drawing tools not available"}), 500

            try:
                drawing_type = DrawingType(drawing_data["type"])
            except ValueError:
                return (
                    jsonify({"error": f"Invalid drawing type: {drawing_data['type']}"}),
                    400,
                )

            drawing = DrawingTool(
                id=f"drawing_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                type=drawing_type,
                points=drawing_data["points"],
                style=drawing_data.get("style", {}),
                metadata=drawing_data.get("metadata", {}),
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )

            # Add to chart
            success = engine.add_drawing_tool(chart_id, drawing)

            if success:
                return jsonify(
                    {
                        "status": "success",
                        "drawing_id": drawing.id,
                        "message": "Drawing added successfully",
                    }
                )
            else:
                return jsonify({"error": "Failed to add drawing"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@advanced_charting_bp.route("/api/charting/annotations", methods=["GET", "POST"])
def manage_annotations():
    """Manage annotations on charts"""
    try:
        engine = get_charting_engine()
        if not engine:
            return jsonify({"error": "Advanced charting module not available"}), 500

        chart_id = request.args.get("chart_id", "default")

        if request.method == "GET":
            # Get existing annotations
            annotations = engine.get_chart_annotations(chart_id)

            return jsonify(
                {
                    "chart_id": chart_id,
                    "annotations": annotations,
                    "annotation_count": len(annotations),
                }
            )

        elif request.method == "POST":
            # Add new annotation
            annotation_data = request.get_json()
            if not annotation_data:
                return jsonify({"error": "No annotation data provided"}), 400

            # Validate required fields
            required_fields = ["x", "y", "text"]
            for field in required_fields:
                if field not in annotation_data:
                    return jsonify({"error": f"Missing required field: {field}"}), 400

            # Add annotation
            success = engine.add_annotation(chart_id, annotation_data)

            if success:
                return jsonify({"status": "success", "message": "Annotation added successfully"})
            else:
                return jsonify({"error": "Failed to add annotation"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@advanced_charting_bp.route("/api/charting/price-data", methods=["GET"])
def get_price_data():
    """Get price data for charting"""
    try:
        asset = request.args.get("asset", "BTC")
        timeframe = request.args.get("timeframe", "1h")
        limit = request.args.get("limit", 500, type=int)

        # Load price data
        price_data = load_sample_price_data(asset, timeframe)
        if price_data.empty:
            return jsonify({"error": "No price data available"}), 404

        # Limit data points
        if len(price_data) > limit:
            price_data = price_data.tail(limit)

        # Convert to format suitable for charting
        chart_data = []
        for timestamp, row in price_data.iterrows():
            chart_data.append(
                {
                    "timestamp": timestamp.isoformat(),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                }
            )

        return jsonify(
            {
                "asset": asset,
                "timeframe": timeframe,
                "data": chart_data,
                "data_points": len(chart_data),
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@advanced_charting_bp.route("/api/charting/fibonacci", methods=["POST"])
def calculate_fibonacci():
    """Calculate Fibonacci retracement levels"""
    try:
        data = request.get_json()
        if not data or "high" not in data or "low" not in data:
            return jsonify({"error": "High and low prices required"}), 400

        high = float(data["high"])
        low = float(data["low"])

        # Calculate Fibonacci levels
        diff = high - low
        levels = {
            "0.0": high,
            "23.6": high - (diff * 0.236),
            "38.2": high - (diff * 0.382),
            "50.0": high - (diff * 0.5),
            "61.8": high - (diff * 0.618),
            "78.6": high - (diff * 0.786),
            "100.0": low,
        }

        # Extension levels
        extensions = {
            "127.2": low - (diff * 0.272),
            "161.8": low - (diff * 0.618),
            "261.8": low - (diff * 1.618),
        }

        return jsonify(
            {
                "high": high,
                "low": low,
                "retracement_levels": levels,
                "extension_levels": extensions,
                "range": diff,
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@advanced_charting_bp.route("/api/charting/support-resistance", methods=["GET"])
def find_support_resistance():
    """Find support and resistance levels"""
    try:
        asset = request.args.get("asset", "BTC")
        timeframe = request.args.get("timeframe", "1h")

        # Load price data
        price_data = load_sample_price_data(asset, timeframe)
        if price_data.empty:
            return jsonify({"error": "No price data available"}), 404

        # Find support and resistance levels
        highs = price_data["high"]
        lows = price_data["low"]

        window = 20
        resistance_levels = []
        support_levels = []

        # Find local maxima and minima
        for i in range(window, len(highs) - window):
            if highs.iloc[i] == highs.iloc[i - window : i + window + 1].max():
                resistance_levels.append(
                    {
                        "price": float(highs.iloc[i]),
                        "timestamp": highs.index[i].isoformat(),
                        "strength": 1,  # Could be calculated based on how many times level was tested
                    }
                )

            if lows.iloc[i] == lows.iloc[i - window : i + window + 1].min():
                support_levels.append(
                    {
                        "price": float(lows.iloc[i]),
                        "timestamp": lows.index[i].isoformat(),
                        "strength": 1,
                    }
                )

        # Sort by price and remove duplicates
        resistance_levels = sorted(resistance_levels, key=lambda x: x["price"], reverse=True)
        support_levels = sorted(support_levels, key=lambda x: x["price"])

        # Remove levels that are too close to each other (within 1%)
        filtered_resistance = []
        filtered_support = []

        for level in resistance_levels:
            if (
                not filtered_resistance
                or abs(level["price"] - filtered_resistance[-1]["price"]) / level["price"] > 0.01
            ):
                filtered_resistance.append(level)

        for level in support_levels:
            if (
                not filtered_support
                or abs(level["price"] - filtered_support[-1]["price"]) / level["price"] > 0.01
            ):
                filtered_support.append(level)

        return jsonify(
            {
                "asset": asset,
                "timeframe": timeframe,
                "resistance_levels": filtered_resistance[:10],  # Top 10
                "support_levels": filtered_support[:10],  # Top 10
                "current_price": float(price_data["close"].iloc[-1]),
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@advanced_charting_bp.route("/api/charting/status", methods=["GET"])
def charting_status():
    """Get advanced charting module status"""
    try:
        engine = get_charting_engine()

        if not engine:
            return jsonify(
                {
                    "status": "unavailable",
                    "error": "Advanced charting module not available",
                }
            )

        return jsonify(
            {
                "status": "available",
                "features": [
                    "technical_indicators",
                    "volume_profile",
                    "pattern_recognition",
                    "multi_timeframe_analysis",
                    "drawing_tools",
                    "fibonacci_retracements",
                    "support_resistance_detection",
                    "interactive_annotations",
                ],
                "supported_indicators": [
                    "RSI",
                    "MACD",
                    "Bollinger Bands",
                    "SMA",
                    "EMA",
                    "Stochastic",
                    "ATR",
                ],
                "supported_patterns": [
                    "Double Top",
                    "Double Bottom",
                    "Head and Shoulders",
                    "Triangle",
                    "Flag",
                    "Pennant",
                ],
                "supported_timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"],
                "drawing_tools": [
                    "trend_line",
                    "horizontal_line",
                    "fibonacci_retracement",
                    "support_resistance",
                    "rectangle",
                    "arrow",
                    "text_annotation",
                ],
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500
