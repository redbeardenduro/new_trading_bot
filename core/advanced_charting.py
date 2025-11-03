"""
Advanced Charting Module

Provides comprehensive charting enhancements including:
- Drawing Tools (trend lines, support/resistance, Fibonacci retracements)
- Custom Indicators (RSI, MACD, Bollinger Bands with customization)
- Multi-timeframe Analysis
- Volume Profile
- Pattern Recognition
- Interactive Annotations
"""

import logging
import warnings
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class DrawingType(Enum):
    TREND_LINE = "trend_line"
    HORIZONTAL_LINE = "horizontal_line"
    FIBONACCI_RETRACEMENT = "fibonacci_retracement"
    SUPPORT_RESISTANCE = "support_resistance"
    RECTANGLE = "rectangle"
    ARROW = "arrow"
    TEXT_ANNOTATION = "text_annotation"


class PatternType(Enum):
    HEAD_SHOULDERS = "head_shoulders"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    TRIANGLE = "triangle"
    WEDGE = "wedge"
    FLAG = "flag"
    PENNANT = "pennant"


@dataclass
class DrawingTool:
    """Container for drawing tool data"""

    id: str
    type: DrawingType
    points: List[Dict[str, float]]
    style: Dict[str, Any]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


@dataclass
class TechnicalIndicator:
    """Container for technical indicator data"""

    name: str
    type: str
    parameters: Dict[str, Any]
    data: pd.Series
    metadata: Dict[str, Any]


@dataclass
class ChartPattern:
    """Container for detected chart pattern"""

    type: PatternType
    confidence: float
    start_time: datetime
    end_time: datetime
    key_points: List[Dict[str, float]]
    description: str
    metadata: Dict[str, Any]


class AdvancedCharting:
    """Advanced charting engine with drawing tools and analysis"""

    def __init__(self, config: Dict[str, Any] = None) -> None:
        """
        Initialize advanced charting

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.drawings = {}
        self.indicators = {}
        self.patterns = {}
        self.annotations = {}

    def calculate_technical_indicators(
        self, price_data: pd.DataFrame, indicators: List[str] = None
    ) -> Dict[str, TechnicalIndicator]:
        """
        Calculate technical indicators

        Args:
            price_data: DataFrame with OHLCV data
            indicators: List of indicators to calculate

        Returns:
            Dict of calculated indicators
        """
        try:
            if indicators is None:
                indicators = ["RSI", "MACD", "BB", "SMA", "EMA", "STOCH", "ATR"]
            results = {}
            if "close" not in price_data.columns:
                return {"error": "Price data must contain 'close' column"}
            prices = price_data["close"]
            highs = price_data.get("high", prices)
            lows = price_data.get("low", prices)
            volumes = price_data.get("volume", pd.Series(index=prices.index, data=0))
            if "RSI" in indicators:
                rsi_data = self._calculate_rsi(prices)
                results["RSI"] = TechnicalIndicator(
                    name="RSI",
                    type="oscillator",
                    parameters={"period": 14},
                    data=rsi_data,
                    metadata={"range": [0, 100], "overbought": 70, "oversold": 30},
                )
            if "MACD" in indicators:
                macd_data = self._calculate_macd(prices)
                results["MACD"] = TechnicalIndicator(
                    name="MACD",
                    type="momentum",
                    parameters={"fast": 12, "slow": 26, "signal": 9},
                    data=macd_data,
                    metadata={"components": ["macd", "signal", "histogram"]},
                )
            if "BB" in indicators:
                bb_data = self._calculate_bollinger_bands(prices)
                results["BB"] = TechnicalIndicator(
                    name="Bollinger Bands",
                    type="volatility",
                    parameters={"period": 20, "std_dev": 2},
                    data=bb_data,
                    metadata={"components": ["upper", "middle", "lower"]},
                )
            if "SMA" in indicators:
                sma_data = self._calculate_sma(prices, [20, 50, 200])
                results["SMA"] = TechnicalIndicator(
                    name="Simple Moving Average",
                    type="trend",
                    parameters={"periods": [20, 50, 200]},
                    data=sma_data,
                    metadata={"components": ["SMA_20", "SMA_50", "SMA_200"]},
                )
            if "EMA" in indicators:
                ema_data = self._calculate_ema(prices, [12, 26, 50])
                results["EMA"] = TechnicalIndicator(
                    name="Exponential Moving Average",
                    type="trend",
                    parameters={"periods": [12, 26, 50]},
                    data=ema_data,
                    metadata={"components": ["EMA_12", "EMA_26", "EMA_50"]},
                )
            if "STOCH" in indicators:
                stoch_data = self._calculate_stochastic(highs, lows, prices)
                results["STOCH"] = TechnicalIndicator(
                    name="Stochastic Oscillator",
                    type="oscillator",
                    parameters={"k_period": 14, "d_period": 3},
                    data=stoch_data,
                    metadata={"range": [0, 100], "overbought": 80, "oversold": 20},
                )
            if "ATR" in indicators:
                atr_data = self._calculate_atr(highs, lows, prices)
                results["ATR"] = TechnicalIndicator(
                    name="Average True Range",
                    type="volatility",
                    parameters={"period": 14},
                    data=atr_data,
                    metadata={"description": "Measures market volatility"},
                )
            return results
        except Exception as e:
            logger.error("Error calculating technical indicators: %s", e)
            return {"error": str(e)}

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - 100 / (1 + rs)
        return rsi

    def _calculate_macd(
        self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> pd.DataFrame:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        return pd.DataFrame({"macd": macd, "signal": signal_line, "histogram": histogram})

    def _calculate_bollinger_bands(
        self, prices: pd.Series, period: int = 20, std_dev: float = 2
    ) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + std * std_dev
        lower = sma - std * std_dev
        return pd.DataFrame({"upper": upper, "middle": sma, "lower": lower})

    def _calculate_sma(self, prices: pd.Series, periods: List[int]) -> pd.DataFrame:
        """Calculate Simple Moving Averages"""
        result = pd.DataFrame(index=prices.index)
        for period in periods:
            result[f"SMA_{period}"] = prices.rolling(window=period).mean()
        return result

    def _calculate_ema(self, prices: pd.Series, periods: List[int]) -> pd.DataFrame:
        """Calculate Exponential Moving Averages"""
        result = pd.DataFrame(index=prices.index)
        for period in periods:
            result[f"EMA_{period}"] = prices.ewm(span=period).mean()
        return result

    def _calculate_stochastic(
        self,
        highs: pd.Series,
        lows: pd.Series,
        closes: pd.Series,
        k_period: int = 14,
        d_period: int = 3,
    ) -> pd.DataFrame:
        """Calculate Stochastic Oscillator"""
        lowest_low = lows.rolling(window=k_period).min()
        highest_high = highs.rolling(window=k_period).max()
        k_percent = 100 * ((closes - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return pd.DataFrame({"k_percent": k_percent, "d_percent": d_percent})

    def _calculate_atr(
        self, highs: pd.Series, lows: pd.Series, closes: pd.Series, period: int = 14
    ) -> pd.Series:
        """Calculate Average True Range"""
        high_low = highs - lows
        high_close = np.abs(highs - closes.shift())
        low_close = np.abs(lows - closes.shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr

    def calculate_volume_profile(self, price_data: pd.DataFrame, bins: int = 50) -> Dict[str, Any]:
        """
        Calculate volume profile

        Args:
            price_data: DataFrame with OHLCV data
            bins: Number of price bins

        Returns:
            Dict containing volume profile data
        """
        try:
            if "volume" not in price_data.columns or "close" not in price_data.columns:
                return {"error": "Price data must contain 'volume' and 'close' columns"}
            prices = price_data["close"]
            volumes = price_data["volume"]
            price_min = prices.min()
            price_max = prices.max()
            price_bins = np.linspace(price_min, price_max, bins + 1)
            volume_profile = []
            for i in range(len(price_bins) - 1):
                bin_min = price_bins[i]
                bin_max = price_bins[i + 1]
                in_bin = (prices >= bin_min) & (prices < bin_max)
                bin_volume = volumes[in_bin].sum()
                volume_profile.append(
                    {
                        "price_level": (bin_min + bin_max) / 2,
                        "volume": bin_volume,
                        "price_min": bin_min,
                        "price_max": bin_max,
                    }
                )
            volume_profile.sort(key=lambda x: x["volume"], reverse=True)
            total_volume = sum((vp["volume"] for vp in volume_profile))
            poc_level = volume_profile[0]
            value_area_volume = total_volume * 0.7
            cumulative_volume = 0
            value_area_levels = []
            for vp in volume_profile:
                if cumulative_volume < value_area_volume:
                    value_area_levels.append(vp)
                    cumulative_volume += vp["volume"]
                else:
                    break
            value_area_levels.sort(key=lambda x: x["price_level"])
            value_area_high = (
                value_area_levels[-1]["price_level"]
                if value_area_levels
                else poc_level["price_level"]
            )
            value_area_low = (
                value_area_levels[0]["price_level"]
                if value_area_levels
                else poc_level["price_level"]
            )
            return {
                "timestamp": datetime.now().isoformat(),
                "volume_profile": volume_profile,
                "poc_level": poc_level,
                "value_area_high": value_area_high,
                "value_area_low": value_area_low,
                "total_volume": total_volume,
                "price_range": {"min": price_min, "max": price_max},
                "analysis": {
                    "dominant_price_level": poc_level["price_level"],
                    "value_area_percentage": 70,
                    "volume_distribution": (
                        "normal"
                        if abs(poc_level["price_level"] - (price_min + price_max) / 2)
                        < (price_max - price_min) * 0.1
                        else "skewed"
                    ),
                },
            }
        except Exception as e:
            logger.error("Error calculating volume profile: %s", e)
            return {"error": str(e)}

    def detect_chart_patterns(self, price_data: pd.DataFrame) -> List[ChartPattern]:
        """
        Detect chart patterns

        Args:
            price_data: DataFrame with OHLCV data

        Returns:
            List of detected patterns
        """
        try:
            patterns = []
            if "high" not in price_data.columns or "low" not in price_data.columns:
                return patterns
            highs = price_data["high"]
            lows = price_data["low"]
            closes = price_data["close"]
            double_tops = self._detect_double_top(highs, closes)
            patterns.extend(double_tops)
            double_bottoms = self._detect_double_bottom(lows, closes)
            patterns.extend(double_bottoms)
            head_shoulders = self._detect_head_shoulders(highs, lows, closes)
            patterns.extend(head_shoulders)
            triangles = self._detect_triangles(highs, lows)
            patterns.extend(triangles)
            flags = self._detect_flags(highs, lows, closes)
            patterns.extend(flags)
            return patterns
        except Exception as e:
            logger.error("Error detecting chart patterns: %s", e)
            return []

    def _detect_double_top(self, highs: pd.Series, closes: pd.Series) -> List[ChartPattern]:
        """Detect Double Top patterns"""
        patterns = []
        try:
            window = 10
            local_maxima = []
            for i in range(window, len(highs) - window):
                if highs.iloc[i] == highs.iloc[i - window : i + window + 1].max():
                    local_maxima.append((i, highs.iloc[i]))
            for i in range(len(local_maxima) - 1):
                (peak1_idx, peak1_price) = local_maxima[i]
                (peak2_idx, peak2_price) = local_maxima[i + 1]
                if abs(peak1_price - peak2_price) / peak1_price < 0.02:
                    valley_start = peak1_idx
                    valley_end = peak2_idx
                    valley_idx = valley_start + highs.iloc[valley_start:valley_end].idxmin()
                    valley_price = highs.iloc[valley_idx]
                    if (peak1_price - valley_price) / peak1_price > 0.03:
                        confidence = min(0.9, 1 - abs(peak1_price - peak2_price) / peak1_price * 10)
                        patterns.append(
                            ChartPattern(
                                type=PatternType.DOUBLE_TOP,
                                confidence=confidence,
                                start_time=highs.index[peak1_idx],
                                end_time=highs.index[peak2_idx],
                                key_points=[
                                    {"x": peak1_idx, "y": peak1_price, "type": "peak1"},
                                    {"x": valley_idx, "y": valley_price, "type": "valley"},
                                    {"x": peak2_idx, "y": peak2_price, "type": "peak2"},
                                ],
                                description=f"Double Top pattern with peaks at {peak1_price:.2f} and {peak2_price:.2f}",
                                metadata={"resistance_level": max(peak1_price, peak2_price)},
                            )
                        )
        except Exception as e:
            logger.error("Error detecting double top: %s", e)
        return patterns

    def _detect_double_bottom(self, lows: pd.Series, closes: pd.Series) -> List[ChartPattern]:
        """Detect Double Bottom patterns"""
        patterns = []
        try:
            window = 10
            local_minima = []
            for i in range(window, len(lows) - window):
                if lows.iloc[i] == lows.iloc[i - window : i + window + 1].min():
                    local_minima.append((i, lows.iloc[i]))
            for i in range(len(local_minima) - 1):
                (bottom1_idx, bottom1_price) = local_minima[i]
                (bottom2_idx, bottom2_price) = local_minima[i + 1]
                if abs(bottom1_price - bottom2_price) / bottom1_price < 0.02:
                    peak_start = bottom1_idx
                    peak_end = bottom2_idx
                    peak_idx = peak_start + lows.iloc[peak_start:peak_end].idxmax()
                    peak_price = lows.iloc[peak_idx]
                    if (peak_price - bottom1_price) / bottom1_price > 0.03:
                        confidence = min(
                            0.9, 1 - abs(bottom1_price - bottom2_price) / bottom1_price * 10
                        )
                        patterns.append(
                            ChartPattern(
                                type=PatternType.DOUBLE_BOTTOM,
                                confidence=confidence,
                                start_time=lows.index[bottom1_idx],
                                end_time=lows.index[bottom2_idx],
                                key_points=[
                                    {"x": bottom1_idx, "y": bottom1_price, "type": "bottom1"},
                                    {"x": peak_idx, "y": peak_price, "type": "peak"},
                                    {"x": bottom2_idx, "y": bottom2_price, "type": "bottom2"},
                                ],
                                description=f"Double Bottom pattern with bottoms at {bottom1_price:.2f} and {bottom2_price:.2f}",
                                metadata={"support_level": min(bottom1_price, bottom2_price)},
                            )
                        )
        except Exception as e:
            logger.error("Error detecting double bottom: %s", e)
        return patterns

    def _detect_head_shoulders(
        self, highs: pd.Series, lows: pd.Series, closes: pd.Series
    ) -> List[ChartPattern]:
        """Detect Head and Shoulders patterns"""
        patterns = []
        try:
            window = 15
            local_maxima = []
            for i in range(window, len(highs) - window):
                if highs.iloc[i] == highs.iloc[i - window : i + window + 1].max():
                    local_maxima.append((i, highs.iloc[i]))
            for i in range(len(local_maxima) - 2):
                (left_shoulder_idx, left_shoulder_price) = local_maxima[i]
                (head_idx, head_price) = local_maxima[i + 1]
                (right_shoulder_idx, right_shoulder_price) = local_maxima[i + 2]
                if (
                    head_price > left_shoulder_price
                    and head_price > right_shoulder_price
                    and (
                        abs(left_shoulder_price - right_shoulder_price) / left_shoulder_price < 0.05
                    )
                ):
                    left_valley_idx = (
                        left_shoulder_idx + lows.iloc[left_shoulder_idx:head_idx].idxmin()
                    )
                    right_valley_idx = head_idx + lows.iloc[head_idx:right_shoulder_idx].idxmin()
                    left_valley_price = lows.iloc[left_valley_idx]
                    right_valley_price = lows.iloc[right_valley_idx]
                    neckline_price = (left_valley_price + right_valley_price) / 2
                    shoulder_symmetry = (
                        1 - abs(left_shoulder_price - right_shoulder_price) / left_shoulder_price
                    )
                    head_prominence = (
                        head_price - max(left_shoulder_price, right_shoulder_price)
                    ) / head_price
                    confidence = min(0.9, (shoulder_symmetry + head_prominence) / 2)
                    patterns.append(
                        ChartPattern(
                            type=PatternType.HEAD_SHOULDERS,
                            confidence=confidence,
                            start_time=highs.index[left_shoulder_idx],
                            end_time=highs.index[right_shoulder_idx],
                            key_points=[
                                {
                                    "x": left_shoulder_idx,
                                    "y": left_shoulder_price,
                                    "type": "left_shoulder",
                                },
                                {"x": head_idx, "y": head_price, "type": "head"},
                                {
                                    "x": right_shoulder_idx,
                                    "y": right_shoulder_price,
                                    "type": "right_shoulder",
                                },
                                {
                                    "x": left_valley_idx,
                                    "y": left_valley_price,
                                    "type": "left_valley",
                                },
                                {
                                    "x": right_valley_idx,
                                    "y": right_valley_price,
                                    "type": "right_valley",
                                },
                            ],
                            description=f"Head and Shoulders pattern with head at {head_price:.2f}",
                            metadata={"neckline_level": neckline_price},
                        )
                    )
        except Exception as e:
            logger.error("Error detecting head and shoulders: %s", e)
        return patterns

    def _detect_triangles(self, highs: pd.Series, lows: pd.Series) -> List[ChartPattern]:
        """Detect Triangle patterns"""
        patterns = []
        try:
            window = 20
            for i in range(window, len(highs) - window):
                recent_highs = highs.iloc[i - window : i + window]
                recent_lows = lows.iloc[i - window : i + window]
                high_trend = np.polyfit(range(len(recent_highs)), recent_highs, 1)
                low_trend = np.polyfit(range(len(recent_lows)), recent_lows, 1)
                if abs(high_trend[0]) > 0.01 and abs(low_trend[0]) > 0.01:
                    if high_trend[0] < 0 and low_trend[0] > 0:
                        convergence_x = (low_trend[1] - high_trend[1]) / (
                            high_trend[0] - low_trend[0]
                        )
                        if 0 < convergence_x < len(recent_highs) * 2:
                            confidence = min(0.8, 1 / (1 + abs(convergence_x - len(recent_highs))))
                            patterns.append(
                                ChartPattern(
                                    type=PatternType.TRIANGLE,
                                    confidence=confidence,
                                    start_time=highs.index[i - window],
                                    end_time=highs.index[i + window],
                                    key_points=[
                                        {
                                            "x": i - window,
                                            "y": recent_highs.iloc[0],
                                            "type": "start_high",
                                        },
                                        {
                                            "x": i + window,
                                            "y": recent_highs.iloc[-1],
                                            "type": "end_high",
                                        },
                                        {
                                            "x": i - window,
                                            "y": recent_lows.iloc[0],
                                            "type": "start_low",
                                        },
                                        {
                                            "x": i + window,
                                            "y": recent_lows.iloc[-1],
                                            "type": "end_low",
                                        },
                                    ],
                                    description="Triangle pattern with converging trend lines",
                                    metadata={"convergence_point": convergence_x},
                                )
                            )
        except Exception as e:
            logger.error("Error detecting triangles: %s", e)
        return patterns

    def _detect_flags(
        self, highs: pd.Series, lows: pd.Series, closes: pd.Series
    ) -> List[ChartPattern]:
        """Detect Flag patterns"""
        patterns = []
        try:
            window = 10
            for i in range(window * 2, len(closes) - window):
                flagpole_start = i - window * 2
                flagpole_end = i - window
                flagpole_move = (
                    closes.iloc[flagpole_end] - closes.iloc[flagpole_start]
                ) / closes.iloc[flagpole_start]
                if abs(flagpole_move) > 0.05:
                    flag_start = flagpole_end
                    flag_end = i
                    flag_high = highs.iloc[flag_start:flag_end].max()
                    flag_low = lows.iloc[flag_start:flag_end].min()
                    flag_range = (flag_high - flag_low) / flag_low
                    if flag_range < 0.03:
                        confidence = min(0.8, abs(flagpole_move) * 10)
                        patterns.append(
                            ChartPattern(
                                type=PatternType.FLAG,
                                confidence=confidence,
                                start_time=closes.index[flagpole_start],
                                end_time=closes.index[flag_end],
                                key_points=[
                                    {
                                        "x": flagpole_start,
                                        "y": closes.iloc[flagpole_start],
                                        "type": "flagpole_start",
                                    },
                                    {
                                        "x": flagpole_end,
                                        "y": closes.iloc[flagpole_end],
                                        "type": "flagpole_end",
                                    },
                                    {"x": flag_start, "y": flag_high, "type": "flag_high"},
                                    {"x": flag_end, "y": flag_low, "type": "flag_low"},
                                ],
                                description=f"Flag pattern after {flagpole_move * 100:.1f}% move",
                                metadata={"flagpole_move": flagpole_move, "flag_range": flag_range},
                            )
                        )
        except Exception as e:
            logger.error("Error detecting flags: %s", e)
        return patterns

    def create_multi_timeframe_analysis(
        self, price_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Create multi-timeframe analysis

        Args:
            price_data: Dict of timeframe -> DataFrame

        Returns:
            Dict containing multi-timeframe analysis
        """
        try:
            analysis = {
                "timestamp": datetime.now().isoformat(),
                "timeframes": {},
                "alignment": {},
                "signals": {},
            }
            for timeframe, data in price_data.items():
                if "close" not in data.columns:
                    continue
                indicators = self.calculate_technical_indicators(data, ["RSI", "MACD", "SMA"])
                trend = self._determine_trend(data, indicators)
                momentum = self._calculate_momentum(data)
                analysis["timeframes"][timeframe] = {
                    "trend": trend,
                    "momentum": momentum,
                    "indicators": {
                        "rsi": (
                            indicators.get("RSI", {}).data.iloc[-1] if "RSI" in indicators else None
                        ),
                        "macd": (
                            indicators.get("MACD", {}).data.iloc[-1].to_dict()
                            if "MACD" in indicators
                            else None
                        ),
                        "sma_alignment": (
                            self._check_sma_alignment(indicators.get("SMA", {}).data)
                            if "SMA" in indicators
                            else None
                        ),
                    },
                    "support_resistance": self._find_support_resistance(data),
                }
            analysis["alignment"] = self._check_timeframe_alignment(analysis["timeframes"])
            analysis["signals"] = self._generate_mtf_signals(
                analysis["timeframes"], analysis["alignment"]
            )
            return analysis
        except Exception as e:
            logger.error("Error in multi-timeframe analysis: %s", e)
            return {"error": str(e)}

    def _determine_trend(
        self, data: pd.DataFrame, indicators: Dict[str, TechnicalIndicator]
    ) -> str:
        """Determine trend direction"""
        try:
            if "SMA" not in indicators:
                return "neutral"
            sma_data = indicators["SMA"].data
            current_price = data["close"].iloc[-1]
            if "SMA_20" in sma_data.columns and "SMA_50" in sma_data.columns:
                sma_20 = sma_data["SMA_20"].iloc[-1]
                sma_50 = sma_data["SMA_50"].iloc[-1]
                if current_price > sma_20 > sma_50:
                    return "bullish"
                elif current_price < sma_20 < sma_50:
                    return "bearish"
            return "neutral"
        except Exception:
            return "neutral"

    def _calculate_momentum(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate momentum metrics"""
        try:
            prices = data["close"]
            return {
                "roc_5": (
                    (prices.iloc[-1] - prices.iloc[-6]) / prices.iloc[-6] if len(prices) >= 6 else 0
                ),
                "roc_10": (
                    (prices.iloc[-1] - prices.iloc[-11]) / prices.iloc[-11]
                    if len(prices) >= 11
                    else 0
                ),
                "roc_20": (
                    (prices.iloc[-1] - prices.iloc[-21]) / prices.iloc[-21]
                    if len(prices) >= 21
                    else 0
                ),
            }
        except Exception:
            return {"roc_5": 0, "roc_10": 0, "roc_20": 0}

    def _check_sma_alignment(self, sma_data: pd.DataFrame) -> str:
        """Check SMA alignment"""
        try:
            if len(sma_data.columns) < 2:
                return "insufficient_data"
            latest = sma_data.iloc[-1]
            sma_values = latest.sort_values(ascending=False)
            columns = ["SMA_20", "SMA_50", "SMA_200"]
            available_columns = [col for col in columns if col in sma_data.columns]
            if len(available_columns) >= 2:
                bullish = all(
                    (
                        latest[available_columns[i]] > latest[available_columns[i + 1]]
                        for i in range(len(available_columns) - 1)
                    )
                )
                bearish = all(
                    (
                        latest[available_columns[i]] < latest[available_columns[i + 1]]
                        for i in range(len(available_columns) - 1)
                    )
                )
                if bullish:
                    return "bullish"
                elif bearish:
                    return "bearish"
            return "mixed"
        except Exception:
            return "unknown"

    def _find_support_resistance(self, data: pd.DataFrame) -> Dict[str, List[float]]:
        """Find support and resistance levels"""
        try:
            highs = data["high"]
            lows = data["low"]
            window = 10
            resistance_levels = []
            support_levels = []
            for i in range(window, len(highs) - window):
                if highs.iloc[i] == highs.iloc[i - window : i + window + 1].max():
                    resistance_levels.append(highs.iloc[i])
                if lows.iloc[i] == lows.iloc[i - window : i + window + 1].min():
                    support_levels.append(lows.iloc[i])
            resistance_levels = sorted(list(set(resistance_levels)), reverse=True)[:5]
            support_levels = sorted(list(set(support_levels)))[:5]
            return {"resistance": resistance_levels, "support": support_levels}
        except Exception:
            return {"resistance": [], "support": []}

    def _check_timeframe_alignment(self, timeframes: Dict[str, Any]) -> Dict[str, Any]:
        """Check alignment across timeframes"""
        try:
            trends = [tf_data["trend"] for tf_data in timeframes.values()]
            bullish_count = trends.count("bullish")
            bearish_count = trends.count("bearish")
            neutral_count = trends.count("neutral")
            total_timeframes = len(trends)
            if bullish_count / total_timeframes >= 0.7:
                overall_alignment = "strongly_bullish"
            elif bearish_count / total_timeframes >= 0.7:
                overall_alignment = "strongly_bearish"
            elif bullish_count > bearish_count:
                overall_alignment = "moderately_bullish"
            elif bearish_count > bullish_count:
                overall_alignment = "moderately_bearish"
            else:
                overall_alignment = "mixed"
            return {
                "overall_alignment": overall_alignment,
                "bullish_timeframes": bullish_count,
                "bearish_timeframes": bearish_count,
                "neutral_timeframes": neutral_count,
                "alignment_strength": max(bullish_count, bearish_count) / total_timeframes,
            }
        except Exception:
            return {"overall_alignment": "unknown", "alignment_strength": 0}

    def _generate_mtf_signals(
        self, timeframes: Dict[str, Any], alignment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate multi-timeframe trading signals"""
        try:
            signals = {
                "primary_signal": "neutral",
                "confidence": 0,
                "supporting_factors": [],
                "conflicting_factors": [],
            }
            overall_alignment = alignment.get("overall_alignment", "mixed")
            alignment_strength = alignment.get("alignment_strength", 0)
            if "strongly_bullish" in overall_alignment:
                signals["primary_signal"] = "buy"
                signals["confidence"] = min(0.9, alignment_strength)
            elif "strongly_bearish" in overall_alignment:
                signals["primary_signal"] = "sell"
                signals["confidence"] = min(0.9, alignment_strength)
            elif "moderately_bullish" in overall_alignment:
                signals["primary_signal"] = "buy"
                signals["confidence"] = min(0.6, alignment_strength)
            elif "moderately_bearish" in overall_alignment:
                signals["primary_signal"] = "sell"
                signals["confidence"] = min(0.6, alignment_strength)
            for tf_name, tf_data in timeframes.items():
                trend = tf_data.get("trend", "neutral")
                if trend == "bullish" and signals["primary_signal"] == "buy":
                    signals["supporting_factors"].append(f"{tf_name} bullish trend")
                elif trend == "bearish" and signals["primary_signal"] == "sell":
                    signals["supporting_factors"].append(f"{tf_name} bearish trend")
                elif trend != "neutral" and trend != signals["primary_signal"].replace(
                    "buy", "bullish"
                ).replace("sell", "bearish"):
                    signals["conflicting_factors"].append(f"{tf_name} {trend} trend")
            return signals
        except Exception:
            return {"primary_signal": "neutral", "confidence": 0}

    def add_drawing_tool(self, chart_id: str, drawing: DrawingTool) -> bool:
        """Add a drawing tool to a chart"""
        try:
            if chart_id not in self.drawings:
                self.drawings[chart_id] = []
            self.drawings[chart_id].append(drawing)
            return True
        except Exception as e:
            logger.error("Error adding drawing tool: %s", e)
            return False

    def get_chart_drawings(self, chart_id: str) -> List[DrawingTool]:
        """Get all drawings for a chart"""
        return self.drawings.get(chart_id, [])

    def add_annotation(self, chart_id: str, annotation: Dict[str, Any]) -> bool:
        """Add an annotation to a chart"""
        try:
            if chart_id not in self.annotations:
                self.annotations[chart_id] = []
            annotation["id"] = f"annotation_{len(self.annotations[chart_id])}"
            annotation["created_at"] = datetime.now().isoformat()
            self.annotations[chart_id].append(annotation)
            return True
        except Exception as e:
            logger.error("Error adding annotation: %s", e)
            return False

    def get_chart_annotations(self, chart_id: str) -> List[Dict[str, Any]]:
        """Get all annotations for a chart"""
        return self.annotations.get(chart_id, [])
