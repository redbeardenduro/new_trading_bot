"""
Common utilities for sentiment analysis and tracking.

Includes:
- SentimentTracker: Tracks sentiment history for assets.
- SentimentAnalyzer: Analyzes text sentiment using specified HuggingFace
  transformer models (e.g., FinBERT, RoBERTa) or VADER fallback.
- Helper functions: For preprocessing text, calculating weighted sentiment,
  adjusting by volume, getting statistics, and normalizing scores.
"""

import math
import statistics
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

from common.common_logger import get_logger

logger = get_logger("sentiment_utils")
hf_transformers_available = False
vader_available = False
(pipeline, AutoTokenizer, AutoModelForSequenceClassification, Accelerator, torch) = (
    None,
    None,
    None,
    None,
    None,
)
fallback_analyzer = None
try:
    import torch
    from accelerate import Accelerator
    from transformers import (AutoModelForSequenceClassification,
                              AutoTokenizer, pipeline)

    hf_transformers_available = True
    logger.debug("HuggingFace Transformers and PyTorch loaded successfully.")
except ImportError:
    logger.warning(
        "HuggingFace 'transformers', 'accelerate', or 'torch' library not found. Transformer models unavailable."
    )
try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    try:
        fallback_analyzer = SentimentIntensityAnalyzer()
        vader_available = True
        logger.debug("NLTK and VADER lexicon loaded successfully.")
    except LookupError:
        logger.warning(
            "NLTK VADER lexicon not found. VADER fallback unavailable. Run: python -m nltk.downloader vader_lexicon"
        )
        vader_available = False
except ImportError:
    logger.warning("NLTK library not found. VADER fallback disabled. Run: pip install nltk")
    vader_available = False


class SentimentTracker:
    """Tracks sentiment history for assets."""

    def __init__(self, history_length: int = 30) -> None:
        """
        Initializes the SentimentTracker.

        Args:
            history_length (int): Max number of sentiment entries to keep per symbol.
        """
        self.history: Dict[str, List[Tuple[datetime, float, float]]] = {}
        try:
            self.history_length = max(1, int(history_length))
        except (ValueError, TypeError):
            logger.warning("Invalid history_length '%s'. Using default 30.", history_length)
            self.history_length = 30
        logger.info("SentimentTracker initialized with history length: %s", self.history_length)

    def add_sentiment(self, symbol: str, sentiment_value: float, confidence: float = 1.0) -> None:
        """Adds a sentiment data point for a given symbol, validating inputs."""
        if not isinstance(symbol, str) or not symbol.strip():
            logger.warning("Invalid symbol provided for sentiment tracking. Skipping.")
            return
        symbol_upper = symbol.strip().upper()
        try:
            sentiment_value_f = max(-1.0, min(1.0, float(sentiment_value)))
            confidence_f = max(0.0, min(1.0, float(confidence)))
        except (ValueError, TypeError):
            logger.warning(
                "Invalid sentiment (%s) or confidence (%s) for %s. Skipping add.",
                sentiment_value,
                confidence,
                symbol_upper,
            )
            return
        timestamp = datetime.now(timezone.utc)
        if symbol_upper not in self.history:
            self.history[symbol_upper] = []
        self.history[symbol_upper].append((timestamp, sentiment_value_f, confidence_f))
        if len(self.history[symbol_upper]) > self.history_length:
            self.history[symbol_upper] = self.history[symbol_upper][-self.history_length :]

    def get_sentiment_trend(self, symbol: str, window: int = 5) -> float:
        """Calculates the sentiment trend over a recent window."""
        if not isinstance(symbol, str) or not symbol.strip():
            return 0.0
        symbol_upper = symbol.strip().upper()
        try:
            window_val = max(2, int(window))
        except (ValueError, TypeError):
            window_val = 5
        history_data = self.history.get(symbol_upper)
        if not history_data or len(history_data) < 2:
            return 0.0
        history_len = len(history_data)
        num_recent = min(window_val, history_len)
        if num_recent < 2:
            return 0.0
        recent_history = history_data[-num_recent:]
        try:
            oldest_sentiment = recent_history[0][1]
            newest_sentiment = recent_history[-1][1]
            return newest_sentiment - oldest_sentiment
        except IndexError:
            logger.error(
                "IndexError during trend calculation for %s (window=%s)",
                symbol_upper,
                num_recent,
                exc_info=True,
            )
            return 0.0

    def get_statistics(self, symbol: str, window: Optional[int] = None) -> Dict[str, float]:
        """Calculates sentiment statistics over a specified window or full history."""
        default_stats = {
            "mean": 0.0,
            "median": 0.0,
            "stdev": 0.0,
            "min": 0.0,
            "max": 0.0,
            "count": 0.0,
            "trend": 0.0,
        }
        if not isinstance(symbol, str) or not symbol.strip():
            return default_stats
        symbol_upper = symbol.strip().upper()
        history_data = self.history.get(symbol_upper)
        if not history_data:
            return default_stats
        history_len = len(history_data)
        stats_window_size = history_len
        if window is not None:
            try:
                window_int = int(window)
                if window_int > 0:
                    stats_window_size = min(window_int, history_len)
            except (ValueError, TypeError):
                pass
        data_for_stats = history_data[-stats_window_size:]
        values = [item[1] for item in data_for_stats]
        if not values:
            return default_stats
        trend = self.get_sentiment_trend(symbol_upper, window=window if window is not None else 5)
        try:
            stats = {
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
                "count": float(len(values)),
                "trend": trend,
            }
            return stats
        except statistics.StatisticsError as e:
            logger.error("Error calculating statistics for %s: %s", symbol_upper, e)
            default_stats["count"] = float(len(values))
            default_stats["trend"] = trend
            return default_stats
        except Exception as e:
            logger.error(
                "Unexpected error calculating statistics for %s: %s", symbol_upper, e, exc_info=True
            )
            default_stats["count"] = float(len(values))
            default_stats["trend"] = trend
            return default_stats


class SentimentAnalyzer:
    """
    Analyzes text sentiment using a specified HuggingFace model or VADER fallback.
    Handles model loading errors gracefully.
    """

    SUPPORTED_MODELS = {
        "finbert": "yiyanghkust/finbert-tone",
        "roberta": "cardiffnlp/twitter-roberta-base-sentiment",
        "vader": "vader",
    }

    def __init__(self, model_type: str = "roberta") -> None:
        """
        Initializes the SentimentAnalyzer, attempting to load the specified model.

        Args:
            model_type (str): Type of model to load ('finbert', 'roberta', or 'vader').
                              Defaults to 'roberta'.
        """
        self.model_type = model_type.lower() if isinstance(model_type, str) else "roberta"
        self.pipeline = None
        self.vader_analyzer = fallback_analyzer
        self.use_hf = False
        self.hf_model_name: Optional[str] = None
        if self.model_type not in self.SUPPORTED_MODELS:
            logger.error(
                "Unsupported sentiment model type '%s'. Supported: %s. Falling back to VADER if available.",
                self.model_type,
                list(self.SUPPORTED_MODELS.keys()),
            )
            self.model_type = "vader"
        if self.model_type != "vader":
            if not hf_transformers_available:
                logger.warning(
                    "Cannot load HF model '%s': Required libraries (transformers, accelerate, torch) not available. Falling back to VADER.",
                    self.model_type,
                )
                self.model_type = "vader"
            else:
                self.hf_model_name = self.SUPPORTED_MODELS[self.model_type]
                try:
                    device = -1
                    if torch and torch.cuda.is_available():
                        device = 0
                        logger.info("CUDA available, attempting to use GPU for sentiment model.")
                    tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name)
                    model = AutoModelForSequenceClassification.from_pretrained(self.hf_model_name)
                    reported_max_length = getattr(tokenizer, "model_max_length", 512)
                    if not isinstance(reported_max_length, int) or reported_max_length > 1024:
                        safe_max_length = 512
                        logger.warning(
                            "Tokenizer reported max_length %s, which is excessive. Using safe default: %s",
                            reported_max_length,
                            safe_max_length,
                        )
                    else:
                        safe_max_length = reported_max_length
                        logger.debug("Using tokenizer reported max_length: %s", safe_max_length)
                    self.pipeline = pipeline(  # type: ignore[misc]
                        "sentiment-analysis",
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        truncation=True,
                        max_length=safe_max_length,
                    )
                    self.use_hf = True
                    logger.info(
                        "Initialized HF '%s' pipeline (%s) successfully on device %s with max_length %s.",
                        self.model_type,
                        self.hf_model_name,
                        device,
                        safe_max_length,
                    )
                except OSError as e:
                    logger.error(
                        "OSError loading HF model '%s': %s. Check model name and internet connection. Falling back to VADER.",
                        self.hf_model_name,
                        e,
                    )
                    self.use_hf = False
                    self.pipeline = None
                    self.model_type = "vader"
                except Exception as e:
                    logger.error(
                        "Failed to initialize HF '%s' pipeline (%s): %s",
                        self.model_type,
                        self.hf_model_name,
                        e,
                        exc_info=True,
                    )
                    self.use_hf = False
                    self.pipeline = None
                    logger.warning(
                        "Falling back to VADER for '%s' due to HF init error.", self.model_type
                    )
                    self.model_type = "vader"
        if self.model_type == "vader":
            if not self.vader_analyzer:
                logger.error(
                    "VADER model selected or fallback required, but VADER analyzer is not available (check NLTK lexicon). Sentiment analysis disabled."
                )
            else:
                logger.info("SentimentAnalyzer initialized using VADER.")
        elif not self.use_hf:
            logger.error(
                "Failed to load HF model '%s' and VADER is unavailable. Sentiment analysis disabled.",
                self.model_type,
            )

    def analyze(self, text: str) -> Dict[str, float]:
        """
        Analyzes the sentiment of the provided text.

        Args:
            text (str): The text to analyze.

        Returns:
            Dict[str, float]: Dictionary containing 'score' and 'confidence'.
                              Returns {'score': 0.0, 'confidence': 0.0} on error or if no analyzer available.
        """
        default_result = {"score": 0.0, "confidence": 0.0}
        if not isinstance(text, str) or not text.strip():
            return default_result
        if self.use_hf and self.pipeline:
            try:
                result_list = self.pipeline(text)
                if (
                    not result_list
                    or not isinstance(result_list, list)
                    or (not isinstance(result_list[0], dict))
                ):
                    logger.warning(
                        "HF pipeline returned unexpected result format for text: '%s...'", text[:50]
                    )
                    raise ValueError("Invalid result format from HF pipeline")
                result = result_list[0]
                label = str(result.get("label", "")).upper()
                hf_confidence = float(result.get("score", 0.0))
                score = 0.0
                if self.model_type == "roberta":
                    score_map = {"LABEL_2": 1.0, "LABEL_1": 0.0, "LABEL_0": -1.0}
                    score = score_map.get(label, 0.0)
                elif self.model_type == "finbert":
                    score_map = {"POSITIVE": 1.0, "NEGATIVE": -1.0, "NEUTRAL": 0.0}
                    score = score_map.get(label, 0.0)
                return {"score": score, "confidence": hf_confidence}
            except Exception as e:
                logger.error(
                    "HF '%s' analysis runtime error: %s", self.model_type, e, exc_info=True
                )
        if self.vader_analyzer:
            try:
                vader_scores = self.vader_analyzer.polarity_scores(text)
                compound_score = vader_scores.get("compound", 0.0)
                if compound_score >= 0.05:
                    score = 1.0
                elif compound_score <= -0.05:
                    score = -1.0
                else:
                    score = 0.0
                confidence = (
                    min(1.0, abs(compound_score) * 1.5)
                    if score != 0
                    else max(0.1, 1.0 - abs(compound_score) * 5)
                )
                logger.debug(
                    "Using VADER fallback. Score=%.1f (Compound=%.3f), Est.Conf=%.3f",
                    score,
                    compound_score,
                    confidence,
                )
                return {"score": score, "confidence": confidence}
            except Exception as e:
                logger.error("VADER analysis error: %s", e, exc_info=True)
                return default_result
        else:
            if not (self.use_hf and self.pipeline):
                logger.error(
                    "No sentiment analyzers available (HF failed/disabled and VADER unavailable)."
                )
            return default_result


def calculate_time_weighted_sentiment(
    items: List[Dict], timestamp_key: str = "created_utc"
) -> float:
    """Calculates time-weighted average sentiment from a list of items."""
    weighted_sum = 0.0
    weight_sum = 0.0
    now_utc = datetime.now(timezone.utc)
    decay_factor = -0.029
    if not isinstance(items, list):
        return 0.0
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            sentiment = float(item.get("sentiment", 0.0))
            confidence = float(item.get("confidence", 1.0))
            timestamp_val = item.get(timestamp_key)
            timestamp = None
            if isinstance(timestamp_val, (int, float)):
                try:
                    if timestamp_val > 1000000000000.0:
                        timestamp = datetime.fromtimestamp(timestamp_val / 1000, timezone.utc)
                    else:
                        timestamp = datetime.fromtimestamp(timestamp_val, timezone.utc)
                except (OSError, ValueError):
                    logger.debug("Invalid numeric timestamp: %s", timestamp_val)
                    timestamp = now_utc
            elif isinstance(timestamp_val, str):
                try:
                    timestamp = datetime.fromisoformat(
                        timestamp_val.replace("Z", "+00:00")
                    ).astimezone(timezone.utc)
                except ValueError:
                    logger.debug("Could not parse timestamp string: %s", timestamp_val)
                    timestamp = now_utc
            else:
                timestamp = now_utc
            age_seconds = (now_utc - timestamp).total_seconds()
            age_hours = max(0.0, age_seconds / 3600.0)
            time_weight = math.exp(decay_factor * age_hours)
            adjusted_weight = time_weight * max(0.0, min(1.0, confidence))
            weighted_sum += sentiment * adjusted_weight
            weight_sum += adjusted_weight
        except (ValueError, TypeError) as e:
            item_id = item.get("id", item.get("url", "N/A"))
            logger.warning("Skipping item '%s' due to calculation error: %s", item_id, e)
            continue
        except Exception as e:
            item_id = item.get("id", item.get("url", "N/A"))
            logger.error("Unexpected error weighting item '%s': %s", item_id, e, exc_info=True)
            continue
    if weight_sum < 1e-09:
        return 0.0
    else:
        return max(-1.0, min(1.0, weighted_sum / weight_sum))


def adjust_sentiment_by_volume(
    sentiment_value: float, item_count: int, baseline_count: int = 20
) -> float:
    """Adjusts sentiment score based on the volume (item count) relative to a baseline."""
    try:
        sentiment_value_f = float(sentiment_value)
        item_count_i = int(item_count)
        baseline_count_i = int(baseline_count)
    except (ValueError, TypeError):
        logger.warning(
            "Invalid input for volume adjustment: sentiment=%s, count=%s, baseline=%s",
            sentiment_value,
            item_count,
            baseline_count,
        )
        return 0.0
    if item_count_i <= 0 or baseline_count_i <= 0:
        return 0.0
    try:
        volume_factor = min(2.0, math.sqrt(item_count_i / baseline_count_i))
    except ZeroDivisionError:
        volume_factor = 1.0
    result = sentiment_value_f * volume_factor
    return max(-1.0, min(1.0, result))


def get_sentiment_statistics(items: List[Dict]) -> Dict[str, float]:
    """Calculates basic statistics for sentiment scores in a list of items."""
    default_stats = {"mean": 0.0, "median": 0.0, "stdev": 0.0, "min": 0.0, "max": 0.0, "count": 0.0}
    if not isinstance(items, list) or not items:
        return default_stats
    sentiments: list = []
    for item in items:
        if isinstance(item, dict):
            sentiment_val = item.get("sentiment")
            if isinstance(sentiment_val, (int, float)):
                sentiments.append(sentiment_val)
    count = len(sentiments)
    if count == 0:
        return default_stats
    try:
        stats = {
            "mean": statistics.mean(sentiments),
            "median": statistics.median(sentiments),
            "stdev": statistics.stdev(sentiments) if count > 1 else 0.0,
            "min": min(sentiments),
            "max": max(sentiments),
            "count": float(count),
        }
        return stats
    except statistics.StatisticsError as e:
        logger.error("Error calculating sentiment statistics: %s", e)
        default_stats["count"] = float(count)
        return default_stats
    except Exception as e:
        logger.error("Unexpected error calculating sentiment statistics: %s", e, exc_info=True)
        default_stats["count"] = float(count)
        return default_stats


def normalize_sentiment_score(
    score: float, min_score: float = -1.0, max_score: float = 1.0
) -> float:
    """Normalizes or scales a sentiment score, e.g., applying power scaling."""
    try:
        score_f = float(score)
        clamped_score = max(min_score, min(max_score, score_f))
        scale_power = 0.8
        scaled_score = math.copysign(math.pow(abs(clamped_score), scale_power), clamped_score)
        return scaled_score
    except (ValueError, TypeError):
        logger.warning("Invalid score '%s' for normalization. Returning 0.0.", score)
        return 0.0


def preprocess_text(text: str) -> str:
    """Basic text preprocessing (lowercase, remove extra whitespace)."""
    if not isinstance(text, str):
        return ""
    return " ".join(text.lower().split())


def normalize_sentiment_value(sentiment: Union[str, int, float]) -> float:
    """Converts sentiment labels or scores to a numeric value [-1, 1]."""
    default_value = 0.0
    if isinstance(sentiment, str):
        sentiment_lower = sentiment.lower().strip()
        if "bullish" in sentiment_lower:
            return 0.7
        if "bearish" in sentiment_lower:
            return -0.7
        if "positive" in sentiment_lower:
            return 0.7
        if "negative" in sentiment_lower:
            return -0.7
        if "neutral" in sentiment_lower:
            return 0.0
        logger.debug("Unknown sentiment string label '%s'. Returning %s.", sentiment, default_value)
        return default_value
    elif isinstance(sentiment, (int, float)):
        try:
            return max(-1.0, min(1.0, float(sentiment)))
        except (ValueError, TypeError):
            logger.warning(
                "Invalid numeric sentiment value '%s'. Returning %s.", sentiment, default_value
            )
            return default_value
    else:
        logger.warning(
            "Unsupported sentiment type '%s'. Returning %s.", type(sentiment), default_value
        )
        return default_value


def convert_recommendation_to_sentiment(recommendation: Optional[str]) -> float:
    """Converts buy/sell/hold recommendations to a numeric sentiment value."""
    default_value = 0.0
    if isinstance(recommendation, str):
        rec_lower = recommendation.lower().strip()
        if "strong buy" in rec_lower:
            return 0.9
        if "buy" in rec_lower:
            return 0.7
        if "strong sell" in rec_lower:
            return -0.9
        if "sell" in rec_lower:
            return -0.7
        if "hold" in rec_lower:
            return 0.0
        logger.debug(
            "Unknown recommendation string '%s'. Returning %s.", recommendation, default_value
        )
        return default_value
    else:
        return default_value


def extract_confidence(confidence_value: Any) -> float:
    """Extracts and normalizes confidence score to [0, 1] range."""
    default_confidence = 0.5
    if isinstance(confidence_value, (int, float)):
        try:
            val = float(confidence_value)
            if val > 1.0 and val <= 10.0:
                return max(0.0, min(1.0, val / 10.0))
            elif val >= 0.0 and val <= 1.0:
                return max(0.0, min(1.0, val))
            else:
                logger.warning(
                    "Numeric confidence value '%s' out of expected range [0, 1] or [1, 10]. Using default.",
                    val,
                )
                return default_confidence
        except (ValueError, TypeError):
            logger.warning(
                "Could not convert numeric confidence '%s' to float. Using default.",
                confidence_value,
            )
            return default_confidence
    elif isinstance(confidence_value, str):
        cl = confidence_value.lower().strip()
        if "high" in cl:
            return 0.8
        if "medium" in cl:
            return 0.5
        if "low" in cl:
            return 0.3
        try:
            cf = float(confidence_value)
            return extract_confidence(cf)
        except (ValueError, TypeError):
            logger.warning("Unknown string confidence label '%s'. Using default.", confidence_value)
            return default_confidence
    else:
        return default_confidence


if __name__ == "__main__":
    print("--- Testing Sentiment Utils ---")
    tracker = SentimentTracker(history_length=5)
    tracker.add_sentiment("BTC", 0.5)
    tracker.add_sentiment("BTC", 0.7, 0.9)
    time.sleep(0.1)
    tracker.add_sentiment("BTC", 0.6)
    tracker.add_sentiment("ETH", -0.3)
    tracker.add_sentiment("BTC", 0.8)
    tracker.add_sentiment("BTC", 0.75)
    tracker.add_sentiment("BTC", 0.7)
    print("Tracker History (BTC):", tracker.history.get("BTC"))
    print("Tracker History (ETH):", tracker.history.get("ETH"))
    print("BTC Stats (Full):", tracker.get_statistics("BTC"))
    print("BTC Stats (Window 3):", tracker.get_statistics("BTC", window=3))
    print("BTC Trend (Window 3):", tracker.get_sentiment_trend("BTC", window=3))
    print("ETH Trend (Window 5):", tracker.get_sentiment_trend("ETH", window=5))
    print("\n--- Testing Analyzer ---")
    text_pos = "This is great news, very optimistic outlook!"
    text_neg = "Terrible decline, market is crashing hard."
    text_neu = "The market opened today."
    print("\nTesting RoBERTa Analyzer:")
    try:
        roberta_analyzer = SentimentAnalyzer(model_type="roberta")
        if roberta_analyzer.use_hf or roberta_analyzer.vader_analyzer:
            print(f"'{text_pos}' -> {roberta_analyzer.analyze(text_pos)}")
            print(f"'{text_neg}' -> {roberta_analyzer.analyze(text_neg)}")
            print(f"'{text_neu}' -> {roberta_analyzer.analyze(text_neu)}")
        else:
            print("RoBERTa Analyzer disabled (model load failed/VADER unavailable).")
    except Exception as e:
        print(f"Error initializing/using RoBERTa: {e}")
    print("\nTesting FinBERT Analyzer:")
    try:
        finbert_analyzer = SentimentAnalyzer(model_type="finbert")
        if finbert_analyzer.use_hf or finbert_analyzer.vader_analyzer:
            print(f"'{text_pos}' -> {finbert_analyzer.analyze(text_pos)}")
            print(f"'{text_neg}' -> {finbert_analyzer.analyze(text_neg)}")
            print(f"'{text_neu}' -> {finbert_analyzer.analyze(text_neu)}")
        else:
            print("FinBERT Analyzer disabled (model load failed/VADER unavailable).")
    except Exception as e:
        print(f"Error initializing/using FinBERT: {e}")
    print("\nTesting VADER Analyzer (as fallback or direct):")
    try:
        vader_analyzer = SentimentAnalyzer(model_type="vader")
        if vader_analyzer.vader_analyzer:
            print(f"'{text_pos}' -> {vader_analyzer.analyze(text_pos)}")
            print(f"'{text_neg}' -> {vader_analyzer.analyze(text_neg)}")
            print(f"'{text_neu}' -> {vader_analyzer.analyze(text_neu)}")
        else:
            print("VADER Analyzer disabled (NLTK/lexicon not available).")
    except Exception as e:
        print(f"Error initializing/using VADER: {e}")
    print("\n--- Testing Helpers ---")
    sample_items = [
        {
            "sentiment": 0.8,
            "confidence": 0.9,
            "created_utc": (datetime.now(timezone.utc) - timedelta(hours=1)).timestamp(),
        },
        {
            "sentiment": 0.5,
            "confidence": 0.7,
            "created_utc": (datetime.now(timezone.utc) - timedelta(hours=10)).timestamp(),
        },
        {
            "sentiment": -0.4,
            "confidence": 0.8,
            "created_utc": (datetime.now(timezone.utc) - timedelta(hours=48)).timestamp(),
        },
        {
            "sentiment": 0.1,
            "confidence": 0.5,
            "published_at": (datetime.now(timezone.utc) - timedelta(hours=72)).isoformat(),
        },
        {
            "sentiment": "invalid",
            "confidence": 0.6,
            "created_utc": datetime.now(timezone.utc).timestamp(),
        },
        {
            "sentiment": 0.9,
            "confidence": "high",
            "created_utc": datetime.now(timezone.utc).timestamp(),
        },
        {},
        None,
    ]
    print("Sample Items for Helpers:", sample_items)
    print(
        "Time Weighted Sentiment (UTC):",
        calculate_time_weighted_sentiment(sample_items, timestamp_key="created_utc"),
    )
    print(
        "Time Weighted Sentiment (PublishedAt):",
        calculate_time_weighted_sentiment(sample_items, timestamp_key="published_at"),
    )
    print("Volume Adjusted (0.5, 40 items, baseline 20):", adjust_sentiment_by_volume(0.5, 40, 20))
    print("Volume Adjusted (0.5, 5 items, baseline 20):", adjust_sentiment_by_volume(0.5, 5, 20))
    print("Sentiment Stats:", get_sentiment_statistics(sample_items))
    print("Normalize Score (0.8):", normalize_sentiment_score(0.8))
    print("Normalize Score (-0.3):", normalize_sentiment_score(-0.3))
    print("Normalize OpenAI Sentiment (Bullish):", normalize_sentiment_value("Bullish"))
    print("Normalize OpenAI Sentiment (Invalid):", normalize_sentiment_value(None))
    print(
        "Normalize OpenAI Rec (Sell):", convert_recommendation_to_sentiment("Sell signal detected")
    )
    print("Normalize OpenAI Rec (None):", convert_recommendation_to_sentiment(None))
    print("Extract Confidence (8):", extract_confidence(8))
    print("Extract Confidence ('Low'):", extract_confidence("Low"))
    print("Extract Confidence (None):", extract_confidence(None))
    print("\n--- Sentiment Utils Test Complete ---")
