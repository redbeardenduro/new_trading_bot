"""
Core module for the Multi-Crypto Trading Bot.

Contains the main bot logic, including market data handling, signal generation,
trade execution, portfolio management integration, and the main trading loop orchestration.
Relies on a central configuration and injected dependencies for external services.
Decoupled from PortfolioManager; passes necessary market state during rebalance checks.
"""

import argparse
import asyncio
import concurrent.futures
import json
import time
from datetime import datetime, timezone
from decimal import Decimal, DivisionByZero, InvalidOperation
from functools import wraps
from typing import Any, Dict, List, Optional

import ccxt
import numpy as np
import prawcore
import requests

from common.common_logger import (CACHE_DIR, METRICS_DIR, SKIPPED_TRADES_DIR,
                                  get_logger)
from core.alerting import Alerter
from core.circuit_breaker import CircuitBreaker
from core.config import BotConfig
from core.interfaces import (IAIAnalyzer, IExchangeClient, IPortfolioManager,
                             ISentimentSource)
from core.order_manager import OrderManager, OrderStatus
from core.safety import SafetyGuard

logger = get_logger("core_bot")


def timed_operation(func) -> None:
    """Decorator to log timing information for operations."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs) -> None:
        start_time = time.time()
        operation = func.__name__
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.debug("Operation %s completed in %.3fs", operation, elapsed)
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.exception(f"Operation {operation} failed after {elapsed:.3f}s: {e}")
            raise

    return wrapper


def calculate_sma(data: np.ndarray, window: int) -> np.ndarray:
    """Calculates Simple Moving Average."""
    if len(data) < window:
        logger.debug("SMA calculation skipped: data length %s < window %s", len(data), window)
        return np.full(len(data), np.nan)
    try:
        data_float = np.asarray(data, dtype=np.float64)
        if np.isnan(data_float).all():
            return np.full(len(data), np.nan)
        sma_valid = np.convolve(data_float, np.ones(window), "valid") / window
        nan_padding = np.full(window - 1, np.nan)
        return np.concatenate((nan_padding, sma_valid))
    except (ValueError, TypeError) as e:
        logger.error("Error calculating SMA: %s", e, exc_info=True)
        return np.full(len(data), np.nan)


def calculate_ema(data: np.ndarray, window: int) -> np.ndarray:
    """Calculates Exponential Moving Average more robustly."""
    if len(data) < window or window <= 0:
        logger.debug("EMA calculation skipped: data length %s < window %s", len(data), window)
        return np.full(len(data), np.nan)
    try:
        data_float = np.asarray(data, dtype=np.float64)
        alpha = 2.0 / (window + 1.0)
        ema = np.full(len(data_float), np.nan)
        first_valid_index = -1
        for i in range(window - 1, len(data_float)):
            if not np.isnan(data_float[i - window + 1 : i + 1]).any():
                initial_sma = np.mean(data_float[i - window + 1 : i + 1])
                if not np.isnan(initial_sma):
                    ema[i] = initial_sma
                    first_valid_index = i
                    break
        if first_valid_index == -1:
            logger.warning(
                "Could not find initial valid window for EMA (window=%s). Returning NaNs.", window
            )
            return ema
        for i in range(first_valid_index + 1, len(data_float)):
            current_val = data_float[i]
            prev_ema = ema[i - 1]
            if np.isnan(current_val):
                ema[i] = prev_ema
            elif np.isnan(prev_ema):
                if i >= window - 1:
                    window_data = data_float[i - window + 1 : i + 1]
                    if not np.isnan(window_data).any():
                        sma_restart = np.mean(window_data)
                        if not np.isnan(sma_restart):
                            ema[i] = sma_restart
                        else:
                            ema[i] = np.nan
                    else:
                        ema[i] = np.nan
                else:
                    ema[i] = np.nan
            else:
                ema[i] = alpha * current_val + (1 - alpha) * prev_ema
        return ema
    except (ValueError, TypeError, IndexError) as e:
        logger.error("Error calculating EMA: %s", e, exc_info=True)
        return np.full(len(data), np.nan)


class MultiCryptoTradingBot:
    """
    Orchestrates the cryptocurrency trading process using centralized configuration
    and injected dependencies for external services. Passes market state to PortfolioManager.
    """

    def __init__(
        self,
        config: BotConfig,
        exchange_client: IExchangeClient,
        portfolio_manager: IPortfolioManager,
        sentiment_sources: List[ISentimentSource],
        ai_analyzer: Optional[IAIAnalyzer] = None,
    ) -> None:
        """
        Initializes the MultiCryptoTradingBot with configuration and injected dependencies.

        Args:
            config (BotConfig): The central configuration object.
            exchange_client (IExchangeClient): Client for exchange interactions.
            portfolio_manager (IPortfolioManager): Component for portfolio management.
            sentiment_sources (List[ISentimentSource]): List of clients for sentiment data.
            ai_analyzer (Optional[IAIAnalyzer]): Client for AI analysis (optional).
        """
        if config is None:
            raise ValueError("BotConfig cannot be None.")
        if exchange_client is None:
            raise ValueError("IExchangeClient cannot be None.")
        if portfolio_manager is None:
            raise ValueError("IPortfolioManager cannot be None.")
        self.config = config
        self.exchange_client = exchange_client
        self.portfolio_manager = portfolio_manager
        self.sentiment_sources = sentiment_sources if sentiment_sources is not None else []
        self.ai_analyzer = ai_analyzer
        self.safety_guard = SafetyGuard(config)
        self.order_manager = OrderManager(exchange_client)
        self.exchange_circuit_breaker = CircuitBreaker(name="exchange")
        self.alerter = Alerter(config)
        self.base_currencies = config.get("bot.base_currencies", [])
        self.quote_currencies = config.get("bot.quote_currencies", ["USD"])
        self.timeframe = config.get("bot.timeframe", "1h")
        self.paper_trading = config.get("bot.paper_trading", True)
        self.strategy = config.get("bot.strategy", "combined")
        self.disable_twitter = config.get("bot.disable_twitter", True)
        max_concurrent = config.get("bot.max_concurrent_pairs", 1)
        if not self.base_currencies or not self.quote_currencies:
            logger.warning(
                "Base or Quote currencies are empty in config. Trading pairs might be empty."
            )
            self.trading_pairs = []
        else:
            self.trading_pairs = [
                f"{base}/{quote}"
                for base in self.base_currencies
                for quote in self.quote_currencies
            ]
        self.market_data: Dict[str, Dict[str, Any]] = {}
        self.opportunity_scores: Dict[str, float] = {}
        skipped_dir = SKIPPED_TRADES_DIR
        skipped_dir.mkdir(parents=True, exist_ok=True)
        self.bot_skipped_trades_log_file = skipped_dir / "bot_skipped_trades.json"
        self.news_cache_file = CACHE_DIR / "news_cache.json"
        self._initialize_news_cache()
        self.performance_metrics: Dict[str, Any] = {
            "trades_executed": 0,
            "successful_trades": 0,
            "failed_trades": 0,
            "buy_trades": 0,
            "sell_trades": 0,
            "total_profit_loss": 0.0,
            "start_portfolio_value": 0.0,
            "current_portfolio_value": 0.0,
            "profit_loss_pct": 0.0,
        }
        self.metrics_dir = METRICS_DIR
        self.exchange_available = self.exchange_client is not None
        self.sentiment_available = bool(self.sentiment_sources)
        self.ai_available = self.ai_analyzer is not None
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, max_concurrent))
        self.initialize_market_data()
        self.running = False
        self.total_portfolio_value: float = 0.0
        logger.info("Trading bot initialized with config: %s", self.config.config_file_path)
        logger.info("Exchange Client: %s", "Provided" if self.exchange_available else "Missing")
        logger.info("Portfolio Manager: %s", "Provided" if self.portfolio_manager else "Missing")
        logger.info("Sentiment Sources: %s provided", len(self.sentiment_sources))
        logger.info("AI Analyzer: %s", "Provided" if self.ai_available else "Missing/Disabled")

    def _initialize_news_cache(self) -> None:
        """Initializes the news cache from a file, handling potential errors."""
        self.news_cache: Dict[str, Dict[str, Any]] = {}
        if self.news_cache_file.exists():
            try:
                with self.news_cache_file.open("r", encoding="utf-8") as f:
                    content = f.read()
                    if content.strip():
                        self.news_cache = json.loads(content)
                    else:
                        logger.info("News cache file is empty. Initializing empty cache.")
                logger.info("Loaded news cache from %s", self.news_cache_file)
            except json.JSONDecodeError as e:
                logger.error(
                    "Error decoding JSON from news cache file %s: %s. Initializing empty cache.",
                    self.news_cache_file,
                    e,
                )
                self.news_cache: dict = {}
            except OSError as e:
                logger.error(
                    "OS error reading news cache file %s: %s. Initializing empty cache.",
                    self.news_cache_file,
                    e,
                )
                self.news_cache: dict = {}
            except Exception as e:
                logger.error(
                    "Unexpected error loading news cache: %s. Initializing empty cache.",
                    e,
                    exc_info=True,
                )
                self.news_cache: dict = {}
        else:
            logger.info("News cache file not found. Initializing empty cache.")
            self.news_cache: dict = {}

    def _update_news_cache(self, asset: str, news_data: Optional[dict]) -> None:
        """Updates the news cache and saves it atomically."""
        if not hasattr(self, "news_cache"):
            self.news_cache: dict = {}
        if news_data is None:
            logger.debug("Skipping news cache update for %s: data is None.", asset)
            return
        self.news_cache[asset] = {
            "data": news_data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        temp_file = self.news_cache_file.with_suffix(f".tmp_{time.time_ns()}")
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with temp_file.open("w", encoding="utf-8") as f:
                json.dump(self.news_cache, f, indent=2)
            temp_file.replace(self.news_cache_file)
            logger.debug("Updated news cache file.")
        except (OSError, TypeError, ValueError) as e:
            logger.error("Failed to save news cache: %s", e, exc_info=True)
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass

    def _get_cached_news(self, asset: str) -> Optional[dict]:
        """Retrieves news data from cache if valid and not expired."""
        if not hasattr(self, "news_cache") or asset not in self.news_cache:
            return None
        cache_entry = self.news_cache.get(asset)
        if cache_entry is None or not isinstance(cache_entry, dict):
            return None
        expiry = self.config.get("news_client.cache_expiry_seconds", 1800)
        try:
            timestamp_str = cache_entry.get("timestamp")
            data = cache_entry.get("data")
            if timestamp_str is None or data is None:
                return None
            cache_time = datetime.fromisoformat(timestamp_str).astimezone(timezone.utc)
            if (datetime.now(timezone.utc) - cache_time).total_seconds() > expiry:
                logger.info("News cache for %s expired.", asset)
                return None
            logger.info("Using cached news data for %s", asset)
            return data
        except (ValueError, TypeError) as e:
            logger.warning("Error processing cache timestamp for %s: %s", asset, e)
            return None
        except Exception as e:
            logger.error("Unexpected error reading cache for %s: %s", asset, e, exc_info=True)
            return None

    def initialize_market_data(self) -> None:
        """Initializes market data structures, filtering unsupported pairs."""
        logger.info("Initializing market data...")
        if self.exchange_available:
            if hasattr(self.exchange_client, "_validate_symbol") and callable(
                getattr(self.exchange_client, "_validate_symbol")
            ):
                self.filter_unsupported_trading_pairs()
            else:
                logger.warning(
                    "Exchange client does not implement '_validate_symbol'. Cannot filter pairs."
                )
        else:
            logger.warning("Exchange client not available, cannot filter trading pairs.")
        self.market_data = {
            pair: self._default_market_data_structure() for pair in self.trading_pairs
        }
        self.opportunity_scores = {pair: 0.5 for pair in self.trading_pairs}
        expiry = self.config.get("kraken_client.cache_expiry_seconds.ohlcv", 300)
        for pair in self.trading_pairs:
            cache_file = CACHE_DIR / f"{pair.replace('/', '_')}_{self.timeframe}_cache.json"
            if cache_file.exists():
                try:
                    with cache_file.open("r", encoding="utf-8") as f:
                        cached_content = f.read()
                        if not cached_content.strip():
                            continue
                        cached_data = json.loads(cached_content)
                    if self._validate_cached_data(cached_data, expiry):
                        for key, value in cached_data.items():
                            if key in self.market_data[pair]:
                                self.market_data[pair][key] = value
                        logger.info("Loaded valid cached data for %s", pair)
                    else:
                        logger.warning("Cached data for %s invalid/expired.", pair)
                except json.JSONDecodeError as e:
                    logger.warning("Failed decoding cache for %s: %s", pair, e)
                except (OSError, ValueError, TypeError) as e:
                    logger.warning("Failed loading cache for %s: %s", pair, e, exc_info=True)

    def _default_market_data_structure(self) -> Dict[str, Any]:
        """Returns the default dictionary structure for storing market data per pair."""
        return {
            "ohlcv": [],
            "ticker": None,
            "order_book": None,
            "indicators": {},
            "sentiment": {},
            "sentiment_metrics": {},
            "ai_analysis": {},
            "ai_metrics": {},
            "signals": {"technical": "hold", "sentiment": "hold", "ai": "hold", "combined": "hold"},
            "last_update": {},
            "opportunity_components": {},
            "combined_score": 0.0,
        }

    def _validate_cached_data(self, cached_data: Optional[Dict], max_age_seconds: int) -> bool:
        """Validates the structure and freshness of cached market data."""
        if not isinstance(cached_data, dict):
            return False
        req_keys = ["ohlcv", "ticker", "indicators", "signals", "last_update"]
        if not all((key in cached_data for key in req_keys)):
            return False
        last_update = cached_data.get("last_update", {})
        if not isinstance(last_update, dict) or not last_update:
            return False
        newest_ts = None
        for ts_val in last_update.values():
            if ts_val and isinstance(ts_val, str):
                try:
                    dt = datetime.fromisoformat(ts_val).astimezone(timezone.utc)
                    newest_ts = max(newest_ts, dt) if newest_ts else dt
                except (ValueError, TypeError):
                    continue
        if newest_ts is None:
            return False
        age = (datetime.now(timezone.utc) - newest_ts).total_seconds()
        if age > max_age_seconds:
            logger.debug("Cache expired. Age: %.0fs > Max: %ss", age, max_age_seconds)
            return False
        return True

    def filter_unsupported_trading_pairs(self) -> None:
        """Filters the trading_pairs list based on exchange validation."""
        if not self.exchange_available or not hasattr(self.exchange_client, "_validate_symbol"):
            logger.error("Cannot filter pairs: Exchange unavailable or _validate_symbol missing.")
            return
        logger.info("Filtering trading pairs...")
        markets = None
        try:
            if hasattr(self.exchange_client, "markets"):
                markets = self.exchange_client.markets
            if (
                not markets
                and hasattr(self.exchange_client, "refresh_markets")
                and callable(getattr(self.exchange_client, "refresh_markets"))
            ):
                self.exchange_client.refresh_markets()
                markets = self.exchange_client.markets
            if not markets:
                logger.error("Failed to load markets from exchange client. Cannot filter pairs.")
                return
            supported = [p for p in self.trading_pairs if self.exchange_client._validate_symbol(p)]
            unsupported = set(self.trading_pairs) - set(supported)
            if unsupported:
                logger.warning(
                    "Removing %s unsupported pairs: %s", len(unsupported), ", ".join(unsupported)
                )
            self.trading_pairs = supported
            logger.info(
                "Trading %s pairs: %s", len(self.trading_pairs), ", ".join(self.trading_pairs)
            )
            self.market_data = {
                pair: self._default_market_data_structure() for pair in self.trading_pairs
            }
            self.opportunity_scores = {pair: 0.5 for pair in self.trading_pairs}
        except ccxt.NetworkError as e:
            logger.error(
                "Network error while filtering pairs: %s. Retaining previous pair list.", e
            )
        except Exception as e:
            logger.error("Error filtering unsupported pairs: %s", e, exc_info=True)

    def save_cached_data(self, pair: str) -> None:
        """Saves the current market data for a pair to its cache file atomically."""
        if pair not in self.market_data:
            return
        cache_file = CACHE_DIR / f"{pair.replace('/', '_')}_{self.timeframe}_cache.json"
        temp_file = cache_file.with_suffix(f".tmp_{time.time_ns()}")
        try:
            cached_data = {k: v for (k, v) in self.market_data[pair].items() if k != "order_book"}
            if "last_update" in cached_data and isinstance(cached_data["last_update"], dict):
                cached_data["last_update"] = {
                    k: v for (k, v) in cached_data["last_update"].items() if v is not None
                }
            with temp_file.open("w", encoding="utf-8") as f:
                json.dump(cached_data, f, indent=2, default=str)
            temp_file.replace(cache_file)
            logger.debug("Saved cache for %s", pair)
        except (OSError, TypeError, ValueError) as e:
            logger.error("Error saving cache for %s: %s", pair, e, exc_info=True)
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass

    @timed_operation
    def update_market_data(self, pair: str) -> None:
        """Fetches and updates all relevant market data for a single pair."""
        if pair not in self.trading_pairs:
            logger.warning("Skipping market data update for inactive/invalid pair: %s", pair)
            return
        try:
            (base_currency, quote_currency) = pair.split("/")
            logger.debug("Updating data for %s", pair)
            if pair not in self.market_data:
                self.market_data[pair] = self._default_market_data_structure()
            if not isinstance(self.market_data[pair].get("last_update"), dict):
                self.market_data[pair]["last_update"] = {}
            current_ts_iso = datetime.now(timezone.utc).isoformat()
            if self.exchange_available:
                if self.exchange_circuit_breaker.is_open:
                    logger.warning(
                        "Exchange circuit breaker is open. Skipping exchange data update for %s.",
                        pair,
                    )
                else:
                    try:
                        ohlcv_limit = self.config.get("trading.indicator_lookback_max", 200)
                        ohlcv = self.exchange_client.get_ohlcv(
                            pair, self.timeframe, limit=ohlcv_limit
                        )
                        if ohlcv is not None and isinstance(ohlcv, list):
                            self.market_data[pair]["ohlcv"] = ohlcv
                            self.market_data[pair]["last_update"]["ohlcv"] = current_ts_iso
                            logger.debug("Updated OHLCV for %s (%s records)", pair, len(ohlcv))
                        else:
                            logger.warning("Received invalid/empty OHLCV for %s", pair)
                        ticker = self.exchange_client.get_ticker(pair)
                        if ticker is not None and isinstance(ticker, dict):
                            self.market_data[pair]["ticker"] = ticker
                            self.market_data[pair]["last_update"]["ticker"] = current_ts_iso
                            logger.debug("Updated ticker for %s", pair)
                        else:
                            logger.warning("Received invalid/empty ticker for %s", pair)
                        self.exchange_circuit_breaker.record_success()
                    except (ccxt.NetworkError, ccxt.ExchangeError, TimeoutError) as e:
                        logger.error("Exchange error fetching data for %s: %s", pair, e)
                        self.exchange_circuit_breaker.record_failure()
                    except Exception as e:
                        logger.error(
                            "Unexpected error fetching exchange data for %s: %s",
                            pair,
                            e,
                            exc_info=False,
                        )
                        self.exchange_circuit_breaker.record_failure()
            else:
                logger.warning("Skipping exchange data fetch for %s: Client unavailable.", pair)
            if self.sentiment_available:
                logger.debug("Updating sentiment data for %s...", pair)
                self.market_data[pair]["sentiment_metrics"] = {}
                sentiment_updated = False
                for source in self.sentiment_sources:
                    source_name = type(source).__name__
                    try:
                        result = source.get_sentiment_analysis(base_currency)
                        source_name_from_result = result.get("source_name", source_name)
                    except Exception as e:
                        logger.error("Error getting sentiment from %s: %s", source_name, e)
                        continue
        except Exception as e:
            logger.error("Error updating market data for %s: %s", pair, e, exc_info=True)

    @timed_operation
    def update_market_data_old(self, pair: str) -> None:
        """Old version of update_market_data for reference."""
        try:
            (base_currency, quote_currency) = pair.split("/")
            logger.debug("Updating data for %s", pair)
            if pair not in self.market_data:
                self.market_data[pair] = self._default_market_data_structure()
            if not isinstance(self.market_data[pair].get("last_update"), dict):
                self.market_data[pair]["last_update"] = {}
            current_ts_iso = datetime.now(timezone.utc).isoformat()
            if self.exchange_available:
                try:
                    ohlcv_limit = self.config.get("trading.indicator_lookback_max", 200)
                    ohlcv = self.exchange_client.get_ohlcv(pair, self.timeframe, limit=ohlcv_limit)
                    if ohlcv is not None and isinstance(ohlcv, list):
                        self.market_data[pair]["ohlcv"] = ohlcv
                        self.market_data[pair]["last_update"]["ohlcv"] = current_ts_iso
                        logger.debug("Updated OHLCV for %s (%s records)", pair, len(ohlcv))
                    else:
                        logger.warning("Received invalid/empty OHLCV for %s", pair)
                except (ccxt.NetworkError, ccxt.ExchangeError, TimeoutError) as e:
                    logger.error("Exchange error fetching OHLCV %s: %s", pair, e)
                except Exception as e:
                    logger.error("Unexpected error fetching OHLCV %s: %s", pair, e, exc_info=False)
                try:
                    ticker = self.exchange_client.get_ticker(pair)
                    if ticker is not None and isinstance(ticker, dict):
                        self.market_data[pair]["ticker"] = ticker
                        self.market_data[pair]["last_update"]["ticker"] = current_ts_iso
                        logger.debug("Updated ticker for %s", pair)
                    else:
                        logger.warning("Received invalid/empty ticker for %s", pair)
                except (ccxt.NetworkError, ccxt.ExchangeError, TimeoutError) as e:
                    logger.error("Exchange error fetching ticker %s: %s", pair, e)
                except Exception as e:
                    logger.error("Unexpected error fetching ticker %s: %s", pair, e, exc_info=False)
            else:
                logger.warning("Skipping exchange data fetch for %s: Client unavailable.", pair)
            if self.sentiment_available:
                logger.debug("Updating sentiment data for %s...", pair)
                self.market_data[pair]["sentiment_metrics"] = {}
                sentiment_updated = False
                for source in self.sentiment_sources:
                    source_name = type(source).__name__
                    try:
                        result = source.get_sentiment_analysis(base_currency)
                        source_name_from_result = result.get("source_name", source_name)
                        if (
                            result is not None
                            and isinstance(result, dict)
                            and (not result.get("error"))
                        ):
                            self.market_data[pair]["sentiment_metrics"][
                                source_name_from_result
                            ] = result
                            self.market_data[pair]["last_update"][
                                f"{source_name_from_result}_sentiment"
                            ] = current_ts_iso
                            logger.debug(
                                "Updated sentiment for %s from %s",
                                base_currency,
                                source_name_from_result,
                            )
                            sentiment_updated = True
                        else:
                            logger.warning(
                                "Failed sentiment from %s: %s",
                                source_name,
                                (
                                    result.get("error", "Invalid/Empty result")
                                    if isinstance(result, dict)
                                    else "Invalid result type"
                                ),
                            )
                    except (
                        requests.exceptions.RequestException,
                        prawcore.exceptions.PrawcoreException,
                    ) as api_e:
                        logger.error("API Error getting sentiment from %s: %s", source_name, api_e)
                    except Exception as e:
                        logger.error(
                            "Error getting sentiment from %s: %s", source_name, e, exc_info=True
                        )
                if not sentiment_updated:
                    logger.warning("No sentiment sources updated for %s.", base_currency)
            else:
                logger.debug("Skipping sentiment data fetch for %s: Sources unavailable.", pair)
            if self.market_data[pair]["last_update"].get("ohlcv") == current_ts_iso:
                logger.debug("Calculating indicators for %s...", pair)
                self.calculate_indicators(pair)
            else:
                logger.debug(
                    "Skipping indicator calculation for %s: OHLCV not updated this cycle.", pair
                )
            if self.ai_available and self.ai_analyzer:
                logger.debug("Updating AI analysis for %s...", pair)
                try:
                    ai_context = self._simplify_market_data(self.market_data[pair], pair)
                    ai_result = asyncio.run(self.ai_analyzer.generate_analysis(ai_context, pair))
                    if (
                        ai_result is not None
                        and isinstance(ai_result, dict)
                        and (not ai_result.get("error"))
                    ):
                        self.market_data[pair]["ai_analysis"] = ai_result.get(
                            "raw_analysis", ai_result
                        )
                        self.market_data[pair]["ai_metrics"] = ai_result
                        self.market_data[pair]["last_update"]["ai_analysis"] = current_ts_iso
                        logger.info(
                            "Updated AI analysis for %s: Strategy=%s, Strength=%.2f",
                            pair,
                            ai_result.get("trading_strategy", "N/A"),
                            ai_result.get("strength", 0.0),
                        )
                    else:
                        logger.error(
                            "AI analysis failed for %s: %s",
                            pair,
                            (
                                ai_result.get("error", "Unknown/Invalid result")
                                if isinstance(ai_result, dict)
                                else "Invalid result type"
                            ),
                        )
                except Exception as e:
                    logger.error(
                        "Unexpected error during AI analysis update %s: %s", pair, e, exc_info=True
                    )
            else:
                logger.debug("Skipping AI analysis for %s: Analyzer unavailable.", pair)
            self.save_cached_data(pair)
        except Exception as e:
            logger.error("Major error during market data update for %s: %s", pair, e, exc_info=True)

    def _simplify_market_data(self, market_data_pair: Dict, symbol: str) -> Dict:
        """Creates a simplified market data dictionary suitable for AI prompts, handling potential None values."""
        if not isinstance(market_data_pair, dict):
            return {"symbol": symbol}
        ohlcv = market_data_pair.get("ohlcv")
        indicators = market_data_pair.get("indicators")
        sentiment_metrics = market_data_pair.get("sentiment_metrics")
        ticker = market_data_pair.get("ticker")
        current_price = None
        if ticker and isinstance(ticker, dict) and (ticker.get("last") is not None):
            try:
                current_price = float(ticker["last"])
            except (ValueError, TypeError):
                pass
        if current_price is None and ohlcv and isinstance(ohlcv, list) and (len(ohlcv) > 0):
            last_candle = ohlcv[-1]
            try:
                if isinstance(last_candle, dict):
                    price_val = last_candle.get("close")
                elif isinstance(last_candle, list) and len(last_candle) > 4:
                    price_val = last_candle[4]
                else:
                    price_val = None
                if price_val is not None:
                    current_price = float(price_val)
            except (IndexError, ValueError, TypeError):
                pass
        ohlcv_recent: list = []
        if ohlcv and isinstance(ohlcv, list):
            try:
                ohlcv_recent = [
                    {
                        k: v
                        for (k, v) in c.items()
                        if k in ["timestamp", "open", "high", "low", "close", "volume"]
                    }
                    for c in ohlcv[-20:]
                    if isinstance(c, dict)
                ]
            except Exception as e:
                logger.warning("Could not simplify OHLCV for %s: %s", symbol, e)
        simplified_indicators = {}
        if indicators and isinstance(indicators, dict):
            for k, v in indicators.items():
                if v is not None:
                    try:
                        simplified_indicators[k] = (
                            round(float(v), 4) if isinstance(v, (float, int, Decimal)) else v
                        )
                    except (ValueError, TypeError):
                        simplified_indicators[k] = v
        agg_sentiment_scores = {}
        if sentiment_metrics and isinstance(sentiment_metrics, dict):
            for source_name, metrics in sentiment_metrics.items():
                if isinstance(metrics, dict):
                    strength = metrics.get("strength")
                    if strength is not None:
                        try:
                            agg_sentiment_scores[source_name] = round(float(strength), 3)
                        except (ValueError, TypeError):
                            pass
        return {
            "symbol": symbol,
            "current_price": current_price,
            "ohlcv_recent": ohlcv_recent,
            "indicators": simplified_indicators,
            "sentiment_scores": agg_sentiment_scores,
        }

    def calculate_indicators(self, pair: str) -> None:
        """Calculates technical indicators using available OHLCV data."""
        logger.debug("Calculating indicators for %s", pair)
        indicators = {}
        try:
            pair_data = self.market_data.get(pair)
            if pair_data is None:
                logger.warning("No market data for %s. Cannot calc indicators.", pair)
                return
            ohlcv = pair_data.get("ohlcv")
            if not isinstance(ohlcv, list) or len(ohlcv) < 2:
                logger.warning(
                    "Insufficient/Invalid OHLCV data for %s (records: %s). Cannot calc indicators.",
                    pair,
                    len(ohlcv) if isinstance(ohlcv, list) else 0,
                )
                self.market_data[pair]["indicators"] = {}
                return
            (closes, highs, lows) = ([], [], [])
            valid_candles = 0
            for c in ohlcv:
                try:
                    if isinstance(c, dict):
                        c_close = float(c["close"])
                        c_high = float(c["high"])
                        c_low = float(c["low"])
                    elif isinstance(c, list) and len(c) > 4:
                        c_close = float(c[4])
                        c_high = float(c[2])
                        c_low = float(c[3])
                    else:
                        continue
                    closes.append(c_close)
                    highs.append(c_high)
                    lows.append(c_low)
                    valid_candles += 1
                except (ValueError, TypeError, KeyError, IndexError):
                    logger.debug("Skipping malformed candle in %s: %s", pair, c)
                    continue
            if valid_candles < 2:
                logger.warning(
                    "Insufficient valid candles (%s) for %s. Cannot calc indicators.",
                    valid_candles,
                    pair,
                )
                return
            closes = np.array(closes)
            highs = np.array(highs)
            lows = np.array(lows)
            sma_short = self.config.get("trading.indicator_sma_short", 9)
            sma_long = self.config.get("trading.indicator_sma_long", 21)
            if len(closes) >= sma_short:
                indicators["SMA_Short"] = calculate_sma(closes, sma_short)[-1]
            if len(closes) >= sma_long:
                indicators["SMA_Long"] = calculate_sma(closes, sma_long)[-1]
            ema_short = self.config.get("trading.indicator_ema_short", 12)
            ema_long = self.config.get("trading.indicator_ema_long", 26)
            if len(closes) >= ema_short:
                indicators["EMA_Short"] = calculate_ema(closes, ema_short)[-1]
            if len(closes) >= ema_long:
                indicators["EMA_Long"] = calculate_ema(closes, ema_long)[-1]
            rsi_period = self.config.get("trading.indicator_rsi_period", 14)
            if len(closes) > rsi_period:
                delta = np.diff(closes)
                gain = np.where(delta > 0, delta, 0)
                loss = np.where(delta < 0, -delta, 0)
                avg_gain_series = calculate_ema(gain, rsi_period)
                avg_loss_series = calculate_ema(loss, rsi_period)
                avg_gain = (
                    avg_gain_series[~np.isnan(avg_gain_series)][-1]
                    if not np.all(np.isnan(avg_gain_series))
                    else np.nan
                )
                avg_loss = (
                    avg_loss_series[~np.isnan(avg_loss_series)][-1]
                    if not np.all(np.isnan(avg_loss_series))
                    else np.nan
                )
                if not np.isnan(avg_gain) and (not np.isnan(avg_loss)):
                    if avg_loss < 1e-09:
                        indicators["RSI"] = 100.0
                    else:
                        rs = avg_gain / avg_loss
                        indicators["RSI"] = 100.0 - 100.0 / (1.0 + rs)
                else:
                    indicators["RSI"] = np.nan
            if (
                "EMA_Short" in indicators
                and "EMA_Long" in indicators
                and (not np.isnan(indicators["EMA_Short"]))
                and (not np.isnan(indicators["EMA_Long"]))
            ):
                ema_short_full = calculate_ema(closes, ema_short)
                ema_long_full = calculate_ema(closes, ema_long)
                macd_line = ema_short_full - ema_long_full
                signal_period = self.config.get("trading.indicator_macd_signal", 9)
                valid_macd_indices = ~np.isnan(macd_line)
                if np.sum(valid_macd_indices) >= signal_period:
                    valid_macd_values = macd_line[valid_macd_indices]
                    signal_line_series = calculate_ema(valid_macd_values, signal_period)
                    if len(signal_line_series) > 0 and len(valid_macd_values) >= len(
                        signal_line_series
                    ):
                        macd_hist_series = (
                            valid_macd_values[-len(signal_line_series) :] - signal_line_series
                        )
                    else:
                        macd_hist_series = np.array([np.nan])
                    indicators["MACD_Line"] = (
                        macd_line[-1] if not np.isnan(macd_line[-1]) else np.nan
                    )
                    indicators["MACD_Signal"] = (
                        signal_line_series[-1]
                        if len(signal_line_series) > 0 and (not np.isnan(signal_line_series[-1]))
                        else np.nan
                    )
                    indicators["MACD_Hist"] = (
                        macd_hist_series[-1]
                        if len(macd_hist_series) > 0 and (not np.isnan(macd_hist_series[-1]))
                        else np.nan
                    )
            else:
                logger.debug("Skipping MACD for %s: EMAs missing or NaN.", pair)
            bb_period = self.config.get("trading.indicator_bb_period", 20)
            bb_std = self.config.get("trading.indicator_bb_stddev", 2.0)
            if len(closes) >= bb_period:
                sma_bb_series = calculate_sma(closes, bb_period)
                sma_bb = sma_bb_series[-1]
                if not np.isnan(sma_bb):
                    window_data = closes[-bb_period:]
                    roll_std = np.std(window_data)
                    indicators["BB_Middle"] = sma_bb
                    indicators["BB_Upper"] = sma_bb + bb_std * roll_std
                    indicators["BB_Lower"] = sma_bb - bb_std * roll_std
            self.market_data[pair]["indicators"] = {
                k: v for (k, v) in indicators.items() if v is not None and (not np.isnan(v))
            }
            self.market_data[pair]["last_update"]["indicators"] = datetime.now(
                timezone.utc
            ).isoformat()
            logger.info(
                "Calculated indicators for %s: %s",
                pair,
                list(self.market_data[pair]["indicators"].keys()),
            )
        except Exception as e:
            logger.error("Error calculating indicators for %s: %s", pair, e, exc_info=True)
            self.market_data[pair]["indicators"] = {}

    def generate_technical_signal(self, pair: str) -> str:
        """Generates a trading signal based on technical indicators."""
        logger.debug("Generating technical signal for %s", pair)
        indicators = self.market_data[pair].get("indicators", {})
        signals: list = []
        if not indicators:
            return "hold"
        rsi = indicators.get("RSI")
        rsi_sell = self.config.portfolio.technical_factor_thresholds.rsi_overbought
        rsi_buy = self.config.portfolio.technical_factor_thresholds.rsi_oversold
        if rsi is not None:
            try:
                if float(rsi) < rsi_buy:
                    signals.append("buy")
                elif float(rsi) > rsi_sell:
                    signals.append("sell")
            except (ValueError, TypeError):
                pass
        macd_hist = indicators.get("MACD_Hist")
        macd_thresh = self.config.portfolio.technical_factor_thresholds.macd_threshold
        if macd_hist is not None:
            try:
                if float(macd_hist) > macd_thresh:
                    signals.append("buy")
                elif float(macd_hist) < -macd_thresh:
                    signals.append("sell")
            except (ValueError, TypeError):
                pass
        sma_s = indicators.get("SMA_Short")
        sma_l = indicators.get("SMA_Long")
        if sma_s is not None and sma_l is not None:
            try:
                if float(sma_s) > float(sma_l):
                    signals.append("buy")
                elif float(sma_s) < float(sma_l):
                    signals.append("sell")
            except (ValueError, TypeError):
                pass
        bb_l = indicators.get("BB_Lower")
        bb_u = indicators.get("BB_Upper")
        price = self._get_current_price(pair)
        if bb_l is not None and bb_u is not None and (price is not None):
            try:
                (bb_l_f, bb_u_f, price_f) = (float(bb_l), float(bb_u), float(price))
                if price_f < bb_l_f * 1.005:
                    signals.append("buy")
                elif price_f > bb_u_f * 0.995:
                    signals.append("sell")
            except (ValueError, TypeError):
                pass
        if not signals:
            return "hold"
        buys = signals.count("buy")
        sells = signals.count("sell")
        return "buy" if buys > sells else "sell" if sells > buys else "hold"

    def generate_sentiment_signal(self, pair: str) -> str:
        """Generates a trading signal based on aggregated sentiment metrics."""
        logger.debug("Generating sentiment signal for %s", pair)
        metrics_dict = self.market_data[pair].get("sentiment_metrics", {})
        if not metrics_dict or not isinstance(metrics_dict, dict):
            return "hold"
        strengths: list = []
        for metrics in metrics_dict.values():
            if isinstance(metrics, dict) and "strength" in metrics:
                try:
                    strengths.append(float(metrics["strength"]))
                except (ValueError, TypeError):
                    pass
        if not strengths:
            return "hold"
        try:
            combined = sum(strengths) / len(strengths)
            bullish_thresh = self.config.sentiment.bullish_threshold
            bearish_thresh = self.config.sentiment.bearish_threshold
            return (
                "buy"
                if combined > bullish_thresh
                else "sell" if combined < bearish_thresh else "hold"
            )
        except ZeroDivisionError:
            return "hold"

    def generate_ai_signal(self, pair: str) -> str:
        """Generates a trading signal based on AI analysis."""
        logger.debug("Generating AI signal for %s", pair)
        ai_metrics = self.market_data[pair].get("ai_metrics", {})
        if not ai_metrics or not isinstance(ai_metrics, dict) or ai_metrics.get("error"):
            return "hold"
        strategy = ai_metrics.get("trading_strategy")
        if isinstance(strategy, str):
            strategy = strategy.lower()
            return strategy if strategy in ["buy", "sell"] else "hold"
        return "hold"

    def generate_combined_signal(self, pair: str) -> str:
        """Generates a combined signal based on configured strategy weights."""
        logger.debug("Generating combined signal for %s using strategy: %s", pair, self.strategy)
        signals = self.market_data[pair].get("signals", {})
        if not signals or not isinstance(signals, dict):
            return "hold"
        tech = signals.get("technical", "hold")
        sent = signals.get("sentiment", "hold")
        ai = signals.get("ai", "hold")
        valid_signals = ["buy", "sell", "hold"]
        tech = tech if tech in valid_signals else "hold"
        sent = sent if sent in valid_signals else "hold"
        ai = ai if ai in valid_signals else "hold"
        if self.strategy == "technical":
            return tech
        elif self.strategy == "sentiment":
            return sent
        elif self.strategy == "ai":
            return ai
        elif self.strategy == "combined":
            weights = self.config.portfolio.target_allocation_weights
            try:
                w_tech = float(weights.technical)
            except (AttributeError, ValueError, TypeError):
                w_tech = 0.0
            try:
                w_sent = float(weights.sentiment)
            except (AttributeError, ValueError, TypeError):
                w_sent = 0.0
            try:
                w_ai = float(weights.ai)
            except (AttributeError, ValueError, TypeError):
                w_ai = 0.0
            score = 0.0
            if tech == "buy":
                score += w_tech
            elif tech == "sell":
                score -= w_tech
            if sent == "buy":
                score += w_sent
            elif sent == "sell":
                score -= w_sent
            if ai == "buy":
                score += w_ai
            elif ai == "sell":
                score -= w_ai
            buy_thresh = self.config.get("trading.combined_signal_buy_threshold", 0.2)
            sell_thresh = self.config.get("trading.combined_signal_sell_threshold", -0.2)
            try:
                buy_thresh = float(buy_thresh)
                sell_thresh = float(sell_thresh)
            except (ValueError, TypeError):
                logger.warning("Invalid combined signal thresholds in config, using defaults.")
                buy_thresh = 0.2
                sell_thresh = -0.2
            return "buy" if score > buy_thresh else "sell" if score < sell_thresh else "hold"
        else:
            logger.warning("Unknown strategy '%s'. Defaulting to hold.", self.strategy)
            return "hold"

    def calculate_opportunity_score(self, pair: str) -> float:
        """Calculates an opportunity score based on signal agreement and confidence."""
        logger.debug("Calculating opportunity score for %s", pair)
        md = self.market_data.get(pair)
        if not md or not isinstance(md, dict):
            return 0.5
        signals = md.get("signals", {})
        combined_sig = signals.get("combined", "hold")
        base = 0.75 if combined_sig == "buy" else 0.25 if combined_sig == "sell" else 0.5
        signal_list = [
            s
            for s in [signals.get("technical"), signals.get("sentiment"), signals.get("ai")]
            if s in ["buy", "sell"]
        ]
        agreement = 0.0
        if len(signal_list) > 1:
            if all((s == signal_list[0] for s in signal_list)):
                agreement = 0.15
            elif len(set(signal_list)) == len(signal_list):
                agreement = -0.15
            else:
                agreement = 0.05
        confidences: list = []
        tech_conf = 0.5
        confidences.append(tech_conf)
        sent_metrics = md.get("sentiment_metrics", {})
        if isinstance(sent_metrics, dict) and sent_metrics:
            sent_confs = [
                m.get("confidence", 0.0) for m in sent_metrics.values() if isinstance(m, dict)
            ]
            valid_sent_confs = [
                c for c in sent_confs if isinstance(c, (float, int)) and 0.0 <= c <= 1.0
            ]
            if valid_sent_confs:
                confidences.append(np.mean(valid_sent_confs))
        ai_metrics = md.get("ai_metrics", {})
        if isinstance(ai_metrics, dict) and "confidence" in ai_metrics:
            ai_conf_val = ai_metrics["confidence"]
            if isinstance(ai_conf_val, (float, int)) and 0.0 <= ai_conf_val <= 1.0:
                confidences.append(ai_conf_val)
        avg_conf = np.mean(confidences) if confidences else 0.5
        final_score = base + agreement + (avg_conf - 0.5) * 0.2
        final_score = max(0.0, min(1.0, final_score))
        md["opportunity_components"] = {
            "base": base,
            "agreement": agreement,
            "avg_confidence": avg_conf,
            "final": final_score,
        }
        if isinstance(md.get("last_update"), dict):
            md["last_update"]["opportunity_score"] = datetime.now(timezone.utc).isoformat()
        else:
            md["last_update"] = {"opportunity_score": datetime.now(timezone.utc).isoformat()}
        return final_score

    def determine_confidence_band(self, pair: str) -> str:
        """Determines the confidence band based on signal agreement and average confidence."""
        logger.debug("Determining confidence band for %s", pair)
        md = self.market_data.get(pair)
        if not md or not isinstance(md, dict):
            return "low"
        signals = md.get("signals", {})
        signal_list = [
            s
            for s in [signals.get("technical"), signals.get("sentiment"), signals.get("ai")]
            if s in ["buy", "sell"]
        ]
        agreement = 0
        if len(signal_list) > 1:
            if all((s == signal_list[0] for s in signal_list)):
                agreement = 2
            elif len(set(signal_list)) < len(signal_list):
                agreement = 1
        confidences: list = []
        sent_metrics = md.get("sentiment_metrics", {})
        if isinstance(sent_metrics, dict) and sent_metrics:
            sent_confs = [
                m.get("confidence", 0.0) for m in sent_metrics.values() if isinstance(m, dict)
            ]
            valid_sent_confs = [
                c for c in sent_confs if isinstance(c, (float, int)) and 0.0 <= c <= 1.0
            ]
            if valid_sent_confs:
                confidences.append(np.mean(valid_sent_confs))
        ai_metrics = md.get("ai_metrics", {})
        if isinstance(ai_metrics, dict) and "confidence" in ai_metrics:
            ai_conf_val = ai_metrics["confidence"]
            if isinstance(ai_conf_val, (float, int)) and 0.0 <= ai_conf_val <= 1.0:
                confidences.append(ai_conf_val)
        avg_conf = np.mean(confidences) if confidences else 0.5
        bands = self.config.trading.confidence_bands
        if not bands or not isinstance(bands, dict):
            bands = {"low": [0.0, 0.4], "medium": [0.4, 0.7], "high": [0.7, 1.0]}
        med_band = bands.get("medium", [0.4, 0.7])
        high_band = bands.get("high", [0.7, 1.0])
        if agreement == 2 and avg_conf >= med_band[1]:
            return "high"
        elif agreement == 0 or avg_conf < med_band[0]:
            return "low"
        else:
            return "medium"

    def _calculate_atr(self, pair: str, period: int) -> Optional[float]:
        """Calculate Average True Range (ATR)."""
        pair_data = self.market_data.get(pair, {})
        ohlcv = pair_data.get("ohlcv")
        if not isinstance(ohlcv, list) or len(ohlcv) < period + 1:
            logger.warning(
                "Insufficient OHLCV data for ATR %s (Need %s, have %s)",
                pair,
                period + 1,
                len(ohlcv) if isinstance(ohlcv, list) else 0,
            )
            return None
        try:
            (highs, lows, closes) = ([], [], [])
            valid_candles = 0
            for c in ohlcv:
                try:
                    if isinstance(c, dict):
                        (h, l, cl) = (float(c["high"]), float(c["low"]), float(c["close"]))
                    elif isinstance(c, list) and len(c) > 4:
                        (h, l, cl) = (float(c[2]), float(c[3]), float(c[4]))
                    else:
                        continue
                    highs.append(h)
                    lows.append(l)
                    closes.append(cl)
                    valid_candles += 1
                except (ValueError, TypeError, KeyError, IndexError):
                    continue
            if valid_candles < period + 1:
                logger.warning("Insufficient valid candles for ATR %s.", pair)
                return None
            highs = np.array(highs)
            lows = np.array(lows)
            closes = np.array(closes)
            high_low = highs - lows
            high_close = np.abs(highs[1:] - closes[:-1])
            low_close = np.abs(lows[1:] - closes[:-1])
            tr = np.full_like(highs, np.nan)
            tr[1:] = np.maximum(high_low[1:], high_close)
            tr[1:] = np.maximum(tr[1:], low_close)
            atr_series = np.full_like(closes, np.nan)
            if len(tr[1 : period + 1]) > 0:
                first_atr = np.nanmean(tr[1 : period + 1])
                if not np.isnan(first_atr):
                    atr_series[period] = first_atr
                    alpha = 1.0 / period
                    for i in range(period + 1, len(closes)):
                        if not np.isnan(tr[i]) and (not np.isnan(atr_series[i - 1])):
                            atr_series[i] = alpha * tr[i] + (1 - alpha) * atr_series[i - 1]
                        elif not np.isnan(atr_series[i - 1]):
                            atr_series[i] = atr_series[i - 1]
            last_atr = atr_series[-1]
            return last_atr if not np.isnan(last_atr) else None
        except Exception as e:
            logger.error("Error calculating ATR for %s: %s", pair, e, exc_info=True)
            return None

    def calculate_position_size(self, pair: str) -> Optional[Decimal]:
        """Calculate position size using Decimal precision."""
        logger.debug("Calculating position size for %s", pair)
        if not self.exchange_available:
            logger.error("Cannot size position: Exchange client unavailable.")
            return None
        current_price_dec = self._get_current_price_decimal(pair)
        if current_price_dec is None or current_price_dec <= Decimal("0"):
            logger.error("Cannot size position for %s: Invalid current price.", pair)
            return None
        balance_data = self.exchange_client.get_balance()
        if balance_data is None:
            logger.error("Cannot size position for %s: Failed to fetch balance.", pair)
            return None
        (base_currency, quote_currency) = pair.split("/")
        available_quote_float = balance_data.get(quote_currency.upper(), 0.0)
        try:
            available_quote_dec = Decimal(str(available_quote_float))
            if available_quote_dec <= Decimal("0.0"):
                logger.warning(
                    "Cannot size position for %s: Available %s balance is zero or negative.",
                    pair,
                    quote_currency,
                )
                return None
        except InvalidOperation:
            logger.error(
                "Cannot size position for %s: Invalid available quote balance value %s.",
                pair,
                available_quote_float,
            )
            return None
        try:
            base_value_multiplier = Decimal(str(self.config.trading.position_size_multiplier))
            calculated_base_value = available_quote_dec * base_value_multiplier
            atr_float = self._calculate_atr(pair, self.config.trading.volatility_lookback_period)
            risk_adj_value = calculated_base_value
            if atr_float is not None and atr_float > 1e-09:
                logger.debug(
                    "%s: ATR found (%.4f), but using %% of available balance for sizing.",
                    pair,
                    atr_float,
                )
            else:
                logger.debug(
                    "%s: Skipping ATR risk adjustment (ATR unavailable/zero). Using base value.",
                    pair,
                )
            conf_band = self.determine_confidence_band(pair)
            band_mults = self.config.trading.confidence_band_multipliers
            conf_mult = Decimal(str(getattr(band_mults, conf_band, "1.0")))
            desired_value_dec = risk_adj_value * conf_mult
        except (AttributeError, ValueError, TypeError, InvalidOperation) as calc_e:
            logger.error(
                "Error during initial value calculation for %s: %s. Cannot size position.",
                pair,
                calc_e,
                exc_info=True,
            )
            return None
        try:
            fee_rate_dec = Decimal(
                str(self.config.get("portfolio.simulation.fee_rate_multiplier", "0.001"))
            )
            max_affordable_value_gross = available_quote_dec / (Decimal("1.0") + fee_rate_dec)
            final_value_dec = desired_value_dec
            if final_value_dec > max_affordable_value_gross:
                logger.warning(
                    "Insufficient %s balance for desired size %.2f. Capping size.",
                    quote_currency,
                    desired_value_dec,
                )
                final_value_dec = max_affordable_value_gross * Decimal("0.999")
            if final_value_dec <= Decimal("0.0"):
                logger.error(
                    "Final value for %s is zero or negative after balance cap. Cannot trade.", pair
                )
                return None
        except (AttributeError, InvalidOperation, ValueError, TypeError) as cap_err:
            logger.error(
                "Error applying balance cap for %s: %s. Cannot size position.",
                pair,
                cap_err,
                exc_info=True,
            )
            return None
        try:
            quantity_dec = final_value_dec / current_price_dec
            if quantity_dec <= Decimal("0"):
                logger.error("Calculated quantity for %s is zero or negative.", pair)
                return None
            (min_amount, min_cost) = (None, None)
            if (
                self.exchange_available
                and hasattr(self.exchange_client, "markets")
                and self.exchange_client.markets
            ):
                market_info = self.exchange_client.markets.get(pair, {})
                if market_info:
                    limits_info = market_info.get("limits", {})
                    if limits_info:
                        min_amount = limits_info.get("amount", {}).get("min")
                        min_cost = limits_info.get("cost", {}).get("min")
            if min_amount is not None:
                min_amount_dec = Decimal(str(min_amount))
                if quantity_dec < min_amount_dec:
                    logger.warning(
                        "%s calc qty %s < min %s. Adjusting.", pair, quantity_dec, min_amount_dec
                    )
                    quantity_dec = min_amount_dec
                    final_value_dec = quantity_dec * current_price_dec
            if min_cost is not None:
                min_cost_dec = Decimal(str(min_cost))
                if final_value_dec < min_cost_dec:
                    logger.error(
                        "%s final value %.2f < min cost %.2f. Cannot trade.",
                        pair,
                        final_value_dec,
                        min_cost_dec,
                    )
                    return None
            if quantity_dec <= Decimal("0"):
                logger.error("Final quantity for %s is zero/negative after limit checks.", pair)
                return None
            logger.info(
                "Calculated Position Size for %s: Qty=%s, Value=%.2f",
                pair,
                quantity_dec,
                final_value_dec,
            )
            return quantity_dec
        except (DivisionByZero, InvalidOperation, ValueError, TypeError) as final_calc_e:
            logger.error(
                "Error during final quantity/limit calculation for %s: %s. Cannot size position.",
                pair,
                final_calc_e,
                exc_info=True,
            )
            return None
        except Exception as e:
            logger.error(
                "Unexpected error calculating position size %s: %s", pair, e, exc_info=True
            )
            return None
        if not self.exchange_available:
            logger.error("Cannot size position: Exchange client unavailable.")
            return None
        current_price_float = self._get_current_price(pair)
        if current_price_float is None or current_price_float <= 1e-09:
            logger.error(
                "Cannot size position for %s: Invalid current price (%s).",
                pair,
                current_price_float,
            )
            return None
        try:
            current_price_dec = Decimal(str(current_price_float))
        except InvalidOperation:
            logger.error(
                "Cannot size position for %s: Could not convert price to Decimal (%s).",
                pair,
                current_price_float,
            )
            return None
        balance_data = self.exchange_client.get_balance()
        if balance_data is None:
            logger.error("Cannot size position for %s: Failed to fetch balance.", pair)
            return None
        (base_currency, quote_currency) = pair.split("/")
        available_quote_float = balance_data.get(quote_currency.upper(), 0.0)
        try:
            available_quote_dec = Decimal(str(available_quote_float))
            if available_quote_dec <= Decimal("0.0"):
                logger.warning(
                    "Cannot size position for %s: Available %s balance is zero or negative.",
                    pair,
                    quote_currency,
                )
                return None
        except InvalidOperation:
            logger.error(
                "Cannot size position for %s: Invalid available quote balance value %s.",
                pair,
                available_quote_float,
            )
            return None
        try:
            base_value_multiplier = Decimal(str(self.config.trading.position_size_multiplier))
            calculated_base_value = available_quote_dec * base_value_multiplier
            atr = self._calculate_atr(pair, self.config.trading.volatility_lookback_period)
            risk_adj_value = calculated_base_value
            if atr is not None and atr > 1e-09:
                logger.debug(
                    "%s: ATR found (%.4f), but using %% of available balance for sizing.", pair, atr
                )
                risk_adj_value = calculated_base_value
            else:
                logger.debug(
                    "%s: Skipping ATR risk adjustment (ATR unavailable/zero). Using base value.",
                    pair,
                )
            conf_band = self.determine_confidence_band(pair)
            band_mults = self.config.trading.confidence_band_multipliers
            conf_mult = Decimal("1.0")
            try:
                conf_mult = Decimal(str(getattr(band_mults, conf_band, 1.0)))
            except (AttributeError, ValueError, TypeError, InvalidOperation):
                logger.warning("Invalid confidence multiplier for band '%s'. Using 1.0.", conf_band)
            desired_value_dec = risk_adj_value * conf_mult
            logger.debug(
                "%s: ConfBand=%s, ConfMult=%.2f, DesiredValue=%.2f",
                pair,
                conf_band,
                conf_mult,
                desired_value_dec,
            )
        except (AttributeError, ValueError, TypeError, InvalidOperation) as calc_e:
            logger.error(
                "Error during initial value calculation for %s: %s. Cannot size position.",
                pair,
                calc_e,
                exc_info=True,
            )
            return None
        try:
            fee_rate_dec = Decimal(
                str(self.config.get("portfolio.simulation.fee_rate_multiplier", 0.001))
            )
            max_affordable_value_gross = available_quote_dec / (Decimal("1.0") + fee_rate_dec)
            final_value_dec = desired_value_dec
            if final_value_dec > max_affordable_value_gross:
                logger.warning(
                    "Insufficient %s balance for desired size %.2f. Available (pre-fee): %.2f. Capping size.",
                    quote_currency,
                    desired_value_dec,
                    max_affordable_value_gross,
                )
                final_value_dec = max_affordable_value_gross
                final_value_dec *= Decimal("0.999")
            if final_value_dec <= Decimal("0.0"):
                logger.error(
                    "Final value for %s is zero or negative after balance cap. Cannot trade.", pair
                )
                return None
        except (AttributeError, InvalidOperation, ValueError, TypeError) as cap_err:
            logger.error(
                "Error applying balance cap for %s: %s. Cannot size position.",
                pair,
                cap_err,
                exc_info=True,
            )
            return None
        try:
            quantity_dec = final_value_dec / current_price_dec
            quantity_float = float(quantity_dec)
            if quantity_float <= 0:
                logger.error(
                    "Calculated quantity for %s is zero or negative (%.8f).", pair, quantity_float
                )
                return None
            (min_amount, min_cost) = (None, None)
            if (
                self.exchange_available
                and hasattr(self.exchange_client, "markets")
                and self.exchange_client.markets
            ):
                market_info = self.exchange_client.markets.get(pair, {})
                if market_info and isinstance(market_info, dict):
                    limits_info = market_info.get("limits", {})
                    if isinstance(limits_info, dict):
                        amount_limits = limits_info.get("amount", {})
                        cost_limits = limits_info.get("cost", {})
                        if isinstance(amount_limits, dict):
                            min_amount = amount_limits.get("min")
                        if isinstance(cost_limits, dict):
                            min_cost = cost_limits.get("min")
            else:
                logger.warning("Exchange market limits unavailable.")
            if min_amount is not None:
                try:
                    min_amount_f = float(min_amount)
                except (ValueError, TypeError):
                    min_amount_f = None
                if min_amount_f is not None and quantity_float < min_amount_f:
                    logger.warning(
                        "%s calc qty %.8f < min %.8f. Adjusting.",
                        pair,
                        quantity_float,
                        min_amount_f,
                    )
                    quantity_float = min_amount_f
                    final_value_dec = Decimal(str(quantity_float)) * current_price_dec
            if min_cost is not None:
                try:
                    min_cost_f = float(min_cost)
                except (ValueError, TypeError):
                    min_cost_f = None
                if min_cost_f is not None and final_value_dec < Decimal(str(min_cost_f)):
                    logger.error(
                        "%s final value %.2f < min cost %.2f. Cannot trade.",
                        pair,
                        final_value_dec,
                        min_cost_f,
                    )
                    return None
            if quantity_float <= 0:
                logger.error(
                    "Final quantity for %s is zero/negative (%.8f) after limit checks.",
                    pair,
                    quantity_float,
                )
                return None
            logger.info(
                "Calculated Position Size for %s: Qty=%.8f, Value=%.2f",
                pair,
                quantity_float,
                final_value_dec,
            )
            return quantity_float
        except (DivisionByZero, InvalidOperation, ValueError, TypeError) as final_calc_e:
            logger.error(
                "Error during final quantity/limit calculation for %s: %s. Cannot size position.",
                pair,
                final_calc_e,
                exc_info=True,
            )
            return None
        except Exception as e:
            logger.error(
                "Unexpected error calculating position size %s: %s", pair, e, exc_info=True
            )
            return None

    def _log_skipped_trade(
        self, pair: str, action: str, reason: str, details: Optional[Dict] = None
    ) -> None:
        """Logs skipped bot trades to a JSON file atomically."""
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pair": pair,
            "action_intended": action,
            "reason": reason,
            "details": details or {},
        }
        logger.warning("Skipping %s %s: %s", action.upper(), pair, reason)
        temp_file = self.bot_skipped_trades_log_file.with_suffix(f".tmp_{time.time_ns()}")
        try:
            log_list: list = []
            if self.bot_skipped_trades_log_file.exists():
                try:
                    with self.bot_skipped_trades_log_file.open("r", encoding="utf-8") as f:
                        content = f.read()
                        if content.strip():
                            loaded_list = json.loads(content)
                            if isinstance(loaded_list, list):
                                log_list = loaded_list
                except json.JSONDecodeError:
                    log_list: list = []
                except OSError:
                    log_list: list = []
            log_list.append(log_entry)
            max_log_entries = self.config.get("logging.max_skipped_trade_log_entries", 5000)
            if len(log_list) > max_log_entries:
                log_list = log_list[-max_log_entries:]
            with temp_file.open("w", encoding="utf-8") as f:
                json.dump(log_list, f, indent=2)
            temp_file.replace(self.bot_skipped_trades_log_file)
            logger.debug("Logged skipped bot trade.")
        except (OSError, TypeError, ValueError) as e:
            logger.error("Failed to log skipped bot trade: %s", e, exc_info=True)
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass

    @timed_operation
    def generate_signals(self, pair: str) -> None:
        """Generates technical, sentiment, AI, and combined signals for a pair."""
        logger.debug("Generating signals and score for %s", pair)
        try:
            if pair not in self.market_data:
                self.market_data[pair] = self._default_market_data_structure()
            signals = self.market_data[pair].setdefault(
                "signals",
                {"technical": "hold", "sentiment": "hold", "ai": "hold", "combined": "hold"},
            )
            if not isinstance(signals, dict):
                signals = {
                    "technical": "hold",
                    "sentiment": "hold",
                    "ai": "hold",
                    "combined": "hold",
                }
                self.market_data[pair]["signals"] = signals
            signals["technical"] = self.generate_technical_signal(pair)
            signals["sentiment"] = self.generate_sentiment_signal(pair)
            signals["ai"] = self.generate_ai_signal(pair)
            signals["combined"] = self.generate_combined_signal(pair)
            opp_score = self.calculate_opportunity_score(pair)
            self.opportunity_scores[pair] = opp_score
            conf_band = self.determine_confidence_band(pair)
            logger.info(
                "Signals for %s: Comb=%s, Score=%.2f, Conf=%s (Tech=%s, Sent=%s, AI=%s)",
                pair,
                signals["combined"],
                opp_score,
                conf_band,
                signals["technical"],
                signals["sentiment"],
                signals["ai"],
            )
        except Exception as e:
            logger.error("Error generating signals/score for %s: %s", pair, e, exc_info=True)
            self.market_data[pair]["signals"] = {
                "technical": "hold",
                "sentiment": "hold",
                "ai": "hold",
                "combined": "hold",
            }
            self.opportunity_scores[pair] = 0.5

    def _get_current_price_decimal(self, pair: str) -> Optional[Decimal]:
        """Get current price as Decimal for precision calculations."""
        price_float = self._get_current_price(pair)
        if price_float is None:
            return None
        try:
            return Decimal(str(price_float))
        except InvalidOperation:
            logger.error("Could not convert price %s to Decimal for %s.", price_float, pair)
            return None

    def _get_current_price(self, pair: str) -> Optional[float]:
        """Safely retrieves the current price for a pair from cached data or exchange."""
        price = None
        md = self.market_data.get(pair)
        if isinstance(md, dict) and isinstance(md.get("ticker"), dict):
            ticker = md["ticker"]
            last_price = ticker.get("last")
            ticker_time_str = ticker.get("datetime")
            if last_price is not None and ticker_time_str:
                try:
                    price_f = float(last_price)
                    if (
                        isinstance(ticker_time_str, str)
                        and "T" in ticker_time_str
                        and (
                            "Z" in ticker_time_str
                            or "+" in ticker_time_str
                            or "-" in ticker_time_str[10:]
                        )
                    ):
                        ticker_time = datetime.fromisoformat(ticker_time_str).astimezone(
                            timezone.utc
                        )
                        if (datetime.now(timezone.utc) - ticker_time).total_seconds() < 300:
                            price = price_f
                        else:
                            logger.debug("Ticker price for %s is stale.", pair)
                    else:
                        logger.debug(
                            "Invalid ticker datetime format for %s: %s", pair, ticker_time_str
                        )
                except (ValueError, TypeError):
                    pass
        if (
            price is None
            and isinstance(md, dict)
            and isinstance(md.get("ohlcv"), list)
            and md["ohlcv"]
        ):
            last_candle = md["ohlcv"][-1]
            try:
                if isinstance(last_candle, dict):
                    close_val = last_candle.get("close")
                elif isinstance(last_candle, list) and len(last_candle) > 4:
                    close_val = last_candle[4]
                else:
                    close_val = None
                if close_val is not None:
                    price = float(close_val)
            except (IndexError, ValueError, TypeError):
                pass
        if price is None and self.exchange_available:
            logger.debug("No fresh cached price for %s, attempting direct fetch...", pair)
            try:
                ticker = self.exchange_client.get_ticker(pair, force_refresh=True)
                if isinstance(ticker, dict) and ticker.get("last") is not None:
                    price = float(ticker["last"])
            except (ccxt.NetworkError, ccxt.ExchangeError, TimeoutError) as e:
                logger.warning("Direct price fetch failed for %s: %s", pair, e)
            except (ValueError, TypeError) as e:
                logger.warning("Invalid price format from direct fetch %s: %s", pair, e)
            except Exception as e:
                logger.warning("Unexpected error during direct price fetch %s: %s", pair, e)
        if price is None:
            logger.error("Could not determine current price for %s.", pair)
        elif price <= 1e-09:
            logger.error("Determined price for %s is zero or negative (%s).", pair, price)
            return None
        return price

    @timed_operation
    def execute_trading_decision(self, pair: str) -> bool:
        """Determines if a trade should be placed and executes it via the OrderManager."""
        logger.debug("Executing decision for %s", pair)
        if not self.exchange_available:
            self._log_skipped_trade(pair, "N/A", "Exchange client unavailable")
            return False
        md = self.market_data.get(pair, {})
        signal = md.get("signals", {}).get("combined", "hold")
        score = self.opportunity_scores.get(pair, 0.0)
        min_score = self.config.trading.min_opportunity_score
        if signal == "hold" or score < min_score:
            return False
        if signal != "buy":
            return False
        quantity = self.calculate_position_size(pair)
        if quantity is None or quantity <= Decimal("0"):
            self._log_skipped_trade(
                pair, signal, "Position sizing failed or resulted in zero quantity"
            )
            return False
        price = self._get_current_price_decimal(pair)
        if price is None:
            self._log_skipped_trade(pair, signal, "Could not determine current price for order")
            return False
        order = self.order_manager.create_order(pair, signal, "market", quantity, price)
        self.order_manager.submit_order(order)
        if order.status == OrderStatus.REJECTED:
            self.performance_metrics["failed_trades"] += 1
            return False
        self.performance_metrics["trades_executed"] += 1
        if signal == "buy":
            self.performance_metrics["buy_trades"] += 1
        return True

    @timed_operation
    def run_trading_cycle(self) -> None:
        """Runs a single trading cycle: data update, signal generation, execution, rebalancing."""
        logger.info("--- Starting Trading Cycle (Paper=%s) ---", self.paper_trading)
        cycle_start_time = time.time()
        try:
            logger.info("Step 1/7: Updating market data...")
            self.update_market_data_concurrently()
            logger.info("Step 2/7 & 3/7: Generating signals and calculating scores...")
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.config.bot.max_concurrent_pairs
            ) as executor:
                futures = [
                    executor.submit(self.generate_signals, pair) for pair in self.trading_pairs
                ]
                concurrent.futures.wait(futures)
                for future in futures:
                    if future.exception():
                        logger.error("Error in generate_signals task: %s", future.exception())
            logger.info("Step 4/7: Updating portfolio value...")
            self._update_portfolio_value()
            logger.info("Step 5/7: Executing trading decisions...")
            valid_score_pairs = [
                p for p in self.trading_pairs if self.opportunity_scores.get(p) is not None
            ]
            sorted_pairs = sorted(
                valid_score_pairs, key=lambda p: self.opportunity_scores.get(p, 0.0), reverse=True
            )
            executed_trade_count = 0
            for pair in sorted_pairs:
                try:
                    if self.execute_trading_decision(pair):
                        executed_trade_count += 1
                except Exception as exec_e:
                    logger.error(
                        "Unexpected error during decision execution for %s: %s",
                        pair,
                        exec_e,
                        exc_info=True,
                    )
            logger.info("Attempted %s bot trades this cycle.", executed_trade_count)
            logger.info("Step 6/7: Checking portfolio balance...")
            if self.portfolio_manager:
                try:
                    current_market_state = {}
                    for pair in self.trading_pairs:
                        if pair in self.market_data:
                            pair_data = self.market_data[pair]
                            state_component = {}
                            if pair_data.get("indicators"):
                                state_component["indicators"] = pair_data["indicators"]
                            if pair_data.get("sentiment_metrics"):
                                state_component["sentiment_metrics"] = pair_data[
                                    "sentiment_metrics"
                                ]
                            if pair_data.get("ai_metrics"):
                                state_component["ai_metrics"] = pair_data["ai_metrics"]
                            if pair_data.get("ohlcv"):
                                state_component["ohlcv"] = pair_data["ohlcv"]
                            if pair_data.get("ticker"):
                                state_component["ticker"] = pair_data["ticker"]
                            current_market_state[pair] = state_component
                        else:
                            logger.warning(
                                "Market data for %s missing, cannot include in state for PortfolioManager.",
                                pair,
                            )
                    rebalance_actions = self.portfolio_manager.determine_rebalance_actions(
                        current_market_state
                    )
                    if rebalance_actions and isinstance(rebalance_actions, list):
                        logger.info(
                            "Portfolio Manager simulated %s rebalance actions.",
                            len(rebalance_actions),
                        )
                        for action in rebalance_actions:
                            logger.debug("  Simulated Rebalance: %s", action)
                    else:
                        logger.info("No portfolio rebalance actions needed or simulated.")
                except Exception as e:
                    logger.error("Error during portfolio rebalancing check: %s", e, exc_info=True)
            else:
                logger.warning(
                    "Skipping portfolio rebalance check (Portfolio Manager not provided)."
                )
            logger.info("Step 7/7: Logging performance metrics...")
            self._log_performance_metrics()
            cycle_duration = time.time() - cycle_start_time
            logger.info("--- Trading Cycle Completed in %.2f seconds ---", cycle_duration)
            return True
        except Exception as e:
            logger.error("Critical error in trading cycle: %s", e, exc_info=True)
            return False

    def update_market_data_concurrently(self) -> None:
        """Updates market data for all trading pairs using a thread pool."""
        logger.info("Updating market data for %s pairs concurrently...", len(self.trading_pairs))
        futures = {
            self.thread_pool.submit(self.update_market_data, pair): pair
            for pair in self.trading_pairs
        }
        completed_count = 0
        start_time = time.time()
        for future in concurrent.futures.as_completed(futures):
            pair = futures[future]
            try:
                future.result()
                completed_count += 1
            except Exception as e:
                logger.error(
                    "Error updating market data for %s in thread: %s", pair, e, exc_info=True
                )
        duration = time.time() - start_time
        logger.info(
            "Concurrent market data update complete (%s/%s successful) in %.2fs.",
            completed_count,
            len(self.trading_pairs),
            duration,
        )

    def _update_portfolio_value(self) -> None:
        """Calculate and update the total portfolio value using PortfolioManager methods or direct exchange client."""
        if (
            self.portfolio_manager
            and hasattr(self.portfolio_manager, "get_current_holdings")
            and callable(getattr(self.portfolio_manager, "get_current_holdings"))
            and hasattr(self.portfolio_manager, "_calculate_current_values")
            and callable(getattr(self.portfolio_manager, "_calculate_current_values"))
        ):
            try:
                valuation_currency = self.config.get("bot.quote_currencies", ["USD"])[0].upper()
                logger.debug(
                    "Updating portfolio value in %s via PortfolioManager...", valuation_currency
                )
                holdings = self.portfolio_manager.get_current_holdings()
                if holdings is None or not isinstance(holdings, dict):
                    logger.error("Failed to retrieve valid holdings for value update (via PM).")
                    return
                (asset_values, total_value_dec) = self.portfolio_manager._calculate_current_values(
                    holdings
                )
                if total_value_dec is None:
                    logger.error("Failed to calculate total portfolio value (via PM).")
                    return
                self.total_portfolio_value = float(total_value_dec)
                logger.info(
                    "Updated total portfolio value (via PM): %.2f %s",
                    self.total_portfolio_value,
                    valuation_currency,
                )
            except Exception as e:
                logger.error(
                    "Unexpected error updating portfolio value via PM: %s", e, exc_info=True
                )
                return
        elif self.exchange_available:
            logger.warning(
                "Portfolio Manager unavailable/missing methods. Attempting direct balance valuation."
            )
            try:
                balance_data = self.exchange_client.get_balance()
                if balance_data is None:
                    logger.error("Direct balance valuation failed: Could not fetch balance.")
                    return
                quote_currency = self.config.get("bot.quote_currencies", ["USD"])[0].upper()
                total_value = Decimal("0.0")
                balances_dec = {}
                for asset, amount_float in balance_data.items():
                    try:
                        balances_dec[asset.upper()] = Decimal(str(amount_float))
                    except InvalidOperation:
                        logger.warning(
                            "Invalid balance value %s for %s. Skipping.", amount_float, asset
                        )
                for asset, amount_dec in balances_dec.items():
                    if amount_dec <= Decimal("1e-9"):
                        continue
                    if asset == quote_currency:
                        total_value += amount_dec
                    else:
                        pair = f"{asset}/{quote_currency}"
                        price = self._get_current_price(pair)
                        if price is not None and price > 1e-09:
                            try:
                                total_value += amount_dec * Decimal(str(price))
                            except InvalidOperation:
                                logger.warning(
                                    "Invalid price %s for %s. Skipping value.", price, pair
                                )
                        else:
                            logger.warning("Could not get price for %s. Skipping value.", pair)
                self.total_portfolio_value = float(total_value)
                logger.info(
                    "Updated total portfolio value (Direct): %.2f %s",
                    self.total_portfolio_value,
                    quote_currency,
                )
            except Exception as e:
                logger.error("Error during direct portfolio value update: %s", e, exc_info=True)
                return
        else:
            logger.error(
                "Cannot update portfolio value: Exchange client and Portfolio Manager unavailable/invalid."
            )
            return
        if (
            self.performance_metrics["start_portfolio_value"] == 0
            and self.total_portfolio_value > 0
        ):
            self.performance_metrics["start_portfolio_value"] = self.total_portfolio_value
        self.performance_metrics["current_portfolio_value"] = self.total_portfolio_value

    def _log_performance_metrics(self) -> None:
        """Logs current performance metrics and saves them atomically."""
        start = self.performance_metrics.get("start_portfolio_value", 0.0)
        current = self.performance_metrics.get("current_portfolio_value", 0.0)
        pl = 0.0
        pl_pct = 0.0
        if (
            isinstance(start, (int, float))
            and isinstance(current, (int, float))
            and (start > 1e-09)
        ):
            pl = current - start
            pl_pct = pl / start * 100
        self.performance_metrics["total_profit_loss"] = pl
        self.performance_metrics["profit_loss_pct"] = pl_pct
        metrics_log = f"Performance: Value=${current:.2f}, P/L=${pl:+.2f} ({pl_pct:+.2f}%), TradesExec={self.performance_metrics.get('trades_executed', 0)} (Success:{self.performance_metrics.get('successful_trades', 0)}/Fail:{self.performance_metrics.get('failed_trades', 0)})"
        logger.info(metrics_log)
        fname = f"performance_{datetime.now().strftime('%Y%m%d')}.json"
        mfile = self.metrics_dir / fname
        temp_file = mfile.with_suffix(f".tmp_{time.time_ns()}")
        try:
            self.metrics_dir.mkdir(parents=True, exist_ok=True)
            data_to_save = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metrics": self.performance_metrics,
            }
            log_list: list = []
            if mfile.exists():
                try:
                    with mfile.open("r", encoding="utf-8") as f:
                        content = f.read()
                        if content.strip():
                            loaded_list = json.loads(content)
                            if isinstance(loaded_list, list):
                                log_list = loaded_list
                except json.JSONDecodeError:
                    log_list: list = []
                except OSError:
                    log_list: list = []
            log_list.append(data_to_save)
            with temp_file.open("w", encoding="utf-8") as f:
                json.dump(log_list, f, indent=2)
            temp_file.replace(mfile)
        except (OSError, TypeError, ValueError) as e:
            logger.error("Failed to save performance metrics: %s", e, exc_info=True)
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass

    def start(self, num_cycles: Optional[int], interval: int) -> None:
        """Starts the main trading loop."""
        if self.running:
            logger.warning("Bot is already running.")
            return
        self.running = True
        cycle_count = 0
        max_cycles = num_cycles if num_cycles is not None else float("inf")
        logger.info(
            "Starting bot: cycles=%s, interval=%ss",
            "infinite" if max_cycles == float("inf") else num_cycles,
            interval,
        )
        self._test_connections()
        logger.info("Performing initial portfolio value update...")
        self._update_portfolio_value()
        if self.performance_metrics["start_portfolio_value"] == 0:
            logger.warning("Initial portfolio value is zero.")
        try:
            while self.running and cycle_count < max_cycles:
                cycle_start_time = time.time()
                logger.info(
                    "--- Starting Cycle %s/%s ---",
                    cycle_count + 1,
                    "Inf" if max_cycles == float("inf") else max_cycles,
                )
                success = self.run_trading_cycle()
                if success:
                    cycle_count += 1
                else:
                    logger.warning("Trading cycle failed. Continuing...")
                if not self.running:
                    break
                if not self.safety_guard.can_trade():
                    logger.critical(
                        "HALTING TRADING due to safety guard: %s",
                        self.safety_guard.circuit_breaker_reason,
                    )
                    self.stop()
                    break
                if self.market_data:
                    sample_pair = next(iter(self.market_data), None)
                    if sample_pair and self.market_data[sample_pair].get("ticker"):
                        ticker_time_str = self.market_data[sample_pair]["ticker"].get("datetime")
                        if ticker_time_str:
                            try:
                                ticker_dt = datetime.fromisoformat(
                                    ticker_time_str.replace("Z", "+00:00")
                                )
                                self.safety_guard.check_clock_drift(ticker_dt.timestamp() * 1000)
                            except (ValueError, TypeError):
                                pass
                realized_pnl_update = Decimal("0")
                unrealized_pnl_update = Decimal("0")
                self.safety_guard.update_pnl(realized_pnl_update, unrealized_pnl_update)
                self.safety_guard.check_daily_loss_limit()
                duration = time.time() - cycle_start_time
                sleep = max(0, interval - duration)
                if sleep > 0:
                    logger.info("Sleeping for %.2f seconds...", sleep)
                    for _ in range(int(sleep)):
                        if not self.running:
                            break
                        time.sleep(1)
                    if self.running:
                        time.sleep(sleep % 1)
                else:
                    logger.info(
                        "Cycle duration (%.2fs) exceeded interval (%ss). Starting next cycle immediately.",
                        duration,
                        interval,
                    )
                if not self.running:
                    break
            logger.info("Trading bot stopping after %s cycles.", cycle_count)
            realized_pnl_update = Decimal("0")
            unrealized_pnl_update = Decimal("0")
            self.safety_guard.update_pnl(realized_pnl_update, unrealized_pnl_update)
            self.safety_guard.check_daily_loss_limit()
            logger.info("Trading bot stopping after %s cycles.", cycle_count)
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received. Stopping...")
        except Exception as e:
            logger.critical("Unhandled critical error in main loop: %s", e, exc_info=True)
        finally:
            self.running = False
            self._cleanup()

    def _test_connections(self) -> None:
        """Tests connections to essential external services."""
        logger.info("Performing dependency connection tests...")
        all_ok = True
        if self.exchange_client:
            try:
                (success, details) = self.exchange_client.test_connection()
                status_msg = (
                    "OK" if success else f"FAILED ({details.get('error', 'Unknown reason')})"
                )
                logger.info(" - Exchange (%s): %s", type(self.exchange_client).__name__, status_msg)
                if not success:
                    all_ok = False
            except Exception as e:
                logger.error(
                    " - Exchange (%s): Test Exception: %s",
                    type(self.exchange_client).__name__,
                    e,
                    exc_info=True,
                )
                all_ok = False
        else:
            logger.warning(" - Exchange: Not provided.")
            all_ok = False
        if self.sentiment_sources:
            for source in self.sentiment_sources:
                source_name = type(source).__name__
                try:
                    (success, details) = source.test_connection()
                    status_msg = "OK" if success else f"FAILED ({details})"
                    logger.info(" - Sentiment (%s): %s", source_name, status_msg)
                except Exception as e:
                    logger.error(
                        " - Sentiment (%s): Test Exception: %s", source_name, e, exc_info=True
                    )
        else:
            logger.info(" - Sentiment: None provided.")
        if self.ai_analyzer:
            source_name = type(self.ai_analyzer).__name__
            try:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                if loop.is_running():
                    logger.warning(
                        "Cannot run async test_connection easily within running loop. Skipping AI test."
                    )
                else:
                    (success, details) = loop.run_until_complete(self.ai_analyzer.test_connection())
                    status_msg = (
                        "OK" if success else f"FAILED ({details.get('error', 'Unknown reason')})"
                    )
                    logger.info(" - AI Analyzer (%s): %s", source_name, status_msg)
            except RuntimeError as rt_err:
                logger.error(
                    " - AI Analyzer (%s): Event loop error during test: %s", source_name, rt_err
                )
            except Exception as e:
                logger.error(
                    " - AI Analyzer (%s): Test Exception: %s", source_name, e, exc_info=True
                )
        else:
            logger.info(" - AI Analyzer: Not provided.")
        if not all_ok:
            logger.critical(
                "One or more critical dependencies (Exchange) failed connection tests. Bot may not function correctly."
            )
        else:
            logger.info("All critical dependency connections seem OK.")

    def _cleanup(self) -> None:
        """Performs cleanup actions before the bot exits."""
        logger.info("Initiating cleanup...")
        if hasattr(self, "thread_pool") and self.thread_pool:
            logger.debug("Shutting down thread pool...")
            try:
                self.thread_pool.shutdown(wait=True, cancel_futures=False)
                logger.info("Thread pool shut down.")
            except Exception as e:
                logger.error("Error shutting down thread pool: %s", e, exc_info=True)
        self._log_performance_metrics()
        logger.info("Cleanup finished.")

    def stop(self) -> None:
        """Signals the bot to stop the main loop."""
        if self.running:
            logger.info("Stop signal received...")
            self.running = False
        else:
            logger.info("Bot already stopped.")


def parse_arguments() -> None:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Crypto Trading Bot")
    parser.add_argument("--paper", action="store_true", help="Force paper trading mode.")
    parser.add_argument("--live", action="store_true", help="Force live trading mode.")
    parser.add_argument("--cycles", type=int, default=None, help="Number of cycles to run.")
    parser.add_argument(
        "--interval", type=int, default=None, help="Interval between cycles (seconds)."
    )
    parser.add_argument("--config", type=str, default=None, help="Path to custom user_config.json.")
    return parser.parse_args()


def main() -> None:
    """Initializes dependencies and starts the trading bot."""
    args = parse_arguments()
    try:
        config = BotConfig(config_path=args.config)
    except Exception as e:
        import logging
        import sys

        logging.critical(f"FATAL: Failed to initialize BotConfig: {e}", exc_info=True)
        sys.exit(1)
    initial_paper_mode = config.bot.paper_trading
    if args.paper and args.live:
        logger.error("Cannot specify both --paper and --live. Using config setting.")
    elif args.paper:
        config.settings["bot"]["paper_trading"] = True
        logger.info("CLI Override: Forcing PAPER trading.")
    elif args.live:
        config.settings["bot"]["paper_trading"] = False
        logger.info("CLI Override: Forcing LIVE trading.")
    config.bot.paper_trading = config.settings["bot"]["paper_trading"]
    if initial_paper_mode != config.bot.paper_trading:
        logger.warning(
            "Trading mode changed by CLI to %s.", "PAPER" if config.bot.paper_trading else "LIVE"
        )
    cycles_to_run = args.cycles if args.cycles is not None else config.get("bot.max_cycles", None)
    interval_seconds = (
        args.interval
        if args.interval is not None
        else config.get("bot.trading_interval_seconds", 300)
    )
    if not isinstance(interval_seconds, int) or interval_seconds <= 0:
        logger.warning("Invalid interval (%s). Using 60s.", interval_seconds)
        interval_seconds = 60
    exchange_client = None
    portfolio_manager = None
    sentiment_sources: list = []
    ai_analyzer = None
    sentiment_tracker = None
    try:
        from core.portfolio_manager import PortfolioManager
        from integrations.ai.openai import OpenAIAPI
        from integrations.data.news import NewsAPI
        from integrations.data.reddit import RedditAPI
        from integrations.exchange.kraken import KrakenClient
        from utils.sentiment_utils import SentimentTracker

        try:
            exchange_client = KrakenClient(config=config)
            logger.info("Instantiated Exchange Client: %s", type(exchange_client).__name__)
        except Exception as e:
            logger.critical("CRITICAL: Failed Exchange Client init: %s", e, exc_info=True)
            import sys

            sys.exit(1)
        try:
            sentiment_tracker = SentimentTracker(
                history_length=config.get("sentiment.history_length", 30)
            )
        except Exception as e:
            logger.error("Failed Sentiment Tracker init: %s", e, exc_info=True)
            sentiment_tracker = None
        if config.get("api_credentials.reddit.client_id"):
            try:
                reddit_api = RedditAPI(config=config)
                if reddit_api.reddit:
                    sentiment_sources.append(reddit_api)
                    logger.info("RedditAPI added as sentiment source.")
                else:
                    logger.error(
                        "RedditAPI initialization failed (PRAW client None), source skipped."
                    )
            except Exception as e:
                logger.error("Failed RedditAPI init: %s", e, exc_info=True)
        else:
            logger.info("Skipping RedditAPI (no client_id in config).")
        if config.get("api_credentials.news.api_key"):
            try:
                news_api = NewsAPI(config=config)
                if news_api.api_key:
                    sentiment_sources.append(news_api)
                    logger.info("NewsAPI added as sentiment source.")
                else:
                    logger.error(
                        "NewsAPI initialization failed (API key likely missing/invalid), source skipped."
                    )
            except Exception as e:
                logger.error("Failed NewsAPI init: %s", e, exc_info=True)
        else:
            logger.info("Skipping NewsAPI (no api_key in config).")
        if config.get("api_credentials.openai.api_key") and sentiment_tracker:
            try:
                ai_analyzer = OpenAIAPI(config=config, sentiment_tracker=sentiment_tracker)
                if ai_analyzer.client:
                    logger.info("OpenAIAPI added as AI analyzer.")
                else:
                    logger.error(
                        "OpenAIAPI initialization failed (client None), analyzer unavailable."
                    )
                    ai_analyzer = None
            except Exception as e:
                logger.error("Failed OpenAIAPI init: %s", e, exc_info=True)
        elif not sentiment_tracker:
            logger.warning("Skipping OpenAI Analyzer (Sentiment Tracker unavailable).")
        else:
            logger.info("Skipping OpenAI Analyzer (no api_key in config).")
        try:
            portfolio_manager = PortfolioManager(config=config, exchange_client=exchange_client)
            logger.info("PortfolioManager instantiated.")
        except Exception as e:
            logger.critical("CRITICAL: Failed Portfolio Manager init: %s", e, exc_info=True)
            import sys

            sys.exit(1)
        bot = MultiCryptoTradingBot(
            config=config,
            exchange_client=exchange_client,
            portfolio_manager=portfolio_manager,
            sentiment_sources=sentiment_sources,
            ai_analyzer=ai_analyzer,
        )
        logger.info("MultiCryptoTradingBot instantiated.")
        bot.start(num_cycles=cycles_to_run, interval=interval_seconds)
    except Exception as e:
        logger.critical(
            "CRITICAL: Failed bot setup or unexpected runtime error: %s", e, exc_info=True
        )
        import sys

        sys.exit(1)


if __name__ == "__main__":
    main()
