"""
Integration module for interacting with the OpenAI API, using centralized configuration.

Uses the `openai` library (async client) to generate market sentiment analysis,
trading strategies, and technical indicator interpretations based on provided data.
Includes caching, rate limiting via backoff, controlled concurrency, and JSON
response parsing. Relies on common sentiment utilities for tracking and processing.
"""

import asyncio
import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import backoff

try:
    from core.config import BotConfig

    BotConfigType: Any = BotConfig
except ImportError:
    import logging

    logging.error("CRITICAL: Failed to import BotConfig. OpenAIAPI may not function.")
    BotConfigType = None
try:
    from openai import (APIConnectionError, APIError, APITimeoutError,
                        AsyncOpenAI, AuthenticationError, BadRequestError,
                        OpenAIError, RateLimitError)

    openai_available = True
except ImportError:
    print(
        "WARNING: OpenAI library (v1.x+) not found or import failed. AI integration disabled. Run: pip install --upgrade openai"
    )
    openai_available = False
    AsyncOpenAI = type("AsyncOpenAI", (), {})
    OpenAIError = type("OpenAIError", (Exception,), {})
    APIError = type("APIError", (Exception,), {})
    RateLimitError = type("RateLimitError", (Exception,), {})
    AuthenticationError = type("AuthenticationError", (Exception,), {})
    BadRequestError = type("BadRequestError", (Exception,), {})
    APITimeoutError = type("APITimeoutError", (Exception,), {})
    APIConnectionError = type("APIConnectionError", (Exception,), {})
try:
    from common.common_logger import CACHE_DIR, get_logger

    logger = get_logger("openai_api")
except Exception as e:
    import logging

    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger("openai_api_fallback")
    logger.error("Failed to get central logger: %s. Using basic fallback logger.", e)
    CACHE_DIR = Path(".") / "data" / "cache"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
try:
    from core.interfaces import IAIAnalyzer

    IAIAnalyzerType: Any = IAIAnalyzer
except ImportError:
    logger.error("Failed to import IAIAnalyzer interface.")
    IAIAnalyzerType = object
try:
    from utils.sentiment_utils import (SentimentTracker,
                                       convert_recommendation_to_sentiment,
                                       extract_confidence,
                                       normalize_sentiment_value)

    sentiment_utils_available = True
except ImportError:
    print(
        "ERROR: Failed to import sentiment utilities from utils.sentiment_utils. OpenAI analysis may be impaired."
    )
    sentiment_utils_available = False
    SentimentTracker = type("SentimentTracker", (), {})  # type: ignore[misc]

    def normalize_sentiment_value(s: Any) -> float:
        return 0.0

    def convert_recommendation_to_sentiment(r: Any) -> float:
        return 0.0

    def extract_confidence(c: Any) -> float:
        return 0.5


logger = get_logger("openai_api")


class PromptManager:
    """Manages prompts for different OpenAI analysis tasks."""

    @staticmethod
    def get_sentiment_prompt(
        symbol: str,
        market_data: Dict,
        reddit_score: Union[float, str],
        news_data: Optional[List[Dict]] = None,
    ) -> str:
        market_data_subset = {
            "symbol": market_data.get("symbol"),
            "current_price": market_data.get("current_price"),
            "indicators": market_data.get("indicators", {}),
            "sentiment_scores": market_data.get("sentiment_scores", {}),
        }
        try:
            market_data_str = json.dumps(
                market_data_subset, indent=2, default=str, ensure_ascii=False
            )
            _MAX_LEN = 2000
        except (TypeError, ValueError) as json_err:
            market_data_str = f"[Error serializing market data: {json_err}]"
            logger.warning(json_err)
        if len(market_data_str) > _MAX_LEN:
            market_data_str = market_data_str[:_MAX_LEN] + "\n... (truncated)"
        prompt = f"Analyze market sentiment for {symbol}. Respond ONLY in JSON.\n\nData:\n{market_data_str}\n\nReddit Score: {reddit_score}\n"
        if news_data and isinstance(news_data, list):
            news_titles = [
                f"- {n.get('title', 'N/A')}" for n in news_data[:3] if isinstance(n, dict)
            ]
            if news_titles:
                prompt += "\nRecent News Titles:\n" + "\n".join(news_titles) + "\n"
        prompt += '\nInstructions:\n- Overall sentiment (Bullish/Bearish/Neutral).\n- Key factors summary (concise).\n- Simple recommendation (\'buy\'/\'sell\'/\'hold\').\n- Confidence score (integer 1-10).\n\nJSON Output:\n{\n  "symbol": "string",\n  "sentiment": "string",\n  "summary": "string",\n  "recommendation": "string",\n  "confidence": integer\n}'
        return prompt

    @staticmethod
    def get_strategy_prompt(symbol: str, market_data: Dict, risk_profile: str = "moderate") -> str:
        market_data_subset = {
            "symbol": market_data.get("symbol"),
            "current_price": market_data.get("current_price"),
            "indicators": market_data.get("indicators", {}),
            "sentiment_metrics": market_data.get("sentiment_metrics", {}),
        }
        try:
            market_data_str = json.dumps(
                market_data_subset, indent=2, default=str, ensure_ascii=False
            )
            _MAX_LEN = 2000
        except (TypeError, ValueError) as json_err:
            market_data_str = f"[Error serializing market data: {json_err}]"
            logger.warning(json_err)
        if len(market_data_str) > _MAX_LEN:
            market_data_str = market_data_str[:_MAX_LEN] + "\n... (truncated)"
        prompt = f"""Generate a {risk_profile}-risk trading strategy for {symbol}. Respond ONLY in JSON.\n\nData:\n{market_data_str}\n\nInstructions:\n- Entry/exit/stop-loss price levels or conditions.\n- Take-profit price level or condition.\n- Position size suggestion (e.g., '% portfolio', 'standard unit').\n- Key indicators to monitor.\n- Suitable timeframe (e.g., '1h', '4h', '1d').\n- Concise rationale.\n\nJSON Output:\n{{\n  "symbol": "string",\n  "entry_point": "string",\n  "exit_point": "string",\n  "stop_loss": "string",\n  "take_profit": "string",\n  "position_size": "string",\n  "indicators_to_watch": ["string"],\n  "timeframe": "string",\n  "notes": "string"\n}}"""
        return prompt

    @staticmethod
    def get_technical_prompt(indicator_data: Dict[str, Any]) -> str:
        if not isinstance(indicator_data, dict):
            return "[Error: Invalid indicator data format]"
        _MAX_LEN = 2000
        indicator_data_str: str = ""
        try:
            indicator_data_str = json.dumps(
                indicator_data, indent=2, default=str, ensure_ascii=False
            )
        except (TypeError, ValueError) as json_err:
            indicator_data_str = f"[Error serializing indicator data: {json_err}]"
            logger.warning(str(json_err))
        if indicator_data_str and len(indicator_data_str) > _MAX_LEN:
            indicator_data_str = indicator_data_str[:_MAX_LEN] + "\n... (truncated)"
        prompt = f'Analyze technical indicators for a crypto asset. Provide signal (buy/sell/hold), identify conflicting signals, rate confidence (1-10). Respond ONLY in JSON.\n\nIndicators:\n{indicator_data_str}\n\nJSON Output:\n{{\n  "indicator_interpretations": {{ "IndicatorName": "interpretation string" }},\n  "overall_signal": "string",\n  "conflicts": ["string"],\n  "confidence": integer\n}}'
        return prompt

    @staticmethod
    def get_system_prompt(task_type: str) -> str:
        prompts = {
            "sentiment": "You are a concise crypto market sentiment analyst for an automated trading bot. Analyze the provided market data, news headlines, and social media scores. Focus on identifying the prevailing sentiment (Bullish, Bearish, Neutral) and provide a confidence level. Respond ONLY with the requested JSON structure.",
            "strategy": "You are a quantitative trading strategist designing plans for an automated bot. Based on the market data and risk profile, generate a structured, actionable short-term trading plan (entry, exit, stop-loss, size suggestion, key indicators, timeframe, rationale). Respond ONLY with the requested JSON structure.",
            "technical": "You are a technical analyst interpreting indicator data for an automated trading bot. Analyze the provided technical indicator values. Determine an overall signal (buy/sell/hold), identify any conflicting indicators, and provide a confidence score. Respond ONLY with the requested JSON structure.",
        }
        return prompts.get(task_type, prompts["sentiment"])


class OpenAIAPI(IAIAnalyzerType):
    """Async client for OpenAI API calls using BotConfig."""

    def __init__(self, config: Any, sentiment_tracker: Any) -> None:
        """
        Initializes the AsyncOpenAI client using BotConfig.

        Args:
            config (BotConfig): The central configuration object.
            sentiment_tracker (SentimentTracker): An instance for tracking sentiment history.
        """
        if BotConfigType is None:
            raise ValueError("BotConfig is not available.")
        if not sentiment_utils_available:
            raise ImportError("Sentiment utilities could not be imported.")
        if not openai_available:
            raise ImportError("OpenAI library not installed.")
        self.main_config = config
        self.client_config = config.openai_client
        self.api_key = config.get("api_credentials.openai.api_key")
        self.sentiment_tracker = sentiment_tracker
        self.client: Optional[AsyncOpenAI] = None
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.rate_limited_until: Optional[datetime] = None
        self.settings = {
            "model": getattr(self.client_config, "model", "gpt-4o-mini"),
            "fallback_model": getattr(self.client_config, "fallback_model", "gpt-3.5-turbo"),
            "temp_analysis": getattr(
                getattr(self.client_config, "temperature", {}), "analysis", 0.3
            ),
            "temp_strategy": getattr(
                getattr(self.client_config, "temperature", {}), "strategy", 0.3
            ),
            "temp_technical": getattr(
                getattr(self.client_config, "temperature", {}), "technical", 0.3
            ),
            "max_retries": getattr(self.client_config, "max_retries", 3),
            "cache_expiry": getattr(self.client_config, "cache_expiry_seconds", 900),
            "rate_limit_reset": 60,
            "max_tokens_analysis": getattr(
                getattr(self.client_config, "max_tokens", {}), "analysis", 700
            ),
            "max_tokens_strategy": getattr(
                getattr(self.client_config, "max_tokens", {}), "strategy", 700
            ),
            "max_tokens_technical": getattr(
                getattr(self.client_config, "max_tokens", {}), "technical", 700
            ),
            "request_timeout": getattr(self.client_config, "request_timeout_seconds", 60),
            "concurrency_limit": getattr(self.client_config, "concurrency_limit", 3),
        }
        try:
            self.settings["concurrency_limit"] = max(1, int(self.settings["concurrency_limit"]))
        except (ValueError, TypeError):
            logger.warning("Invalid openai_client.concurrency_limit value. Using default 3.")
            self.settings["concurrency_limit"] = 3
        if not self.api_key:
            logger.error("Cannot initialize OpenAIAPI: API key missing or empty in config.")
            return
        try:
            timeout_val = float(self.settings["request_timeout"])
            self.client = AsyncOpenAI(api_key=self.api_key, timeout=timeout_val)
            logger.info(
                "AsyncOpenAI client initialized. Model: %s, Timeout: %ss, Concurrency: %s",
                self.settings["model"],
                timeout_val,
                self.settings["concurrency_limit"],
            )
        except (ValueError, TypeError) as e:
            logger.error(
                "Invalid timeout value for OpenAI client: %s. Error: %s",
                self.settings["request_timeout"],
                e,
            )
            self.client = None
        except Exception as e:
            logger.error("Unexpected error initializing OpenAI client: %s", e, exc_info=True)
            self.client = None

    def _check_client(self) -> bool:
        """Check if the OpenAI client was initialized successfully."""
        if not self.client:
            return False
        return True

    def _check_cache(self, cache_key: str) -> Optional[Any]:
        """Retrieves data from cache if valid and not expired."""
        entry = self.cache.get(cache_key)
        if entry is None or not isinstance(entry, dict):
            return None
        now = time.time()
        expiry = entry.get("timestamp", 0) + self.settings["cache_expiry"]
        if now < expiry:
            logger.debug("OpenAI Cache HIT: %s", cache_key)
            return entry.get("data")
        logger.debug("OpenAI Cache MISS/EXPIRED: %s", cache_key)
        self.cache.pop(cache_key, None)
        return None

    def _update_cache(self, cache_key: str, data: Any) -> None:
        """Updates the cache, managing its size."""
        if data is None:
            return
        self.cache[cache_key] = {"timestamp": time.time(), "data": data}
        if len(self.cache) > 100:
            try:
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k].get("timestamp", 0))
                self.cache.pop(oldest_key, None)
                logger.debug("Cleaned OpenAI cache (removed oldest entry)")
            except ValueError:
                pass
            except Exception as e:
                logger.warning("Error during OpenAI cache cleanup: %s", e)

    def _get_cache_key(self, method_name: str, **kwargs: Any) -> str:
        """Generates a unique cache key based on method and arguments."""
        try:
            params_str = json.dumps(kwargs, sort_keys=True, default=str)
            key_hash = hashlib.md5(params_str.encode("utf-8")).hexdigest()
            return f"openai:{method_name}:{key_hash}"
        except (TypeError, ValueError) as e:
            logger.warning("Failed OpenAI cache key gen (%s): %s. Using fallback.", method_name, e)
            fallback_args_str = str(kwargs)[:100]
            return f"openai:{method_name}:{fallback_args_str}"

    async def _handle_rate_limiting(self) -> None:
        """Async wait if internal rate limit flag is set."""
        if self.rate_limited_until and datetime.now(timezone.utc) < self.rate_limited_until:
            wait = (self.rate_limited_until - datetime.now(timezone.utc)).total_seconds()
            if wait > 0:
                logger.warning("OpenAI internal rate limit active, waiting %.1fs...", wait)
                await asyncio.sleep(wait)
            self.rate_limited_until = None

    @staticmethod
    def _is_retryable_openai_error(e: Exception) -> bool:
        return isinstance(e, (RateLimitError, APITimeoutError, APIConnectionError, APIError)) and (
            not isinstance(e, (AuthenticationError, BadRequestError))
        )

    async def _call_openai_api(
        self, model: Any, messages: List[Dict[str, Any]], temperature: Any, max_tokens: Any
    ) -> Optional[Any]:
        """Async call to OpenAI ChatCompletion using config, with backoff and error handling."""
        if not self._check_client():
            return None
        max_retry_time_seconds = 60

        @backoff.on_exception(
            backoff.expo,
            (RateLimitError, APITimeoutError, APIConnectionError, APIError),
            max_tries=int(self.settings["max_retries"]) + 1,
            max_time=max_retry_time_seconds,
            giveup=lambda e: not OpenAIAPI._is_retryable_openai_error(e),
            logger=None,
            on_backoff=lambda d: logger.warning(
                "OpenAI Backoff: Wait %.1fs after %s tries (%s model=%s) (%s)",
                d["wait"],
                d["tries"],
                d["target"].__name__,
                model,
                type(d["exception"]).__name__,
            ),
            on_giveup=lambda d: logger.error(
                "OpenAI Giveup: %s after %s tries (elapsed %.1fs > max_time %ss?) (%s)",
                d["target"].__name__,
                d["tries"],
                d["elapsed"],
                max_retry_time_seconds,
                type(d["exception"]).__name__,
            ),
        )
        async def make_api_call_with_backoff() -> Any:
            await self._handle_rate_limiting()
            try:
                logger.debug(
                    "Calling OpenAI API: model=%s, temp=%s, tokens=%s",
                    model,
                    temperature,
                    max_tokens,
                )
                if self.client is None:
                    raise OpenAIError("OpenAI client is not initialized")
                response = await self.client.chat.completions.create(
                    model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
                )
                usage = getattr(response, "usage", None)
                if usage:
                    logger.debug("OpenAI Usage: %s", usage)
                return response
            except RateLimitError as rle:
                reset_seconds = int(self.settings["rate_limit_reset"])
                self.rate_limited_until = datetime.now(timezone.utc) + timedelta(
                    seconds=reset_seconds
                )
                logger.warning(
                    "OpenAI RateLimitError encountered: %s. Setting internal wait flag for %ss.",
                    rle,
                    reset_seconds,
                )
                raise rle
            except AuthenticationError as ae:
                logger.critical("OpenAI Authentication Error: %s", ae)
                raise
            except BadRequestError as bre:
                logger.error("OpenAI Bad Request Error (check prompt/input): %s", bre)
                raise
            except APITimeoutError as te:
                logger.warning("OpenAI API Timeout Error: %s", te)
                raise
            except APIConnectionError as ce:
                logger.warning("OpenAI API Connection Error: %s", ce)
                raise
            except APIError as apie:
                logger.warning("OpenAI API Error (potentially temporary): %s", apie)
                raise
            except OpenAIError as oe:
                logger.error("OpenAI Library Error: %s", oe, exc_info=True)
                raise
            except Exception as exc:
                logger.error("OpenAI Unexpected Error during API call: %s", exc, exc_info=True)
                raise OpenAIError(f"Unexpected error: {exc}") from exc

        try:
            return await make_api_call_with_backoff()
        except Exception as final_error:
            logger.error(
                "OpenAI API call failed permanently for model %s after retries: %s - %s",
                model,
                type(final_error).__name__,
                final_error,
            )
            if model != self.settings["fallback_model"] and OpenAIAPI._is_retryable_openai_error(
                final_error
            ):
                logger.warning(
                    "Attempting OpenAI fallback model: %s", self.settings["fallback_model"]
                )
                try:
                    await self._handle_rate_limiting()
                    if self.client is None:
                        return None
                    fallback_response = await self.client.chat.completions.create(
                        model=self.settings["fallback_model"],
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    return fallback_response
                except Exception as fallback_error:
                    logger.error(
                        "OpenAI fallback model call failed: %s - %s",
                        type(fallback_error).__name__,
                        fallback_error,
                    )
                    return None
            else:
                return None

    async def test_connection(self, force_refresh: bool = False) -> Tuple[bool, Dict]:
        """Async test OpenAI API connection."""
        if not self._check_client():
            return (False, {"error": "Client not initialized"})
        cache_key = self._get_cache_key("test_connection")
        if not force_refresh:
            cached = self._check_cache(cache_key)
            if cached and isinstance(cached, dict) and ("status" in cached):
                return (True, cached)
        logger.info("Testing OpenAI connection (listing models)...")
        try:
            if self.client is None:
                raise OpenAIError("OpenAI client is not initialized")
            models_response = await self.client.models.list()
            models = models_response.data if hasattr(models_response, "data") else []
            model_names = [m.id for m in models if hasattr(m, "id")]
            default_ok = self.settings["model"] in model_names
            fallback_ok = self.settings["fallback_model"] in model_names
            result = {
                "status": "OK",
                "models_count": len(model_names),
                "default_model_ok": default_ok,
                "fallback_model_ok": fallback_ok,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self._update_cache(cache_key, result)
            logger.info(
                "OpenAI connection OK. Default OK: %s, Fallback OK: %s", default_ok, fallback_ok
            )
            return (True, result)
        except AuthenticationError as e:
            logger.error("OpenAI Auth Error during test: %s", e)
            return (False, {"status": "Auth Error", "error": str(e)})
        except APIConnectionError as e:
            logger.error("OpenAI Connection Error during test: %s", e)
            return (False, {"status": "Connection Error", "error": str(e)})
        except OpenAIError as e:
            logger.error("OpenAI API Error during test: %s", e)
            return (False, {"status": "API Error", "error": str(e)})
        except Exception as e:
            logger.error("OpenAI connection test failed unexpectedly: %s", e, exc_info=True)
            return (False, {"status": "Failed", "error": f"Unexpected error: {e}"})

    async def analyze_market_sentiment(
        self,
        market_data: Dict[str, Any],
        news_data: Optional[List[Dict[str, Any]]] = None,
        force_refresh: bool = False,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Async analyze market sentiment using config settings."""
        default_error_response: Dict[str, Any] = {
            "sentiment": "neutral",
            "strength": 0.0,
            "confidence": 0.0,
            "error": None,
        }
        if not self._check_client():
            default_error_response["error"] = "Client not initialized"
            return (False, default_error_response)
        if not isinstance(market_data, dict):
            default_error_response["error"] = "Invalid market_data input"
            return (False, default_error_response)
        symbol = market_data.get("symbol", "UNKNOWN")
        cache_key = self._get_cache_key(
            "analyze_market_sentiment", symbol=symbol, market_data=market_data, news_data=news_data
        )
        if not force_refresh:
            cached = self._check_cache(cache_key)
            if cached and isinstance(cached, dict):
                return (True, cached)
        try:
            reddit_score_val = market_data.get("sentiment_scores", {}).get("reddit", 0.0)
            reddit_score_str = (
                f"{reddit_score_val:.2f}"
                if isinstance(reddit_score_val, (float, int))
                else str(reddit_score_val)
            )
            system_prompt = PromptManager.get_system_prompt("sentiment")
            user_prompt = PromptManager.get_sentiment_prompt(
                symbol, market_data, reddit_score_str, news_data
            )
            model_to_use = self.settings["model"]
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            response = await self._call_openai_api(
                model=model_to_use,
                messages=messages,
                temperature=self.settings["temp_analysis"],
                max_tokens=self.settings["max_tokens_analysis"],
            )
            if response is None or not hasattr(response, "choices") or (not response.choices):
                error_msg = "No valid response from OpenAI API (sentiment)"
                logger.error(error_msg)
                default_error_response["error"] = error_msg
                return (False, default_error_response)
            message = response.choices[0].message
            content = getattr(message, "content", None)
            if content is None:
                error_msg = "OpenAI response message content is missing (sentiment)"
                logger.error(error_msg)
                default_error_response["error"] = error_msg
                return (False, default_error_response)
            analysis_json = self._parse_json_response(content.strip(), "sentiment", symbol)
            if analysis_json is None or not isinstance(analysis_json, dict):
                default_error_response["error"] = (
                    "Failed to parse valid JSON from OpenAI response (sentiment)"
                )
                return (False, default_error_response)
            sentiment_label = analysis_json.get("sentiment", "neutral")
            recommendation = analysis_json.get("recommendation", "hold")
            confidence_raw = analysis_json.get("confidence")
            sentiment_value = normalize_sentiment_value(sentiment_label)
            rec_value = convert_recommendation_to_sentiment(recommendation)
            combined = sentiment_value * 0.7 + rec_value * 0.3
            confidence = extract_confidence(confidence_raw)
            standardized_result = {
                "sentiment": (
                    sentiment_label.lower() if isinstance(sentiment_label, str) else "neutral"
                ),
                "strength": combined,
                "confidence": confidence,
                "summary": analysis_json.get("summary", ""),
                "recommendation": (
                    recommendation.lower() if isinstance(recommendation, str) else "hold"
                ),
                "raw_analysis": analysis_json,
            }
            if symbol != "UNKNOWN" and self.sentiment_tracker:
                try:
                    self.sentiment_tracker.add_sentiment(symbol, combined, confidence)
                except Exception as track_e:
                    logger.warning(
                        "Failed to add AI sentiment to tracker for %s: %s", symbol, track_e
                    )
            try:
                log_openai_analysis(
                    symbol=symbol if symbol != "UNKNOWN" else None,
                    analysis_type="sentiment",
                    prompt=user_prompt,
                    response_data=response,
                    result_metadata=standardized_result,
                    model_used=model_to_use,
                )
            except Exception as log_e:
                logger.error("Failed to log OpenAI sentiment interaction: %s", log_e, exc_info=True)
            logger.info("Analyzed market sentiment for %s.", symbol)
            self._update_cache(cache_key, standardized_result)
            return (True, standardized_result)
        except Exception as e:
            logger.error(
                "Unexpected error in analyze_market_sentiment %s: %s", symbol, e, exc_info=True
            )
            default_error_response["error"] = f"Unexpected error: {str(e)}"
            return (False, default_error_response)

    async def generate_trading_strategy(
        self, market_data: Dict, risk_profile: str = "moderate", force_refresh: bool = False
    ) -> Tuple[bool, Dict]:
        """Async generate trading strategy using config settings."""
        default_error_response = {
            "strategy_type": "hold",
            "strength": 0.0,
            "confidence": 0.0,
            "error": None,
        }
        if not self._check_client():
            default_error_response["error"] = "Client not initialized"
            return (False, default_error_response)
        if not isinstance(market_data, dict):
            default_error_response["error"] = "Invalid market_data input"
            return (False, default_error_response)
        symbol = market_data.get("symbol", "UNKNOWN")
        cache_key = self._get_cache_key(
            "generate_trading_strategy",
            symbol=symbol,
            market_data=market_data,
            risk_profile=risk_profile,
        )
        if not force_refresh:
            cached = self._check_cache(cache_key)
            if cached and isinstance(cached, dict):
                return (True, cached)
        try:
            system_prompt = PromptManager.get_system_prompt("strategy")
            user_prompt = PromptManager.get_strategy_prompt(symbol, market_data, risk_profile)
            model_to_use = self.settings["model"]
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            response = await self._call_openai_api(
                model=model_to_use,
                messages=messages,
                temperature=self.settings["temp_strategy"],
                max_tokens=self.settings["max_tokens_strategy"],
            )
            if response is None or not hasattr(response, "choices") or (not response.choices):
                error_msg = "No valid response from OpenAI API (strategy)"
                logger.error(error_msg)
                default_error_response["error"] = error_msg
                return (False, default_error_response)
            message = response.choices[0].message
            content = getattr(message, "content", None)
            if content is None:
                error_msg = "OpenAI response message content is missing (strategy)"
                logger.error(error_msg)
                default_error_response["error"] = error_msg
                return (False, default_error_response)
            strategy_json = self._parse_json_response(content.strip(), "strategy", symbol)
            if strategy_json is None or not isinstance(strategy_json, dict):
                default_error_response["error"] = (
                    "Failed to parse valid JSON from OpenAI response (strategy)"
                )
                return (False, default_error_response)
            strategy_type = self._extract_strategy_type(strategy_json)
            strategy_sentiment = convert_recommendation_to_sentiment(strategy_type)
            confidence = self._estimate_strategy_confidence(strategy_json)
            standardized_result = {
                "strategy_type": strategy_type,
                "strength": strategy_sentiment,
                "confidence": confidence,
                "timeframe": strategy_json.get("timeframe", "Medium-term"),
                "position_size": strategy_json.get("position_size", ""),
                "raw_strategy": strategy_json,
            }
            try:
                log_openai_analysis(
                    symbol=symbol if symbol != "UNKNOWN" else None,
                    analysis_type="strategy",
                    prompt=user_prompt,
                    response_data=response,
                    result_metadata=standardized_result,
                    model_used=model_to_use,
                )
            except Exception as log_e:
                logger.error("Failed to log OpenAI strategy interaction: %s", log_e, exc_info=True)
            logger.info("Generated trading strategy for %s.", symbol)
            self._update_cache(cache_key, standardized_result)
            return (True, standardized_result)
        except Exception as e:
            logger.error(
                "Unexpected error in generate_trading_strategy %s: %s", symbol, e, exc_info=True
            )
            default_error_response["error"] = f"Unexpected error: {str(e)}"
            return (False, default_error_response)

    async def analyze_technical_indicators(
        self, indicator_data: Dict, force_refresh: bool = False
    ) -> Tuple[bool, Dict]:
        """Async analyze technical indicators using config settings."""
        default_error_response = {
            "signal": "hold",
            "strength": 0.0,
            "confidence": 0.0,
            "error": None,
        }
        if not self._check_client():
            default_error_response["error"] = "Client not initialized"
            return (False, default_error_response)
        if not isinstance(indicator_data, dict) or not indicator_data:
            error_msg = "No valid indicator data provided"
            logger.warning(error_msg)
            default_error_response["error"] = error_msg
            return (False, default_error_response)
        symbol_for_log = indicator_data.get("symbol", "generic")
        cache_key = self._get_cache_key(
            "analyze_technical_indicators", indicator_data=indicator_data
        )
        if not force_refresh:
            cached = self._check_cache(cache_key)
            if cached and isinstance(cached, dict):
                return (True, cached)
        try:
            system_prompt = PromptManager.get_system_prompt("technical")
            user_prompt = PromptManager.get_technical_prompt(indicator_data)
            model_to_use = self.settings["model"]
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            response = await self._call_openai_api(
                model=model_to_use,
                messages=messages,
                temperature=self.settings["temp_technical"],
                max_tokens=self.settings["max_tokens_technical"],
            )
            if response is None or not hasattr(response, "choices") or (not response.choices):
                error_msg = "No valid response from OpenAI API (technical)"
                logger.error(error_msg)
                default_error_response["error"] = error_msg
                return (False, default_error_response)
            message = response.choices[0].message
            content = getattr(message, "content", None)
            if content is None:
                error_msg = "OpenAI response message content is missing (technical)"
                logger.error(error_msg)
                default_error_response["error"] = error_msg
                return (False, default_error_response)
            analysis_json = self._parse_json_response(content.strip(), "technical", "Indicators")
            if analysis_json is None or not isinstance(analysis_json, dict):
                default_error_response["error"] = (
                    "Failed to parse valid JSON from OpenAI response (technical)"
                )
                return (False, default_error_response)
            signal_raw = analysis_json.get("overall_signal", "hold")
            signal = signal_raw.lower() if isinstance(signal_raw, str) else "hold"
            signal = signal if signal in ["buy", "sell"] else "hold"
            signal_value = convert_recommendation_to_sentiment(signal)
            confidence_raw = analysis_json.get("confidence")
            confidence = extract_confidence(confidence_raw)
            conflicts = analysis_json.get("conflicts", [])
            if isinstance(conflicts, list) and conflicts:
                penalty = min(0.3, len(conflicts) * 0.1)
                confidence = max(0.1, confidence - penalty)
            standardized_result = {
                "signal": signal,
                "strength": signal_value,
                "confidence": confidence,
                "conflicts": conflicts if isinstance(conflicts, list) else [],
                "interpretations": (
                    analysis_json.get("indicator_interpretations", {})
                    if isinstance(analysis_json.get("indicator_interpretations"), dict)
                    else {}
                ),
                "raw_analysis": analysis_json,
            }
            try:
                log_openai_analysis(
                    symbol=symbol_for_log,
                    analysis_type="technical",
                    prompt=user_prompt,
                    response_data=response,
                    result_metadata=standardized_result,
                    model_used=model_to_use,
                )
            except Exception as log_e:
                logger.error("Failed to log OpenAI technical interaction: %s", log_e, exc_info=True)
            logger.info("Analyzed technical indicators.")
            self._update_cache(cache_key, standardized_result)
            return (True, standardized_result)
        except Exception as e:
            logger.error("Unexpected error in analyze_technical_indicators: %s", e, exc_info=True)
            default_error_response["error"] = f"Unexpected error: {str(e)}"
            return (False, default_error_response)

    async def generate_analysis(
        self, market_data: Dict[str, Any], symbol: str, force_refresh: bool = False
    ) -> Dict[str, Any]:
        """Async generate comprehensive analysis concurrently for a single asset."""
        default_result: Dict[str, Any] = {
            "market_sentiment": "neutral",
            "trading_strategy": "hold",
            "strength": 0.0,
            "confidence": 0.5,
            "trend": "neutral",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sentiment_analysis": None,
            "strategy_analysis": None,
            "technical_analysis": None,
            "historical_trend": None,
            "error": None,
        }
        if not self._check_client():
            default_result["error"] = "Client not initialized"
            return default_result
        if not isinstance(market_data, dict):
            default_result["error"] = "Invalid market_data input"
            return default_result
        logger.info("Generating comprehensive analysis for %s...", symbol)
        cache_key = self._get_cache_key("generate_analysis", symbol=symbol, market_data=market_data)
        if not force_refresh:
            cached = self._check_cache(cache_key)
            if cached and isinstance(cached, dict):
                return cached
        try:
            simplified_market_data = self._simplify_market_data(market_data, symbol)
            news_data_raw = market_data.get("sentiment", {}).get("news", [])
            news_data = news_data_raw if isinstance(news_data_raw, list) else []
            tasks = [
                self.analyze_market_sentiment(simplified_market_data, news_data),
                self.generate_trading_strategy(simplified_market_data),
            ]
            indicator_data = simplified_market_data.get("indicators", {})
            if isinstance(indicator_data, dict) and indicator_data:
                tasks.append(self.analyze_technical_indicators(indicator_data))
            else:

                async def placeholder_tech():
                    return (
                        True,
                        {"signal": "hold", "strength": 0, "confidence": 0.5, "skipped": True},
                    )

                tasks.append(placeholder_tech())
            task_results = await asyncio.gather(*tasks, return_exceptions=True)
            (sentiment_success, sentiment_result) = self._process_task_result(
                task_results[0], "Sentiment", symbol
            )
            (strategy_success, strategy_result) = self._process_task_result(
                task_results[1], "Strategy", symbol
            )
            (technical_success, technical_result) = self._process_task_result(
                task_results[2], "Technical", symbol
            )
            combined_analysis = self._combine_analysis_results(
                sentiment_result if sentiment_success else None,
                strategy_result if strategy_success else None,
                technical_result if technical_success else None,
                symbol,
            )
            if self.sentiment_tracker:
                try:
                    combined_analysis["historical_trend"] = self.sentiment_tracker.get_statistics(
                        symbol
                    )
                except Exception as trend_e:
                    logger.warning("Failed to get trend stats for %s: %s", symbol, trend_e)
            self._update_cache(cache_key, combined_analysis)
            logger.info("Generated comprehensive analysis for %s.", symbol)
            return combined_analysis
        except Exception as e:
            logger.error(
                "Error generating comprehensive analysis for %s: %s", symbol, e, exc_info=True
            )
            default_result["error"] = f"Analysis failed: {str(e)}"
            return default_result

    async def _run_analysis_with_semaphore(
        self, semaphore: asyncio.Semaphore, data: Dict[str, Any], symbol: str
    ) -> Tuple[str, Dict[str, Any]]:
        """Helper function to run generate_analysis under semaphore control."""
        async with semaphore:
            logger.debug("Acquired semaphore for %s analysis.", symbol)
            result = await self.generate_analysis(data, symbol)
            logger.debug("Released semaphore for %s analysis.", symbol)
            return (symbol, result)

    async def analyze_multiple_assets(
        self, assets_data: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Async analyze multiple assets concurrently with controlled parallelism."""
        default_error: Dict[str, Any] = {"error": "Client not initialized"}
        if not self._check_client():
            return {s: default_error for s in assets_data}
        if not isinstance(assets_data, dict):
            return {"error": "Invalid assets_data input"}  # type: ignore[dict-item]
        concurrency = int(self.settings["concurrency_limit"])
        logger.info(
            "Analyzing %s assets concurrently (Max Concurrency: %s)...",
            len(assets_data),
            concurrency,
        )
        semaphore = asyncio.Semaphore(concurrency)
        tasks: list = []
        for symbol, data in assets_data.items():
            if isinstance(data, dict):
                tasks.append(self._run_analysis_with_semaphore(semaphore, data, symbol))
            else:
                logger.warning(
                    "Skipping analysis for %s: Invalid data format (%s).", symbol, type(data)
                )
        if not tasks:
            return {}
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
        final_results = {}
        for result_item in results_list:
            if isinstance(result_item, Exception):
                logger.error("Analysis task failed unexpectedly: %s", result_item, exc_info=True)
            elif isinstance(result_item, tuple) and len(result_item) == 2:
                (symbol, analysis_result) = result_item
                if isinstance(analysis_result, dict):
                    final_results[symbol] = analysis_result
                    if analysis_result.get("error"):
                        logger.error(
                            "Analysis task completed with error for %s: %s",
                            symbol,
                            analysis_result["error"],
                        )
                else:
                    logger.error(
                        "Analysis task for %s returned unexpected result type: %s",
                        symbol,
                        type(analysis_result),
                    )
                    final_results[symbol] = {"error": "Analysis returned unexpected result type."}
            else:
                logger.error("Analysis task returned unexpected type: %s", type(result_item))
        logger.info("Completed analysis for %s assets.", len(final_results))
        return final_results

    def _parse_json_response(
        self, text: Optional[str], context: str, identifier: str
    ) -> Optional[Dict]:
        """Safely parses JSON from OpenAI response, handling potential markdown code blocks."""
        if not isinstance(text, str) or not text.strip():
            return None
        try:
            cleaned_text = text.strip()
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]
            cleaned_text = cleaned_text.strip()
            if not cleaned_text.startswith("{") or not cleaned_text.endswith("}"):
                start = cleaned_text.find("{")
                end = cleaned_text.rfind("}") + 1
                if start != -1 and end > start:
                    cleaned_text = cleaned_text[start:end]
                else:
                    raise json.JSONDecodeError(
                        "Text does not appear to contain a JSON object.", cleaned_text, 0
                    )
            parsed_json = json.loads(cleaned_text)
            if not isinstance(parsed_json, dict):
                logger.warning(
                    "Parsed JSON is not a dictionary (%s for '%s').", context, identifier
                )
                return None
            return parsed_json
        except json.JSONDecodeError as e:
            logger.error(
                "Failed JSON parsing (%s for '%s'). Error: %s. Response snippet: '%s...'",
                context,
                identifier,
                e,
                text[:150],
            )
            return None
        except Exception as e:
            logger.error(
                "Unexpected error parsing JSON (%s for '%s'): %s",
                context,
                identifier,
                e,
                exc_info=True,
            )
            return None

    def _extract_strategy_type(self, strategy_json: Optional[Dict]) -> str:
        """Extracts overall strategy type (buy/sell/hold) from generated strategy safely."""
        if not isinstance(strategy_json, dict):
            return "hold"
        entry = str(strategy_json.get("entry_point", "")).lower()
        if any((w in entry for w in ["buy", "long"])):
            return "buy"
        if any((w in entry for w in ["sell", "short"])):
            return "sell"
        notes = str(strategy_json.get("notes", "")).lower()
        if any((w in notes for w in ["buy", "long", "bullish"])):
            return "buy"
        if any((w in notes for w in ["sell", "short", "bearish"])):
            return "sell"
        sentiment = str(strategy_json.get("sentiment", "")).lower()
        if "bullish" in sentiment:
            return "buy"
        if "bearish" in sentiment:
            return "sell"
        return "hold"

    def _estimate_strategy_confidence(self, strategy_json: Optional[Dict]) -> float:
        """Estimates confidence based on completeness of strategy details safely."""
        if not isinstance(strategy_json, dict):
            return 0.1
        factors = [
            "entry_point",
            "exit_point",
            "stop_loss",
            "take_profit",
            "indicators_to_watch",
            "notes",
        ]
        score = sum((1 for f in factors if strategy_json.get(f)))
        max_score = len(factors)
        return max(0.1, min(1.0, 0.1 + score / max_score * 0.9)) if max_score > 0 else 0.1

    def _simplify_market_data(self, market_data: Dict, symbol: str) -> Dict:
        """Creates a simplified market data dictionary suitable for AI prompts, handling potential None values."""
        if not isinstance(market_data, dict):
            return {"symbol": symbol}
        ohlcv = market_data.get("ohlcv")
        indicators = market_data.get("indicators")
        sentiment = market_data.get("sentiment_metrics")
        ticker = market_data.get("ticker")
        current_price = None
        if ticker and isinstance(ticker, dict) and (ticker.get("last") is not None):
            try:
                current_price = float(ticker["last"])
            except (ValueError, TypeError):
                pass
        if current_price is None and isinstance(ohlcv, list) and ohlcv:
            try:
                last_candle = ohlcv[-1]
                price_val = None
                if isinstance(last_candle, dict):
                    price_val = last_candle.get("close")
                elif isinstance(last_candle, list) and len(last_candle) > 4:
                    price_val = last_candle[4]
                if price_val is not None:
                    current_price = float(price_val)
            except (IndexError, ValueError, TypeError):
                pass
        ohlcv_recent: list = []
        if isinstance(ohlcv, list):
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
        if isinstance(indicators, dict):
            for k, v in indicators.items():
                if v is not None:
                    try:
                        simplified_indicators[k] = (
                            round(float(v), 4) if isinstance(v, (float, int)) else v
                        )
                    except (ValueError, TypeError):
                        simplified_indicators[k] = v
        agg_sentiment_scores = {}
        if isinstance(sentiment, dict):
            for source_name, metrics in sentiment.items():
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

    def _process_task_result(
        self, result: Any, task_name: str, symbol: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """Safely processes results from asyncio.gather."""
        if isinstance(result, Exception):
            logger.error("%s task failed for %s: %s", task_name, symbol, result, exc_info=True)
            return (False, {"error": f"{task_name} failed: {str(result)}"})
        elif isinstance(result, tuple) and len(result) == 2:
            (success, data) = result
            if success and isinstance(data, dict):
                return (True, data)
            else:
                return (
                    False,
                    (
                        data
                        if isinstance(data, dict)
                        else {"error": f"Invalid data from {task_name}: {str(data)}"}
                    ),
                )
        else:
            logger.error(
                "%s task returned unexpected result type for %s: %s",
                task_name,
                symbol,
                type(result),
            )
            return (False, {"error": f"{task_name} returned unexpected format."})

    def _combine_analysis_results(
        self,
        sentiment_res: Optional[Dict[str, Any]],
        strategy_res: Optional[Dict[str, Any]],
        technical_res: Optional[Dict[str, Any]],
        symbol: str,
    ) -> Dict[str, Any]:
        """Combines results from sentiment, strategy, and technical analysis safely."""
        sentiment_res = sentiment_res if isinstance(sentiment_res, dict) else {}
        strategy_res = strategy_res if isinstance(strategy_res, dict) else {}
        technical_res = technical_res if isinstance(technical_res, dict) else {}
        sentiment_str = sentiment_res.get("strength", 0.0)
        sentiment_conf = sentiment_res.get("confidence", 0.0)
        strategy_str = strategy_res.get("strength", 0.0)
        strategy_conf = strategy_res.get("confidence", 0.0)
        technical_str = technical_res.get("strength", 0.0)
        technical_conf = technical_res.get("confidence", 0.0)
        try:
            (sentiment_str, sentiment_conf) = (float(sentiment_str), float(sentiment_conf))
        except (ValueError, TypeError):
            (sentiment_str, sentiment_conf) = (0.0, 0.0)
        try:
            (strategy_str, strategy_conf) = (float(strategy_str), float(strategy_conf))
        except (ValueError, TypeError):
            (strategy_str, strategy_conf) = (0.0, 0.0)
        try:
            (technical_str, technical_conf) = (float(technical_str), float(technical_conf))
        except (ValueError, TypeError):
            (technical_str, technical_conf) = (0.0, 0.0)
        weights = {"sentiment": 0.3, "strategy": 0.35, "technical": 0.35}
        total_conf_weighted_sum = (
            sentiment_conf * weights["sentiment"]
            + strategy_conf * weights["strategy"]
            + technical_conf * weights["technical"]
        )
        total_weight_sum = sum(weights.values())
        overall_confidence = (
            total_conf_weighted_sum / total_weight_sum if total_weight_sum > 1e-09 else 0.0
        )
        weighted_strength_sum = (
            sentiment_str * sentiment_conf * weights["sentiment"]
            + strategy_str * strategy_conf * weights["strategy"]
            + technical_str * technical_conf * weights["technical"]
        )
        weighted_strength = (
            weighted_strength_sum / total_conf_weighted_sum
            if total_conf_weighted_sum > 1e-09
            else 0.0
        )
        market_sentiment = (
            "bullish"
            if weighted_strength > 0.15
            else "bearish" if weighted_strength < -0.15 else "neutral"
        )
        trading_strategy = "hold"
        if weighted_strength > 0.3 and overall_confidence > 0.55:
            trading_strategy = "buy"
        elif weighted_strength < -0.3 and overall_confidence > 0.55:
            trading_strategy = "sell"
        final_result = {
            "market_sentiment": market_sentiment,
            "trading_strategy": trading_strategy,
            "strength": round(weighted_strength, 4),
            "confidence": round(overall_confidence, 4),
            "trend": "neutral",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sentiment_analysis": sentiment_res,
            "strategy_analysis": strategy_res,
            "technical_analysis": technical_res,
            "historical_trend": None,
            "error": None,
        }
        component_errors = [
            res.get("error")
            for res in [sentiment_res, strategy_res, technical_res]
            if res.get("error")
        ]
        if component_errors:
            error_msg = "One or more analysis components failed: " + "; ".join(
                filter(None, component_errors)
            )
            final_result["error"] = error_msg
            logger.warning(
                "Combined analysis for %s includes component failures: %s", symbol, error_msg
            )
        return final_result


def log_openai_analysis(
    symbol: Optional[str],
    analysis_type: str,
    prompt: str,
    response_data: Any,
    result_metadata: Optional[Dict[str, Any]] = None,
    model_used: Any = None,
) -> None:
    """
    Logs OpenAI API interaction details (prompt, response, metadata) to a
    JSON file specific to the asset being analyzed.

    Args:
        symbol (Optional[str]): The crypto symbol (e.g., 'BTC'). If None or 'generic',
                                logs to a generic file.
        analysis_type (str): Type of analysis (e.g., 'sentiment', 'strategy').
        prompt (str): The full prompt sent to the API.
        response_data (Any): The raw response object from the OpenAI API client.
                             Should contain 'usage' and 'choices'.
        result_metadata (Optional[Dict]): Standardized result from the analysis
                                           (e.g., containing strength, confidence).
        model_used (Optional[str]): The specific model name used for the request.
    """
    log_symbol = "GENERIC"
    if symbol and isinstance(symbol, str) and (symbol != "generic"):
        log_symbol = symbol.upper().replace("/", "_")
    elif symbol == "generic":
        log_symbol = "GENERIC_TECHNICAL"
    log_file = CACHE_DIR / f"{log_symbol}_OpenAI_analysis.json"
    logger.debug(
        "Logging OpenAI analysis interaction for %s (%s) to %s", log_symbol, analysis_type, log_file
    )
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": log_symbol,
        "analysis_type": analysis_type,
        "prompt": prompt,
        "response": None,
        "strategy_result": None,
        "strength_score": None,
        "confidence_score": None,
        "token_usage": None,
        "model_used": model_used or "unknown",
    }
    raw_response_content = None
    if response_data:
        try:
            if hasattr(response_data, "choices") and response_data.choices:
                message = response_data.choices[0].message
                raw_response_content = getattr(message, "content", None)
            log_entry["response"] = raw_response_content
            usage_data = getattr(response_data, "usage", None)
            if usage_data:
                if hasattr(usage_data, "model_dump"):
                    log_entry["token_usage"] = usage_data.model_dump()
                elif (
                    hasattr(usage_data, "prompt_tokens")
                    and hasattr(usage_data, "completion_tokens")
                    and hasattr(usage_data, "total_tokens")
                ):
                    log_entry["token_usage"] = {
                        "prompt_tokens": usage_data.prompt_tokens,
                        "completion_tokens": usage_data.completion_tokens,
                        "total_tokens": usage_data.total_tokens,
                    }
                else:
                    try:
                        log_entry["token_usage"] = vars(usage_data)
                    except TypeError:
                        log_entry["token_usage"] = "[Cannot dump usage data]"
            if not model_used and hasattr(response_data, "model"):
                log_entry["model_used"] = str(response_data.model)
        except Exception as e:
            logger.warning("Could not fully extract response details for logging: %s", e)
            log_entry["response"] = (
                "[Error extracting response content]"
                if raw_response_content is None
                else raw_response_content
            )
            log_entry["token_usage"] = "[Error extracting usage]"
    if isinstance(result_metadata, dict):
        log_entry["strength_score"] = result_metadata.get("strength")
        log_entry["confidence_score"] = result_metadata.get("confidence")
        if analysis_type == "strategy":
            log_entry["strategy_result"] = result_metadata.get("strategy_type")
        elif analysis_type == "sentiment":
            log_entry["strategy_result"] = result_metadata.get("recommendation")
        elif analysis_type == "technical":
            log_entry["strategy_result"] = result_metadata.get("signal")
    existing_logs: list = []
    if log_file.exists():
        try:
            with log_file.open("r", encoding="utf-8") as f:
                content = f.read()
                if content.strip():
                    loaded_data = json.loads(content)
                    if isinstance(loaded_data, list):
                        existing_logs = loaded_data
                    else:
                        logger.warning(
                            "Existing OpenAI log file %s is not a list. Overwriting.", log_file
                        )
        except json.JSONDecodeError:
            logger.error("Error decoding JSON from %s. Overwriting.", log_file)
        except OSError as e:
            logger.error(
                "OS error reading OpenAI log file %s: %s. Will attempt to overwrite.", log_file, e
            )
        except Exception as e:
            logger.error(
                "Unexpected error loading OpenAI log file %s: %s. Will attempt to overwrite.",
                log_file,
                e,
                exc_info=True,
            )
    all_logs = existing_logs + [log_entry]
    temp_file = log_file.with_suffix(f".tmp_{time.time_ns()}")
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with temp_file.open("w", encoding="utf-8") as f:
            json.dump(all_logs, f, indent=2, default=str)
        temp_file.replace(log_file)
        logger.debug("Appended OpenAI analysis log entry for %s (%s)", log_symbol, analysis_type)
    except (OSError, TypeError, ValueError) as e:
        logger.error("Error saving OpenAI analysis log to %s: %s", log_file, e, exc_info=True)
        if temp_file.exists():
            try:
                temp_file.unlink()
            except OSError:
                pass
    except Exception as e:
        logger.error("Unexpected error saving OpenAI analysis log: %s", e, exc_info=True)
        if temp_file.exists():
            try:
                temp_file.unlink()
            except OSError:
                pass
