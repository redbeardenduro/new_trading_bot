"""
Enhanced Rate Limiting Module for Exchange APIs

Implements per-endpoint rate limiting with exponential backoff and jitter
specifically tuned for Kraken's rate limit structure.
"""

import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional

from common.common_logger import get_logger

logger = get_logger("rate_limiter")


@dataclass
class RateLimitConfig:
    """Rate limit configuration for an endpoint."""

    calls_per_second: float
    weight: int = 1
    backoff_base: float = 2.0
    max_backoff: float = 60.0
    jitter: bool = True


class RateLimiter:
    """
    Per-endpoint rate limiter with exponential backoff and jitter.

    Kraken rate limits:
    - Public endpoints: ~1 call/sec
    - Private endpoints: ~1-2 calls/sec depending on tier
    - Trading endpoints: Lower limits, higher weights
    """

    DEFAULT_CONFIGS = {
        "public": RateLimitConfig(calls_per_second=1.0, weight=1),
        "fetchTicker": RateLimitConfig(calls_per_second=1.0, weight=1),
        "fetchOHLCV": RateLimitConfig(calls_per_second=1.0, weight=1),
        "fetchOrderBook": RateLimitConfig(calls_per_second=1.0, weight=1),
        "private": RateLimitConfig(calls_per_second=1.0, weight=2),
        "fetchBalance": RateLimitConfig(calls_per_second=1.0, weight=2),
        "fetchOpenOrders": RateLimitConfig(calls_per_second=1.0, weight=2),
        "fetchClosedOrders": RateLimitConfig(calls_per_second=0.5, weight=2),
        "trading": RateLimitConfig(calls_per_second=0.5, weight=3),
        "createOrder": RateLimitConfig(calls_per_second=0.5, weight=3),
        "cancelOrder": RateLimitConfig(calls_per_second=0.5, weight=2),
        "editOrder": RateLimitConfig(calls_per_second=0.5, weight=3),
    }

    def __init__(self, window_size: int = 60) -> None:
        """
        Initialize rate limiter.

        Args:
            window_size: Size of the sliding window in seconds
        """
        self.window_size = window_size
        self.call_history: Dict[str, deque] = {}
        self.error_counts: Dict[str, int] = {}
        self.last_error_time: Dict[str, float] = {}
        self.configs = self.DEFAULT_CONFIGS.copy()

    def set_config(self, endpoint: str, config: RateLimitConfig) -> None:
        """
        Set custom rate limit configuration for an endpoint.

        Args:
            endpoint: Endpoint name
            config: Rate limit configuration
        """
        self.configs[endpoint] = config
        logger.info("Set rate limit config for %s: %s", endpoint, config)

    def _get_config(self, endpoint: str) -> RateLimitConfig:
        """Get configuration for an endpoint."""
        if endpoint in self.configs:
            return self.configs[endpoint]
        if "create" in endpoint.lower() or "place" in endpoint.lower():
            return self.configs["trading"]
        elif "cancel" in endpoint.lower():
            return self.configs["cancelOrder"]
        elif "fetch" in endpoint.lower() and any(
            (x in endpoint.lower() for x in ["balance", "order", "position"])
        ):
            return self.configs["private"]
        else:
            return self.configs["public"]

    def _clean_old_calls(self, endpoint: str, now: float) -> None:
        """Remove calls outside the sliding window."""
        if endpoint not in self.call_history:
            self.call_history[endpoint] = deque()
            return
        history = self.call_history[endpoint]
        cutoff = now - self.window_size
        while history and history[0] < cutoff:
            history.popleft()

    def _calculate_backoff(self, endpoint: str, config: RateLimitConfig) -> float:
        """
        Calculate backoff time based on error count.

        Args:
            endpoint: Endpoint name
            config: Rate limit configuration

        Returns:
            Backoff time in seconds
        """
        error_count = self.error_counts.get(endpoint, 0)
        if error_count == 0:
            return 0.0
        backoff = min(config.backoff_base**error_count, config.max_backoff)
        if config.jitter:
            jitter_amount = backoff * 0.3
            backoff += random.uniform(-jitter_amount, jitter_amount)
        return max(0.0, backoff)

    def wait_if_needed(self, endpoint: str) -> float:
        """
        Wait if rate limit would be exceeded.

        Args:
            endpoint: Endpoint name

        Returns:
            Time waited in seconds
        """
        config = self._get_config(endpoint)
        now = time.time()
        self._clean_old_calls(endpoint, now)
        history = self.call_history.get(endpoint, deque())
        if len(history) > 0:
            time_since_last = now - history[-1]
            min_interval = 1.0 / config.calls_per_second
            if time_since_last < min_interval:
                wait_time = min_interval - time_since_last
                backoff = self._calculate_backoff(endpoint, config)
                total_wait = wait_time + backoff
                if total_wait > 0:
                    logger.debug(
                        "Rate limiting %s: waiting %.2fs (rate limit: %.2fs, backoff: %.2fs)",
                        endpoint,
                        total_wait,
                        wait_time,
                        backoff,
                    )
                    time.sleep(total_wait)
                    return total_wait
        backoff = self._calculate_backoff(endpoint, config)
        if backoff > 0:
            logger.debug("Backoff for %s: waiting %.2fs", endpoint, backoff)
            time.sleep(backoff)
            return backoff
        return 0.0

    def record_call(self, endpoint: str) -> None:
        """
        Record a successful API call.

        Args:
            endpoint: Endpoint name
        """
        now = time.time()
        if endpoint not in self.call_history:
            self.call_history[endpoint] = deque()
        self.call_history[endpoint].append(now)
        if endpoint in self.error_counts:
            self.error_counts[endpoint] = 0

    def record_error(self, endpoint: str, error_code: Optional[int] = None) -> None:
        """
        Record an API error for backoff calculation.

        Args:
            endpoint: Endpoint name
            error_code: HTTP error code (if applicable)
        """
        now = time.time()
        self.error_counts[endpoint] = self.error_counts.get(endpoint, 0) + 1
        self.last_error_time[endpoint] = now
        error_count = self.error_counts[endpoint]
        logger.warning(
            "Recorded error for %s (count: %s, code: %s)", endpoint, error_count, error_code
        )
        if error_count >= 3:
            logger.error(
                "High error count for %s: %s consecutive errors. Consider investigating the issue.",
                endpoint,
                error_count,
            )

    def reset_errors(self, endpoint: str) -> None:
        """
        Reset error count for an endpoint.

        Args:
            endpoint: Endpoint name
        """
        if endpoint in self.error_counts:
            del self.error_counts[endpoint]
        if endpoint in self.last_error_time:
            del self.last_error_time[endpoint]
        logger.info("Reset error count for %s", endpoint)

    def get_stats(self, endpoint: Optional[str] = None) -> Dict:
        """
        Get rate limiter statistics.

        Args:
            endpoint: Specific endpoint (None for all)

        Returns:
            Statistics dictionary
        """
        if endpoint:
            history = self.call_history.get(endpoint, deque())
            return {
                "endpoint": endpoint,
                "calls_in_window": len(history),
                "error_count": self.error_counts.get(endpoint, 0),
                "last_error": self.last_error_time.get(endpoint),
                "config": self._get_config(endpoint).__dict__,
            }
        else:
            return {
                ep: {
                    "calls_in_window": len(hist),
                    "error_count": self.error_counts.get(ep, 0),
                    "last_error": self.last_error_time.get(ep),
                }
                for (ep, hist) in self.call_history.items()
            }


_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get or create the global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter
