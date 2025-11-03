"""
Enhanced Circuit Breaker Implementation with Per-Scope Breakers.

Provides circuit breakers at multiple scopes:
- Per-exchange (auth failures, DDoS protection)
- Per-operation (place/cancel/get-balance)
- Per-symbol (if a market is misbehaving)

Integrates with Prometheus metrics for monitoring.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, Optional, Type

try:
    from core.metrics import (circuit_breaker_failure_count,
                              record_circuit_breaker_trip,
                              update_circuit_breaker_state)

    METRICS_AVAILABLE = True
except ImportError:  # pragma: no cover
    METRICS_AVAILABLE = False

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = 0
    OPEN = 1
    HALF_OPEN = 2


@dataclass
class CircuitBreakerConfig:
    """Configuration for a circuit breaker."""

    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    half_open_max_calls: int = 3
    success_threshold: int = 2
    expected_exception: Type[BaseException] = Exception


@dataclass
class CircuitBreakerStats:
    """Statistics for a circuit breaker."""

    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[float] = None
    last_state_change: float = field(default_factory=time.time)
    total_trips: int = 0


class CircuitBreaker:
    """
    Circuit breaker implementation with state management.

    States:
    - CLOSED: Normal operation, failures increment counter
    - OPEN: Blocking all calls, waiting for recovery timeout
    - HALF_OPEN: Allowing limited calls to test recovery
    """

    def __init__(
        self,
        name: str,
        scope: str,
        config: Optional[CircuitBreakerConfig] = None,
    ) -> None:
        """
        Initialize circuit breaker.

        Args:
            name: Name of the circuit breaker
            scope: Scope (exchange, operation, symbol)
            config: Configuration settings
        """
        self.name = name
        self.scope = scope
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.stats = CircuitBreakerStats()
        self.lock = threading.RLock()
        self._half_open_calls = 0
        self._update_state_metric()

    # ----- Internal helpers -----

    def _update_state_metric(self) -> None:
        """Update Prometheus state metric."""
        if METRICS_AVAILABLE:
            update_circuit_breaker_state(self.name, self.scope, self.state.value)
            circuit_breaker_failure_count.labels(breaker_name=self.name, scope=self.scope).set(
                self.stats.failure_count
            )

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.stats.last_failure_time is None:
            return False
        elapsed = time.time() - self.stats.last_failure_time
        return elapsed >= self.config.recovery_timeout

    def _trip(self) -> None:
        """Trip the circuit breaker to OPEN state."""
        logger.warning(
            "Circuit breaker '%s' (%s) tripped: %s failures",
            self.name,
            self.scope,
            self.stats.failure_count,
        )
        self.state = CircuitState.OPEN
        self.stats.total_trips += 1
        self.stats.last_state_change = time.time()
        if METRICS_AVAILABLE:
            record_circuit_breaker_trip(self.name, self.scope)
        self._update_state_metric()

    def _reset(self) -> None:
        """Reset the circuit breaker to CLOSED state."""
        logger.info("Circuit breaker '%s' (%s) reset to CLOSED", self.name, self.scope)
        self.state = CircuitState.CLOSED
        self.stats.failure_count = 0
        self.stats.success_count = 0
        self.stats.last_state_change = time.time()
        self._half_open_calls = 0
        self._update_state_metric()

    def _enter_half_open(self) -> None:
        """Enter HALF_OPEN state to test recovery."""
        logger.info("Circuit breaker '%s' (%s) entering HALF_OPEN", self.name, self.scope)
        self.state = CircuitState.HALF_OPEN
        self.stats.success_count = 0
        self.stats.last_state_change = time.time()
        self._half_open_calls = 0
        self._update_state_metric()

    # ----- Public API -----

    def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerOpenError: If circuit is open
            Exception: Original exception from function
        """
        with self.lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._enter_half_open()
                else:
                    raise CircuitBreakerOpenError("Circuit breaker '%s' is OPEN" % self.name)
            if self.state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.config.half_open_max_calls:
                    raise CircuitBreakerOpenError(
                        "Circuit breaker '%s' half-open limit reached" % self.name
                    )
                self._half_open_calls += 1

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.config.expected_exception:
            self._on_failure()
            raise

    def _on_success(self) -> None:
        """Handle successful call."""
        with self.lock:
            if self.state == CircuitState.HALF_OPEN:
                self.stats.success_count += 1
                if self.stats.success_count >= self.config.success_threshold:
                    self._reset()
            elif self.state == CircuitState.CLOSED:
                # Success in CLOSED clears consecutive failure streak
                self.stats.failure_count = 0
                self._update_state_metric()

    def _on_failure(self) -> None:
        """Handle failed call."""
        with self.lock:
            self.stats.failure_count += 1
            self.stats.last_failure_time = time.time()
            if self.state == CircuitState.HALF_OPEN:
                # Any failure in HALF_OPEN should re-trip immediately
                self._trip()
            elif self.state == CircuitState.CLOSED:
                if self.stats.failure_count >= self.config.failure_threshold:
                    self._trip()
                else:
                    self._update_state_metric()

    def get_state(self) -> CircuitState:
        """Get current circuit breaker state."""
        with self.lock:
            return self.state

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        with self.lock:
            return {
                "name": self.name,
                "scope": self.scope,
                "state": self.state.name,
                "failure_count": self.stats.failure_count,
                "success_count": self.stats.success_count,
                "total_trips": self.stats.total_trips,
                "last_failure_time": self.stats.last_failure_time,
                "last_state_change": self.stats.last_state_change,
            }


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""

    pass


class CircuitBreakerManager:
    """
    Manages multiple circuit breakers with different scopes.

    Supports hierarchical scopes:
    - exchange: kraken
    - operation: kraken.place_order
    - symbol: kraken.place_order.BTC/USD
    """

    def __init__(self) -> None:
        """Initialize circuit breaker manager."""
        self.breakers: Dict[str, CircuitBreaker] = {}
        self.lock = threading.RLock()
        self.default_configs: Dict[str, CircuitBreakerConfig] = {
            "exchange": CircuitBreakerConfig(failure_threshold=10, recovery_timeout=120.0),
            "operation": CircuitBreakerConfig(failure_threshold=5, recovery_timeout=60.0),
            "symbol": CircuitBreakerConfig(failure_threshold=3, recovery_timeout=30.0),
        }

    def get_breaker(
        self,
        name: str,
        scope: str,
        config: Optional[CircuitBreakerConfig] = None,
    ) -> CircuitBreaker:
        """
        Get or create a circuit breaker.

        Args:
            name: Breaker name (e.g., 'kraken.place_order.BTC/USD')
            scope: Scope type (exchange, operation, symbol)
            config: Optional custom configuration

        Returns:
            CircuitBreaker instance
        """
        with self.lock:
            if name not in self.breakers:
                if config is None:
                    config = self.default_configs.get(scope, CircuitBreakerConfig())
                self.breakers[name] = CircuitBreaker(name, scope, config)
            return self.breakers[name]

    def call_with_breaker(
        self,
        name: str,
        scope: str,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Execute function with circuit breaker protection.

        Args:
            name: Breaker name
            scope: Scope type
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result
        """
        breaker = self.get_breaker(name, scope)
        return breaker.call(func, *args, **kwargs)

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers."""
        with self.lock:
            return {name: breaker.get_stats() for (name, breaker) in self.breakers.items()}

    def reset_breaker(self, name: str) -> None:
        """Manually reset a circuit breaker."""
        with self.lock:
            if name in self.breakers:
                self.breakers[name]._reset()


# Global manager instance
_manager = CircuitBreakerManager()


def get_circuit_breaker_manager() -> CircuitBreakerManager:
    """Get the global circuit breaker manager instance."""
    return _manager


def get_circuit_breaker(
    name: str,
    scope: str,
    config: Optional[CircuitBreakerConfig] = None,
) -> CircuitBreaker:
    """
    Convenience accessor for a single circuit breaker.
    Mirrors the manager API but avoids needing to import the manager in callers/tests.
    """
    return _manager.get_breaker(name, scope, config=config)


def with_circuit_breaker(
    name: str, scope: str
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to protect a function with a circuit breaker.

    Args:
        name: Breaker name
        scope: Scope type (exchange, operation, symbol)

    Example:
        @with_circuit_breaker('kraken.place_order', 'operation')
        def place_order(symbol, side, amount):
            ...
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return _manager.call_with_breaker(name, scope, func, *args, **kwargs)

        return wrapper

    return decorator
