"""
Prometheus Metrics Instrumentation Module.

Provides comprehensive metrics collection for monitoring trading bot performance,
API operations, order execution, and system health.
"""

import time
from functools import wraps
from typing import Any, Callable, TypeVar, cast

from prometheus_client import (CONTENT_TYPE_LATEST, REGISTRY, Counter, Gauge,
                               Histogram, Info, Summary, generate_latest)

F = TypeVar("F", bound=Callable[..., Any])
order_attempts_total = Counter(
    "trading_bot_order_attempts_total",
    "Total number of order placement attempts",
    ["symbol", "side", "order_type"],
)
order_success_total = Counter(
    "trading_bot_order_success_total",
    "Total number of successful order placements",
    ["symbol", "side", "order_type"],
)
order_failures_total = Counter(
    "trading_bot_order_failures_total",
    "Total number of failed order placements",
    ["symbol", "side", "order_type", "error_type"],
)
order_cancellations_total = Counter(
    "trading_bot_order_cancellations_total",
    "Total number of order cancellations",
    ["symbol", "reason"],
)
order_fills_total = Counter(
    "trading_bot_order_fills_total", "Total number of filled orders", ["symbol", "side"]
)
order_fill_amount = Summary(
    "trading_bot_order_fill_amount", "Amount of filled orders", ["symbol", "side"]
)
order_lifecycle_seconds = Histogram(
    "trading_bot_order_lifecycle_seconds",
    "Time from order placement to final status (filled/canceled/rejected) in seconds",
    ["symbol"],
    buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0],
)
internal_api_request_duration_seconds = Histogram(
    "trading_bot_internal_api_request_duration_seconds",
    "Internal API request duration in seconds",
    ["endpoint", "method", "status"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)
exchange_call_seconds = Histogram(
    "trading_bot_exchange_call_seconds",
    "Exchange API call duration in seconds",
    ["exchange", "endpoint", "status"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)
exchange_call_total = Counter(
    "trading_bot_exchange_call_total",
    "Total number of exchange API calls",
    ["exchange", "endpoint", "status"],
)
internal_api_requests_total = Counter(
    "trading_bot_internal_api_requests_total",
    "Total number of internal API requests",
    ["endpoint", "method"],
)
internal_api_errors_total = Counter(
    "trading_bot_internal_api_errors_total",
    "Total number of internal API errors",
    ["endpoint", "error_type"],
)
api_rate_limit_hits_total = Counter(
    "trading_bot_api_rate_limit_hits_total",
    "Total number of rate limit hits",
    ["exchange", "endpoint"],
)
circuit_breaker_trips_total = Counter(
    "trading_bot_circuit_breaker_trips_total",
    "Total number of circuit breaker trips",
    ["breaker_name", "scope"],
)
circuit_breaker_state = Gauge(
    "trading_bot_circuit_breaker_state",
    "Current circuit breaker state (0=closed, 1=open, 2=half_open)",
    ["breaker_name", "scope"],
)
circuit_breaker_failure_count = Gauge(
    "trading_bot_circuit_breaker_failure_count",
    "Current failure count for circuit breaker",
    ["breaker_name", "scope"],
)
portfolio_value_usd = Gauge(
    "trading_bot_portfolio_value_usd", "Current portfolio value in USD", ["mode"]
)
portfolio_pnl_usd = Gauge("trading_bot_portfolio_pnl_usd", "Current profit/loss in USD", ["mode"])
portfolio_pnl_percent = Gauge(
    "trading_bot_portfolio_pnl_percent", "Current profit/loss percentage", ["mode"]
)
portfolio_drawdown_percent = Gauge(
    "trading_bot_portfolio_drawdown_percent", "Current drawdown percentage from peak", ["mode"]
)
portfolio_position_count = Gauge(
    "trading_bot_portfolio_position_count", "Number of open positions", ["mode"]
)
portfolio_asset_allocation_percent = Gauge(
    "trading_bot_portfolio_asset_allocation_percent",
    "Asset allocation percentage",
    ["symbol", "mode"],
)
portfolio_var_usd = Gauge(
    "trading_bot_portfolio_var_usd", "Value at Risk in USD", ["confidence_level", "mode"]
)
reconciliation_mismatches_total = Counter(
    "trading_bot_reconciliation_mismatches_total",
    "Total number of reconciliation mismatches detected",
    ["mismatch_type"],
)
reconciliation_duration_seconds = Histogram(
    "trading_bot_reconciliation_duration_seconds",
    "Reconciliation operation duration in seconds",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)
strategy_signals_total = Counter(
    "trading_bot_strategy_signals_total",
    "Total number of strategy signals generated",
    ["symbol", "strategy", "signal_type"],
)
strategy_signal_strength = Histogram(
    "trading_bot_strategy_signal_strength",
    "Distribution of strategy signal strengths",
    ["symbol", "strategy"],
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
opportunity_score = Histogram(
    "trading_bot_opportunity_score",
    "Distribution of opportunity scores",
    ["symbol"],
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
backtest_duration_seconds = Histogram(
    "trading_bot_backtest_duration_seconds",
    "Backtest execution duration in seconds",
    buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0],
)
backtest_trades_total = Counter(
    "trading_bot_backtest_trades_total",
    "Total number of trades in backtest",
    ["symbol", "strategy"],
)
backtest_final_pnl_percent = Gauge(
    "trading_bot_backtest_final_pnl_percent",
    "Final PnL percentage from backtest",
    ["strategy", "run_id"],
)
backtest_sharpe_ratio = Gauge(
    "trading_bot_backtest_sharpe_ratio", "Sharpe ratio from backtest", ["strategy", "run_id"]
)
backtest_max_drawdown_percent = Gauge(
    "trading_bot_backtest_max_drawdown_percent",
    "Maximum drawdown percentage from backtest",
    ["strategy", "run_id"],
)
bot_uptime_seconds = Gauge("trading_bot_uptime_seconds", "Bot uptime in seconds")
clock_drift_seconds = Gauge(
    "trading_bot_clock_drift_seconds", "Clock drift from exchange server in seconds", ["exchange"]
)
cache_hits_total = Counter(
    "trading_bot_cache_hits_total", "Total number of cache hits", ["cache_type"]
)
cache_misses_total = Counter(
    "trading_bot_cache_misses_total", "Total number of cache misses", ["cache_type"]
)
safety_check_violations_total = Counter(
    "trading_bot_safety_check_violations_total",
    "Total number of safety check violations",
    ["check_type"],
)
bot_info = Info("trading_bot_info", "Trading bot version and configuration information")


def track_exchange_api_call(exchange: str, endpoint: str) -> Callable[[F], F]:
    """
    Decorator to track exchange API call metrics.

    Args:
        exchange: Name of the exchange (e.g., 'kraken')
        endpoint: API endpoint being called
    """

    def decorator(func: F) -> F:

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:  # type: ignore[no-untyped-def]
            start_time = time.time()
            status = "success"
            try:
                result = func(*args, **kwargs)
                return result
            except Exception:
                status = "failure"
                raise
            finally:
                duration = time.time() - start_time
                exchange_call_total.labels(
                    exchange=exchange, endpoint=endpoint, status=status
                ).inc()
                exchange_call_seconds.labels(
                    exchange=exchange, endpoint=endpoint, status=status
                ).observe(duration)

        return cast(F, wrapper)

    return decorator


def track_internal_api_call(endpoint: str, method: str = "GET") -> Callable[[F], F]:
    """
    Decorator to track internal API call metrics.

    Args:
        endpoint: API endpoint being called
        method: HTTP method
    """

    def decorator(func: F) -> F:

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:  # type: ignore[no-untyped-def]
            internal_api_requests_total.labels(endpoint=endpoint, method=method).inc()
            start_time = time.time()
            status = "success"
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = "failure"
                error_type = type(e).__name__
                internal_api_errors_total.labels(endpoint=endpoint, error_type=error_type).inc()
                raise
            finally:
                duration = time.time() - start_time
                internal_api_request_duration_seconds.labels(
                    endpoint=endpoint, method=method, status=status
                ).observe(duration)

        return cast(F, wrapper)

    return decorator


def record_order_attempt(symbol: str, side: str, order_type: str) -> None:
    """Record an order placement attempt."""
    order_attempts_total.labels(symbol=symbol, side=side, order_type=order_type).inc()


def record_order_success(symbol: str, side: str, order_type: str) -> None:
    """Record a successful order placement."""
    order_success_total.labels(symbol=symbol, side=side, order_type=order_type).inc()


def record_order_failure(symbol: str, side: str, order_type: str, error_type: str) -> None:
    """Record a failed order placement."""
    order_failures_total.labels(
        symbol=symbol, side=side, order_type=order_type, error_type=error_type
    ).inc()


def record_order_fill(symbol: str, side: str, amount: float) -> None:
    """Record an order fill."""
    order_fills_total.labels(symbol=symbol, side=side).inc()
    order_fill_amount.labels(symbol=symbol, side=side).observe(amount)


def record_circuit_breaker_trip(breaker_name: str, scope: str) -> None:
    """Record a circuit breaker trip."""
    circuit_breaker_trips_total.labels(breaker_name=breaker_name, scope=scope).inc()


def update_circuit_breaker_state(breaker_name: str, scope: str, state: int) -> None:
    """
    Update circuit breaker state.

    Args:
        breaker_name: Name of the circuit breaker
        scope: Scope (exchange, operation, symbol)
        state: 0=closed, 1=open, 2=half_open
    """
    circuit_breaker_state.labels(breaker_name=breaker_name, scope=scope).set(state)


def update_portfolio_metrics(
    value_usd: float,
    pnl_usd: float,
    pnl_percent: float,
    drawdown_percent: float,
    position_count: int,
    mode: str = "paper",
) -> None:
    """Update portfolio metrics."""
    portfolio_value_usd.labels(mode=mode).set(value_usd)
    portfolio_pnl_usd.labels(mode=mode).set(pnl_usd)
    portfolio_pnl_percent.labels(mode=mode).set(pnl_percent)
    portfolio_drawdown_percent.labels(mode=mode).set(drawdown_percent)
    portfolio_position_count.labels(mode=mode).set(position_count)


def update_asset_allocation(symbol: str, allocation_percent: float, mode: str = "paper") -> None:
    """Update asset allocation percentage."""
    portfolio_asset_allocation_percent.labels(symbol=symbol, mode=mode).set(allocation_percent)


def record_strategy_signal(symbol: str, strategy: str, signal_type: str, strength: float) -> None:
    """Record a strategy signal."""
    strategy_signals_total.labels(symbol=symbol, strategy=strategy, signal_type=signal_type).inc()
    strategy_signal_strength.labels(symbol=symbol, strategy=strategy).observe(strength)


def record_opportunity_score(symbol: str, score: float) -> None:
    """Record an opportunity score."""
    opportunity_score.labels(symbol=symbol).observe(score)


def record_reconciliation_mismatch(mismatch_type: str) -> None:
    """Record a reconciliation mismatch."""
    reconciliation_mismatches_total.labels(mismatch_type=mismatch_type).inc()


def record_safety_violation(check_type: str) -> None:
    """Record a safety check violation."""
    safety_check_violations_total.labels(check_type=check_type).inc()


def update_clock_drift(exchange: str, drift_seconds: float) -> None:
    """Update clock drift metric."""
    clock_drift_seconds.labels(exchange=exchange).set(drift_seconds)


def record_cache_hit(cache_type: str) -> None:
    """Record a cache hit."""
    cache_hits_total.labels(cache_type=cache_type).inc()


def record_cache_miss(cache_type: str) -> None:
    """Record a cache miss."""
    cache_misses_total.labels(cache_type=cache_type).inc()


def get_metrics() -> bytes:
    """
    Generate Prometheus metrics in text format.

    Returns:
        Metrics in Prometheus text format
    """
    return generate_latest(REGISTRY)  # type: ignore[no-any-return]


def get_content_type() -> str:
    """
    Get the content type for Prometheus metrics.

    Returns:
        Content type string
    """
    return CONTENT_TYPE_LATEST  # type: ignore[no-any-return]


def initialize_bot_info(version: str, mode: str, strategy: str, config_hash: str) -> None:
    """
    Initialize bot information metric.

    Args:
        version: Bot version
        mode: Trading mode (paper/live)
        strategy: Trading strategy
        config_hash: Hash of configuration for tracking
    """
    bot_info.info(
        {"version": version, "mode": mode, "strategy": strategy, "config_hash": config_hash}
    )
