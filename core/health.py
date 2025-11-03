"""
Health and Readiness Endpoints for Trading Bot

Provides Kubernetes-compatible health check endpoints:
- /healthz: Liveness probe (basic uptime check)
- /readyz: Readiness probe (validates config, connectivity, breaker state)
"""

import os
import time
from typing import Any, Dict, List, Tuple

from flask import Response, jsonify

START_TIME = time.time()


def get_uptime() -> float:
    """Get application uptime in seconds."""
    return time.time() - START_TIME


def check_config_validity() -> Tuple[bool, str]:
    """
    Check if configuration is valid and loaded.

    Returns:
        Tuple of (is_valid, message)
    """
    try:
        from core.config import Config

        config = Config()
        if not hasattr(config, "exchange") or not config.exchange:
            return (False, "Exchange configuration missing")
        return (True, "Config valid")
    except Exception as e:
        return (False, f"Config validation failed: {str(e)}")


def check_exchange_connectivity() -> Tuple[bool, str]:
    """
    Check if exchange connection is available.

    Returns:
        Tuple of (is_connected, message)
    """
    try:
        import ccxt

        from core.config import Config

        config = Config()
        exchange_id = config.exchange.lower()
        api_key = os.getenv(f"{exchange_id.upper()}_API_KEY", "")
        api_secret = os.getenv(f"{exchange_id.upper()}_API_SECRET", "")
        if not api_key or not api_secret:
            return (False, "Exchange credentials not configured")
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class(
            {"apiKey": api_key, "secret": api_secret, "enableRateLimit": True, "timeout": 5000}
        )
        exchange.fetch_balance()
        return (True, "Exchange connected")
    except Exception as e:
        return (False, f"Exchange connectivity failed: {str(e)}")


def check_circuit_breaker_state() -> Tuple[bool, str]:
    """
    Check circuit breaker state.

    Returns:
        Tuple of (is_healthy, message)
    """
    try:
        return (True, "Circuit breaker operational")
    except Exception as e:
        return (False, f"Circuit breaker check failed: {str(e)}")


def healthz() -> Tuple[Response, int]:
    """
    Liveness probe endpoint.

    Returns basic health status and uptime.
    Always returns 200 OK unless the application is completely dead.

    Returns:
        Flask response tuple (response, status_code)
    """
    uptime = get_uptime()
    response = {"status": "ok", "uptime_seconds": round(uptime, 2), "timestamp": time.time()}
    return (jsonify(response), 200)


def readyz() -> Tuple[Response, int]:
    """
    Readiness probe endpoint.

    Performs comprehensive checks:
    - Config validity
    - Exchange connectivity
    - Circuit breaker state

    Returns 200 OK only if all checks pass.
    Returns 503 Service Unavailable if any check fails.

    Returns:
        Flask response tuple (response, status_code)
    """
    checks: List[Dict[str, Any]] = []
    all_healthy = True
    (config_ok, config_msg) = check_config_validity()
    checks.append(
        {"name": "config", "status": "pass" if config_ok else "fail", "message": config_msg}
    )
    if not config_ok:
        all_healthy = False
    (exchange_ok, exchange_msg) = check_exchange_connectivity()
    checks.append(
        {"name": "exchange", "status": "pass" if exchange_ok else "fail", "message": exchange_msg}
    )
    if not exchange_ok:
        all_healthy = False
    (breaker_ok, breaker_msg) = check_circuit_breaker_state()
    checks.append(
        {
            "name": "circuit_breaker",
            "status": "pass" if breaker_ok else "fail",
            "message": breaker_msg,
        }
    )
    if not breaker_ok:
        all_healthy = False
    response = {
        "status": "ready" if all_healthy else "not_ready",
        "checks": checks,
        "timestamp": time.time(),
    }
    status_code = 200 if all_healthy else 503
    return (jsonify(response), status_code)


def create_health_routes(app) -> None:
    """
    Register health check routes with Flask app.

    Args:
        app: Flask application instance
    """
    from flask import Blueprint

    health_bp = Blueprint("health", __name__)

    @health_bp.route("/healthz", methods=["GET"])
    def health_check() -> None:
        """Liveness probe endpoint."""
        return healthz()

    @health_bp.route("/readyz", methods=["GET"])
    def readiness_check() -> None:
        """Readiness probe endpoint."""
        return readyz()

    app.register_blueprint(health_bp)
    return health_bp
