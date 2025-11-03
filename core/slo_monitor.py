"""
Service Level Objective (SLO) Monitoring and Alerting System.

Tracks key performance indicators and service level objectives for the trading bot,
including latency, availability, error rates, and trading performance metrics.
Provides alerting when SLOs are breached.
"""

import json
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from prometheus_client import Counter, Gauge, Histogram, start_http_server

from common.common_logger import DATA_DIR, get_logger

logger = get_logger("slo_monitor")


class SLOStatus(Enum):
    """SLO status enumeration."""

    OK = "ok"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class SLODefinition:
    """
    Service Level Objective definition.

    Attributes:
        name: SLO name
        description: SLO description
        target_value: Target value for the SLO
        warning_threshold: Warning threshold (percentage of target)
        critical_threshold: Critical threshold (percentage of target)
        measurement_window: Time window for measurement (seconds)
        comparison: Comparison operator ('lt', 'gt', 'eq')
    """

    name: str
    description: str
    target_value: float
    warning_threshold: float
    critical_threshold: float
    measurement_window: int
    comparison: str = "lt"


@dataclass
class SLOViolation:
    """
    Represents an SLO violation.

    Attributes:
        slo_name: Name of violated SLO
        timestamp: Violation timestamp
        current_value: Current measured value
        target_value: Target value
        severity: Violation severity
        message: Violation message
    """

    slo_name: str
    timestamp: datetime
    current_value: float
    target_value: float
    severity: AlertSeverity
    message: str


class SLOMonitor:
    """
    SLO monitoring and alerting system.

    Features:
    - Define and track multiple SLOs
    - Real-time monitoring and alerting
    - Prometheus metrics integration
    - Historical SLO compliance tracking
    - Alert callbacks for notifications
    """

    def __init__(
        self,
        storage_dir: Optional[Path] = None,
        prometheus_port: int = 8000,
        enable_prometheus: bool = True,
    ) -> None:
        """
        Initialize the SLO monitor.

        Args:
            storage_dir: Directory for storing SLO data
            prometheus_port: Port for Prometheus metrics endpoint
            enable_prometheus: Enable Prometheus metrics export
        """
        if storage_dir is None:
            storage_dir = DATA_DIR / "slo_metrics"
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.enable_prometheus = enable_prometheus
        self.prometheus_port = prometheus_port
        self.slos: Dict[str, SLODefinition] = {}
        self.measurements: Dict[str, deque] = {}
        self.slo_status: Dict[str, SLOStatus] = {}
        self.violations: List[SLOViolation] = []
        self.alert_callbacks: List[Callable[[SLOViolation], None]] = []
        self.lock = threading.RLock()
        if self.enable_prometheus:
            self._init_prometheus_metrics()
            self._start_prometheus_server()
        self._define_default_slos()
        logger.info(
            "Initialized SLOMonitor (storage: %s, prometheus: %s)",
            self.storage_dir,
            enable_prometheus,
        )

    def _init_prometheus_metrics(self) -> None:
        """Initialize Prometheus metrics."""
        self.api_latency = Histogram(
            "trading_bot_api_latency_seconds", "API call latency in seconds", ["endpoint", "method"]
        )
        self.order_latency = Histogram(
            "trading_bot_order_latency_seconds",
            "Order execution latency in seconds",
            ["order_type"],
        )
        self.error_counter = Counter(
            "trading_bot_errors_total", "Total number of errors", ["error_type", "component"]
        )
        self.success_counter = Counter(
            "trading_bot_successes_total",
            "Total number of successful operations",
            ["operation_type"],
        )
        self.trades_counter = Counter(
            "trading_bot_trades_total", "Total number of trades", ["side", "status"]
        )
        self.pnl_gauge = Gauge("trading_bot_pnl", "Current profit/loss", ["currency"])
        self.portfolio_value = Gauge(
            "trading_bot_portfolio_value", "Current portfolio value", ["currency"]
        )
        self.slo_compliance = Gauge(
            "trading_bot_slo_compliance", "SLO compliance percentage", ["slo_name"]
        )
        self.slo_violations = Counter(
            "trading_bot_slo_violations_total", "Total SLO violations", ["slo_name", "severity"]
        )
        self.uptime_gauge = Gauge("trading_bot_uptime_seconds", "Bot uptime in seconds")
        self.circuit_breaker_state = Gauge(
            "trading_bot_circuit_breaker_state",
            "Circuit breaker state (0=closed, 1=open)",
            ["breaker_name"],
        )
        logger.info("Initialized Prometheus metrics")

    def _start_prometheus_server(self) -> None:
        """Start Prometheus metrics HTTP server."""
        try:
            start_http_server(self.prometheus_port)
            logger.info("Started Prometheus metrics server on port %s", self.prometheus_port)
        except Exception as e:
            logger.error("Failed to start Prometheus server: %s", e)

    def _define_default_slos(self) -> None:
        """Define default SLOs for the trading bot."""
        self.define_slo(
            SLODefinition(
                name="api_latency_p95",
                description="95th percentile API latency should be < 1s",
                target_value=1.0,
                warning_threshold=0.8,
                critical_threshold=0.6,
                measurement_window=300,
                comparison="lt",
            )
        )
        self.define_slo(
            SLODefinition(
                name="order_placement_latency_p95",
                description="95th percentile order placement latency should be < 500ms",
                target_value=0.5,
                warning_threshold=0.8,
                critical_threshold=1.2,
                measurement_window=3600,
                comparison="lt",
            )
        )
        self.define_slo(
            SLODefinition(
                name="breaker_trips_per_hour",
                description="Circuit breaker trips should be < 3 per hour",
                target_value=3.0,
                warning_threshold=1.5,
                critical_threshold=2.0,
                measurement_window=3600,
                comparison="lt",
            )
        )
        self.define_slo(
            SLODefinition(
                name="reconciliation_mismatches_per_day",
                description="Reconciliation mismatches should be < 1 per day",
                target_value=1.0,
                warning_threshold=2.0,
                critical_threshold=3.0,
                measurement_window=86400,
                comparison="lt",
            )
        )
        self.define_slo(
            SLODefinition(
                name="order_latency_p99",
                description="99th percentile order execution latency should be < 2s",
                target_value=2.0,
                warning_threshold=0.8,
                critical_threshold=0.6,
                measurement_window=300,
                comparison="lt",
            )
        )
        self.define_slo(
            SLODefinition(
                name="error_rate",
                description="Error rate should be < 1%",
                target_value=1.0,
                warning_threshold=1.5,
                critical_threshold=2.0,
                measurement_window=300,
                comparison="lt",
            )
        )
        self.define_slo(
            SLODefinition(
                name="availability",
                description="System availability should be > 99.9%",
                target_value=99.9,
                warning_threshold=99.5,
                critical_threshold=99.0,
                measurement_window=3600,
                comparison="gt",
            )
        )
        self.define_slo(
            SLODefinition(
                name="order_success_rate",
                description="Order success rate should be > 95%",
                target_value=95.0,
                warning_threshold=90.0,
                critical_threshold=85.0,
                measurement_window=3600,
                comparison="gt",
            )
        )

    def define_slo(self, slo: SLODefinition) -> None:
        """
        Define a new SLO.

        Args:
            slo: SLO definition
        """
        with self.lock:
            self.slos[slo.name] = slo
            self.measurements[slo.name] = deque(maxlen=10000)
            self.slo_status[slo.name] = SLOStatus.UNKNOWN
            logger.info("Defined SLO: %s - %s", slo.name, slo.description)

    def record_measurement(
        self, slo_name: str, value: float, timestamp: Optional[datetime] = None
    ) -> None:
        """
        Record a measurement for an SLO.

        Args:
            slo_name: Name of the SLO
            value: Measured value
            timestamp: Measurement timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        with self.lock:
            if slo_name not in self.measurements:
                logger.warning("Unknown SLO: %s", slo_name)
                return
            self.measurements[slo_name].append({"timestamp": timestamp, "value": value})
            self._check_slo_compliance(slo_name)

    def _check_slo_compliance(self, slo_name: str) -> None:
        """
        Check if an SLO is being met.

        Args:
            slo_name: Name of the SLO to check
        """
        if slo_name not in self.slos:
            return
        slo = self.slos[slo_name]
        measurements = self.measurements[slo_name]
        if not measurements:
            return
        cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=slo.measurement_window)
        recent_measurements = [m for m in measurements if m["timestamp"] >= cutoff_time]
        if not recent_measurements:
            return
        values = [m["value"] for m in recent_measurements]
        if "latency" in slo_name or "p95" in slo_name or "p99" in slo_name:
            import numpy as np

            if "p99" in slo_name:
                current_value = np.percentile(values, 99)
            else:
                current_value = np.percentile(values, 95)
        elif "rate" in slo_name or "availability" in slo_name:
            current_value = sum(values) / len(values)
        else:
            current_value = sum(values) / len(values)
        old_status = self.slo_status[slo_name]
        new_status = self._determine_slo_status(slo, current_value)
        self.slo_status[slo_name] = new_status
        if self.enable_prometheus:
            compliance = self._calculate_compliance(slo, current_value)
            self.slo_compliance.labels(slo_name=slo_name).set(compliance)
        if new_status in [SLOStatus.WARNING, SLOStatus.CRITICAL]:
            if old_status != new_status:
                severity = (
                    AlertSeverity.WARNING
                    if new_status == SLOStatus.WARNING
                    else AlertSeverity.CRITICAL
                )
                self._record_violation(slo_name, current_value, slo.target_value, severity)

    def _determine_slo_status(self, slo: SLODefinition, current_value: float) -> SLOStatus:
        """
        Determine SLO status based on current value.

        Args:
            slo: SLO definition
            current_value: Current measured value

        Returns:
            SLO status
        """
        if slo.comparison == "lt":
            if current_value <= slo.target_value:
                return SLOStatus.OK
            elif current_value <= slo.target_value / slo.warning_threshold:
                return SLOStatus.WARNING
            else:
                return SLOStatus.CRITICAL
        elif slo.comparison == "gt":
            if current_value >= slo.target_value:
                return SLOStatus.OK
            elif current_value >= slo.target_value * slo.warning_threshold / 100:
                return SLOStatus.WARNING
            else:
                return SLOStatus.CRITICAL
        else:
            return SLOStatus.UNKNOWN

    def _calculate_compliance(self, slo: SLODefinition, current_value: float) -> float:
        """
        Calculate SLO compliance percentage.

        Args:
            slo: SLO definition
            current_value: Current measured value

        Returns:
            Compliance percentage (0-100)
        """
        if slo.comparison == "lt":
            if current_value <= slo.target_value:
                return 100.0
            else:
                return max(0, 100 * (1 - (current_value - slo.target_value) / slo.target_value))
        elif slo.comparison == "gt":
            if current_value >= slo.target_value:
                return 100.0
            else:
                return max(0, 100 * current_value / slo.target_value)
        else:
            return 0.0

    def _record_violation(
        self, slo_name: str, current_value: float, target_value: float, severity: AlertSeverity
    ) -> None:
        """
        Record an SLO violation.

        Args:
            slo_name: Name of violated SLO
            current_value: Current value
            target_value: Target value
            severity: Violation severity
        """
        violation = SLOViolation(
            slo_name=slo_name,
            timestamp=datetime.now(timezone.utc),
            current_value=current_value,
            target_value=target_value,
            severity=severity,
            message=f"SLO '{slo_name}' violated: current={current_value:.3f}, target={target_value:.3f}",
        )
        with self.lock:
            self.violations.append(violation)
        if self.enable_prometheus:
            self.slo_violations.labels(slo_name=slo_name, severity=severity.value).inc()
        self._trigger_alerts(violation)
        logger.warning(violation.message)

    def _trigger_alerts(self, violation: SLOViolation) -> None:
        """
        Trigger alert callbacks for a violation.

        Args:
            violation: SLO violation
        """
        for callback in self.alert_callbacks:
            try:
                callback(violation)
            except Exception as e:
                logger.error("Error in alert callback: %s", e, exc_info=True)

    def add_alert_callback(self, callback: Callable[[SLOViolation], None]) -> None:
        """
        Add an alert callback function.

        Args:
            callback: Callback function that takes an SLOViolation
        """
        self.alert_callbacks.append(callback)
        logger.info("Added alert callback: %s", callback.__name__)

    def record_api_call(self, endpoint: str, method: str, duration: float, success: bool) -> None:
        """Record an API call metric."""
        if self.enable_prometheus:
            self.api_latency.labels(endpoint=endpoint, method=method).observe(duration)
            if success:
                self.success_counter.labels(operation_type="api_call").inc()
            else:
                self.error_counter.labels(error_type="api_error", component="exchange").inc()
        self.record_measurement("api_latency_p95", duration)
        error_value = 0.0 if success else 100.0
        self.record_measurement("error_rate", error_value)

    def record_order_execution(self, order_type: str, duration: float, success: bool) -> None:
        """Record an order execution metric."""
        if self.enable_prometheus:
            self.order_latency.labels(order_type=order_type).observe(duration)
            status = "success" if success else "failed"
            self.trades_counter.labels(side=order_type, status=status).inc()
        self.record_measurement("order_latency_p99", duration)
        success_value = 100.0 if success else 0.0
        self.record_measurement("order_success_rate", success_value)

    def record_trade(
        self, side: str, status: str, pnl: Optional[float] = None, currency: str = "USD"
    ) -> None:
        """Record a trade metric."""
        if self.enable_prometheus:
            self.trades_counter.labels(side=side, status=status).inc()
            if pnl is not None:
                self.pnl_gauge.labels(currency=currency).set(pnl)

    def update_portfolio_value(self, value: float, currency: str = "USD") -> None:
        """Update portfolio value metric."""
        if self.enable_prometheus:
            self.portfolio_value.labels(currency=currency).set(value)

    def record_circuit_breaker_state(self, breaker_name: str, is_open: bool) -> None:
        """Record circuit breaker state."""
        if self.enable_prometheus:
            state = 1.0 if is_open else 0.0
            self.circuit_breaker_state.labels(breaker_name=breaker_name).set(state)

    def get_slo_status(self, slo_name: str) -> SLOStatus:
        """
        Get current status of an SLO.

        Args:
            slo_name: Name of the SLO

        Returns:
            Current SLO status
        """
        with self.lock:
            return self.slo_status.get(slo_name, SLOStatus.UNKNOWN)

    def get_all_slo_status(self) -> Dict[str, SLOStatus]:
        """
        Get status of all SLOs.

        Returns:
            Dictionary of SLO statuses
        """
        with self.lock:
            return self.slo_status.copy()

    def get_violations(
        self, since: Optional[datetime] = None, severity: Optional[AlertSeverity] = None
    ) -> List[SLOViolation]:
        """
        Get SLO violations.

        Args:
            since: Only return violations after this time  # type: ignore[unreachable]
            severity: Filter by severity

        Returns:
            List of violations
        """
        with self.lock:
            violations = self.violations.copy()
        if since:
            violations = [v for v in violations if v.timestamp >= since]
        if severity:
            violations = [v for v in violations if v.severity == severity]
        return violations

    def get_compliance_report(self) -> Dict[str, Any]:
        """
        Generate an SLO compliance report.

        Returns:
            Compliance report dictionary
        """
        report = {"timestamp": datetime.now(timezone.utc).isoformat(), "slos": {}}
        with self.lock:
            for slo_name, slo in self.slos.items():
                measurements = self.measurements[slo_name]
                if not measurements:
                    continue
                cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=slo.measurement_window)
                recent = [m for m in measurements if m["timestamp"] >= cutoff_time]
                if not recent:
                    continue
                values = [m["value"] for m in recent]
                report["slos"][slo_name] = {
                    "description": slo.description,
                    "target_value": slo.target_value,
                    "current_status": self.slo_status[slo_name].value,
                    "measurement_count": len(recent),
                    "avg_value": sum(values) / len(values),
                    "min_value": min(values),
                    "max_value": max(values),
                    "compliance": self._calculate_compliance(slo, sum(values) / len(values)),
                }
        return report

    def save_report(
        self, report: Optional[Dict[str, Any]] = None, output_file: Optional[Path] = None
    ) -> Path:
        """
        Save compliance report to file.

        Args:
            report: Report dictionary (generated if None)
            output_file: Output file path

        Returns:
            Path to saved report
        """
        if report is None:
            report = self.get_compliance_report()
        if output_file is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            output_file = self.storage_dir / f"slo_report_{timestamp}.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info("Saved SLO compliance report to %s", output_file)
        return output_file


_slo_monitor: Optional[SLOMonitor] = None


def get_slo_monitor() -> SLOMonitor:
    """
    Get the global SLO monitor instance.

    Returns:
        SLOMonitor instance
    """
    global _slo_monitor
    if _slo_monitor is None:
        _slo_monitor = SLOMonitor()
    return _slo_monitor
