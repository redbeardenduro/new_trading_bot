"""
Comprehensive Audit Logging System.

Provides structured audit logging for all critical trading operations,
including order submissions, balance changes, configuration updates,
and system events. Supports multiple output formats and retention policies.
"""

import gzip
import hashlib
import json
import shutil
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from common.common_logger import DATA_DIR, get_logger

logger = get_logger("audit_logger")


class AuditEventType(Enum):
    """Types of audit events."""

    ORDER_CREATED = "order_created"
    ORDER_SUBMITTED = "order_submitted"
    ORDER_FILLED = "order_filled"
    ORDER_PARTIALLY_FILLED = "order_partially_filled"
    ORDER_CANCELED = "order_canceled"
    ORDER_REJECTED = "order_rejected"
    ORDER_DUPLICATE = "order_duplicate"
    BALANCE_UPDATED = "balance_updated"
    BALANCE_SNAPSHOT = "balance_snapshot"
    PORTFOLIO_REBALANCE = "portfolio_rebalance"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    POSITION_ADJUSTED = "position_adjusted"
    CONFIG_LOADED = "config_loaded"
    CONFIG_UPDATED = "config_updated"
    PARAMETER_CHANGED = "parameter_changed"
    BOT_STARTED = "bot_started"
    BOT_STOPPED = "bot_stopped"
    BOT_ERROR = "bot_error"
    CIRCUIT_BREAKER_TRIGGERED = "circuit_breaker_triggered"
    CIRCUIT_BREAKER_RESET = "circuit_breaker_reset"
    API_KEY_USED = "api_key_used"
    AUTHENTICATION_FAILED = "authentication_failed"
    RATE_LIMIT_HIT = "rate_limit_hit"
    MARKET_DATA_FETCHED = "market_data_fetched"
    PRICE_ALERT = "price_alert"
    ANOMALY_DETECTED = "anomaly_detected"


class AuditSeverity(Enum):
    """Severity levels for audit events."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuditEvent:
    """
    Represents a single audit event with full context.

    Attributes:
        event_id: Unique event identifier
        event_type: Type of audit event
        severity: Event severity level
        timestamp: Event timestamp
        actor: Entity that triggered the event
        resource: Resource affected by the event
        action: Action performed
        details: Additional event details
        metadata: Event metadata
        checksum: Event integrity checksum
    """

    def __init__(
        self,
        event_type: AuditEventType,
        severity: AuditSeverity,
        actor: str,
        resource: str,
        action: str,
        details: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize an audit event.

        Args:
            event_type: Type of audit event
            severity: Event severity level
            actor: Entity that triggered the event (e.g., 'bot', 'user', 'system')
            resource: Resource affected (e.g., 'order:123', 'balance:BTC')
            action: Action performed (e.g., 'create', 'update', 'delete')
            details: Additional event details
            metadata: Event metadata
        """
        self.event_id = self._generate_event_id()
        self.event_type = event_type
        self.severity = severity
        self.timestamp = datetime.now(timezone.utc)
        self.actor = actor
        self.resource = resource
        self.action = action
        self.details = details or {}
        self.metadata = metadata or {}
        self.metadata["hostname"] = self._get_hostname()
        self.metadata["process_id"] = self._get_process_id()
        self.checksum = self._generate_checksum()

    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        import uuid

        return str(uuid.uuid4())

    def _get_hostname(self) -> str:
        """Get system hostname."""
        import socket

        try:
            return socket.gethostname()
        except Exception:
            return "unknown"

    def _get_process_id(self) -> int:
        """Get process ID."""
        import os

        return os.getpid()

    def _generate_checksum(self) -> str:
        """
        Generate integrity checksum for the event.

        Returns:
            SHA256 checksum of event data
        """
        event_data = {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "actor": self.actor,
            "resource": self.resource,
            "action": self.action,
            "details": self._serialize_details(self.details),
        }
        event_str = json.dumps(event_data, sort_keys=True)
        return hashlib.sha256(event_str.encode()).hexdigest()

    def _serialize_details(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Serialize details for checksum calculation.

        Args:
            details: Details dictionary

        Returns:
            Serialized details
        """
        serialized = {}
        for key, value in details.items():
            if isinstance(value, Decimal):
                serialized[key] = str(value)
            elif isinstance(value, (datetime,)):
                serialized[key] = value.isoformat()
            elif isinstance(value, (dict, list)):
                serialized[key] = json.dumps(value, sort_keys=True)
            else:
                serialized[key] = str(value)
        return serialized

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert event to dictionary representation.

        Returns:
            Dictionary representation of the event
        """
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "actor": self.actor,
            "resource": self.resource,
            "action": self.action,
            "details": self.details,
            "metadata": self.metadata,
            "checksum": self.checksum,
        }

    def verify_integrity(self) -> bool:
        """
        Verify event integrity by recalculating checksum.

        Returns:
            True if integrity is valid, False otherwise
        """
        current_checksum = self.checksum
        recalculated_checksum = self._generate_checksum()
        return current_checksum == recalculated_checksum


class AuditLogger:
    """
    Comprehensive audit logging system.

    Features:
    - Structured audit event logging
    - Multiple output formats (JSON, JSONL)
    - Event integrity verification
    - Automatic log rotation and compression
    - Retention policy enforcement
    - Query and search capabilities
    """

    def __init__(
        self,
        storage_dir: Optional[Path] = None,
        retention_days: int = 90,
        compress_after_days: int = 7,
        enable_integrity_check: bool = True,
    ) -> None:
        """
        Initialize the audit logger.

        Args:
            storage_dir: Directory for audit log storage
            retention_days: Number of days to retain audit logs
            compress_after_days: Compress logs older than this many days
            enable_integrity_check: Enable event integrity verification
        """
        if storage_dir is None:
            storage_dir = DATA_DIR / "audit_logs"
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.retention_days = retention_days
        self.compress_after_days = compress_after_days
        self.enable_integrity_check = enable_integrity_check
        self.current_log_file = self._get_current_log_file()
        logger.info(
            "Initialized AuditLogger (storage: %s, retention: %s days)",
            self.storage_dir,
            retention_days,
        )

    def _get_current_log_file(self) -> Path:
        """Get the current log file path."""
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        return self.storage_dir / f"audit_{date_str}.jsonl"

    def log_event(self, event: AuditEvent) -> None:
        """
        Log an audit event.

        Args:
            event: Audit event to log
        """
        try:
            if self.enable_integrity_check and (not event.verify_integrity()):
                logger.error("Event integrity check failed for %s", event.event_id)
                return
            current_log_file = self._get_current_log_file()
            with current_log_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event.to_dict()) + "\n")
            log_msg = f"[AUDIT] {event.event_type.value} | {event.actor} -> {event.resource} | {event.action}"
            if event.severity == AuditSeverity.DEBUG:
                logger.debug(log_msg)
            elif event.severity == AuditSeverity.INFO:
                logger.info(log_msg)
            elif event.severity == AuditSeverity.WARNING:
                logger.warning(log_msg)
            elif event.severity == AuditSeverity.ERROR:
                logger.error(log_msg)
            elif event.severity == AuditSeverity.CRITICAL:
                logger.critical(log_msg)
        except Exception as e:
            logger.error("Failed to log audit event: %s", e, exc_info=True)

    def log_order_event(
        self, event_type: AuditEventType, order_id: str, action: str, details: Dict[str, Any]
    ) -> None:
        """
        Log an order-related audit event.

        Args:
            event_type: Type of order event
            order_id: Order identifier
            action: Action performed
            details: Event details
        """
        event = AuditEvent(
            event_type=event_type,
            severity=AuditSeverity.INFO,
            actor="bot",
            resource=f"order:{order_id}",
            action=action,
            details=details,
        )
        self.log_event(event)

    def log_balance_event(
        self, currency: str, old_balance: Decimal, new_balance: Decimal, reason: str
    ) -> None:
        """
        Log a balance change event.

        Args:
            currency: Currency code
            old_balance: Previous balance
            new_balance: New balance
            reason: Reason for balance change
        """
        event = AuditEvent(
            event_type=AuditEventType.BALANCE_UPDATED,
            severity=AuditSeverity.INFO,
            actor="bot",
            resource=f"balance:{currency}",
            action="update",
            details={
                "currency": currency,
                "old_balance": str(old_balance),
                "new_balance": str(new_balance),
                "change": str(new_balance - old_balance),
                "reason": reason,
            },
        )
        self.log_event(event)

    def log_system_event(
        self,
        event_type: AuditEventType,
        severity: AuditSeverity,
        action: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log a system event.

        Args:
            event_type: Type of system event
            severity: Event severity
            action: Action performed
            details: Event details
        """
        event = AuditEvent(
            event_type=event_type,
            severity=severity,
            actor="system",
            resource="bot",
            action=action,
            details=details or {},
        )
        self.log_event(event)

    def query_events(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        event_types: Optional[List[AuditEventType]] = None,
        actor: Optional[str] = None,
        resource: Optional[str] = None,
        severity: Optional[AuditSeverity] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query audit events with filters.

        Args:
            start_date: Filter events after this date
            end_date: Filter events before this date
            event_types: Filter by event types
            actor: Filter by actor
            resource: Filter by resource
            severity: Filter by severity

        Returns:
            List of matching events
        """
        matching_events: List[Dict[str, Any]] = []
        log_files = self._get_log_files_in_range(start_date, end_date)
        for log_file in log_files:
            try:
                if log_file.suffix == ".gz":
                    with gzip.open(log_file, "rt", encoding="utf-8") as f:
                        lines = f.readlines()
                else:
                    with log_file.open("r", encoding="utf-8") as f:
                        lines = f.readlines()
                for line in lines:
                    try:
                        event_data = json.loads(line.strip())
                        if (
                            start_date
                            and datetime.fromisoformat(event_data["timestamp"]) < start_date
                        ):
                            continue
                        if end_date and datetime.fromisoformat(event_data["timestamp"]) > end_date:
                            continue
                        if event_types and event_data["event_type"] not in [
                            et.value for et in event_types
                        ]:
                            continue
                        if actor and event_data["actor"] != actor:
                            continue
                        if resource and (not event_data["resource"].startswith(resource)):
                            continue
                        if severity and event_data["severity"] != severity.value:
                            continue
                        matching_events.append(event_data)
                    except json.JSONDecodeError:
                        continue
            except Exception as e:
                logger.error("Error reading log file %s: %s", log_file, e)
        return matching_events

    def _get_log_files_in_range(
        self, start_date: Optional[datetime], end_date: Optional[datetime]
    ) -> List[Path]:
        """Get log files within date range."""
        all_log_files = sorted(self.storage_dir.glob("audit_*.jsonl*"))
        if not start_date and (not end_date):
            return all_log_files
        filtered_files: list = []
        for log_file in all_log_files:
            try:
                date_str = log_file.stem.replace("audit_", "").replace(".jsonl", "")
                file_date = datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=timezone.utc)
                if start_date and file_date < start_date:
                    continue
                if end_date and file_date > end_date:
                    continue
                filtered_files.append(log_file)
            except ValueError:
                continue
        return filtered_files

    def compress_old_logs(self) -> int:
        """
        Compress old log files.

        Returns:
            Number of files compressed
        """
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.compress_after_days)
        compressed_count = 0
        for log_file in self.storage_dir.glob("audit_*.jsonl"):
            try:
                date_str = log_file.stem.replace("audit_", "")
                file_date = datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=timezone.utc)
                if file_date < cutoff_date:
                    compressed_file = log_file.with_suffix(".jsonl.gz")
                    with log_file.open("rb") as f_in:
                        with gzip.open(compressed_file, "wb") as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    log_file.unlink()
                    compressed_count += 1
                    logger.info("Compressed audit log: %s", log_file.name)
            except Exception as e:
                logger.error("Error compressing log file %s: %s", log_file, e)
        return compressed_count

    def cleanup_old_logs(self) -> int:
        """
        Remove old log files based on retention policy.

        Returns:
            Number of files removed
        """
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
        removed_count = 0
        for log_file in self.storage_dir.glob("audit_*"):
            try:
                date_str = log_file.stem.replace("audit_", "").replace(".jsonl", "")
                file_date = datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=timezone.utc)
                if file_date < cutoff_date:
                    log_file.unlink()
                    removed_count += 1
                    logger.info("Removed old audit log: %s", log_file.name)
            except Exception as e:
                logger.error("Error removing log file %s: %s", log_file, e)
        return removed_count

    def get_statistics(
        self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get audit log statistics.

        Args:
            start_date: Start of date range
            end_date: End of date range

        Returns:
            Dictionary of statistics
        """
        events = self.query_events(start_date=start_date, end_date=end_date)
        event_type_counts: Dict[str, int] = {}
        severity_counts: Dict[str, int] = {}
        actor_counts: Dict[str, int] = {}
        for event in events:
            event_type = event["event_type"]
            severity = event["severity"]
            actor = event["actor"]
            event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            actor_counts[actor] = actor_counts.get(actor, 0) + 1
        return {
            "total_events": len(events),
            "event_type_counts": event_type_counts,
            "severity_counts": severity_counts,
            "actor_counts": actor_counts,
            "date_range": {
                "start": start_date.isoformat() if start_date else None,
                "end": end_date.isoformat() if end_date else None,
            },
        }


_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """
    Get the global audit logger instance.

    Returns:
        AuditLogger instance
    """
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger
