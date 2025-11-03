"""
Enhanced Logging Utility Module.

Provides the EnhancedLogger class for setting up a configured logger instance
with console output and timed rotating file output. Includes helper methods
for logging application-specific events with basic serialization safety.
"""

import json
import logging
import logging.handlers
import sys
from logging import LogRecord
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union


def _print_error(message: str) -> None:
    print(f"LOGGER SETUP ERROR: {message}", file=sys.stderr)


import uuid

_RUN_ID = str(uuid.uuid4())
_CONFIG_HASH = "UNKNOWN"
_CODE_SHA = "UNKNOWN"


def set_manifest_values(config_hash: str, code_sha: str) -> None:
    global _CONFIG_HASH, _CODE_SHA
    _CONFIG_HASH = config_hash
    _CODE_SHA = code_sha


class ManifestFilter(logging.Filter):
    """A logging filter that injects manifest fields into LogRecord."""

    def filter(self, record: LogRecord) -> bool:
        record.run_id = _RUN_ID
        record.config_hash = _CONFIG_HASH
        record.code_sha = _CODE_SHA
        return True


class EnhancedLogger:
    """
    Configures and provides a logger instance with console and rotating file handlers.

    Handles log formatting, log levels, configurable log rotation, and provides
    convenience methods for standard and application-specific logging tasks.
    Designed to be instantiated once via a central configuration module.
    """

    def __init__(
        self,
        name: str = "trading_bot",
        log_level: int = logging.INFO,
        log_dir: Optional[Path] = None,
        log_file_name: Optional[str] = None,
        rotation_interval: str = "midnight",
        rotation_backup_count: int = 7,
    ):
        """
        Initializes the EnhancedLogger.

        Args:
            name (str): Base name for the logger (e.g., "trading_bot").
            log_level (int): The logging level (e.g., logging.INFO, logging.DEBUG).
            log_dir (Optional[Path]): Directory to store log files. If None or invalid, file logging is disabled.
            log_file_name (Optional[str]): Base name for the log file. If None or invalid, file logging is disabled.
            rotation_interval (str): The type of rotation interval ('S', 'M', 'H', 'D', 'W0'-'W6', 'midnight').
            rotation_backup_count (int): How many backup log files to keep.
        """
        try:
            self.logger = logging.getLogger(name)
            self.logger.setLevel(log_level)
            self.logger.propagate = False
            self.logger.addFilter(ManifestFilter())
            self.log_dir = None
            self.log_file_base_name = None
            file_logging_enabled = False
            if (
                isinstance(log_dir, Path)
                and log_file_name
                and isinstance(log_file_name, str)
                and log_file_name.strip()
            ):
                try:
                    log_dir.mkdir(parents=True, exist_ok=True)
                    if log_dir.is_dir():
                        self.log_dir = log_dir
                        self.log_file_base_name = log_file_name.strip()
                        file_logging_enabled = True
                    else:
                        _print_error(
                            f"Provided log path '{log_dir}' is not a directory. File logging disabled."
                        )
                except OSError as e:
                    _print_error(
                        f"Could not create/access log directory '{log_dir}'. File logging disabled. Error: {e}"
                    )
                except Exception as e:
                    _print_error(
                        f"Unexpected error setting up log directory '{log_dir}': {e}. File logging disabled."
                    )
            else:
                _print_error(
                    f"Log directory ('{log_dir}') or filename ('{log_file_name}') is invalid. File logging disabled."
                )
            if self.logger.hasHandlers():
                _print_error(f"Logger '{name}' already had handlers. Clearing them.")
                for handler in self.logger.handlers[:]:
                    try:
                        handler.flush()
                        handler.close()
                    except Exception as e:
                        _print_error(f"Warning: Error flushing/closing existing handler: {e}")
                    self.logger.removeHandler(handler)
            log_format = "%(asctime)s - %(name)s - %(levelname)s - [Manifest: %(run_id)s|%(config_hash)s|%(code_sha)s] - %(message)s"
            date_format = "%Y-%m-%d %H:%M:%S"
            formatter = logging.Formatter(log_format, datefmt=date_format)
            if file_logging_enabled:
                self._setup_file_handler(formatter, rotation_interval, rotation_backup_count)
            else:
                pass
            self._setup_console_handler(formatter)
            if self.logger.hasHandlers():
                self.logger.info(
                    f"EnhancedLogger initialized for '{name}'. Level: {logging.getLevelName(log_level)}. Log Dir: '{(self.log_dir if file_logging_enabled else 'Disabled')}'. Rotation: {rotation_interval}, Backups: {rotation_backup_count}"
                )
            else:
                _print_error(
                    f"CRITICAL: EnhancedLogger for '{name}' FAILED to add ANY handlers. Logging will not work."
                )
        except Exception as init_error:
            _print_error(
                f"CRITICAL error during EnhancedLogger initialization for '{name}': {init_error}"
            )
            if not hasattr(self, "logger"):
                self.logger = logging.getLogger(f"{name}_init_failed")
                self.logger.addHandler(logging.NullHandler())

    def _setup_file_handler(
        self, formatter: logging.Formatter, interval: str, backup_count: int
    ) -> None:
        """Sets up the TimedRotatingFileHandler with configured rotation."""
        if not self.log_dir or not self.log_file_base_name:
            _print_error("File handler setup skipped: Invalid log directory or filename.")
            return
        log_file_path = self.log_dir / self.log_file_base_name
        try:
            valid_intervals = ["S", "M", "H", "D", "MIDNIGHT"] + [f"W{i}" for i in range(7)]
            safe_interval = interval.upper() if isinstance(interval, str) else "MIDNIGHT"
            if safe_interval not in valid_intervals:
                _print_error(f"Invalid rotation interval '{interval}'. Using 'midnight'.")
                safe_interval = "MIDNIGHT"
            safe_backup_count = max(0, int(backup_count)) if isinstance(backup_count, int) else 7
            file_handler = logging.handlers.TimedRotatingFileHandler(
                filename=log_file_path,
                when=safe_interval,
                interval=1,
                backupCount=safe_backup_count,
                encoding="utf-8",
                delay=False,
                utc=True,
            )
            if safe_interval == "MIDNIGHT" or safe_interval == "D":
                file_handler.suffix = "%Y-%m-%d.log"
            elif safe_interval == "H":
                file_handler.suffix = "%Y-%m-%d_%H.log"
            else:
                file_handler.suffix = "%Y-%m-%d_%H-%M-%S.log"
            if file_handler.baseFilename.endswith(".log"):
                file_handler.baseFilename = file_handler.baseFilename[:-4]
            file_handler.setLevel(self.logger.level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            print(
                f"File handler configured: Path='{log_file_path}', Interval='{safe_interval}', Backups={safe_backup_count}"
            )
        except (OSError, ValueError, TypeError) as e:
            _print_error(f"Failed to set up file handler for '{log_file_path}'. Error: {e}")
        except Exception as e:
            _print_error(
                f"Unexpected error setting up file handler for '{log_file_path}'. Error: {e}"
            )

    def _setup_console_handler(self, formatter: logging.Formatter) -> None:
        """Sets up the StreamHandler to log to the console (stdout)."""
        try:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.logger.level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            print("Console handler configured.")
        except Exception as e:
            _print_error(f"CRITICAL: Failed to set up console handler. Error: {e}")

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        try:
            self.logger.debug(message, *args, **kwargs)
        except Exception as e:
            _print_error(f"Logging failed: {e}")

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        try:
            self.logger.info(message, *args, **kwargs)
        except Exception as e:
            _print_error(f"Logging failed: {e}")

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        try:
            self.logger.warning(message, *args, **kwargs)
        except Exception as e:
            _print_error(f"Logging failed: {e}")

    def error(self, message: str, *args: Any, exc_info: bool = False, **kwargs: Any) -> None:
        try:
            self.logger.error(message, *args, exc_info=exc_info, **kwargs)
        except Exception as e:
            _print_error(f"Logging failed: {e}")

    def critical(self, message: str, *args: Any, exc_info: bool = True, **kwargs: Any) -> None:
        try:
            self.logger.critical(message, *args, exc_info=exc_info, **kwargs)
        except Exception as e:
            _print_error(f"Logging failed: {e}")

    def exception(self, message: str, *args: Any, exc_info: bool = True, **kwargs: Any) -> None:
        try:
            self.logger.exception(message, *args, exc_info=exc_info, **kwargs)
        except Exception as e:
            _print_error(f"Logging failed: {e}")

    def _safe_json_dumps(self, data: Any) -> str:
        """Safely converts data to JSON string, handling common non-serializable types."""
        try:
            return json.dumps(data, default=str, ensure_ascii=False, indent=None)
        except (TypeError, ValueError) as e:
            self.logger.debug(
                f"Could not JSON serialize data for logging: {e}. Data Type: {type(data)}",
                exc_info=False,
            )
            return "[Unserializable Data]"
        except Exception as e:
            self.logger.debug(
                f"Unexpected error during JSON serialization for logging: {e}", exc_info=False
            )
            return "[Serialization Error]"

    def log_api_request(self, api_name: str, endpoint: str, params: Optional[Dict] = None) -> None:
        self.debug(
            f"API Request -> {api_name}: {endpoint}, Params: {(self._safe_json_dumps(params) if params else 'None')}"
        )

    def log_api_response(
        self, api_name: str, endpoint: str, status_code: int, response: Optional[Any] = None
    ) -> None:
        response_str = self._safe_json_dumps(response)
        log_level = logging.DEBUG if 200 <= status_code < 300 else logging.WARNING
        response_summary = response_str
        _TRUNC_LEN = 200
        if log_level > logging.DEBUG and len(response_summary) > _TRUNC_LEN:
            response_summary = response_summary[:_TRUNC_LEN] + "...(truncated)"
        self.logger.log(
            log_level,
            f"API Response <- {api_name}: {endpoint}, Status: {status_code}, Response: {response_summary}",
        )
        if response_summary != response_str:
            self.debug(f"Full API Response {api_name} {endpoint}: {response_str}")

    def log_trade_signal(
        self, pair: str, signal_type: str, confidence: float, data: Optional[Dict] = None
    ) -> None:
        try:
            conf_str = f"{float(confidence):.2f}"
        except (ValueError, TypeError):
            conf_str = "N/A"
        self.info(f"Signal Generated: {pair} - {signal_type.upper()} (Conf: {conf_str})")
        if data is not None:
            self.debug(f"Signal Data ({pair}): {self._safe_json_dumps(data)}")

    def log_trade_execution(
        self,
        pair: str,
        order_type: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        order_id: Optional[str] = None,
        status: str = "N/A",
        paper_trading: bool = True,
    ) -> None:
        mode = "PAPER" if paper_trading else "LIVE"
        try:
            amount_str = f"{float(amount):.8f}"
        except (ValueError, TypeError):
            amount_str = str(amount)
        price_str = f" @ {float(price):.4f}" if price is not None else ""
        id_str = f" (ID: {order_id})" if order_id else ""
        status_str = f" -> Status: {status}"
        self.info(
            f"Trade Execution [{mode}]: {side.upper()} {amount_str} {pair}{price_str} ({order_type}){id_str}{status_str}"
        )

    def log_portfolio_update(self, portfolio: Dict, paper_trading: bool = True) -> None:
        mode = "PAPER" if paper_trading else "LIVE"
        total_value = portfolio.get("total_value_usd", "N/A")
        balances = portfolio.get("balances", {})
        balances_summary = {}
        if isinstance(balances, dict):
            for k, v in balances.items():
                try:
                    if isinstance(v, (int, float)) and v > 1e-09:
                        balances_summary[k] = f"{float(v):.4f}"
                except (ValueError, TypeError):
                    pass
        self.info(f"Portfolio Update [{mode}]: Value ~ {total_value}, Balances: {balances_summary}")
        self.debug(f"Full Portfolio [{mode}]: {self._safe_json_dumps(portfolio)}")

    def log_error_with_context(
        self,
        message: str,
        context: Optional[Dict] = None,
        exc_info: Union[bool, Tuple, BaseException, None] = None,
    ) -> None:
        error_msg = f"{message}"
        if context is not None:
            context_str = self._safe_json_dumps(context)
            error_msg += f" | Context: {context_str}"
        self.error(error_msg, exc_info=exc_info)
