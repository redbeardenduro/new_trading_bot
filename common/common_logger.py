"""
Centralized setup for project paths and the primary logger instance.

This module determines the project's root directory, defines key data
subdirectories using the BotConfig, and sets up the central logger instance
using EnhancedLogger, configured via BotConfig. Ensures robust handling
of configuration and filesystem errors during setup.
"""

import hashlib
import json
import logging
import sys
from pathlib import Path

import git


def _init_print_error(message: str) -> None:
    """Prints error messages to stderr during early initialization."""
    print(f"ERROR [common_logger]: {message}", file=sys.stderr)


def _init_print_info(message: str) -> None:
    """Prints info messages to stdout during early initialization."""
    print(f"INFO [common_logger]: {message}", file=sys.stdout)


config = None
config_available = False
try:
    try:
        from core.config import BotConfig
    except ImportError as import_err:
        _init_print_error(f"Failed to import BotConfig: {import_err}. Cannot load configuration.")
        raise
    config = BotConfig()
    config_available = True
    _init_print_info("BotConfig loaded successfully.")
except Exception as config_init_err:
    _init_print_error(
        f"Failed to initialize BotConfig: {config_init_err}. Logging will use basic defaults."
    )
    config = None
    config_available = False
try:
    if (
        config_available
        and hasattr(config, "project_root")
        and isinstance(config.project_root, Path)
    ):
        PROJECT_ROOT = config.project_root
    else:
        PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = Path(".").resolve()
except Exception as path_err:
    _init_print_error(f"Failed to determine project root: {path_err}. Using current directory.")
    PROJECT_ROOT = Path(".").resolve()
if config_available and config:
    BASE_DATA_DIR_NAME = config.get("paths.data", "data")
    LOG_DIR_NAME = config.get("paths.logs", "data/logs")
    CACHE_DIR_NAME = config.get("paths.cache", "data/cache")
    TRADES_DIR_NAME = config.get("paths.trades", "data/trades")
    METRICS_DIR_NAME = config.get("paths.metrics", "data/metrics")
else:
    BASE_DATA_DIR_NAME = "data"
    LOG_DIR_NAME = "data/logs"
    CACHE_DIR_NAME = "data/cache"
    TRADES_DIR_NAME = "data/trades"
    METRICS_DIR_NAME = "data/metrics"
try:
    DATA_DIR = PROJECT_ROOT / BASE_DATA_DIR_NAME
    LOG_DIR = PROJECT_ROOT / LOG_DIR_NAME
    CACHE_DIR = PROJECT_ROOT / CACHE_DIR_NAME
    TRADES_DIR = PROJECT_ROOT / TRADES_DIR_NAME
    METRICS_DIR = PROJECT_ROOT / METRICS_DIR_NAME
    LIVE_TRADES_DIR = TRADES_DIR / "live"
    PAPER_TRADES_DIR = TRADES_DIR / "paper"
    SKIPPED_TRADES_DIR = TRADES_DIR / "skipped"
except TypeError as path_build_err:
    _init_print_error(f"Failed to construct paths (likely invalid config value): {path_build_err}")
    DATA_DIR = PROJECT_ROOT / "data"
    LOG_DIR = DATA_DIR / "logs"
    CACHE_DIR = DATA_DIR / "cache"
    TRADES_DIR = DATA_DIR / "trades"
    METRICS_DIR = DATA_DIR / "metrics"
    LIVE_TRADES_DIR = TRADES_DIR / "live"
    PAPER_TRADES_DIR = TRADES_DIR / "paper"
    SKIPPED_TRADES_DIR = TRADES_DIR / "skipped"
dirs_to_create = {
    LOG_DIR,
    CACHE_DIR,
    TRADES_DIR,
    METRICS_DIR,
    LIVE_TRADES_DIR,
    PAPER_TRADES_DIR,
    SKIPPED_TRADES_DIR,
}
for dir_path in dirs_to_create:
    if not isinstance(dir_path, Path):
        continue
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
        _init_print_info(f"Ensured directory exists: {dir_path}")
    except OSError as e:
        _init_print_error(f"Could not create directory {dir_path}. Check permissions. Error: {e}")
    except Exception as e:
        _init_print_error(f"Unexpected error creating directory {dir_path}: {e}")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    _init_print_info(f"Added project root to sys.path: {PROJECT_ROOT}")
central_logger = None
log_file_path_for_fallback = None
try:
    log_level_name = "INFO"
    log_filename = "trading_bot_fallback.log"
    log_rotation = "midnight"
    log_backup_count = 7
    log_to_file_enabled = True
    if config_available and config:
        log_level_name = config.get("logging.level", "INFO").upper()
        log_filename = config.get("logging.log_filename", "trading_bot_default.log")
        log_rotation = config.get("logging.rotation_interval", "midnight")
        log_backup_count = config.get("logging.rotation_backup_count", 7)
        log_to_file_enabled = config.get("logging.log_to_file", True)
    valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if log_level_name not in valid_log_levels:
        _init_print_error(f"Invalid logging level '{log_level_name}' in config. Using INFO.")
        log_level_name = "INFO"
    log_level = getattr(logging, log_level_name, logging.INFO)
    logger_log_dir = None
    can_write_to_log_dir = False
    if isinstance(LOG_DIR, Path):
        if LOG_DIR.is_dir():
            logger_log_dir = LOG_DIR
            can_write_to_log_dir = True
        else:
            _init_print_error(f"Log directory '{LOG_DIR}' exists but is not a directory.")
    else:
        _init_print_error(f"Log directory path '{LOG_DIR}' is invalid.")
    if log_to_file_enabled and (not can_write_to_log_dir):
        _init_print_error(
            f"File logging enabled in config, but log directory '{LOG_DIR}' is not accessible/valid. Disabling file logging."
        )
        log_to_file_enabled = False
        logger_log_dir = None
    if log_to_file_enabled and logger_log_dir and log_filename:
        log_file_path_for_fallback = logger_log_dir / log_filename
    try:
        from utils.enhanced_logging import EnhancedLogger, set_manifest_values

        config_hash = "UNKNOWN"
        if config_available and config:
            try:
                config_json = json.dumps(config.to_dict(), sort_keys=True)
                config_hash = hashlib.sha256(config_json.encode()).hexdigest()[:8]
            except Exception as e:
                _init_print_error(f"Failed to calculate config hash: {e}")
        code_sha = "UNKNOWN"
        try:
            repo = git.Repo(PROJECT_ROOT)
            code_sha = repo.head.commit.hexsha[:8]
        except Exception as e:
            _init_print_error(f"Failed to get code SHA: {e}")
        set_manifest_values(config_hash, code_sha)
        central_logger_instance = EnhancedLogger(
            name="trading_bot",
            log_level=log_level,
            log_dir=logger_log_dir,
            log_file_name=log_filename if log_to_file_enabled else None,
            rotation_interval=log_rotation,
            rotation_backup_count=log_backup_count,
        )
        central_logger = central_logger_instance.logger
        if not central_logger.hasHandlers():
            raise RuntimeError("EnhancedLogger initialized but failed to add handlers.")
        _init_print_info("Central logger initialized using EnhancedLogger.")
    except ImportError as e:
        _init_print_error(f"EnhancedLogger not found ({e}). Falling back to basic logging.")
        raise
    except Exception as e:
        _init_print_error(f"Error initializing EnhancedLogger: {e}. Falling back to basic logging.")
        raise
except Exception:
    _init_print_info("Setting up fallback basic logger.")
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file_path_for_fallback:
        try:
            log_file_path_for_fallback.parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(log_file_path_for_fallback, encoding="utf-8"))
            _init_print_info(f"Fallback logger will write to: {log_file_path_for_fallback}")
        except OSError as file_err:
            _init_print_error(
                f"Cannot open fallback log file {log_file_path_for_fallback}: {file_err}"
            )
        except Exception as file_err_other:
            _init_print_error(
                f"Unexpected error setting up fallback log file {log_file_path_for_fallback}: {file_err_other}"
            )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - [Fallback] %(message)s",
        handlers=handlers,
        force=True,
    )
    central_logger = logging.getLogger("trading_bot_fallback")
    central_logger.warning("Using fallback basic logger configuration.")


def get_logger(name: str) -> logging.Logger:
    """
    Return a child logger derived from the central logger configuration.

    Args:
        name: The name for the child logger (often __name__).

    Returns:
        A logging.Logger instance configured via the central setup.
    """
    if central_logger is None:
        _init_print_error("CRITICAL: Central logger (including fallback) is not initialized!")
        dummy_logger = logging.getLogger(f"dummy_{name}")
        dummy_logger.addHandler(logging.NullHandler())
        return dummy_logger
    try:
        name_str = str(name)
        if name_str.startswith(central_logger.name + "."):
            full_logger_name = name_str
        else:
            logger_name_suffix = name_str.split(".")[-1]
            full_logger_name = f"{central_logger.name}.{logger_name_suffix}"
        child_logger = logging.getLogger(full_logger_name)
        if child_logger.level == logging.NOTSET:
            child_logger.setLevel(central_logger.level)
        child_logger.propagate = True
        return child_logger
    except Exception as e:
        _init_print_error(f"Error getting child logger for '{name}': {e}")
        return central_logger


__all__ = [
    "get_logger",
    "PROJECT_ROOT",
    "DATA_DIR",
    "LOG_DIR",
    "CACHE_DIR",
    "TRADES_DIR",
    "LIVE_TRADES_DIR",
    "PAPER_TRADES_DIR",
    "SKIPPED_TRADES_DIR",
    "METRICS_DIR",
]
if __name__ == "__main__":
    print("\n--- common_logger.py Execution Test ---")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Dir: {DATA_DIR}")
    print(f"Log Dir: {LOG_DIR}")
    print(f"Cache Dir: {CACHE_DIR}")
    print(f"Trades Dir: {TRADES_DIR}")
    print(f"Metrics Dir: {METRICS_DIR}")
    print("-" * 20)
    main_log = get_logger(__name__)
    main_log.info("Info message from __main__ via get_logger.")
    main_log.warning("Warning message from __main__.")
    main_log.debug("Debug message (should appear only if log level is DEBUG).")
    core_log = get_logger("core.bot")
    core_log.error("Error message from core.bot logger.")
    dup_log = get_logger("trading_bot.some_module")
    dup_log.info("Info message from potentially duplicate named logger.")
    print("\nFinished logger test.")
