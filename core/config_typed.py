"""
Centralized Configuration Management with Type Hints.

Loads configuration settings from user_config.json, provides default values,
validates critical settings, and makes them accessible throughout the
application via the BotConfig singleton instance.

This is a type-annotated version for mypy strict mode compliance.
"""

import copy
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

root_logger = logging.getLogger()
if not root_logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
config_logger = logging.getLogger("config_loader")
ConfigValue = Union[str, int, float, bool, None, List[Any], Dict[str, Any]]
ConfigDict = Dict[str, ConfigValue]
DEFAULT_CONFIG: ConfigDict = {
    "bot": {
        "base_currencies": ["BTC", "ETH"],
        "quote_currencies": ["USD"],
        "timeframe": "1h",
        "strategy": "combined",
        "paper_trading": True,
        "trading_interval_seconds": 300,
        "max_concurrent_pairs": 5,
        "max_cycles": None,
        "disable_twitter": True,
    },
    "paper_trading_settings": {"initial_capital": 10000.0},
    "trading": {
        "min_opportunity_score": 0.55,
        "position_size_percent": 5.0,
        "volatility_lookback_period": 14,
        "performance_lookback_period": 30,
        "confidence_bands": {"low": [0.0, 0.4], "medium": [0.4, 0.7], "high": [0.7, 1.0]},
        "confidence_band_multipliers": {"low": 1.05, "medium": 1.0, "high": 0.95},
        "dynamic_threshold_enabled": True,
        "rebalance_threshold_percent": 5.0,
    },
    "portfolio": {
        "max_allocation_per_asset_percent": 7.5,
        "min_allocation_per_asset_percent": 1.0,
        "target_allocation_weights": {"technical": 0.4, "sentiment": 0.3, "ai": 0.3},
        "technical_factor_thresholds": {
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "macd_threshold": 0,
            "bb_threshold_percent": 5.0,
        },
        "risk_management": {
            "var_confidence_level": 0.95,
            "var_time_horizon_days": 1,
            "use_monte_carlo_var": False,
            "stress_test_scenarios_percent": {"market_crash": -20.0, "moderate_decline": -10.0},
        },
        "simulation": {
            "fee_rate_percent": 0.1,
            "slippage_percent": 0.2,
            "min_order_value_quote": 10.0,
        },
    },
    "api_credentials": {
        "kraken": {"api_key": None, "api_secret": None},
        "twitter": {
            "api_key": None,
            "api_secret": None,
            "access_token": None,
            "access_token_secret": None,
            "bearer_token": None,
        },
        "reddit": {
            "client_id": None,
            "client_secret": None,
            "user_agent": "TradingBot/1.0",
            "username": None,
            "password": None,
        },
        "news": {"api_key": None},
        "openai": {"api_key": None},
    },
    "kraken_client": {
        "request_timeout_ms": 30000,
        "max_retries": 3,
        "retry_delay_seconds": 2,
        "cache_expiry_seconds": {
            "ticker": 30,
            "ohlcv": 300,
            "order_book": 10,
            "balance": 120,
            "markets": 3600,
        },
    },
    "news_client": {
        "request_timeout_seconds": 15,
        "max_results_per_query": 20,
        "search_days_back": 1,
        "cache_expiry_seconds": 1800,
        "debug_mode": False,
        "asset_keywords": {
            "DEFAULT": ["cryptocurrency", "crypto"],
            "BTC": ["Bitcoin", "BTC"],
            "ETH": ["Ethereum", "ETH", "Ether"],
            "XRP": ["XRP", "Ripple"],
            "LTC": ["Litecoin", "LTC"],
            "DOGE": ["Dogecoin", "DOGE"],
            "DOT": ["Polkadot", "DOT"],
            "SOL": ["Solana", "SOL"],
            "ADA": ["Cardano", "ADA"],
        },
    },
    "reddit_client": {
        "limit_per_subreddit": 10,
        "search_time_filter": "week",
        "cache_expiry_seconds": 1800,
    },
    "openai_client": {
        "request_timeout_seconds": 60,
        "model": "gpt-4o-mini",
        "fallback_model": "gpt-3.5-turbo",
        "max_retries": 3,
        "cache_expiry_seconds": 900,
        "temperature": {"analysis": 0.3, "strategy": 0.3, "technical": 0.3},
        "max_tokens": {"analysis": 700, "strategy": 700, "technical": 700},
        "concurrency_limit": 3,
    },
    "sentiment": {
        "bullish_threshold": 0.05,
        "bearish_threshold": -0.05,
        "history_length": 30,
        "trend_window": 5,
        "volume_baseline_count": 20,
    },
    "logging": {
        "level": "INFO",
        "log_to_file": True,
        "log_filename": "trading_bot.log",
        "rotation_interval": "midnight",
        "rotation_backup_count": 7,
    },
    "paths": {
        "data": "data",
        "logs": "data/logs",
        "cache": "data/cache",
        "trades": "data/trades",
        "metrics": "data/metrics",
    },
}


class ConfigNamespace:
    """Helper class for attribute-style access to configuration."""

    def __init__(self, data: ConfigDict) -> None:
        self._data = data

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        value = self._data.get(name)
        if isinstance(value, dict):
            return ConfigNamespace(cast(ConfigDict, value))
        return value

    def __getitem__(self, key: str) -> Any:
        return self.__getattr__(key)

    def __repr__(self) -> str:
        return f"ConfigNamespace({self._data})"


class BotConfig:
    """
    Centralized configuration manager with type safety.

    Loads configuration from JSON file, merges with defaults,
    and provides type-safe access to configuration values.
    """

    _instance: Optional["BotConfig"] = None
    _initialized: bool = False

    def __new__(cls, config_file_path: Optional[Path] = None) -> "BotConfig":
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super(BotConfig, cls).__new__(cls)
        return cls._instance

    def __init__(self, config_file_path: Optional[Path] = None) -> None:
        """
        Initialize the configuration manager.

        Args:
            config_file_path: Path to the user configuration JSON file
        """
        if BotConfig._initialized:
            return
        self.config_file_path: Optional[Path] = config_file_path
        self._config: ConfigDict = {}
        self._load_config()
        BotConfig._initialized = True

    def _load_config(self) -> None:
        """Load configuration from file and merge with defaults."""
        self._config = copy.deepcopy(DEFAULT_CONFIG)
        if self.config_file_path is None:
            possible_paths = [
                Path("config/user_config.json"),
                Path("user_config.json"),
                Path(__file__).parent.parent / "config" / "user_config.json",
            ]
            for path in possible_paths:
                if path.exists():
                    self.config_file_path = path
                    config_logger.info(f"Found config file at: {path}")
                    break
            if self.config_file_path is None:
                config_logger.warning("No user config file found. Using defaults.")
                return
        try:
            if not self.config_file_path.exists():
                config_logger.warning(
                    f"Config file not found: {self.config_file_path}. Using defaults."
                )
                return
            with self.config_file_path.open("r", encoding="utf-8") as f:
                user_config = json.load(f)
            if not isinstance(user_config, dict):
                config_logger.error("Invalid config file format. Using defaults.")
                return
            self._config = self._deep_merge(self._config, cast(ConfigDict, user_config))
            config_logger.info(f"Loaded configuration from {self.config_file_path}")
        except json.JSONDecodeError as e:
            config_logger.error(
                f"Error parsing config file {self.config_file_path}: {e}. Using defaults."
            )
        except OSError as e:
            config_logger.error(
                f"Error reading config file {self.config_file_path}: {e}. Using defaults."
            )
        except Exception as e:
            config_logger.error(
                f"Unexpected error loading config: {e}. Using defaults.", exc_info=True
            )

    def _deep_merge(self, base: ConfigDict, override: ConfigDict) -> ConfigDict:
        """
        Deep merge two dictionaries.

        Args:
            base: Base configuration dictionary
            override: Override configuration dictionary

        Returns:
            Merged configuration dictionary
        """
        result = copy.deepcopy(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(
                    cast(ConfigDict, result[key]), cast(ConfigDict, value)
                )
            else:
                result[key] = copy.deepcopy(value)
        return result

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.

        Args:
            key_path: Dot-separated path to the configuration value
            default: Default value if key not found

        Returns:
            Configuration value or default

        Example:
            config.get('bot.paper_trading', True)
        """
        keys = key_path.split(".")
        value: Any = self._config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        return value if value is not None else default

    def set(self, key_path: str, value: Any) -> None:
        """
        Set a configuration value using dot notation.

        Args:
            key_path: Dot-separated path to the configuration value
            value: Value to set

        Example:
            config.set('bot.paper_trading', False)
        """
        keys = key_path.split(".")
        current: Any = self._config
        for i, key in enumerate(keys[:-1]):
            if key not in current:
                current[key] = {}
            current = current[key]
            if not isinstance(current, dict):
                raise ValueError(
                    f"Cannot set nested value: '{'.'.join(keys[:i + 1])}' is not a dict"
                )
        current[keys[-1]] = value

    def as_namespace(self) -> ConfigNamespace:
        """
        Get configuration as a namespace for attribute-style access.

        Returns:
            ConfigNamespace object

        Example:
            ns = config.as_namespace()
            paper_trading = ns.bot.paper_trading
        """
        return ConfigNamespace(self._config)

    def to_dict(self) -> ConfigDict:
        """
        Get the entire configuration as a dictionary.

        Returns:
            Configuration dictionary
        """
        return copy.deepcopy(self._config)

    def save(self, file_path: Optional[Path] = None) -> bool:
        """
        Save current configuration to a JSON file.

        Args:
            file_path: Path to save the configuration (uses loaded path if None)

        Returns:
            True if successful, False otherwise
        """
        save_path = file_path or self.config_file_path
        if save_path is None:
            config_logger.error("No file path specified for saving configuration")
            return False
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with save_path.open("w", encoding="utf-8") as f:
                json.dump(self._config, f, indent=2)
            config_logger.info(f"Saved configuration to {save_path}")
            return True
        except OSError as e:
            config_logger.error(f"Error saving config to {save_path}: {e}")
            return False
        except Exception as e:
            config_logger.error(f"Unexpected error saving config: {e}", exc_info=True)
            return False

    def validate(self) -> bool:
        """
        Validate critical configuration values.

        Returns:
            True if configuration is valid, False otherwise
        """
        is_valid = True
        base_currencies = self.get("bot.base_currencies", [])
        if not isinstance(base_currencies, list) or not base_currencies:
            config_logger.error("Invalid or empty bot.base_currencies")
            is_valid = False
        quote_currencies = self.get("bot.quote_currencies", [])
        if not isinstance(quote_currencies, list) or not quote_currencies:
            config_logger.error("Invalid or empty bot.quote_currencies")
            is_valid = False
        min_score = self.get("trading.min_opportunity_score", 0.55)
        if not isinstance(min_score, (int, float)) or not 0 <= min_score <= 1:
            config_logger.error("Invalid trading.min_opportunity_score (must be 0-1)")
            is_valid = False
        position_size = self.get("trading.position_size_percent", 5.0)
        if not isinstance(position_size, (int, float)) or not 0 < position_size <= 100:
            config_logger.error("Invalid trading.position_size_percent (must be 0-100)")
            is_valid = False
        max_alloc = self.get("portfolio.max_allocation_per_asset_percent", 7.5)
        if not isinstance(max_alloc, (int, float)) or not 0 < max_alloc <= 100:
            config_logger.error("Invalid portfolio.max_allocation_per_asset_percent")
            is_valid = False
        min_alloc = self.get("portfolio.min_allocation_per_asset_percent", 1.0)
        if not isinstance(min_alloc, (int, float)) or not 0 < min_alloc <= 100:
            config_logger.error("Invalid portfolio.min_allocation_per_asset_percent")
            is_valid = False
        if is_valid:
            config_logger.info("Configuration validation passed")
        else:
            config_logger.error("Configuration validation failed")
        return is_valid

    def __repr__(self) -> str:
        """String representation of the configuration."""
        return f"BotConfig(config_file={self.config_file_path})"


def get_config(config_file_path: Optional[Path] = None) -> BotConfig:
    """
    Get the BotConfig singleton instance.

    Args:
        config_file_path: Path to configuration file (only used on first call)

    Returns:
        BotConfig instance
    """
    return BotConfig(config_file_path)
