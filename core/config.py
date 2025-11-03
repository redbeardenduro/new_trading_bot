"""
Centralized Configuration Management.

Loads configuration settings from user_config.json, provides default values,
validates critical settings, and makes them accessible throughout the
application via the BotConfig singleton instance.
"""

import copy
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

root_logger = logging.getLogger()
if not root_logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
config_logger = logging.getLogger("config_loader")
DEFAULT_CONFIG = {
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
    "security": {"api_jwt_secret": None, "dash_api_key": None},
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

    def __init__(self, data: Dict[str, Any]) -> None:
        self._data = data

    def __getattr__(self, name: str) -> Any:
        if name in self._data:
            value = self._data[name]
            if isinstance(value, dict):
                return ConfigNamespace(value)
            return value
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __repr__(self) -> str:
        return repr(self._data)

    def __contains__(self, item: str) -> bool:
        return item in self._data

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)


class BotConfig:
    """
    Loads, validates, and provides access to bot configuration settings. Singleton pattern.
    """

    _instance: Optional["BotConfig"] = None
    _initialized: bool

    def __new__(cls, config_path: Optional[Union[str, Path]] = None) -> "BotConfig":
        """Implement Singleton pattern."""
        if cls._instance is None:
            cls._instance = super(BotConfig, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """
        Initializes the BotConfig instance (only once). Loads and validates config.

        Args:
            config_path (Optional[Union[str, Path]]): Path to the user_config.json file.
                                                     If None, defaults relative to project root.
        """
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._initialized = True
        self.settings: Dict[str, Any] = {}
        try:
            self.project_root = Path(__file__).resolve().parent.parent
        except NameError:
            self.project_root = Path(".").resolve()
        if config_path is None:
            self.config_file_path = self.project_root / "config" / "user_config.json"
        else:
            try:
                self.config_file_path = Path(config_path).resolve()
            except Exception as e:
                config_logger.error(
                    f"Invalid config path provided: {config_path}. Error: {e}. Using default path."
                )
                self.config_file_path = self.project_root / "config" / "user_config.json"
        try:
            self._load_config()
            self._process_percentage_values()
            self._validate_config()
            config_logger.info("Configuration loaded, processed, and validated successfully.")
            config_logger.info(f"Trading Mode: {('PAPER' if self.bot.paper_trading else 'LIVE')}")
            config_logger.info(f"Base Currencies: {self.bot.base_currencies}")
            config_logger.info(f"Logging Level: {self.logging.level}")
        except Exception as e:
            config_logger.critical(
                f"CRITICAL ERROR during BotConfig initialization: {e}", exc_info=True
            )
            config_logger.warning(
                "Proceeding with default configuration due to initialization error."
            )
            self.settings = copy.deepcopy(DEFAULT_CONFIG)
            try:
                self._process_percentage_values()
                self._validate_config()
            except Exception as nested_e:
                config_logger.error(f"Error processing/validating default config: {nested_e}")

    def _load_config(self) -> None:
        """Loads user config from JSON, merges with defaults. Handles file/JSON errors."""
        user_config = {}
        if self.config_file_path.exists() and self.config_file_path.is_file():
            try:
                with self.config_file_path.open("r", encoding="utf-8") as f:
                    content_lines = f.readlines()
                    content_no_comments = "".join(
                        (line for line in content_lines if not line.strip().startswith("//"))
                    )
                    if content_no_comments.strip():
                        user_config = json.loads(content_no_comments)
                        config_logger.info(
                            f"Successfully loaded user configuration from: {self.config_file_path}"
                        )
                    else:
                        config_logger.info(
                            f"User config file {self.config_file_path} is empty or only contains comments. Using defaults."
                        )
            except json.JSONDecodeError as e:
                config_logger.error(
                    f"Error decoding JSON from {self.config_file_path}: {e}. Fix the JSON or use defaults."
                )
                user_config = {}
            except OSError as e:
                config_logger.error(
                    f"OS error reading config file {self.config_file_path}: {e}. Using defaults."
                )
                user_config = {}
            except Exception as e:
                config_logger.error(
                    f"Unexpected error reading config file {self.config_file_path}: {e}. Using defaults.",
                    exc_info=True,
                )
                user_config = {}
        else:
            config_logger.warning(
                f"Configuration file not found at {self.config_file_path}. Using default settings."
            )
        merged_config = self._deep_merge_dicts(copy.deepcopy(DEFAULT_CONFIG), user_config)
        env_config = self._load_env_config()
        self.settings = self._deep_merge_dicts(merged_config, env_config)

    def _deep_merge_dicts(self, base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge overlay dict into base dict."""
        if not isinstance(base, dict) or not isinstance(overlay, dict):
            return overlay
        for key, value in overlay.items():
            if key in base:
                if isinstance(base[key], dict) and isinstance(value, dict):
                    base[key] = self._deep_merge_dicts(base[key], value)
                elif value is not None:
                    base[key] = value
            else:
                base[key] = value
        return base

    def _load_env_config(self) -> Dict[str, Any]:
        """Loads configuration from environment variables, converting types as needed."""
        env_config: Dict[str, Any] = {}
        for section, section_data in DEFAULT_CONFIG.items():
            if not isinstance(section_data, dict):
                continue
            for key, default_value in section_data.items():
                if section == "security" and key == "api_jwt_secret":
                    env_var_name = "API_JWT_SECRET"
                elif section == "security" and key == "dash_api_key":
                    env_var_name = "DASH_API_KEY"
                else:
                    env_var_name = f"{section.upper()}_{key.upper()}"
                env_value = os.getenv(env_var_name)
                if env_value is not None:
                    try:
                        converted_value: Any
                        if isinstance(default_value, bool):
                            converted_value = env_value.lower() in ("true", "1", "t", "y", "yes")
                        elif isinstance(default_value, int):
                            converted_value = int(env_value)
                        elif isinstance(default_value, float):
                            converted_value = float(env_value)
                        elif isinstance(default_value, list):
                            converted_value = [item.strip() for item in env_value.split(",")]
                        else:
                            converted_value = env_value
                        if section not in env_config:
                            env_config[section] = {}
                        env_config[section][key] = converted_value
                        config_logger.info(f"Loaded {env_var_name} from environment variables.")
                    except ValueError:
                        config_logger.warning(
                            f'Could not convert environment variable {env_var_name}="{env_value}" to expected type. Skipping.'
                        )
        return env_config

    def _process_percentage_values(self) -> None:
        """Converts percentage values from config to decimal multipliers safely."""
        percentages_to_convert = [
            (("trading",), "position_size_percent", "position_size_multiplier"),
            (("trading",), "rebalance_threshold_percent", "rebalance_threshold_multiplier"),
            (
                ("portfolio",),
                "max_allocation_per_asset_percent",
                "max_allocation_per_asset_multiplier",
            ),
            (
                ("portfolio",),
                "min_allocation_per_asset_percent",
                "min_allocation_per_asset_multiplier",
            ),
            (
                ("portfolio", "technical_factor_thresholds"),
                "bb_threshold_percent",
                "bb_threshold_multiplier",
            ),
            (("portfolio", "simulation"), "fee_rate_percent", "fee_rate_multiplier"),
            (("portfolio", "simulation"), "slippage_percent", "slippage_multiplier"),
            (
                ("portfolio", "risk_management"),
                "stress_test_scenarios_percent",
                "stress_test_scenarios_multiplier",
            ),
        ]
        for path_info in percentages_to_convert:
            try:
                target_dict = self.settings
                path_keys = path_info[0]
                percent_key = path_info[1]
                multiplier_key = path_info[2]
                valid_path = True
                for key in path_keys:
                    next_dict = target_dict.get(key)
                    if not isinstance(next_dict, dict):
                        valid_path = False
                        break
                    target_dict = next_dict
                if not valid_path:
                    continue
                percentage_value = target_dict.get(percent_key)
                if percentage_value is None:
                    continue
                if isinstance(percentage_value, dict):
                    multiplier_dict = {}
                    for k, v_percent in percentage_value.items():
                        if isinstance(v_percent, (int, float)):
                            if -1000.0 <= v_percent <= 1000.0:
                                multiplier_dict[k] = v_percent / 100.0
                            else:
                                config_logger.warning(
                                    f"Percentage value {v_percent}% for '{k}' in '{percent_key}' is outside expected range [-1000, 1000]. Using 0."
                                )
                                multiplier_dict[k] = 0.0
                        else:
                            config_logger.warning(
                                f"Non-numeric value '{v_percent}' found for key '{k}' in percentage dict '{percent_key}'. Using 0."
                            )
                            multiplier_dict[k] = 0.0
                    target_dict[multiplier_key] = multiplier_dict
                elif isinstance(percentage_value, (int, float)):
                    if 0.0 <= percentage_value <= 100.0:
                        target_dict[multiplier_key] = percentage_value / 100.0
                    elif -1000.0 <= percentage_value < 0.0:
                        target_dict[multiplier_key] = percentage_value / 100.0
                    else:
                        config_logger.warning(
                            f"Percentage value {percentage_value}% for '{percent_key}' is outside expected range [0, 100] or negative range. Using 0."
                        )
                        target_dict[multiplier_key] = 0.0
                else:
                    config_logger.warning(
                        f"Invalid type '{type(percentage_value)}' for percentage key '{percent_key}'. Skipping conversion."
                    )
            except Exception as e:
                config_logger.error(
                    f"Error processing percentage conversion for path {path_info}: {e}",
                    exc_info=True,
                )

    def _validate_config(self) -> None:
        """Validates the configuration using the Pydantic schema."""
        try:
            from .config_schema import ConfigSchema

            ConfigSchema(**self.settings)
            config_logger.info("Configuration validated successfully against Pydantic schema.")
        except Exception as e:
            config_logger.error(f"Configuration validation failed: {e}")
            raise ValueError(f"Invalid configuration: {e}")
        config_logger.debug("Running configuration validation checks...")
        validation_errors = 0

        def check_numeric(
            section_name: str,
            key: str,
            min_val: Optional[float] = None,
            max_val: Optional[float] = None,
            is_int: bool = False,
            default_on_fail: Optional[Any] = None,
        ) -> bool:
            nonlocal validation_errors
            value = self.get(f"{section_name}.{key}")
            valid = True
            error_msg = None
            try:
                if value is None:
                    raise TypeError("Value is None")
                num_value = int(value) if is_int else float(value)
                if min_val is not None and num_value < min_val:
                    valid = False
                    error_msg = f"below min {min_val}"
                if max_val is not None and num_value > max_val:
                    valid = False
                    error_msg = f"above max {max_val}"
            except (ValueError, TypeError) as e:
                valid = False
                error_msg = f"invalid type ({type(value).__name__}): {e}"
            if not valid:
                config_logger.error(
                    f"Config Validation Error: '{section_name}.{key}' ({value}) is invalid ({error_msg})."
                )
                validation_errors += 1
            return valid

        def check_string_enum(
            section_name: str, key: str, allowed_values: List[str], case_sensitive: bool = False
        ) -> bool:
            nonlocal validation_errors
            value = self.get(f"{section_name}.{key}")
            valid = True
            if not isinstance(value, str):
                valid = False
            else:
                check_val = value if case_sensitive else value.upper()
                check_allowed = (
                    allowed_values if case_sensitive else [v.upper() for v in allowed_values]
                )
                if check_val not in check_allowed:
                    valid = False
            if not valid:
                config_logger.error(
                    f"Config Validation Error: '{section_name}.{key}' ('{value}') must be one of {allowed_values}."
                )
                validation_errors += 1
            return valid

        if not isinstance(self.get("bot.base_currencies"), list) or not self.get(
            "bot.base_currencies"
        ):
            config_logger.error(
                "Config Validation Error: 'bot.base_currencies' must be a non-empty list."
            )
            validation_errors += 1
        if not isinstance(self.get("bot.quote_currencies"), list) or not self.get(
            "bot.quote_currencies"
        ):
            config_logger.error(
                "Config Validation Error: 'bot.quote_currencies' must be a non-empty list."
            )
            validation_errors += 1
        check_numeric("bot", "trading_interval_seconds", min_val=10, is_int=True)
        check_numeric("bot", "max_concurrent_pairs", min_val=1, is_int=True)
        check_string_enum("bot", "strategy", ["technical", "sentiment", "ai", "combined"])
        check_numeric("paper_trading_settings", "initial_capital", min_val=0.0)
        check_numeric("trading", "min_opportunity_score", min_val=0.0, max_val=1.0)
        check_numeric("trading", "position_size_multiplier", min_val=0.0, max_val=1.0)
        check_numeric("trading", "volatility_lookback_period", min_val=2, is_int=True)
        check_numeric("trading", "rebalance_threshold_multiplier", min_val=0.0, max_val=0.5)
        check_numeric("portfolio", "max_allocation_per_asset_multiplier", min_val=0.0, max_val=1.0)
        check_numeric("portfolio", "min_allocation_per_asset_multiplier", min_val=0.0, max_val=1.0)
        weights = self.get("portfolio.target_allocation_weights", {})
        if isinstance(weights, dict):
            total_weight = sum((float(v) for v in weights.values() if isinstance(v, (int, float))))
            if abs(total_weight - 1.0) > 1e-06:
                config_logger.warning(
                    f"Config Validation Warning: 'portfolio.target_allocation_weights' sum ({total_weight:.3f}) is not 1.0."
                )
        else:
            config_logger.error(
                "Config Validation Error: 'portfolio.target_allocation_weights' must be a dictionary."
            )
            validation_errors += 1
        check_numeric("portfolio.simulation", "fee_rate_multiplier", min_val=0.0, max_val=0.1)
        check_numeric("portfolio.simulation", "slippage_multiplier", min_val=0.0, max_val=0.1)
        check_numeric("portfolio.simulation", "min_order_value_quote", min_val=0.0)
        keywords = self.get("news_client.asset_keywords", {})
        if not isinstance(keywords, dict):
            config_logger.error(
                "Config Validation Error: 'news_client.asset_keywords' must be a dictionary."
            )
            validation_errors += 1
        else:
            for asset, terms in keywords.items():
                if not isinstance(terms, list) or not all((isinstance(t, str) for t in terms)):
                    config_logger.error(
                        f"Config Validation Error: Keywords for asset '{asset}' in 'news_client.asset_keywords' must be a list of strings."
                    )
                    validation_errors += 1
        if not isinstance(self.get("news_client.debug_mode", False), bool):
            config_logger.error(
                "Config Validation Error: 'news_client.debug_mode' must be true or false."
            )
            validation_errors += 1
        check_string_enum(
            "logging",
            "level",
            ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            case_sensitive=False,
        )
        check_numeric("logging", "rotation_backup_count", min_val=0, is_int=True)
        if not self.get("bot.paper_trading"):
            if not self.get("api_credentials.kraken.api_key") or not self.get(
                "api_credentials.kraken.api_secret"
            ):
                config_logger.critical(
                    "Config Validation CRITICAL: LIVE TRADING ENABLED but Kraken API credentials are missing!"
                )
                validation_errors += 1
        if not self.get("security.api_jwt_secret"):
            config_logger.critical(
                "Config Validation CRITICAL: API_JWT_SECRET is missing. JWT authentication will not work."
            )
            validation_errors += 1
        if not self.get("security.dash_api_key"):
            config_logger.critical(
                "Config Validation CRITICAL: DASH_API_KEY is missing. Token minting will not work."
            )
            validation_errors += 1
        if validation_errors > 0:
            config_logger.error(
                f"Configuration validation failed with {validation_errors} errors. Review config file."
            )
        else:
            config_logger.info("Configuration validation passed.")

    def __getattr__(self, name: str) -> Any:
        """Allow accessing config sections like attributes (e.g., config.bot)."""
        if name in self.settings:
            value = self.settings[name]
            if isinstance(value, dict):
                return ConfigNamespace(value)
            return value
        raise AttributeError(f"'{type(self).__name__}' object has no attribute or key '{name}'")

    def get(self, key: str, default: Any = None) -> Any:
        """Provide dict-like get access with dot notation and a default value."""
        keys = key.split(".")
        value: Any = self.settings
        try:
            for k in keys:
                if isinstance(value, dict):
                    value = value.get(k)
                    if value is None:
                        return default
                else:
                    return default
            return value
        except Exception as e:
            config_logger.warning(f"Error accessing config key '{key}': {e}")
            return default


if __name__ == "__main__":
    print("--- Testing BotConfig ---")
    print("\n1. Testing with potential user_config.json:")
    try:
        config_default = BotConfig()
        print(f"   Paper Trading: {config_default.bot.paper_trading}")
        print(f"   Paper Initial Cap: {config_default.paper_trading_settings.initial_capital}")
        print(f"   Base Currencies: {config_default.bot.base_currencies}")
        print(
            f"   Kraken API Key: {(config_default.api_credentials.kraken.api_key if config_default.api_credentials.kraken else 'N/A')}"
        )
        print(f"   Position Size Multiplier: {config_default.trading.position_size_multiplier}")
        print(f"   Logging Level: {config_default.logging.level}")
        print(f"   OpenAI Model: {config_default.openai_client.model}")
        print(f"   Fee Rate Multiplier: {config_default.portfolio.simulation.fee_rate_multiplier}")
        print(
            f"   News Asset Keywords (BTC): {config_default.news_client.asset_keywords.get('BTC')}"
        )
        print(f"   News Debug Mode: {config_default.news_client.debug_mode}")
        print(f"   Using get('bot.timeframe'): {config_default.get('bot.timeframe', 'default_tf')}")
        print(
            f"   Using get('paper_trading_settings.initial_capital'): {config_default.get('paper_trading_settings.initial_capital', 0.0)}"
        )
        print(
            f"   Using get('news_client.asset_keywords.ETH'): {config_default.get('news_client.asset_keywords.ETH', [])}"
        )
        print(
            f"   Using get('news_client.debug_mode'): {config_default.get('news_client.debug_mode')}"
        )
        print(
            f"   Using get('nonexistent.key', 'default_val'): {config_default.get('nonexistent.key', 'default_val')}"
        )
        print(
            f"   Using get('portfolio.risk_management.var_confidence_level'): {config_default.get('portfolio.risk_management.var_confidence_level')}"
        )
        print(f"   Using get('logging.level'): {config_default.get('logging.level')}")
        print(f"   Using get('trading'): {config_default.get('trading')}")
    except Exception as e:
        print(f"   ERROR loading/accessing default config: {e}")
        config_logger.exception("Exception during default config test:")
    print("\n2. Testing with an invalid dummy user_config.json:")
    dummy_config_path_invalid = Path("./dummy_user_config_invalid.json")
    invalid_json_content = '\n    {\n        "bot": { "paper_trading": "maybe", "trading_interval_seconds": "fast" }, // Invalid types\n        "paper_trading_settings": { "initial_capital": "lots" }, // Invalid numeric type\n        "news_client": { "asset_keywords": "not a dict", "debug_mode": "yes" }, // Invalid types\n        "trading": { "position_size_percent": 150 }, // Invalid range\n        "logging": { "level": "VERBOSE" } // Invalid enum value\n        "portfolio": { // Missing comma\n            "max_allocation_per_asset_percent": "20%" // Invalid type for percentage processing\n        }\n    }\n    '
    try:
        with dummy_config_path_invalid.open("w") as f:
            f.write(invalid_json_content)
        print(f"   Created invalid dummy config file: {dummy_config_path_invalid}")
        BotConfig._instance = None
        config_invalid = BotConfig(config_path=dummy_config_path_invalid)
        print(f"   Paper Trading (invalid load): {config_invalid.bot.paper_trading}")
        print(
            f"   Paper Initial Cap (invalid load): {config_invalid.paper_trading_settings.initial_capital}"
        )
        print(f"   News Keywords (invalid load): {config_invalid.news_client.asset_keywords}")
        print(f"   News Debug Mode (invalid load): {config_invalid.news_client.debug_mode}")
        print(f"   Trading Interval (invalid load): {config_invalid.bot.trading_interval_seconds}")
        print(
            f"   Position Size Multiplier (invalid load): {config_invalid.trading.position_size_multiplier}"
        )
        print(f"   Logging Level (invalid load): {config_invalid.logging.level}")
        print(
            f"   Max Alloc Multiplier (invalid load): {config_invalid.portfolio.max_allocation_per_asset_multiplier}"
        )
    except Exception as e:
        print(f"   ERROR loading invalid dummy config: {e}")
        config_logger.exception("Exception during invalid config test:")
    finally:
        if dummy_config_path_invalid.exists():
            dummy_config_path_invalid.unlink()
            print(f"   Removed invalid dummy config file: {dummy_config_path_invalid}")
    print("\n--- BotConfig Test Complete ---")
