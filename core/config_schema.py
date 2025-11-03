"""
Enhanced Pydantic Configuration Schema with Strict Validation.

Provides comprehensive type checking, validation, and coercion for all configuration
settings following best practices from the review.
"""

import os
from decimal import Decimal
from typing import Any, Dict, List, Literal, Optional

try:  # Python 3.11+
    from typing import Self
except ImportError:  # Python 3.10 and below
    from typing_extensions import Self

from pydantic import BaseModel, Field, field_validator, model_validator


class KrakenCredentials(BaseModel):
    """Kraken API credentials loaded from environment."""

    api_key: Optional[str] = Field(default=None, description="Kraken API key")
    api_secret: Optional[str] = Field(default=None, description="Kraken API secret")

    @model_validator(mode="after")
    def validate_credentials(self) -> Self:
        """Validate that both key and secret are provided together."""
        if (self.api_key is None) != (self.api_secret is None):
            raise ValueError("Both api_key and api_secret must be provided together for Kraken")
        return self


class TwitterCredentials(BaseModel):
    """Twitter API credentials loaded from environment."""

    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    access_token: Optional[str] = None
    access_token_secret: Optional[str] = None
    bearer_token: Optional[str] = None

    @model_validator(mode="after")
    def validate_credentials(self) -> Self:
        """Validate that all required Twitter credentials are provided together."""
        fields = [
            self.api_key,
            self.api_secret,
            self.access_token,
            self.access_token_secret,
            self.bearer_token,
        ]
        if any((f is not None for f in fields)) and (not all((f is not None for f in fields))):
            raise ValueError("All Twitter credentials must be provided together")
        return self


class RedditCredentials(BaseModel):
    """Reddit API credentials loaded from environment."""

    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    user_agent: str = "TradingBot/2.0"
    username: Optional[str] = None
    password: Optional[str] = None

    @model_validator(mode="after")
    def validate_credentials(self) -> Self:
        """Validate that all required Reddit credentials are provided together."""
        if (self.client_id is None) != (self.client_secret is None):
            raise ValueError(
                "Both client_id and client_secret must be provided together for Reddit"
            )
        if self.client_id is not None and (self.username is None or self.password is None):
            raise ValueError("Username and password required when Reddit credentials are provided")
        return self


class NewsCredentials(BaseModel):
    """News API credentials loaded from environment."""

    api_key: Optional[str] = None


class OpenAICredentials(BaseModel):
    """OpenAI API credentials loaded from environment."""

    api_key: Optional[str] = None


class APICredentials(BaseModel):
    """All API credentials container."""

    kraken: KrakenCredentials = Field(default_factory=KrakenCredentials)
    twitter: TwitterCredentials = Field(default_factory=TwitterCredentials)
    reddit: RedditCredentials = Field(default_factory=RedditCredentials)
    news: NewsCredentials = Field(default_factory=NewsCredentials)
    openai: OpenAICredentials = Field(default_factory=OpenAICredentials)


class BotSettings(BaseModel):
    """Core bot operational settings."""

    base_currencies: List[str] = Field(min_length=1, description="List of base currencies to trade")
    quote_currencies: List[str] = Field(min_length=1, description="List of quote currencies")
    timeframe: str = Field(default="1h", pattern="^(1m|5m|15m|30m|1h|4h|1d)$")
    strategy: Literal["combined", "technical", "sentiment", "ai"] = "combined"
    paper_trading: bool = True
    trading_interval_seconds: int = Field(default=300, ge=30, le=3600)
    max_concurrent_pairs: int = Field(default=5, ge=1, le=20)
    max_cycles: Optional[int] = Field(default=None, ge=1)
    disable_twitter: bool = True

    @field_validator("base_currencies", "quote_currencies")
    @classmethod
    def validate_currencies(cls, v: Any) -> Any:
        """Ensure currencies are uppercase."""
        return [c.upper() for c in v]


class PaperTradingSettings(BaseModel):
    """Paper trading specific settings."""

    initial_capital: Decimal = Field(
        default=Decimal("10000.0"), gt=0, description="Starting capital for paper trading"
    )


class ConfidenceBands(BaseModel):
    """Confidence band ranges."""

    low: List[float] = Field(default=[0.0, 0.4], min_length=2, max_length=2)
    medium: List[float] = Field(default=[0.4, 0.7], min_length=2, max_length=2)
    high: List[float] = Field(default=[0.7, 1.0], min_length=2, max_length=2)

    @field_validator("low", "medium", "high")
    @classmethod
    def validate_range(cls, v: Any) -> Any:
        """Ensure ranges are valid [min, max] with min < max."""
        if v[0] >= v[1]:
            raise ValueError(f"Range must have min < max, got {v}")
        if not (0.0 <= v[0] <= 1.0 and 0.0 <= v[1] <= 1.0):
            raise ValueError(f"Range values must be between 0.0 and 1.0, got {v}")
        return v


class ConfidenceBandMultipliers(BaseModel):
    """Multipliers for each confidence band."""

    low: float = Field(default=1.05, gt=0, le=2.0)
    medium: float = Field(default=1.0, gt=0, le=2.0)
    high: float = Field(default=0.95, gt=0, le=2.0)


class TradingSettings(BaseModel):
    """Trading strategy and execution settings."""

    min_opportunity_score: float = Field(default=0.6, ge=0.0, le=1.0)
    position_size_percent: float = Field(default=5.0, gt=0, le=100)
    volatility_lookback_period: int = Field(default=14, ge=5, le=100)
    performance_lookback_period: int = Field(default=30, ge=10, le=365)
    confidence_bands: ConfidenceBands = Field(default_factory=ConfidenceBands)
    confidence_band_multipliers: ConfidenceBandMultipliers = Field(
        default_factory=ConfidenceBandMultipliers
    )
    dynamic_threshold_enabled: bool = True
    rebalance_threshold_percent: float = Field(default=5.0, ge=0, le=50)


class TechnicalFactorThresholds(BaseModel):
    """Thresholds for technical indicators."""

    rsi_oversold: int = Field(default=30, ge=0, le=100)
    rsi_overbought: int = Field(default=70, ge=0, le=100)
    macd_threshold: float = 0.0
    bb_threshold_percent: float = Field(default=5.0, ge=0, le=50)

    @model_validator(mode="after")
    def validate_rsi_thresholds(self) -> Self:
        """Ensure RSI oversold < overbought."""
        if self.rsi_oversold >= self.rsi_overbought:
            raise ValueError("rsi_oversold must be less than rsi_overbought")
        return self


class RiskManagement(BaseModel):
    """Risk management parameters."""

    var_confidence_level: float = Field(default=0.95, gt=0, lt=1.0)
    var_time_horizon_days: int = Field(default=1, ge=1, le=30)
    use_monte_carlo_var: bool = False
    stress_test_scenarios_percent: Dict[str, float] = Field(
        default_factory=lambda: {
            "market_crash": -20.0,
            "moderate_decline": -10.0,
            "sector_rotation": -15.0,
        }
    )

    @field_validator("stress_test_scenarios_percent")
    @classmethod
    def validate_scenarios(cls, v: Any) -> Any:
        """Ensure stress test percentages are within reasonable bounds."""
        for scenario, pct in v.items():
            if not -100.0 <= pct <= 100.0:
                raise ValueError(f"Stress test scenario '{scenario}' has invalid percentage: {pct}")
        return v


class SimulationSettings(BaseModel):
    """Simulation parameters for backtesting and paper trading."""

    fee_rate_percent: float = Field(default=0.1, ge=0, le=5.0)
    slippage_percent: float = Field(default=0.2, ge=0, le=10.0)
    min_order_value_quote: Decimal = Field(default=Decimal("10.0"), gt=0)


class PortfolioSettings(BaseModel):
    """Portfolio management and allocation settings."""

    max_allocation_per_asset_percent: float = Field(default=7.5, gt=0, le=100)
    min_allocation_per_asset_percent: float = Field(default=1.0, gt=0, le=100)
    target_allocation_weights: Dict[str, float] = Field(
        default_factory=lambda: {"technical": 0.4, "sentiment": 0.3, "ai": 0.3}
    )
    technical_factor_thresholds: TechnicalFactorThresholds = Field(
        default_factory=TechnicalFactorThresholds
    )
    risk_management: RiskManagement = Field(default_factory=RiskManagement)
    simulation: SimulationSettings = Field(default_factory=SimulationSettings)

    @model_validator(mode="after")
    def validate_allocation(self) -> Self:
        """Ensure min < max allocation and weights sum to 1.0."""
        if self.min_allocation_per_asset_percent >= self.max_allocation_per_asset_percent:
            raise ValueError("min_allocation must be less than max_allocation")
        weight_sum = sum(self.target_allocation_weights.values())
        if not 0.99 <= weight_sum <= 1.01:
            raise ValueError(f"target_allocation_weights must sum to 1.0, got {weight_sum}")
        return self


class CacheExpirySettings(BaseModel):
    """Cache expiry times for different data types."""

    ticker: int = Field(default=30, ge=1)
    ohlcv: int = Field(default=300, ge=1)
    order_book: int = Field(default=10, ge=1)
    balance: int = Field(default=120, ge=1)
    markets: int = Field(default=3600, ge=1)


class KrakenClientSettings(BaseModel):
    """Kraken client configuration with explicit rate limiting."""

    request_timeout_ms: int = Field(default=30000, ge=1000, le=120000)
    max_retries: int = Field(default=3, ge=1, le=10)
    retry_delay_seconds: float = Field(default=2.0, ge=0.1, le=60.0)
    cache_expiry_seconds: CacheExpirySettings = Field(default_factory=CacheExpirySettings)
    enable_rate_limit: bool = Field(default=True, description="Enable CCXT rate limiting")
    rate_limit_per_second: Optional[float] = Field(
        default=None, ge=0.1, description="Custom rate limit override"
    )


class NewsClientSettings(BaseModel):
    """News API client configuration."""

    request_timeout_seconds: int = Field(default=15, ge=5, le=120)
    max_results_per_query: int = Field(default=20, ge=1, le=100)
    search_days_back: int = Field(default=1, ge=1, le=30)
    cache_expiry_seconds: int = Field(default=1800, ge=60)
    debug_mode: bool = False
    asset_keywords: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            "DEFAULT": ["cryptocurrency", "crypto"],
            "BTC": ["Bitcoin", "BTC"],
            "ETH": ["Ethereum", "ETH", "Ether"],
            "XRP": ["XRP", "Ripple"],
            "LTC": ["Litecoin", "LTC"],
            "DOGE": ["Dogecoin", "DOGE"],
            "DOT": ["Polkadot", "DOT"],
            "SOL": ["Solana", "SOL"],
            "ADA": ["Cardano", "ADA"],
        }
    )


class RedditClientSettings(BaseModel):
    """Reddit client configuration."""

    limit_per_subreddit: int = Field(default=10, ge=1, le=100)
    search_time_filter: Literal["hour", "day", "week", "month", "year", "all"] = "week"
    cache_expiry_seconds: int = Field(default=1800, ge=60)


class TemperatureSettings(BaseModel):
    """OpenAI temperature settings for different analysis types."""

    analysis: float = Field(default=0.3, ge=0.0, le=2.0)
    strategy: float = Field(default=0.3, ge=0.0, le=2.0)
    technical: float = Field(default=0.3, ge=0.0, le=2.0)


class MaxTokensSettings(BaseModel):
    """OpenAI max tokens settings for different analysis types."""

    analysis: int = Field(default=700, ge=100, le=4000)
    strategy: int = Field(default=700, ge=100, le=4000)
    technical: int = Field(default=700, ge=100, le=4000)


class OpenAIClientSettings(BaseModel):
    """OpenAI client configuration."""

    request_timeout_seconds: int = Field(default=60, ge=10, le=300)
    model: str = Field(default="gpt-4o-mini")
    fallback_model: str = Field(default="gpt-3.5-turbo")
    max_retries: int = Field(default=3, ge=1, le=10)
    cache_expiry_seconds: int = Field(default=900, ge=60)
    temperature: TemperatureSettings = Field(default_factory=TemperatureSettings)
    max_tokens: MaxTokensSettings = Field(default_factory=MaxTokensSettings)
    concurrency_limit: int = Field(default=3, ge=1, le=10)


class SentimentSettings(BaseModel):
    """Sentiment analysis configuration."""

    bullish_threshold: float = Field(default=0.05, ge=-1.0, le=1.0)
    bearish_threshold: float = Field(default=-0.05, ge=-1.0, le=1.0)
    history_length: int = Field(default=30, ge=1, le=365)
    trend_window: int = Field(default=5, ge=1, le=30)
    volume_baseline_count: int = Field(default=20, ge=5, le=100)

    @model_validator(mode="after")
    def validate_thresholds(self) -> Self:
        """Ensure bearish < bullish threshold."""
        if self.bearish_threshold >= self.bullish_threshold:
            raise ValueError("bearish_threshold must be less than bullish_threshold")
        return self


class LoggingSettings(BaseModel):
    """Logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_to_file: bool = True
    log_filename: str = "trading_bot.log"
    rotation_interval: Literal["midnight", "H", "D", "W0"] = "midnight"
    rotation_backup_count: int = Field(default=7, ge=1, le=365)


class PathSettings(BaseModel):
    """File system paths configuration."""

    data: str = "data"
    logs: str = "data/logs"
    cache: str = "data/cache"
    trades: str = "data/trades"
    metrics: str = "data/metrics"


class ConfigSchema(BaseModel):
    """
    Complete configuration schema with strict validation.

    This schema enforces type checking, range validation, and business logic
    constraints across all configuration settings.
    """

    model_config = {"extra": "forbid", "validate_assignment": True}
    api_credentials: APICredentials = Field(default_factory=APICredentials)
    bot: BotSettings
    paper_trading_settings: PaperTradingSettings = Field(default_factory=PaperTradingSettings)
    trading: TradingSettings = Field(default_factory=TradingSettings)
    portfolio: PortfolioSettings = Field(default_factory=PortfolioSettings)
    kraken_client: KrakenClientSettings = Field(default_factory=KrakenClientSettings)
    news_client: NewsClientSettings = Field(default_factory=NewsClientSettings)
    reddit_client: RedditClientSettings = Field(default_factory=RedditClientSettings)
    openai_client: OpenAIClientSettings = Field(default_factory=OpenAIClientSettings)
    sentiment: SentimentSettings = Field(default_factory=SentimentSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    paths: PathSettings = Field(default_factory=PathSettings)

    @classmethod
    def from_env_and_file(cls, config_dict: Dict) -> "ConfigSchema":
        """
        Create ConfigSchema from config dict, loading secrets from environment.

        Args:
            config_dict: Configuration dictionary from JSON file

        Returns:
            Validated ConfigSchema instance
        """
        api_creds = {
            "kraken": {
                "api_key": os.getenv("KRAKEN_API_KEY"),
                "api_secret": os.getenv("KRAKEN_API_SECRET"),
            },
            "twitter": {
                "api_key": os.getenv("TWITTER_API_KEY"),
                "api_secret": os.getenv("TWITTER_API_SECRET"),
                "access_token": os.getenv("TWITTER_ACCESS_TOKEN"),
                "access_token_secret": os.getenv("TWITTER_ACCESS_TOKEN_SECRET"),
                "bearer_token": os.getenv("TWITTER_BEARER_TOKEN"),
            },
            "reddit": {
                "client_id": os.getenv("REDDIT_CLIENT_ID"),
                "client_secret": os.getenv("REDDIT_CLIENT_SECRET"),
                "user_agent": os.getenv("REDDIT_USER_AGENT", "TradingBot/2.0"),
                "username": os.getenv("REDDIT_USERNAME"),
                "password": os.getenv("REDDIT_PASSWORD"),
            },
            "news": {"api_key": os.getenv("NEWS_API_KEY")},
            "openai": {"api_key": os.getenv("OPENAI_API_KEY")},
        }
        config_dict["api_credentials"] = api_creds
        return cls(**config_dict)
