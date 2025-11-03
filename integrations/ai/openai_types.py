"""
Type definitions for OpenAI API integration.

This module contains TypedDict definitions for OpenAI API responses and related data structures.
"""

from typing import Any, List, Literal, Optional, TypedDict


class OpenAIMessage(TypedDict):
    """Represents a message in the OpenAI chat completion."""

    role: Literal["system", "user", "assistant"]
    content: str


class OpenAIUsage(TypedDict):
    """Token usage information from OpenAI API response."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OpenAIChoice(TypedDict):
    """A single choice from the OpenAI API response."""

    index: int
    message: OpenAIMessage
    finish_reason: str


class OpenAIResponse(TypedDict):
    """Complete OpenAI chat completion response."""

    id: str
    object: str
    created: int
    model: str
    choices: List[OpenAIChoice]
    usage: OpenAIUsage


class SentimentResult(TypedDict):
    """Result from sentiment analysis."""

    sentiment: str
    strength: float
    confidence: float
    error: Optional[str]


class StrategyResult(TypedDict):
    """Result from strategy analysis."""

    recommendation: str
    confidence: float
    reasoning: str
    error: Optional[str]


class TechnicalAnalysisResult(TypedDict):
    """Result from technical analysis."""

    signal: str
    confidence: float
    interpretation: str
    error: Optional[str]


class MarketIndicators(TypedDict, total=False):
    """Technical indicators for market data."""

    rsi: float
    macd: float
    sma: float
    ema: float
    bollinger_upper: float
    bollinger_lower: float
    volume: float


class MarketData(TypedDict, total=False):
    """Market data structure for analysis."""

    symbol: str
    current_price: float
    indicators: MarketIndicators
    sentiment_scores: dict
    volume_24h: float
    price_change_24h: float


class NewsArticle(TypedDict, total=False):
    """News article structure."""

    title: str
    description: Optional[str]
    url: str
    publishedAt: str
    source: dict
    content: Optional[str]


class CacheEntry(TypedDict):
    """Cache entry structure."""

    timestamp: float
    data: Any


class ConnectionTestResult(TypedDict, total=False):
    """Result from connection test."""

    status: str
    models_count: int
    default_model_ok: bool
    fallback_model_ok: bool
    timestamp: str
    error: str
