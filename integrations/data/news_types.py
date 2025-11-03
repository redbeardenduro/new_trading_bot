"""
Type definitions for News API integration.

This module contains TypedDict definitions for NewsAPI responses and related data structures.
"""
from typing import Any, List, Optional, TypedDict


class NewsSource(TypedDict, total=False):
    """News article source information."""
    id: Optional[str]
    name: str

class NewsArticle(TypedDict, total=False):
    """Individual news article from NewsAPI."""
    source: NewsSource
    author: Optional[str]
    title: str
    description: Optional[str]
    url: str
    urlToImage: Optional[str]
    publishedAt: str
    content: Optional[str]

class NewsAPIResponse(TypedDict):
    """Complete NewsAPI response."""
    status: str
    totalResults: int
    articles: List[NewsArticle]
    code: Optional[str]
    message: Optional[str]

class SentimentScore(TypedDict):
    """Sentiment score for a single article."""
    score: float
    confidence: float
    label: str

class AnalyzedArticle(TypedDict):
    """News article with sentiment analysis."""
    headline: str
    source_name: str
    published_at: str
    url: str
    sentiment_label: str
    confidence_score: float
    model: str
    sentiment_score: float

class NewsSentimentResult(TypedDict):
    """Result from news sentiment analysis."""
    sentiment_score: float
    confidence: float
    article_count: int
    analyzed_articles: List[AnalyzedArticle]
    statistics: dict
    error: Optional[str]

class NewsCacheEntry(TypedDict):
    """Cache entry for news data."""
    timestamp: float
    data: Any

class NewsAPIConfig(TypedDict, total=False):
    """Configuration for NewsAPI."""
    api_key: str
    base_url: str
    request_timeout: int
    max_retries: int
    cache_expiry: int
    articles_per_request: int
    lookback_days: int