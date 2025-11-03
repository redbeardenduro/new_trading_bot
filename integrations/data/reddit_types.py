"""
Type definitions for Reddit API integration.

This module contains TypedDict definitions for Reddit (PRAW) responses and related data structures.
"""
from typing import Any, List, Optional, Protocol, TypedDict


class RedditSubmissionProtocol(Protocol):
    """Protocol for Reddit submission objects from PRAW."""
    id: str
    title: str
    selftext: str
    score: int
    created_utc: float
    num_comments: int
    url: str
    subreddit: Any
    author: Any

    def __getattribute__(self, name: str) -> Any:
        ...

class RedditCommentProtocol(Protocol):
    """Protocol for Reddit comment objects from PRAW."""
    id: str
    body: str
    score: int
    created_utc: float
    author: Any

    def __getattribute__(self, name: str) -> Any:
        ...

class AnalyzedSubmission(TypedDict):
    """Reddit submission with sentiment analysis."""
    title: str
    selftext: str
    score: int
    num_comments: int
    created_utc: float
    url: str
    sentiment_label: str
    confidence_score: float
    sentiment_score: float
    model: str

class AnalyzedComment(TypedDict):
    """Reddit comment with sentiment analysis."""
    body: str
    score: int
    created_utc: float
    sentiment_label: str
    confidence_score: float
    sentiment_score: float
    model: str

class RedditSentimentResult(TypedDict):
    """Result from Reddit sentiment analysis."""
    sentiment_score: float
    confidence: float
    submission_count: int
    comment_count: int
    analyzed_submissions: List[AnalyzedSubmission]
    analyzed_comments: List[AnalyzedComment]
    statistics: dict
    error: Optional[str]

class RedditCacheEntry(TypedDict):
    """Cache entry for Reddit data."""
    timestamp: float
    data: Any

class RedditAPIConfig(TypedDict, total=False):
    """Configuration for Reddit API."""
    client_id: str
    client_secret: str
    user_agent: str
    username: str
    password: str
    request_timeout: int
    max_retries: int
    cache_expiry: int
    submissions_limit: int
    comments_limit: int
    subreddits: List[str]