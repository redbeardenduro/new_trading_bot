"""Type stubs for praw (Python Reddit API Wrapper) library."""

from typing import Any, Iterator, List, Optional

class Reddit:
    """Reddit API client."""

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        user_agent: Optional[str] = None,
        **kwargs: Any
    ) -> None: ...
    def subreddit(self, display_name: str) -> Subreddit: ...

class Subreddit:
    """Reddit subreddit."""

    def __init__(self, reddit: Reddit, display_name: str) -> None: ...
    def hot(self, limit: Optional[int] = None) -> Iterator[Submission]: ...
    def new(self, limit: Optional[int] = None) -> Iterator[Submission]: ...
    def top(
        self, time_filter: str = "all", limit: Optional[int] = None
    ) -> Iterator[Submission]: ...
    def search(
        self,
        query: str,
        sort: str = "relevance",
        time_filter: str = "all",
        limit: Optional[int] = None,
    ) -> Iterator[Submission]: ...

class Submission:
    """Reddit submission (post)."""

    id: str
    title: str
    selftext: str
    score: int
    upvote_ratio: float
    num_comments: int
    created_utc: float
    author: Optional[Redditor]
    subreddit: Subreddit
    url: str

    def comments(self) -> CommentForest: ...

class Comment:
    """Reddit comment."""

    id: str
    body: str
    score: int
    created_utc: float
    author: Optional[Redditor]

class CommentForest:
    """Collection of comments."""

    def list(self) -> List[Comment]: ...

class Redditor:
    """Reddit user."""

    name: str
