"""
Defines abstract base classes (interfaces) for core components
to enable dependency injection and improve modularity.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class IExchangeClient(ABC):
    """Interface for exchange clients (e.g., Kraken, Binance, Simulated)."""

    @abstractmethod
    def test_connection(self) -> Tuple[bool, Dict[str, Any]]:
        """Test API connection."""
        pass

    @abstractmethod
    def get_balance(self, force_refresh: bool = False) -> Dict[str, float]:
        """Fetch account balances."""
        pass

    @abstractmethod
    def get_ticker(self, symbol: str, force_refresh: bool = False) -> Optional[Dict]:
        """Fetch ticker information for a symbol."""
        pass

    @abstractmethod
    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 100,
        since: Optional[int] = None,
        force_refresh: bool = False,
    ) -> Optional[List[Dict]]:
        """Fetch OHLCV data for a symbol."""
        pass

    @abstractmethod
    def get_order_book(
        self, symbol: str, limit: int = 10, force_refresh: bool = False
    ) -> Optional[Dict]:
        """Fetch order book data for a symbol."""
        pass

    @abstractmethod
    def create_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        params: Dict[str, Any] = {},
    ) -> Dict[str, Any]:
        """Create an order."""
        pass

    def create_market_buy_order(
        self, symbol: str, amount: float, params: Dict[str, Any] = {}
    ) -> Dict[str, Any]:
        return self.create_order(symbol, "market", "buy", amount, None, params)

    def create_market_sell_order(
        self, symbol: str, amount: float, params: Dict[str, Any] = {}
    ) -> Dict[str, Any]:
        return self.create_order(symbol, "market", "sell", amount, None, params)

    def create_limit_buy_order(
        self, symbol: str, amount: float, price: float, params: Dict[str, Any] = {}
    ) -> Dict[str, Any]:
        return self.create_order(symbol, "limit", "buy", amount, price, params)

    def create_limit_sell_order(
        self, symbol: str, amount: float, price: float, params: Dict[str, Any] = {}
    ) -> Dict[str, Any]:
        return self.create_order(symbol, "limit", "sell", amount, price, params)


class ISentimentSource(ABC):
    """Interface for sources providing sentiment analysis (e.g., Reddit, News)."""

    @abstractmethod
    def test_connection(self) -> Tuple[bool, str]:
        """Test connection to the sentiment data source."""
        pass

    @abstractmethod
    def get_sentiment_analysis(self, crypto_symbol: str, **kwargs: Any) -> Dict[str, Any]:
        """
        Fetch and analyze sentiment for a given crypto symbol.

        Should return a dictionary with standardized keys like:
        'sentiment': str ('bullish'/'bearish'/'neutral')
        'strength': float (e.g., score from -1 to 1)
        'confidence': float (0 to 1)
        'posts_analyzed': int
        'source_name': str (e.g., 'Reddit', 'NewsAPI')
        ... plus potentially source-specific metrics.
        """
        pass


class IAIAnalyzer(ABC):
    """Interface for AI-based analysis services (e.g., OpenAI)."""

    @abstractmethod
    async def test_connection(self, force_refresh: bool = False) -> Tuple[bool, Dict]:
        """Test connection to the AI service."""
        pass

    @abstractmethod
    async def generate_analysis(
        self, market_data: Dict, symbol: str, force_refresh: bool = False
    ) -> Dict:
        """
        Generate comprehensive AI analysis (sentiment, strategy, technical interpretation).

        Should return a dictionary with standardized keys like:
        'market_sentiment': str
        'trading_strategy': str ('buy'/'sell'/'hold')
        'strength': float
        'confidence': float
        'timestamp': str (ISO format)
        'error': Optional[str]
        ... plus raw analysis components.
        """
        pass


class IPortfolioManager(ABC):
    """Interface for the portfolio management component."""

    @abstractmethod
    def get_current_holdings(self) -> Dict[str, float]:
        """Get current asset holdings."""
        pass

    @abstractmethod
    def determine_rebalance_actions(self) -> list:
        """
        Determine and simulate necessary rebalancing trades.

        Returns a list of simulated trade dictionaries or an empty list.
        """
        pass
