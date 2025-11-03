"""
Idempotency Module for Order Management

Provides idempotency key generation and duplicate detection
to prevent duplicate order submissions.
"""

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Optional

from common.common_logger import get_logger

logger = get_logger("idempotency")


@dataclass
class OrderIntent:
    """Represents an order intent with idempotency key."""

    key: str
    symbol: str
    side: str
    amount: float
    price: Optional[float]
    order_type: str
    timestamp: float
    order_id: Optional[str] = None
    status: str = "pending"


class IdempotencyManager:
    """
    Manages idempotency keys for order intents.

    Features:
    - Generate deterministic idempotency keys
    - Cache active intents to detect duplicates
    - Automatic cleanup of old intents
    """

    def __init__(self, cache_size: int = 1000, cache_ttl: int = 3600) -> None:
        """
        Initialize idempotency manager.

        Args:
            cache_size: Maximum number of intents to cache
            cache_ttl: Time-to-live for cached intents in seconds
        """
        self.cache_size = cache_size
        self.cache_ttl = cache_ttl
        self.intent_cache: OrderedDict[str, OrderIntent] = OrderedDict()
        logger.info(
            "Initialized IdempotencyManager (cache_size: %s, ttl: %ss)", cache_size, cache_ttl
        )

    def generate_key(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: Optional[float],
        order_type: str,
        extra_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate an idempotency key for an order intent.

        The key is deterministic based on order parameters.

        Args:
            symbol: Trading symbol
            side: Order side ('buy' or 'sell')
            amount: Order amount
            price: Order price (None for market orders)
            order_type: Order type ('market', 'limit', etc.)
            extra_data: Additional data to include in key generation

        Returns:
            Idempotency key (SHA256 hash)
        """
        key_parts = [
            symbol,
            side,
            f"{amount:.8f}",
            f"{price:.8f}" if price is not None else "market",
            order_type,
        ]
        if extra_data:
            for k in sorted(extra_data.keys()):
                key_parts.append(f"{k}:{extra_data[k]}")
        key_string = "|".join(key_parts)
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()
        return key_hash

    def create_intent(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: Optional[float],
        order_type: str,
        extra_data: Optional[Dict[str, Any]] = None,
    ) -> OrderIntent:
        """
        Create a new order intent with idempotency key.

        Args:
            symbol: Trading symbol
            side: Order side ('buy' or 'sell')
            amount: Order amount
            price: Order price (None for market orders)
            order_type: Order type ('market', 'limit', etc.)
            extra_data: Additional data to include in key generation

        Returns:
            Created order intent
        """
        key = self.generate_key(symbol, side, amount, price, order_type, extra_data)
        intent = OrderIntent(
            key=key,
            symbol=symbol,
            side=side,
            amount=amount,
            price=price,
            order_type=order_type,
            timestamp=time.time(),
        )
        return intent

    def check_duplicate(self, intent: OrderIntent) -> bool:
        """
        Check if an order intent is a duplicate.

        Args:
            intent: Order intent to check

        Returns:
            True if duplicate, False otherwise
        """
        self._clean_expired()
        if intent.key in self.intent_cache:
            cached_intent = self.intent_cache[intent.key]
            if cached_intent.status in ["pending", "submitted"]:
                logger.warning(
                    "Duplicate order intent detected: %s %s %s (key: %s...)",
                    intent.symbol,
                    intent.side,
                    intent.amount,
                    intent.key[:16],
                )
                return True
        return False

    def register_intent(self, intent: OrderIntent) -> bool:
        """
        Register an order intent in the cache.

        Args:
            intent: Order intent to register

        Returns:
            True if registered successfully, False if duplicate
        """
        if self.check_duplicate(intent):
            return False
        self.intent_cache[intent.key] = intent
        if len(self.intent_cache) > self.cache_size:
            self.intent_cache.popitem(last=False)
        logger.debug(
            "Registered order intent: %s %s %s (key: %s...)",
            intent.symbol,
            intent.side,
            intent.amount,
            intent.key[:16],
        )
        return True

    def update_intent_status(self, key: str, status: str, order_id: Optional[str] = None) -> None:
        """
        Update the status of an order intent.

        Args:
            key: Idempotency key
            status: New status
            order_id: Order ID (if submitted)
        """
        if key in self.intent_cache:
            intent = self.intent_cache[key]
            intent.status = status
            if order_id:
                intent.order_id = order_id
            logger.debug("Updated intent %s... status to %s", key[:16], status)
        else:
            logger.warning("Attempted to update unknown intent: %s...", key[:16])

    def get_intent(self, key: str) -> Optional[OrderIntent]:
        """
        Get an order intent by key.

        Args:
            key: Idempotency key

        Returns:
            Order intent if found, None otherwise
        """
        return self.intent_cache.get(key)

    def _clean_expired(self) -> None:
        """Remove expired intents from cache."""
        now = time.time()
        expired_keys = []
        for key, intent in self.intent_cache.items():
            if now - intent.timestamp > self.cache_ttl or intent.status in [
                "filled",
                "rejected",
                "cancelled",
            ]:
                expired_keys.append(key)
        for key in expired_keys:
            del self.intent_cache[key]
        if expired_keys:
            logger.debug("Cleaned %s expired intents from cache", len(expired_keys))

    def get_stats(self) -> Dict[str, Any]:
        """
        Get idempotency manager statistics.

        Returns:
            Statistics dictionary
        """
        self._clean_expired()
        status_counts = {}
        for intent in self.intent_cache.values():
            status_counts[intent.status] = status_counts.get(intent.status, 0) + 1
        return {
            "total_cached": len(self.intent_cache),
            "cache_size_limit": self.cache_size,
            "cache_ttl": self.cache_ttl,
            "status_breakdown": status_counts,
        }


_idempotency_manager: Optional[IdempotencyManager] = None


def get_idempotency_manager() -> IdempotencyManager:
    """Get or create the global idempotency manager instance."""
    global _idempotency_manager
    if _idempotency_manager is None:
        _idempotency_manager = IdempotencyManager()
    return _idempotency_manager
