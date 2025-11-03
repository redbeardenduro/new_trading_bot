"""
Enhanced Order Manager with Idempotency Keys.

Manages the state of all orders with idempotency support to prevent
duplicate order submissions, providing a state machine for each order
and reconciliation mechanism to ensure local state matches the exchange.
"""

import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from common.common_logger import DATA_DIR, get_logger

logger = get_logger("order_manager_enhanced")


class OrderStatus(Enum):
    """Order status enumeration."""

    NEW = "new"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    DUPLICATE = "duplicate"


class Order:
    """
    Represents a trading order with full lifecycle tracking.

    Attributes:
        client_order_id: Unique client-side order identifier
        idempotency_key: Key to prevent duplicate submissions
        exchange_order_id: Exchange-assigned order ID
        pair: Trading pair (e.g., 'BTC/USD')
        side: Order side ('buy' or 'sell')
        order_type: Order type ('market' or 'limit')
        quantity: Order quantity
        price: Order price (for limit orders)
        status: Current order status
        filled_quantity: Amount filled so far
        average_fill_price: Average execution price
        created_at: Order creation timestamp
        updated_at: Last update timestamp
        history: List of status changes
        metadata: Additional order metadata
    """

    def __init__(
        self,
        pair: str,
        side: str,
        order_type: str,
        quantity: Decimal,
        price: Optional[Decimal] = None,
        idempotency_key: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize a new order.

        Args:
            pair: Trading pair
            side: Order side ('buy' or 'sell')
            order_type: Order type ('market' or 'limit')
            quantity: Order quantity
            price: Order price (required for limit orders)
            idempotency_key: Optional idempotency key (generated if not provided)
            metadata: Optional metadata dictionary
        """
        self.client_order_id = str(uuid.uuid4())
        if idempotency_key is None:
            key_data = f"{pair}:{side}:{order_type}:{quantity}:{price}:{time.time_ns()}"
            self.idempotency_key = hashlib.sha256(key_data.encode()).hexdigest()
        else:
            self.idempotency_key = idempotency_key
        self.exchange_order_id: Optional[str] = None
        self.pair = pair
        self.side = side
        self.order_type = order_type
        self.quantity = quantity
        self.price = price
        self.status = OrderStatus.NEW
        self.filled_quantity = Decimal("0")
        self.average_fill_price = Decimal("0")
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = self.created_at
        self.history: List[Tuple[datetime, OrderStatus, str]] = [
            (self.created_at, self.status, "Order created")
        ]
        self.metadata = metadata or {}
        self.submission_attempts = 0
        self.last_submission_attempt: Optional[datetime] = None

    def update_status(self, new_status: OrderStatus, reason: str) -> None:
        """
        Update order status and record in history.

        Args:
            new_status: New order status
            reason: Reason for status change
        """
        if self.status != new_status:
            self.status = new_status
            self.updated_at = datetime.now(timezone.utc)
            self.history.append((self.updated_at, new_status, reason))
            logger.info(
                "Order %s (idem: %s...) status updated to %s: %s",
                self.client_order_id,
                self.idempotency_key[:8],
                new_status.value,
                reason,
            )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert order to dictionary representation.

        Returns:
            Dictionary representation of the order
        """
        return {
            "client_order_id": self.client_order_id,
            "idempotency_key": self.idempotency_key,
            "exchange_order_id": self.exchange_order_id,
            "pair": self.pair,
            "side": self.side,
            "order_type": self.order_type,
            "quantity": str(self.quantity),
            "price": str(self.price) if self.price else None,
            "status": self.status.value,
            "filled_quantity": str(self.filled_quantity),
            "average_fill_price": str(self.average_fill_price),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "submission_attempts": self.submission_attempts,
            "last_submission_attempt": (
                self.last_submission_attempt.isoformat() if self.last_submission_attempt else None
            ),
            "history": [
                (ts.isoformat(), status.value, reason) for (ts, status, reason) in self.history
            ],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Order":
        """
        Create order from dictionary representation.

        Args:
            data: Dictionary representation of order

        Returns:
            Order instance
        """
        order = cls(
            pair=data["pair"],
            side=data["side"],
            order_type=data["order_type"],
            quantity=Decimal(data["quantity"]),
            price=Decimal(data["price"]) if data.get("price") else None,
            idempotency_key=data["idempotency_key"],
            metadata=data.get("metadata", {}),
        )
        order.client_order_id = data["client_order_id"]
        order.exchange_order_id = data.get("exchange_order_id")
        order.status = OrderStatus(data["status"])
        order.filled_quantity = Decimal(data["filled_quantity"])
        order.average_fill_price = Decimal(data["average_fill_price"])
        order.created_at = datetime.fromisoformat(data["created_at"])
        order.updated_at = datetime.fromisoformat(data["updated_at"])
        order.submission_attempts = data.get("submission_attempts", 0)
        if data.get("last_submission_attempt"):
            order.last_submission_attempt = datetime.fromisoformat(data["last_submission_attempt"])
        order.history = [
            (datetime.fromisoformat(ts), OrderStatus(status), reason)
            for (ts, status, reason) in data.get("history", [])
        ]
        return order


class OrderManagerEnhanced:
    """
    Enhanced order manager with idempotency support.

    Features:
    - Idempotency key tracking to prevent duplicate submissions
    - Persistent order state storage
    - Order reconciliation with exchange
    - Comprehensive order lifecycle management
    """

    def __init__(self, exchange_client: Any, storage_dir: Optional[Path] = None) -> None:
        """
        Initialize the enhanced order manager.

        Args:
            exchange_client: Exchange client instance
            storage_dir: Directory for persistent storage
        """
        self.exchange_client = exchange_client
        if storage_dir is None:
            storage_dir = DATA_DIR / "orders"
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.orders: Dict[str, Order] = {}
        self.idempotency_map: Dict[str, str] = {}
        self._load_orders()
        logger.info("Initialized OrderManagerEnhanced with %s existing orders", len(self.orders))

    def _get_storage_path(self) -> Path:
        """Get the path to the orders storage file."""
        return self.storage_dir / "orders.json"

    def _load_orders(self) -> None:
        """Load orders from persistent storage."""
        storage_path = self._get_storage_path()
        if not storage_path.exists():
            logger.info("No existing orders file found")
            return
        try:
            with storage_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            for order_data in data.get("orders", []):
                order = Order.from_dict(order_data)
                self.orders[order.client_order_id] = order
                self.idempotency_map[order.idempotency_key] = order.client_order_id
            logger.info("Loaded %s orders from storage", len(self.orders))
        except json.JSONDecodeError as e:
            logger.error("Error decoding orders file: %s", e)
        except Exception as e:
            logger.error("Error loading orders: %s", e, exc_info=True)

    def _save_orders(self) -> None:
        """Save orders to persistent storage."""
        storage_path = self._get_storage_path()
        temp_path = storage_path.with_suffix(".tmp")
        try:
            data = {
                "orders": [order.to_dict() for order in self.orders.values()],
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            with temp_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            temp_path.replace(storage_path)
            logger.debug("Saved %s orders to storage", len(self.orders))
        except Exception as e:
            logger.error("Error saving orders: %s", e, exc_info=True)
            if temp_path.exists():
                temp_path.unlink()

    def create_order(
        self,
        pair: str,
        side: str,
        order_type: str,
        quantity: Decimal,
        price: Optional[Decimal] = None,
        idempotency_key: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Order:
        """
        Create a new order with idempotency support.

        Args:
            pair: Trading pair
            side: Order side ('buy' or 'sell')
            order_type: Order type ('market' or 'limit')
            quantity: Order quantity
            price: Order price (for limit orders)
            idempotency_key: Optional idempotency key
            metadata: Optional metadata

        Returns:
            Order instance

        Raises:
            ValueError: If duplicate idempotency key is detected
        """
        if idempotency_key and idempotency_key in self.idempotency_map:
            existing_order_id = self.idempotency_map[idempotency_key]
            existing_order = self.orders[existing_order_id]
            logger.warning(
                "Duplicate idempotency key detected: %s... (existing order: %s)",
                idempotency_key[:8],
                existing_order_id,
            )
            existing_order.update_status(
                OrderStatus.DUPLICATE, "Duplicate submission attempt detected"
            )
            return existing_order
        order = Order(
            pair=pair,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            idempotency_key=idempotency_key,
            metadata=metadata,
        )
        self.orders[order.client_order_id] = order
        self.idempotency_map[order.idempotency_key] = order.client_order_id
        self._save_orders()
        logger.info(
            "Created new order %s (idem: %s...) for %s: %s %s @ %s",
            order.client_order_id,
            order.idempotency_key[:8],
            pair,
            side,
            quantity,
            order_type,
        )
        return order

    def submit_order(self, order: Order) -> Optional[Dict[str, Any]]:
        """
        Submit order to exchange with idempotency protection.

        Args:
            order: Order to submit

        Returns:
            Exchange order response or None if submission failed
        """
        if order.status in [
            OrderStatus.SUBMITTED,
            OrderStatus.FILLED,
            OrderStatus.PARTIALLY_FILLED,
        ]:
            logger.warning(
                "Order %s already in status %s, skipping submission",
                order.client_order_id,
                order.status.value,
            )
            return None
        order.submission_attempts += 1
        order.last_submission_attempt = datetime.now(timezone.utc)
        try:
            params = {
                "userref": order.client_order_id,
                "idempotency_key": order.idempotency_key,
                **order.metadata.get("params", {}),
            }
            exchange_order = self.exchange_client.create_order(
                pair=order.pair,
                order_type=order.order_type,
                side=order.side,
                amount=float(order.quantity),
                price=float(order.price) if order.price else None,
                params=params,
            )
            order.exchange_order_id = exchange_order.get("id")
            order.update_status(
                OrderStatus.SUBMITTED, f"Submitted to exchange with ID {order.exchange_order_id}"
            )
            self._save_orders()
            return exchange_order
        except Exception as e:
            error_msg = str(e)
            if any(
                (
                    keyword in error_msg.lower()
                    for keyword in ["duplicate", "already exists", "idempotency"]
                )
            ):
                order.update_status(
                    OrderStatus.DUPLICATE, f"Duplicate detected by exchange: {error_msg}"
                )
                logger.warning(
                    "Exchange detected duplicate order %s: %s", order.client_order_id, error_msg
                )
            else:
                order.update_status(OrderStatus.REJECTED, f"Submission failed: {error_msg}")
                logger.error(
                    "Failed to submit order %s: %s", order.client_order_id, error_msg, exc_info=True
                )
            self._save_orders()
            return None

    def get_order(self, client_order_id: str) -> Optional[Order]:
        """
        Get order by client order ID.

        Args:
            client_order_id: Client order ID

        Returns:
            Order instance or None if not found
        """
        return self.orders.get(client_order_id)

    def get_order_by_idempotency_key(self, idempotency_key: str) -> Optional[Order]:
        """
        Get order by idempotency key.

        Args:
            idempotency_key: Idempotency key

        Returns:
            Order instance or None if not found
        """
        client_order_id = self.idempotency_map.get(idempotency_key)
        if client_order_id:
            return self.orders.get(client_order_id)
        return None

    def reconcile_orders(self) -> None:
        """
        Reconcile order states with exchange.

        Fetches current order status from exchange and updates local state.
        """
        logger.info("Starting order reconciliation...")
        open_orders = [
            o
            for o in self.orders.values()
            if o.status in [OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]
        ]
        for order in open_orders:
            if not order.exchange_order_id:
                logger.warning(
                    "Order %s has no exchange ID, skipping reconciliation", order.client_order_id
                )
                continue
            try:
                exchange_order = self.exchange_client.fetch_order(
                    order.exchange_order_id, order.pair
                )
                exchange_status = exchange_order.get("status", "").lower()
                if exchange_status == "closed":
                    order.update_status(OrderStatus.FILLED, "Reconciled as filled")
                    order.filled_quantity = Decimal(str(exchange_order.get("filled", 0)))
                    order.average_fill_price = Decimal(str(exchange_order.get("average", 0)))
                elif exchange_status == "canceled":
                    order.update_status(OrderStatus.CANCELED, "Reconciled as canceled")
                elif exchange_status == "open":
                    filled = exchange_order.get("filled", 0)
                    if filled > 0:
                        order.update_status(
                            OrderStatus.PARTIALLY_FILLED, "Reconciled as partially filled"
                        )
                        order.filled_quantity = Decimal(str(filled))
            except Exception as e:
                logger.error(
                    "Failed to reconcile order %s: %s", order.client_order_id, e, exc_info=True
                )
        self._save_orders()
        logger.info("Order reconciliation complete. Processed %s orders", len(open_orders))

    def get_orders_by_status(self, status: OrderStatus) -> List[Order]:
        """
        Get all orders with a specific status.

        Args:
            status: Order status to filter by

        Returns:
            List of orders with the specified status
        """
        return [o for o in self.orders.values() if o.status == status]

    def get_orders_by_pair(self, pair: str) -> List[Order]:
        """
        Get all orders for a specific trading pair.

        Args:
            pair: Trading pair

        Returns:
            List of orders for the specified pair
        """
        return [o for o in self.orders.values() if o.pair == pair]

    def cleanup_old_orders(self, days: int = 30) -> int:
        """
        Clean up old completed orders.

        Args:
            days: Remove orders older than this many days

        Returns:
            Number of orders removed
        """
        cutoff_time = datetime.now(timezone.utc).timestamp() - days * 86400
        removed_count = 0
        orders_to_remove = []
        for order_id, order in self.orders.items():
            if order.status in [OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED]:
                if order.updated_at.timestamp() < cutoff_time:
                    orders_to_remove.append(order_id)
        for order_id in orders_to_remove:
            order = self.orders[order_id]
            del self.idempotency_map[order.idempotency_key]
            del self.orders[order_id]
            removed_count += 1
        if removed_count > 0:
            self._save_orders()
            logger.info("Cleaned up %s old orders", removed_count)
        return removed_count
