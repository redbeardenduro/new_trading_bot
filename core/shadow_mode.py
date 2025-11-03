"""
Shadow Mode Trading Module

Implements shadow trading mode where real decisions are mirrored as paper trades
alongside live trades to measure divergence and validate strategies.
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from common.common_logger import get_logger

logger = get_logger("shadow_mode")


@dataclass
class ShadowTrade:
    """Represents a shadow (paper) trade."""

    trade_id: str
    timestamp: float
    symbol: str
    side: str
    amount: float
    price: float
    order_type: str
    status: str = "open"
    fill_price: Optional[float] = None
    fill_time: Optional[float] = None
    pnl: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ShadowPosition:
    """Represents a shadow position."""

    symbol: str
    amount: float
    entry_price: float
    entry_time: float
    current_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


class ShadowModeTracker:
    """
    Tracks shadow trades and compares with live trades.

    Features:
    - Mirror live trading decisions as paper trades
    - Track shadow portfolio separately
    - Detect divergence between live and shadow performance
    - Alert on significant divergence
    """

    def __init__(
        self,
        initial_balance: float = 10000.0,
        divergence_threshold: float = 0.1,
        storage_dir: Optional[str] = None,
    ) -> None:
        """
        Initialize shadow mode tracker.

        Args:
            initial_balance: Initial balance for shadow trading
            divergence_threshold: Alert threshold for performance divergence (e.g., 0.10 = 10%)
            storage_dir: Directory to store shadow trade data
        """
        self.initial_balance = initial_balance
        self.shadow_balance = initial_balance
        self.divergence_threshold = divergence_threshold
        self.shadow_trades: List[ShadowTrade] = []
        self.shadow_positions: Dict[str, ShadowPosition] = {}
        self.live_pnl: float = 0.0
        self.shadow_pnl: float = 0.0
        if storage_dir is None:
            storage_dir = "data/shadow_mode"
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Initialized ShadowModeTracker with balance: %s", initial_balance)

    def mirror_trade(self, live_trade: Dict[str, Any]) -> ShadowTrade:
        """
        Mirror a live trade as a shadow trade.

        Args:
            live_trade: Live trade information

        Returns:
            Created shadow trade
        """
        shadow_trade = ShadowTrade(
            trade_id=f"shadow_{live_trade.get('order_id', time.time())}",
            timestamp=time.time(),
            symbol=live_trade["symbol"],
            side=live_trade["side"],
            amount=live_trade["amount"],
            price=live_trade.get("price", 0.0),
            order_type=live_trade.get("type", "market"),
            metadata={
                "live_order_id": live_trade.get("order_id"),
                "mirrored_at": datetime.now().isoformat(),
            },
        )
        self.shadow_trades.append(shadow_trade)
        logger.info(
            "Mirrored live trade as shadow: %s %s %s @ %s",
            shadow_trade.symbol,
            shadow_trade.side,
            shadow_trade.amount,
            shadow_trade.price,
        )
        return shadow_trade

    def fill_shadow_trade(self, trade_id: str, fill_price: float) -> None:
        """
        Mark a shadow trade as filled.

        Args:
            trade_id: Shadow trade ID
            fill_price: Fill price
        """
        for trade in self.shadow_trades:
            if trade.trade_id == trade_id and trade.status == "open":
                trade.status = "filled"
                trade.fill_price = fill_price
                trade.fill_time = time.time()
                self._update_shadow_position(trade)
                logger.info("Filled shadow trade %s at %s", trade_id, fill_price)
                break

    def _update_shadow_position(self, trade: ShadowTrade) -> None:
        """Update shadow position based on filled trade."""
        symbol = trade.symbol
        if symbol not in self.shadow_positions:
            if trade.side == "buy":
                self.shadow_positions[symbol] = ShadowPosition(
                    symbol=symbol,
                    amount=trade.amount,
                    entry_price=trade.fill_price or trade.price,
                    entry_time=trade.fill_time or trade.timestamp,
                    current_price=trade.fill_price or trade.price,
                )
        else:
            position = self.shadow_positions[symbol]
            if trade.side == "buy":
                total_cost = position.amount * position.entry_price + trade.amount * (
                    trade.fill_price or trade.price
                )
                total_amount = position.amount + trade.amount
                position.entry_price = total_cost / total_amount
                position.amount = total_amount
            elif trade.amount >= position.amount:
                realized_pnl = (
                    trade.fill_price or trade.price - position.entry_price
                ) * position.amount
                position.realized_pnl += realized_pnl
                self.shadow_pnl += realized_pnl
                del self.shadow_positions[symbol]
                logger.info("Closed shadow position %s, PnL: %.2f", symbol, realized_pnl)
            else:
                realized_pnl = (
                    trade.fill_price or trade.price - position.entry_price
                ) * trade.amount
                position.realized_pnl += realized_pnl
                position.amount -= trade.amount
                self.shadow_pnl += realized_pnl

    def update_prices(self, prices: Dict[str, float]) -> None:
        """
        Update current prices for shadow positions.

        Args:
            prices: Dictionary of symbol -> current price
        """
        for symbol, position in self.shadow_positions.items():
            if symbol in prices:
                position.current_price = prices[symbol]
                position.unrealized_pnl = (
                    position.current_price - position.entry_price
                ) * position.amount

    def update_live_pnl(self, live_pnl: float) -> None:
        """
        Update live trading PnL for comparison.

        Args:
            live_pnl: Current live trading PnL
        """
        self.live_pnl = live_pnl
        self._check_divergence()

    def _check_divergence(self) -> None:
        """Check for significant divergence between live and shadow performance."""
        if self.live_pnl == 0:
            return
        divergence = abs(self.shadow_pnl - self.live_pnl) / abs(self.live_pnl)
        if divergence > self.divergence_threshold:
            logger.warning(
                "DIVERGENCE ALERT: Shadow PnL (%.2f) diverges from live PnL (%.2f) by %.1f%%",
                self.shadow_pnl,
                self.live_pnl,
                divergence * 100,
            )
            self._save_divergence_report(divergence)

    def _save_divergence_report(self, divergence: float) -> None:
        """Save a divergence report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "live_pnl": self.live_pnl,
            "shadow_pnl": self.shadow_pnl,
            "divergence_pct": divergence * 100,
            "shadow_positions": {
                symbol: {
                    "amount": pos.amount,
                    "entry_price": pos.entry_price,
                    "current_price": pos.current_price,
                    "unrealized_pnl": pos.unrealized_pnl,
                }
                for (symbol, pos) in self.shadow_positions.items()
            },
            "recent_trades": [
                {
                    "trade_id": trade.trade_id,
                    "symbol": trade.symbol,
                    "side": trade.side,
                    "amount": trade.amount,
                    "price": trade.price,
                    "status": trade.status,
                }
                for trade in self.shadow_trades[-10:]
            ],
        }
        report_file = self.storage_dir / f"divergence_{int(time.time())}.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("Saved divergence report to %s", report_file)

    def get_shadow_performance(self) -> Dict[str, Any]:
        """
        Get shadow trading performance summary.

        Returns:
            Performance summary dictionary
        """
        total_unrealized_pnl = sum((pos.unrealized_pnl for pos in self.shadow_positions.values()))
        total_pnl = self.shadow_pnl + total_unrealized_pnl
        return {
            "initial_balance": self.initial_balance,
            "current_balance": self.shadow_balance + total_pnl,
            "realized_pnl": self.shadow_pnl,
            "unrealized_pnl": total_unrealized_pnl,
            "total_pnl": total_pnl,
            "return_pct": total_pnl / self.initial_balance * 100,
            "open_positions": len(self.shadow_positions),
            "total_trades": len(self.shadow_trades),
            "live_pnl": self.live_pnl,
            "divergence_pct": (
                abs(total_pnl - self.live_pnl) / abs(self.live_pnl) * 100
                if self.live_pnl != 0
                else 0
            ),
        }

    def save_state(self) -> None:
        """Save shadow mode state to disk."""
        state = {
            "timestamp": datetime.now().isoformat(),
            "initial_balance": self.initial_balance,
            "shadow_balance": self.shadow_balance,
            "shadow_pnl": self.shadow_pnl,
            "live_pnl": self.live_pnl,
            "positions": {
                symbol: {
                    "amount": pos.amount,
                    "entry_price": pos.entry_price,
                    "entry_time": pos.entry_time,
                    "current_price": pos.current_price,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "realized_pnl": pos.realized_pnl,
                }
                for (symbol, pos) in self.shadow_positions.items()
            },
            "trades": [
                {
                    "trade_id": trade.trade_id,
                    "timestamp": trade.timestamp,
                    "symbol": trade.symbol,
                    "side": trade.side,
                    "amount": trade.amount,
                    "price": trade.price,
                    "status": trade.status,
                    "fill_price": trade.fill_price,
                    "fill_time": trade.fill_time,
                }
                for trade in self.shadow_trades
            ],
            "performance": self.get_shadow_performance(),
        }
        state_file = self.storage_dir / "shadow_state.json"
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)
        logger.info("Saved shadow mode state to %s", state_file)

    def load_state(self) -> bool:
        """
        Load shadow mode state from disk.

        Returns:
            True if state was loaded successfully
        """
        state_file = self.storage_dir / "shadow_state.json"
        if not state_file.exists():
            return False
        try:
            with open(state_file, "r") as f:
                state = json.load(f)
            self.shadow_balance = state["shadow_balance"]
            self.shadow_pnl = state["shadow_pnl"]
            self.live_pnl = state["live_pnl"]
            for symbol, pos_data in state["positions"].items():
                self.shadow_positions[symbol] = ShadowPosition(
                    symbol=symbol,
                    amount=pos_data["amount"],
                    entry_price=pos_data["entry_price"],
                    entry_time=pos_data["entry_time"],
                    current_price=pos_data["current_price"],
                    unrealized_pnl=pos_data["unrealized_pnl"],
                    realized_pnl=pos_data["realized_pnl"],
                )
            for trade_data in state["trades"]:
                self.shadow_trades.append(
                    ShadowTrade(
                        trade_id=trade_data["trade_id"],
                        timestamp=trade_data["timestamp"],
                        symbol=trade_data["symbol"],
                        side=trade_data["side"],
                        amount=trade_data["amount"],
                        price=trade_data["price"],
                        order_type=trade_data.get("order_type", "market"),
                        status=trade_data["status"],
                        fill_price=trade_data.get("fill_price"),
                        fill_time=trade_data.get("fill_time"),
                    )
                )
            logger.info("Loaded shadow mode state successfully")
            return True
        except Exception as e:
            logger.error("Failed to load shadow mode state: %s", e)
            return False
