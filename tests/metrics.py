# enhanced_trading_bot/tests/metrics.py
# MODIFIED - Re-applying defensive formatting before logging metrics
"""
Backtest Metrics Calculation Module.

Contains the BacktestMetrics class responsible for calculating and storing
detailed performance metrics for a backtest run, including per-trade
analysis using FIFO logic for position tracking.
"""

import json
import logging
import math
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from decimal import Decimal  # Added InvalidOperation
from decimal import DivisionByZero, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Use basic logging if central logger isn't set up or needed here
# If using central logger: from common.common_logger import get_logger, DATA_DIR
# logger = get_logger("backtest_metrics")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("backtest_metrics")


class BacktestMetrics:
    """
    Calculates and stores performance metrics for a backtest run.
    Uses FIFO logic for position tracking and Decimal for precision.
    """

    def __init__(self, initial_capital: Decimal, quote_currency: str):
        if not isinstance(initial_capital, Decimal):
            try:
                self.initial_capital = Decimal(str(initial_capital))
            except (InvalidOperation, ValueError, TypeError):
                logger.error(f"Invalid initial_capital '{initial_capital}'. Using 0.")
                self.initial_capital = Decimal("0.0")
        else:
            self.initial_capital = initial_capital

        if not isinstance(quote_currency, str) or not quote_currency.strip():
            logger.error(f"Invalid quote_currency '{quote_currency}'. Using 'USD'.")
            self.quote_currency = "USD"
        else:
            self.quote_currency = quote_currency.strip().upper()

        self.final_capital = self.initial_capital  # Start final = initial
        self.executed_trades: List[Dict] = []  # All closed trades from simulator
        self.portfolio_value_history: List[Tuple[datetime, Decimal]] = []  # Store values as Decimal
        self.metrics: Dict[str, Any] = {}
        # Tracks open positions using FIFO logic. Structure: {symbol: deque([{'amount': Decimal, 'price': Decimal, 'timestamp': datetime, 'fee': Decimal}])}
        self.open_positions: Dict[str, deque] = {}
        self.round_trips: List[Dict] = []  # Stores details of completed round trips

        logger.info(
            f"BacktestMetrics initialized. Initial Capital: {self.initial_capital} {self.quote_currency}"
        )

    def record_trade(self, trade: Dict):
        """Record an executed trade and update open positions/round trips. Validates trade data."""
        # --- Input Validation ---
        if not isinstance(trade, dict):
            logger.warning("Skipping trade record: Input is not a dictionary.")
            return
        if trade.get("status") != "closed":
            logger.debug(f"Skipping trade record (not closed): {trade.get('id', 'N/A')}")
            return

        symbol = trade.get("symbol")
        side = trade.get("side")
        if not isinstance(symbol, str) or "/" not in symbol:
            logger.warning(
                f"Skipping trade record for PnL: Invalid symbol '{symbol}' in trade {trade.get('id', 'N/A')}."
            )
            return
        if side not in ["buy", "sell"]:
            logger.warning(
                f"Skipping trade record for PnL: Invalid side '{side}' in trade {trade.get('id', 'N/A')}."
            )
            return

        try:
            # Convert numeric fields to Decimal, ensuring they are valid
            amount_str = str(trade.get("amount", "0.0"))
            avg_price_str = str(trade.get("average", "0.0"))
            fee_cost_str = str(trade.get("fee", {}).get("cost", "0.0"))  # Safely get fee cost

            amount = Decimal(amount_str)
            avg_price = Decimal(avg_price_str)
            fee = Decimal(fee_cost_str)

            if amount <= Decimal("1e-9"):
                raise ValueError("Amount is zero or negative")
            if avg_price <= Decimal("1e-9"):
                raise ValueError("Average price is zero or negative")
            if fee < Decimal("0"):
                raise ValueError("Fee cannot be negative")  # Allow zero fee

        except (InvalidOperation, ValueError, TypeError, KeyError) as e:
            logger.warning(
                f"Skipping trade record for PnL due to invalid/missing numeric data in trade {trade.get('id', 'N/A')}: {e}"
            )
            return

        # --- Timestamp Parsing ---
        try:
            ts_input = trade.get("datetime", trade.get("timestamp"))  # Prioritize datetime string
            if isinstance(ts_input, (int, float)):  # Handle ms or s timestamp
                unit = "ms" if ts_input > 1e12 else "s"
                timestamp = pd.to_datetime(ts_input, unit=unit, utc=True).to_pydatetime()
            elif isinstance(ts_input, str):
                timestamp = pd.to_datetime(
                    ts_input, errors="raise", utc=True
                ).to_pydatetime()  # Raise error on parse failure
            elif isinstance(ts_input, datetime):
                timestamp = (
                    ts_input.astimezone(timezone.utc)
                    if ts_input.tzinfo
                    else ts_input.replace(tzinfo=timezone.utc)
                )
            else:
                raise ValueError(f"Invalid timestamp type: {type(ts_input)}")
            # Ensure timestamp is timezone-aware UTC
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            elif timestamp.tzinfo != timezone.utc:
                timestamp = timestamp.astimezone(timezone.utc)

        except Exception as e:
            logger.error(
                f"Failed to parse valid timestamp for trade {trade.get('id', 'N/A')}: '{ts_input}'. Error: {e}. Using current time."
            )
            timestamp = datetime.now(timezone.utc)  # Fallback timestamp

        # --- Append Validated Trade ---
        # Store a copy to avoid modifying original trade dict if passed by reference
        validated_trade = trade.copy()
        validated_trade["amount_dec"] = amount
        validated_trade["avg_price_dec"] = avg_price
        validated_trade["fee_dec"] = fee
        validated_trade["timestamp_dt"] = timestamp
        self.executed_trades.append(validated_trade)  # Store validated trade

        # --- Process Trade for PnL ---
        try:
            base, quote = symbol.split("/")
            base = base.upper()
            quote = quote.upper()

            if side == "buy":
                if symbol not in self.open_positions:
                    self.open_positions[symbol] = deque()
                # Add bought lot to the queue (FIFO) using Decimal values
                self.open_positions[symbol].append(
                    {
                        "amount": amount,
                        "price": avg_price,
                        "timestamp": timestamp,
                        "fee": fee,
                        "order_id": trade.get("id"),  # Optionally store entry order ID
                    }
                )
                logger.debug(
                    f"Position Update: Bought {amount:.8f} {base} @ {avg_price:.4f}. Queue size: {len(self.open_positions[symbol])}"
                )

            elif side == "sell":
                self._process_sell_trade(
                    symbol,
                    base,
                    quote,
                    amount,
                    avg_price,
                    fee,
                    timestamp,
                    trade.get("id"),
                )

        except Exception as e:
            # Catch unexpected errors during processing
            logger.error(
                f"Unexpected error processing trade {trade.get('id', 'N/A')}: {e}",
                exc_info=True,
            )

    def _process_sell_trade(
        self,
        symbol: str,
        base: str,
        quote: str,
        sell_amount: Decimal,
        sell_price: Decimal,
        sell_fee: Decimal,
        sell_timestamp: datetime,
        sell_order_id: Optional[str],
    ):
        """Processes a sell trade against open positions using FIFO."""
        if symbol not in self.open_positions or not self.open_positions[symbol]:
            logger.warning(
                f"Received sell trade {sell_order_id or 'N/A'} for {symbol} but no open position found. Ignoring for PnL."
            )
            return

        sell_amount_remaining = sell_amount
        total_proceeds_net = Decimal("0.0")  # Proceeds after exit fee
        total_cost_basis = Decimal("0.0")  # Cost of lots closed
        total_entry_fees = Decimal("0.0")  # Fees from lots closed
        first_entry_timestamp = self.open_positions[symbol][0][
            "timestamp"
        ]  # Timestamp of the *first* lot contributing
        buy_order_ids_for_rt = []  # Track buy IDs contributing to this round trip
        amount_closed_in_rt = Decimal("0.0")  # Track total amount closed for this specific sell

        logger.debug(
            f"Position Update: Selling {sell_amount_remaining:.8f} {base}. Open queue size: {len(self.open_positions[symbol])}"
        )

        # Iterate through open buy lots (FIFO)
        while sell_amount_remaining > Decimal("1e-9") and self.open_positions[symbol]:
            entry_lot = self.open_positions[symbol][0]
            entry_amount = entry_lot["amount"]
            entry_price = entry_lot["price"]
            entry_fee = entry_lot["fee"]
            entry_order_id = entry_lot.get("order_id")

            # Determine amount to close from this lot
            amount_to_close = min(sell_amount_remaining, entry_amount)
            amount_closed_in_rt += amount_to_close

            # Calculate cost basis and fees for the portion being closed
            cost_basis_lot = amount_to_close * entry_price
            # Calculate pro-rata entry fee for the closed portion
            try:
                pro_rata_entry_fee = (
                    (entry_fee / entry_amount) * amount_to_close
                    if entry_amount > Decimal("1e-9")
                    else Decimal("0")
                )
            except DivisionByZero:
                pro_rata_entry_fee = Decimal("0")

            total_cost_basis += cost_basis_lot
            total_entry_fees += pro_rata_entry_fee
            if entry_order_id:
                buy_order_ids_for_rt.append(entry_order_id)

            # Reduce the amount in the entry lot or remove it
            remaining_in_lot = entry_amount - amount_to_close
            if remaining_in_lot > Decimal("1e-9"):  # Partial close of the lot
                entry_lot["amount"] = remaining_in_lot
                # Reduce fee proportionally (optional but good practice)
                try:
                    entry_lot["fee"] = (
                        (entry_fee / entry_amount) * remaining_in_lot
                        if entry_amount > Decimal("1e-9")
                        else Decimal("0")
                    )
                except DivisionByZero:
                    entry_lot["fee"] = Decimal("0")
                logger.debug(
                    f"  Partial close: Used {amount_to_close:.8f} from lot bought at {entry_price:.4f}. Remaining in lot: {entry_lot['amount']:.8f}"
                )
            else:  # Full close of the lot
                self.open_positions[symbol].popleft()  # Remove the fully closed lot
                logger.debug(
                    f"  Full close: Used {amount_to_close:.8f} from lot bought at {entry_price:.4f}. Lot removed."
                )

            sell_amount_remaining -= amount_to_close

        # --- Calculate Round Trip PnL ---
        if amount_closed_in_rt <= Decimal("1e-9"):
            logger.warning(
                f"Sell trade {sell_order_id or 'N/A'} ({symbol}) resulted in closing negligible amount ({amount_closed_in_rt:.8f}). Skipping round trip record."
            )
            return

        if sell_amount_remaining > Decimal("1e-9"):  # Allow small tolerance
            logger.warning(
                f"Sell amount {sell_amount:.8f} for {symbol} exceeded open position amount. PnL calculation based on closed amount {amount_closed_in_rt:.8f}."
            )

        # Calculate net proceeds for the closed amount
        total_proceeds_gross = amount_closed_in_rt * sell_price
        # Assume sell_fee applies to the entire sell order amount initially recorded
        # Pro-rata exit fee based on the portion actually closed *if* the sell order wasn't fully matched
        try:
            pro_rata_exit_fee = (
                (sell_fee / sell_amount) * amount_closed_in_rt
                if sell_amount > Decimal("1e-9")
                else Decimal("0")
            )
        except DivisionByZero:
            pro_rata_exit_fee = Decimal("0")

        total_proceeds_net = total_proceeds_gross - pro_rata_exit_fee

        # Calculate PnL
        pnl = total_proceeds_net - (total_cost_basis + total_entry_fees)
        duration = sell_timestamp - first_entry_timestamp

        # Calculate derived metrics safely (checking for zero denominators)
        try:
            avg_entry_price_calc = (
                total_cost_basis / amount_closed_in_rt
                if amount_closed_in_rt > Decimal("1e-9")
                else Decimal("0")
            )
        except DivisionByZero:
            avg_entry_price_calc = Decimal("0")
        try:
            pnl_pct_calc = (
                (pnl / total_cost_basis) * 100
                if total_cost_basis > Decimal("1e-9")
                else Decimal("0")
            )
        except DivisionByZero:
            pnl_pct_calc = Decimal("0")  # PnL is 0 if cost basis is 0

        # Record round trip (store values as floats for easier analysis later)
        round_trip_info = {
            "symbol": symbol,
            "entry_timestamp": first_entry_timestamp.isoformat(),
            "exit_timestamp": sell_timestamp.isoformat(),
            "duration_seconds": duration.total_seconds(),
            "amount_closed": float(amount_closed_in_rt),
            "entry_price_avg": float(avg_entry_price_calc),
            "exit_price": float(sell_price),
            "pnl": float(pnl),
            "pnl_pct": float(pnl_pct_calc),
            "entry_fees": float(total_entry_fees),
            "exit_fee": float(pro_rata_exit_fee),
            "buy_order_ids": list(set(buy_order_ids_for_rt)),  # Unique buy order IDs
            "sell_order_id": sell_order_id,
        }
        self.round_trips.append(round_trip_info)
        logger.info(
            f"Round Trip Closed: {symbol}, Amount: {round_trip_info['amount_closed']:.6f}, PnL: {pnl:.4f} {self.quote_currency}"
        )

        # Clean up empty deque for the symbol
        if symbol in self.open_positions and not self.open_positions[symbol]:
            del self.open_positions[symbol]
            logger.debug(f"  Position fully closed for {symbol}.")

    def record_portfolio_value(self, timestamp: datetime, value: Union[float, Decimal, str]):
        """Record the portfolio value at a specific timestamp, ensuring Decimal."""
        try:
            # Ensure value is Decimal
            if not isinstance(value, Decimal):
                value_dec = Decimal(str(value))
            else:
                value_dec = value

            # Ensure timestamp is timezone-aware UTC
            if not isinstance(timestamp, datetime):
                raise TypeError(f"Invalid timestamp type: {type(timestamp)}")
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            elif timestamp.tzinfo != timezone.utc:
                timestamp = timestamp.astimezone(timezone.utc)

            self.portfolio_value_history.append((timestamp, value_dec))
            self.final_capital = value_dec  # Update final capital with the latest value

        except (InvalidOperation, ValueError, TypeError) as e:
            logger.error(f"Failed to record portfolio value ({value}) at {timestamp}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error recording portfolio value: {e}", exc_info=True)

    def calculate_metrics(self) -> Dict:
        """Calculate final performance metrics including per-trade stats."""
        logger.info("Calculating backtest performance metrics...")
        num_total_executed = len(self.executed_trades)  # All closed trades recorded
        num_round_trips = len(self.round_trips)

        # --- Basic Metrics ---
        start_ts, end_ts = None, None
        start_val, end_val = self.initial_capital, self.final_capital
        if self.portfolio_value_history:
            # Sort just in case, although should be appended in order
            self.portfolio_value_history.sort(key=lambda x: x[0])
            start_ts = self.portfolio_value_history[0][0]
            end_ts = self.portfolio_value_history[-1][0]
            end_val = self.portfolio_value_history[-1][1]
        duration_days = (end_ts - start_ts).days if start_ts and end_ts and end_ts > start_ts else 0

        total_fees = sum(trade.get("fee_dec", Decimal("0")) for trade in self.executed_trades)
        total_pnl = end_val - self.initial_capital
        total_pnl_pct = Decimal("0.0")
        if self.initial_capital > Decimal("1e-9"):
            try:
                total_pnl_pct = (total_pnl / self.initial_capital) * 100
            except DivisionByZero:
                pass

        # --- Round Trip Metrics ---
        wins = 0
        losses = 0
        neutral = 0
        total_win_pnl = Decimal("0.0")
        total_loss_pnl = Decimal("0.0")
        max_win = Decimal("0.0")
        max_loss = Decimal("0.0")
        total_duration_seconds = Decimal("0.0")

        if num_round_trips > 0:
            for rt in self.round_trips:
                pnl = Decimal(str(rt.get("pnl", 0.0)))
                if pnl > Decimal("1e-9"):
                    wins += 1
                    total_win_pnl += pnl
                    max_win = max(max_win, pnl)
                elif pnl < Decimal("-1e-9"):
                    losses += 1
                    total_loss_pnl += pnl
                    max_loss = min(max_loss, pnl)
                else:
                    neutral += 1
                try:
                    total_duration_seconds += Decimal(str(rt.get("duration_seconds", 0.0)))
                except InvalidOperation:
                    pass

            try:
                win_rate = (
                    (Decimal(wins) / Decimal(num_round_trips)) * 100
                    if num_round_trips > 0
                    else Decimal("0")
                )
            except DivisionByZero:
                win_rate = Decimal("0")
            try:
                avg_win = total_win_pnl / Decimal(wins) if wins > 0 else Decimal("0")
            except DivisionByZero:
                avg_win = Decimal("0")
            try:
                avg_loss = total_loss_pnl / Decimal(losses) if losses > 0 else Decimal("0")
            except DivisionByZero:
                avg_loss = Decimal("0")
            try:
                avg_duration_seconds = (
                    total_duration_seconds / Decimal(num_round_trips)
                    if num_round_trips > 0
                    else Decimal("0")
                )
            except DivisionByZero:
                avg_duration_seconds = Decimal("0")
            profit_factor = Decimal("0.0")
            if total_loss_pnl < Decimal("-1e-9"):
                try:
                    profit_factor = total_win_pnl / abs(total_loss_pnl)
                except DivisionByZero:
                    profit_factor = Decimal("inf")
            elif total_win_pnl > Decimal("1e-9"):
                profit_factor = Decimal("inf")
        else:
            win_rate = avg_win = avg_loss = avg_duration_seconds = profit_factor = Decimal("0.0")

        # --- Risk Metrics (Sharpe, Sortino, Drawdown) ---
        returns_list = []
        sharpe_ratio = np.nan
        sortino_ratio = np.nan
        if len(self.portfolio_value_history) > 1:
            values = np.array([float(v) for _, v in self.portfolio_value_history])
            valid_indices = np.where(values[:-1] > 1e-9)[0]
            if len(valid_indices) > 0:
                returns_list = (values[1:][valid_indices] / values[:-1][valid_indices]) - 1
            else:
                logger.warning(
                    "Could not calculate returns (all starting portfolio values zero/negative)."
                )

        if len(returns_list) > 1:
            mean_return = np.mean(returns_list)
            std_dev_return = np.std(returns_list)
            negative_returns = returns_list[returns_list < 0]
            downside_std_dev = np.std(negative_returns) if len(negative_returns) > 1 else 0.0

            annualization_factor = 252
            if len(self.portfolio_value_history) > 1:
                try:
                    time_diffs = np.diff([t.timestamp() for t, _ in self.portfolio_value_history])
                    median_diff_seconds = np.median(time_diffs)
                    if median_diff_seconds > 0:
                        periods_per_year = (365.25 * 24 * 3600) / median_diff_seconds
                        annualization_factor = math.sqrt(max(1.0, periods_per_year))
                    else:
                        logger.warning("Median time difference between portfolio values is zero.")
                except Exception as time_calc_e:
                    logger.warning(
                        f"Could not calculate annualization factor from timestamps: {time_calc_e}. Using default {annualization_factor}."
                    )

            if std_dev_return > 1e-9:
                # Ensure mean_return is finite before calculating Sharpe
                if np.isfinite(mean_return):
                    sharpe_ratio = (mean_return * annualization_factor) / std_dev_return
                else:
                    logger.warning("Mean return is non-finite, cannot calculate Sharpe Ratio.")
            else:
                logger.warning(
                    "Standard deviation of returns is zero, cannot calculate Sharpe Ratio."
                )

            if downside_std_dev > 1e-9:
                # Ensure mean_return is finite before calculating Sortino
                if np.isfinite(mean_return):
                    sortino_ratio = (mean_return * annualization_factor) / downside_std_dev
                else:
                    logger.warning("Mean return is non-finite, cannot calculate Sortino Ratio.")
            else:
                logger.warning(
                    "Downside deviation of returns is zero, cannot calculate Sortino Ratio."
                )

        # --- Max Drawdown ---
        max_drawdown_pct = 0.0
        peak_value = float("-inf")
        if len(self.portfolio_value_history) > 1:
            values = np.array([float(v) for _, v in self.portfolio_value_history])
            if np.any(values <= 1e-9):
                logger.warning(
                    "Portfolio value hit zero or negative during backtest. Drawdown calculation might be affected."
                )
            cumulative_max = np.maximum.accumulate(np.maximum(values, 1e-9))
            drawdowns = (values - cumulative_max) / cumulative_max
            max_drawdown_pct = abs(np.min(drawdowns) * 100) if len(drawdowns) > 0 else 0.0

        # --- Assemble Metrics Dictionary (using float for final storage) ---
        self.metrics = {
            "run_details": {
                "start_timestamp": start_ts.isoformat() if start_ts else None,
                "end_timestamp": end_ts.isoformat() if end_ts else None,
                "duration_days": duration_days,
                "initial_capital": float(self.initial_capital),
                "final_capital": float(end_val),
                "quote_currency": self.quote_currency,
            },
            "performance": {
                "total_pnl": float(total_pnl) if total_pnl.is_finite() else "N/A",
                "total_pnl_pct": (float(total_pnl_pct) if total_pnl_pct.is_finite() else "N/A"),
                "total_fees": float(total_fees),
                "max_drawdown_pct": (
                    float(max_drawdown_pct) if np.isfinite(max_drawdown_pct) else "N/A"
                ),
                "sharpe_ratio": (float(sharpe_ratio) if np.isfinite(sharpe_ratio) else "N/A"),
                "sortino_ratio": (float(sortino_ratio) if np.isfinite(sortino_ratio) else "N/A"),
            },
            "trade_stats": {
                "total_trades_executed": num_total_executed,
                "total_round_trips": num_round_trips,
                "wins": wins,
                "losses": losses,
                "neutral": neutral,
                "win_rate_pct": float(win_rate) if win_rate.is_finite() else "N/A",
                "profit_factor": (
                    float(profit_factor) if profit_factor.is_finite() else "inf"
                ),  # Keep 'inf' possible here
                "avg_win_pnl": float(avg_win) if avg_win.is_finite() else "N/A",
                "avg_loss_pnl": float(avg_loss) if avg_loss.is_finite() else "N/A",
                "max_win_pnl": float(max_win),
                "max_loss_pnl": float(max_loss),
                "avg_trade_duration_seconds": (
                    float(avg_duration_seconds) if avg_duration_seconds.is_finite() else "N/A"
                ),
                "avg_trade_duration_human": (
                    str(timedelta(seconds=float(avg_duration_seconds)))
                    if avg_duration_seconds.is_finite() and avg_duration_seconds > 0
                    else "N/A"
                ),
            },
        }

        # --- FIX: Format values defensively before logging ---
        try:
            total_pnl_str = f"{float(total_pnl):.2f}" if total_pnl.is_finite() else "N/A"
        except (ValueError, TypeError):
            total_pnl_str = "ERR"
        try:
            total_pnl_pct_str = (
                f"{float(total_pnl_pct):.2f}" if total_pnl_pct.is_finite() else "N/A"
            )
        except (ValueError, TypeError):
            total_pnl_pct_str = "ERR"
        try:
            win_rate_str = f"{float(win_rate):.2f}" if win_rate.is_finite() else "N/A"
        except (ValueError, TypeError):
            win_rate_str = "ERR"
        try:
            sharpe_ratio_str = f"{sharpe_ratio:.3f}" if np.isfinite(sharpe_ratio) else "N/A"
        except (ValueError, TypeError):
            sharpe_ratio_str = "ERR"
        try:
            max_drawdown_pct_str = (
                f"{float(max_drawdown_pct):.2f}" if np.isfinite(max_drawdown_pct) else "N/A"
            )
        except (ValueError, TypeError):
            max_drawdown_pct_str = "ERR"

        # Use the safe string variables in the log message
        logger.info(
            f"Calculated Metrics: PnL={total_pnl_str} ({total_pnl_pct_str}%), RoundTrips={num_round_trips}, WinRate={win_rate_str}%, Sharpe={sharpe_ratio_str}, MaxDrawdown={max_drawdown_pct_str}%"
        )
        # ----------------------------------------------------
        return self.metrics

    def save_results(self, output_dir: Path, run_id: str):
        """Save backtest trades, metrics, portfolio history, and round trips to JSON files."""
        if not self.metrics:
            logger.warning("No metrics calculated yet. Call calculate_metrics() first.")
            # Optionally calculate metrics here if not already done
            self.calculate_metrics()
            if not self.metrics:  # If still no metrics after calculation
                logger.error("Failed to calculate metrics. Cannot save results.")
                return

        try:
            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create output directory {output_dir}: {e}")
            return  # Cannot save if directory cannot be created

        # Define files to save
        files_to_save = {
            # Store executed trades as recorded (should have Decimals converted already if needed by JSON)
            "trades": (
                output_dir / f"backtest_trades_{run_id}.json",
                self.executed_trades,
            ),
            # Metrics dict now contains strings for potentially non-finite values
            "metrics": (output_dir / f"backtest_metrics_{run_id}.json", self.metrics),
            # Convert portfolio history Timestamps and Decimals for JSON
            "portfolio": (
                output_dir / f"backtest_portfolio_{run_id}.json",
                [
                    {
                        "timestamp": ts.isoformat(),
                        "value": float(val) if val.is_finite() else None,
                    }
                    for ts, val in self.portfolio_value_history
                ],
            ),  # Handle non-finite portfolio values
            # Round trip data should be floats, but add check just in case
            "round_trips": (
                output_dir / f"backtest_round_trips_{run_id}.json",
                [
                    {
                        k: (float(v) if isinstance(v, Decimal) and v.is_finite() else v)
                        for k, v in rt.items()
                    }
                    for rt in self.round_trips
                ],
            ),
        }

        # Define a robust JSON encoder
        def robust_json_default(obj):
            if isinstance(obj, Decimal):
                # Convert finite Decimals to float, represent non-finite as strings
                return float(obj) if obj.is_finite() else str(obj)
            if isinstance(obj, (datetime, pd.Timestamp)):
                return obj.isoformat()
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                # Convert finite floats, represent non-finite as strings
                return float(obj) if np.isfinite(obj) else str(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.bool_, bool)):
                return bool(obj)  # Handle numpy bool
            # Add handling for other types if necessary
            try:
                return str(obj)  # Fallback to string representation
            except Exception:
                return "[Unserializable Object]"

        # Save each file atomically
        for name, (file_path, data) in files_to_save.items():
            temp_file = file_path.with_suffix(f".tmp_{time.time_ns()}")
            try:
                with temp_file.open("w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, default=robust_json_default)
                temp_file.replace(file_path)  # Atomic rename/replace
                logger.info(f"Saved {name} data ({len(data)} items) to {file_path.name}")
            except (OSError, TypeError, ValueError) as e:
                logger.error(
                    f"Failed to save backtest {name} data to {file_path}: {e}",
                    exc_info=True,
                )
                if temp_file.exists():
                    try:
                        temp_file.unlink()
                    except OSError:
                        pass  # Ignore cleanup error
            except Exception as e:  # Catch other unexpected errors
                logger.error(f"Unexpected error saving {name} data: {e}", exc_info=True)
                if temp_file.exists():
                    try:
                        temp_file.unlink()
                    except OSError:
                        pass
