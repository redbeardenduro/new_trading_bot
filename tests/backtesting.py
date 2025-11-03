# enhanced_trading_bot/tests/backtesting.py
"""
Backtesting Engine for the Enhanced Crypto Trading Bot.

Simulates trading strategies using historical OHLCV data, injects historical
sentiment and AI signals, replicates the decision-making logic of the
core bot and portfolio manager (now decoupled), and calculates detailed
performance metrics using the BacktestMetrics class.
"""

import argparse
import bisect  # For efficient timestamp searching
import hashlib  # For unique run IDs
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal  # Added InvalidOperation
from decimal import DivisionByZero, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# --- Core Imports (Relative to project root) ---
# Assuming common_logger setup adds project root to sys.path
try:
    from common.common_logger import (CACHE_DIR, DATA_DIR, LOG_DIR,
                                      PROJECT_ROOT, get_logger)
    from core.bot import \
        MultiCryptoTradingBot  # For reusing logic/calculations
    from core.config import BotConfig
    # Import the interface
    from core.interfaces import IExchangeClient, IPortfolioManager
    from core.portfolio_manager import \
        PortfolioManager  # Import concrete class for instantiation
    # --- Import the metrics class from its new location ---
    from tests.metrics import BacktestMetrics
    from utils.enhanced_logging import \
        EnhancedLogger  # For separate backtest logging
except ImportError as e:
    # Logger might not be available yet
    print(f"ERROR: Failed to import core modules: {e}", file=sys.stderr)
    # Attempt to add project root if running script directly
    import sys

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    try:
        from common.common_logger import (CACHE_DIR, DATA_DIR, LOG_DIR,
                                          PROJECT_ROOT, get_logger)
        from core.bot import MultiCryptoTradingBot
        from core.config import BotConfig
        from core.interfaces import IExchangeClient, IPortfolioManager
        from core.portfolio_manager import PortfolioManager
        from tests.metrics import BacktestMetrics
        from utils.enhanced_logging import EnhancedLogger
    except ImportError as inner_e:
        print(
            f"ERROR: Failed import after adding project root: {inner_e}",
            file=sys.stderr,
        )
        sys.exit(1)

# Configure a dedicated logger for backtesting
try:
    bt_log_dir = DATA_DIR / "logs" / "backtests"
    bt_log_dir.mkdir(parents=True, exist_ok=True)
    backtest_logger_instance = EnhancedLogger(
        name="backtester",
        log_level=logging.INFO,  # Default, can be overridden
        log_dir=bt_log_dir,
        log_file_name="backtest_run.log",
        rotation_interval="H",  # Hourly rotation for potentially long runs
        rotation_backup_count=24,
    )
    logger = backtest_logger_instance.logger  # Use the configured logger instance
except Exception as log_e:
    # Fallback basic logging if EnhancedLogger fails
    print(
        f"ERROR: Failed to set up enhanced logger: {log_e}. Using basic logging.",
        file=sys.stderr,
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("backtester_fallback")


# --- Constants ---
DEFAULT_INITIAL_BALANCE_QUOTE = 10000.0
DEFAULT_INITIAL_BALANCE_BASE = 0.0
MIN_LOOKBACK_PERIOD = 50  # Minimum candles needed before starting simulation for TA indicators
MAX_TIMESTAMP_DIFF_SECONDS = 3600 * 3  # Max diff (3 hours) allowed for matching signal data


# --- Helper Classes ---


# Implementing IExchangeClient (SimulatedExchange)
class SimulatedExchange(IExchangeClient):
    """
    Mocks the exchange client interface for backtesting. Handles errors gracefully.
    """

    def __init__(
        self,
        config: BotConfig,
        pairs: List[str],
        timeframe: str,
        data_dir: Path,  # Base data dir (for logs, results, signals)
        historical_ohlcv_source_dir: Path,
        historical_data: Optional[Dict[str, pd.DataFrame]] = None,
    ):  # Allow pre-loaded data

        if config is None:
            raise ValueError("BotConfig cannot be None for BacktestRunner.")
        if not isinstance(pairs, list) or not pairs:
            raise ValueError("Pairs list cannot be empty.")
        if not isinstance(timeframe, str) or not timeframe:
            raise ValueError("Timeframe cannot be empty.")
        if not isinstance(data_dir, Path) or not data_dir.is_dir():
            raise ValueError(f"Invalid data directory: {data_dir}")
        if (
            not isinstance(historical_ohlcv_source_dir, Path)
            or not historical_ohlcv_source_dir.is_dir()
        ):
            logger.error(
                f"Invalid or non-existent historical OHLCV data directory provided: {historical_ohlcv_source_dir}"
            )
            raise ValueError(
                f"Invalid historical OHLCV data directory: {historical_ohlcv_source_dir}"
            )

        self.config = config
        self.pairs = pairs
        self.timeframe = timeframe
        self.historical_ohlcv_dir = historical_ohlcv_source_dir
        self.sentiment_data_dir = DATA_DIR / "sentiment"
        self.ai_signals_data_dir = DATA_DIR / "ai_signals"
        self.historical_data: Dict[str, pd.DataFrame] = (
            historical_data if historical_data is not None else {}
        )
        self.preloaded_data_used = historical_data is not None
        self.loaded_sentiment_data: Dict[str, List[Tuple[datetime, Dict]]] = {}
        self.loaded_ai_signals_data: Dict[str, List[Tuple[datetime, Dict]]] = {}

        # Simulation State Variables
        self.current_step = 0  # Initialize step counter

        # ------ FIX 1: Access quote_currency from config.bot.quote_currencies list ------
        try:
            # Check if quote_currencies exists and is a non-empty list
            if (
                not hasattr(self.config.bot, "quote_currencies")
                or not isinstance(self.config.bot.quote_currencies, list)
                or not self.config.bot.quote_currencies
            ):
                raise ValueError(
                    "Configuration missing 'bot.quote_currencies' list or list is empty."
                )
            # Take the first quote currency from the list
            self.quote_currency = self.config.bot.quote_currencies[0].upper()
            logger.info(f"SimulatedExchange using quote currency: {self.quote_currency}")
        except AttributeError:
            logger.critical("FATAL: 'quote_currencies' not found under [bot] in configuration.")
            raise AttributeError("Configuration missing 'bot.quote_currencies'")
        except (ValueError, TypeError, IndexError) as e:
            logger.critical(f"FATAL: Error processing bot.quote_currencies: {e}")
            raise ValueError(f"Error processing bot.quote_currencies: {e}")
        # ---------------------------------------------------------------------------------

        # ------ FIX 2: Access simulation parameters from config.portfolio.simulation ------
        try:
            sim_config = self.config.portfolio.simulation
            # Convert percentages to decimal rates
            fee_rate_pct = Decimal(str(sim_config.fee_rate_percent))
            slippage_pct = Decimal(str(sim_config.slippage_percent))
            self.fee_rate = fee_rate_pct / Decimal("100.0")
            self.slippage = slippage_pct / Decimal("100.0")
            self.min_order_value_quote = Decimal(str(sim_config.min_order_value_quote))
            logger.info(
                f"SimulatedExchange using Fee Rate: {self.fee_rate}, Slippage: {self.slippage}, Min Order: {self.min_order_value_quote}"
            )
        except AttributeError as e:
            logger.critical(
                f"FATAL: Missing required simulation parameter under [portfolio][simulation] in configuration: {e}"
            )
            raise AttributeError(f"Configuration missing required simulation parameter: {e}")
        except (InvalidOperation, TypeError) as e:
            logger.critical(
                f"FATAL: Invalid numeric value for simulation parameter in configuration: {e}"
            )
            raise ValueError(f"Invalid numeric value for simulation parameter: {e}")
        # ----------------------------------------------------------------------------------

        self.balances: Dict[str, Decimal] = {}  # Balances stored as Decimal
        self.trades: List[Dict] = []  # Record of simulated trades

        # Initialize markets and balances after basic config is processed
        if not self.historical_data:
            logger.warning(
                "SimulatedExchange initialized without historical_data. Markets/Balances might be incomplete until data is loaded."
            )
            self.markets: Dict = {}  # Init empty
            self.balances = {
                self.quote_currency: Decimal(str(DEFAULT_INITIAL_BALANCE_QUOTE))
            }  # Init with default quote balance
        else:
            self.markets = self._create_mock_markets()  # Create markets based on data
            self._initialize_balances()  # Initialize balances based on markets/config

    def _create_mock_markets(self) -> Dict:
        """Create a basic market structure needed for validation/precision."""
        markets = {}
        if not isinstance(self.historical_data, dict) or not self.historical_data:
            logger.warning("Cannot create mock markets: historical_data not available or empty.")
            return markets

        for pair, df in self.historical_data.items():
            if not isinstance(pair, str) or "/" not in pair:
                continue
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            try:
                base, quote = pair.split("/")
                # Ensure quote currency matches the simulation's quote currency
                if quote.upper() != self.quote_currency:
                    logger.warning(
                        f"Skipping market creation for {pair}: Quote currency '{quote}' does not match simulation quote '{self.quote_currency}'."
                    )
                    continue

                min_cost = float(self.min_order_value_quote)
                markets[pair] = {
                    "symbol": pair,
                    "base": base.upper(),
                    "quote": quote.upper(),
                    "active": True,
                    "precision": {
                        "amount": 8,
                        "price": 8,
                    },  # Increased price precision might be safer
                    "limits": {
                        "amount": {"min": 1e-8, "max": 1e8},
                        "price": {"min": 1e-8, "max": 1e8},
                        "cost": {"min": min_cost, "max": 1e8},
                    },
                    "info": {"simulated": True},
                }
            except Exception as e:
                logger.error(f"Error creating mock market for {pair}: {e}")
                continue
        logger.debug(f"Created mock markets for: {list(markets.keys())}")
        return markets

    def _initialize_balances(self):
        """Set initial balances for quote and base currencies safely."""
        try:
            self.balances[self.quote_currency] = Decimal(str(DEFAULT_INITIAL_BALANCE_QUOTE))
            all_base_currencies = set()
            if isinstance(self.markets, dict):
                for market_info in self.markets.values():
                    if isinstance(market_info, dict) and "base" in market_info:
                        all_base_currencies.add(market_info["base"])

            for base in all_base_currencies:
                self.balances[base] = Decimal(str(DEFAULT_INITIAL_BALANCE_BASE))
            logger.info(
                f"Initialized balances: { {k: float(v) for k, v in self.balances.items()} }"
            )
        except (InvalidOperation, TypeError) as e:
            logger.error(f"Error initializing balances: {e}. Using empty balances.")
            self.balances = {self.quote_currency: Decimal("0")}

    def set_step(self, step: int):
        """Set the current simulation step (index in the historical data)."""
        if isinstance(step, int) and step >= 0:
            self.current_step = step
        else:
            logger.error(
                f"Invalid step value received: {step}. Keeping previous step {self.current_step}."
            )

    def get_current_timestamp(self, pair: str) -> Optional[datetime]:
        """Get the timestamp for the current simulation step, returns tz-aware datetime."""
        if not isinstance(self.historical_data, dict):
            return None
        df = self.historical_data.get(pair)
        if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex):
            return None
        if not (0 <= self.current_step < len(df)):
            return None
        try:
            ts = df.index[self.current_step]
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            return ts.to_pydatetime()
        except IndexError:
            logger.error(
                f"IndexError getting timestamp step {self.current_step} for pair {pair} (len={len(df)})."
            )
            return None
        except Exception as e:
            logger.error(
                f"Error getting timestamp step {self.current_step} pair {pair}: {e}",
                exc_info=True,
            )
            return None

    def get_current_price(self, pair: str) -> Optional[Decimal]:
        """Get the closing price for the current simulation step as Decimal."""
        if not isinstance(self.historical_data, dict):
            return None
        df = self.historical_data.get(pair)
        if df is None or df.empty:
            return None
        if not (0 <= self.current_step < len(df)):
            return None
        try:
            price_val = df.iloc[self.current_step]["close"]
            if pd.isna(price_val):
                return None
            price = Decimal(str(price_val))
            return price if price > Decimal("1e-9") else None
        except (IndexError, KeyError):
            logger.error(
                f"Error accessing price at step {self.current_step} for pair {pair} (Key/Index)."
            )
            return None
        except (InvalidOperation, TypeError) as e:
            logger.error(
                f"Error converting price to Decimal step {self.current_step} pair {pair}: {e}"
            )
            return None
        except Exception as e:
            logger.error(
                f"Unexpected error getting price step {self.current_step} pair {pair}: {e}",
                exc_info=True,
            )
            return None

    def test_connection(self) -> Tuple[bool, Dict[str, Any]]:
        logger.info("SimulatedExchange: test_connection() called.")
        return True, {
            "connected": True,
            "server_time": datetime.now(timezone.utc).isoformat(),
            "latency_ms": 1,
            "markets_loaded": bool(self.markets),
            "simulated": True,
        }

    def get_balance(self, force_refresh: bool = False) -> Dict[str, float]:
        return {k: float(v) for k, v in self.balances.items()}

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 100,
        since: Optional[int] = None,
        force_refresh: bool = False,
    ) -> Optional[List[Dict]]:
        """Gets OHLCV data up to the current simulation step."""
        if not isinstance(self.historical_data, dict):
            return None
        df = self.historical_data.get(symbol)
        if df is None or df.empty:
            return None
        end_index = self.current_step + 1
        if end_index < 1:
            return []
        start_index = max(0, end_index - limit)
        data_slice = df.iloc[start_index:end_index]
        if data_slice.empty:
            return []

        ohlcv_list = []
        for timestamp, row in data_slice.iterrows():
            try:
                ts_ms = int(timestamp.timestamp() * 1000)
                o = float(row["open"])
                h = float(row["high"])
                l = float(row["low"])
                c = float(row["close"])
                v = float(row["volume"])
                ohlcv_list.append(
                    {
                        "timestamp_ms": ts_ms,
                        "timestamp": timestamp.isoformat(),
                        "open": o,
                        "high": h,
                        "low": l,
                        "close": c,
                        "volume": v,
                    }
                )
            except (ValueError, TypeError, AttributeError, KeyError) as e:
                logger.warning(f"Skipping invalid OHLCV data at {timestamp} for {symbol}: {e}")
                continue
        return ohlcv_list

    def get_ticker(self, symbol: str, force_refresh: bool = False) -> Optional[Dict]:
        """Generates a ticker based on the current simulation step's candle."""
        if not isinstance(self.historical_data, dict):
            return None
        df = self.historical_data.get(symbol)
        if df is None or df.empty:
            return None
        if not (0 <= self.current_step < len(df)):
            return None

        try:
            current_candle = df.iloc[self.current_step]
            ts = self.get_current_timestamp(symbol)
            if ts is None:
                return None

            close_price_dec = Decimal(str(current_candle["close"]))
            open_price_dec = Decimal(str(current_candle["open"]))
            high_price_dec = Decimal(str(current_candle["high"]))
            low_price_dec = Decimal(str(current_candle["low"]))
            volume_dec = Decimal(str(current_candle["volume"]))

            spread = close_price_dec * (self.slippage / Decimal("2.0"))
            bid = float(close_price_dec - spread)
            ask = float(close_price_dec + spread)
            last = float(close_price_dec)
            open_price = float(open_price_dec)

            change_pct = 0.0
            if open_price_dec > Decimal("1e-9"):
                try:
                    change_pct = float(
                        ((close_price_dec - open_price_dec) / open_price_dec) * Decimal("100")
                    )
                except DivisionByZero:
                    pass

            return {
                "symbol": symbol,
                "timestamp": int(ts.timestamp() * 1000),
                "datetime": ts.isoformat(),
                "high": float(high_price_dec),
                "low": float(low_price_dec),
                "bid": bid,
                "ask": ask,
                "last": last,
                "open": open_price,
                "volume": float(volume_dec),
                "change_percent": change_pct,
            }
        except (IndexError, KeyError):
            logger.error(
                f"Error accessing ticker data at step {self.current_step} for {symbol} (Key/Index)."
            )
            return None
        except (InvalidOperation, TypeError) as e:
            logger.error(f"Error converting ticker data to Decimal/float for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting ticker for {symbol}: {e}", exc_info=True)
            return None

    def get_order_book(
        self, symbol: str, limit: int = 10, force_refresh: bool = False
    ) -> Optional[Dict]:
        """Generates a mock order book based on the current ticker."""
        ticker = self.get_ticker(symbol)
        if not ticker or not isinstance(ticker, dict):
            return None
        bid = ticker.get("bid")
        ask = ticker.get("ask")
        if bid is None or ask is None:
            return None

        try:
            bid_dec = Decimal(str(bid))
            ask_dec = Decimal(str(ask))
            limit_i = max(1, int(limit))

            mock_bids = [
                [
                    float(bid_dec * (Decimal("1") - Decimal(str(i * 0.0001)))),
                    float(Decimal("0.1") * Decimal(str(limit_i - i))),
                ]
                for i in range(limit_i)
            ]
            mock_asks = [
                [
                    float(ask_dec * (Decimal("1") + Decimal(str(i * 0.0001)))),
                    float(Decimal("0.1") * Decimal(str(limit_i - i))),
                ]
                for i in range(limit_i)
            ]

            spread = float(ask_dec - bid_dec)
            spread_pct = (
                float(((ask_dec - bid_dec) / bid_dec) * Decimal("100"))
                if bid_dec > Decimal("1e-9")
                else 0.0
            )

            return {
                "symbol": symbol,
                "bids": mock_bids,
                "asks": mock_asks,
                "timestamp": ticker.get("timestamp", int(time.time() * 1000)),
                "datetime": ticker.get("datetime", datetime.now(timezone.utc).isoformat()),
                "nonce": self.current_step,
                "spread": spread,
                "spread_percent": spread_pct,
            }
        except (ValueError, TypeError, InvalidOperation, IndexError) as e:
            logger.error(f"Error generating mock order book for {symbol}: {e}")
            return None

    def _validate_symbol(self, symbol: str) -> bool:
        if not isinstance(self.historical_data, dict):
            return False
        df = self.historical_data.get(symbol)
        return isinstance(df, pd.DataFrame) and not df.empty

    def create_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        params: Dict = {},
    ) -> Dict:
        """Simulates order execution based on current step's data."""
        ts_now = datetime.now(timezone.utc)
        order_id_base = f"sim_{side}_{symbol.replace('/', '')}_{self.current_step}_{int(ts_now.timestamp()*1000)}"

        if not self._validate_symbol(symbol):
            return self._create_order_response(
                order_id_base,
                symbol,
                side,
                amount,
                status="rejected",
                reason="Invalid symbol",
            )
        if order_type != "market":
            logger.warning(
                f"SimulatedExchange only supports market orders. Treating {order_type} as market."
            )
            order_type = "market"
        try:
            order_amount_dec = Decimal(str(amount))
            if order_amount_dec <= Decimal("1e-9"):
                raise ValueError("Order amount must be positive")
        except (InvalidOperation, ValueError, TypeError) as e:
            return self._create_order_response(
                order_id_base,
                symbol,
                side,
                amount,
                status="rejected",
                reason=f"Invalid amount: {e}",
            )

        try:
            base_currency, quote_currency_pair = symbol.split("/")
            base_currency = base_currency.upper()
            quote_currency_pair = quote_currency_pair.upper()
            # Validate pair's quote currency matches simulation's quote currency
            if quote_currency_pair != self.quote_currency:
                reason = f"Pair quote currency '{quote_currency_pair}' does not match simulation quote '{self.quote_currency}'"
                return self._create_order_response(
                    order_id_base,
                    symbol,
                    side,
                    amount,
                    status="rejected",
                    reason=reason,
                )
        except ValueError:
            return self._create_order_response(
                order_id_base,
                symbol,
                side,
                amount,
                status="rejected",
                reason=f"Invalid symbol format: {symbol}",
            )

        exec_price_dec = self.get_current_price(symbol)

        if exec_price_dec is None:
            return self._create_order_response(
                order_id_base,
                symbol,
                side,
                amount,
                status="rejected",
                reason="No valid execution price available",
            )

        try:
            exec_price_dec = (
                exec_price_dec * (Decimal("1") + self.slippage)
                if side == "buy"
                else exec_price_dec * (Decimal("1") - self.slippage)
            )
            order_value_dec = order_amount_dec * exec_price_dec
            fee_dec = abs(order_value_dec * self.fee_rate)
        except (InvalidOperation, DivisionByZero, OverflowError) as calc_err:
            return self._create_order_response(
                order_id_base,
                symbol,
                side,
                amount,
                status="rejected",
                reason=f"Calculation error: {calc_err}",
            )

        if order_value_dec < self.min_order_value_quote:
            return self._create_order_response(
                order_id_base,
                symbol,
                side,
                amount,
                status="rejected",
                reason=f"Value {order_value_dec:.2f} < min {self.min_order_value_quote}",
            )

        current_quote_balance = self.balances.get(
            self.quote_currency, Decimal("0")
        )  # Use self.quote_currency
        current_base_balance = self.balances.get(base_currency, Decimal("0"))

        if side == "buy":
            required_quote = order_value_dec + fee_dec
            if current_quote_balance < required_quote:
                return self._create_order_response(
                    order_id_base,
                    symbol,
                    side,
                    amount,
                    status="rejected",
                    reason=f"Insufficient {self.quote_currency} ({current_quote_balance:.4f} < {required_quote:.4f})",
                )
            self.balances[self.quote_currency] = current_quote_balance - required_quote
            self.balances[base_currency] = current_base_balance + order_amount_dec
        else:  # sell
            required_base = order_amount_dec
            if current_base_balance < required_base:
                return self._create_order_response(
                    order_id_base,
                    symbol,
                    side,
                    amount,
                    status="rejected",
                    reason=f"Insufficient {base_currency} ({current_base_balance:.8f} < {required_base:.8f})",
                )
            proceeds_quote = order_value_dec - fee_dec
            self.balances[base_currency] = current_base_balance - required_base
            self.balances[self.quote_currency] = (
                current_quote_balance + proceeds_quote
            )  # Add to self.quote_currency

        exec_ts = self.get_current_timestamp(symbol) or ts_now
        trade_info = {
            "id": f"{order_id_base}_filled",
            "info": {"simulated": True, "step": self.current_step},
            "timestamp": int(exec_ts.timestamp() * 1000),
            "datetime": exec_ts.isoformat(),
            "symbol": symbol,
            "type": order_type,
            "side": side,
            "price": float(exec_price_dec),
            "amount": float(order_amount_dec),
            "cost": float(order_value_dec),
            "fee": {
                "cost": float(fee_dec),
                "currency": self.quote_currency,
                "rate": float(self.fee_rate),
            },  # Use self.quote_currency
            "filled": float(order_amount_dec),
            "remaining": 0.0,
            "status": "closed",
            "average": float(exec_price_dec),
            "trades": [],
        }
        self.trades.append(trade_info)
        logger.info(
            f"Simulated Trade Executed: {trade_info['side']} {trade_info['amount']:.6f} {trade_info['symbol']} @ {trade_info['average']:.4f}, Fee: {trade_info['fee']['cost']:.4f} {trade_info['fee']['currency']}"
        )
        return trade_info

    def _create_order_response(
        self,
        base_id: str,
        symbol: str,
        side: str,
        amount: float,
        status: str,
        reason: str = "Unknown",
    ) -> Dict:
        """Helper to create a consistent rejected/failed order response."""
        ts = datetime.now(timezone.utc)
        order_id = f"{base_id}_{status}"
        logger.warning(f"Order {status}: {side} {amount} {symbol}. Reason: {reason}")
        return {
            "id": order_id,
            "info": {
                "simulated": True,
                "step": self.current_step,
                "error": reason,
                "status_reason": status,
            },
            "timestamp": int(ts.timestamp() * 1000),
            "datetime": ts.isoformat(),
            "symbol": symbol,
            "type": "market",
            "side": side,
            "amount": float(amount),
            "price": None,
            "cost": 0.0,
            "fee": None,
            "filled": 0.0,
            "remaining": float(amount),
            "status": status,
            "average": None,
            "trades": [],
        }

    # --- Convenience Methods ---
    def create_market_buy_order(self, symbol: str, amount: float, params={}) -> Dict:
        return self.create_order(symbol, "market", "buy", amount, None, params)

    def create_market_sell_order(self, symbol: str, amount: float, params={}) -> Dict:
        return self.create_order(symbol, "market", "sell", amount, None, params)

    def create_limit_buy_order(self, symbol: str, amount: float, price: float, params={}) -> Dict:
        return self.create_order(symbol, "limit", "buy", amount, price, params)

    def create_limit_sell_order(self, symbol: str, amount: float, price: float, params={}) -> Dict:
        return self.create_order(symbol, "limit", "sell", amount, price, params)


# --- BacktestRunner Class ---
class BacktestRunner:
    """
    Orchestrates the backtesting process using refactored components and BacktestMetrics.
    Handles data loading, simulation setup, loop execution, and results calculation.
    """

    def __init__(
        self,
        config: BotConfig,
        pairs: List[str],
        timeframe: str,
        data_dir: Path,
        historical_ohlcv_source_dir: Path,
        historical_data: Optional[Dict[str, pd.DataFrame]] = None,
    ):

        if config is None:
            raise ValueError("BotConfig cannot be None for BacktestRunner.")
        if not isinstance(pairs, list) or not pairs:
            raise ValueError("Pairs list cannot be empty.")
        if not isinstance(timeframe, str) or not timeframe:
            raise ValueError("Timeframe cannot be empty.")
        if not isinstance(data_dir, Path) or not data_dir.is_dir():
            raise ValueError(f"Invalid data directory: {data_dir}")
        if (
            not isinstance(historical_ohlcv_source_dir, Path)
            or not historical_ohlcv_source_dir.is_dir()
        ):
            logger.error(
                f"Invalid or non-existent historical OHLCV data directory provided: {historical_ohlcv_source_dir}"
            )
            raise ValueError(
                f"Invalid historical OHLCV data directory: {historical_ohlcv_source_dir}"
            )

        self.config = config
        self.pairs = pairs  # Initial list from args
        self.timeframe = timeframe
        self.historical_ohlcv_dir = historical_ohlcv_source_dir
        self.sentiment_data_dir = DATA_DIR / "sentiment"
        self.ai_signals_data_dir = DATA_DIR / "ai_signals"
        self.historical_data: Dict[str, pd.DataFrame] = (
            historical_data if historical_data is not None else {}
        )
        self.preloaded_data_used = historical_data is not None
        self.loaded_sentiment_data: Dict[str, List[Tuple[datetime, Dict]]] = {}
        self.loaded_ai_signals_data: Dict[str, List[Tuple[datetime, Dict]]] = {}

        self.sim_exchange: Optional[SimulatedExchange] = None
        self.bot_logic: Optional[MultiCryptoTradingBot] = None
        self.portfolio_manager_logic: Optional[IPortfolioManager] = None
        self.metrics_calculator: Optional[BacktestMetrics] = None

        try:
            param_hash = hashlib.md5(
                json.dumps(self.config.settings, sort_keys=True, default=str).encode()
            ).hexdigest()[:6]
        except Exception:
            param_hash = "nohash"
        self.run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{param_hash}"
        self.results_dir = DATA_DIR / "backtest_results" / self.run_id
        self.run_metrics: Optional[Dict] = None

    def _load_historical_data(self):
        """Loads historical OHLCV data from the specified historical data directory."""
        logger.info(
            f"Loading historical OHLCV data: {self.pairs} ({self.timeframe}) from {self.historical_ohlcv_dir}"
        )
        missing_pairs = []
        loaded_data = {}

        if not self.historical_ohlcv_dir.exists():
            logger.error(
                f"CRITICAL: Historical OHLCV data directory not found: {self.historical_ohlcv_dir}"
            )
            raise FileNotFoundError(f"Directory not found: {self.historical_ohlcv_dir}")

        # Determine the single quote currency for this run
        try:
            run_quote_currency = self.config.bot.quote_currencies[0].upper()
        except (AttributeError, IndexError, TypeError):
            logger.critical("Could not determine quote currency from config.bot.quote_currencies")
            raise ValueError("Invalid quote currency configuration")

        for pair in self.pairs:
            # Validate pair format and quote currency match
            if "/" not in pair or pair.split("/")[1].upper() != run_quote_currency:
                logger.warning(
                    f"Skipping pair {pair}: Invalid format or quote currency does not match run quote currency '{run_quote_currency}'."
                )
                missing_pairs.append(pair)
                continue

            ohlcv_file = (
                self.historical_ohlcv_dir / f"{pair.replace('/', '_')}_{self.timeframe}_cache.json"
            )

            if not ohlcv_file.exists():
                logger.error(
                    f"CRITICAL: OHLCV file not found: {ohlcv_file}. Cannot backtest {pair}."
                )
                missing_pairs.append(pair)
                continue
            try:
                with ohlcv_file.open("r", encoding="utf-8") as f:
                    content = f.read()
                    if not content.strip():
                        logger.warning(f"OHLCV file is empty: {ohlcv_file}")
                        continue
                    data = json.loads(content)  # Expecting list of dictionaries

                if not isinstance(data, list):
                    logger.error(
                        f"Unexpected format in {ohlcv_file}. Expected top-level list of dictionaries."
                    )
                    missing_pairs.append(pair)
                    continue

                if not data:
                    logger.warning(f"Empty OHLCV list in {ohlcv_file}")
                    continue

                try:
                    df = pd.DataFrame(data)
                except Exception as e:
                    logger.error(
                        f"Error creating DataFrame for {pair} from {ohlcv_file}. Check data structure. Error: {e}",
                        exc_info=True,
                    )
                    missing_pairs.append(pair)
                    continue

                if "timestamp_ms" not in df.columns and "timestamp" not in df.columns:
                    logger.error(
                        f"Missing required timestamp column ('timestamp_ms' or 'timestamp') in {ohlcv_file}"
                    )
                    missing_pairs.append(pair)
                    continue

                if "timestamp_ms" in df.columns:
                    df["timestamp"] = pd.to_datetime(
                        df["timestamp_ms"], unit="ms", utc=True, errors="coerce"
                    )
                else:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

                df = df.dropna(subset=["timestamp"])
                if df.empty:
                    logger.warning(f"No valid timestamps found in {ohlcv_file} after conversion.")
                    continue

                df = df.set_index("timestamp")

                numeric_cols = ["open", "high", "low", "close", "volume"]
                missing_numeric = [col for col in numeric_cols if col not in df.columns]
                if missing_numeric:
                    logger.error(f"Missing numeric columns {missing_numeric} in {ohlcv_file}")
                    missing_pairs.append(pair)
                    continue

                for col in numeric_cols:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

                df = df.dropna(subset=numeric_cols)
                if df.empty:
                    logger.warning(f"No valid numeric data after conversion/dropna in {ohlcv_file}")
                    continue

                df = df.sort_index()

                if len(df) < MIN_LOOKBACK_PERIOD:
                    logger.warning(
                        f"Insufficient data for {pair} ({len(df)} < {MIN_LOOKBACK_PERIOD}) in {ohlcv_file}. Skipping pair."
                    )
                    missing_pairs.append(pair)
                    continue

                loaded_data[pair] = df[numeric_cols]
                logger.info(f"Loaded {len(df)} valid OHLCV rows for {pair} from {ohlcv_file.name}")

            except FileNotFoundError:
                logger.error(f"CRITICAL: OHLCV file vanished during load: {ohlcv_file}.")
                missing_pairs.append(pair)
            except json.JSONDecodeError as e:
                logger.error(f"Failed decoding JSON for {pair} from {ohlcv_file}: {e}")
                missing_pairs.append(pair)
            except (ValueError, TypeError, KeyError, AttributeError) as e:
                logger.error(
                    f"Error processing OHLCV data for {pair} from {ohlcv_file}: {e}",
                    exc_info=True,
                )
                missing_pairs.append(pair)
            except Exception as e:
                logger.error(
                    f"Unexpected error loading OHLCV for {pair} from {ohlcv_file}: {e}",
                    exc_info=True,
                )
                missing_pairs.append(pair)

        # Update internal state ONLY with successfully loaded pairs matching quote currency
        self.historical_data = loaded_data
        # Filter the original self.pairs list based on successful loading
        original_requested_pairs = set(self.pairs)
        successfully_loaded_pairs = set(loaded_data.keys())
        self.pairs = [
            p for p in self.pairs if p in successfully_loaded_pairs
        ]  # Keep original order

        if not self.pairs:
            logger.critical(
                f"Failed backtest initialization: No valid historical OHLCV data loaded for requested pairs matching quote currency '{run_quote_currency}'. Checked: {list(original_requested_pairs)}. Review logs above for specific reasons why pairs were skipped."
            )
            raise ValueError(
                "No valid historical OHLCV data could be loaded for any specified pair matching the run's quote currency."
            )
        else:
            logger.info(
                f"Successfully loaded historical data for pairs: {self.pairs}"
            )  # Log the final list of pairs used

    def _load_external_signal_data(self):
        """Loads historical sentiment and AI signal data."""
        assets_to_load = set()
        # Load signals only for assets involved in the successfully loaded pairs
        for pair in self.pairs:  # Use self.pairs which now only contains loaded pairs
            if isinstance(pair, str) and "/" in pair:
                try:
                    assets_to_load.add(pair.split("/")[0].upper())
                except IndexError:
                    pass
        if not assets_to_load:
            return

        logger.info(f"Loading external signal data for assets: {assets_to_load}")
        self.loaded_sentiment_data = {asset: [] for asset in assets_to_load}
        self.loaded_ai_signals_data = {asset: [] for asset in assets_to_load}

        sources = [
            (self.sentiment_data_dir, "reddit_sentiment_*", self.loaded_sentiment_data),
            (self.sentiment_data_dir, "news_sentiment_*", self.loaded_sentiment_data),
            (self.ai_signals_data_dir, "ai_response_*", self.loaded_ai_signals_data),
        ]

        for data_dir, pattern, target_dict in sources:
            if not data_dir.exists():
                logger.warning(f"Signal data directory not found: {data_dir}")
                continue
            for asset in assets_to_load:
                self._load_single_asset_signal_files(asset, data_dir, pattern, target_dict)

        for asset in assets_to_load:
            if asset in self.loaded_sentiment_data:
                self.loaded_sentiment_data[asset].sort(key=lambda x: x[0])
                logger.info(
                    f"Loaded {len(self.loaded_sentiment_data[asset])} sentiment entries for {asset}"
                )
            if asset in self.loaded_ai_signals_data:
                self.loaded_ai_signals_data[asset].sort(key=lambda x: x[0])
                logger.info(
                    f"Loaded {len(self.loaded_ai_signals_data[asset])} AI signal entries for {asset}"
                )

    def _load_single_asset_signal_files(
        self, asset: str, data_dir: Path, pattern: str, target_dict: Dict
    ):
        """Loads signal data for a specific asset from files matching a pattern."""
        files_found = list(data_dir.glob(pattern))
        logger.debug(
            f"Checking {len(files_found)} files in {data_dir} matching '{pattern}' for asset '{asset}'"
        )
        loaded_count = 0
        for file_path in files_found:
            try:
                with file_path.open("r", encoding="utf-8") as f:
                    content = f.read()
                    if not content.strip():
                        continue
                    data_list = json.loads(content)
                if isinstance(data_list, dict):
                    data_list = [data_list]
                if not isinstance(data_list, list):
                    continue

                for item in data_list:
                    if not isinstance(item, dict):
                        continue
                    ts_str = item.get("timestamp")
                    item_data = item.get("data", item)
                    if not isinstance(item_data, dict):
                        continue

                    data_asset = item_data.get("symbol", item_data.get("asset", asset))
                    if not isinstance(data_asset, str) or data_asset.upper() != asset:
                        continue

                    if ts_str and isinstance(ts_str, str):
                        try:
                            timestamp = datetime.fromisoformat(
                                ts_str.replace("Z", "+00:00")
                            ).astimezone(timezone.utc)
                            target_dict[asset].append((timestamp, item_data))
                            loaded_count += 1
                        except ValueError:
                            logger.warning(
                                f"Invalid timestamp format '{ts_str}' in {file_path}. Skipping item."
                            )

            except FileNotFoundError:
                logger.error(f"Signal file vanished during load: {file_path}")
            except json.JSONDecodeError:
                logger.error(f"Failed decoding JSON from signal file: {file_path}")
            except OSError as e:
                logger.error(f"OS error reading signal file {file_path}: {e}")
            except Exception as e:
                logger.error(
                    f"Unexpected error processing signal file {file_path}: {e}",
                    exc_info=True,
                )

    def _align_data_steps(self) -> int:
        """Aligns historical data to have the same number of steps based on the shortest DataFrame."""
        if not self.historical_data:
            return 0
        try:
            min_steps = min(
                len(df) for df in self.historical_data.values() if isinstance(df, pd.DataFrame)
            )
        except ValueError:
            min_steps = 0

        if min_steps == 0:
            logger.error("Could not align data: Found 0 common steps across pairs.")
            return 0
        logger.info(f"Aligning OHLCV data to {min_steps} common steps.")
        aligned_data = {}
        for pair, df in self.historical_data.items():
            if isinstance(df, pd.DataFrame) and len(df) >= min_steps:
                aligned_data[pair] = df.iloc[-min_steps:]
            else:
                logger.warning(
                    f"Excluding pair {pair} from alignment due to insufficient length or invalid type."
                )
        self.historical_data = aligned_data
        original_pairs = self.pairs[:]
        self.pairs = list(
            self.historical_data.keys()
        )  # Update pairs to ONLY those successfully aligned
        removed_pairs = set(original_pairs) - set(self.pairs)
        if removed_pairs:
            logger.warning(f"Removed pairs due to alignment issues: {removed_pairs}")
        if not self.pairs:
            logger.error("No pairs remaining after data alignment.")
            return 0
        logger.info(f"Pairs after alignment: {self.pairs}")
        return min_steps

    def _initialize_simulation_env(self):
        """Initialize the simulated exchange, bot logic, decoupled PM, and metrics calculator."""
        logger.info("Initializing simulation environment...")
        if not self.historical_data:
            raise RuntimeError(
                "Cannot initialize simulation: Historical data not loaded or aligned."
            )

        try:
            self.results_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise RuntimeError(f"Failed to create results directory {self.results_dir}: {e}") from e

        # 1. Simulated Exchange
        try:
            # Pass the necessary parameters, including the now FINALIZED list of pairs
            self.sim_exchange = SimulatedExchange(
                config=self.config,
                pairs=self.pairs,  # Use the final list of pairs after loading/alignment
                timeframe=self.timeframe,
                data_dir=DATA_DIR,
                historical_ohlcv_source_dir=self.historical_ohlcv_dir,
                historical_data=self.historical_data,  # Pass the loaded and aligned data
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize SimulatedExchange: {e}") from e

        # 2. Load External Signals (after exchange init and pair finalization)
        self._load_external_signal_data()  # Uses self.pairs which is now finalized

        # 3. Portfolio Manager (Decoupled)
        try:
            self.portfolio_manager_logic = PortfolioManager(
                config=self.config, exchange_client=self.sim_exchange
            )
            logger.info("PortfolioManager initialized for backtest.")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize PortfolioManager: {e}") from e

        # 4. Bot Logic (Decoupled)
        try:
            backtest_config = self.config
            # Force paper trading mode (access config safely)
            paper_trading_path_found = False
            if hasattr(backtest_config, "bot"):
                if hasattr(backtest_config.bot, "paper_trading"):
                    backtest_config.bot.paper_trading = True
                    paper_trading_path_found = True
                # Also check settings dict if it exists
                if (
                    not paper_trading_path_found
                    and hasattr(backtest_config, "settings")
                    and isinstance(backtest_config.settings, dict)
                    and "bot" in backtest_config.settings
                    and isinstance(backtest_config.settings["bot"], dict)
                    and "paper_trading" in backtest_config.settings["bot"]
                ):
                    backtest_config.settings["bot"]["paper_trading"] = True
                    paper_trading_path_found = True

            if not paper_trading_path_found:
                logger.warning(
                    "Could not automatically force paper_trading=True in backtest_config. Path not found."
                )

            self.bot_logic = MultiCryptoTradingBot(
                config=backtest_config,
                exchange_client=self.sim_exchange,
                portfolio_manager=self.portfolio_manager_logic,
                sentiment_sources=[],
                ai_analyzer=None,
            )
            self.bot_logic.exchange_available = True
            logger.info("MultiCryptoTradingBot initialized for backtest.")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize MultiCryptoTradingBot: {e}") from e

        # 5. Metrics Calculator
        try:
            if self.sim_exchange is None:
                raise RuntimeError("SimulatedExchange not initialized before Metrics Calculator.")

            initial_capital_dec = self.sim_exchange.balances.get(
                self.sim_exchange.quote_currency, Decimal("0.0")
            )
            if initial_capital_dec <= Decimal("0"):
                logger.warning(
                    f"Initial capital for metrics is zero or negative ({initial_capital_dec}). Metrics might be invalid."
                )
            self.metrics_calculator = BacktestMetrics(
                initial_capital=initial_capital_dec,
                quote_currency=self.sim_exchange.quote_currency,
            )
            logger.info(
                f"BacktestMetrics initialized with capital {initial_capital_dec} {self.sim_exchange.quote_currency}."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize BacktestMetrics: {e}") from e

        logger.info("Simulation environment initialization complete.")

    def _find_closest_signal_data(
        self,
        asset: str,
        timestamp: datetime,
        source_data: Dict[str, List[Tuple[datetime, Dict]]],
    ) -> Optional[Dict]:
        """Finds signal data closest to the target timestamp within a threshold."""
        if asset not in source_data or not source_data[asset]:
            return None
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)  # Ensure tz-aware

        data_list = source_data[asset]  # Assumes this is sorted by timestamp
        timestamps = [ts for ts, _ in data_list]  # Extract timestamps

        insertion_point = bisect.bisect_left(timestamps, timestamp)
        best_match = None
        min_diff = timedelta(seconds=MAX_TIMESTAMP_DIFF_SECONDS + 1)

        # Check insertion point
        if insertion_point < len(data_list):
            ts_candidate, data_candidate = data_list[insertion_point]
            diff = abs(timestamp - ts_candidate)
            if diff <= min_diff:
                min_diff = diff
                best_match = data_candidate

        # Check point before insertion point
        if insertion_point > 0:
            ts_candidate, data_candidate = data_list[insertion_point - 1]
            diff = abs(timestamp - ts_candidate)
            if diff <= min_diff:
                min_diff = diff
                best_match = data_candidate

        if best_match and min_diff.total_seconds() <= MAX_TIMESTAMP_DIFF_SECONDS:
            return best_match
        else:
            return None

    def run(self) -> Optional[Dict]:  # Returns metrics dict or None on failure
        """Execute the backtest simulation loop."""
        try:
            if not self.preloaded_data_used:
                self._load_historical_data()  # Now filters pairs by quote currency internally
            elif not self.historical_data:
                logger.error("Pre-loaded historical data is empty.")
                return None
            else:
                logger.info("Using pre-loaded historical data.")
                # Ensure preloaded pairs match the expected quote currency? Maybe not needed if filtered later.
                self.pairs = list(self.historical_data.keys())

            # Check if any valid pairs remain *after* loading and quote currency filtering
            if not self.pairs:
                logger.error(
                    "No pairs matching the configured quote currency were successfully loaded."
                )
                return None  # Or raise ValueError? Returning None seems safer here.

            num_steps = self._align_data_steps()  # Uses the filtered self.pairs list
            if num_steps < MIN_LOOKBACK_PERIOD:
                logger.error(
                    f"Insufficient common steps ({num_steps}) after alignment. Minimum required: {MIN_LOOKBACK_PERIOD}."
                )
                return None
            if not self.pairs:
                logger.error("No pairs remaining after data alignment.")
                return None

            self._initialize_simulation_env()  # Uses the final self.pairs list

            if not all(
                [
                    self.sim_exchange,
                    self.bot_logic,
                    self.portfolio_manager_logic,
                    self.metrics_calculator,
                ]
            ):
                logger.critical(
                    "Simulation environment components not fully initialized. Aborting backtest."
                )
                return None

        except (
            ValueError,
            RuntimeError,
            FileNotFoundError,
            AttributeError,
        ) as init_err:
            logger.critical(f"Failed backtest initialization: {init_err}", exc_info=True)
            return None
        except Exception as init_e:
            logger.critical(
                f"Unexpected error during backtest initialization: {init_e}",
                exc_info=True,
            )
            return None

        logger.info(f"--- Starting Backtest Run ({self.run_id}) ---")
        logger.info(
            f"Pairs: {self.pairs}, TF: {self.timeframe}, Steps: {num_steps} (Simulating {num_steps - MIN_LOOKBACK_PERIOD} steps)"
        )
        start_time = time.time()

        # --- Simulation Loop ---
        for step in range(MIN_LOOKBACK_PERIOD, num_steps):
            step_start_time = time.time()
            try:
                if not all(
                    [
                        self.sim_exchange,
                        self.bot_logic,
                        self.portfolio_manager_logic,
                        self.metrics_calculator,
                    ]
                ):
                    logger.critical(
                        f"Simulation component became invalid at step {step}. Aborting."
                    )
                    break

                self.sim_exchange.set_step(step)
                # Use first available pair's timestamp (pairs list is guaranteed non-empty here)
                current_sim_time = self.sim_exchange.get_current_timestamp(self.pairs[0])
                if current_sim_time is None:
                    logger.warning(f"Skipping step {step}: Invalid simulation timestamp.")
                    continue
                logger.debug(
                    f"\n--- Step {step}/{num_steps-1} ({current_sim_time.strftime('%Y-%m-%d %H:%M:%S %Z')}) ---"
                )

                # --- 1. Update Bot's Market Data (OHLCV, Ticker, Injected Signals) ---
                for pair in self.pairs:  # Iterate through the final list of pairs
                    try:
                        asset, quote = pair.split("/")
                        asset = asset.upper()
                        if pair not in self.bot_logic.market_data:
                            self.bot_logic.market_data[pair] = (
                                self.bot_logic._default_market_data_structure()
                            )

                        ohlcv = self.sim_exchange.get_ohlcv(pair, self.timeframe, limit=200)
                        ticker = self.sim_exchange.get_ticker(pair)
                        if ohlcv is None or ticker is None:
                            logger.warning(
                                f"Skipping signal generation for {pair} at step {step}: Missing OHLCV or Ticker."
                            )
                            continue
                        self.bot_logic.market_data[pair]["ohlcv"] = ohlcv
                        self.bot_logic.market_data[pair]["ticker"] = ticker

                        sentiment_data = self._find_closest_signal_data(
                            asset, current_sim_time, self.loaded_sentiment_data
                        )
                        ai_data = self._find_closest_signal_data(
                            asset, current_sim_time, self.loaded_ai_signals_data
                        )

                        self.bot_logic.market_data[pair]["sentiment"] = (
                            sentiment_data if sentiment_data else {}
                        )
                        self.bot_logic.market_data[pair]["ai_analysis"] = ai_data if ai_data else {}

                        current_ts_iso = current_sim_time.isoformat()
                        if not isinstance(
                            self.bot_logic.market_data[pair].get("last_update"), dict
                        ):
                            self.bot_logic.market_data[pair]["last_update"] = {}
                        self.bot_logic.market_data[pair]["last_update"].update(
                            {
                                "ohlcv": current_ts_iso,
                                "ticker": current_ts_iso,
                                "sentiment": current_ts_iso if sentiment_data else None,
                                "ai": current_ts_iso if ai_data else None,
                            }
                        )

                        self.bot_logic.calculate_indicators(pair)
                        self.bot_logic.generate_signals(pair)

                    except Exception as pair_update_e:
                        logger.error(
                            f"Error processing market data/signals for {pair} at step {step}: {pair_update_e}",
                            exc_info=True,
                        )

                # --- 2. Update & Record Portfolio Value ---
                self.bot_logic._update_portfolio_value()
                current_portfolio_value = self.bot_logic.total_portfolio_value
                if self.metrics_calculator and current_sim_time is not None:
                    self.metrics_calculator.record_portfolio_value(
                        current_sim_time, current_portfolio_value
                    )

                # --- 3. Execute Bot's Trading Decisions ---
                sorted_pairs = sorted(
                    self.pairs,
                    key=lambda p: self.bot_logic.opportunity_scores.get(p, 0.0),
                    reverse=True,
                )
                for pair in sorted_pairs:
                    try:
                        self.bot_logic.execute_trading_decision(pair)
                    except Exception as exec_e:
                        logger.error(
                            f"Error executing bot decision for {pair} at step {step}: {exec_e}",
                            exc_info=True,
                        )

                # --- 4. Record Bot Trades for Metrics ---
                if self.metrics_calculator:
                    current_step_bot_trades = [
                        t
                        for t in self.sim_exchange.trades
                        if isinstance(t, dict)
                        and t.get("info", {}).get("step") == step
                        and not t.get("info", {}).get("rebalance")
                    ]
                    for trade in current_step_bot_trades:
                        self.metrics_calculator.record_trade(trade)

                # --- 5. Portfolio Rebalancing Check ---
                if self.portfolio_manager_logic:
                    try:
                        current_market_state = {}
                        for pair in self.pairs:
                            if pair in self.bot_logic.market_data:
                                pair_data = self.bot_logic.market_data[pair]
                                current_market_state[pair] = {
                                    k: v
                                    for k, v in pair_data.items()
                                    if k
                                    in [
                                        "indicators",
                                        "sentiment_metrics",
                                        "ai_metrics",
                                        "ohlcv",
                                        "ticker",
                                    ]
                                    and v is not None
                                }
                        rebalance_actions = (
                            self.portfolio_manager_logic.determine_rebalance_actions(
                                current_market_state
                            )
                        )

                        # --- 6. Record Rebalance Trades ---
                        if (
                            rebalance_actions
                            and isinstance(rebalance_actions, list)
                            and self.metrics_calculator
                        ):
                            logger.info(
                                f"Step {step}: Simulated {len(rebalance_actions)} rebalance actions by PM."
                            )
                            for trade in rebalance_actions:
                                if isinstance(trade, dict) and trade.get("status") not in [
                                    "rejected",
                                    "failed",
                                    "canceled",
                                    "error",
                                ]:
                                    trade["info"] = trade.get("info", {})
                                    trade["info"]["step"] = step
                                    trade["info"]["rebalance"] = True
                                    trade["datetime"] = current_sim_time.isoformat()
                                    trade["timestamp"] = int(current_sim_time.timestamp() * 1000)
                                    self.metrics_calculator.record_trade(trade)
                                else:
                                    reason = "N/A"
                                    if isinstance(trade, dict) and isinstance(
                                        trade.get("info"), dict
                                    ):
                                        reason = trade["info"].get(
                                            "reason", trade["info"].get("error", "N/A")
                                        )
                                    logger.warning(
                                        f"Skipping recording failed/rejected rebalance action: {trade.get('symbol')} - {reason}"
                                    )
                    except Exception as pm_e:
                        logger.error(
                            f"Error during simulated portfolio rebalance check at step {step}: {pm_e}",
                            exc_info=True,
                        )

            except Exception as step_e:
                logger.error(
                    f"CRITICAL ERROR in simulation loop at step {step}: {step_e}",
                    exc_info=True,
                )
                logger.critical("Aborting backtest loop due to critical step error.")
                break

            step_duration = time.time() - step_start_time
            logger.debug(f"--- Step {step} took {step_duration:.3f}s ---")
        # --- End Loop ---

        end_time = time.time()
        total_duration = end_time - start_time
        logger.info(f"--- Backtest Run ({self.run_id}) Finished ---")
        logger.info(f"Total Duration: {total_duration:.2f}s ({total_duration / 60:.2f} min)")

        # --- Final Metrics Calculation & Saving ---
        if self.metrics_calculator:
            try:
                self.run_metrics = self.metrics_calculator.calculate_metrics()
                if self.run_metrics:
                    perf = self.run_metrics.get("performance", {})
                    stats = self.run_metrics.get("trade_stats", {})
                    logger.info("--- Backtest Results Summary ---")
                    logger.info(
                        f"  Initial Capital: {self.run_metrics.get('run_details', {}).get('initial_capital', 0.0):.2f} {self.metrics_calculator.quote_currency}"
                    )
                    logger.info(
                        f"  Final Capital:    {perf.get('final_capital', 0.0):.2f} {self.metrics_calculator.quote_currency}"
                    )
                    logger.info(
                        f"  Total PnL:        {perf.get('total_pnl', 0.0):+.2f} ({perf.get('total_pnl_pct', 0.0):+.2f}%)"
                    )
                    logger.info(f"  Max Drawdown:     {perf.get('max_drawdown_pct', 0.0):.2f}%")
                    logger.info(f"  Sharpe Ratio:     {perf.get('sharpe_ratio', 'N/A')}")
                    logger.info(f"  Round Trips:      {stats.get('total_round_trips', 0)}")
                    logger.info(f"  Win Rate:         {stats.get('win_rate_pct', 0.0):.2f}%")
                    logger.info(f"  Profit Factor:    {stats.get('profit_factor', 'N/A')}")
                    logger.info("---------------------------------")

                    self.metrics_calculator.save_results(self.results_dir, self.run_id)
                    logger.info(f"Full backtest results saved to: {self.results_dir}")
                else:
                    logger.error("Metrics calculation failed.")
            except Exception as metric_e:
                logger.error(
                    f"Error calculating or saving final metrics: {metric_e}",
                    exc_info=True,
                )
        else:
            logger.error("Metrics calculator not available. Cannot calculate or save results.")

        return self.run_metrics


# --- Main Execution ---
def run_backtest_from_cli():
    """Parses CLI arguments and runs the backtester."""
    parser = argparse.ArgumentParser(description="Enhanced Trading Bot Backtester")
    parser.add_argument(
        "--pairs", required=True, help="Comma-separated pairs (e.g., BTC/USD,ETH/USD)"
    )
    parser.add_argument("--timeframe", required=True, help="Timeframe (e.g., 1h, 4h, 1d)")
    parser.add_argument("--config", default=None, help="Path to user_config.json")
    parser.add_argument(
        "--historical-data-dir",
        default=str(DATA_DIR / "backtest_data" / "ohlcv"),
        help=f'Directory containing historical OHLCV JSON files (default: {DATA_DIR / "backtest_data" / "ohlcv"})',
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level for backtester",
    )
    args = parser.parse_args()

    try:
        log_level_upper = args.log_level.upper()
        logger.setLevel(log_level_upper)
        if hasattr(backtest_logger_instance, "logger"):
            for handler in backtest_logger_instance.logger.handlers:
                handler.setLevel(log_level_upper)
        logger.info(f"Backtester log level set to {log_level_upper}")
    except Exception as e:
        logger.error(f"Failed to set log level: {e}")

    try:
        config = BotConfig(config_path=args.config)
    except Exception as e:
        logger.critical(f"FATAL: Failed BotConfig initialization: {e}", exc_info=True)
        sys.exit(1)

    pairs_list = [p.strip().upper().replace("_", "/") for p in args.pairs.split(",") if p.strip()]
    if not pairs_list:
        logger.critical("No valid pairs provided.")
        sys.exit(1)
    historical_data_directory = Path(args.historical_data_dir)
    if not historical_data_directory.is_dir():
        logger.critical(
            f"Historical OHLCV data directory not found or invalid: {historical_data_directory}"
        )
        sys.exit(1)

    try:
        runner = BacktestRunner(
            config=config,
            pairs=pairs_list,  # Pass the initial list from args here
            timeframe=args.timeframe,
            data_dir=DATA_DIR,
            historical_ohlcv_source_dir=historical_data_directory,
        )
        runner.run()
    except Exception as e:
        logger.critical(f"Backtest execution failed with unhandled exception: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    import bisect
    import hashlib
    import json
    import sys
    from decimal import Decimal, DivisionByZero, InvalidOperation
    from pathlib import Path

    import pandas as pd

    run_backtest_from_cli()
