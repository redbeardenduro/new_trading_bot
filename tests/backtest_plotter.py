# enhanced_trading_bot/tests/backtest_plotter.py
"""
Backtest Performance Visualization Module.

Loads results from a backtest run (specified by run_id) and generates plots:
- Portfolio Equity Curve
- Drawdown Curve
- OHLCV Price Chart with Buy/Sell Trade Markers (for a specific pair)

Saves the combined plot as an image file. Handles potential errors during
data loading and plotting gracefully.
"""

import argparse
import json
import logging
from datetime import datetime  # Import timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle  # For drawing candlestick body
from matplotlib.ticker import FuncFormatter  # For formatting y-axis

# --- Core Imports ---
try:
    from common.common_logger import (CACHE_DIR, DATA_DIR, LOG_DIR,
                                      PROJECT_ROOT, get_logger)
except ImportError:
    import sys

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    try:
        from common.common_logger import (CACHE_DIR, DATA_DIR, LOG_DIR,
                                          PROJECT_ROOT, get_logger)
    except ImportError as inner_e:
        print(f"ERROR: Failed to import common_logger: {inner_e}", file=sys.stderr)
        # Fallback basic logging if logger setup fails
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        logger = logging.getLogger("backtest_plotter_fallback")
        # Define fallback paths if needed, though script might fail later without config/common_logger
        DATA_DIR = project_root / "data"
        CACHE_DIR = DATA_DIR / "cache"
        # ... other paths if necessary
else:
    # Use a logger for the plotter, inherit level from central config if possible
    logger = get_logger("backtest_plotter")
    logger.setLevel(logging.INFO)  # Default log level if central config doesn't override


# --- Plotting Configuration ---
PLOT_STYLE = "seaborn_v0_8_darkgrid"  # Example style
EQUITY_CURVE_COLOR = "cyan"
DRAWDOWN_COLOR = "red"
OHLC_UP_COLOR = "#2ECC71"  # More distinct green
OHLC_DOWN_COLOR = "#E74C3C"  # More distinct red
TRADE_BUY_MARKER = "^"
TRADE_SELL_MARKER = "v"
TRADE_BUY_COLOR = "lime"
TRADE_SELL_COLOR = "red"
PLOT_WIDTH_INCHES = 16  # Slightly wider
PLOT_HEIGHT_PER_CHART = 5

# --- Data Loading Functions ---


def load_json_data(file_path: Path) -> Optional[Any]:
    """Loads data from a JSON file with error handling."""
    if not isinstance(file_path, Path):
        logger.error(f"Invalid file path type provided: {type(file_path)}")
        return None
    if not file_path.exists():
        logger.error(f"Data file not found: {file_path}")
        return None
    if not file_path.is_file():
        logger.error(f"Path is not a file: {file_path}")
        return None
    try:
        with file_path.open("r", encoding="utf-8") as f:
            content = f.read()
            if not content.strip():  # Handle empty file
                logger.warning(f"JSON file is empty: {file_path}")
                return None  # Return None for empty file
            data = json.loads(content)
        logger.info(f"Successfully loaded data from {file_path.name}")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from {file_path}: {e}")
        return None
    except OSError as e:
        logger.error(f"OS error reading file {file_path}: {e}")
        return None
    except Exception as e:  # Catch any other unexpected errors
        logger.error(f"Unexpected error reading file {file_path}: {e}", exc_info=True)
        return None


def load_portfolio_history(results_dir: Path, run_id: str) -> Optional[pd.DataFrame]:
    """Loads portfolio history, converts to DataFrame, validates data."""
    portfolio_file = results_dir / f"backtest_portfolio_{run_id}.json"
    data = load_json_data(portfolio_file)
    if data is None:
        return None
    if not isinstance(data, list) or not data:  # Check if it's a non-empty list
        logger.error(f"Portfolio data in {portfolio_file.name} is not a valid non-empty list.")
        return None

    try:
        df = pd.DataFrame(data)
        # Validate required columns
        if "timestamp" not in df.columns or "value" not in df.columns:
            logger.error("Portfolio data missing required 'timestamp' or 'value' columns.")
            return None

        # Convert timestamp safely
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.dropna(subset=["timestamp"])  # Drop rows with invalid timestamps
        if df.empty:
            logger.error("No valid timestamps found in portfolio data.")
            return None
        df = df.set_index("timestamp")

        # Convert value safely
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])  # Drop rows with invalid values
        if df.empty:
            logger.error("No valid portfolio values found after conversion.")
            return None

        df = df.sort_index()  # Ensure chronological order
        logger.info(f"Processed {len(df)} valid portfolio history points.")
        return df[["value"]]  # Return only the 'value' column
    except KeyError as e:
        logger.error(f"Missing expected key during portfolio data processing: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error processing portfolio data: {e}", exc_info=True)
        return None


def load_trades(results_dir: Path, run_id: str) -> Optional[List[Dict]]:
    """Loads executed trades list and parses timestamps."""
    trades_file = results_dir / f"backtest_trades_{run_id}.json"
    data = load_json_data(trades_file)
    if data is None:
        return None
    if not isinstance(data, list):  # Trades should be a list
        logger.error(f"Trades data in {trades_file.name} is not a list.")
        return None

    processed_trades = []
    for trade in data:
        if not isinstance(trade, dict):
            continue  # Skip non-dict items

        # Convert timestamp safely
        ts_input = trade.get("datetime", trade.get("timestamp"))
        timestamp_dt = None
        if ts_input:
            try:
                if isinstance(ts_input, (int, float)):  # Handle ms or s timestamp
                    unit = "ms" if ts_input > 1e12 else "s"
                    timestamp_dt = pd.to_datetime(ts_input, unit=unit, utc=True)
                elif isinstance(ts_input, str):
                    timestamp_dt = pd.to_datetime(ts_input, errors="coerce", utc=True)
                # Add handling for datetime objects if they might appear
                elif isinstance(ts_input, datetime):
                    timestamp_dt = pd.to_datetime(ts_input, utc=True)

                if pd.isna(timestamp_dt):  # Check if conversion failed
                    timestamp_dt = None
                    logger.warning(
                        f"Could not parse timestamp '{ts_input}' for trade {trade.get('id')}"
                    )
            except Exception as e:
                logger.warning(
                    f"Error parsing timestamp '{ts_input}' for trade {trade.get('id')}: {e}"
                )
                timestamp_dt = None
        trade["timestamp_dt"] = timestamp_dt  # Add parsed timestamp or None
        # Validate other essential fields (optional, but good practice)
        if not all(k in trade for k in ["symbol", "side", "average", "amount"]):
            logger.warning(f"Skipping trade with missing essential fields: {trade.get('id')}")
            continue
        processed_trades.append(trade)

    logger.info(f"Processed {len(processed_trades)} trades.")
    return processed_trades


def load_ohlcv_data(pair: str, timeframe: str, ohlcv_data_dir: Path) -> Optional[pd.DataFrame]:
    """Loads historical OHLCV data used in the backtest with validation."""
    if not isinstance(pair, str) or not pair:
        logger.error("Invalid pair provided.")
        return None
    if not isinstance(timeframe, str) or not timeframe:
        logger.error("Invalid timeframe provided.")
        return None
    if not isinstance(ohlcv_data_dir, Path):
        logger.error("Invalid OHLCV data directory provided.")
        return None

    pair_filename = f"{pair.replace('/', '_')}_{timeframe}_cache.json"
    cache_file = ohlcv_data_dir / pair_filename
    data = load_json_data(cache_file)  # Handles file not found, JSON errors etc.
    if data is None:
        return None

    try:
        # Determine structure and get list of candles
        if isinstance(data, dict) and "ohlcv" in data:
            ohlcv_list = data["ohlcv"]
        elif isinstance(data, list):
            ohlcv_list = data
        else:
            logger.error(f"Unexpected data format in {cache_file.name}")
            return None
        if not isinstance(ohlcv_list, list) or not ohlcv_list:
            logger.warning(f"Empty OHLCV list in {cache_file.name}")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(ohlcv_list)
        # Validate required columns exist
        required_cols = ["open", "high", "low", "close", "volume"]
        ts_col = None
        if "timestamp_ms" in df.columns:
            ts_col = "timestamp_ms"
            unit = "ms"
        elif "timestamp" in df.columns:
            ts_col = "timestamp"
            unit = None  # Infer unit later if needed
        else:
            logger.error("Missing timestamp column ('timestamp' or 'timestamp_ms').")
            return None
        if not all(col in df.columns for col in required_cols):
            logger.error(f"Missing one or more required OHLCV columns ({required_cols}).")
            return None

        # Process Timestamp column robustly
        df["timestamp"] = pd.to_datetime(df[ts_col], errors="coerce", unit=unit, utc=True)
        df = df.dropna(subset=["timestamp"])
        if df.empty:
            logger.error(f"No valid timestamps found in OHLCV data for {pair}.")
            return None
        df = df.set_index("timestamp")

        # Process numeric columns robustly
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=required_cols)  # Drop rows with invalid numeric data
        if df.empty:
            logger.error(f"No valid numeric OHLCV data found for {pair}.")
            return None

        df = df.sort_index()  # Ensure chronological order
        logger.info(f"Processed {len(df)} valid OHLCV data points for {pair}.")
        return df[required_cols]  # Return only the required columns

    except KeyError as e:
        logger.error(f"Missing expected key during OHLCV processing for {pair}: {e}")
        return None
    except Exception as e:
        logger.error(
            f"Failed to load or process OHLCV data for {pair} from {cache_file.name}: {e}",
            exc_info=True,
        )
        return None


# --- Plotting Functions ---


def plot_equity_curve(ax, portfolio_df: pd.DataFrame):
    """Plots the portfolio equity curve."""
    if portfolio_df is None or portfolio_df.empty or "value" not in portfolio_df.columns:
        logger.warning("Cannot plot equity curve: Portfolio data is missing or invalid.")
        ax.set_title("Equity Curve (No Data)")
        return
    try:
        ax.plot(
            portfolio_df.index,
            portfolio_df["value"],
            label="Portfolio Value",
            color=EQUITY_CURVE_COLOR,
            linewidth=1.5,
        )
        ax.set_ylabel("Portfolio Value")
        ax.set_title("Equity Curve")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:,.0f}"))  # Format y-axis
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)
    except Exception as e:
        logger.error(f"Error plotting equity curve: {e}", exc_info=True)
        ax.set_title("Equity Curve (Plotting Error)")


def plot_drawdown(ax, portfolio_df: pd.DataFrame):
    """Calculates and plots the drawdown curve."""
    if (
        portfolio_df is None
        or portfolio_df.empty
        or "value" not in portfolio_df.columns
        or len(portfolio_df) < 2
    ):
        logger.warning("Cannot plot drawdown: Insufficient or invalid portfolio data.")
        ax.set_title("Drawdown (No Data)")
        return

    try:
        values = portfolio_df["value"].astype(float)  # Ensure float
        # Calculate cumulative max, ensuring it doesn't dip below a small positive number
        cumulative_max = np.maximum.accumulate(np.maximum(values, 1e-9))
        # Calculate drawdown percentage: (current - peak) / peak * 100
        # Ensure peak value is positive before division
        drawdowns = np.zeros_like(values)
        valid_peak_mask = cumulative_max > 1e-9
        drawdowns[valid_peak_mask] = (
            (values[valid_peak_mask] - cumulative_max[valid_peak_mask])
            / cumulative_max[valid_peak_mask]
        ) * 100
        drawdowns[~valid_peak_mask] = 0  # Set drawdown to 0 where peak was zero/negative

        max_drawdown = np.min(drawdowns)  # Max drawdown is the most negative value

        ax.fill_between(
            portfolio_df.index,
            drawdowns,
            0,
            color=DRAWDOWN_COLOR,
            alpha=0.3,
            label="Drawdown",
        )
        ax.set_ylabel("Drawdown (%)")
        ax.set_title(f"Drawdown (Max: {max_drawdown:.2f}%)")  # Format as percentage
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda x, p: f"{x:.1f}%")
        )  # Format y-axis as percent
        ax.set_ylim(bottom=min(max_drawdown * 1.1, -1), top=5)  # Adjust y-limits slightly

    except Exception as e:
        logger.error(f"Error calculating or plotting drawdown: {e}", exc_info=True)
        ax.set_title("Drawdown (Calculation/Plotting Error)")


def plot_ohlc_with_trades(ax, ohlcv_df: pd.DataFrame, trades: List[Dict], pair: str):
    """Plots OHLC candles and trade markers, handling potential errors."""
    ax.set_title(f"{pair} Price and Trades")
    ax.set_ylabel("Price")

    if ohlcv_df is None or ohlcv_df.empty:
        logger.warning(f"No OHLCV data available for {pair}. Cannot plot price chart.")
        ax.text(
            0.5,
            0.5,
            "OHLCV Data Missing",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )
        return

    # Filter trades for the current pair and ensure they have valid timestamps and prices
    pair_trades = [
        t
        for t in trades
        if isinstance(t, dict)
        and t.get("symbol") == pair
        and isinstance(
            t.get("timestamp_dt"), pd.Timestamp
        )  # Check if timestamp was parsed correctly
        and isinstance(t.get("average"), (int, float))  # Check if average price is valid
    ]

    # Filter OHLCV data to match trade period if trades exist
    plot_ohlcv_df = ohlcv_df  # Default to all data
    if pair_trades:
        try:
            min_trade_time = min(t["timestamp_dt"] for t in pair_trades)
            max_trade_time = max(t["timestamp_dt"] for t in pair_trades)
            # Add padding
            start_time = min_trade_time - pd.Timedelta(
                days=max(1, (max_trade_time - min_trade_time).days * 0.1)
            )
            end_time = max_trade_time + pd.Timedelta(
                days=max(1, (max_trade_time - min_trade_time).days * 0.1)
            )
            mask = (ohlcv_df.index >= start_time) & (ohlcv_df.index <= end_time)
            if mask.any():  # Only slice if there's data in the range
                plot_ohlcv_df = ohlcv_df[mask]
            if plot_ohlcv_df.empty:
                logger.warning(
                    f"No OHLCV data aligns with trade times for {pair}. Plotting all available OHLCV."
                )
                plot_ohlcv_df = ohlcv_df
        except Exception as e:
            logger.error(
                f"Error aligning OHLCV data with trades for {pair}: {e}. Plotting all OHLCV."
            )
            plot_ohlcv_df = ohlcv_df

    if plot_ohlcv_df.empty:
        logger.warning(f"No OHLCV data to plot for {pair} after filtering.")
        ax.text(
            0.5,
            0.5,
            "No OHLCV Data to Display",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )
        return

    # --- Candlestick plotting ---
    try:
        # Determine candle width based on median time difference
        width = 0.0008  # Default width
        if len(plot_ohlcv_df.index) > 1:
            time_diffs = np.diff(
                mdates.date2num(plot_ohlcv_df.index)
            )  # Calculate time differences in days
            if len(time_diffs) > 0:
                median_diff = np.median(time_diffs)
                width = median_diff * 0.7  # Use 70% of the median interval
                logger.debug(f"Calculated candle width: {width} (days)")

        for index, row in plot_ohlcv_df.iterrows():
            dt_num = mdates.date2num(index)
            # Safely access and convert OHLC values
            o = float(row["open"])
            h = float(row["high"])
            l = float(row["low"])
            c = float(row["close"])
            color = OHLC_UP_COLOR if c >= o else OHLC_DOWN_COLOR
            # Draw the wick (high-low line)
            ax.plot([dt_num, dt_num], [l, h], color=color, linewidth=0.8, zorder=1)
            # Draw the body (rectangle)
            body_height = abs(o - c)
            body_bottom = min(o, c)
            # Ensure width is positive for Rectangle
            rect = Rectangle(
                (dt_num - abs(width) / 2, body_bottom),
                abs(width),
                body_height,
                facecolor=color,
                edgecolor=color,
                zorder=2,
            )
            ax.add_patch(rect)
    except Exception as e:
        logger.error(f"Error plotting OHLCV candles for {pair}: {e}", exc_info=True)
        ax.text(
            0.5,
            0.5,
            "Error Plotting Candles",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )
        # Continue to plot trades if possible

    # --- Plot trade markers ---
    try:
        buy_times = [t["timestamp_dt"] for t in pair_trades if t["side"] == "buy"]
        # Use average execution price, ensure it's float
        buy_prices = [float(t["average"]) for t in pair_trades if t["side"] == "buy"]
        sell_times = [t["timestamp_dt"] for t in pair_trades if t["side"] == "sell"]
        sell_prices = [float(t["average"]) for t in pair_trades if t["side"] == "sell"]

        if buy_times:
            ax.scatter(
                buy_times,
                buy_prices,
                label="Buy",
                marker=TRADE_BUY_MARKER,
                color=TRADE_BUY_COLOR,
                s=100,
                zorder=3,
                alpha=0.9,
                edgecolors="black",
            )
        if sell_times:
            ax.scatter(
                sell_times,
                sell_prices,
                label="Sell",
                marker=TRADE_SELL_MARKER,
                color=TRADE_SELL_COLOR,
                s=100,
                zorder=3,
                alpha=0.9,
                edgecolors="black",
            )
    except (ValueError, TypeError) as e:
        logger.error(f"Error preparing trade marker data for {pair}: {e}")
    except Exception as e:  # Catch other plotting errors
        logger.error(f"Error plotting trade markers for {pair}: {e}", exc_info=True)

    # --- Formatting ---
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    # Improve date formatting based on time range
    time_range_days = (
        (plot_ohlcv_df.index.max() - plot_ohlcv_df.index.min()).days
        if not plot_ohlcv_df.empty
        else 0
    )
    if time_range_days > 365 * 2:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    elif time_range_days > 90:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, time_range_days // 180)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    elif time_range_days > 7:
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=max(1, time_range_days // 30)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    else:
        ax.xaxis.set_major_locator(
            mdates.HourLocator(interval=max(1, int(time_range_days * 24 / 12)))
        )
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))

    ax.figure.autofmt_xdate(rotation=45)  # Auto-rotate date labels
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:,.2f}"))  # Format y-axis price


# --- Main Function ---


def create_backtest_plots(run_id: str, pair_to_plot: str, timeframe: str):
    """Loads data and generates the combined backtest plot."""
    logger.info(
        f"Generating backtest plots for Run ID: {run_id}, Pair: {pair_to_plot}, Timeframe: {timeframe}"
    )

    # Validate inputs
    if not run_id or not isinstance(run_id, str):
        logger.critical("Invalid Run ID provided.")
        return
    if not pair_to_plot or not isinstance(pair_to_plot, str):
        logger.critical("Invalid Pair provided.")
        return
    if not timeframe or not isinstance(timeframe, str):
        logger.critical("Invalid Timeframe provided.")
        return

    # Define results directory relative to DATA_DIR
    results_dir = DATA_DIR / "backtest_results" / run_id
    if not results_dir.is_dir():
        logger.critical(f"Results directory not found: {results_dir}")
        return

    # Load Data (functions handle None return)
    portfolio_df = load_portfolio_history(results_dir, run_id)
    trades = load_trades(results_dir, run_id)
    # Assume OHLCV is in CACHE_DIR as per original code
    ohlcv_df = load_ohlcv_data(pair_to_plot, timeframe, CACHE_DIR)

    # Check if essential data loaded
    if portfolio_df is None or trades is None:
        logger.error("Failed to load essential portfolio history or trades. Cannot generate plots.")
        return
    # OHLCV is plotted on one axis, allow proceeding without it but log error
    if ohlcv_df is None:
        logger.error(f"Failed to load OHLCV data for {pair_to_plot}. Price chart will be empty.")
        # Continue to plot Equity/Drawdown

    # --- Plotting Setup ---
    try:
        plt.style.use(PLOT_STYLE)
    except OSError:
        logger.warning(f"Plot style '{PLOT_STYLE}' not found. Using default.")
    except Exception as e:
        logger.error(f"Error setting plot style: {e}")

    # Create Figure and Axes
    fig, axes = plt.subplots(
        3, 1, figsize=(PLOT_WIDTH_INCHES, PLOT_HEIGHT_PER_CHART * 3), sharex=True
    )
    fig.suptitle(
        f"Backtest Results - Run ID: {run_id}", fontsize=16, y=0.99
    )  # Adjust title position

    # --- Plotting Components ---
    # Plot Equity/Drawdown even if OHLCV is missing
    plot_equity_curve(axes[0], portfolio_df)
    plot_drawdown(axes[1], portfolio_df)
    # Plot OHLCV + Trades (handles None ohlcv_df internally)
    plot_ohlc_with_trades(axes[2], ohlcv_df, trades, pair_to_plot)

    # --- Final Touches ---
    axes[2].set_xlabel("Timestamp (UTC)")  # Label bottom axis
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])  # Adjust layout further

    # Save Plot
    output_filename = (
        results_dir / f"backtest_summary_plot_{run_id}_{pair_to_plot.replace('/', '_')}.png"
    )
    try:
        plt.savefig(output_filename, dpi=150, bbox_inches="tight")  # Use bbox_inches for better fit
        logger.info(f"Successfully saved backtest plot to {output_filename}")
    except OSError as e:
        logger.error(f"OS error saving plot to {output_filename}: {e}")
    except Exception as e:  # Catch other saving errors
        logger.error(f"Failed to save plot: {e}", exc_info=True)

    plt.close(fig)  # Close the figure to free memory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Backtest Performance Plots")
    parser.add_argument(
        "--run-id",
        required=True,
        help="The timestamped Run ID (directory name in data/backtest_results/)",
    )
    parser.add_argument("--pair", required=True, help="The trading pair to plot (e.g., BTC/USD)")
    parser.add_argument(
        "--timeframe",
        required=True,
        help="The timeframe used for the backtest data (e.g., 1h)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level",
    )

    args = parser.parse_args()

    # Set log level based on argument
    try:
        log_level_upper = args.log_level.upper()
        logger.setLevel(log_level_upper)
        # Also set level for handlers if using EnhancedLogger setup
        if "backtest_logger_instance" in globals() and hasattr(backtest_logger_instance, "logger"):
            for handler in backtest_logger_instance.logger.handlers:
                handler.setLevel(log_level_upper)
        logger.info(f"Plotter log level set to {log_level_upper}")
    except Exception as e:
        logger.error(f"Failed to set plotter log level: {e}")

    # Sanitize pair format
    pair_clean = args.pair.strip().upper().replace("_", "/")

    create_backtest_plots(run_id=args.run_id, pair_to_plot=pair_clean, timeframe=args.timeframe)
