# enhanced_trading_bot/scripts/fetch_historical_data.py
"""
Script to fetch historical OHLCV data from Kraken (using the project's
KrakenClient) and save it in the format required by the backtester
(list of dictionaries: [{'timestamp_ms': ..., 'open': ..., ...}])
to a dedicated backtesting data directory.
"""

import json
import logging  # Import standard logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# --- Add project root to sys.path ---
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
# ------------------------------------

try:
    import ccxt

    # Import DATA_DIR (CACHE_DIR not needed here anymore for output)
    from common.common_logger import DATA_DIR
    from core.config import BotConfig
    from integrations.exchange.kraken import KrakenClient
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("Please ensure you are running this script from the project root directory")
    print("and that your virtual environment is active with all requirements installed.")
    sys.exit(1)

# --- Setup Basic Logging for the Script ---
log_dir = DATA_DIR / "logs" / "scripts"
log_dir.mkdir(parents=True, exist_ok=True)
script_log_file = log_dir / "fetch_data.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(script_log_file)],
)
logger = logging.getLogger("fetch_historical_data")
# ------------------------------------------

# --- Configuration ---
CONFIG_PATH = project_root / "config" / "user_config.json"
# *** Adjust these as needed ***
PAIRS_TO_FETCH = ["BTC/USD", "ETH/USD", "XRP/USD", "LTC/USD", "DOT/USD", "DOGE/USD"]
TIMEFRAME = "1h"  # e.g., '1h', '4h', '1d'
DAYS_HISTORY = 730  # How many days of history to fetch (e.g., 365*2 = ~2 years)
# *** End Adjustments ***

END_DATE = datetime.now(timezone.utc) - timedelta(days=1)
START_DATE = END_DATE - timedelta(days=DAYS_HISTORY)
API_LIMIT = 720
SLEEP_INTERVAL = 3
# --- Changed Output Directory ---
OUTPUT_DIR = DATA_DIR / "backtest_data" / "ohlcv"  # Save to dedicated backtest data dir
# --------------------------------


def save_data(pair_data: list, filepath: Path):
    """Saves data to JSON file."""
    try:
        # Create parent directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)  # Ensures OUTPUT_DIR gets created
        with filepath.open("w", encoding="utf-8") as f:
            json.dump(pair_data, f, indent=2)  # Add indent for readability
        logger.info(f"Successfully saved {len(pair_data)} records to {filepath.name}")
    except (IOError, TypeError) as e:
        logger.error(f"Error saving data to {filepath.name}: {e}")


def main():
    """Main function to fetch and save data."""
    if not CONFIG_PATH.exists():
        logger.error(f"Configuration file not found at {CONFIG_PATH}")
        sys.exit(1)

    logger.info("Loading configuration...")
    try:
        config = BotConfig(config_path=CONFIG_PATH)
        kraken_client = KrakenClient(config=config)
        if not hasattr(kraken_client, "exchange") or not kraken_client.exchange:
            logger.error("Failed to initialize Kraken client exchange instance.")
            sys.exit(1)
        logger.info(f"Kraken client initialized for exchange: {kraken_client.exchange.id}")
    except Exception as e:
        logger.error(f"Failed to initialize BotConfig or KrakenClient: {e}", exc_info=True)
        sys.exit(1)

    start_ms = int(START_DATE.timestamp() * 1000)
    end_ms = int(END_DATE.timestamp() * 1000)
    try:
        timeframe_in_ms = kraken_client.exchange.parse_timeframe(TIMEFRAME) * 1000
    except Exception as e:
        logger.error(f"Failed to parse timeframe '{TIMEFRAME}': {e}")
        sys.exit(1)

    # Create the main output directory now
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured output directory exists: {OUTPUT_DIR}")
    except OSError as e:
        logger.error(f"Could not create output directory {OUTPUT_DIR}: {e}")
        sys.exit(1)

    logger.info(f"\nFetching {TIMEFRAME} data for {PAIRS_TO_FETCH}")
    logger.info(f"From: {START_DATE.isoformat()}")
    logger.info(f"To:   {END_DATE.isoformat()}")
    logger.info(f"Saving to: {OUTPUT_DIR}")

    for pair in PAIRS_TO_FETCH:
        logger.info(f"\n----- Processing {pair} -----")
        all_pair_data = []
        last_fetched_timestamp = -1  # Track last timestamp to avoid duplicates
        current_start_ms = start_ms
        fetch_attempts = 0
        max_fetch_attempts = 10  # Limit attempts for a single pair run

        while current_start_ms < end_ms and fetch_attempts < max_fetch_attempts:
            fetch_attempts += 1
            try:
                fetch_start_dt = datetime.fromtimestamp(current_start_ms / 1000, tz=timezone.utc)
                logger.info(
                    f"  Attempt {fetch_attempts}: Fetching {API_LIMIT} candles from {fetch_start_dt.isoformat()}..."
                )

                # Assuming get_ohlcv returns a list of dictionaries like:
                # [{'timestamp_ms': ..., 'timestamp': '...', 'open': ..., ...}, ...]
                ohlcv_data_raw = kraken_client.get_ohlcv(
                    pair, TIMEFRAME, since=current_start_ms, limit=API_LIMIT
                )

                if ohlcv_data_raw is None:
                    logger.warning(
                        "  Received None from get_ohlcv, possibly due to API error or no data."
                    )
                    time.sleep(SLEEP_INTERVAL * 2)
                    continue

                if not ohlcv_data_raw:
                    logger.info(
                        "  No more data received for this period or empty list. Stopping fetch for this pair."
                    )
                    break

                # --- FIX: Process into list of dictionaries ---
                processed_ohlcv = []
                batch_last_ts = -1
                for candle_dict in ohlcv_data_raw:
                    if isinstance(candle_dict, dict):
                        try:
                            ts_ms = int(candle_dict["timestamp_ms"])
                            # Skip if timestamp is duplicate of last added
                            if ts_ms <= last_fetched_timestamp:
                                continue

                            o = float(candle_dict["open"])
                            h = float(candle_dict["high"])
                            l = float(candle_dict["low"])
                            c = float(candle_dict["close"])
                            v = float(candle_dict["volume"])

                            # Create the dictionary to be saved
                            processed_candle = {
                                "timestamp_ms": ts_ms,
                                "open": o,
                                "high": h,
                                "low": l,
                                "close": c,
                                "volume": v,
                            }
                            processed_ohlcv.append(processed_candle)
                            batch_last_ts = ts_ms  # Track last timestamp within this batch

                        except (KeyError, ValueError, TypeError) as e:
                            logger.warning(
                                f"  Warning: Skipping malformed candle dict: {candle_dict}, Error: {e}"
                            )
                    else:
                        logger.warning(
                            f"  Warning: Skipping unexpected item format in OHLCV list: {candle_dict}"
                        )
                # --- End FIX ---

                if not processed_ohlcv:
                    logger.warning("  No valid, new candle data processed in this batch.")
                    # Need to advance time even if no data to avoid infinite loop on gaps
                    current_start_ms += timeframe_in_ms * API_LIMIT
                    time.sleep(SLEEP_INTERVAL)  # Sleep even if no data
                    continue

                all_pair_data.extend(processed_ohlcv)
                last_fetched_timestamp = batch_last_ts  # Update last overall timestamp
                last_candle_dt = datetime.fromtimestamp(
                    last_fetched_timestamp / 1000, tz=timezone.utc
                )

                logger.info(
                    f"  Fetched {len(processed_ohlcv)} new valid records. Last timestamp: {last_candle_dt.isoformat()}"
                )
                # Advance start time for next fetch
                current_start_ms = last_fetched_timestamp + timeframe_in_ms

                if last_fetched_timestamp >= end_ms:
                    logger.info("  Reached target end date.")
                    break

                logger.info(f"  Sleeping for {SLEEP_INTERVAL}s...")
                time.sleep(SLEEP_INTERVAL)

            except ccxt.RateLimitExceeded as e:
                logger.warning(f"  Rate limit exceeded, sleeping longer... ({e})")
                time.sleep(60)
            except ccxt.ExchangeNotAvailable as e:
                logger.warning(f"  Exchange not available (temporary?), sleeping... ({e})")
                time.sleep(30)
            except ccxt.NetworkError as e:
                logger.warning(f"  Network error fetching {pair}, retrying after sleep... ({e})")
                time.sleep(15)
            except ccxt.ExchangeError as e:
                logger.warning(f"  Exchange error fetching {pair}: {e}")
                time.sleep(10)
            except Exception as e:
                logger.error(f"  Unexpected error fetching {pair}: {e}", exc_info=True)
                time.sleep(10)

        if fetch_attempts >= max_fetch_attempts:
            logger.warning(
                f"  Warning: Reached max fetch attempts ({max_fetch_attempts}) for {pair}. Data might be incomplete."
            )

        if not all_pair_data:
            logger.warning(f"----- No data fetched for {pair} -----")
            continue

        # --- Filename uses OUTPUT_DIR ---
        pair_filename = (
            f"{pair.replace('/', '_')}_{TIMEFRAME}_cache.json"  # Keep same filename convention
        )
        output_filepath = OUTPUT_DIR / pair_filename
        # ---------------------------------
        # Sort just in case API returns slightly out of order (unlikely but safe)
        all_pair_data.sort(key=lambda x: x["timestamp_ms"])
        logger.info(f"----- Saving data for {pair} -----")
        save_data(all_pair_data, output_filepath)  # Saves list of dictionaries

    logger.info("\nHistorical data fetching process complete.")


if __name__ == "__main__":
    main()
