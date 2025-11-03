"""
Batch Backtesting and Walk-Forward Analysis Tool.

Runs multiple backtests using BacktestRunner with varying parameters
and/or over sliding date windows (walk-forward). Aggregates results into a CSV.
Provides enhanced error handling for data loading, configuration, and execution.
"""

import argparse
import copy
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pandas as pd

# --- Core Imports ---
try:
    from common.common_logger import (CACHE_DIR, DATA_DIR, LOG_DIR,
                                      PROJECT_ROOT, get_logger)
    from core.config import BotConfig
    from tests.backtesting import BacktestRunner
    from utils.enhanced_logging import EnhancedLogger
except ImportError as e:
    print(f"ERROR: Failed to import core/test modules: {e}", file=sys.stderr)
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    try:
        from common.common_logger import (CACHE_DIR, DATA_DIR, LOG_DIR,
                                          PROJECT_ROOT, get_logger)
        from core.config import BotConfig
        from tests.backtesting import BacktestRunner
        from utils.enhanced_logging import EnhancedLogger
    except ImportError as inner_e:
        print(
            f"ERROR: Failed import after adding project root: {inner_e}",
            file=sys.stderr,
        )
        sys.exit(1)

# --- Configure Dedicated Logger ---
try:
    batch_log_dir = DATA_DIR / "logs" / "batch_tests"
    batch_log_dir.mkdir(parents=True, exist_ok=True)
    batch_logger_instance = EnhancedLogger(
        name="batch_tester",
        log_level=logging.INFO,
        log_dir=batch_log_dir,
        log_file_name="batch_test_run.log",
        rotation_interval="D",
        rotation_backup_count=14,
    )
    logger = batch_logger_instance.logger
except Exception as log_setup_e:
    print(
        f"ERROR: Failed to set up enhanced logger for batch tester: {log_setup_e}. Using basic.",
        file=sys.stderr,
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("batch_tester_fallback")


# --- Global Settings ---
DEFAULT_OUTPUT_DIR = DATA_DIR / "batch_results"


# --- Helper Functions ---


def load_full_ohlcv_data(
    pairs: List[str], timeframe: str, ohlcv_data_dir: Path
) -> Dict[str, pd.DataFrame]:
    """Loads the complete OHLCV data for specified pairs with robust error handling."""
    logger.info(
        f"Loading FULL historical OHLCV data for: {pairs} ({timeframe}) from {ohlcv_data_dir}"
    )
    full_data = {}
    if not isinstance(pairs, list) or not pairs:
        logger.error("No pairs provided for loading OHLCV data.")
        return {}
    if not ohlcv_data_dir.is_dir():
        logger.error(f"OHLCV data directory not found or invalid: {ohlcv_data_dir}")
        raise FileNotFoundError(f"OHLCV directory not found: {ohlcv_data_dir}")

    for pair in pairs:
        pair_filename = f"{pair.replace('/', '_')}_{timeframe}_cache.json"
        cache_file = ohlcv_data_dir / pair_filename
        if not cache_file.exists():
            logger.error(
                f"Required OHLCV data file not found: {cache_file}. Cannot proceed with pair {pair}."
            )
            continue

        try:
            with cache_file.open("r", encoding="utf-8") as f:
                content = f.read()
                if not content.strip():
                    logger.warning(f"OHLCV file is empty: {cache_file}")
                    continue
                data = json.loads(content)

            # Determine data structure
            if isinstance(data, dict) and "ohlcv" in data:
                ohlcv_list = data["ohlcv"]
            elif isinstance(data, list):
                ohlcv_list = data
            else:
                raise ValueError("Unexpected OHLCV data format")

            if not isinstance(ohlcv_list, list) or not ohlcv_list:
                raise ValueError("Empty or invalid OHLCV list")

            # Convert to DataFrame and validate
            df = pd.DataFrame(ohlcv_list)

            # Timestamp processing
            if "timestamp_ms" in df.columns:
                df["timestamp"] = pd.to_datetime(
                    df["timestamp_ms"], errors="coerce", unit="ms", utc=True
                )
            elif "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
            else:
                raise ValueError("Missing timestamp column")

            df = df.dropna(subset=["timestamp"]).set_index("timestamp")
            if df.empty:
                raise ValueError("No valid timestamps after conversion")

            # Numeric processing
            numeric_cols = ["open", "high", "low", "close", "volume"]
            if not all(col in df.columns for col in numeric_cols):
                raise ValueError("Missing required OHLCV columns")

            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.dropna(subset=numeric_cols)
            if df.empty:
                raise ValueError("No valid numeric OHLCV data after conversion")

            df = df.sort_index()
            full_data[pair] = df[["open", "high", "low", "close", "volume"]]
            logger.info(f"Loaded {len(df)} valid OHLCV data points for {pair}")

        except FileNotFoundError:
            logger.error(f"OHLCV file vanished during load: {cache_file}. Skipping pair {pair}.")
            continue
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for {pair} from {cache_file}: {e}. Skipping pair.")
            continue
        except (ValueError, TypeError, KeyError) as e:
            logger.error(
                f"Data processing error for {pair} from {cache_file}: {e}. Skipping pair.",
                exc_info=True,
            )
            continue
        except OSError as e:
            logger.error(f"OS error reading {cache_file} for {pair}: {e}. Skipping pair.")
            continue
        except Exception as e:
            logger.error(
                f"Unexpected error loading OHLCV data for {pair} from {cache_file}: {e}",
                exc_info=True,
            )
            continue

    if not full_data:
        logger.error("Failed to load valid OHLCV data for ANY specified pair.")
    return full_data


def generate_walk_forward_windows(
    start_date_str: str,
    end_date_str: str,
    train_days: int,
    test_days: int,
    step_days: int,
) -> Iterator[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Generates testing windows for walk-forward analysis with validation."""
    try:
        # Validate inputs
        if not all([start_date_str, end_date_str]):
            raise ValueError("Start and end dates are required.")
        if not isinstance(train_days, int) or train_days <= 0:
            raise ValueError("Training days must be a positive integer.")
        if not isinstance(test_days, int) or test_days <= 0:
            raise ValueError("Testing days must be a positive integer.")
        if not isinstance(step_days, int) or step_days <= 0:
            raise ValueError("Step days must be a positive integer.")

        start_date = pd.Timestamp(start_date_str, tz="UTC")
        end_date = pd.Timestamp(end_date_str, tz="UTC")

        if start_date >= end_date:
            raise ValueError("Start date must be before end date.")

        total_period_days = train_days + test_days
        window_start = start_date
        window_end = window_start + pd.Timedelta(days=total_period_days)

        while window_end <= end_date:
            test_start = window_start + pd.Timedelta(days=train_days)
            test_end = window_end
            yield (test_start, test_end)

            # Move the window forward
            window_start += pd.Timedelta(days=step_days)
            window_end = window_start + pd.Timedelta(days=total_period_days)

    except (ValueError, TypeError) as e:
        logger.error(f"Error generating walk-forward windows: {e}")
        raise


def slice_data(
    full_data: Dict[str, pd.DataFrame], start_date: pd.Timestamp, end_date: pd.Timestamp
) -> Dict[str, pd.DataFrame]:
    """Slices the full OHLCV dataframes for a given date range safely."""
    sliced_data = {}
    min_required_length = 50
    logger.debug(f"Slicing data from {start_date} to {end_date}")

    if not isinstance(full_data, dict):
        return {}

    for pair, df in full_data.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue

        try:
            # Ensure index is DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                logger.warning(f"Index for pair {pair} is not DatetimeIndex. Skipping slice.")
                continue

            # Ensure dates are timezone-aware (assume UTC if naive)
            if start_date.tzinfo is None:
                start_utc = start_date.tz_localize("UTC")
            else:
                start_utc = start_date.tz_convert("UTC")

            if end_date.tzinfo is None:
                end_utc = end_date.tz_localize("UTC")
            else:
                end_utc = end_date.tz_convert("UTC")

            mask = (df.index >= start_utc) & (df.index < end_utc)
            sliced_df = df.loc[mask]

            if len(sliced_df) < min_required_length:
                logger.warning(
                    f"Skipping slice for {pair}: Insufficient data ({len(sliced_df)} rows) < "
                    f"{min_required_length} in range {start_utc.date()} - {end_utc.date()}"
                )
                continue

            sliced_data[pair] = sliced_df

        except Exception as e:
            logger.error(f"Error slicing data for pair {pair}: {e}", exc_info=True)
            continue

    return sliced_data


def flatten_dict(d: Dict, parent_key: str = "", sep: str = "_") -> Dict:
    """Flattens a nested dictionary."""
    items: list = []
    if not isinstance(d, dict):
        return {}

    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            items.append((new_key, json.dumps(v)))
        else:
            items.append((new_key, v if v is not None else ""))

    return dict(items)


# --- Main Batch Runner Class ---


class BatchTester:
    """Manages the execution of multiple backtest runs with error handling."""

    def __init__(self, args):
        if args is None:
            raise ValueError("Arguments object cannot be None.")

        self.args = args
        self.csv_header = None  # Initialize CSV header attribute

        try:
            self.base_config = BotConfig(config_path=args.config)
        except Exception as config_e:
            logger.critical(f"FATAL: Failed to load base configuration: {config_e}", exc_info=True)
            raise RuntimeError("Base configuration loading failed.") from config_e

        # Process pairs safely
        self.pairs = []
        if isinstance(args.pairs, str):
            self.pairs = [
                p.strip().upper().replace("_", "/") for p in args.pairs.split(",") if p.strip()
            ]

        if not self.pairs:
            raise ValueError("No valid trading pairs specified.")

        self.timeframe = args.timeframe
        self.ohlcv_data_dir = Path(args.ohlcv_data_dir)

        if not self.ohlcv_data_dir.is_dir():
            raise FileNotFoundError(f"OHLCV data directory not found: {self.ohlcv_data_dir}")

        self.full_ohlcv_data: Dict[str, pd.DataFrame] = {}
        self.all_results: List[Dict] = []

        # Ensure output directory exists
        try:
            DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(
                f"Could not create output directory {DEFAULT_OUTPUT_DIR}: {e}. "
                "Results might not be saved."
            )

    def define_parameter_variations(self) -> List[Dict[str, Any]]:
        """Defines the parameter sets to test, handling potential errors."""
        variations: list = []

        try:
            # Example: Varying position size percentage
            if self.args.pos_sizes and isinstance(self.args.pos_sizes, str):
                sizes = [float(s.strip()) for s in self.args.pos_sizes.split(",") if s.strip()]
                for size_pct in sizes:
                    # Basic validation for percentage range
                    if 0.0 < size_pct <= 50.0:
                        variations.append({"trading.position_size_percent": size_pct})
                    else:
                        logger.warning(f"Skipping invalid position size parameter: {size_pct}%")

            # If no specific variations provided, add one empty dict to run with base config
            if not variations:
                variations.append({})

            logger.info(f"Defined {len(variations)} parameter sets to test.")
            return variations

        except (ValueError, TypeError) as e:
            logger.error(
                f"Error defining parameter variations: {e}. Running only base config.",
                exc_info=True,
            )
            return [{}]

    def run_all(self):
        """Loads data and runs all configured backtest variations."""
        try:
            self.full_ohlcv_data = load_full_ohlcv_data(
                self.pairs, self.timeframe, self.ohlcv_data_dir
            )

            if not self.full_ohlcv_data:
                logger.critical("Failed to load any valid OHLCV data. Aborting batch run.")
                return

            # Update self.pairs based on successfully loaded data
            self.pairs = list(self.full_ohlcv_data.keys())
            if not self.pairs:
                logger.critical("No pairs remaining after loading OHLCV data. Aborting.")
                return

        except FileNotFoundError as e:
            logger.critical(f"{e}. Aborting batch run.")
            return
        except Exception as e:
            logger.critical(
                f"Unhandled error during data loading: {e}. Aborting batch run.",
                exc_info=True,
            )
            return

        parameter_sets = self.define_parameter_variations()
        output_file = (
            DEFAULT_OUTPUT_DIR / f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        logger.info(f"Starting batch run. Results will be saved incrementally to {output_file}")
        run_counter = 0

        # Determine run mode: Walk-Forward or Parameter Optimization
        is_walk_forward = all(
            [
                self.args.wf_start,
                self.args.wf_end,
                self.args.wf_train_days,
                self.args.wf_test_days,
                self.args.wf_step_days,
            ]
        )

        # Initialize CSV file with header
        if parameter_sets:
            self._initialize_csv(output_file, parameter_sets[0])

        # --- Main Execution Loop ---
        try:
            if is_walk_forward:
                logger.info("Running in Walk-Forward Mode.")
                try:
                    date_windows_gen = generate_walk_forward_windows(
                        self.args.wf_start,
                        self.args.wf_end,
                        self.args.wf_train_days,
                        self.args.wf_test_days,
                        self.args.wf_step_days,
                    )
                    date_windows_list = list(date_windows_gen)
                    total_windows = len(date_windows_list)
                    logger.info(f"Generated {total_windows} walk-forward test windows.")
                except ValueError as e:
                    logger.critical(f"Failed to generate walk-forward windows: {e}. Aborting.")
                    return

                for i, (test_start, test_end) in enumerate(date_windows_list):
                    train_start = test_start - pd.Timedelta(days=self.args.wf_train_days)
                    logger.info(
                        f"\n--- Walk-Forward Window {i+1}/{total_windows}: "
                        f"Test Period {test_start.date()} to {test_end.date()} ---"
                    )

                    window_data = slice_data(self.full_ohlcv_data, train_start, test_end)
                    if not window_data:
                        logger.warning(
                            f"Skipping window {i+1} due to insufficient data in slice "
                            f"{train_start.date()} - {test_end.date()}."
                        )
                        continue

                    for params in parameter_sets:
                        run_counter += 1
                        logger.info(
                            f"  Running Test {run_counter} (Window {i+1}) with params: "
                            f"{params or 'Base Config'}"
                        )
                        run_metrics = self.execute_single_run(window_data, params)

                        if run_metrics:
                            flat_metrics = flatten_dict(run_metrics)
                            flat_metrics["run_number"] = run_counter
                            flat_metrics["wf_window"] = i + 1
                            flat_metrics["test_start_date"] = test_start.date().isoformat()
                            flat_metrics["test_end_date"] = test_end.date().isoformat()

                            for p_key, p_val in params.items():
                                flat_metrics[f"param_{p_key}"] = p_val

                            self.all_results.append(flat_metrics)
                            self._append_result_to_csv(output_file, flat_metrics)
                        else:
                            logger.error(
                                f"Run {run_counter} (Window {i+1}, Params: {params or 'Base'}) "
                                "failed to produce metrics."
                            )

            else:
                logger.info("Running in Parameter Optimization Mode (using full dataset).")
                total_runs = len(parameter_sets)

                for i, params in enumerate(parameter_sets):
                    run_counter += 1
                    logger.info(
                        f"\n--- Running Test {run_counter}/{total_runs} with params: "
                        f"{params or 'Base Config'} ---"
                    )

                    run_metrics = self.execute_single_run(self.full_ohlcv_data, params)

                    if run_metrics:
                        flat_metrics = flatten_dict(run_metrics)
                        flat_metrics["run_number"] = run_counter

                        for p_key, p_val in params.items():
                            flat_metrics[f"param_{p_key}"] = p_val

                        self.all_results.append(flat_metrics)
                        self._append_result_to_csv(output_file, flat_metrics)
                    else:
                        logger.error(
                            f"Run {run_counter} (Params: {params or 'Base'}) "
                            "failed to produce metrics."
                        )

            logger.info(f"Batch run completed. {len(self.all_results)} successful runs processed.")

        except KeyboardInterrupt:
            logger.warning("Batch run interrupted by user.")
        except Exception as e:
            logger.critical(f"Unexpected error during batch execution loop: {e}", exc_info=True)
        finally:
            logger.info(f"Final batch results saved to {output_file}")

    def execute_single_run(
        self, historical_data_slice: Dict[str, pd.DataFrame], params_override: Dict
    ) -> Optional[Dict]:
        """Executes a single backtest run with specific data and parameters, handling errors."""
        run_config = None
        runner = None

        try:
            # 1. Create a modified config for this run
            run_config = copy.deepcopy(self.base_config)
            if not run_config:
                raise ValueError("Base config is invalid after copy.")

            # Apply parameter overrides safely
            if isinstance(params_override, dict):
                for key, value in params_override.items():
                    keys = key.split(".")
                    cfg_section = run_config.settings
                    valid_path = True

                    for k in keys[:-1]:
                        cfg_section = cfg_section.setdefault(k, {})
                        if not isinstance(cfg_section, dict):
                            logger.warning(
                                f"Invalid path structure in config override key: {key}. "
                                "Cannot set override."
                            )
                            valid_path = False
                            break

                    if valid_path:
                        cfg_section[keys[-1]] = value
                        logger.debug(f"  Overriding config: {key} = {value}")

                # Re-process percentage values if they were overridden
                if any("percent" in k for k in params_override.keys()):
                    run_config._process_percentage_values()

                # Re-validate the modified config
                run_config._validate_config()

            # 2. Instantiate BacktestRunner
            if not historical_data_slice or not isinstance(historical_data_slice, dict):
                logger.error("Invalid or empty historical data slice provided for backtest run.")
                return None

            runner = BacktestRunner(
                config=run_config,
                pairs=list(historical_data_slice.keys()),
                timeframe=self.timeframe,
                data_dir=DATA_DIR,
                historical_data=historical_data_slice,
            )

            # 3. Run the backtest
            metrics = runner.run()
            return metrics

        except (ValueError, TypeError, AttributeError, KeyError) as config_err:
            logger.error(
                f"Error setting up config or runner for backtest run: {config_err}",
                exc_info=True,
            )
            return None
        except Exception as e:
            logger.error(
                f"Unexpected exception during single backtest run execution: {e}",
                exc_info=True,
            )
            return None

    def _initialize_csv(self, output_file: Path, sample_params: Dict):
        """Creates the CSV file and writes the header row."""
        header = ["run_number", "wf_window", "test_start_date", "test_end_date"]

        # Add parameter columns
        for p_key in sample_params.keys():
            header.append(f"param_{p_key}")

        # Add flattened metrics columns
        metric_keys = [
            "run_details_start_timestamp",
            "run_details_end_timestamp",
            "run_details_duration_days",
            "run_details_initial_capital",
            "run_details_final_capital",
            "run_details_quote_currency",
            "performance_total_pnl",
            "performance_total_pnl_pct",
            "performance_total_fees",
            "performance_max_drawdown_pct",
            "performance_sharpe_ratio",
            "performance_sortino_ratio",
            "trade_stats_total_trades_executed",
            "trade_stats_total_round_trips",
            "trade_stats_wins",
            "trade_stats_losses",
            "trade_stats_neutral",
            "trade_stats_win_rate_pct",
            "trade_stats_profit_factor",
            "trade_stats_avg_win_pnl",
            "trade_stats_avg_loss_pnl",
            "trade_stats_max_win_pnl",
            "trade_stats_max_loss_pnl",
            "trade_stats_avg_trade_duration_seconds",
            "trade_stats_avg_trade_duration_human",
        ]
        header.extend(metric_keys)

        try:
            with output_file.open("w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(header)
            logger.info(f"Initialized results CSV: {output_file}")
            self.csv_header = header
        except OSError as e:
            logger.error(f"Failed to initialize results CSV file {output_file}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error initializing results CSV: {e}", exc_info=True)

    def _append_result_to_csv(self, output_file: Path, result_dict: Dict):
        """Appends a single result dictionary to the CSV file."""
        if not self.csv_header:
            logger.error("CSV header not initialized. Cannot append result.")
            return

        try:
            row_data = [result_dict.get(h, "") for h in self.csv_header]

            with output_file.open("a", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile, quoting=csv.QUOTE_NONNUMERIC)
                writer.writerow(row_data)
        except OSError as e:
            logger.error(f"Failed to append result to CSV file {output_file}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error appending result to CSV: {e}", exc_info=True)


# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Backtester & Walk-Forward Analyzer")

    # General Args
    parser.add_argument(
        "--pairs",
        required=True,
        help="Comma-separated list of trading pairs (e.g., BTC/USD,ETH/USD)",
    )
    parser.add_argument(
        "--timeframe",
        required=True,
        help="Timeframe of the historical data (e.g., 1h, 4h, 1d)",
    )
    parser.add_argument("--config", default=None, help="Path to the base user_config.json file")
    parser.add_argument(
        "--ohlcv-data-dir",
        default=str(CACHE_DIR),
        help=f"Directory containing OHLCV cache files (default: {CACHE_DIR})",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level",
    )

    # Parameter Variation Args
    parser.add_argument(
        "--pos-sizes",
        default=None,
        help="Comma-separated list of position size percentages to test (e.g., 3.0,5.0,7.5)",
    )

    # Walk-Forward Args
    parser.add_argument("--wf-start", default=None, help="Walk-forward start date (YYYY-MM-DD)")
    parser.add_argument("--wf-end", default=None, help="Walk-forward end date (YYYY-MM-DD)")
    parser.add_argument(
        "--wf-train-days",
        type=int,
        default=None,
        help="Number of days in training period",
    )
    parser.add_argument(
        "--wf-test-days",
        type=int,
        default=None,
        help="Number of days in testing period",
    )
    parser.add_argument(
        "--wf-step-days",
        type=int,
        default=None,
        help="Number of days to slide the window forward",
    )

    try:
        args = parser.parse_args()

        # Set log level
        log_level_upper = args.log_level.upper()
        logger.setLevel(log_level_upper)

        if "batch_logger_instance" in globals() and hasattr(batch_logger_instance, "logger"):
            for handler in batch_logger_instance.logger.handlers:
                handler.setLevel(log_level_upper)

        logger.info(f"Batch tester log level set to {log_level_upper}")

        # Validate Walk-Forward Args
        wf_args_present = [
            args.wf_start,
            args.wf_end,
            args.wf_train_days,
            args.wf_test_days,
            args.wf_step_days,
        ]

        if any(a is not None for a in wf_args_present) and not all(
            a is not None for a in wf_args_present
        ):
            parser.error(
                "If using walk-forward, --wf-start, --wf-end, --wf-train-days, "
                "--wf-test-days, and --wf-step-days must ALL be provided."
            )

        # Initialize and run the batch tester
        batch_runner = BatchTester(args)
        batch_runner.run_all()

    except (ValueError, FileNotFoundError, RuntimeError) as init_err:
        logger.critical(f"Failed to initialize or start batch testing: {init_err}", exc_info=True)
        sys.exit(1)
    except argparse.ArgumentError as arg_err:
        logger.critical(f"Invalid command line arguments: {arg_err}")
        sys.exit(1)
    except Exception as e:
        logger.critical(
            f"Batch testing execution failed with unhandled exception: {e}",
            exc_info=True,
        )
        sys.exit(1)
