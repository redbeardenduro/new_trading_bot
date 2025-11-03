"""
Manages the trading portfolio, including allocation targets, rebalancing logic,
risk assessment (VaR, stress tests), and simulated trade execution using centralized configuration.
Decoupled from the main bot instance, relying on injected exchange client and passed-in market state.
"""

import json
import math
import time
from datetime import datetime, timezone
from decimal import Decimal, DivisionByZero, InvalidOperation
from typing import Any, Dict, Optional, Tuple

import ccxt
import numpy as np

from common.common_logger import SKIPPED_TRADES_DIR, get_logger
from core.config import BotConfig
from core.interfaces import IExchangeClient

logger = get_logger("portfolio_manager")


class PortfolioManager:
    """
    Manages portfolio allocations, rebalancing, risk, and simulated execution.

    Reads configuration from the passed BotConfig instance, calculates dynamic
    target allocations based on provided market state, determines rebalancing trades,
    simulates execution using the injected exchange client, and assesses risk.

    Attributes:
        config (BotConfig): The centralized configuration object.
        exchange_client (IExchangeClient): The initialized exchange client instance.
        base_currencies (List[str]): List of base currencies being managed.
        quote_currency (str): The primary quote currency for valuation.
        target_allocations (Dict[str, float]): Current target allocation percentages.
        skipped_trades_log_file (Path): Path to log skipped rebalance trades.
    """

    def __init__(self, config: BotConfig, exchange_client: IExchangeClient) -> None:
        """
        Initialize the PortfolioManager.

        Args:
            config (BotConfig): The centralized configuration object.
            exchange_client (IExchangeClient): The initialized exchange client instance.
        """
        if config is None:
            raise ValueError("BotConfig cannot be None.")
        if exchange_client is None:
            raise ValueError("IExchangeClient cannot be None.")
        self.config = config
        self.exchange_client = exchange_client
        self.base_currencies = self.config.get("bot.base_currencies", [])
        quote_list = self.config.get("bot.quote_currencies", ["USD"])
        self.quote_currency = quote_list[0].upper() if quote_list else "USD"
        logger.info("Portfolio Manager using '%s' as the valuation currency.", self.quote_currency)
        num_assets = len(self.base_currencies)
        if num_assets == 0:
            logger.warning("PortfolioManager initialized with no base currencies!")
            self.target_allocations = {}
        else:
            self.target_allocations = {asset: 1.0 / num_assets for asset in self.base_currencies}
        try:
            self.fee_rate = Decimal(str(self.config.portfolio.simulation.fee_rate_multiplier))
        except (AttributeError, ValueError, TypeError, InvalidOperation):
            self.fee_rate = Decimal("0.001")
        try:
            self.slippage = Decimal(str(self.config.portfolio.simulation.slippage_multiplier))
        except (AttributeError, ValueError, TypeError, InvalidOperation):
            self.slippage = Decimal("0.001")
        try:
            self.min_order_value = Decimal(
                str(self.config.portfolio.simulation.min_order_value_quote)
            )
        except (AttributeError, ValueError, TypeError, InvalidOperation):
            self.min_order_value = Decimal("10.0")
        self.fixed_cost = Decimal("0.0")
        self.variable_cost_rate = self.fee_rate
        try:
            self.technical_weight = float(self.config.portfolio.target_allocation_weights.technical)
        except (AttributeError, ValueError, TypeError):
            self.technical_weight = 0.4
        try:
            self.sentiment_weight = float(self.config.portfolio.target_allocation_weights.sentiment)
        except (AttributeError, ValueError, TypeError):
            self.sentiment_weight = 0.3
        try:
            self.ai_weight = float(self.config.portfolio.target_allocation_weights.ai)
        except (AttributeError, ValueError, TypeError):
            self.ai_weight = 0.3
        try:
            self.rsi_oversold = float(
                self.config.portfolio.technical_factor_thresholds.rsi_oversold
            )
        except (AttributeError, ValueError, TypeError):
            self.rsi_oversold = 30.0
        try:
            self.rsi_overbought = float(
                self.config.portfolio.technical_factor_thresholds.rsi_overbought
            )
        except (AttributeError, ValueError, TypeError):
            self.rsi_overbought = 70.0
        try:
            self.macd_threshold = float(
                self.config.portfolio.technical_factor_thresholds.macd_threshold
            )
        except (AttributeError, ValueError, TypeError):
            self.macd_threshold = 0.0
        try:
            self.bb_threshold = float(
                self.config.portfolio.technical_factor_thresholds.bb_threshold_multiplier
            )
        except (AttributeError, ValueError, TypeError):
            self.bb_threshold = 0.05
        try:
            self.max_allocation_per_asset = float(
                self.config.portfolio.max_allocation_per_asset_multiplier
            )
        except (AttributeError, ValueError, TypeError):
            self.max_allocation_per_asset = 0.075
        try:
            self.min_allocation_per_asset = float(
                self.config.portfolio.min_allocation_per_asset_multiplier
            )
        except (AttributeError, ValueError, TypeError):
            self.min_allocation_per_asset = 0.01
        try:
            self.rebalance_threshold = float(self.config.trading.rebalance_threshold_multiplier)
        except (AttributeError, ValueError, TypeError):
            self.rebalance_threshold = 0.05
        self.dynamic_threshold_enabled = self.config.get("trading.dynamic_threshold_enabled", True)
        try:
            self.var_confidence_level = float(
                self.config.portfolio.risk_management.var_confidence_level
            )
        except (AttributeError, ValueError, TypeError):
            self.var_confidence_level = 0.95
        try:
            self.var_time_horizon = int(self.config.portfolio.risk_management.var_time_horizon_days)
        except (AttributeError, ValueError, TypeError):
            self.var_time_horizon = 1
        self.use_advanced_var = self.config.get(
            "portfolio.risk_management.use_monte_carlo_var", False
        )
        self.stress_test_scenarios = self.config.get(
            "portfolio.risk_management.stress_test_scenarios_multiplier", {}
        )
        try:
            self.volatility_lookback = int(self.config.trading.volatility_lookback_period)
        except (AttributeError, ValueError, TypeError):
            self.volatility_lookback = 14
        skipped_dir = SKIPPED_TRADES_DIR
        skipped_dir.mkdir(parents=True, exist_ok=True)
        self.skipped_trades_log_file = skipped_dir / "portfolio_skipped_trades.json"
        logger.info("PortfolioManager initialized. Targets: %s", self.target_allocations)
        logger.info(
            "Limits: Max Alloc=%.2f%%, Min Alloc=%.2f%%, Rebal Threshold=%.2f%%",
            self.max_allocation_per_asset * 100,
            self.min_allocation_per_asset * 100,
            self.rebalance_threshold * 100,
        )
        logger.info(
            "Dynamic Weights: Tech=%.2f, Sent=%.2f, AI=%.2f",
            self.technical_weight,
            self.sentiment_weight,
            self.ai_weight,
        )

    def determine_rebalance_actions(self, market_state: Dict[str, Dict[str, Any]]) -> list:
        """
        Main API method to determine and simulate rebalance actions.

        Args:
            market_state (Dict[str, Dict[str, Any]]): A dictionary where keys are pairs (e.g., 'BTC/USD')
                                                     and values are dictionaries containing the latest relevant
                                                     market data for that pair needed for dynamic allocation
                                                     (e.g., 'indicators', 'sentiment_metrics', 'ai_metrics').

        Returns:
            List[Dict]: A list of simulated trade execution dictionaries.
        """
        if not isinstance(market_state, dict):
            logger.error(
                "Invalid market_state received (not a dict). Cannot determine rebalance actions."
            )
            return []
        else:
            return self._rebalance_portfolio(market_state)

    def _rebalance_portfolio(self, market_state: Dict[str, Dict[str, Any]]) -> list:
        """Internal method to perform portfolio rebalancing calculations and simulations."""
        if not self.exchange_client:
            logger.error("Exchange client unavailable. Cannot rebalance.")
            return []
        if not self.base_currencies:
            logger.warning("No base currencies defined. Cannot rebalance.")
            return []
        holdings = self.get_current_holdings()
        if holdings is None:
            logger.error("Failed to get current holdings. Cannot rebalance.")
            return []
        (current_values, total_value) = self._calculate_current_values(holdings)
        if total_value is None or total_value <= Decimal("1e-9") or current_values is None:
            logger.warning(
                "Total portfolio value is zero or invalid (%s). Cannot rebalance.", total_value
            )
            return []
        current_allocations = {}
        if total_value > Decimal("1e-9"):
            current_allocations = {
                asset: float(value / total_value) for (asset, value) in current_values.items()
            }
            logger.info("Current Portfolio Value: %.2f %s", total_value, self.quote_currency)
            log_alloc_str = ", ".join([f"{k}: {v:.2%}" for (k, v) in current_allocations.items()])
            logger.info("Current Allocations: %s", log_alloc_str)
        else:
            logger.warning("Total portfolio value is zero, cannot calculate allocations.")
            return []
        target_allocations = self._adjust_target_allocations(market_state)
        if target_allocations is None:
            logger.error("Failed to determine target allocations. Cannot rebalance.")
            return []
        threshold = self.rebalance_threshold
        if self.dynamic_threshold_enabled:
            dynamic_thresh = self._calculate_dynamic_threshold(total_value, market_state)
            if dynamic_thresh is not None:
                threshold = dynamic_thresh
                logger.info("Using dynamic rebalance threshold: %.2f%%", threshold * 100)
            else:
                logger.warning("Failed to calculate dynamic threshold. Using fixed threshold.")
                threshold = self.rebalance_threshold
        logger.info("Rebalance threshold: %.2f%%", threshold * 100)
        trades_to_make = []
        skipped_trades = []
        max_rebalance_cost_pct = Decimal(
            str(self.config.get("portfolio.simulation.max_rebalance_cost_percent", 10.0))
        ) / Decimal("100.0")
        quote_currency_upper = self.quote_currency.upper()
        available_quote_balance = holdings.get(quote_currency_upper, Decimal("0.0"))
        for asset in self.base_currencies:
            asset_upper = asset.upper()
            current_alloc = current_allocations.get(asset_upper, 0.0)
            target_alloc = target_allocations.get(asset_upper, 0.0)
            diff = target_alloc - current_alloc
            if abs(diff) < threshold:
                continue
            trade_value = Decimal(str(diff)) * total_value
            action = "BUY" if diff > 0 else "SELL"
            abs_trade_value = abs(trade_value)
            pair = f"{asset_upper}/{quote_currency_upper}"
            logger.info(
                "Rebalance for %s: Target %.2f%%, Current %.2f%%. Action: %s ~%.2f %s",
                asset_upper,
                target_alloc * 100,
                current_alloc * 100,
                action,
                abs_trade_value,
                quote_currency_upper,
            )
            skip_reason = None
            current_price_dec = None
            if action == "BUY":
                required_quote_gross = abs_trade_value * (Decimal("1.0") + self.fee_rate)
                if required_quote_gross > available_quote_balance:
                    skip_reason = f"Insufficient {quote_currency_upper} balance pre-simulation (Req: ~{required_quote_gross:.2f}, Avail: {available_quote_balance:.2f})"
            elif action == "SELL":
                current_price_dec = self._get_current_price_from_exchange(pair)
                if current_price_dec is None or current_price_dec <= Decimal("1e-9"):
                    skip_reason = f"Could not get valid price ({current_price_dec}) for {pair} to check sell quantity"
                else:
                    try:
                        required_base_qty = abs_trade_value / current_price_dec
                        available_base_balance = holdings.get(asset_upper, Decimal("0.0"))
                        if required_base_qty > available_base_balance:
                            skip_reason = f"Insufficient {asset_upper} balance pre-simulation (Req: ~{required_base_qty:.8f}, Avail: {available_base_balance:.8f})"
                    except (DivisionByZero, InvalidOperation) as e:
                        skip_reason = f"Calculation error checking sell quantity for {pair}: {e}"
            if skip_reason:
                logger.warning("Skipping %s %s: %s", action, asset_upper, skip_reason)
                skipped_trades.append(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "asset": asset_upper,
                        "action": action,
                        "trade_value": float(trade_value),
                        "reason": skip_reason,
                        "current_allocation": current_alloc,
                        "target_allocation": target_alloc,
                    }
                )
                continue
            estimated_cost = self._calculate_transaction_cost(abs_trade_value)
            min_value_check = abs_trade_value >= self.min_order_value
            cost_check = estimated_cost <= abs_trade_value * max_rebalance_cost_pct
            if not min_value_check or not cost_check:
                skip_reason = f"Trade uneconomical. Value: {abs_trade_value:.2f} (Min: {self.min_order_value:.2f}), Est. Cost: {estimated_cost:.4f} (Max %: {max_rebalance_cost_pct:.1%})"
                logger.warning("Skipping %s %s: %s", action, asset_upper, skip_reason)
                skipped_trades.append(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "asset": asset_upper,
                        "action": action,
                        "trade_value": float(trade_value),
                        "reason": skip_reason,
                        "current_allocation": current_alloc,
                        "target_allocation": target_alloc,
                    }
                )
                continue
            trades_to_make.append(
                {
                    "asset": asset_upper,
                    "action": action,
                    "trade_value": float(trade_value),
                    "current_allocation": current_alloc,
                    "target_allocation": target_alloc,
                }
            )
        executed_trades_info = []
        trades_to_make.sort(
            key=lambda x: abs(x["target_allocation"] - x["current_allocation"]), reverse=True
        )
        for trade_info in trades_to_make:
            executed_trade = self._simulate_trade_execution(
                trade_info["action"], trade_info["asset"], trade_info["trade_value"]
            )
            if isinstance(executed_trade, dict) and executed_trade.get("status") not in [
                "rejected",
                "failed",
                "canceled",
                "error",
            ]:
                executed_trade["current_allocation"] = trade_info["current_allocation"]
                executed_trade["target_allocation"] = trade_info["target_allocation"]
                executed_trade["executed"] = True
                executed_trades_info.append(executed_trade)
                logger.info(
                    "Simulated execution for %s %s successful (Sim ID: %s).",
                    trade_info["action"],
                    trade_info["asset"],
                    executed_trade.get("id"),
                )
                if (
                    trade_info["action"] == "BUY"
                    and executed_trade.get("cost") is not None
                    and (executed_trade.get("fee", {}).get("cost") is not None)
                ):
                    try:
                        cost_total_dec = Decimal(str(executed_trade["cost"])) + Decimal(
                            str(executed_trade["fee"]["cost"])
                        )
                        available_quote_balance -= cost_total_dec
                        logger.debug(
                            "Updated available quote balance after sim buy: %.2f",
                            available_quote_balance,
                        )
                    except (InvalidOperation, TypeError):
                        logger.warning(
                            "Could not update available quote balance after simulated buy due to invalid cost/fee."
                        )
            else:
                skip_reason = (
                    executed_trade.get("info", {}).get("error", "Simulation failed/rejected")
                    if isinstance(executed_trade, dict)
                    else "Simulation failed (Invalid response)"
                )
                logger.warning(
                    "Skipping %s %s post-simulation: %s",
                    trade_info["action"],
                    trade_info["asset"],
                    skip_reason,
                )
                skipped_trades.append(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "asset": trade_info["asset"],
                        "action": trade_info["action"],
                        "trade_value": trade_info["trade_value"],
                        "reason": skip_reason,
                        "current_allocation": trade_info["current_allocation"],
                        "target_allocation": trade_info["target_allocation"],
                    }
                )
        if skipped_trades:
            self._log_skipped_trades(skipped_trades)
            self._check_for_anomalous_patterns(skipped_trades)
        try:
            self._perform_risk_assessment(holdings, current_values, total_value, market_state)
        except Exception as risk_e:
            logger.error("Error during risk assessment: %s", risk_e, exc_info=True)
        if executed_trades_info:
            logger.info("--- Post-Rebalance Snapshot (Simulated based on Targets) ---")
            logger.info("Target Portfolio Value: ~$%.2f", total_value)
            asset_target_sum = sum((target_allocations.get(a, 0.0) for a in self.base_currencies))
            quote_target_alloc = max(0.0, 1.0 - asset_target_sum)
            all_assets_to_log = self.base_currencies + [self.quote_currency]
            log_targets = {
                asset: target_allocations.get(asset, 0.0) for asset in self.base_currencies
            }
            log_targets[quote_currency_upper] = quote_target_alloc
            for asset in all_assets_to_log:
                asset_upper = asset.upper()
                target_alloc = log_targets.get(asset_upper, 0.0)
                target_value_dec = Decimal(str(target_alloc)) * total_value
                if target_value_dec > Decimal("0.01") or (
                    asset_upper == quote_currency_upper and target_value_dec > 0
                ):
                    logger.info(
                        "%s: Target Value ~$%.2f (%.2f%%)",
                        asset_upper,
                        target_value_dec,
                        target_alloc * 100,
                    )
            logger.info("-------------------------------------------")
        elif not skipped_trades:
            logger.info("No rebalancing trades needed or executed.")
        return executed_trades_info

    def _adjust_target_allocations(
        self, market_state: Dict[str, Dict[str, Any]]
    ) -> Optional[Dict[str, float]]:
        """Dynamically adjust target allocations based on provided market state and config weights."""
        logger.info("Adjusting target allocations...")
        if not self.base_currencies:
            return {}
        try:
            base_allocations = {
                asset: 1.0 / len(self.base_currencies) for asset in self.base_currencies
            }
            technical_factors = self._calculate_technical_factors(market_state)
            sentiment_factors = self._calculate_sentiment_factors(market_state)
            ai_factors = self._calculate_ai_factors(market_state)
            tech_w = float(self.technical_weight)
            sent_w = float(self.sentiment_weight)
            ai_w = float(self.ai_weight)
            total_weight = tech_w + sent_w + ai_w
            if abs(total_weight - 1.0) > 1e-06:
                logger.warning(
                    "Dynamic allocation weights do not sum to 1 (%.3f). Normalizing.", total_weight
                )
                if total_weight > 1e-09:
                    tech_w /= total_weight
                    sent_w /= total_weight
                    ai_w /= total_weight
                else:
                    tech_w = sent_w = ai_w = 1.0 / 3.0
            combined_factors = {}
            for asset in self.base_currencies:
                tech_f = technical_factors.get(asset, 1.0)
                sent_f = sentiment_factors.get(asset, 1.0)
                ai_f = ai_factors.get(asset, 1.0)
                combined_factors[asset] = max(0.0, tech_w * tech_f + sent_w * sent_f + ai_w * ai_f)
            raw_allocations = {
                asset: base_allocations[asset] * combined_factors.get(asset, 1.0)
                for asset in self.base_currencies
            }
            total_raw_allocation = sum(raw_allocations.values())
            if total_raw_allocation > 1e-09:
                normalized_allocations = {
                    asset: raw / total_raw_allocation for (asset, raw) in raw_allocations.items()
                }
            else:
                logger.warning(
                    "Total raw allocation adjustment is zero or negative. Using base allocations."
                )
                normalized_allocations = base_allocations
            final_allocations = self._apply_concentration_limits(normalized_allocations)
            if final_allocations is None:
                return None
            allocation_changes = {
                asset: f"{base_allocations[asset]:.1%} -> {final_allocations[asset]:.1%}"
                for asset in self.base_currencies
            }
            logger.info("Target Allocation Adjustments: %s", allocation_changes)
            self.target_allocations = final_allocations
            return final_allocations
        except Exception as e:
            logger.error("Error adjusting target allocations: %s", e, exc_info=True)
            return None

    def _apply_concentration_limits(
        self, allocations: Dict[str, float]
    ) -> Optional[Dict[str, float]]:
        """Apply max/min concentration limits from config, redistributing excess/deficit."""
        if not allocations:
            return {}
        min_limit = float(self.min_allocation_per_asset)
        max_limit = float(self.max_allocation_per_asset)
        logger.debug(
            "Applying concentration limits (Min: %.1f%%, Max: %.1f%%)",
            min_limit * 100,
            max_limit * 100,
        )
        adj_alloc = allocations.copy()
        assets = list(allocations.keys())
        max_iterations = len(assets) * 3
        for iteration in range(max_iterations):
            changed_this_iter = False
            assets_over_max = {a for a in assets if adj_alloc.get(a, 0.0) > max_limit}
            excess_to_distribute = sum(
                (adj_alloc[a] - max_limit for a in assets_over_max if a in adj_alloc)
            )
            if excess_to_distribute > 1e-09:
                changed_this_iter = True
                for a in assets_over_max:
                    adj_alloc[a] = max_limit
                assets_under_max = {a for a in assets if a not in assets_over_max}
                if assets_under_max:
                    current_total_under_max = sum((adj_alloc.get(a, 0.0) for a in assets_under_max))
                    if current_total_under_max > 1e-09:
                        for a in assets_under_max:
                            adj_alloc[a] += excess_to_distribute * (
                                adj_alloc.get(a, 0.0) / current_total_under_max
                            )
                    elif len(assets_under_max) > 0:
                        for a in assets_under_max:
                            adj_alloc[a] += excess_to_distribute / len(assets_under_max)
            assets_under_min = {a for a in assets if adj_alloc.get(a, 0.0) < min_limit}
            deficit_to_cover = sum((min_limit - adj_alloc.get(a, 0.0) for a in assets_under_min))
            if deficit_to_cover > 1e-09:
                changed_this_iter = True
                for a in assets_under_min:
                    adj_alloc[a] = min_limit
                assets_over_min = {a for a in assets if a not in assets_under_min}
                if assets_over_min:
                    reducible_amount_total = sum(
                        (max(0.0, adj_alloc.get(a, 0.0) - min_limit) for a in assets_over_min)
                    )
                    if reducible_amount_total >= deficit_to_cover - 1e-09:
                        reduction_needed = deficit_to_cover
                        for a in assets_over_min:
                            available_reduction = max(0.0, adj_alloc.get(a, 0.0) - min_limit)
                            proportion = (
                                available_reduction / reducible_amount_total
                                if reducible_amount_total > 1e-09
                                else 0.0
                            )
                            adj_alloc[a] -= reduction_needed * proportion
                    else:
                        logger.warning(
                            "Cannot fully cover min allocation deficit (%.2f%%) from available (%.2f%%). Allocations might sum < 100%%. Reducing available amounts.",
                            deficit_to_cover * 100,
                            reducible_amount_total * 100,
                        )
                        deficit_per_asset = (
                            deficit_to_cover / len(assets_over_min)
                            if len(assets_over_min) > 0
                            else 0.0
                        )
                        for a in assets_over_min:
                            adj_alloc[a] = max(min_limit, adj_alloc.get(a, 0.0) - deficit_per_asset)
                else:
                    logger.warning(
                        "Cannot cover min allocation deficit (%.2f%%): No assets available to reduce from.",
                        deficit_to_cover * 100,
                    )
            if not changed_this_iter:
                logger.debug("Concentration limits converged after %s iterations.", iteration + 1)
                break
        else:
            logger.warning(
                "Concentration limit application did not converge after %s iterations.",
                max_iterations,
            )
        final_total = sum(adj_alloc.values())
        if final_total > 1e-09 and abs(final_total - 1.0) > 1e-09:
            logger.debug("Renormalizing final allocations (Sum was %.6f)", final_total)
            try:
                adj_alloc = {a: alloc / final_total for (a, alloc) in adj_alloc.items()}
            except ZeroDivisionError:
                logger.error("Final normalization failed due to zero total allocation sum.")
                return None
        elif final_total <= 1e-09:
            logger.warning("Final allocation sum is zero or negative after applying limits.")
            return adj_alloc
        logger.info(
            "Final allocations after limits: %s", {k: f"{v:.2%}" for (k, v) in adj_alloc.items()}
        )
        return adj_alloc

    def _calculate_technical_factors(
        self, market_state: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate adjustment factors based on technical indicators using config thresholds and market_state."""
        logger.debug("Calculating technical factors...")
        factors = {}
        for asset in self.base_currencies:
            pair = f"{asset}/{self.quote_currency}"
            pair_state = market_state.get(pair, {})
            indicators = pair_state.get("indicators", {})
            if not indicators or not isinstance(indicators, dict):
                factors[asset] = 1.0
                continue
            asset_factor = 1.0
            rsi = indicators.get("RSI")
            if rsi is not None:
                try:
                    rsi_f = float(rsi)
                    if rsi_f < self.rsi_oversold:
                        factor_increase = (
                            max(0.0, (self.rsi_oversold - rsi_f) / (self.rsi_oversold or 1.0))
                            if self.rsi_oversold > 0
                            else 0.0
                        )
                        asset_factor *= 1.0 + min(0.5, factor_increase)
                    elif rsi_f > self.rsi_overbought:
                        factor_decrease = (
                            max(
                                0.0,
                                (rsi_f - self.rsi_overbought)
                                / (100.0 - self.rsi_overbought or 1.0),
                            )
                            if self.rsi_overbought < 100
                            else 0.0
                        )
                        asset_factor *= max(0.1, 1.0 - min(0.5, factor_decrease))
                except (ValueError, TypeError, ZeroDivisionError):
                    pass
            macd_hist = indicators.get("MACD_Hist")
            if macd_hist is not None:
                try:
                    macd_hist_f = float(macd_hist)
                    if macd_hist_f > self.macd_threshold:
                        asset_factor *= 1.05
                    elif macd_hist_f < -self.macd_threshold:
                        asset_factor *= 0.95
                except (ValueError, TypeError):
                    pass
            bb_upper = indicators.get("BB_Upper")
            bb_lower = indicators.get("BB_Lower")
            ticker = pair_state.get("ticker", {})
            current_price = ticker.get("last") if isinstance(ticker, dict) else None
            if current_price is None:
                current_price = self._get_current_price_from_exchange(pair)
            if bb_upper is not None and bb_lower is not None and (current_price is not None):
                try:
                    bb_u_f = float(bb_upper)
                    bb_l_f = float(bb_lower)
                    price_f = float(current_price)
                    if bb_u_f > bb_l_f:
                        band_width = bb_u_f - bb_l_f
                        if band_width > 1e-09:
                            lower_thresh = bb_l_f + band_width * self.bb_threshold
                            upper_thresh = bb_u_f - band_width * self.bb_threshold
                            if price_f < lower_thresh:
                                asset_factor *= 1.1
                            elif price_f > upper_thresh:
                                asset_factor *= 0.9
                except (ValueError, TypeError, ZeroDivisionError):
                    pass
            factors[asset] = max(0.0, asset_factor)
        logger.debug("Calculated technical factors: %s", factors)
        return factors

    def _calculate_sentiment_factors(
        self, market_state: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate adjustment factors based on aggregated sentiment data from market_state."""
        logger.debug("Calculating sentiment factors.")
        factors = {}
        for asset in self.base_currencies:
            pair = f"{asset}/{self.quote_currency}"
            sentiment_metrics_dict = market_state.get(pair, {}).get("sentiment_metrics", {})
            if not sentiment_metrics_dict or not isinstance(sentiment_metrics_dict, dict):
                factors[asset] = 1.0
                continue
            strengths = []
            for metrics in sentiment_metrics_dict.values():
                if isinstance(metrics, dict):
                    strength_val = metrics.get("strength")
                    if strength_val is not None:
                        try:
                            strengths.append(float(strength_val))
                        except (ValueError, TypeError):
                            pass
            if strengths:
                try:
                    combined_strength = sum(strengths) / len(strengths)
                    sentiment_factor = 1.0 + combined_strength * 0.5
                    factors[asset] = max(0.1, min(2.0, sentiment_factor))
                except ZeroDivisionError:
                    factors[asset] = 1.0
            else:
                factors[asset] = 1.0
        logger.debug("Calculated sentiment factors: %s", factors)
        return factors

    def _calculate_ai_factors(self, market_state: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Calculate adjustment factors based on AI analysis from market_state."""
        logger.debug("Calculating AI factors.")
        factors = {}
        for asset in self.base_currencies:
            pair = f"{asset}/{self.quote_currency}"
            ai_metrics = market_state.get(pair, {}).get("ai_metrics", {})
            if not ai_metrics or not isinstance(ai_metrics, dict) or ai_metrics.get("error"):
                factors[asset] = 1.0
                continue
            ai_strength = ai_metrics.get("strength", 0.0)
            ai_confidence = ai_metrics.get("confidence", 0.5)
            try:
                ai_strength_f = float(ai_strength)
                ai_confidence_f = float(ai_confidence)
                ai_factor = 1.0 + ai_strength_f * ai_confidence_f * 0.5
                factors[asset] = max(0.1, min(2.0, ai_factor))
            except (ValueError, TypeError):
                logger.warning(
                    "Invalid AI strength (%s) or confidence (%s) for %s. Using default factor 1.0.",
                    ai_strength,
                    ai_confidence,
                    asset,
                )
                factors[asset] = 1.0
        logger.debug("Calculated AI factors: %s", factors)
        return factors

    def get_current_holdings(self) -> Optional[Dict[str, Decimal]]:
        """Get current asset holdings from the exchange client, returning Decimals."""
        holdings_dec = {}
        if not self.exchange_client:
            logger.error("Exchange client unavailable. Cannot fetch holdings.")
            return None
        try:
            balance_float = self.exchange_client.get_balance(force_refresh=True)
            if balance_float is None:
                logger.warning("Received None balance from exchange client.")
                return None
            if not isinstance(balance_float, dict):
                logger.warning("Received invalid/empty balance from exchange client.")
                return None
            relevant_assets = self.base_currencies + [self.quote_currency]
            for asset in relevant_assets:
                asset_upper = asset.upper()
                balance_val = balance_float.get(asset_upper, balance_float.get(asset, 0.0))
                try:
                    holdings_dec[asset_upper] = Decimal(str(balance_val))
                except (InvalidOperation, TypeError):
                    logger.warning(
                        "Invalid balance value '%s' for %s. Setting holding to 0.",
                        balance_val,
                        asset,
                    )
                    holdings_dec[asset_upper] = Decimal("0")
            log_holdings_str = {k: f"{v:.8f}" for (k, v) in holdings_dec.items()}
            logger.info("Fetched current holdings: %s", log_holdings_str)
            return holdings_dec
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            logger.error("Exchange error fetching holdings: %s", e, exc_info=True)
            return None
        except Exception as e:
            logger.error("Unexpected error fetching holdings: %s", e, exc_info=True)
            return None

    def _get_current_price_from_exchange(self, pair: str) -> Optional[Decimal]:
        """Helper to get current price directly via exchange client, returns Decimal."""
        if not self.exchange_client:
            return None
        price_float = None
        try:
            ticker = self.exchange_client.get_ticker(pair, force_refresh=True)
            if ticker and isinstance(ticker, dict) and (ticker.get("last") is not None):
                price_float = ticker["last"]
            else:
                ohlcv = self.exchange_client.get_ohlcv(pair, limit=1, force_refresh=True)
                if ohlcv and isinstance(ohlcv, list) and (len(ohlcv) > 0):
                    last_candle = ohlcv[0]
                    if isinstance(last_candle, dict):
                        if last_candle.get("close") is not None:
                            price_float = last_candle["close"]
                    elif isinstance(last_candle, list):
                        if len(last_candle) > 4:
                            price_float = last_candle[4]
            if price_float is not None:
                try:
                    return Decimal(str(price_float))
                except (InvalidOperation, TypeError):
                    logger.warning(
                        "Invalid price value %s for %s from exchange.", price_float, pair
                    )
                    return None
            else:
                logger.warning("Could not fetch current price for %s from exchange.", pair)
                return None
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            logger.error("Exchange error fetching price for %s: %s", pair, e)
            return None
        except Exception as e:
            logger.error("Unexpected error fetching price for %s: %s", pair, e)
            return None

    def _calculate_current_values(
        self, holdings: Dict[str, Decimal]
    ) -> Tuple[Optional[Dict[str, Decimal]], Optional[Decimal]]:
        """Calculate value of each holding in quote currency and total portfolio value, using Decimals."""
        asset_values = {}
        total_value = Decimal("0.0")
        holdings_upper = {k.upper(): v for (k, v) in holdings.items()}
        quote_upper = self.quote_currency.upper()
        for asset, amount in holdings_upper.items():
            if not isinstance(amount, Decimal):
                amount = Decimal("0")
            if amount <= Decimal("1e-9"):
                asset_values[asset] = Decimal("0.0")
                continue
            if asset == quote_upper:
                value = amount
                asset_values[asset] = value
                total_value += value
            else:
                pair = f"{asset}/{quote_upper}"
                price = self._get_current_price_from_exchange(pair)
                if price is not None and price > Decimal("1e-9"):
                    value = amount * price
                    asset_values[asset] = value
                    total_value += value
                else:
                    logger.warning(
                        "Could not get price for %s, cannot value %s holding (%.8f). Excluding from total value.",
                        pair,
                        asset,
                        amount,
                    )
                    asset_values[asset] = Decimal("0.0")
        for base in self.base_currencies:
            asset_upper = base.upper()
            if asset_upper not in asset_values:
                asset_values[asset_upper] = Decimal("0.0")
        if total_value <= Decimal("1e-9") and any(
            (h > Decimal("1e-9") for (a, h) in holdings_upper.items() if a != quote_upper)
        ):
            logger.error(
                "Failed to calculate a positive total portfolio value despite non-quote holdings."
            )
            return (asset_values, None)
        return (asset_values, total_value)

    def _calculate_atr(
        self, pair: str, period: int, market_state: Dict[str, Dict[str, Any]]
    ) -> Optional[float]:
        """Calculate Average True Range (ATR) using OHLCV from market_state. Returns float."""
        pair_state = market_state.get(pair, {})
        ohlcv_data = pair_state.get("ohlcv")
        if not isinstance(ohlcv_data, list) or len(ohlcv_data) < period + 1:
            logger.warning(
                "Insufficient OHLCV data for ATR %s (Need %s, have %s)",
                pair,
                period + 1,
                len(ohlcv_data) if isinstance(ohlcv_data, list) else 0,
            )
            return None
        try:
            (highs, lows, closes) = ([], [], [])
            for c in ohlcv_data:
                try:
                    if isinstance(c, dict):
                        (h, l, cl) = (float(c["high"]), float(c["low"]), float(c["close"]))
                    elif isinstance(c, list) and len(c) > 4:
                        (h, l, cl) = (float(c[2]), float(c[3]), float(c[4]))
                    else:
                        continue
                    highs.append(h)
                    lows.append(l)
                    closes.append(cl)
                except (ValueError, TypeError, KeyError, IndexError):
                    continue
            if len(highs) < period + 1:
                logger.warning("Insufficient valid candles for ATR %s.", pair)
                return None
            highs_arr: Any = np.array(highs)
            lows_arr: Any = np.array(lows)
            closes_arr: Any = np.array(closes)
            high_low: Any = highs_arr - lows_arr
            high_close: Any = np.abs(highs_arr[1:] - closes_arr[:-1])
            low_close: Any = np.abs(lows_arr[1:] - closes_arr[:-1])
            tr: Any = np.full_like(highs_arr, np.nan)
            tr[1:] = np.maximum(high_low[1:], high_close)
            tr[1:] = np.maximum(tr[1:], low_close)
            atr_series: Any = np.full_like(closes_arr, np.nan)
            if len(tr[1 : period + 1]) > 0:
                first_atr = np.nanmean(tr[1 : period + 1])
                if not np.isnan(first_atr):
                    atr_series[period] = first_atr
                    alpha = 1.0 / period
                    for i in range(period + 1, len(closes)):
                        if not np.isnan(tr[i]) and (not np.isnan(atr_series[i - 1])):
                            atr_series[i] = alpha * tr[i] + (1 - alpha) * atr_series[i - 1]
                        elif not np.isnan(atr_series[i - 1]):
                            atr_series[i] = atr_series[i - 1]
            last_atr = atr_series[-1]
            return float(last_atr) if not np.isnan(last_atr) else None
        except Exception as e:
            logger.error("Error calculating ATR for %s: %s", pair, e, exc_info=True)
            return None

    def _calculate_dynamic_threshold(
        self, total_portfolio_value: Decimal, market_state: Dict[str, Dict[str, Any]]
    ) -> Optional[float]:
        """Calculate dynamic rebalancing threshold based on average portfolio volatility. Returns float."""
        logger.debug("Calculating dynamic rebalance threshold...")
        if total_portfolio_value <= Decimal("1e-9"):
            return None
        volatilities = []
        for asset in self.base_currencies:
            pair = f"{asset}/{self.quote_currency}"
            atr = self._calculate_atr(pair, self.volatility_lookback, market_state)
            ticker = market_state.get(pair, {}).get("ticker", {})
            current_price_float = None
            if isinstance(ticker, dict) and ticker.get("last") is not None:
                try:
                    current_price_float = float(ticker["last"])
                except (ValueError, TypeError):
                    pass
            if current_price_float is None:
                current_price_dec = self._get_current_price_from_exchange(pair)
                if current_price_dec is not None:
                    try:
                        current_price_float = float(current_price_dec)
                    except (ValueError, TypeError):
                        current_price_float = None
            if (
                atr is not None
                and current_price_float is not None
                and (current_price_float > 1e-09)
            ):
                try:
                    volatilities.append(atr / current_price_float)
                except ZeroDivisionError:
                    pass
            else:
                logger.warning(
                    "Could not calculate volatility for %s, skipping for dynamic threshold (ATR=%s, Price=%s)",
                    pair,
                    atr,
                    current_price_float,
                )
        if not volatilities:
            logger.warning("No asset volatilities calculated. Cannot determine dynamic threshold.")
            return None
        avg_volatility = float(np.mean(volatilities))
        volatility_sensitivity = 1.0
        dynamic_thresh = self.rebalance_threshold * (1.0 + avg_volatility * volatility_sensitivity)
        min_thresh = self.rebalance_threshold * 0.5
        max_thresh = self.rebalance_threshold * 2.0
        final_threshold = max(min_thresh, min(max_thresh, dynamic_thresh))
        logger.debug(
            "Average Volatility (ATR %%): %.2f%%. Dynamic Threshold: %.2f%% (Base: %.2f%%)",
            avg_volatility * 100,
            final_threshold * 100,
            self.rebalance_threshold * 100,
        )
        return final_threshold

    def _calculate_returns(
        self, pair: str, period: int, market_state: Dict[str, Dict[str, Any]]
    ) -> Optional[np.ndarray]:
        """Helper to calculate historical log returns from OHLCV data in market_state."""
        pair_state = market_state.get(pair, {})
        ohlcv_data = pair_state.get("ohlcv")
        if not isinstance(ohlcv_data, list) or len(ohlcv_data) < period + 1:
            logger.warning(
                "Insufficient OHLCV for returns %s (Need %s, have %s)",
                pair,
                period + 1,
                len(ohlcv_data) if isinstance(ohlcv_data, list) else 0,
            )
            return None
        try:
            closes = []
            for c in ohlcv_data[-(period + 1) :]:
                try:
                    if isinstance(c, dict):
                        cl = float(c["close"])
                    elif isinstance(c, list) and len(c) > 4:
                        cl = float(c[4])
                    else:
                        continue
                    closes.append(cl)
                except (ValueError, TypeError, KeyError, IndexError):
                    continue
            if len(closes) < period + 1:
                logger.warning("Insufficient valid closes for returns %s.", pair)
                return None
            closes_arr: Any = np.array(closes)
            if np.any(closes_arr <= 1e-09):
                logger.warning(
                    "Non-positive close prices found for %s, cannot calculate log returns.", pair
                )
                return None
            returns: Any = np.log(closes_arr[1:] / closes_arr[:-1])
            return returns
        except (FloatingPointError, RuntimeWarning) as e:
            logger.error("Numerical error calculating returns for %s: %s", pair, e, exc_info=True)
            return None
        except Exception as e:
            logger.error("Unexpected error calculating returns for %s: %s", pair, e, exc_info=True)
            return None

    def _calculate_transaction_cost(self, trade_value: Decimal) -> Decimal:
        """Calculate estimated transaction cost using config fee rate (using Decimal)."""
        if not isinstance(trade_value, Decimal):
            try:
                trade_value = Decimal(str(trade_value))
            except InvalidOperation:
                return Decimal("0.0")
        try:
            return self.fixed_cost + abs(trade_value) * self.variable_cost_rate
        except Exception as e:
            logger.error("Error calculating transaction cost for value %s: %s", trade_value, e)
            return Decimal("0.0")

    def _simulate_trade_execution(self, action: str, asset: str, trade_value: float) -> dict:
        """Simulate trade execution using the exchange client's order mechanism."""
        logger.debug(
            "Simulating trade execution via exchange client: %s %s for value %.2f",
            action,
            asset,
            trade_value,
        )
        pair = f"{asset}/{self.quote_currency}"
        price_dec = self._get_current_price_from_exchange(pair)
        if price_dec is None or price_dec <= Decimal("1e-9"):
            error_msg = f"Cannot simulate trade for {pair}: Invalid price ({price_dec})"
            logger.error(error_msg)
            return {
                "executed": False,
                "status": "rejected",
                "reason": error_msg,
                "info": {"error": error_msg},
            }
        try:
            quantity_dec = abs(Decimal(str(trade_value))) / price_dec
            quantity_float = float(quantity_dec)
        except (DivisionByZero, InvalidOperation) as e:
            error_msg = f"Cannot simulate trade for {pair}: Quantity calculation error ({e})"
            logger.error(error_msg)
            return {
                "executed": False,
                "status": "rejected",
                "reason": error_msg,
                "info": {"error": error_msg},
            }
        if quantity_float <= 0:
            error_msg = f"Cannot simulate trade for {pair}: Calculated quantity zero or negative ({quantity_float:.8f})"
            logger.error(error_msg)
            return {
                "executed": False,
                "status": "rejected",
                "reason": error_msg,
                "info": {"error": error_msg},
            }
        try:
            if action == "BUY":
                order_result = self.exchange_client.create_market_buy_order(pair, quantity_float)
            elif action == "SELL":
                order_result = self.exchange_client.create_market_sell_order(pair, quantity_float)
            else:
                error_msg = f"Unknown action '{action}' for trade simulation."
                logger.error(error_msg)
                return {
                    "executed": False,
                    "status": "rejected",
                    "reason": error_msg,
                    "info": {"error": error_msg},
                }
            if not isinstance(order_result, dict):
                logger.error(
                    "Exchange client simulation returned non-dict result for %s %s: %s",
                    action,
                    asset,
                    order_result,
                )
                return {
                    "executed": False,
                    "status": "failed",
                    "reason": "Invalid simulation response type",
                    "info": {"error": "Invalid simulation response type"},
                }
            logger.debug(
                "Exchange client simulation result for %s %s: %s",
                action,
                asset,
                order_result.get("status", "N/A"),
            )
            return order_result
        except (
            ccxt.NetworkError,
            ccxt.ExchangeError,
            ccxt.InvalidOrder,
            ccxt.InsufficientFunds,
        ) as exchange_e:
            error_msg = f"Exchange error during simulation {action} {asset}: {exchange_e}"
            logger.error(error_msg, exc_info=False)
            return {
                "executed": False,
                "status": "failed",
                "reason": error_msg,
                "info": {"error": str(exchange_e)},
            }
        except Exception as e:
            error_msg = (
                f"Unexpected exception during exchange client simulation for {action} {asset}: {e}"
            )
            logger.error(error_msg, exc_info=True)
            return {
                "executed": False,
                "status": "failed",
                "reason": error_msg,
                "info": {"error": str(e)},
            }

    def _perform_risk_assessment(
        self,
        holdings: Dict[str, Decimal],
        asset_values: Optional[Dict[str, Decimal]],
        total_value: Decimal,
        market_state: Dict[str, Dict[str, Any]],
    ) -> None:
        """Perform and log risk assessment using config parameters and market_state."""
        logger.info("--- Portfolio Risk Assessment ---")
        if total_value <= Decimal("1e-9"):
            logger.warning("Total portfolio value is zero, cannot perform risk assessment.")
            return
        var_loss = 0.0
        valid_assets_for_var = 0
        try:
            timeframe = self.config.bot.timeframe.lower()
            time_unit = timeframe[-1]
            time_val = int(timeframe[:-1])
            if time_unit == "m":
                periods_per_day = 24 * 60 / time_val
            elif time_unit == "h":
                periods_per_day = 24 / time_val
            elif time_unit == "d":
                periods_per_day = 1 / time_val
            else:
                periods_per_day = 24
            var_period = math.ceil(self.var_time_horizon * periods_per_day)
            if var_period <= 1:
                logger.warning(
                    "Calculated VaR period too short (%s). Using minimum of 2.", var_period
                )
                var_period = 2
        except (ValueError, TypeError, ZeroDivisionError):
            logger.error("Invalid timeframe in config for VaR period calculation. Using default.")
            var_period = self.var_time_horizon * 24
        logger.debug(
            "Calculating Historical VaR for period: %s (%s days)", var_period, self.var_time_horizon
        )
        portfolio_returns = None
        for asset, value_dec in asset_values.items():
            asset_upper = asset.upper()
            if asset_upper == self.quote_currency.upper() or value_dec <= Decimal("1e-9"):
                continue
            pair = f"{asset_upper}/{self.quote_currency.upper()}"
            returns = self._calculate_returns(pair, var_period, market_state)
            if returns is not None and len(returns) == var_period:
                asset_weight = float(value_dec / total_value) if total_value > 0 else 0.0
                if portfolio_returns is None:
                    portfolio_returns = returns * asset_weight
                else:
                    min_len = min(len(portfolio_returns), len(returns))
                    portfolio_returns = (
                        portfolio_returns[:min_len] + returns[:min_len] * asset_weight
                    )
                valid_assets_for_var += 1
            else:
                logger.warning("Insufficient/invalid return data for %s for VaR calculation.", pair)
        if portfolio_returns is not None and valid_assets_for_var > 0:
            try:
                alpha = 1.0 - self.var_confidence_level
                sorted_returns = np.sort(portfolio_returns[~np.isnan(portfolio_returns)])
                if len(sorted_returns) > 0:
                    var_return = np.percentile(sorted_returns, alpha * 100)
                    var_loss_dec = (
                        -Decimal(str(var_return)) * total_value
                        if var_return < 0
                        else Decimal("0.0")
                    )
                    var_loss = float(var_loss_dec)
                    logger.info(
                        "Historical VaR (%.0f%%, %sd): Loss <= $%.2f (Based on %s assets)",
                        self.var_confidence_level * 100,
                        self.var_time_horizon,
                        var_loss,
                        valid_assets_for_var,
                    )
                else:
                    logger.warning(
                        "Could not calculate Historical VaR: No valid returns after NaN removal."
                    )
            except (IndexError, ValueError) as np_e:
                logger.error("Numpy error calculating Historical VaR percentile: %s", np_e)
        else:
            logger.warning(
                "Could not calculate Historical VaR due to insufficient data for weighted returns."
            )
        logger.info("Stress Tests:")
        if not isinstance(self.stress_test_scenarios, dict):
            logger.warning("Stress test scenarios configuration is invalid (not a dict). Skipping.")
        else:
            for scenario, shock_multiplier_obj in self.stress_test_scenarios.items():
                try:
                    shock_multiplier = Decimal(str(shock_multiplier_obj))
                    stressed_value = total_value * (Decimal("1.0") + shock_multiplier)
                    loss = total_value - stressed_value
                    logger.info(
                        "  - %s (%f%%): Est. Loss ~$%.2f (New Value ~$%.2f)",
                        scenario,
                        shock_multiplier * 100,
                        loss,
                        stressed_value,
                    )
                except (InvalidOperation, TypeError, ValueError) as dec_err:
                    logger.warning(
                        "Invalid stress test shock multiplier for scenario '%s': %s (%s). Skipping.",
                        scenario,
                        shock_multiplier_obj,
                        dec_err,
                    )
        if self.use_advanced_var:
            logger.info("Calculating Monte Carlo VaR...")
            mc_simulations = 1000
            simulated_portfolio_values = np.zeros(mc_simulations)
            mc_assets_used = 0
            try:
                returns_dict = {}
                std_devs = {}
                means = {}
                mc_calc_period = var_period * 5
                for asset, value_dec in asset_values.items():
                    asset_upper = asset.upper()
                    if asset_upper == self.quote_currency.upper() or value_dec <= Decimal("1e-9"):
                        continue
                    pair = f"{asset_upper}/{self.quote_currency.upper()}"
                    returns_data = self._calculate_returns(pair, mc_calc_period, market_state)
                    if returns_data is not None and len(returns_data) > 1:
                        valid_returns = returns_data[~np.isnan(returns_data)]
                        if len(valid_returns) > 1:
                            means[asset_upper] = np.mean(valid_returns)
                            std_devs[asset_upper] = np.std(valid_returns)
                            mc_assets_used += 1
                        else:
                            logger.warning("Insufficient valid returns for MC VaR on %s.", pair)
                    else:
                        logger.warning("Could not get returns for MC VaR on %s.", pair)
                if mc_assets_used == 0:
                    raise ValueError("No valid asset returns data for Monte Carlo VaR.")
                quote_balance = holdings.get(self.quote_currency.upper(), Decimal("0.0"))
                simulated_portfolio_values = np.full(mc_simulations, float(quote_balance))
                for asset, value_dec in asset_values.items():
                    asset_upper = asset.upper()
                    if (
                        asset_upper == self.quote_currency.upper()
                        or asset_upper not in means
                        or asset_upper not in std_devs
                    ):
                        continue
                    mu = means[asset_upper]
                    sigma = std_devs[asset_upper]
                    T = self.var_time_horizon
                    drift = (mu - 0.5 * sigma**2) * T
                    volatility = sigma * np.sqrt(T)
                    random_shocks = np.random.normal(0, 1, mc_simulations)
                    simulated_end_returns = np.exp(drift + volatility * random_shocks)
                    simulated_end_values = float(value_dec) * simulated_end_returns
                    simulated_portfolio_values += simulated_end_values
                simulated_losses = float(total_value) - simulated_portfolio_values
                sorted_losses = np.sort(simulated_losses)
                mc_var_loss = np.percentile(sorted_losses, self.var_confidence_level * 100)
                logger.info(
                    "Monte Carlo VaR (%.0f%%, %sd): Loss <= $%.2f (Based on %s assets, Note: Ignores correlations)",
                    self.var_confidence_level * 100,
                    self.var_time_horizon,
                    mc_var_loss,
                    mc_assets_used,
                )
            except (ValueError, TypeError, FloatingPointError, IndexError) as mc_e:
                logger.error("Monte Carlo VaR calculation failed: %s", mc_e, exc_info=True)
        logger.info("---------------------------------")

    def _log_skipped_trades(self, skipped_trades_list: list) -> None:
        """Logs skipped portfolio rebalance trades to a file atomically."""
        if not skipped_trades_list:
            return
        temp_file = self.skipped_trades_log_file.with_suffix(f".tmp_{time.time_ns()}")
        try:
            log_list = []
            if self.skipped_trades_log_file.exists():
                try:
                    with self.skipped_trades_log_file.open("r", encoding="utf-8") as f:
                        content = f.read()
                        if content.strip():
                            loaded_list = json.loads(content)
                            if isinstance(loaded_list, list):
                                log_list = loaded_list
                except json.JSONDecodeError:
                    log_list = []
                except OSError:
                    log_list = []
            log_list.extend(skipped_trades_list)
            max_log_entries = self.config.get("logging.max_skipped_trade_log_entries", 5000)
            if len(log_list) > max_log_entries:
                log_list = log_list[-max_log_entries:]
            with temp_file.open("w", encoding="utf-8") as f:
                json.dump(log_list, f, indent=2)
            temp_file.replace(self.skipped_trades_log_file)
            logger.info(
                "Logged %s skipped rebalance trades to %s",
                len(skipped_trades_list),
                self.skipped_trades_log_file.name,
            )
        except (OSError, TypeError, ValueError) as e:
            logger.error("Failed to log skipped trades: %s", e, exc_info=True)
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass

    def _check_for_anomalous_patterns(self, skipped_trades_list: list) -> None:
        """Check for anomalies in skipped trades (e.g., repeated skips for same reason)."""
        if not skipped_trades_list:
            return
        try:
            recent_skips = skipped_trades_list[-20:]
            reasons_count = {}
            for trade in recent_skips:
                asset = trade.get("asset", "UNKNOWN")
                reason = trade.get("reason", "UNKNOWN_REASON")
                key = (asset, reason)
                reasons_count[key] = reasons_count.get(key, 0) + 1
            for (asset, reason), count in reasons_count.items():
                if count >= 3:
                    logger.warning(
                        "Anomaly Detected: Asset '%s' skipped %s times recently for reason: '%s'",
                        asset,
                        count,
                        reason,
                    )
            logger.debug("Checked for anomalies in %s recent skipped trades.", len(recent_skips))
        except Exception as e:
            logger.error("Error checking for anomalies in skipped trades: %s", e, exc_info=True)
