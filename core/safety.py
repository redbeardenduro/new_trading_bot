from datetime import datetime, timedelta
from decimal import Decimal

from core.alerting import Alerter


class SafetyGuard:

    def __init__(self, config) -> None:
        self.config = config
        self.alerter = Alerter(config)
        self.daily_loss_limit = Decimal(self.config.get("daily_loss_limit", "-1000"))
        self.max_per_asset_exposure = Decimal(self.config.get("max_per_asset_exposure", "5000"))
        self.max_clock_drift = timedelta(seconds=self.config.get("max_clock_drift_seconds", 1))
        self.realized_pnl_today = Decimal("0")
        self.unrealized_pnl = Decimal("0")
        self.last_reset_date = datetime.utcnow().date()
        self.circuit_breaker_tripped = False
        self.circuit_breaker_reason = None

    def check_daily_loss_limit(self) -> None:
        """Checks if the daily loss limit has been exceeded."""
        if datetime.utcnow().date() > self.last_reset_date:
            self.realized_pnl_today = Decimal("0")
            self.last_reset_date = datetime.utcnow().date()
        total_pnl = self.realized_pnl_today + self.unrealized_pnl
        if total_pnl <= self.daily_loss_limit:
            self.trip_circuit_breaker(
                f"Daily loss limit of {self.daily_loss_limit} exceeded. Current PnL: {total_pnl}"
            )
            return False
        return True

    def check_asset_exposure(self, asset_notional_value) -> None:
        """Checks if the notional value of an asset exceeds the maximum exposure."""
        if asset_notional_value > self.max_per_asset_exposure:
            self.trip_circuit_breaker(
                f"Max per-asset exposure of {self.max_per_asset_exposure} exceeded for an asset with notional value: {asset_notional_value}"
            )
            return False
        return True

    def check_clock_drift(self, exchange_timestamp) -> None:
        """Checks for clock drift between the system and the exchange."""
        system_timestamp = datetime.utcnow()
        exchange_dt = datetime.fromtimestamp(exchange_timestamp / 1000)
        drift = abs(system_timestamp - exchange_dt)
        if drift > self.max_clock_drift:
            self.trip_circuit_breaker(
                f"Clock drift of {drift} exceeds the maximum of {self.max_clock_drift}"
            )
            return False
        return True

    def trip_circuit_breaker(self, reason) -> None:
        """Trips the global circuit breaker."""
        self.circuit_breaker_tripped = True
        self.circuit_breaker_reason = reason
        print(f"CIRCUIT BREAKER TRIPPED: {reason}")
        self.alerter.send_alert("Circuit Breaker Tripped", reason)

    def can_trade(self) -> None:
        """Checks if trading is allowed."""
        if self.circuit_breaker_tripped:
            print(
                f"Trading is halted due to tripped circuit breaker: {self.circuit_breaker_reason}"
            )
            return False
        return True

    def update_pnl(self, realized_pnl, unrealized_pnl) -> None:
        """Updates the PnL values."""
        self.realized_pnl_today += realized_pnl
        self.unrealized_pnl = unrealized_pnl
