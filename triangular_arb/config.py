"""
Configuration management via Pydantic validated YAML.

Config is loaded once at startup and immutable thereafter.
No more Python files with mutable globals acting as config.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class ExchangeConfig(BaseModel):
    """Connection settings for a single exchange."""
    exchange_id: str
    api_key: str = ""
    api_secret: str = ""
    passphrase: str = ""
    testnet: bool = False
    rate_limit: bool = True
    timeout_ms: int = 30_000


class RiskConfig(BaseModel):
    """Risk management parameters."""
    max_position_pct: Decimal = Field(
        default=Decimal("0.5"),
        description="Max fraction of balance to use per trade",
    )
    min_profit_bps: Decimal = Field(
        default=Decimal("5"),
        description="Minimum net profit in basis points to execute",
    )
    max_daily_loss_pct: Decimal = Field(
        default=Decimal("5"),
        description="Max daily drawdown as % of starting balance before circuit breaker",
    )
    max_consecutive_losses: int = 5
    max_open_triangles: int = 3
    stale_book_ms: int = 2_000


class ExecutionConfig(BaseModel):
    """Order execution parameters."""
    use_limit_orders: bool = True
    limit_order_timeout_s: float = 2.0
    max_retries: int = 3
    max_slippage_bps: Decimal = Field(
        default=Decimal("10"),
        description="Max acceptable slippage per leg in basis points",
    )
    order_book_depth: int = 10


class ScannerConfig(BaseModel):
    """Triangle discovery and scanning parameters."""
    base_currencies: list[str] = Field(default_factory=lambda: ["ETH", "BTC", "USDT"])
    scan_interval_ms: int = 500
    min_book_levels: int = 3
    quote_currencies: list[str] = Field(
        default_factory=lambda: ["ETH", "BTC", "USDT", "BNB"],
    )


class LoggingConfig(BaseModel):
    """Structured logging config."""
    level: str = "INFO"
    json_output: bool = True
    log_dir: str = "logs"


class Config(BaseModel):
    """
    Top-level configuration. Validated at startup — if the config is invalid,
    the process refuses to start rather than silently misbehaving.
    """
    exchange: ExchangeConfig
    risk: RiskConfig = Field(default_factory=RiskConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    scanner: ScannerConfig = Field(default_factory=ScannerConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    dry_run: bool = Field(
        default=True,
        description="Paper trading mode. Set to false for live execution.",
    )


def load_config(path: Path | str = "config.yaml") -> Config:
    """
    Load and validate configuration from a YAML file.

    Fails fast with a clear error message if the config is invalid,
    rather than discovering misconfiguration mid-trade.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            f"Copy config.example.yaml to {path} and fill in your API keys."
        )

    with open(path) as f:
        raw = yaml.safe_load(f)

    return Config.model_validate(raw)
