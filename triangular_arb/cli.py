"""
CLI entry point.

Handles argument parsing, config loading, and signal handling
for graceful shutdown. The engine is started via `asyncio.run()`.
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from pathlib import Path

from triangular_arb import __version__
from triangular_arb.config import load_config
from triangular_arb.engine import Engine


def main() -> None:
    """Entry point for the triangular-arb CLI."""
    parser = argparse.ArgumentParser(
        prog="triangular-arb",
        description="High-frequency triangular arbitrage engine for cryptocurrency exchanges",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to config file (default: config.yaml)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=None,
        help="Override config: enable paper trading mode",
    )

    args = parser.parse_args()

    # Load and validate config
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Config validation error: {e}", file=sys.stderr)
        sys.exit(1)

    # CLI overrides
    if args.dry_run is not None:
        config = config.model_copy(update={"dry_run": args.dry_run})

    # Create engine
    engine = Engine(config)

    # Signal handling for graceful shutdown
    loop = asyncio.new_event_loop()

    background_tasks: set = set()  # type: ignore[type-arg]

    def _shutdown(sig: signal.Signals) -> None:
        print(f"\nReceived {sig.name}, shutting down...")
        task = loop.create_task(engine.stop())
        background_tasks.add(task)
        task.add_done_callback(background_tasks.discard)

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _shutdown, sig)

    try:
        loop.run_until_complete(engine.start())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
