"""
Daily signal generation entry point.

Usage:
    py scripts/run_daily_signal.py                    # Generate signal (dry run)
    py scripts/run_daily_signal.py --date 2026-07-31  # Generate for specific date
    py scripts/run_daily_signal.py --execute          # Generate + execute in paper trader

Workflow:
    1. Generate trading signal from IC-weighted linear model
    2. (Optional) Execute orders in paper trader
    3. Run risk checks
    4. Print summary to console

The signal is always saved to signals/YYYY-MM-DD.json regardless of --execute.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_daily_signal")


def main():
    parser = argparse.ArgumentParser(
        description="Daily signal generation for IC-weighted linear model"
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Signal date (YYYY-MM-DD). Default: latest available data date.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute orders in paper trader (default: dry run only).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/conservative.yaml",
        help="Path to config YAML file.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="model_weights/ic_weights.json",
        help="Path to IC weights JSON file.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=" * 60)
    logger.info("  QUANT-STARTER: Daily Signal Generator")
    logger.info("  Date: %s", args.date or "(latest)")
    logger.info("  Mode: %s", "EXECUTE" if args.execute else "DRY RUN")
    logger.info("=" * 60)

    # ----------------------------------------------------------
    # Step 1: Generate Signal
    # ----------------------------------------------------------
    logger.info("\n[1/3] Generating signal...")

    from quant.production.signal_generator import SignalGenerator

    config_path = PROJECT_ROOT / args.config
    weights_path = PROJECT_ROOT / args.weights

    generator = SignalGenerator(
        config_path=config_path,
        weights_path=weights_path,
    )

    try:
        signal = generator.generate(as_of_date=args.date)
    except Exception as e:
        logger.error("Signal generation failed: %s", e, exc_info=True)
        sys.exit(1)

    # Print signal summary
    print("\n" + "=" * 60)
    print(f"  SIGNAL: {signal['date']}")
    print(f"  Model: {signal['model_version']}")
    print(f"  Rebalance Due: {signal['rebalance_due']}")
    print(f"  Stocks Scored: {signal['metadata']['n_stocks_scored']}")
    print(f"  Active Factors: {signal['metadata']['n_factors_active']}")
    print(f"  Top Factor: {signal['metadata']['top_factor']}")
    print("=" * 60)

    # Print target portfolio (top 10)
    print("\n  TARGET PORTFOLIO (Top 10):")
    print(f"  {'Rank':<5} {'Symbol':<8} {'Weight':<8} {'Score':<8}")
    print("  " + "-" * 30)
    for item in signal["target_portfolio"][:10]:
        print(f"  {item['rank']:<5} {item['symbol']:<8} "
              f"{item['weight']:<8.3f} {item['score']:<8.4f}")

    if len(signal["target_portfolio"]) > 10:
        print(f"  ... and {len(signal['target_portfolio']) - 10} more")

    # Print orders
    orders = signal["orders"]
    if orders:
        print(f"\n  ORDERS ({len(orders)} total):")
        for order in orders:
            icon = "+" if order["action"] == "BUY" else "-"
            print(f"  [{icon}] {order['action']:<4} {order['symbol']:<8} "
                  f"w={order['weight']:.3f}  ({order['reason']})")
    else:
        print("\n  No orders (portfolio unchanged)")

    # Risk check
    risk = signal["risk_check"]
    if risk["passed"]:
        print("\n  RISK CHECK: PASSED")
    else:
        print("\n  RISK CHECK: FAILED")
        for v in risk["violations"]:
            print(f"    ! {v}")

    # ----------------------------------------------------------
    # Step 2: Execute in Paper Trader (optional)
    # ----------------------------------------------------------
    if args.execute and orders:
        logger.info("\n[2/3] Executing in paper trader...")

        from quant.production.paper_trader import PaperTrader
        import pandas as pd

        trader = PaperTrader(initial_capital=1_000_000)

        # Load market data for execution
        market_data = _load_market_data(orders, signal["target_portfolio"])

        if market_data:
            results = trader.execute_orders(orders, market_data)

            # Print execution summary
            filled = [r for r in results if r.status == "filled"]
            rejected = [r for r in results if r.status == "rejected"]

            print(f"\n  EXECUTION RESULTS:")
            print(f"  Filled: {len(filled)}, Rejected: {len(rejected)}")

            for r in filled:
                print(f"    [OK] {r.action:<4} {r.symbol:<8} "
                      f"qty={r.qty} @ {r.price:.2f}")
            for r in rejected:
                print(f"    [X]  {r.action:<4} {r.symbol:<8} "
                      f"reason={r.reject_reason}")

            # Mark to market
            mtm = trader.mark_to_market(market_data)
            print(f"\n  PORTFOLIO VALUE: {mtm['total_equity']:,.0f}")
            print(f"  Cash: {mtm['cash']:,.0f}")
            print(f"  Holdings: {mtm['holdings_value']:,.0f}")
            print(f"  Daily Return: {mtm['daily_return']:.2%}")
        else:
            logger.warning("No market data available for execution")
    else:
        logger.info("\n[2/3] Skipping execution (dry run mode)")

    # ----------------------------------------------------------
    # Step 3: Risk Monitor
    # ----------------------------------------------------------
    logger.info("\n[3/3] Running risk monitor...")

    from quant.production.risk_monitor import RiskMonitor

    # Load portfolio state for risk monitoring
    state_file = PROJECT_ROOT / "paper_trade" / "portfolio.json"
    if state_file.exists():
        with open(state_file, "r", encoding="utf-8") as f:
            portfolio_state = json.load(f)
    else:
        portfolio_state = {"cash": 1_000_000, "positions": {}, "daily_equity": []}

    # Use empty benchmark returns if not available
    benchmark_returns = []

    monitor = RiskMonitor(portfolio_state, benchmark_returns)
    report = monitor.check_all()

    print(f"\n  RISK STATUS: {report.overall_status}")
    for check in report.checks:
        status_icon = {"OK": " ", "WARNING": "!", "CRITICAL": "X"}
        icon = status_icon.get(check["status"], "?")
        print(f"  [{icon}] {check['name']:<18} {check['message']}")

    if report.recommendations:
        print("\n  RECOMMENDATIONS:")
        for rec in report.recommendations:
            print(f"    -> {rec}")

    # ----------------------------------------------------------
    # Done
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    signal_path = PROJECT_ROOT / "signals" / f"{signal['date']}.json"
    print(f"  Signal saved: {signal_path}")
    print(f"  Risk report: {PROJECT_ROOT / 'paper_trade' / 'risk_report.json'}")
    print("=" * 60)


def _load_market_data(orders: list, target_portfolio: list) -> dict:
    """Load market data for all symbols in orders and target portfolio."""
    import pandas as pd

    data_cache_dir = PROJECT_ROOT / "data_cache"
    if not data_cache_dir.exists():
        logger.error("data_cache/ directory not found")
        return {}

    # Collect all symbols we need
    symbols = set()
    for order in orders:
        symbols.add(order["symbol"])
    for item in target_portfolio:
        symbols.add(item["symbol"])

    market_data = {}
    for symbol in symbols:
        path = data_cache_dir / f"{symbol}.parquet"
        if path.exists():
            try:
                df = pd.read_parquet(path)
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.sort_values("date").reset_index(drop=True)
                market_data[symbol] = df
            except Exception as e:
                logger.warning("Failed to load %s: %s", symbol, e)

    logger.info("Loaded market data for %d/%d symbols", len(market_data), len(symbols))
    return market_data


if __name__ == "__main__":
    main()
