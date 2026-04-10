#!/usr/bin/env python3
"""
CLI entry point for the DeerFlow Day Trading Scorer.

Usage:
    python run_scorer.py
    python run_scorer.py --signals-dir /path/to/signals
    python run_scorer.py --dry-run          # score but do not write output
    python run_scorer.py --output /custom/path/trade_signal.json

Exit codes:
    0  — signals written (or dry-run completed)
    1  — no signals met the entry threshold
    2  — error reading signal bus
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from repo root without installing as a package
sys.path.insert(0, str(Path(__file__).parent))

from tools.day_trading_scorer import (
    SIGNALS_DIR,
    OUTPUT_FILE,
    run_scoring,
    write_trade_signal,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score day-trading candidates from the DeerFlow signal bus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--signals-dir",
        type=Path,
        default=SIGNALS_DIR,
        help=f"Signal bus directory (default: {SIGNALS_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override output path for trade_signal.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run scoring but do not write trade_signal.json",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Print full JSON result to stdout instead of a human summary",
    )
    args = parser.parse_args()

    try:
        result = run_scoring(args.signals_dir)
    except Exception as exc:
        print(f"ERROR reading signal bus: {exc}", file=sys.stderr)
        return 2

    signals = result["signals"]

    if args.json_output:
        print(json.dumps(result, indent=2))
    else:
        print(f"Generated at : {result['generated_at']}")
        print(f"Signals found: {len(signals)}")
        if signals:
            print()
            print(f"{'Rank':<5} {'Ticker':<8} {'Dir':<6} {'Score':>7}  {'Size%':>6}  Reasons")
            print("-" * 72)
            for s in signals:
                reasons_str = ", ".join(s["reasons"][:3])
                if len(s["reasons"]) > 3:
                    reasons_str += f" (+{len(s['reasons']) - 3})"
                print(
                    f"{s['rank']:<5} {s['ticker']:<8} {s['direction']:<6} "
                    f"{s['conviction_score']:>7.1f}  {s['size_pct']*100:>5.1f}%  {reasons_str}"
                )
        else:
            print("No tickers met the entry threshold.")

    if not signals:
        return 1

    if not args.dry_run:
        output_path = args.output or (args.signals_dir / "trade_signal.json")
        written = write_trade_signal(result, output_path)
        if not args.json_output:
            print()
            print(f"Wrote: {written}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
