"""
Unit tests for news_veto pre-filter (Wire #8) in day_trading_scorer.

Run with: pytest tools/test_news_veto.py -v
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

# Stub kronos_client before import so gate is always pass.
import types as _types
_stub = _types.ModuleType("kronos_client")
_stub.get_kronos_gate_pass = lambda ticker, direction: True
_stub.get_kronos_gate_reason = lambda ticker, direction: "stub"
sys.modules["kronos_client"] = _stub

# Stub langchain_core.tools.tool decorator.
_lc_core = _types.ModuleType("langchain_core")
_lc_tools = _types.ModuleType("langchain_core.tools")
def _stub_tool(*args, **kwargs):
    def _decorator(fn):
        return fn
    if args and callable(args[0]):
        return _decorator(args[0])
    return _decorator
_lc_tools.tool = _stub_tool
sys.modules.setdefault("langchain_core", _lc_core)
sys.modules["langchain_core.tools"] = _lc_tools

from tools import day_trading_scorer as dts  # noqa: E402


def _write_veto(td: Path, vetoes: list[dict]) -> Path:
    p = td / "news_veto.json"
    p.write_text(json.dumps({"generated_at": "now", "vetoes": vetoes}), encoding="utf-8")
    return p


def _write_base_signals(td: Path, tickers: list[str]) -> None:
    """Write minimal kronos + whale + options fixtures for a set of tickers."""
    td.mkdir(parents=True, exist_ok=True)
    (td / "kronos_predictions.json").write_text(json.dumps({
        t: {"direction": "bullish", "win_probability": 0.9} for t in tickers
    }), encoding="utf-8")
    (td / "whale_signals.json").write_text(json.dumps({
        "signals": [{"ticker": t, "score": 90, "signal_type": "call"} for t in tickers],
    }), encoding="utf-8")
    (td / "options_positions.json").write_text(json.dumps({"positions": []}), encoding="utf-8")


class TestReadNewsVeto(unittest.TestCase):
    def test_missing_file_returns_empty(self):
        with tempfile.TemporaryDirectory() as td:
            result = dts.read_news_veto(Path(td))
            self.assertEqual(result, set())

    def test_ticker_vetoes_returned_uppercased(self):
        with tempfile.TemporaryDirectory() as td:
            _write_veto(Path(td), [
                {"ticker": "nvda", "reason": "SEC probe"},
                {"ticker": "AAPL", "reason": "earnings halt"},
            ])
            result = dts.read_news_veto(Path(td))
            self.assertEqual(result, {"NVDA", "AAPL"})

    def test_category_only_veto_not_returned(self):
        with tempfile.TemporaryDirectory() as td:
            _write_veto(Path(td), [
                {"category": "biotech", "reason": "FDA uncertainty"},
            ])
            result = dts.read_news_veto(Path(td))
            self.assertEqual(result, set())

    def test_stale_file_returns_empty_with_warning(self):
        with tempfile.TemporaryDirectory() as td:
            p = _write_veto(Path(td), [{"ticker": "NVDA", "reason": "old"}])
            # Backdate mtime by 3 hours
            stale_mtime = time.time() - 3 * 3600
            import os; os.utime(p, (stale_mtime, stale_mtime))
            with self.assertLogs("root", level="WARNING") as cm:
                result = dts.read_news_veto(Path(td))
            self.assertEqual(result, set())
            self.assertTrue(any("stale" in m or "old" in m or "threshold" in m for m in cm.output),
                            f"Expected stale warning; got: {cm.output}")

    def test_corrupt_file_returns_empty(self):
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "news_veto.json").write_text("{not valid json", encoding="utf-8")
            result = dts.read_news_veto(Path(td))
            self.assertEqual(result, set())


class TestNewsVetoPreFilter(unittest.TestCase):
    """Integration tests: vetoed tickers must not appear in run_scoring output."""

    def test_vetoed_ticker_dropped_from_output(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            _write_base_signals(tdp, ["NVDA", "TSLA"])
            _write_veto(tdp, [{"ticker": "NVDA", "reason": "SEC probe"}])
            result = dts.run_scoring(signals_dir=tdp)
            tickers = {s["ticker"] for s in result["signals"]}
            self.assertNotIn("NVDA", tickers, "NVDA should be vetoed")
            self.assertIn("TSLA", tickers, "TSLA should still score")

    def test_category_veto_no_ticker_match_no_drops(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            _write_base_signals(tdp, ["NVDA", "TSLA"])
            # Only category veto — no ticker-specific entries
            _write_veto(tdp, [{"category": "semiconductor", "reason": "sector headwind"}])
            result = dts.run_scoring(signals_dir=tdp)
            tickers = {s["ticker"] for s in result["signals"]}
            self.assertIn("NVDA", tickers)
            self.assertIn("TSLA", tickers)

    def test_missing_veto_file_no_drops(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            _write_base_signals(tdp, ["NVDA"])
            # No news_veto.json written
            with self.assertLogs("root", level="WARNING") as cm:
                result = dts.run_scoring(signals_dir=tdp)
            tickers = {s["ticker"] for s in result["signals"]}
            self.assertIn("NVDA", tickers)
            self.assertTrue(any("news_veto" in m for m in cm.output),
                            f"Expected news_veto warning; got: {cm.output}")

    def test_stale_veto_file_no_drops(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            _write_base_signals(tdp, ["NVDA"])
            p = _write_veto(tdp, [{"ticker": "NVDA", "reason": "old veto"}])
            import os
            stale_mtime = time.time() - 3 * 3600
            os.utime(p, (stale_mtime, stale_mtime))
            result = dts.run_scoring(signals_dir=tdp)
            tickers = {s["ticker"] for s in result["signals"]}
            self.assertIn("NVDA", tickers, "Stale veto must not drop ticker")


if __name__ == "__main__":
    unittest.main()
