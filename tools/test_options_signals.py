"""
Unit tests for options_signals integration in day_trading_scorer.

Run with: pytest tools/test_options_signals.py -v
"""
from __future__ import annotations

import json
import sys
import time
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import types as _types
_stub = _types.ModuleType("kronos_client")
_stub.get_kronos_gate_pass = lambda ticker, direction: True
_stub.get_kronos_gate_reason = lambda ticker, direction: "stub"
sys.modules["kronos_client"] = _stub

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

from tools import day_trading_scorer as dts


KRONOS_BULLISH = {"direction": "bullish", "win_probability": 0.6}


def _write_options_signals(td: Path, tickers: list[dict]) -> Path:
    p = td / "options_signals.json"
    p.write_text(
        json.dumps({"updated_at": "2026-05-19T00:00:00+00:00", "tickers": tickers}),
        encoding="utf-8",
    )
    return p


class TestReadOptionsSignals(unittest.TestCase):
    def test_missing_file_returns_empty(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(dts.read_options_signals(Path(td)), {})

    def test_valid_call_sweep_indexed_by_upper_ticker(self):
        with tempfile.TemporaryDirectory() as td:
            _write_options_signals(Path(td), [
                {"ticker": "aapl", "sweep_type": "call", "score": 2, "computed_at": "2026-05-19T00:00:00+00:00"},
            ])
            result = dts.read_options_signals(Path(td))
        self.assertIn("AAPL", result)
        self.assertEqual(result["AAPL"]["sweep_type"], "call")

    def test_valid_put_sweep(self):
        with tempfile.TemporaryDirectory() as td:
            _write_options_signals(Path(td), [
                {"ticker": "SPY", "sweep_type": "put", "score": -2, "computed_at": "2026-05-19T00:00:00+00:00"},
            ])
            result = dts.read_options_signals(Path(td))
        self.assertEqual(result["SPY"]["sweep_type"], "put")

    def test_stale_file_returns_empty(self):
        with tempfile.TemporaryDirectory() as td:
            p = _write_options_signals(Path(td), [
                {"ticker": "AAPL", "sweep_type": "call", "score": 2, "computed_at": "x"},
            ])
            # backdate mtime by 5 hours
            old_mtime = time.time() - (5 * 3600)
            import os
            os.utime(p, (old_mtime, old_mtime))
            result = dts.read_options_signals(Path(td))
        self.assertEqual(result, {})

    def test_corrupt_file_returns_empty(self):
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "options_signals.json").write_text("NOT JSON", encoding="utf-8")
            result = dts.read_options_signals(Path(td))
        self.assertEqual(result, {})

    def test_empty_tickers_list(self):
        with tempfile.TemporaryDirectory() as td:
            _write_options_signals(Path(td), [])
            result = dts.read_options_signals(Path(td))
        self.assertEqual(result, {})


class TestScoreTickerOptionsSweep(unittest.TestCase):
    def _base_kwargs(self):
        return dict(
            ticker="AAPL",
            kronos=KRONOS_BULLISH,
            whale=None,
            sentiment=None,
            market_regime={},
            already_held=False,
        )

    def test_call_sweep_adds_bonus(self):
        sweep = {"sweep_type": "call", "score": 2}
        score, reasons = dts.score_ticker(
            **self._base_kwargs(), options_sweep=sweep
        )
        base_score, _ = dts.score_ticker(**self._base_kwargs())
        self.assertAlmostEqual(score, base_score + dts.OPTIONS_SWEEP_CALL_BONUS)
        self.assertTrue(any("options_sweep_call_bonus" in r for r in reasons))

    def test_put_sweep_applies_penalty(self):
        sweep = {"sweep_type": "put", "score": -2}
        score, reasons = dts.score_ticker(
            **self._base_kwargs(), options_sweep=sweep
        )
        base_score, _ = dts.score_ticker(**self._base_kwargs())
        self.assertAlmostEqual(score, base_score - dts.OPTIONS_SWEEP_PUT_PENALTY)
        self.assertTrue(any("options_sweep_put_penalty" in r for r in reasons))

    def test_no_sweep_no_change(self):
        score_with, _ = dts.score_ticker(**self._base_kwargs(), options_sweep=None)
        score_without, _ = dts.score_ticker(**self._base_kwargs())
        self.assertAlmostEqual(score_with, score_without)

    def test_unknown_sweep_type_no_change(self):
        sweep = {"sweep_type": "unknown", "score": 0}
        score, reasons = dts.score_ticker(**self._base_kwargs(), options_sweep=sweep)
        base_score, _ = dts.score_ticker(**self._base_kwargs())
        self.assertAlmostEqual(score, base_score)
        self.assertFalse(any("options_sweep" in r for r in reasons))


if __name__ == "__main__":
    unittest.main()
