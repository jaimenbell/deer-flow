"""
Unit tests for news_catalyst integration in day_trading_scorer.

Lives next to day_trading_scorer.py because deer-flow has no central tests dir
for tools/. Run with: pytest tools/test_news_catalyst.py -v
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

# Stub kronos_client BEFORE importing day_trading_scorer so the gate is a no-op.
import types as _types
_stub = _types.ModuleType("kronos_client")
_stub.get_kronos_gate_pass = lambda ticker, direction: True
_stub.get_kronos_gate_reason = lambda ticker, direction: "stub"
sys.modules["kronos_client"] = _stub

# Stub langchain_core.tools.tool decorator — the real one fails on parse_docstring
# under Python 3.14/current langchain. The scorer registers a tool at import time
# which would crash unrelated to this test's subject.
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


KRONOS_BULLISH = {"direction": "bullish", "win_probability": 0.6}


def _write_catalyst(td: Path, signals: list[dict]) -> Path:
    p = td / "news_catalyst.json"
    p.write_text(json.dumps({"generated_at": "now", "signals": signals}), encoding="utf-8")
    return p


class TestReadNewsCatalyst(unittest.TestCase):
    def test_missing_file_returns_empty(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(dts.read_news_catalyst(Path(td)), {})

    def test_valid_signals_indexed_by_upper_ticker(self):
        with tempfile.TemporaryDirectory() as td:
            _write_catalyst(Path(td), [
                {"ticker": "nvda", "catalyst_score": 75, "sentiment_label": "bullish"},
                {"ticker": "AAPL", "catalyst_score": 55, "sentiment_label": "bullish"},
            ])
            m = dts.read_news_catalyst(Path(td))
            self.assertIn("NVDA", m)
            self.assertIn("AAPL", m)


class TestScoreTickerCatalyst(unittest.TestCase):
    def _score(self, news_catalyst=None):
        return dts.score_ticker(
            ticker="NVDA",
            kronos=KRONOS_BULLISH,
            whale=None,
            sentiment=None,
            market_regime={},
            already_held=False,
            news_catalyst=news_catalyst,
        )

    def test_no_catalyst_baseline(self):
        score, reasons = self._score(None)
        self.assertFalse(any("news_catalyst" in r for r in reasons))

    def test_bullish_strong_adds_bonus(self):
        score, reasons = self._score({"catalyst_score": 75, "sentiment_label": "bullish"})
        self.assertTrue(any(r.startswith("news_catalyst_strong_") for r in reasons))
        baseline, _ = self._score(None)
        self.assertAlmostEqual(score - baseline, dts.NEWS_CATALYST_STRONG_BONUS)

    def test_bullish_weak_adds_smaller_bonus(self):
        score, reasons = self._score({"catalyst_score": 55, "sentiment_label": "bullish"})
        self.assertTrue(any(r.startswith("news_catalyst_weak_") for r in reasons))
        baseline, _ = self._score(None)
        self.assertAlmostEqual(score - baseline, dts.NEWS_CATALYST_WEAK_BONUS)

    def test_bearish_subtracts_penalty(self):
        score, reasons = self._score({"catalyst_score": 80, "sentiment_label": "bearish"})
        self.assertTrue(any(r.startswith("news_catalyst_bearish_") for r in reasons))
        baseline, _ = self._score(None)
        self.assertAlmostEqual(score - baseline, -dts.NEWS_CATALYST_BEARISH_PENALTY)

    def test_below_weak_floor_no_change(self):
        score, reasons = self._score({"catalyst_score": 40, "sentiment_label": "bullish"})
        self.assertFalse(any("news_catalyst" in r for r in reasons))

    def test_conviction_monotonicity(self):
        s_none, _ = self._score(None)
        s_50, _ = self._score({"catalyst_score": 50, "sentiment_label": "bullish"})
        s_70, _ = self._score({"catalyst_score": 70, "sentiment_label": "bullish"})
        self.assertGreater(s_70, s_50)
        self.assertGreater(s_50, s_none)


class TestEndToEnd(unittest.TestCase):
    """Veto + catalyst collision: covered conceptually by day-trader test_news_veto.
    Here we verify the scorer pipeline picks up the catalyst file."""

    def test_pipeline_reads_catalyst(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            (tdp / "kronos_predictions.json").write_text(json.dumps({
                "NVDA": {"direction": "bullish", "win_probability": 0.6},
            }))
            # whale_signals.json uses {"signals": [...]} list format per read_whale_signals().
            (tdp / "whale_signals.json").write_text(json.dumps({
                "signals": [{"ticker": "NVDA", "score": 85, "signal_type": "call"}],
            }))
            (tdp / "options_positions.json").write_text(json.dumps({"positions": []}))
            _write_catalyst(tdp, [
                {"ticker": "NVDA", "catalyst_score": 80, "sentiment_label": "bullish"},
            ])
            result = dts.run_scoring(signals_dir=tdp)
            sigs = result.get("signals", [])
            nvda = next((s for s in sigs if s["ticker"] == "NVDA"), None)
            self.assertIsNotNone(nvda, f"NVDA not in scored signals: {sigs}")
            self.assertTrue(
                any("news_catalyst_strong_" in r for r in nvda["reasons"]),
                f"news_catalyst tag missing from reasons: {nvda['reasons']}",
            )


if __name__ == "__main__":
    unittest.main()
