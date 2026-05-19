"""
Unit tests for ranked_tickers.json integration in day_trading_scorer.

Run with: pytest tools/test_ranked_tickers.py -v
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

# Stub kronos_client before import so gate is a no-op.
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

from tools import day_trading_scorer as dts  # noqa: E402


KRONOS_BULLISH = {"direction": "bullish", "win_probability": 0.6}

RANKED_DATA = {
    "generated_at": "2026-05-17T03:12:05+00:00",
    "date": "2026-05-17",
    "partial": False,
    "tickers": [
        {"ticker": "TSLA", "score": 90, "direction": "long"},
        {"ticker": "NVDA", "score": 85, "direction": "long"},
        {"ticker": "AAPL", "score": 80, "direction": "long"},
        {"ticker": "MSFT", "score": 75, "direction": "long"},
        {"ticker": "AMD", "score": 70, "direction": "long"},
        {"ticker": "AMZN", "score": 65, "direction": "long"},  # position 5 — lower bonus
    ],
}


def _write_ranked(td: Path, data: dict | None = None, age_seconds: float = 0) -> Path:
    p = td / "ranked_tickers.json"
    p.write_text(json.dumps(data or RANKED_DATA), encoding="utf-8")
    if age_seconds:
        old_mtime = time.time() - age_seconds
        import os
        os.utime(p, (old_mtime, old_mtime))
    return p


class TestReadRankedTickers(unittest.TestCase):
    def test_missing_file_returns_empty(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(dts.read_ranked_tickers(Path(td)), {})

    def test_valid_file_returns_rank_map(self):
        with tempfile.TemporaryDirectory() as td:
            _write_ranked(Path(td))
            m = dts.read_ranked_tickers(Path(td))
            self.assertEqual(m["TSLA"], 0)
            self.assertEqual(m["NVDA"], 1)
            self.assertEqual(m["AMZN"], 5)

    def test_lowercase_ticker_uppercased(self):
        with tempfile.TemporaryDirectory() as td:
            _write_ranked(Path(td), {"tickers": [{"ticker": "tsla", "score": 90}]})
            m = dts.read_ranked_tickers(Path(td))
            self.assertIn("TSLA", m)

    def test_stale_file_returns_empty(self):
        with tempfile.TemporaryDirectory() as td:
            age = dts.RANKED_TICKER_STALE_HOURS * 3600 + 60
            _write_ranked(Path(td), age_seconds=age)
            self.assertEqual(dts.read_ranked_tickers(Path(td)), {})

    def test_fresh_file_not_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            _write_ranked(Path(td))  # just-written = fresh
            m = dts.read_ranked_tickers(Path(td))
            self.assertGreater(len(m), 0)

    def test_corrupt_file_returns_empty(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "ranked_tickers.json"
            p.write_text("not json", encoding="utf-8")
            self.assertEqual(dts.read_ranked_tickers(Path(td)), {})


class TestScoreTickerRankBonus(unittest.TestCase):
    def _score(self, ranked_ticker_rank=None):
        return dts.score_ticker(
            ticker="TSLA",
            kronos=KRONOS_BULLISH,
            whale=None,
            sentiment=None,
            market_regime={},
            already_held=False,
            ranked_ticker_rank=ranked_ticker_rank,
        )

    def test_no_rank_no_bonus(self):
        score, reasons = self._score(None)
        self.assertFalse(any("ranked_ticker" in r for r in reasons))

    def test_top_rank_gives_top_bonus(self):
        baseline, _ = self._score(None)
        score, reasons = self._score(0)
        self.assertEqual(score - baseline, dts.RANKED_TICKER_TOP_BONUS)
        self.assertTrue(any("ranked_ticker_pos0" in r for r in reasons))

    def test_rank_4_still_top_bonus(self):
        baseline, _ = self._score(None)
        score, reasons = self._score(dts.RANKED_TICKER_TOP_N - 1)
        self.assertEqual(score - baseline, dts.RANKED_TICKER_TOP_BONUS)

    def test_rank_5_gives_lower_bonus(self):
        baseline, _ = self._score(None)
        score, reasons = self._score(dts.RANKED_TICKER_TOP_N)
        self.assertEqual(score - baseline, dts.RANKED_TICKER_BONUS)
        self.assertTrue(any("ranked_ticker_pos5" in r for r in reasons))

    def test_rank_10_gives_lower_bonus(self):
        baseline, _ = self._score(None)
        score, _ = self._score(10)
        self.assertEqual(score - baseline, dts.RANKED_TICKER_BONUS)


class TestRunScoringRankedTickers(unittest.TestCase):
    """Integration: ranked ticker appears in pipeline output with bonus reason tag."""

    def _minimal_signals(self, td: Path, extra_ticker: str | None = None) -> None:
        kronos = {"NVDA": {"direction": "bullish", "win_probability": 0.9}}
        if extra_ticker:
            kronos[extra_ticker] = {"direction": "bullish", "win_probability": 0.9}
        (td / "kronos_predictions.json").write_text(json.dumps(kronos))
        (td / "whale_signals.json").write_text(json.dumps({
            "signals": [{"ticker": "NVDA", "score": 90, "signal_type": "call"}],
        }))
        (td / "options_positions.json").write_text(json.dumps({"positions": []}))

    def test_ranked_ticker_union_adds_to_universe(self):
        """TSLA only in ranked_tickers.json + Kronos → appears in output."""
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            self._minimal_signals(tdp, extra_ticker="TSLA")
            _write_ranked(tdp, {"tickers": [{"ticker": "TSLA", "score": 90}]})
            result = dts.run_scoring(signals_dir=tdp)
            tickers = [s["ticker"] for s in result.get("signals", [])]
            self.assertIn("TSLA", tickers)

    def test_ranked_ticker_bonus_in_reasons(self):
        """Ranked ticker conviction reasons include ranked_ticker tag."""
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            self._minimal_signals(tdp, extra_ticker="TSLA")
            _write_ranked(tdp, {"tickers": [{"ticker": "TSLA", "score": 90}]})
            result = dts.run_scoring(signals_dir=tdp)
            tsla = next((s for s in result["signals"] if s["ticker"] == "TSLA"), None)
            self.assertIsNotNone(tsla, "TSLA not in output signals")
            self.assertTrue(
                any("ranked_ticker" in r for r in tsla["reasons"]),
                f"ranked_ticker tag missing from reasons: {tsla['reasons']}",
            )

    def test_stale_file_no_crash_no_bonus(self):
        """Stale ranked_tickers.json → scorer runs normally, no ranked_ticker bonus."""
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            self._minimal_signals(tdp)
            age = dts.RANKED_TICKER_STALE_HOURS * 3600 + 60
            _write_ranked(tdp, age_seconds=age)
            result = dts.run_scoring(signals_dir=tdp)
            for sig in result.get("signals", []):
                self.assertFalse(
                    any("ranked_ticker" in r for r in sig["reasons"]),
                    f"Stale file produced ranked_ticker bonus: {sig['reasons']}",
                )

    def test_missing_file_no_crash(self):
        """Missing ranked_tickers.json → scorer runs normally."""
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            self._minimal_signals(tdp)
            result = dts.run_scoring(signals_dir=tdp)
            self.assertIn("signals", result)


if __name__ == "__main__":
    unittest.main()
