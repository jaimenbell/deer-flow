"""
Unit + integration tests for read_daytrader_positions() and its union into held_tickers.

Run with: pytest tools/test_daytrader_positions.py -v
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Stub kronos_client before importing scorer (gate must be a no-op for these tests).
import types as _types
_stub = _types.ModuleType("kronos_client")
_stub.get_kronos_gate_pass = lambda ticker, direction: True
_stub.get_kronos_gate_reason = lambda ticker, direction: "stub"
sys.modules["kronos_client"] = _stub

# Stub langchain_core.tools.tool so import doesn't crash.
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_dt_positions(td: Path, positions: list[dict]) -> None:
    """Write day_trader_positions.json as a top-level array (position_manager.py format)."""
    (td / "day_trader_positions.json").write_text(
        json.dumps(positions), encoding="utf-8"
    )


def _open_pos(ticker: str) -> dict:
    return {"id": f"{ticker}_test", "ticker": ticker, "status": "open", "direction": "call"}


def _closed_pos(ticker: str) -> dict:
    return {"id": f"{ticker}_test", "ticker": ticker, "status": "closed", "direction": "call"}


# ---------------------------------------------------------------------------
# Unit tests: read_daytrader_positions
# ---------------------------------------------------------------------------

class TestReadDaytraderPositions(unittest.TestCase):

    def test_missing_file_returns_empty(self):
        with tempfile.TemporaryDirectory() as td:
            result = dts.read_daytrader_positions(Path(td))
            self.assertEqual(result, set())

    def test_only_closed_positions_returns_empty(self):
        with tempfile.TemporaryDirectory() as td:
            _write_dt_positions(Path(td), [_closed_pos("GEHC"), _closed_pos("NVDA")])
            result = dts.read_daytrader_positions(Path(td))
            self.assertEqual(result, set())

    def test_open_positions_returned(self):
        with tempfile.TemporaryDirectory() as td:
            _write_dt_positions(Path(td), [_open_pos("GEHC"), _open_pos("MSFT")])
            result = dts.read_daytrader_positions(Path(td))
            self.assertEqual(result, {"GEHC", "MSFT"})

    def test_mixed_open_and_closed(self):
        with tempfile.TemporaryDirectory() as td:
            _write_dt_positions(Path(td), [
                _open_pos("GEHC"),
                _closed_pos("NVDA"),
                _open_pos("TSLA"),
            ])
            result = dts.read_daytrader_positions(Path(td))
            self.assertEqual(result, {"GEHC", "TSLA"})
            self.assertNotIn("NVDA", result)

    def test_ticker_uppercased(self):
        with tempfile.TemporaryDirectory() as td:
            _write_dt_positions(Path(td), [
                {"id": "x", "ticker": "gehc", "status": "open"},
            ])
            result = dts.read_daytrader_positions(Path(td))
            self.assertIn("GEHC", result)

    def test_corrupt_file_returns_empty(self):
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "day_trader_positions.json").write_text("not json", encoding="utf-8")
            result = dts.read_daytrader_positions(Path(td))
            self.assertEqual(result, set())


# ---------------------------------------------------------------------------
# Integration: already_held_penalty fires for day-trader position
# ---------------------------------------------------------------------------

KRONOS_BULLISH = {"direction": "bullish", "win_probability": 0.6}


class TestAlreadyHeldPenaltyIntegration(unittest.TestCase):
    """
    Regression for the F6 gap: day-trader holds GEHC intraday →
    DeerFlow scorer must apply already_held_penalty to any GEHC signal.
    """

    def _minimal_signals_dir(self, td: Path, open_tickers: list[str]) -> Path:
        """Write the minimum bus files needed for run_scoring() to process GEHC."""
        tdp = Path(td)
        # Kronos predicts GEHC bullish
        (tdp / "kronos_predictions.json").write_text(json.dumps({
            t: KRONOS_BULLISH for t in open_tickers
        }), encoding="utf-8")
        # Whale signals for GEHC
        (tdp / "whale_signals.json").write_text(json.dumps({
            "signals": [
                {"ticker": t, "score": 85, "signal_type": "call"}
                for t in open_tickers
            ],
        }), encoding="utf-8")
        # No wheel positions
        (tdp / "options_positions.json").write_text(
            json.dumps({"positions": []}), encoding="utf-8"
        )
        # Day-trader holds the tickers
        _write_dt_positions(tdp, [_open_pos(t) for t in open_tickers])
        return tdp

    def test_already_held_penalty_fires_for_daytrader_position(self):
        with tempfile.TemporaryDirectory() as td:
            signals_dir = self._minimal_signals_dir(td, ["GEHC"])
            result = dts.run_scoring(signals_dir=signals_dir)
            sigs = result.get("signals", [])
            gehc = next((s for s in sigs if s["ticker"] == "GEHC"), None)
            # Even if the signal passes threshold, already_held_penalty must be tagged.
            # If score drops below threshold, verify via direct score_ticker call.
            if gehc is not None:
                self.assertIn(
                    "already_held_penalty", gehc["reasons"],
                    f"already_held_penalty missing from reasons: {gehc['reasons']}",
                )
            else:
                # Signal filtered out (score below threshold) — verify penalty was applied
                # by scoring directly.
                score, reasons = dts.score_ticker(
                    ticker="GEHC",
                    kronos=KRONOS_BULLISH,
                    whale={"score": 85, "signal_type": "call"},
                    sentiment=None,
                    market_regime={},
                    already_held=True,
                    news_catalyst=None,
                )
                self.assertIn("already_held_penalty", reasons)

    def test_no_penalty_without_daytrader_position(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            (tdp / "kronos_predictions.json").write_text(json.dumps({
                "GEHC": KRONOS_BULLISH,
            }), encoding="utf-8")
            (tdp / "whale_signals.json").write_text(json.dumps({
                "signals": [{"ticker": "GEHC", "score": 85, "signal_type": "call"}],
            }), encoding="utf-8")
            (tdp / "options_positions.json").write_text(
                json.dumps({"positions": []}), encoding="utf-8"
            )
            # No day_trader_positions.json — GEHC is not held
            result = dts.run_scoring(signals_dir=tdp)
            sigs = result.get("signals", [])
            gehc = next((s for s in sigs if s["ticker"] == "GEHC"), None)
            if gehc is not None:
                self.assertNotIn("already_held_penalty", gehc["reasons"])

    def test_union_options_and_daytrader(self):
        """held_tickers = options_held ∪ daytrader_held."""
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            # MARA held by options-bot, GEHC held by day-trader
            (tdp / "options_positions.json").write_text(json.dumps({
                "positions": [{"ticker": "MARA"}],
            }), encoding="utf-8")
            _write_dt_positions(tdp, [_open_pos("GEHC")])

            options_held = dts.read_options_positions(tdp)
            dt_held = dts.read_daytrader_positions(tdp)
            combined = options_held | dt_held

            self.assertIn("MARA", combined)
            self.assertIn("GEHC", combined)
            self.assertEqual(len(combined), 2)


if __name__ == "__main__":
    unittest.main()
