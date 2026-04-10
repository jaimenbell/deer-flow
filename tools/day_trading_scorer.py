"""
DeerFlow Day Trading Scorer — Signal Bus Reader & Conviction Engine

Reads the shared signal bus at SIGNALS_DIR, scores each ticker using the
conviction formula, and writes a ranked trade_signal.json artifact.

Can be imported as a LangChain tool (via @tool decorator) for use inside
a DeerFlow agent, or called directly from run_scorer.py.

Signal bus files expected:
  whale_signals.json         — whale tracker output
  kronos_predictions.json    — Kronos ML forecasts
  options_positions.json     — currently held positions
  mirofish_sentiment.json    — optional sentiment overlay
  market_regime.json         — optional news/macro regime + Fear & Greed
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Kronos gate — must be importable for signals to be emitted.
_SIGNALS_CLIENT_DIR = str(Path(r"C:/Users/owner/projects/signals"))
if _SIGNALS_CLIENT_DIR not in sys.path:
    sys.path.insert(0, _SIGNALS_CLIENT_DIR)

try:
    from kronos_client import get_kronos_gate_pass, get_kronos_gate_reason
except ImportError as _e:
    # If kronos_client is unavailable the gate blocks everything — no trades.
    logging.warning("kronos_client not importable (%s); all tickers will be blocked.", _e)

    def get_kronos_gate_pass(ticker: str, intended_direction: str) -> bool:  # type: ignore[misc]
        return False

    def get_kronos_gate_reason(ticker: str, intended_direction: str) -> str:  # type: ignore[misc]
        return "kronos_client import failed"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SIGNALS_DIR = Path(os.environ.get("SIGNALS_DIR", r"C:/Users/owner/projects/signals"))
OUTPUT_FILE = SIGNALS_DIR / "trade_signal.json"

# Risk rules — hard-coded, not configurable
RISK_RULES: dict[str, Any] = {
    "daily_loss_limit_pct": -0.02,
    "max_positions": 6,
    "base_position_size_pct": 0.02,
    "strong_signal_size_pct": 0.035,
    "profit_target_pct": 0.40,
    "stop_loss_pct": -0.25,
    "max_hold_hours": 4,
}

ENTRY_THRESHOLD = 85
STRONG_ENTRY_THRESHOLD = 100


# ---------------------------------------------------------------------------
# Signal bus readers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Any:
    """Load JSON file; return None if file is missing."""
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def read_whale_signals(signals_dir: Path = SIGNALS_DIR) -> dict[str, dict]:
    """Return {ticker: {score, signal_type, ...}} from whale_signals.json."""
    data = _load_json(signals_dir / "whale_signals.json")
    if not data:
        return {}
    result: dict[str, dict] = {}
    for sig in data.get("signals", []):
        ticker = sig.get("ticker", "").upper()
        if ticker:
            result[ticker] = sig
    return result


def read_kronos_predictions(signals_dir: Path = SIGNALS_DIR) -> dict[str, dict]:
    """Return {ticker: {win_probability, predicted_return, confidence, direction}}."""
    data = _load_json(signals_dir / "kronos_predictions.json")
    if not data:
        return {}
    # Support both dict-of-tickers and list-of-predictions formats
    if isinstance(data, dict) and "predictions" in data:
        entries = data["predictions"]
    elif isinstance(data, list):
        entries = data
    elif isinstance(data, dict):
        # Direct {ticker: {...}} mapping
        return {k.upper(): v for k, v in data.items()}
    else:
        return {}
    return {e["ticker"].upper(): e for e in entries if "ticker" in e}


def read_options_positions(signals_dir: Path = SIGNALS_DIR) -> set[str]:
    """Return set of tickers currently held in options positions."""
    data = _load_json(signals_dir / "options_positions.json")
    if not data:
        return set()
    positions = data if isinstance(data, list) else data.get("positions", [])
    return {p["ticker"].upper() for p in positions if "ticker" in p}


def read_mirofish_sentiment(signals_dir: Path = SIGNALS_DIR) -> dict[str, str]:
    """Return {ticker: 'positive'|'negative'|'neutral'} — optional."""
    data = _load_json(signals_dir / "mirofish_sentiment.json")
    if not data:
        return {}
    if isinstance(data, dict) and "sentiments" in data:
        return {e["ticker"].upper(): e.get("sentiment", "neutral")
                for e in data["sentiments"] if "ticker" in e}
    if isinstance(data, dict):
        return {k.upper(): v for k, v in data.items()}
    return {}


def read_market_regime(signals_dir: Path = SIGNALS_DIR) -> dict[str, Any]:
    """Return {news_regime: str, fear_and_greed: int} — optional."""
    data = _load_json(signals_dir / "market_regime.json")
    if not data:
        return {}
    return data


# ---------------------------------------------------------------------------
# Conviction scorer
# ---------------------------------------------------------------------------

def score_ticker(
    ticker: str,
    kronos: dict,
    whale: dict | None,
    sentiment: str | None,
    market_regime: dict,
    already_held: bool,
) -> tuple[float, list[str]]:
    """
    Compute conviction score for a single ticker.

    Returns (score, reasons_list).
    """
    reasons: list[str] = []

    # Base score from Kronos win probability
    win_prob: float = float(kronos.get("win_probability", 0))
    base_score = win_prob * 100
    reasons.append(f"kronos_{kronos.get('direction', 'unknown')}_{win_prob:.2f}")

    score = base_score

    # --- Bonuses ---
    whale_score: int = int(whale["score"]) if whale else 0

    if whale_score >= 80:
        score += 25
        reasons.append(f"whale_confirmed_{whale_score}")
    elif whale_score >= 60:
        score += 10
        reasons.append(f"whale_partial_{whale_score}")

    if sentiment == "positive":
        score += 10
        reasons.append("mirofish_positive")

    news_regime: str = str(market_regime.get("news_regime", "")).lower()
    fear_and_greed: int | None = market_regime.get("fear_and_greed")

    if news_regime == "risk_on":
        score += 5
        reasons.append("news_risk_on")

    # --- Penalties ---
    if news_regime == "risk_off":
        score -= 30
        reasons.append("news_risk_off_penalty")

    if fear_and_greed is not None and fear_and_greed < 20:
        if whale_score >= 90:
            reasons.append("fg_low_bypassed_by_whale")
        else:
            score -= 20
            reasons.append(f"fg_extreme_fear_{fear_and_greed}")

    if already_held:
        score -= 15
        reasons.append("already_held_penalty")

    return score, reasons


# ---------------------------------------------------------------------------
# Direction helper
# ---------------------------------------------------------------------------

def determine_intended_direction(ticker: str, kronos_map: dict, whale_map: dict) -> str:
    """
    Derive intended trade direction for a ticker.

    Uses Kronos as the primary source.  Whale signal_type is checked first as a
    tiebreaker when Kronos is absent or neutral, but in practice Kronos must
    confirm the direction via get_kronos_gate_pass() before any trade proceeds.

    Returns 'call', 'put', or 'neutral'.
    """
    # Whale signal can hint at direction even before Kronos confirmation
    whale = whale_map.get(ticker, {})
    signal_type = str(whale.get("signal_type", "")).lower()
    if any(kw in signal_type for kw in ("call", "bullish", "buy", "long")):
        return "call"
    if any(kw in signal_type for kw in ("put", "bearish", "sell", "short")):
        return "put"

    # Fall back to Kronos direction
    kronos_dir = kronos_map.get(ticker, {}).get("direction", "neutral")
    if kronos_dir == "bullish":
        return "call"
    if kronos_dir == "bearish":
        return "put"
    return "neutral"


# ---------------------------------------------------------------------------
# Main scoring pipeline
# ---------------------------------------------------------------------------

def run_scoring(signals_dir: Path = SIGNALS_DIR) -> dict[str, Any]:
    """
    Full pipeline: read all signal bus files → score each ticker →
    filter → rank → return the trade_signal dict (does NOT write to disk).
    """
    whale_map = read_whale_signals(signals_dir)
    kronos_map = read_kronos_predictions(signals_dir)
    held_tickers = read_options_positions(signals_dir)
    mirofish = read_mirofish_sentiment(signals_dir)
    market_regime = read_market_regime(signals_dir)

    # Candidate universe = union of kronos (primary) and whale tickers
    candidates = set(kronos_map.keys()) | set(whale_map.keys())

    scored: list[dict] = []
    for ticker in sorted(candidates):
        # ── Kronos hard gate ─────────────────────────────────────────────────
        # Must pass BEFORE any other scoring.  No gate pass = no signal emitted,
        # regardless of whale/news/sentiment signals.
        direction = determine_intended_direction(ticker, kronos_map, whale_map)
        if not get_kronos_gate_pass(ticker, direction):
            reason = get_kronos_gate_reason(ticker, direction)
            logging.info("SKIP %s: Kronos gate failed — %s", ticker, reason)
            continue

        kronos = kronos_map.get(ticker)
        if not kronos:
            # Should not be reachable after gate (gate already checked prediction),
            # but guard defensively so score_ticker always receives a valid dict.
            continue

        whale = whale_map.get(ticker)
        sentiment = mirofish.get(ticker)
        already_held = ticker in held_tickers

        conviction_score, reasons = score_ticker(
            ticker=ticker,
            kronos=kronos,
            whale=whale,
            sentiment=sentiment,
            market_regime=market_regime,
            already_held=already_held,
        )

        scored.append(
            {
                "ticker": ticker,
                "conviction_score": round(conviction_score, 2),
                "direction": direction,
                "reasons": reasons,
            }
        )

    # Filter to entry threshold, sort descending
    qualified = [s for s in scored if s["conviction_score"] >= ENTRY_THRESHOLD]
    qualified.sort(key=lambda x: x["conviction_score"], reverse=True)

    # Respect max concurrent positions
    open_slots = RISK_RULES["max_positions"] - len(held_tickers)
    qualified = qualified[:max(open_slots, 0)]

    # Build output signals
    signals: list[dict] = []
    for rank, candidate in enumerate(qualified, start=1):
        score = candidate["conviction_score"]
        is_strong = score >= STRONG_ENTRY_THRESHOLD
        signals.append(
            {
                "ticker": candidate["ticker"],
                "direction": candidate["direction"],
                "conviction_score": score,
                "size_pct": (
                    RISK_RULES["strong_signal_size_pct"]
                    if is_strong
                    else RISK_RULES["base_position_size_pct"]
                ),
                "profit_target_pct": RISK_RULES["profit_target_pct"],
                "stop_loss_pct": RISK_RULES["stop_loss_pct"],
                "max_hold_hours": RISK_RULES["max_hold_hours"],
                "reasons": candidate["reasons"],
                "rank": rank,
            }
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "signals": signals,
        "risk_rules": {
            "daily_loss_limit_pct": RISK_RULES["daily_loss_limit_pct"],
            "max_positions": RISK_RULES["max_positions"],
        },
    }


def write_trade_signal(output: dict[str, Any], output_path: Path = OUTPUT_FILE) -> Path:
    """Write the trade signal dict to JSON and return the output path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    return output_path


# ---------------------------------------------------------------------------
# DeerFlow LangChain tool
# ---------------------------------------------------------------------------

try:
    from langchain_core.tools import tool

    @tool("day_trading_scorer", parse_docstring=True)
    def day_trading_scorer_tool(signals_dir: str = str(SIGNALS_DIR)) -> str:
        """Score day-trading candidates from the shared signal bus and write trade_signal.json.

        Reads whale_signals.json, kronos_predictions.json, options_positions.json
        (and optionally mirofish_sentiment.json, market_regime.json) from the signal
        bus directory, runs the conviction scoring formula, and writes a ranked
        trade_signal.json artifact.

        Args:
            signals_dir: Path to the signals directory. Defaults to the value of
                the SIGNALS_DIR environment variable or
                C:/Users/owner/projects/signals.

        Returns:
            JSON string summarising the scored signals written to trade_signal.json.
        """
        dir_path = Path(signals_dir)
        result = run_scoring(dir_path)
        out_path = write_trade_signal(result, dir_path / "trade_signal.json")
        summary = {
            "status": "ok",
            "output_file": str(out_path),
            "signal_count": len(result["signals"]),
            "top_signals": [
                {
                    "ticker": s["ticker"],
                    "conviction_score": s["conviction_score"],
                    "direction": s["direction"],
                    "size_pct": s["size_pct"],
                }
                for s in result["signals"][:5]
            ],
        }
        return json.dumps(summary, indent=2)

except ImportError:
    # langchain_core not installed — standalone mode only
    day_trading_scorer_tool = None  # type: ignore[assignment]
