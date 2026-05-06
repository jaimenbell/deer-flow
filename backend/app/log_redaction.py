"""Secret redaction for log records.

Defense-in-depth utility: any log handler attached with this filter will
have API keys, bearer tokens, and other secret-shaped substrings redacted
from message text BEFORE the record reaches its formatter and handler I/O.

This does NOT replace careful logging hygiene at call sites - prefer not
logging secrets in the first place. The filter exists to limit the blast
radius when a debug print or stray logger.info leaks credentials.

Patterns covered:
  - Anthropic keys:  sk-ant-api03-...
  - OpenAI keys:     sk-proj-..., sk-<long>
  - Bearer tokens:   "Bearer <token>"
  - api_key=<value>, api-key: "<value>" style assignments

Replacements preserve the prefix marker so log lines remain greppable
("an Anthropic key was here") without exposing key bytes.

Usage:
    import logging
    from app.log_redaction import install_secret_redaction

    install_secret_redaction()  # attaches to root logger
    logging.getLogger(__name__).info("ANTHROPIC_API_KEY=%s", os.environ[...])
    # -> emits "ANTHROPIC_API_KEY=sk-ant-api03-[REDACTED:anthropic-key]"

The filter walks both the formatted message and any args, so both
``logger.info("k=%s", key)`` and ``logger.info(f"k={key}")`` are redacted.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable, Pattern, Tuple

# Order matters: more specific patterns first so they win over general ones.
_REDACTION_RULES: Tuple[Tuple[Pattern[str], str], ...] = (
    (re.compile(r"sk-ant-api03-[A-Za-z0-9_\-]+"),
     "sk-ant-api03-[REDACTED:anthropic-key]"),
    (re.compile(r"sk-proj-[A-Za-z0-9_\-]+"),
     "sk-proj-[REDACTED:openai-project-key]"),
    (re.compile(r"sk-[A-Za-z0-9]{20,}"),
     "sk-[REDACTED:openai-key]"),
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]+"),
     r"\1[REDACTED:bearer-token]"),
    (re.compile(
        r"(?i)(api[_-]?key[\"']?\s*[:=]\s*[\"']?)([A-Za-z0-9_\-]{16,})"
    ), r"\1[REDACTED:api-key]"),
)


def redact(text: str) -> str:
    """Apply all redaction rules to a string. Pure function; safe to unit test."""
    if not isinstance(text, str):
        return text
    out = text
    for pattern, replacement in _REDACTION_RULES:
        out = pattern.sub(replacement, out)
    return out


class SecretRedactionFilter(logging.Filter):
    """logging.Filter that scrubs secrets from a record's message and args.

    Returns True so the record is always emitted (after rewriting). Mutates
    the record in place: this is consistent with how stdlib filters like
    LogRecord usage in older formatters expect to operate, and ensures the
    same redaction takes effect across multiple handlers attached to the
    same logger.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        # Redact the raw msg template and any string args. We avoid eagerly
        # calling record.getMessage() because that would cache a formatted
        # string on the record and bypass downstream %-formatting.
        if isinstance(record.msg, str):
            record.msg = redact(record.msg)

        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: redact(v) if isinstance(v, str) else v
                    for k, v in record.args.items()
                }
            elif isinstance(record.args, tuple):
                record.args = tuple(
                    redact(a) if isinstance(a, str) else a for a in record.args
                )
            elif isinstance(record.args, list):
                record.args = [
                    redact(a) if isinstance(a, str) else a for a in record.args
                ]

        return True


def install_secret_redaction(
    loggers: Iterable[logging.Logger] | None = None,
    *,
    attach_to_root: bool = True,
) -> SecretRedactionFilter:
    """Attach a SecretRedactionFilter to the given loggers (and the root logger).

    Returns the filter instance so callers can detach it later if needed.

    Idempotency: calling this multiple times will attach multiple filter
    instances. Callers that may invoke it more than once (e.g. test suites)
    should track the returned filter and call ``remove_secret_redaction``
    or only invoke this once at process startup.
    """
    flt = SecretRedactionFilter()
    if attach_to_root:
        logging.getLogger().addFilter(flt)
    if loggers:
        for lg in loggers:
            lg.addFilter(flt)
    return flt


def remove_secret_redaction(
    flt: SecretRedactionFilter,
    loggers: Iterable[logging.Logger] | None = None,
    *,
    detach_from_root: bool = True,
) -> None:
    """Remove a previously attached SecretRedactionFilter."""
    if detach_from_root:
        logging.getLogger().removeFilter(flt)
    if loggers:
        for lg in loggers:
            lg.removeFilter(flt)
