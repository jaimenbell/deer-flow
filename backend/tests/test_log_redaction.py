"""Tests for app.log_redaction.

These tests prove the redaction filter scrubs the patterns we care about
before log records reach a handler, so a future debug print of an API key
cannot leak the key value to disk or console.
"""

from __future__ import annotations

import io
import logging

import pytest

from app.log_redaction import (
    SecretRedactionFilter,
    install_secret_redaction,
    redact,
    remove_secret_redaction,
)


# -- pure redact() function --------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected_substr_present, expected_substr_absent",
    [
        # Anthropic
        (
            "ANTHROPIC_API_KEY=sk-ant-api03-AAAAbbbb1111ZZZZ_-yyyy",
            "sk-ant-api03-[REDACTED:anthropic-key]",
            "AAAAbbbb1111ZZZZ",
        ),
        # OpenAI project key
        (
            "key sk-proj-AbCdEf1234567890XYZ_-abc",
            "sk-proj-[REDACTED:openai-project-key]",
            "AbCdEf1234567890XYZ",
        ),
        # Generic OpenAI sk-
        (
            "X-Token: sk-ABCDEFGHIJKLMNOPQRST",
            "sk-[REDACTED:openai-key]",
            "ABCDEFGHIJKLMNOPQRST",
        ),
        # Bearer token
        (
            "Authorization: Bearer abc.def.ghi-jkl_mno",
            "Bearer [REDACTED:bearer-token]",
            "abc.def.ghi-jkl_mno",
        ),
        # api_key= assignment
        (
            'config: api_key="my-secret-token-1234567"',
            "[REDACTED:api-key]",
            "my-secret-token-1234567",
        ),
    ],
)
def test_redact_replaces_known_patterns(raw, expected_substr_present, expected_substr_absent):
    out = redact(raw)
    assert expected_substr_present in out, f"redaction marker missing in: {out!r}"
    assert expected_substr_absent not in out, f"original secret leaked through: {out!r}"


def test_redact_passes_through_non_secret_text():
    s = "user_id=42 status=ok latency_ms=87"
    assert redact(s) == s


def test_redact_handles_non_string_input_gracefully():
    # The pure function only redacts strings; other types pass through unchanged.
    assert redact(123) == 123  # type: ignore[arg-type]
    assert redact(None) is None  # type: ignore[arg-type]


# -- SecretRedactionFilter on real LogRecords --------------------------------

def _capture_logs(filter_obj: SecretRedactionFilter) -> tuple[logging.Logger, io.StringIO]:
    """Build an isolated logger -> StringIO handler with the filter attached."""
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.addFilter(filter_obj)

    logger = logging.getLogger(f"test_log_redaction_{id(buf)}")
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    return logger, buf


def test_filter_redacts_message_template():
    flt = SecretRedactionFilter()
    logger, buf = _capture_logs(flt)
    logger.info("ANTHROPIC_API_KEY=sk-ant-api03-LEAKEDKEYBYTES_abcdef")
    output = buf.getvalue()
    assert "[REDACTED:anthropic-key]" in output
    assert "LEAKEDKEYBYTES_abcdef" not in output


def test_filter_redacts_args_via_percent_format():
    flt = SecretRedactionFilter()
    logger, buf = _capture_logs(flt)
    logger.info("key=%s", "sk-ant-api03-AnotherLeakValue123")
    output = buf.getvalue()
    assert "[REDACTED:anthropic-key]" in output
    assert "AnotherLeakValue123" not in output


def test_filter_redacts_dict_args():
    flt = SecretRedactionFilter()
    logger, buf = _capture_logs(flt)
    logger.info("key=%(k)s", {"k": "sk-ant-api03-DictArgLeakXYZ_-9"})
    output = buf.getvalue()
    assert "[REDACTED:anthropic-key]" in output
    assert "DictArgLeakXYZ" not in output


def test_filter_returns_true_so_records_propagate():
    # Sanity: filter should not suppress log emission, just rewrite content.
    flt = SecretRedactionFilter()
    logger, buf = _capture_logs(flt)
    logger.info("plain message no secret")
    assert "plain message no secret" in buf.getvalue()


def test_install_and_remove_secret_redaction_roundtrip():
    root = logging.getLogger()
    before = list(root.filters)
    flt = install_secret_redaction()
    try:
        assert flt in root.filters
    finally:
        remove_secret_redaction(flt)
    assert list(root.filters) == before


def test_filter_redacts_bearer_in_real_log_call():
    flt = SecretRedactionFilter()
    logger, buf = _capture_logs(flt)
    logger.warning("auth header: Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig-part")
    output = buf.getvalue()
    assert "[REDACTED:bearer-token]" in output
    assert "eyJhbGciOiJIUzI1NiJ9" not in output
