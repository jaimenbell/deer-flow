"""Process-local authentication for Gateway internal callers."""

from __future__ import annotations

import os
import secrets
from types import SimpleNamespace

from deerflow.runtime.user_context import DEFAULT_USER_ID

INTERNAL_AUTH_HEADER_NAME = "X-DeerFlow-Internal-Token"
# Allow a pre-shared fixed token via env var so external callers (e.g. overnight-pipeline)
# can authenticate without knowing a per-boot random token.
_INTERNAL_AUTH_TOKEN = os.getenv("DEERFLOW_INTERNAL_TOKEN") or secrets.token_urlsafe(32)


def create_internal_auth_headers() -> dict[str, str]:
    """Return headers that authenticate same-process Gateway internal calls."""
    return {INTERNAL_AUTH_HEADER_NAME: _INTERNAL_AUTH_TOKEN}


def is_valid_internal_auth_token(token: str | None) -> bool:
    """Return True when *token* matches the process-local internal token."""
    return bool(token) and secrets.compare_digest(token, _INTERNAL_AUTH_TOKEN)


def get_internal_user():
    """Return the synthetic user used for trusted internal channel calls."""
    return SimpleNamespace(id=DEFAULT_USER_ID, system_role="internal")
