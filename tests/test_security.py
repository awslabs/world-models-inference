# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for inference/lib/security.py.

These exercise the auth, CORS-config, path-traversal, and rate-limit logic
without a GPU. They require fastapi/starlette (the inference-container deps);
where those are absent the whole module is skipped, matching the repo's
existing skip-when-deps-missing convention.
"""

import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

# Make `import lib.security` resolve against the inference package.
INFERENCE_DIR = Path(__file__).resolve().parent.parent / "inference"
if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))

from fastapi import HTTPException  # noqa: E402

from lib.security import (  # noqa: E402
    RateLimitMiddleware,
    SecurityConfig,
    extract_token,
    safe_subdir,
    token_valid,
)


# --------------------------------------------------------------------------
# SecurityConfig — env parsing
# --------------------------------------------------------------------------


def test_auth_disabled_when_token_unset(monkeypatch):
    monkeypatch.delenv("WORLD_MODEL_API_TOKEN", raising=False)
    assert SecurityConfig().auth_enabled is False


def test_auth_enabled_when_token_set(monkeypatch):
    monkeypatch.setenv("WORLD_MODEL_API_TOKEN", "secret")
    assert SecurityConfig().auth_enabled is True


def test_cors_denies_by_default(monkeypatch):
    monkeypatch.delenv("WORLD_MODEL_ALLOWED_ORIGINS", raising=False)
    assert SecurityConfig().allowed_origins == []


def test_cors_parses_allowlist(monkeypatch):
    monkeypatch.setenv("WORLD_MODEL_ALLOWED_ORIGINS", "https://a.example, https://b.example")
    assert SecurityConfig().allowed_origins == ["https://a.example", "https://b.example"]


def test_rate_limit_default_and_override(monkeypatch):
    monkeypatch.delenv("WORLD_MODEL_RATE_LIMIT", raising=False)
    assert SecurityConfig().rate_limit == 60
    monkeypatch.setenv("WORLD_MODEL_RATE_LIMIT", "5")
    assert SecurityConfig().rate_limit == 5
    monkeypatch.setenv("WORLD_MODEL_RATE_LIMIT", "not-a-number")
    assert SecurityConfig().rate_limit == 60  # falls back on garbage


# --------------------------------------------------------------------------
# Token extraction / verification
# --------------------------------------------------------------------------


def test_extract_token_bearer_header():
    assert extract_token("Bearer abc123", None) == "abc123"
    assert extract_token("bearer abc123", None) == "abc123"  # case-insensitive


def test_extract_token_bare_header_and_query():
    assert extract_token("abc123", None) == "abc123"
    assert extract_token(None, "qtok") == "qtok"
    assert extract_token(None, None) is None


def test_token_valid_disabled_allows_anything(monkeypatch):
    monkeypatch.delenv("WORLD_MODEL_API_TOKEN", raising=False)
    cfg = SecurityConfig()
    assert token_valid(cfg, None) is True
    assert token_valid(cfg, "whatever") is True


def test_token_valid_enforced_when_enabled(monkeypatch):
    monkeypatch.setenv("WORLD_MODEL_API_TOKEN", "s3cr3t")
    cfg = SecurityConfig()
    assert token_valid(cfg, "s3cr3t") is True
    assert token_valid(cfg, "wrong") is False
    assert token_valid(cfg, None) is False


# --------------------------------------------------------------------------
# Path traversal — safe_subdir
# --------------------------------------------------------------------------


def test_safe_subdir_accepts_valid_id(tmp_path):
    (tmp_path / "example1").mkdir()
    assert safe_subdir(tmp_path, "example1") == (tmp_path / "example1").resolve()


@pytest.mark.parametrize("evil", ["../etc", "../../etc/passwd", "a/b", "..", "", "foo/../bar", "/etc/passwd", "a b"])
def test_safe_subdir_rejects_traversal_and_bad_chars(tmp_path, evil):
    with pytest.raises(HTTPException) as exc:
        safe_subdir(tmp_path, evil)
    assert exc.value.status_code == 400


def test_safe_subdir_allows_dash_underscore(tmp_path):
    # Valid ids may contain dashes/underscores; result stays inside base.
    result = safe_subdir(tmp_path, "my-example_01")
    assert str(result).startswith(str(tmp_path.resolve()))


# --------------------------------------------------------------------------
# Rate-limit client key — X-Forwarded-For handling (N1)
# --------------------------------------------------------------------------


class _FakeRequest:
    """Minimal stand-in for starlette Request: headers + client peer."""

    def __init__(self, xff=None, peer="10.0.0.1"):
        self.headers = {"x-forwarded-for": xff} if xff is not None else {}

        class _Client:
            host = peer

        self.client = _Client() if peer else None


def _client_key(xff=None, peer="10.0.0.1"):
    mw = RateLimitMiddleware(app=None, config=SecurityConfig())
    return mw._client_key(_FakeRequest(xff=xff, peer=peer))


def test_client_key_uses_last_xff_hop(monkeypatch):
    monkeypatch.delenv("WORLD_MODEL_RATE_LIMIT", raising=False)
    # ALB appends the real peer as the rightmost hop; a client-forged leading
    # entry must NOT become the key (that would let callers dodge the limit).
    assert _client_key("1.1.1.1, 2.2.2.2, 203.0.113.9") == "203.0.113.9"
    # A single spoofed value is still the last hop, but two different forged
    # prefixes now collapse to the same real peer instead of distinct keys.
    assert _client_key("9.9.9.9, 203.0.113.9") == "203.0.113.9"


def test_client_key_falls_back_to_peer_without_xff(monkeypatch):
    monkeypatch.delenv("WORLD_MODEL_RATE_LIMIT", raising=False)
    assert _client_key(xff=None, peer="198.51.100.4") == "198.51.100.4"
    assert _client_key(xff="   ", peer="198.51.100.4") == "198.51.100.4"
    assert _client_key(xff=None, peer=None) == "unknown"
