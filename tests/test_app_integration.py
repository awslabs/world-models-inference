# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end HTTP/WebSocket tests for the inference app's security controls.

Unlike test_security.py (which unit-tests the helpers), this builds the REAL
FastAPI app via lib.app.create_app() and drives it through Starlette's
TestClient, exercising the full middleware stack over actual requests:
auth (HTTP + WebSocket), CORS deny-by-default, rate limiting, path-traversal
rejection, and security headers.

Requires fastapi/starlette/httpx (inference-container deps); skipped when
absent, matching the repo's skip-when-deps-missing convention. torch is
stubbed so the app imports without a GPU.
"""

import importlib
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

INFERENCE_DIR = Path(__file__).resolve().parent.parent / "inference"
if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))


def _stub_torch():
    """Stub torch + torch.distributed so lib.app imports without a GPU."""
    if "torch" not in sys.modules or not hasattr(sys.modules["torch"], "_wm_stub"):
        torch = types.ModuleType("torch")
        torch.__path__ = []
        torch._wm_stub = True
        torch.cuda = types.SimpleNamespace(
            is_available=lambda: False, device_count=lambda: 0
        )
        dist = types.ModuleType("torch.distributed")
        dist.is_available = lambda: False
        dist.is_initialized = lambda: False
        torch.distributed = dist
        sys.modules["torch"] = torch
        sys.modules["torch.distributed"] = dist


def _build_app(monkeypatch, *, token=None, origins=None, rate=None, streaming=False):
    _stub_torch()
    for k in ("WORLD_MODEL_API_TOKEN", "WORLD_MODEL_ALLOWED_ORIGINS", "WORLD_MODEL_RATE_LIMIT"):
        monkeypatch.delenv(k, raising=False)
    if token is not None:
        monkeypatch.setenv("WORLD_MODEL_API_TOKEN", token)
    if origins is not None:
        monkeypatch.setenv("WORLD_MODEL_ALLOWED_ORIGINS", origins)
    if rate is not None:
        monkeypatch.setenv("WORLD_MODEL_RATE_LIMIT", str(rate))

    import lib.app as app_mod
    importlib.reload(app_mod)
    from lib.jobs import JobStore
    from lib.runner import Runner

    class _Runner(Runner):
        def setup(self, *a, **k):
            pass

        def generate(self, **k):
            return "/tmp/out.mp4"  # nosec B108 - dummy path in a test stub

        @property
        def supports_streaming(self):
            return streaming

        def reset(self):
            pass

        def stream(self, actions):
            yield b"frame-bytes"

    return app_mod.create_app(_Runner(), JobStore())


def _client(app):
    from starlette.testclient import TestClient
    return TestClient(app)


# --------------------------------------------------------------------------
# Auth — HTTP
# --------------------------------------------------------------------------


def test_health_open_regardless_of_auth(monkeypatch):
    c = _client(_build_app(monkeypatch, token="secret"))  # nosec
    assert c.get("/ping").status_code == 200
    assert c.get("/health").status_code == 200


def test_route_open_when_auth_disabled(monkeypatch):
    c = _client(_build_app(monkeypatch))  # no token
    # 404 (unknown job) means it passed through — not blocked by auth.
    assert c.get("/jobs/nope").status_code == 404


def test_missing_token_rejected(monkeypatch):
    c = _client(_build_app(monkeypatch, token="secret"))  # nosec
    assert c.get("/jobs/nope").status_code == 401


def test_wrong_token_rejected(monkeypatch):
    c = _client(_build_app(monkeypatch, token="secret"))  # nosec
    assert c.get("/jobs/nope", headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_bearer_token_accepted(monkeypatch):
    c = _client(_build_app(monkeypatch, token="secret"))  # nosec
    assert c.get("/jobs/nope", headers={"Authorization": "Bearer secret"}).status_code == 404  # nosec


def test_bare_token_accepted(monkeypatch):
    c = _client(_build_app(monkeypatch, token="secret"))  # nosec
    assert c.get("/jobs/nope", headers={"Authorization": "secret"}).status_code == 404  # nosec


def test_query_token_accepted(monkeypatch):
    c = _client(_build_app(monkeypatch, token="secret"))  # nosec
    assert c.get("/jobs/nope?token=secret").status_code == 404  # nosec


# --------------------------------------------------------------------------
# Path traversal (auth off so we isolate the validation behaviour)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["..%2F..%2Fetc%2Fpasswd", "a%2Fb", "%2Fetc%2Fpasswd"])
def test_example_traversal_blocked(monkeypatch, bad):
    c = _client(_build_app(monkeypatch))
    # 400 (invalid id) — never 200/500 from actually touching a path.
    assert c.get(f"/examples/{bad}/image").status_code in (400, 404)


def test_valid_example_id_passes_validation(monkeypatch):
    c = _client(_build_app(monkeypatch))
    # No such example on disk → 404, but it cleared validation (not 400).
    assert c.get("/examples/valid-id_1/image").status_code == 404


# --------------------------------------------------------------------------
# Rate limiting
# --------------------------------------------------------------------------


def test_rate_limit_triggers(monkeypatch):
    c = _client(_build_app(monkeypatch, rate=3))
    codes = [c.post("/generate", data={"prompt": "x"}).status_code for _ in range(6)]
    assert 429 in codes
    assert codes[:3] == [200, 200, 200]


def test_health_never_rate_limited(monkeypatch):
    c = _client(_build_app(monkeypatch, rate=1))
    assert all(c.get("/ping").status_code == 200 for _ in range(5))


# --------------------------------------------------------------------------
# CORS
# --------------------------------------------------------------------------


def test_cors_deny_by_default(monkeypatch):
    c = _client(_build_app(monkeypatch, origins=""))
    r = c.get("/ping", headers={"Origin": "https://evil.example"})
    assert r.headers.get("access-control-allow-origin") is None


def test_cors_allows_listed_origin(monkeypatch):
    c = _client(_build_app(monkeypatch, origins="https://good.example"))
    r = c.get("/ping", headers={"Origin": "https://good.example"})
    assert r.headers.get("access-control-allow-origin") == "https://good.example"


# --------------------------------------------------------------------------
# Security headers
# --------------------------------------------------------------------------


def test_security_headers_present(monkeypatch):
    c = _client(_build_app(monkeypatch))
    h = c.get("/ping").headers
    assert h.get("x-content-type-options") == "nosniff"
    assert h.get("x-frame-options") == "DENY"
    assert h.get("referrer-policy") == "no-referrer"


# --------------------------------------------------------------------------
# WebSocket auth
# --------------------------------------------------------------------------


def test_ws_rejected_without_token(monkeypatch):
    c = _client(_build_app(monkeypatch, token="wss", streaming=True))  # nosec
    with pytest.raises(Exception):
        with c.websocket_connect("/ws") as ws:
            ws.receive_bytes()


def test_ws_connects_with_token(monkeypatch):
    c = _client(_build_app(monkeypatch, token="wss", streaming=True))  # nosec
    with c.websocket_connect("/ws?token=wss") as ws:  # nosec
        assert ws.receive_bytes() == b"frame-bytes"


def test_ws_open_when_auth_disabled(monkeypatch):
    c = _client(_build_app(monkeypatch, streaming=True))
    with c.websocket_connect("/ws") as ws:
        assert ws.receive_bytes() == b"frame-bytes"
