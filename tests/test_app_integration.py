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
import threading
import time
import types
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

import cognito_testing  # noqa: E402

cognito_testing.requires_jwt()
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


def _build_app(monkeypatch, *, auth=False, origins=None, rate=None, streaming=False,
               hold=False, flow_window=None, seen=None, reset_hook=None,
               close_delay=None):
    """Build the app. ``auth=True`` configures Cognito; the default disables auth.

    Auth is off by default here only so the CORS/rate-limit/traversal tests isolate
    what they are about. The *server's* default is the opposite: authentication is
    required and a missing config is fatal (see test_security.py).
    """
    _stub_torch()
    for k in ("WORLD_MODEL_ALLOWED_ORIGINS", "WORLD_MODEL_RATE_LIMIT",
              "WORLD_MODEL_FLOW_WINDOW"):
        monkeypatch.delenv(k, raising=False)
    if auth:
        cognito_testing.use_cognito(monkeypatch)
    else:
        cognito_testing.disable_auth(monkeypatch)
    if origins is not None:
        monkeypatch.setenv("WORLD_MODEL_ALLOWED_ORIGINS", origins)
    if rate is not None:
        monkeypatch.setenv("WORLD_MODEL_RATE_LIMIT", str(rate))
    if flow_window is not None:
        monkeypatch.setenv("WORLD_MODEL_FLOW_WINDOW", str(flow_window))

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
            # reset_hook lets a test stall a session inside reset(), the window
            # where the gate is held but no action buffer is published yet.
            if reset_hook is not None:
                reset_hook()

        def stream(self, actions):
            try:
                yield b"frame-bytes"
                # hold=True keeps the session open (as a real-time runner does)
                # so a test can observe server state during a live session.
                while hold and actions.active:
                    time.sleep(0.01)
                    # seen collects what the runner is actually handed, so a test
                    # can prove protocol chatter never lands in the action buffer.
                    if seen is not None:
                        seen.append(actions.get())
                    yield b"frame-bytes"
            finally:
                # close_delay models a real-time runner whose teardown waits for
                # the in-flight device call (waypoint blocks up to 30 s here).
                if close_delay:
                    time.sleep(close_delay)

    return app_mod.create_app(_Runner(), JobStore())


def _client(app):
    from starlette.testclient import TestClient
    return TestClient(app)


# --------------------------------------------------------------------------
# Auth — HTTP
# --------------------------------------------------------------------------


def test_health_open_regardless_of_auth(monkeypatch):
    """LB health checks must never need a token, or the target goes unhealthy."""
    c = _client(_build_app(monkeypatch, auth=True))
    assert c.get("/ping").status_code == 200
    assert c.get("/health").status_code == 200


def test_route_open_when_auth_disabled(monkeypatch):
    c = _client(_build_app(monkeypatch))  # WORLD_MODEL_AUTH_MODE=disabled
    # 404 (unknown job) means it passed through — not blocked by auth.
    assert c.get("/jobs/nope").status_code == 404


def test_missing_token_rejected(monkeypatch):
    c = _client(_build_app(monkeypatch, auth=True))
    assert c.get("/jobs/nope").status_code == 401


def test_garbage_token_rejected(monkeypatch):
    c = _client(_build_app(monkeypatch, auth=True))
    assert c.get("/jobs/nope", headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_token_signed_by_another_key_rejected(monkeypatch):
    """A well-formed JWT from the wrong issuer must not be enough."""
    from cryptography.hazmat.primitives.asymmetric import rsa

    forged = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    c = _client(_build_app(monkeypatch, auth=True))
    token = cognito_testing.make_access_token(key=forged)
    assert c.get("/jobs/nope", headers={"Authorization": f"Bearer {token}"}).status_code == 401


def test_id_token_rejected(monkeypatch):
    """The old frontend sent an ID token; the API must require an access token."""
    c = _client(_build_app(monkeypatch, auth=True))
    token = cognito_testing.make_access_token(token_use="id")
    assert c.get("/jobs/nope", headers={"Authorization": f"Bearer {token}"}).status_code == 401


def test_token_without_scope_rejected(monkeypatch):
    c = _client(_build_app(monkeypatch, auth=True))
    token = cognito_testing.make_access_token(scope="openid")
    assert c.get("/jobs/nope", headers={"Authorization": f"Bearer {token}"}).status_code == 401


def test_expired_token_rejected(monkeypatch):
    c = _client(_build_app(monkeypatch, auth=True))
    token = cognito_testing.make_access_token(expires_in=-60)
    assert c.get("/jobs/nope", headers={"Authorization": f"Bearer {token}"}).status_code == 401


def test_bearer_access_token_accepted(monkeypatch):
    c = _client(_build_app(monkeypatch, auth=True))
    token = cognito_testing.make_access_token()
    assert c.get("/jobs/nope", headers={"Authorization": f"Bearer {token}"}).status_code == 404


def test_machine_client_token_accepted(monkeypatch):
    """client_credentials grant — how deploy.sh bench and CI authenticate."""
    c = _client(_build_app(monkeypatch, auth=True))
    token = cognito_testing.make_access_token(client_id=cognito_testing.MACHINE_CLIENT)
    assert c.get("/jobs/nope", headers={"Authorization": f"Bearer {token}"}).status_code == 404


def test_bare_token_accepted(monkeypatch):
    c = _client(_build_app(monkeypatch, auth=True))
    token = cognito_testing.make_access_token()
    assert c.get("/jobs/nope", headers={"Authorization": token}).status_code == 404


def test_query_token_accepted(monkeypatch):
    c = _client(_build_app(monkeypatch, auth=True))
    token = cognito_testing.make_access_token()
    assert c.get(f"/jobs/nope?token={token}").status_code == 404


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
    c = _client(_build_app(monkeypatch, auth=True, streaming=True))
    with pytest.raises(Exception):
        with c.websocket_connect("/ws") as ws:
            ws.receive_bytes()


def test_ws_rejected_with_id_token(monkeypatch):
    """WebSocket auth must apply the same checks as HTTP, not a weaker subset."""
    c = _client(_build_app(monkeypatch, auth=True, streaming=True))
    token = cognito_testing.make_access_token(token_use="id")
    with pytest.raises(Exception):
        with c.websocket_connect(f"/ws?token={token}") as ws:
            ws.receive_bytes()


def test_ws_rejected_with_expired_token(monkeypatch):
    c = _client(_build_app(monkeypatch, auth=True, streaming=True))
    token = cognito_testing.make_access_token(expires_in=-60)
    with pytest.raises(Exception):
        with c.websocket_connect(f"/ws?token={token}") as ws:
            ws.receive_bytes()


def test_ws_connects_with_access_token(monkeypatch):
    c = _client(_build_app(monkeypatch, auth=True, streaming=True))
    token = cognito_testing.make_access_token()
    with c.websocket_connect(f"/ws?token={token}") as ws:
        assert ws.receive_json()["type"] == "connected"
        assert ws.receive_bytes() == b"frame-bytes"


def test_ws_open_when_auth_disabled(monkeypatch):
    c = _client(_build_app(monkeypatch, streaming=True))
    with c.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "connected"
        assert ws.receive_bytes() == b"frame-bytes"


def test_ws_second_session_preempts_the_first(monkeypatch):
    # Real-time runners hold one session's state, so sessions are serialised:
    # a new connection closes the active session's action buffer and takes
    # over once its loop unwinds (last player wins). Preemption rather than
    # refusal, because a refused-second design locks everyone out after a
    # dirty disconnect and React StrictMode's double socket would refuse the
    # real player.
    c = _client(_build_app(monkeypatch, streaming=True, hold=True))
    with c.websocket_connect("/ws") as first:
        assert first.receive_json()["type"] == "connected"
        assert first.receive_bytes() == b"frame-bytes"
        with c.websocket_connect("/ws") as second:
            assert second.receive_json()["type"] == "connected"
            assert second.receive_bytes() == b"frame-bytes"


# --------------------------------------------------------------------------
# Frame flow control
# --------------------------------------------------------------------------


def test_ws_advertises_the_flow_window(monkeypatch):
    # Clients only acknowledge frames when the server asks them to, so the
    # window has to be announced on the handshake.
    c = _client(_build_app(monkeypatch, streaming=True, flow_window=4))
    with c.websocket_connect("/ws") as ws:
        assert ws.receive_json() == {"type": "connected", "flow_window": 4}


def test_ws_keeps_sending_while_the_client_acknowledges(monkeypatch):
    # An acknowledging client must not be throttled: frames keep flowing as
    # long as it stays within the window.
    c = _client(_build_app(monkeypatch, streaming=True, hold=True, flow_window=2))
    with c.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "connected"
        started = time.monotonic()
        for n in range(1, 9):
            assert ws.receive_bytes() == b"frame-bytes"
            ws.send_json({"type": "ack", "n": n})
        # No stall means no wait for the acknowledgement timeout.
        assert time.monotonic() - started < 1.0


def test_ws_falls_back_to_unbounded_for_a_silent_client(monkeypatch):
    # A client that predates flow control never acknowledges anything. It must
    # still receive frames — one stall, then the window is abandoned.
    import lib.app as app_mod
    app = _build_app(monkeypatch, streaming=True, hold=True, flow_window=2)
    monkeypatch.setattr(app_mod, "FLOW_ACK_TIMEOUT", 0.25)
    c = _client(app)
    with c.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "connected"
        started = time.monotonic()
        for _ in range(6):
            assert ws.receive_bytes() == b"frame-bytes"
        elapsed = time.monotonic() - started
        assert elapsed >= 0.25, "expected one stall before giving up on acks"
        assert elapsed < 2.0, "expected the window to be abandoned, not re-stalled"


def test_ws_acknowledgements_never_reach_the_runner(monkeypatch):
    # The runner reads an unrecognised message type as a neutral action, so an
    # acknowledgement reaching the action buffer would cancel held keys.
    seen: list = []
    c = _client(_build_app(monkeypatch, streaming=True, hold=True, flow_window=8,
                           seen=seen))
    with c.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "connected"
        ws.receive_bytes()
        ws.send_json({"type": "control", "buttons": ["W"]})
        for n in range(1, 5):
            ws.send_json({"type": "ack", "n": n})
            ws.receive_bytes()
    assert seen, "runner should have been handed at least one action"
    assert not any(b"ack" in a for a in seen if a)


# --------------------------------------------------------------------------
# Session gate under contention
# --------------------------------------------------------------------------


def test_ws_every_contender_is_served_or_closed_never_left_hanging(monkeypatch):
    # Two connections arriving together on a held session both preempt it
    # before either has published a buffer of its own. One wins the gate and
    # becomes the session; the other checked the shared slot while it was
    # empty, so it closed nothing and — with a single check-then-block gate
    # wait — sleeps on a gate that the winner (a real-time session, which runs
    # until its player leaves) will not release. The socket is accepted and
    # then silent forever: "last player wins" quietly becomes "this player
    # hangs". Re-running the preempt on every gate-wait timeout is what makes
    # the loser's turn arrive; the deadline guarantees it is at worst a clean
    # close. Both contenders must therefore finish, not just one.
    app = _build_app(monkeypatch, streaming=True, hold=True)
    import lib.app as app_mod
    monkeypatch.setattr(app_mod, "GATE_RETRY_INTERVAL", 0.05, raising=False)
    c = _client(app)

    resolved: list[int] = []
    release = threading.Event()

    def contend(tag):
        try:
            with c.websocket_connect("/ws") as ws:
                assert ws.receive_json()["type"] == "connected"
                assert ws.receive_bytes() == b"frame-bytes"
                resolved.append(tag)
                # Hold the session, as a player would. This is the shape that
                # exposes the bug: the winner does not hand the gate back on
                # its own, so the loser depends entirely on retrying.
                release.wait(timeout=10)
        except Exception:
            # A clean refusal (1013) also counts as resolved; a silent socket
            # that never says anything does not.
            resolved.append(tag)

    with c.websocket_connect("/ws") as first:
        assert first.receive_json()["type"] == "connected"
        assert first.receive_bytes() == b"frame-bytes"
        threads = [threading.Thread(target=contend, args=(n,), daemon=True)
                   for n in (1, 2)]
        for t in threads:
            t.start()
        deadline = time.monotonic() + 6
        while len(resolved) < 2 and time.monotonic() < deadline:
            time.sleep(0.05)
        # Read the result BEFORE releasing anything: the release and the outer
        # socket's close both free the gate, and either would let a stuck
        # contender recover just in time to mask the bug.
        served = sorted(resolved)
        release.set()
        for t in threads:
            t.join(timeout=5)
    assert served == [1, 2], f"only {served} of 2 contenders were ever answered"


def test_ping_stays_responsive_while_a_preempted_session_tears_down(monkeypatch):
    # A preempted session leaves its frame generator unexhausted. If that
    # generator is closed on the event loop (which is what garbage collection
    # does), the runner's teardown — waypoint waits up to 30 s for the
    # in-flight device call — freezes the loop, and /ping with it: the load
    # balancer then declares the instance dead mid-preemption. Uses a single
    # portal (`with TestClient(...)`) so every request shares one event loop,
    # as in production, and a 1 s teardown to make any on-loop close visible.
    app = _build_app(monkeypatch, streaming=True, hold=True, close_delay=1.0)
    with _client(app) as c:
        with c.websocket_connect("/ws") as first:
            assert first.receive_json()["type"] == "connected"
            assert first.receive_bytes() == b"frame-bytes"
            with c.websocket_connect("/ws") as second:
                # The second connection preempts the first, whose generator
                # close now blocks for close_delay. The loop must keep serving.
                worst = 0.0
                start = time.monotonic()
                while time.monotonic() - start < 1.5:
                    sent = time.monotonic()
                    assert c.get("/ping").status_code == 200
                    worst = max(worst, time.monotonic() - sent)
                assert worst < 0.5, (
                    f"/ping stalled {worst:.2f}s — teardown ran on the event loop"
                )
                assert second.receive_json()["type"] == "connected"
