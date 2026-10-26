# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end session tests for the Matrix Game 2 cartridge.

Drives MatrixGame2Runner.stream() through the real shared server (lib.app) over
a WebSocket, with a fake engine standing in for the model. What is under test
is the adapter: seeding, the action pump, chunk-to-frame delivery and order,
and that sessions start, end and hand over cleanly.
"""

import importlib
import importlib.util
import io
import json
import struct
import sys
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
np = pytest.importorskip("numpy")
PIL_Image = pytest.importorskip("PIL.Image")

REPO_ROOT = Path(__file__).resolve().parent.parent
INFERENCE_DIR = REPO_ROOT / "inference"
CARTRIDGE_DIR = INFERENCE_DIR / "models" / "matrix-game-2"

if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))

FRAME_H, FRAME_W = 64, 64


def _stub_torch():
    """Same `_wm_stub` torch as the other session tests, so they share one."""
    if "torch" not in sys.modules or not hasattr(sys.modules["torch"], "_wm_stub"):
        torch = types.ModuleType("torch")
        torch.__path__ = []
        torch._wm_stub = True
        torch.cuda = types.SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
        dist = types.ModuleType("torch.distributed")
        dist.is_available = lambda: False
        dist.is_initialized = lambda: False
        torch.distributed = dist
        sys.modules["torch"] = torch
        sys.modules["torch.distributed"] = dist


def _load_runner_module():
    """Import runner.py with *our* actions/frames bound to its bare imports."""
    _stub_torch()
    saved = {name: sys.modules.get(name) for name in ("actions", "frames", "runner")}
    try:
        for name in ("actions", "frames", "runner"):
            spec = importlib.util.spec_from_file_location(name, CARTRIDGE_DIR / f"{name}.py")
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
        return sys.modules["runner"]
    finally:
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


runner_mod = _load_runner_module()


def frame_value(chunk: int, t: int) -> int:
    """Grey level identifying (chunk, frame in chunk); survives JPEG intact."""
    return 20 + (chunk % 4) * 50 + t * 3


class FakeEngine:
    """The engine surface runner.py uses, recording how it was driven."""

    def __init__(self, frames_per_chunk: int = 4, chunk_seconds: float = 0.0):
        self.frames_per_chunk = frames_per_chunk
        self.chunk_seconds = chunk_seconds
        self.starts: list = []
        self.ends = 0
        self.steps: list = []
        self.reseeds = 0
        self._started = False
        self.fail_at = None

    def default_seed(self):
        return "upstream-seed"

    @property
    def started(self) -> bool:
        return self._started

    def start(self, image) -> None:
        self.starts.append(image)
        self._started = True

    def end(self) -> None:
        self.ends += 1
        self._started = False

    def step(self, keyboard, yaw):
        if self.fail_at is not None and len(self.steps) >= self.fail_at:
            raise RuntimeError("simulated CUDA failure")
        if self.chunk_seconds:
            time.sleep(self.chunk_seconds)
        chunk = len(self.steps)
        self.steps.append((tuple(keyboard), yaw))
        out = np.empty((self.frames_per_chunk, FRAME_H, FRAME_W, 3), dtype=np.uint8)
        for t in range(self.frames_per_chunk):
            out[t] = frame_value(chunk, t)
        return out


def make_runner(engine: FakeEngine, seed_wait: float = 0.3):
    """A MatrixGame2Runner wired to a fake engine, bypassing setup()."""
    r = runner_mod.MatrixGame2Runner()
    r.engine = engine
    r._dev = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mg2-test")
    r.jpeg_quality = 90
    r.seed_wait = seed_wait
    r._default_seed = "default-seed"
    # base64 → PIL needs a real image; a token keeps the seed path observable.
    r._image_from_b64 = lambda b64: f"client:{b64}"
    return r


def build_app(monkeypatch, engine: FakeEngine, **runner_kw):
    import cognito_testing

    for k in ("WORLD_MODEL_ALLOWED_ORIGINS", "WORLD_MODEL_RATE_LIMIT", "WORLD_MODEL_FLOW_WINDOW"):
        monkeypatch.delenv(k, raising=False)
    cognito_testing.disable_auth(monkeypatch)

    import lib.app as app_mod
    importlib.reload(app_mod)
    from lib.jobs import JobStore

    return app_mod.create_app(make_runner(engine, **runner_kw), JobStore())


def client(app):
    from starlette.testclient import TestClient
    return TestClient(app)


def next_frame(ws) -> bytes:
    while True:
        msg = ws.receive()
        if msg.get("bytes") is not None:
            return msg["bytes"]
        if msg["type"] == "websocket.close":
            raise AssertionError(f"socket closed early: {msg}")


def drain(ws, n: int, ack_from: int = 0) -> list:
    out = []
    for i in range(n):
        out.append(next_frame(ws))
        ws.send_json({"type": "ack", "n": ack_from + i + 1})
    return out


def grey(payload: bytes) -> int:
    img = PIL_Image.open(io.BytesIO(payload)).convert("L")
    return round(np.asarray(img).mean())


def binary(keys: str = "", yaw: float = 0.0) -> bytes:
    flags = sum(1 << i for i, k in enumerate("wsad") if k in keys)
    return struct.pack("<Iff", flags, 0.0, yaw)


def wait_for(predicate, timeout: float = 3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


# --------------------------------------------------------------------------


def test_handshake_then_jpeg_frames(monkeypatch):
    with client(build_app(monkeypatch, FakeEngine())).websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "connected"
        ws.send_bytes(binary())
        assert next_frame(ws)[:2] == b"\xff\xd8"


def test_without_a_client_seed_the_default_world_starts(monkeypatch):
    engine = FakeEngine()
    with client(build_app(monkeypatch, engine)).websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_bytes(binary())
        next_frame(ws)
    assert engine.starts[0] == "default-seed"


def test_a_start_message_seeds_the_world_before_the_first_chunk(monkeypatch):
    # The kiosk sends `start` with its chosen world right after `connected`;
    # the runner must not burn a chunk (and a seed encode) on the default first.
    engine = FakeEngine()
    with client(build_app(monkeypatch, engine)).websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_text(json.dumps({"type": "start", "image_data": "data:image/png;base64,QUJD"}))
        next_frame(ws)
    assert engine.starts[0] == "client:QUJD"
    assert "default-seed" not in engine.starts


def test_a_start_followed_by_ticks_still_seeds_the_world(monkeypatch):
    # ActionBuffer keeps only the latest payload; a kiosk that sends `start`
    # and then its first key ticks must not lose the world it chose.
    engine = FakeEngine()
    with client(build_app(monkeypatch, engine)).websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_text(json.dumps({"type": "start", "image_data": "QUJD"}))
        for _ in range(5):
            ws.send_bytes(binary("w"))
        drain(ws, 8)
    assert engine.starts == ["client:QUJD"]


def test_a_start_is_applied_once_not_every_tick(monkeypatch):
    # The buffer re-returns its latest payload every 0.1 s; re-applying a
    # `start` would re-seed the world ten times a second.
    engine = FakeEngine()
    with client(build_app(monkeypatch, engine)).websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_text(json.dumps({"type": "start", "image_data": "QUJD"}))
        drain(ws, 12)
        time.sleep(0.4)
        drain(ws, 4, ack_from=12)
    assert engine.starts == ["client:QUJD"]


def test_a_malformed_message_does_not_freeze_input(monkeypatch):
    engine = FakeEngine()
    with client(build_app(monkeypatch, engine)).websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_text('{"type":"control","buttons":5}')
        ws.send_bytes(binary("w"))
        assert wait_for(lambda: ((1, 0, 0, 0), 0) in engine.steps)
        drain(ws, 4)


def test_a_failed_step_ends_the_session_and_the_next_one_works(monkeypatch):
    engine = FakeEngine()
    engine.fail_at = 2
    c = client(build_app(monkeypatch, engine))
    with c.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_bytes(binary())
        with pytest.raises(Exception):
            drain(ws, 50)
    engine.fail_at = None
    with c.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_bytes(binary())
        assert next_frame(ws)[:2] == b"\xff\xd8"


def test_a_later_start_reseeds_mid_session(monkeypatch):
    engine = FakeEngine()
    with client(build_app(monkeypatch, engine)).websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_bytes(binary())
        drain(ws, 4)
        ws.send_text(json.dumps({"type": "start", "image_data": "WFla"}))
        assert wait_for(lambda: "client:WFla" in engine.starts)
        drain(ws, 8, ack_from=4)


def test_catalogue_binary_action_reaches_the_engine(monkeypatch):
    engine = FakeEngine()
    with client(build_app(monkeypatch, engine)).websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_bytes(binary("wd", yaw=0.1))
        assert wait_for(lambda: ((1, 0, 0, 1), 1) in engine.steps)
        drain(ws, 4)


def test_unknown_message_types_do_not_cancel_held_keys(monkeypatch):
    # ping/ack never reach the runner (lib.app answers them); an unknown type
    # does, and must read as "no change" rather than a release.
    engine = FakeEngine()
    with client(build_app(monkeypatch, engine)).websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_bytes(binary("w"))
        assert wait_for(lambda: engine.steps and engine.steps[-1][0] == (1, 0, 0, 0))
        ws.send_text(json.dumps({"type": "hello", "client": "kiosk"}))
        drain(ws, 8)
        n = len(engine.steps)
        assert wait_for(lambda: len(engine.steps) > n + 1)
        assert engine.steps[-1][0] == (1, 0, 0, 0)


def test_releasing_every_key_stops_the_walk(monkeypatch):
    engine = FakeEngine()
    with client(build_app(monkeypatch, engine)).websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_bytes(binary("w"))
        assert wait_for(lambda: ((1, 0, 0, 0), 0) in engine.steps)
        ws.send_text('{"type":"control","buttons":[]}')
        drain(ws, 8)
        assert wait_for(lambda: engine.steps[-1] == ((0, 0, 0, 0), 0))


def test_every_frame_of_every_chunk_arrives_in_order(monkeypatch):
    engine = FakeEngine(frames_per_chunk=4)
    with client(build_app(monkeypatch, engine)).websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_bytes(binary())
        greys = [grey(p) for p in drain(ws, 12)]
    expected = [frame_value(c, t) for c in range(3) for t in range(4)]
    assert greys == pytest.approx(expected, abs=2)


def test_frames_are_paced_not_burst(monkeypatch):
    # 12 frames per chunk sent at once would be mostly dropped by the player,
    # which keeps only the newest undecoded frame.
    engine = FakeEngine(frames_per_chunk=12, chunk_seconds=0.24)
    with client(build_app(monkeypatch, engine)).websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_bytes(binary())
        drain(ws, 24)  # chunks 0-1 calibrate the pacer
        t0 = time.perf_counter()
        drain(ws, 12, ack_from=24)
        took = time.perf_counter() - t0
    # Paced at ~0.24 s / 12 frames; a burst would arrive in a few ms.
    assert took > 0.12


def test_reset_ends_the_previous_world(monkeypatch):
    engine = FakeEngine()
    c = client(build_app(monkeypatch, engine))
    for _ in range(2):
        with c.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.send_bytes(binary())
            next_frame(ws)
    assert engine.ends >= 2
    assert engine.starts == ["default-seed", "default-seed"]


def test_a_second_player_preempts_the_first(monkeypatch):
    engine = FakeEngine()
    c = client(build_app(monkeypatch, engine))
    with c.websocket_connect("/ws") as first:
        first.receive_json()
        first.send_bytes(binary())
        next_frame(first)
        with c.websocket_connect("/ws") as second:
            assert second.receive_json()["type"] == "connected"
            second.send_bytes(binary())
            assert next_frame(second)[:2] == b"\xff\xd8"
    # Each session start ends the previous world on the engine.
    assert engine.ends >= 2


def test_generation_stops_when_the_player_leaves(monkeypatch):
    engine = FakeEngine()
    with client(build_app(monkeypatch, engine)).websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_bytes(binary())
        drain(ws, 8)
    last = -1
    for _ in range(40):
        time.sleep(0.05)
        if len(engine.steps) == last:
            break
        last = len(engine.steps)
    else:
        raise AssertionError("engine still generating after the player left")


def test_action_buffer_keeps_a_seed_apart_from_actions():
    from lib.runner import ActionBuffer

    buf = ActionBuffer()
    buf.put_seed(b'{"type":"start"}')
    buf.put(b"tick")
    assert buf.get() == b"tick"
    assert buf.take_seed() == b'{"type":"start"}'
    assert buf.take_seed() is None
