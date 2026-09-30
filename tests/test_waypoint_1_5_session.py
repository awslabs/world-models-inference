# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end session tests for the Waypoint-1.5 cartridge.

test_waypoint_1_5.py unit-tests the codecs in isolation. This drives a whole
session the way a player does: the real FastAPI /ws route from lib.app, the real
WaypointRunner.stream() generator with its action pump and one-step-ahead
pipeline, and the real JPEG encoder — with only the GPU-side pieces faked.

What is faked, and why:
  world_engine   a GPL-3.0 CUDA library that exists only inside the cartridge
                 image. A fake with the same three-method surface
                 (reset/append_frame/gen_frame) lets the adapter run anywhere.
  torch          stubbed at import time, as in test_app_integration.py.
  _seed_from_b64 base64 → device tensor needs real torch; the tests replace it
                 with a token so they can assert on the *reaction* to a client
                 seed (reset + append) rather than on tensor plumbing.

Everything between the socket and those seams is the shipping code, so these
tests catch what the unit tests cannot: frames arriving in the wrong order,
held keys being cancelled by protocol chatter, a seed not reaching the engine,
or a session that never ends when the player leaves.
"""

import importlib
import importlib.util
import io
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
CARTRIDGE_DIR = INFERENCE_DIR / "models" / "waypoint-1-5"

if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))

W, A, S = 87, 65, 83  # Windows virtual-key codes, what world_engine expects
FRAME_H, FRAME_W = 64, 64


# --------------------------------------------------------------------------
# Importing the cartridge
# --------------------------------------------------------------------------


def _stub_torch():
    """Stub torch so lib.app and the cartridge import without a GPU.

    Mirrors tests/test_app_integration.py: same `_wm_stub` marker, so whichever
    file runs first the other reuses it instead of installing a second stub.
    """
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


def _load_runner_module():
    """Import the cartridge's runner.py with its sibling modules bound.

    runner.py says `import actions` / `import frames` — bare names that resolve
    against the cartridge directory inside the container. Other cartridges ship
    modules by the same names, so bind ours explicitly rather than trusting
    sys.path ordering or whatever a previously-run test file left behind.
    """
    _stub_torch()
    saved = {name: sys.modules.get(name) for name in ("actions", "frames", "runner")}
    try:
        for name in ("actions", "frames", "runner"):
            spec = importlib.util.spec_from_file_location(
                name, CARTRIDGE_DIR / f"{name}.py"
            )
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
        return sys.modules["runner"], sys.modules["frames"]
    finally:
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


runner_mod, frames_mod = _load_runner_module()


# --------------------------------------------------------------------------
# Fake engine
# --------------------------------------------------------------------------


class FakeCtrl:
    """Stand-in for world_engine.CtrlInput — a plain record of the inputs."""

    def __init__(self, button=(), mouse=(0.0, 0.0)):
        self.button = set(button)
        self.mouse = tuple(mouse)


def frame_value(step: int, t: int) -> int:
    """Grey level identifying (engine step, frame within the step).

    Flat frames 10 grey levels apart survive JPEG quality 60 intact, so a test
    can read the step and slot back out of a delivered frame and prove ordering.
    """
    return 20 + (step % 5) * 40 + t * 10


class FakeEngine:
    """The three methods the adapter calls, plus a log of how it called them."""

    def __init__(self, temporal_compression: int = 4):
        self.model_cfg = types.SimpleNamespace(temporal_compression=temporal_compression)
        self.temporal_compression = temporal_compression
        self.resets = 0
        self.appended: list = []
        self.ctrls: list = []

    def reset(self) -> None:
        self.resets += 1

    def append_frame(self, seed) -> None:
        self.appended.append(seed)

    def gen_frame(self, ctrl):
        step = len(self.ctrls)
        self.ctrls.append(ctrl)
        out = np.empty((self.temporal_compression, FRAME_H, FRAME_W, 3), dtype=np.uint8)
        for t in range(self.temporal_compression):
            out[t] = frame_value(step, t)
        return out


def make_runner(engine: FakeEngine):
    """A WaypointRunner wired to a fake engine, bypassing setup().

    setup() only loads weights and compiles CUDA graphs; every field it sets is
    set here instead, so stream()/reset() are the shipping implementations.
    """
    r = runner_mod.WaypointRunner()
    r.engine = engine
    r._dev = ThreadPoolExecutor(max_workers=1, thread_name_prefix="waypoint-test")
    r._ctrl_cls = FakeCtrl
    r.temporal_compression = engine.temporal_compression
    r.jpeg_quality = frames_mod.DEFAULT_QUALITY
    r.jpeg_subsampling = frames_mod.DEFAULT_SUBSAMPLING
    r._seed = "boot-seed"  # opaque to the engine; the fake just records it
    # base64 → device tensor needs real torch (covered by the deploy smoke
    # test). Returning a token keeps the reaction to a seed observable.
    r._seed_from_b64 = lambda b64: f"client-seed:{b64}"
    return r


def build_app(monkeypatch, engine: FakeEngine, *, flow_window=None):
    import cognito_testing

    for k in ("WORLD_MODEL_ALLOWED_ORIGINS",
              "WORLD_MODEL_RATE_LIMIT", "WORLD_MODEL_FLOW_WINDOW"):
        monkeypatch.delenv(k, raising=False)
    # These tests exercise session behaviour, not auth, so run the app open —
    # the equivalent of the old no-token default now the server fails closed.
    cognito_testing.disable_auth(monkeypatch)
    if flow_window is not None:
        monkeypatch.setenv("WORLD_MODEL_FLOW_WINDOW", str(flow_window))

    import lib.app as app_mod
    importlib.reload(app_mod)
    from lib.jobs import JobStore

    runner = make_runner(engine)
    app = app_mod.create_app(runner, JobStore())
    # create_app does not publish the runner; stash it so a test can reach the
    # instance the route actually holds.
    app.state.runner = runner
    return app


def client(app):
    from starlette.testclient import TestClient
    return TestClient(app)


# --------------------------------------------------------------------------
# Reading the socket
# --------------------------------------------------------------------------


def next_frame(ws) -> bytes:
    """Next binary frame, skipping the server's text messages.

    The route interleaves JSON ('started', 'pong') with frames, so a test that
    called receive_bytes() blindly would fail on the reply to its own message.
    """
    while True:
        msg = ws.receive()
        if msg.get("bytes") is not None:
            return msg["bytes"]
        if msg["type"] == "websocket.close":
            raise AssertionError(f"socket closed early: {msg}")


def grey(payload: bytes) -> int:
    img = PIL_Image.open(io.BytesIO(payload)).convert("L")
    assert img.size == (FRAME_W, FRAME_H)
    return round(np.asarray(img).mean())


def drain(ws, n: int, ack_from: int = 0) -> list:
    """Receive n frames, acknowledging as a real client does."""
    out = []
    for i in range(n):
        out.append(next_frame(ws))
        ws.send_json({"type": "ack", "n": ack_from + i + 1})
    return out


# --------------------------------------------------------------------------
# Handshake and session lifecycle
# --------------------------------------------------------------------------


def test_handshake_then_frames(monkeypatch):
    engine = FakeEngine()
    with client(build_app(monkeypatch, engine)).websocket_connect("/ws") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "connected"
        assert hello["flow_window"] == 16
        assert next_frame(ws)[:2] == b"\xff\xd8"  # JPEG SOI


def test_session_start_seeds_a_fresh_world(monkeypatch):
    # Each player must get the boot scene, not the previous player's world.
    engine = FakeEngine()
    with client(build_app(monkeypatch, engine)).websocket_connect("/ws") as ws:
        ws.receive_json()
        next_frame(ws)
    assert engine.resets == 1
    assert engine.appended == ["boot-seed"]


def test_a_second_player_preempts_the_first(monkeypatch):
    # One engine, one KV cache: two concurrent sessions would reset the world
    # under each other. The route serialises them, last player wins — the new
    # connection closes the active session's action buffer and takes over once
    # its loop unwinds. (Preemption rather than refusal: a refused-second
    # design can lock everyone out after a dirty disconnect, and React
    # StrictMode's double socket on mount would refuse the real player.)
    c = client(build_app(monkeypatch, FakeEngine()))
    with c.websocket_connect("/ws") as first:
        first.receive_json()
        next_frame(first)
        with c.websocket_connect("/ws") as second:
            assert second.receive_json()["type"] == "connected"
            assert next_frame(second)[:2] == b"\xff\xd8"


def test_the_next_player_gets_a_session(monkeypatch):
    # The single-session guard has to clear on disconnect, or the endpoint is
    # dead to everyone after the first player leaves.
    c = client(build_app(monkeypatch, FakeEngine()))
    for _ in range(2):
        with c.websocket_connect("/ws") as ws:
            assert ws.receive_json()["type"] == "connected"
            assert next_frame(ws)[:2] == b"\xff\xd8"


def test_generation_stops_when_the_player_leaves(monkeypatch):
    # A leaked generator would hold the GPU and lock out every later session.
    engine = FakeEngine()
    with client(build_app(monkeypatch, engine)).websocket_connect("/ws") as ws:
        ws.receive_json()
        drain(ws, 8)
    settled = _wait_until_quiet(engine)
    assert settled >= 2, "engine should have produced frames during the session"


def _wait_until_quiet(engine: FakeEngine, tries: int = 40) -> int:
    """Poll until gen_frame stops being called, and return the final count."""
    import time
    last = -1
    for _ in range(tries):
        time.sleep(0.05)
        count = len(engine.ctrls)
        if count == last:
            return count
        last = count
    raise AssertionError(f"engine still generating after teardown ({last} steps)")


# --------------------------------------------------------------------------
# Frame delivery
# --------------------------------------------------------------------------


def test_each_engine_step_delivers_four_frames_in_order(monkeypatch):
    # The TAEHV autoencoder decodes one latent step into temporal_compression
    # RGB frames. They must reach the player individually and in order — a
    # dropped or reordered frame is visible as a stutter.
    engine = FakeEngine()
    with client(build_app(monkeypatch, engine)).websocket_connect("/ws") as ws:
        ws.receive_json()
        greys = [grey(p) for p in drain(ws, 8)]
    expected = [frame_value(step, t) for step in range(2) for t in range(4)]
    assert greys == pytest.approx(expected, abs=2)


class TexturedEngine(FakeEngine):
    """Detailed frames, so JPEG settings actually change the byte count.

    FakeEngine's flat greys are all DC coefficient — they compress to the same
    size at any quality, which would make a quality assertion pass vacuously.
    """

    def gen_frame(self, ctrl):
        step = len(self.ctrls)
        self.ctrls.append(ctrl)
        rng = np.random.default_rng(step)
        return rng.integers(
            0, 256, size=(self.temporal_compression, FRAME_H, FRAME_W, 3),
            dtype=np.uint8,
        )


def test_frames_honour_the_configured_quality(monkeypatch):
    # jpeg_quality is the operator's bandwidth dial; if stream() ignored it the
    # knob documented in the README would silently do nothing.
    app = build_app(monkeypatch, TexturedEngine())
    with client(app).websocket_connect("/ws") as ws:
        ws.receive_json()
        default = len(next_frame(ws))

    app2 = build_app(monkeypatch, TexturedEngine())
    app2.state.runner.jpeg_quality = 95
    with client(app2).websocket_connect("/ws") as ws:
        ws.receive_json()
        high = len(next_frame(ws))
    assert high > default


def test_frames_honour_the_configured_subsampling(monkeypatch):
    # 4:2:0 is the shipped default because it is 15.2% fewer bytes than 4:4:4 on
    # this model's output, and delivered fps is bytes-bound. Prove the choice
    # reaches the wire rather than sitting unread in the runner.
    app = build_app(monkeypatch, TexturedEngine())
    with client(app).websocket_connect("/ws") as ws:
        ws.receive_json()
        default = len(next_frame(ws))

    app2 = build_app(monkeypatch, TexturedEngine())
    app2.state.runner.jpeg_subsampling = "444"
    with client(app2).websocket_connect("/ws") as ws:
        ws.receive_json()
        full_chroma = len(next_frame(ws))
    assert full_chroma > default


# --------------------------------------------------------------------------
# Controls
# --------------------------------------------------------------------------


def test_the_world_starts_neutral(monkeypatch):
    # Nothing is pressed before the player presses anything: a phantom input
    # here would make every session drift on connect.
    engine = FakeEngine()
    with client(build_app(monkeypatch, engine)).websocket_connect("/ws") as ws:
        ws.receive_json()
        drain(ws, 4)
    assert engine.ctrls[0].button == set()
    assert engine.ctrls[0].mouse == (0.0, 0.0)


def test_held_keys_reach_the_engine_as_vk_codes(monkeypatch):
    engine = FakeEngine()
    monkeypatch.setenv("WAYPOINT_MOUSE_SCALE", "10")
    with client(build_app(monkeypatch, engine)).websocket_connect("/ws") as ws:
        ws.receive_json()
        drain(ws, 4)
        ws.send_json({"type": "control", "buttons": ["W", "A"],
                      "mouse_dx": 100, "mouse_dy": 0})
        ctrl = _wait_for_ctrl(engine, ws, lambda c: c.button == {W, A})
    assert ctrl.mouse == pytest.approx((10.0, 0.0))


def test_keys_stay_held_while_the_client_acknowledges_frames(monkeypatch):
    # Acknowledgements are protocol chatter. If one reached the action buffer,
    # the runner would decode it as a neutral action and drop the player's
    # keys — the bug this asserts against is a walk that stops by itself.
    engine = FakeEngine()
    with client(build_app(monkeypatch, engine)).websocket_connect("/ws") as ws:
        ws.receive_json()
        drain(ws, 2)
        ws.send_json({"type": "control", "buttons": ["W"], "mouse_dx": 0, "mouse_dy": 0})
        _wait_for_ctrl(engine, ws, lambda c: c.button == {W})
        first_held = len(engine.ctrls)
        drain(ws, 12, ack_from=2)  # nothing but acks from here on
    later = engine.ctrls[first_held:]
    assert later, "expected further engine steps while acknowledging"
    assert all(c.button == {W} for c in later)


def test_a_released_key_stops_the_player(monkeypatch):
    engine = FakeEngine()
    with client(build_app(monkeypatch, engine)).websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "control", "buttons": ["S"], "mouse_dx": 0, "mouse_dy": 0})
        _wait_for_ctrl(engine, ws, lambda c: c.button == {S})
        ws.send_json({"type": "control", "buttons": [], "mouse_dx": 0, "mouse_dy": 0})
        _wait_for_ctrl(engine, ws, lambda c: c.button == set())


def test_binary_actions_from_a_legacy_client(monkeypatch):
    # The matrix-game-3 test harness sends packed binary actions; the shared
    # route forwards raw bytes, so this cartridge has to decode them too.
    import struct
    engine = FakeEngine()
    monkeypatch.setenv("WAYPOINT_MOUSE_SCALE", "10")
    with client(build_app(monkeypatch, engine)).websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_bytes(struct.pack("<Iff", 0b0001, 0.0, 0.1))  # w held, yaw right
        ctrl = _wait_for_ctrl(engine, ws, lambda c: c.button == {W})
    assert ctrl.mouse == pytest.approx((10.0, 0.0))


def _wait_for_ctrl(engine: FakeEngine, ws, predicate, frames: int = 60):
    """Consume frames until the engine sees a matching control, and return it.

    Actions and generation are deliberately decoupled — the pump thread stores
    the latest input and the device thread picks it up on its next step — so a
    control message takes effect within a step or two, not instantly.
    """
    for _ in range(frames):
        for ctrl in engine.ctrls:
            if predicate(ctrl):
                return ctrl
        next_frame(ws)
    raise AssertionError(
        "engine never saw a matching control; saw "
        f"{[(sorted(c.button), c.mouse) for c in engine.ctrls]}"
    )


# --------------------------------------------------------------------------
# Client-supplied seed image
# --------------------------------------------------------------------------


def test_a_client_seed_restarts_the_world(monkeypatch):
    # Picking a scene in the catalogue UI sends `start` with the image inline.
    engine = FakeEngine()
    with client(build_app(monkeypatch, engine)).websocket_connect("/ws") as ws:
        ws.receive_json()
        next_frame(ws)
        ws.send_json({"type": "start", "image_data": "data:image/png;base64,QUJD"})
        for _ in range(60):
            if len(engine.appended) > 1:
                break
            next_frame(ws)
    assert engine.appended == ["boot-seed", "client-seed:QUJD"]
    assert engine.resets == 2, "a new scene must clear the frame context first"


def test_a_seed_is_applied_once_even_if_the_client_goes_quiet(monkeypatch):
    # ActionBuffer.get() re-returns the last payload every 0.1s whether or not
    # the client sent anything, so the pump sees the same `start` over and over.
    # It used to act on each one: reset() + append_frame() ~10x a second, which
    # silently threw away the frame context of anyone who seeded a scene and
    # then stopped pressing keys. Ack only — no further input — for well over
    # one poll interval, and the world must still have been seeded exactly once.
    engine = FakeEngine()
    with client(build_app(monkeypatch, engine)).websocket_connect("/ws") as ws:
        ws.receive_json()
        next_frame(ws)
        ws.send_json({"type": "start", "image_data": "data:image/png;base64,QUJD"})
        for _ in range(60):
            if len(engine.appended) > 1:
                break
            next_frame(ws)
        deadline = time.monotonic() + 0.5      # 5x the 0.1s ActionBuffer poll
        received = 0
        while time.monotonic() < deadline:
            next_frame(ws)
            received += 1
            ws.send_json({"type": "ack", "n": received})
    assert engine.appended == ["boot-seed", "client-seed:QUJD"]
    assert engine.resets == 2


def test_an_undecodable_seed_leaves_the_world_alone(monkeypatch):
    # A corrupt upload should cost the player nothing; resetting on a failed
    # decode would blank the world they were already playing.
    engine = FakeEngine()
    app = build_app(monkeypatch, engine)
    app.state.runner._seed_from_b64 = lambda b64: None
    with client(app).websocket_connect("/ws") as ws:
        ws.receive_json()
        next_frame(ws)
        ws.send_json({"type": "start", "image_data": "data:image/png;base64,!!!"})
        drain(ws, 8)
    assert engine.appended == ["boot-seed"]
    assert engine.resets == 1


def test_start_without_an_image_keeps_the_boot_scene(monkeypatch):
    # The plain 'start' handshake (benchmark script, matrix-game-3 clients)
    # carries no image and must not be mistaken for a scene change.
    engine = FakeEngine()
    with client(build_app(monkeypatch, engine)).websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "start"})
        drain(ws, 8)
    assert engine.appended == ["boot-seed"]


# --------------------------------------------------------------------------
# Contract with the shared framework
# --------------------------------------------------------------------------


def test_runner_declares_streaming_support():
    assert make_runner(FakeEngine()).supports_streaming is True


def test_http_generate_is_refused(monkeypatch):
    # A real-time cartridge has no batch mode. /generate must fail loudly
    # rather than return an empty job someone waits on.
    with pytest.raises(NotImplementedError, match="/ws"):
        make_runner(FakeEngine()).generate(prompt="x")


def test_follower_ranks_hold_no_engine():
    # Single-GPU model. If it is ever launched multi-rank, followers must not
    # try to serve; stream() and reset() return immediately.
    r = runner_mod.WaypointRunner()
    r.engine = None
    r.reset()  # must not raise
    assert list(r.stream(None)) == []


# --------------------------------------------------------------------------
# setup() — the env knobs
# --------------------------------------------------------------------------
#
# make_runner() sets jpeg_quality/jpeg_subsampling by hand, so every test above
# proves the *encoder* honours them and none prove setup() ever reads the
# container environment. That is the half that ships: a knob documented in the
# README but never wired would be invisible here and silently ignored in
# production. Follower ranks return from setup() right after resolving the two
# knobs and before touching a GPU, which is the seam these tests use.


def _setup_follower(monkeypatch, **env):
    for key in ("WAYPOINT_JPEG_QUALITY", "WAYPOINT_JPEG_SUBSAMPLING"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    r = runner_mod.WaypointRunner()
    r.setup("/nonexistent", device=None, rank=1, world_size=2)
    assert r.engine is None, "the follower path must not build an engine"
    return r


def test_setup_defaults_the_encoding_knobs(monkeypatch):
    r = _setup_follower(monkeypatch)
    assert r.jpeg_quality == frames_mod.DEFAULT_QUALITY
    assert r.jpeg_subsampling == frames_mod.DEFAULT_SUBSAMPLING


def test_setup_reads_the_encoding_knobs_from_the_environment(monkeypatch):
    r = _setup_follower(
        monkeypatch,
        WAYPOINT_JPEG_QUALITY="85",
        WAYPOINT_JPEG_SUBSAMPLING="444",
    )
    assert r.jpeg_quality == 85
    assert r.jpeg_subsampling == "444"


@pytest.mark.parametrize("quality,subsampling", [
    ("not-a-number", "411"),   # unparseable and unsupported
    ("0", ""),                 # out of libjpeg's range, and empty
    ("500", "4:2:0"),          # above the range, and the wrong spelling
])
def test_setup_survives_a_mistyped_knob(monkeypatch, quality, subsampling):
    # A typo in the container environment must cost one warning at boot, not a
    # dead endpoint or an exception on every frame. Whatever comes back has to
    # be something the encoder will actually accept.
    r = _setup_follower(
        monkeypatch,
        WAYPOINT_JPEG_QUALITY=quality,
        WAYPOINT_JPEG_SUBSAMPLING=subsampling,
    )
    assert 1 <= r.jpeg_quality <= 100
    assert r.jpeg_subsampling in frames_mod.SUBSAMPLINGS
    frames_mod.encode_jpeg(
        np.zeros((8, 8, 3), dtype=np.uint8), r.jpeg_quality, r.jpeg_subsampling
    )


# --------------------------------------------------------------------------
# Flow control
# --------------------------------------------------------------------------


def test_the_flow_window_is_configurable(monkeypatch):
    # The window is the latency/throughput dial (README: 1799 ms unbounded vs
    # 201 ms bounded). The client is told the value at handshake so it knows to
    # acknowledge, so a mis-plumbed env var would leave every client guessing.
    app = build_app(monkeypatch, FakeEngine(), flow_window=4)
    with client(app).websocket_connect("/ws") as ws:
        assert ws.receive_json() == {"type": "connected", "flow_window": 4}


def test_a_zero_flow_window_sends_unbounded(monkeypatch):
    # 0 is the documented escape hatch for a client that cannot ack. It must
    # actually disable the gate: with a window of 4 a non-acking client would
    # stall after 4 frames, so reading many more than that without ever acking
    # is the proof.
    app = build_app(monkeypatch, FakeEngine(), flow_window=0)
    with client(app).websocket_connect("/ws") as ws:
        assert ws.receive_json() == {"type": "connected", "flow_window": 0}
        for _ in range(40):
            assert len(next_frame(ws)) > 0


# --------------------------------------------------------------------------
# setup() — engine construction
# --------------------------------------------------------------------------
#
# Everything above bypasses setup() on the rank-0 path, so none of it proves
# the engine is *constructed* correctly: which quantisation reaches
# WorldEngine (the fp8→int8 downgrade exists because warmup died on an A10G),
# whether the locally staged autoencoder is used instead of the Hub, and
# whether warmup actually compiles a frame. These fake `world_engine` itself
# and record the constructor call.


class RecordingWorldEngine(FakeEngine):
    """FakeEngine that also records how the adapter constructed it."""

    last_build: dict = {}

    def __init__(self, ckpt_dir, device=None, quant=None, dtype=None,
                 model_config_overrides=None):
        super().__init__()
        type(self).last_build = {
            "ckpt_dir": ckpt_dir,
            "device": device,
            "quant": quant,
            "model_config_overrides": model_config_overrides,
        }


def _setup_rank0(monkeypatch, ckpt_dir, *, capability=(9, 0), **env):
    """Run the real setup() with world_engine and the GPU probe faked."""
    for key in ("WAYPOINT_QUANT", "WAYPOINT_WARMUP", "WAYPOINT_SEED",
                "WAYPOINT_JPEG_QUALITY", "WAYPOINT_JPEG_SUBSAMPLING"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("WAYPOINT_WARMUP", "0")
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    monkeypatch.setattr(runner_mod.torch, "cuda", types.SimpleNamespace(
        is_available=lambda: True,
        get_device_capability=lambda device: capability,
    ), raising=False)
    # The stub torch has no dtypes; setup() passes torch.bfloat16 through to
    # the (faked) engine, so any sentinel will do.
    monkeypatch.setattr(runner_mod.torch, "bfloat16", object(), raising=False)

    fake_we = types.ModuleType("world_engine")
    fake_we.WorldEngine = RecordingWorldEngine
    fake_we.CtrlInput = FakeCtrl
    monkeypatch.setitem(sys.modules, "world_engine", fake_we)

    # Seed loading is tensor plumbing (real torch); its selection logic has its
    # own tests below.
    monkeypatch.setattr(runner_mod.WaypointRunner, "_load_default_seed",
                        lambda self: "boot-seed")

    r = runner_mod.WaypointRunner()
    r.setup(str(ckpt_dir), device="cuda:0", rank=0, world_size=1)
    return r


def test_fp8_downgrades_to_int8_below_ada(monkeypatch, tmp_path):
    # torch._scaled_mm needs compute capability 8.9+. On an Ampere A10G the
    # default fp8w8a8 config used to die inside warmup; the downgrade must
    # happen before WorldEngine ever sees the quant string.
    _setup_rank0(monkeypatch, tmp_path, capability=(8, 6))
    assert RecordingWorldEngine.last_build["quant"] == "intw8a8"


def test_fp8_is_kept_on_ada_and_newer(monkeypatch, tmp_path):
    _setup_rank0(monkeypatch, tmp_path, capability=(8, 9))
    assert RecordingWorldEngine.last_build["quant"] == "fp8w8a8"


def test_quant_none_disables_quantisation(monkeypatch, tmp_path):
    # The README documents WAYPOINT_QUANT=none for debugging quality issues;
    # world_engine expects a literal None, not the string "none".
    _setup_rank0(monkeypatch, tmp_path, WAYPOINT_QUANT="none")
    assert RecordingWorldEngine.last_build["quant"] is None


def test_staged_autoencoder_beats_the_hub(monkeypatch, tmp_path):
    # The weights sync stages taehv1_5/ next to the model. If the override is
    # not passed the engine fetches the autoencoder from Hugging Face at boot —
    # which fails inside the VPC and is exactly the kind of silent network
    # dependency the staging exists to remove.
    (tmp_path / "taehv1_5").mkdir()
    _setup_rank0(monkeypatch, tmp_path)
    overrides = RecordingWorldEngine.last_build["model_config_overrides"]
    assert overrides == {"ae_uri": str(tmp_path / "taehv1_5")}


def test_no_staged_autoencoder_means_no_override(monkeypatch, tmp_path):
    _setup_rank0(monkeypatch, tmp_path)
    assert RecordingWorldEngine.last_build["model_config_overrides"] is None


def test_warmup_compiles_exactly_one_frame(monkeypatch, tmp_path):
    # WAYPOINT_WARMUP=1 must drive the whole compile path — reset, seed,
    # one gen_frame — on the device thread. A warmup that silently does
    # nothing moves the ~40s torch.compile stall onto the first player.
    r = _setup_rank0(monkeypatch, tmp_path, WAYPOINT_WARMUP="1")
    assert r.engine.resets == 1
    assert r.engine.appended == ["boot-seed"]
    assert len(r.engine.ctrls) == 1


def test_create_runner_factory():
    assert isinstance(runner_mod.create_runner(), runner_mod.WaypointRunner)


# --------------------------------------------------------------------------
# _load_default_seed — where the boot scene comes from
# --------------------------------------------------------------------------
#
# The chain is: WAYPOINT_SEED file → first frame of the demo video in the
# weights repo → a synthetic gradient. Only the selection is tested here —
# _seed_from_array (resize + dtype + device) is real-torch tensor plumbing,
# so it is replaced with a recorder.


def _seed_runner(monkeypatch, ckpt_dir):
    captured = {}

    def record(self, rgb):
        captured["rgb"] = rgb
        return "seed-tensor"

    monkeypatch.setattr(runner_mod.WaypointRunner, "_seed_from_array", record)
    r = runner_mod.WaypointRunner()
    r.ckpt_dir = str(ckpt_dir)
    return r, captured


def test_seed_env_var_wins(monkeypatch, tmp_path):
    img_path = tmp_path / "seed.png"
    PIL_Image.new("RGB", (10, 8), (200, 30, 30)).save(img_path)
    monkeypatch.setenv("WAYPOINT_SEED", str(img_path))

    r, captured = _seed_runner(monkeypatch, tmp_path)
    assert r._load_default_seed() == "seed-tensor"
    assert captured["rgb"].shape == (8, 10, 3)
    assert (captured["rgb"] == (200, 30, 30)).all()


def test_missing_everything_falls_back_to_a_gradient(monkeypatch, tmp_path):
    # No env var, no demo video in the weights: setup must still succeed
    # (README: "never fail setup over a missing picture"), with a full-size
    # ramp so the first generated frames have structure to anchor on.
    monkeypatch.delenv("WAYPOINT_SEED", raising=False)

    r, captured = _seed_runner(monkeypatch, tmp_path)
    assert r._load_default_seed() == "seed-tensor"
    rgb = captured["rgb"]
    assert rgb.shape == (720, 1280, 3)
    assert rgb.dtype == np.uint8
    assert rgb[0, 0, 0] < rgb[-1, 0, 0], "the ramp should darken top to bottom"


def test_a_dangling_seed_path_falls_through(monkeypatch, tmp_path):
    # A typo'd WAYPOINT_SEED must not crash the boot — the chain continues to
    # the fallbacks as if it were unset.
    monkeypatch.setenv("WAYPOINT_SEED", str(tmp_path / "nope.png"))

    r, captured = _seed_runner(monkeypatch, tmp_path)
    assert r._load_default_seed() == "seed-tensor"
    assert captured["rgb"].shape == (720, 1280, 3)
