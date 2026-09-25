# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end session tests for the vjepa2 cartridge.

vjepa2 is the odd one out in the catalogue. Every other real-time cartridge
*generates* pixels: the client sends input, the server sends JPEG frames. This
one runs the other way — the client streams video frames in and the server
sends embeddings back, with an alert flag when the scene stops resembling the
one the session started with. Nothing exercised that data flow, so the wire
format, the 16-frame warm-up and the alert threshold were all unverified.

These tests drive the real FastAPI /ws route from lib.app and the real
VJepa2Runner.stream(), with the GPU-side pieces faked:

  torch         stubbed at import time, as in test_app_integration.py, plus the
                small tensor surface stream() touches (mean/clone/squeeze/half
                and F.cosine_similarity).
  transformers  a fake AutoModel/AutoVideoProcessor pair. The encoder is
                deterministic in the pixels it is given (see FakeEncoder), so
                cosine distance is a real function of what the client sent
                rather than a constant the assertions could not distinguish
                from a broken window.
  PIL, numpy    real. The client sends real JPEGs.

Everything between the socket and those seams is shipping code, so what these
catch is what a GPU would otherwise have to: a frame counter that does not match
the frames sent, a window that never fills, an embedding truncated by a bad
struct format, or an alert that fires on the baseline itself.
"""

import importlib
import importlib.util
import io
import json
import struct
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
np = pytest.importorskip("numpy")
PIL_Image = pytest.importorskip("PIL.Image")

REPO_ROOT = Path(__file__).resolve().parent.parent
INFERENCE_DIR = REPO_ROOT / "inference"
CARTRIDGE_DIR = INFERENCE_DIR / "models" / "vjepa2"

if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))

EMBED_DIM = 8
FRAME_H, FRAME_W = 64, 64


# --------------------------------------------------------------------------
# Stubs
# --------------------------------------------------------------------------


class FakeTensor:
    """The torch.Tensor surface runner.py touches, over a numpy array.

    Deliberately tiny: if stream() starts using a method that is not here the
    test fails with an AttributeError naming it, which is the signal you want.
    """

    def __init__(self, array):
        self.array = np.asarray(array)

    @property
    def shape(self):
        return self.array.shape

    def mean(self, dim):
        return FakeTensor(self.array.mean(axis=dim))

    def clone(self):
        return FakeTensor(self.array.copy())

    def squeeze(self, dim):
        return FakeTensor(np.squeeze(self.array, axis=dim))

    def half(self):
        return FakeTensor(self.array.astype(np.float16))

    def cpu(self):
        return self

    def numpy(self):
        return self.array


def _cosine_similarity(a, b, dim=1):
    """Real cosine similarity, returning something with .item()."""
    x = np.asarray(a.array, dtype=float).ravel()
    y = np.asarray(b.array, dtype=float).ravel()
    value = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
    return types.SimpleNamespace(item=lambda: value)


class _NoGrad:
    """`@torch.no_grad()` — used as a decorator here, so __call__ returns fn."""

    def __call__(self, fn):
        return fn

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _stub_torch():
    """Stub torch so lib.app and the cartridge import without a GPU.

    Shares the `_wm_stub` marker with test_app_integration.py and
    test_waypoint_1_5_session.py so the three files cooperate rather than
    installing competing stubs; the extras this cartridge needs (no_grad,
    torch.nn.functional) are added to whichever stub got there first.
    """
    torch = sys.modules.get("torch")
    if torch is None or not getattr(torch, "_wm_stub", False):
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

    torch.no_grad = _NoGrad
    if "torch.nn.functional" not in sys.modules:
        nn = types.ModuleType("torch.nn")
        nn.__path__ = []
        functional = types.ModuleType("torch.nn.functional")
        functional.cosine_similarity = _cosine_similarity
        nn.functional = functional
        torch.nn = nn
        sys.modules["torch.nn"] = nn
        sys.modules["torch.nn.functional"] = functional


class FakeEncoder:
    """Deterministic stand-in for the ViT-g vision tower.

    Maps a window to a unit vector whose angle tracks the window's mean
    brightness: theta = mean/255 * pi, embedding = [cos, sin, 0...]. So the
    cosine distance the runner computes is `1 - cos(dtheta)` between two
    windows' brightness, which means

      - the baseline scores exactly 0 against itself,
      - a large scene change moves distance past the 0.5 threshold,
      - and one odd frame among fifteen normal ones is diluted by the window
        mean, so it does not — which is what makes the sliding window testable.

    Returns [1, T*P, D] with two identical patch rows, so the runner's
    .mean(dim=1) recovers the vector exactly.
    """

    def __init__(self):
        self.calls = 0
        self.windows: list = []

    def get_vision_features(self, pixel_values_videos=None, **kwargs):
        self.calls += 1
        video = np.asarray(pixel_values_videos, dtype=float)
        self.windows.append(video.shape)
        theta = float(video.mean() / 255.0 * np.pi)
        vector = np.zeros(EMBED_DIM)
        vector[0], vector[1] = np.cos(theta), np.sin(theta)
        return FakeTensor(np.stack([vector, vector])[None, ...])

    def to(self, device):
        return self

    def eval(self):
        return self


class _Inputs(dict):
    """What the processor returns: a mapping, since stream() does **inputs."""

    def to(self, device):
        return self


class FakeProcessor:
    def __call__(self, video, return_tensors=None):
        return _Inputs(pixel_values_videos=np.asarray(video))


def _stub_transformers():
    """AutoModel/AutoVideoProcessor, recording the from_pretrained arguments.

    setup() picks between a local directory and a pinned Hub revision, and the
    pin is the supply-chain control — worth asserting on, and it cannot be
    asserted on without capturing the call.
    """
    module = types.ModuleType("transformers")
    module.calls = []

    class _Auto:
        kind = "model"

        @classmethod
        def from_pretrained(cls, path, revision=None, **kwargs):
            module.calls.append({"kind": cls.kind, "path": path, "revision": revision})
            return FakeEncoder() if cls.kind == "model" else FakeProcessor()

    class AutoModel(_Auto):
        kind = "model"

    class AutoVideoProcessor(_Auto):
        kind = "processor"

    module.AutoModel = AutoModel
    module.AutoVideoProcessor = AutoVideoProcessor
    sys.modules["transformers"] = module
    return module


def _load_runner_module():
    """Import the cartridge's runner.py under a private name.

    Several cartridges ship a module called `runner`; binding this one to
    `vjepa2_runner` keeps it out of the way of whatever another test file left
    in sys.modules.
    """
    _stub_torch()
    spec = importlib.util.spec_from_file_location(
        "vjepa2_runner", CARTRIDGE_DIR / "runner.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["vjepa2_runner"] = module
    spec.loader.exec_module(module)
    return module


runner_mod = _load_runner_module()


# --------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------


def make_runner(encoder=None):
    """A VJepa2Runner wired to a fake encoder, bypassing setup().

    setup() only downloads and moves weights; every field it sets is set here,
    so reset() and stream() are the shipping implementations.
    """
    runner = runner_mod.VJepa2Runner()
    runner.device = "cpu"
    runner.model = encoder or FakeEncoder()
    runner.processor = FakeProcessor()
    runner.reset_state = None
    import collections

    runner.window = collections.deque(maxlen=runner_mod.WINDOW_SIZE)
    runner.reference = None
    runner.frame_count = 0
    return runner


def build_app(monkeypatch, runner, *, flow_window=None):
    import cognito_testing

    for key in ("WORLD_MODEL_ALLOWED_ORIGINS",
                "WORLD_MODEL_RATE_LIMIT", "WORLD_MODEL_FLOW_WINDOW"):
        monkeypatch.delenv(key, raising=False)
    # These tests exercise session behaviour, not auth, so run the app open —
    # the equivalent of the old no-token default now the server fails closed.
    cognito_testing.disable_auth(monkeypatch)
    if flow_window is not None:
        monkeypatch.setenv("WORLD_MODEL_FLOW_WINDOW", str(flow_window))

    import lib.app as app_mod
    importlib.reload(app_mod)
    from lib.jobs import JobStore

    return app_mod.create_app(runner, JobStore())


def client(app):
    from starlette.testclient import TestClient
    return TestClient(app)


def jpeg(grey: int) -> bytes:
    """A flat 64x64 JPEG at the given grey level.

    Flat frames survive JPEG quantisation intact, so the brightness the
    FakeEncoder reads back is the brightness the test asked for.
    """
    buf = io.BytesIO()
    PIL_Image.new("RGB", (FRAME_W, FRAME_H), (grey, grey, grey)).save(buf, "JPEG")
    return buf.getvalue()


def parse(payload: bytes) -> tuple:
    """Unpack one server message: (frame_num, meta, embedding)."""
    frame_num, meta_len = struct.unpack("<II", payload[:8])
    meta = json.loads(payload[8:8 + meta_len])
    embedding = np.frombuffer(payload[8 + meta_len:], dtype=np.float16)
    return frame_num, meta, embedding


def next_message(ws) -> bytes:
    """Next binary message, skipping the route's own JSON chatter."""
    while True:
        msg = ws.receive()
        if msg.get("bytes") is not None:
            return msg["bytes"]
        if msg["type"] == "websocket.close":
            raise AssertionError(f"socket closed early: {msg}")


def send_frames(ws, greys) -> list:
    """Stream frames in lockstep, one reply per frame.

    Lockstep because ActionBuffer is a latest-wins slot rather than a queue: a
    client that fires frames faster than the encoder consumes them has its
    frames overwritten, which is correct for live video but would make an
    assertion on the reply count flaky. One-in-one-out is also what a real
    client does under the flow window.
    """
    out = []
    for i, grey in enumerate(greys):
        ws.send_bytes(jpeg(grey))
        out.append(parse(next_message(ws)))
        ws.send_json({"type": "ack", "n": i + 1})
    return out


WINDOW = runner_mod.WINDOW_SIZE


# --------------------------------------------------------------------------
# Handshake and warm-up
# --------------------------------------------------------------------------


def test_handshake_announces_the_flow_window(monkeypatch):
    with client(build_app(monkeypatch, make_runner())).websocket_connect("/ws") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "connected"
        assert hello["flow_window"] == 16


def test_no_embedding_until_the_window_is_full(monkeypatch):
    # The encoder wants 16 frames. Until then the server has to say so rather
    # than go quiet, or a client cannot tell warm-up from a hung endpoint.
    encoder = FakeEncoder()
    app = build_app(monkeypatch, make_runner(encoder), flow_window=0)
    with client(app).websocket_connect("/ws") as ws:
        ws.receive_json()
        replies = send_frames(ws, [40] * (WINDOW - 1))

    assert encoder.calls == 0, "encoder ran before the window was full"
    for i, (frame_num, meta, embedding) in enumerate(replies, start=1):
        assert frame_num == i
        assert meta == {"frame": i, "warming_up": True, "frames_buffered": i}
        assert embedding.size == 0


def test_the_sixteenth_frame_produces_an_embedding(monkeypatch):
    encoder = FakeEncoder()
    app = build_app(monkeypatch, make_runner(encoder), flow_window=0)
    with client(app).websocket_connect("/ws") as ws:
        ws.receive_json()
        replies = send_frames(ws, [40] * WINDOW)

    frame_num, meta, embedding = replies[-1]
    assert encoder.calls == 1
    assert encoder.windows == [(WINDOW, FRAME_H, FRAME_W, 3)], "wrong window shape"
    assert frame_num == WINDOW
    assert meta["frame"] == WINDOW
    assert meta["dtype"] == "float16"
    assert meta["embed_dim"] == EMBED_DIM
    assert embedding.size == EMBED_DIM
    # The first full window is the baseline, so it is by definition 0 away from
    # itself. An alert here would mean every session opens by crying wolf.
    assert meta["distance"] == 0.0
    assert meta["alert"] is False


def test_the_frame_counter_matches_the_frames_the_client_sent(monkeypatch):
    # The counter is how a client correlates an alert with a moment in its own
    # video. If it drifts, the alert points at the wrong frame.
    app = build_app(monkeypatch, make_runner(), flow_window=0)
    with client(app).websocket_connect("/ws") as ws:
        ws.receive_json()
        replies = send_frames(ws, [40] * (WINDOW + 4))

    assert [r[0] for r in replies] == list(range(1, WINDOW + 5))


# --------------------------------------------------------------------------
# Alerting
# --------------------------------------------------------------------------


def test_a_scene_change_raises_an_alert(monkeypatch):
    # The point of the cartridge: after a baseline is established, a window
    # that no longer resembles it crosses ALERT_THRESHOLD.
    app = build_app(monkeypatch, make_runner(), flow_window=0)
    with client(app).websocket_connect("/ws") as ws:
        ws.receive_json()
        baseline = send_frames(ws, [0] * WINDOW)
        changed = send_frames(ws, [255] * WINDOW)

    assert baseline[-1][1]["alert"] is False
    assert changed[-1][1]["alert"] is True
    assert changed[-1][1]["distance"] > runner_mod.ALERT_THRESHOLD


def test_a_similar_scene_does_not_raise_an_alert(monkeypatch):
    # The other half of a threshold: it has to stay down for ordinary drift, or
    # the alert carries no information.
    app = build_app(monkeypatch, make_runner(), flow_window=0)
    with client(app).websocket_connect("/ws") as ws:
        ws.receive_json()
        send_frames(ws, [40] * WINDOW)
        drifted = send_frames(ws, [55] * WINDOW)

    assert drifted[-1][1]["alert"] is False
    assert 0.0 < drifted[-1][1]["distance"] < runner_mod.ALERT_THRESHOLD


def test_the_window_slides_so_one_odd_frame_is_not_an_alert(monkeypatch):
    # A deque(maxlen=16) is the whole reason this is a video model rather than
    # a frame classifier: the decision is made over a window, so a single
    # aberrant frame is diluted by the fifteen around it. Were the window
    # replaced rather than slid, this frame alone would score 2.0 and alert.
    app = build_app(monkeypatch, make_runner(), flow_window=0)
    with client(app).websocket_connect("/ws") as ws:
        ws.receive_json()
        send_frames(ws, [0] * WINDOW)
        spike = send_frames(ws, [255])

    assert spike[-1][1]["alert"] is False


def test_the_baseline_survives_a_scene_change(monkeypatch):
    # The reference is set once, on the first full window. If a later window
    # overwrote it, the detector would silently accept any new scene as normal
    # and never alert again.
    runner = make_runner()
    app = build_app(monkeypatch, runner, flow_window=0)
    with client(app).websocket_connect("/ws") as ws:
        ws.receive_json()
        send_frames(ws, [0] * WINDOW)
        reference = runner.reference.array.copy()
        send_frames(ws, [255] * WINDOW)
        assert np.allclose(runner.reference.array, reference)
        # Still alerting on the changed scene, i.e. the baseline is the old one.
        back = send_frames(ws, [255])
    assert back[-1][1]["alert"] is True


# --------------------------------------------------------------------------
# Session lifecycle
# --------------------------------------------------------------------------


def test_each_session_starts_from_a_clean_window(monkeypatch):
    # reset() runs per session. Without it the next client inherits the last
    # one's baseline and frame numbers, and alerts against a scene it never saw.
    runner = make_runner()
    connect = client(build_app(monkeypatch, runner, flow_window=0)).websocket_connect
    with connect("/ws") as ws:
        ws.receive_json()
        send_frames(ws, [0] * WINDOW)
    with connect("/ws") as ws:
        ws.receive_json()
        replies = send_frames(ws, [255] * WINDOW)

    assert replies[0][0] == 1, "frame counter carried over from the last session"
    assert replies[-1][1]["alert"] is False, "second session inherited a baseline"


def test_a_second_client_preempts_the_first(monkeypatch):
    # max_concurrent: 1 in endpoint.yaml, because the runner holds one window
    # and one baseline. Sessions are serialised (last player wins): the second
    # connection ends the first session and gets a fresh window rather than
    # interleaving two cameras into one.
    c = client(build_app(monkeypatch, make_runner(), flow_window=0))
    with c.websocket_connect("/ws") as first:
        first.receive_json()
        send_frames(first, [40])
        with c.websocket_connect("/ws") as second:
            assert second.receive_json()["type"] == "connected"


def test_an_undecodable_frame_is_skipped_not_fatal(monkeypatch):
    # A truncated frame off a live camera must not end the session or advance
    # the counter, or one bad packet costs the client its baseline.
    app = build_app(monkeypatch, make_runner(), flow_window=0)
    with client(app).websocket_connect("/ws") as ws:
        ws.receive_json()
        first = send_frames(ws, [40])
        ws.send_bytes(b"\xff\xd8not-a-jpeg")
        second = send_frames(ws, [45])

    assert first[0][0] == 1
    assert second[0][0] == 2, "a rejected frame should not consume a frame number"


def test_a_quiet_client_does_not_advance_the_window(monkeypatch):
    # Regression test. ActionBuffer.get() re-returns the latest payload every
    # <=0.1s whether or not the client sent anything, so without a guard a
    # camera that pauses has its last frame appended ~10x a second: the window
    # fills with copies of one frame, frame_count races ahead of the frames
    # actually sent, and — if the pause happens during warm-up — that duplicate
    # becomes the session's baseline. Sending one frame and then waiting is the
    # cheapest way to catch it.
    import time

    runner = make_runner()
    app = build_app(monkeypatch, runner, flow_window=0)
    with client(app).websocket_connect("/ws") as ws:
        ws.receive_json()
        send_frames(ws, [40])
        time.sleep(0.6)  # ~6 re-deliveries at the 0.1s ActionBuffer timeout
        assert runner.frame_count == 1, (
            f"frame_count reached {runner.frame_count} while the client was idle"
        )
        assert len(runner.window) == 1
        resumed = send_frames(ws, [45])

    assert resumed[0][0] == 2


# --------------------------------------------------------------------------
# setup()
# --------------------------------------------------------------------------


def test_setup_pins_the_hub_revision_when_weights_are_not_staged(tmp_path):
    # The Hub fallback is a network fetch at container boot. Unpinned it would
    # take whatever HEAD is that day; the pin is the supply-chain control.
    transformers = _stub_transformers()
    runner_mod.VJepa2Runner().setup(str(tmp_path), device="cpu", rank=0, world_size=1)

    assert [c["path"] for c in transformers.calls] == [
        "facebook/vjepa2-vitg-fpc64-384"
    ] * 2
    assert {c["revision"] for c in transformers.calls} == {
        "12ca91694b230e0d4b5b0078af6f4ae1d51e933d"
    }


def test_setup_loads_staged_weights_from_disk(tmp_path):
    # The deployed path: stage-weights.py has put the repo in the model dir, so
    # nothing should reach for the network — a private-subnet instance cannot.
    (tmp_path / "config.json").write_text("{}")
    transformers = _stub_transformers()
    runner = runner_mod.VJepa2Runner()
    runner.setup(str(tmp_path), device="cpu", rank=0, world_size=1)

    assert [c["path"] for c in transformers.calls] == [str(tmp_path)] * 2
    assert {c["revision"] for c in transformers.calls} == {None}
    assert runner.frame_count == 0 and runner.reference is None
    assert runner.window.maxlen == WINDOW


def test_batch_generation_is_refused(tmp_path):
    # The async routes exist for every cartridge, so the failure has to be an
    # explicit one rather than a partial result.
    with pytest.raises(NotImplementedError, match="realtime-only"):
        make_runner().generate(prompt="x")
