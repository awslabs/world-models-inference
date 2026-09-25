#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark a real-time world-model endpoint and write a performance report.

Drives a live session the way a player does -- seed image, then a scripted
input pattern -- and records the arrival time and size of every frame. From
that it reports three things, kept separate because they have three different
causes and three different fixes:

  Throughput & latency  What the player feels. Delivered fps, frame-interval
                        percentiles, jitter, stalls, and in-session lag. Lag is
                        probed with a ping on the same socket as the frames, so
                        the pong queues behind them: the round-trip is the
                        standing queue plus the network, which is exactly the
                        delay between pressing a key and seeing it.

  Bandwidth & encoding  Why the fps is what it is. A 720p JPEG is ~85 KiB, so
                        delivered fps is usually pinned by the link, not the
                        GPU: this section reports the measured bitrate and the
                        fps ceiling it implies, so the two can't be confused.

  Image quality         How good the frames actually are, and for how long.
                        Sharpness (variance of the Laplacian) and inter-frame
                        motion are reported per time window, because this model
                        family degrades as the context window fills -- the
                        world visibly softens and then dissolves. A single
                        average would hide the one thing worth knowing.

Run it from anywhere with network reach to the endpoint. Running it ON the
instance over loopback removes the WAN and measures generation rate instead of
delivery rate; the report labels which one it measured.

Usage:
  benchmark.py --url ws://<host>/ws [--token T] [--seed img.png]
               [--seconds 60] [--out report.md] [--json report.json]
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

try:
    import numpy as np
    from PIL import Image
except ImportError:  # pragma: no cover - dependency guidance
    sys.exit("benchmark.py needs numpy and pillow: pip install numpy pillow")

try:
    import websockets
except ImportError:  # pragma: no cover - dependency guidance
    sys.exit("benchmark.py needs websockets: pip install websockets")


# Frames are sampled for image analysis rather than analysed wholesale: decoding
# and running a Laplacian over every frame of a 60 s 60 fps run costs more time
# than the run itself, and would throttle the very socket we are measuring.
ANALYSE_EVERY = 15

# Width of the reporting windows for the quality trend, in seconds.
QUALITY_WINDOW_S = 10.0

# A gap longer than this is a stall the player would notice as a freeze, not
# jitter. 100 ms is ~6 frames at 60 fps.
STALL_MS = 100.0


def pct(values: list[float], p: float) -> float:
    """Percentile by nearest rank. Avoids a scipy/numpy dependency here so the
    latency stats are computed identically whether or not the image analysis
    ran."""
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(math.ceil(p / 100.0 * len(ordered))) - 1))
    return ordered[k]


def sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian — the standard no-reference focus measure.

    High on crisp edges, low on blur. There is no ground-truth frame to compare
    a generated world against, so an absolute value means little; the *trend*
    across a session is the signal, and it tracks the model losing coherence as
    its context window fills.
    """
    lap = (
        -4.0 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1] + gray[2:, 1:-1]
        + gray[1:-1, :-2] + gray[1:-1, 2:]
    )
    return float(lap.var())


def colourfulness(rgb: np.ndarray) -> float:
    """Hasler-Süsstrunk colourfulness. Falls towards zero as the world dissolves
    into the flat sand/fog this model family ends in, so it catches a failure
    that sharpness alone can miss."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    return float(
        math.sqrt(rg.std() ** 2 + yb.std() ** 2)
        + 0.3 * math.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    )


@dataclass
class Frame:
    t: float           # arrival time, seconds since first frame
    nbytes: int


@dataclass
class Sample:
    """Image analysis of one sampled frame."""
    t: float
    sharp: float
    colour: float
    motion: float      # mean abs diff vs the previous sampled frame, 0-255
    width: int
    height: int


@dataclass
class Run:
    frames: list[Frame] = field(default_factory=list)
    samples: list[Sample] = field(default_factory=list)
    pings: list[float] = field(default_factory=list)
    ttff: float = 0.0          # time to first frame, seconds
    flow_window: int | None = None
    elapsed: float = 0.0
    error: str | None = None


async def run_session(url: str, token: str, seed: bytes | None,
                      seconds: float, analyse: bool) -> Run:
    run = Run()
    qs = f"?token={token}" if token else ""

    # A benchmark that only sends neutral input measures the model doing almost
    # nothing. Hold W and sweep the camera so the engine denoises real motion,
    # which is the load a player actually generates.
    async def drive(ws, stop_at):
        phase = 0
        while time.monotonic() < stop_at:
            await asyncio.sleep(0.05)
            phase += 1
            look = 0
            if phase % 60 < 20:
                look = 12
            elif phase % 60 < 40:
                look = -12
            try:
                await ws.send(json.dumps({
                    "type": "control", "buttons": ["W"],
                    "mouse_dx": look, "mouse_dy": 0,
                }))
            except Exception:
                return

    async def pinger(ws, stop_at):
        while time.monotonic() < stop_at:
            await asyncio.sleep(1.0)
            try:
                await ws.send(json.dumps(
                    {"type": "ping", "timestamp": int(time.time() * 1000)}))
            except Exception:
                return

    connect_started = time.monotonic()
    async with websockets.connect(
        f"{url}{qs}", max_size=None, ping_interval=None, open_timeout=30,
    ) as ws:
        stop_at = time.monotonic() + seconds
        first = None
        prev_gray = None
        acks = 0
        tasks: list[asyncio.Task] = []

        while True:
            remaining = stop_at - time.monotonic()
            if remaining <= 0:
                break
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=max(remaining, 1))
            except (asyncio.TimeoutError, websockets.ConnectionClosed):
                break

            if isinstance(msg, bytes):
                now = time.monotonic()
                if first is None:
                    first = now
                    run.ttff = now - connect_started
                    tasks = [asyncio.create_task(drive(ws, stop_at)),
                             asyncio.create_task(pinger(ws, stop_at))]
                run.frames.append(Frame(t=now - first, nbytes=len(msg)))

                # Acknowledge every frame: the endpoint bounds how many frames
                # it sends ahead of the client's acks, so a benchmark that stays
                # silent would be throttled to the window and under-report fps.
                if run.flow_window:
                    acks += 1
                    await ws.send(json.dumps({"type": "ack", "n": acks}))

                if analyse and len(run.frames) % ANALYSE_EVERY == 0:
                    try:
                        img = Image.open(io.BytesIO(msg)).convert("RGB")
                        rgb = np.asarray(img, dtype=np.float32)
                        gray = rgb.mean(axis=2)
                        motion = (0.0 if prev_gray is None
                                  or prev_gray.shape != gray.shape
                                  else float(np.abs(gray - prev_gray).mean()))
                        prev_gray = gray
                        run.samples.append(Sample(
                            t=now - first, sharp=sharpness(gray),
                            colour=colourfulness(rgb), motion=motion,
                            width=img.width, height=img.height,
                        ))
                    except Exception:
                        pass  # a truncated frame must not end the benchmark
                continue

            obj = json.loads(msg)
            kind = obj.get("type")
            if kind == "connected":
                run.flow_window = obj.get("flow_window") or 0
                if run.flow_window:
                    acks = 0
                payload = {"type": "start"}
                if seed:
                    payload["image_data"] = base64.b64encode(seed).decode()
                await ws.send(json.dumps(payload))
            elif kind == "pong":
                run.pings.append(time.time() * 1000 - int(obj["timestamp"]))
            elif kind == "error":
                run.error = obj.get("message")
                break

        for t in tasks:
            t.cancel()
        run.elapsed = (time.monotonic() - first) if first else 0.0

    return run


def summarise(run: Run, url: str, seconds: float) -> dict:
    frames = run.frames
    n = len(frames)
    fps = n / run.elapsed if run.elapsed else 0.0

    gaps = [1000.0 * (b.t - a.t) for a, b in zip(frames, frames[1:])]
    sizes = [f.nbytes for f in frames]
    total_bytes = sum(sizes)
    mbps = total_bytes * 8 / 1e6 / run.elapsed if run.elapsed else 0.0
    mean_frame_bits = (total_bytes / n * 8) if n else 0.0

    host = urlparse(url).hostname or ""
    loopback = host in ("127.0.0.1", "localhost", "::1")

    res = ""
    if run.samples:
        res = f"{run.samples[0].width}x{run.samples[0].height}"

    # Quality trend, bucketed by time so degradation is visible rather than
    # averaged away.
    windows = []
    if run.samples:
        span = max(s.t for s in run.samples)
        nwin = max(1, int(math.ceil(span / QUALITY_WINDOW_S)))
        for w in range(nwin):
            lo, hi = w * QUALITY_WINDOW_S, (w + 1) * QUALITY_WINDOW_S
            bucket = [s for s in run.samples if lo <= s.t < hi]
            if not bucket:
                continue
            windows.append({
                "from_s": round(lo, 1),
                "to_s": round(min(hi, span), 1),
                "frames_generated_by_end": int(hi * fps),
                "sharpness": round(statistics.mean(s.sharp for s in bucket), 1),
                "colourfulness": round(statistics.mean(s.colour for s in bucket), 2),
                "motion": round(statistics.mean(s.motion for s in bucket), 2),
                "n": len(bucket),
            })

    return {
        "endpoint": url,
        "path": "loopback (no WAN — measures generation rate)" if loopback
                else "network (measures delivered rate)",
        "requested_seconds": seconds,
        "elapsed_s": round(run.elapsed, 1),
        "error": run.error,
        "flow_window": run.flow_window,
        "resolution": res,
        "throughput": {
            "frames": n,
            "fps": round(fps, 1),
            "time_to_first_frame_s": round(run.ttff, 1),
            "frame_interval_ms": {
                "mean": round(statistics.mean(gaps), 1) if gaps else 0.0,
                "p50": round(pct(gaps, 50), 1),
                "p95": round(pct(gaps, 95), 1),
                "p99": round(pct(gaps, 99), 1),
                "max": round(max(gaps), 1) if gaps else 0.0,
                "jitter_stdev": round(statistics.pstdev(gaps), 1) if len(gaps) > 1 else 0.0,
            },
            "stalls_over_100ms": sum(1 for g in gaps if g > STALL_MS),
            "stall_rate_pct": round(100.0 * sum(1 for g in gaps if g > STALL_MS) / len(gaps), 2) if gaps else 0.0,
        },
        "lag_ms": {
            "n": len(run.pings),
            "min": round(min(run.pings), 0) if run.pings else 0,
            "p50": round(pct(run.pings, 50), 0),
            "p95": round(pct(run.pings, 95), 0),
            "max": round(max(run.pings), 0) if run.pings else 0,
        },
        "bandwidth": {
            "bytes_per_frame_mean": int(total_bytes / n) if n else 0,
            "bytes_per_frame_p95": int(pct([float(s) for s in sizes], 95)) if sizes else 0,
            "bitrate_mbit_s": round(mbps, 1),
            "total_mib": round(total_bytes / 1048576, 1),
            # The headline number: at this frame size, this is the link speed a
            # player needs for 60 fps. Compare with bitrate_mbit_s above.
            "mbit_s_needed_for_60fps": round(mean_frame_bits * 60 / 1e6, 1),
            "fps_ceiling_at_this_bitrate": round(mbps * 1e6 / mean_frame_bits, 1) if mean_frame_bits else 0.0,
        },
        "quality_windows": windows,
    }


def to_markdown(s: dict) -> str:
    t, b, lag = s["throughput"], s["bandwidth"], s["lag_ms"]
    fi = t["frame_interval_ms"]
    out = [
        "# Performance report",
        "",
        f"- Endpoint: `{s['endpoint']}`",
        f"- Path: {s['path']}",
        f"- Duration: {s['elapsed_s']} s, resolution {s['resolution'] or 'n/a'}"
        + (f", flow window {s['flow_window']}" if s["flow_window"] is not None else ""),
    ]
    if s["error"]:
        out += ["", f"**Endpoint returned an error:** {s['error']}"]

    out += [
        "",
        "## Throughput & latency",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Frames delivered | {t['frames']} |",
        f"| **Frame rate** | **{t['fps']} fps** |",
        f"| Time to first frame | {t['time_to_first_frame_s']} s |",
        f"| Frame interval mean | {fi['mean']} ms |",
        f"| Frame interval p50 / p95 / p99 | {fi['p50']} / {fi['p95']} / {fi['p99']} ms |",
        f"| Worst frame interval | {fi['max']} ms |",
        f"| Jitter (stdev) | {fi['jitter_stdev']} ms |",
        f"| Stalls > {int(STALL_MS)} ms | {t['stalls_over_100ms']} ({t['stall_rate_pct']}% of frames) |",
        f"| **Input lag p50** | **{lag['p50']:.0f} ms** (min {lag['min']:.0f}, p95 {lag['p95']:.0f}, max {lag['max']:.0f}) |",
        "",
        "Input lag is measured with a ping on the frame socket, so it includes any",
        "queue of undelivered frames — it is the delay a player feels, not a bare",
        "network round-trip.",
        "",
        "## Bandwidth & encoding",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Bytes per frame (mean / p95) | {b['bytes_per_frame_mean'] / 1024:.1f} / {b['bytes_per_frame_p95'] / 1024:.1f} KiB |",
        f"| Measured bitrate | {b['bitrate_mbit_s']} Mbit/s |",
        f"| Transferred | {b['total_mib']} MiB |",
        f"| **Link needed for 60 fps** | **{b['mbit_s_needed_for_60fps']} Mbit/s** |",
        f"| fps ceiling at this bitrate | {b['fps_ceiling_at_this_bitrate']} fps |",
        "",
    ]

    if s["quality_windows"]:
        first = s["quality_windows"][0]
        last = s["quality_windows"][-1]
        # Signed change, so a loss of sharpness reads negative.
        drop = (100.0 * (last["sharpness"] / first["sharpness"] - 1)
                if first["sharpness"] else 0.0)
        out += [
            "## Image quality over the session",
            "",
            "Sharpness is the variance of the Laplacian (higher = crisper edges);",
            "colourfulness falls as the world dissolves into flat fog or sand; motion",
            "is the mean absolute change between sampled frames, so a value near zero",
            "means the picture froze. Absolute values are only comparable within a run —",
            "the trend is the point.",
            "",
            "| Window | Frames generated by end | Sharpness | Colourfulness | Motion |",
            "|---|---|---|---|---|",
        ]
        for w in s["quality_windows"]:
            out.append(
                f"| {w['from_s']:.0f}–{w['to_s']:.0f} s | ~{w['frames_generated_by_end']} | "
                f"{w['sharpness']} | {w['colourfulness']} | {w['motion']} |"
            )
        out += [
            "",
            f"Sharpness moved {drop:+.0f}% from the first window to the last "
            f"({first['sharpness']} → {last['sharpness']}).",
            "",
            "This model family carries a context window measured in frames generated,",
            "not seconds elapsed, so a faster link spends the coherence budget sooner.",
            "Read the decay against the frame count, not the clock.",
        ]

    return "\n".join(out) + "\n"


async def amain() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True, help="ws://host/ws or wss://host/ws")
    ap.add_argument("--token", default="", help="shared API token, if auth is enabled")
    ap.add_argument("--seed", help="seed image; omitted → the model's own prior")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--out", help="write the markdown report here (default: stdout)")
    ap.add_argument("--json", dest="json_out", help="also write the raw numbers here")
    ap.add_argument("--no-image-analysis", action="store_true",
                    help="skip decoding frames (throughput and lag only)")
    args = ap.parse_args()

    seed = Path(args.seed).read_bytes() if args.seed else None
    try:
        run = await run_session(args.url, args.token, seed, args.seconds,
                                analyse=not args.no_image_analysis)
    except (TimeoutError, asyncio.TimeoutError):
        # The endpoint serves one session at a time. A client that vanished
        # without closing cleanly (browser tab, Ctrl-C) leaves the socket open
        # through the load balancer, so the endpoint stays busy and a new
        # handshake stalls rather than being refused. Waiting out the LB idle
        # timeout clears it — say so, because the bare timeout looks like a
        # dead endpoint.
        print("Handshake timed out. The endpoint answers /health but takes one "
              "session at a time; if a previous client dropped without closing, "
              "wait ~60 s for the load balancer to drop the stale socket and "
              "retry.", file=sys.stderr)
        return 1

    if not run.frames:
        print(f"No frames received. {run.error or 'Check the URL, token and that the endpoint is up.'}",
              file=sys.stderr)
        return 1

    summary = summarise(run, args.url, args.seconds)
    report = to_markdown(summary)

    if args.out:
        Path(args.out).write_text(report)
        print(f"Wrote {args.out}")
    else:
        print(report)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2))
        print(f"Wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(amain()))
