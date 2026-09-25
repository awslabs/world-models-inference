"""Measure the input-sampling period: how long a keypress can wait before the
model even looks at it. That period is the dominant term in perceived lag.

The model reads one action per chunk and emits 40 new frames from it, so frames
arrive in bursts separated by a generation pause. The interval between bursts IS
the chunk period.
"""
import os, struct, time
from websockets.sync.client import connect

ALB = os.environ.get("MG3_ALB", "")  # ALB DNS name, no scheme
TOKEN = os.environ.get("WORLD_MODEL_API_TOKEN", "")
FWD = struct.pack("<Iff", 0b0001, 0.0, 0.0)
N = 220

times = []
t0 = time.time()
with connect(f"ws://{ALB}/ws?token={TOKEN}", open_timeout=40, max_size=None) as ws:
    for _ in range(4):
        ws.send(FWD)
    while len(times) < N:
        try:
            msg = ws.recv(timeout=120)
        except TimeoutError:
            break
        times.append(time.time())
        if len(times) == 1:
            print(f"cold first frame: {times[0]-t0:.1f}s", flush=True)
        ws.send(FWD)

print(f"frames={len(times)}")
# Chunk boundaries: the generation pause between bursts.
gaps = [(times[i + 1] - times[i], i) for i in range(len(times) - 1)]
bounds = [(g, i) for g, i in gaps if g > 0.25]
print(f"boundaries detected: {len(bounds)}")
print("pause at each boundary: " + ", ".join(f"{g:.2f}s" for g, _ in bounds[:12]))

# Period between consecutive boundaries = one chunk of input sampling.
bi = [times[i] for _, i in bounds]
periods = [bi[i + 1] - bi[i] for i in range(len(bi) - 1)]
if periods:
    periods_sorted = sorted(periods)
    med = periods_sorted[len(periods_sorted) // 2]
    print(f"chunk period: median {med:.2f}s  min {periods_sorted[0]:.2f}s  max {periods_sorted[-1]:.2f}s")
    print(f"=> input sampled every ~{med:.1f}s")
    print(f"=> typical lag ~{med:.1f}s, worst case ~{2*med:.1f}s")

warm = times[80:]
if len(warm) > 2:
    span = warm[-1] - warm[0]
    print(f"warm fps: {(len(warm)-1)/span:.1f}")
