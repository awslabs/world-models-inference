# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Praxis-1 runner — closed-loop robot-policy rollouts in MuJoCo, as an async job.

Praxis-1 (Runway) is a world action model: a video-action policy that predicts the
next chunk of robot actions together with the video of what those actions will do.
One job runs the policy in closed loop on one task of the praxis-sim suite, one
episode per seed, and returns a single MP4 of every episode back to back. Each
episode's frames show the rendered sim cameras (overview, exterior, wrist) with the
video the model predicted underneath.

Next to the returned MP4 the runner leaves a directory with the per-episode MP4s
and a ``summary.json`` (success per seed) — see README.md.

Each job runs upstream's own entry point (``scripts/run_sim.py`` for the DROID
single-arm rig, ``scripts/run_sim_yam.py`` for the bimanual YAM rig) in a child
process — the exact command the cartridge was validated with. The child owns the
GPU for the length of the job; this process never initialises CUDA. Upstream's
on-disk kernel cache (``PRAXIS_CACHE_DIR``) keeps later jobs from recompiling.

Job parameters (all optional):
  rig     "droid" (default) or "yam"
  task    a task of that rig's suite (DROID_TASKS / YAM_TASKS below)
  prompt  replaces the task's default instruction; the success check is unchanged
  seeds   list of scene seeds, one episode each, or "0,1,2"; else ``seed``; else [0]
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess  # nosec B404 - fixed argv, no shell; every user value is validated first
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from lib.runner import Runner

logger = logging.getLogger(__name__)

# Upstream checkout baked into the image (Dockerfile, PRAXIS_SHA).
PRAXIS_REPO = os.environ.get("PRAXIS_REPO", "/opt/praxis-1")

# Weight layout under the model dir — the `subdir`s of endpoint.yaml's `hf:` list.
VAE_SUBDIR, VAE_FILE = "wan2_2", "Wan2.2_VAE.pth"
TEXT_ENCODER = "qwen3_5-2b"

# The DROID single-arm suite of praxis-sim (eval_suites.py at PRAXIS_SHA): the 7
# DROID task families and the 11 harder tasks.
DROID_TASKS = (
    "lift_cube", "cube_on_plate", "block_in_bowl", "block_out_of_bowl", "move_block",
    "stack_block", "press_button",
    "cube_in_tray", "cube_on_color", "pick_color_block", "color_block_in_bowl",
    "stack_color_blocks", "peg_in_hole_6mm", "block_in_small_bowl", "cube_on_small_plate",
    "stack_small_block", "two_blocks_in_bowl", "stack_three_blocks",
)
# The bag tasks the released bimanual checkpoint solves (run_sim_yam.py's default set).
YAM_TASKS = ("dice_in_bag", "tennis_ball_in_bag", "apple_in_bag", "bell_in_bag")

MAX_SEEDS = 8
MAX_SEED = 1_000_000
MAX_PROMPT_CHARS = 500

# The child gets the whole job, but not forever: compile + CUDA-graph capture is a
# few minutes on first use, then each episode is roughly one to three minutes.
BASE_TIMEOUT_S = int(os.environ.get("PRAXIS_BASE_TIMEOUT_S", "1200"))
PER_EPISODE_TIMEOUT_S = int(os.environ.get("PRAXIS_EPISODE_TIMEOUT_S", "600"))

# torchrun's variables: the child is a plain single-GPU process, and inheriting
# these makes libraries that look for them try to join the server's process group.
_TORCHRUN_ENV = (
    "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK", "GROUP_WORLD_SIZE",
    "ROLE_RANK", "ROLE_NAME", "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
    "TORCHELASTIC_RESTART_COUNT", "TORCHELASTIC_MAX_RESTARTS", "TORCHELASTIC_RUN_ID",
    "TORCHELASTIC_USE_AGENT_STORE", "TORCHELASTIC_ERROR_FILE", "TORCH_NCCL_ASYNC_ERROR_HANDLING",
)


@dataclass(frozen=True)
class Rig:
    script: str
    config: str
    checkpoint: str     # subdir of the model dir
    tasks: tuple
    default_task: str


RIGS = {
    "droid": Rig("scripts/run_sim.py", "configs/sim_droid.yaml", "praxis-1-droid", DROID_TASKS, "lift_cube"),
    "yam": Rig("scripts/run_sim_yam.py", "configs/sim_yam.yaml", "praxis-1-yam", YAM_TASKS, "dice_in_bag"),
}


def create_runner():
    return Praxis1Runner()


def weights_dir(model_dir: Path, subdir: str) -> Path:
    """Where one `hf:` entry's files are: ``<model>/<subdir>``.

    ``scripts/stage-weights.py --subdir`` currently lands files one level deeper
    (``<model>/<subdir>/<subdir>``, see tests/test_launch_buildspecs.py), while the
    Launch Stack stages them once; accept either layout.
    """
    nested = model_dir / subdir / subdir
    return nested if nested.is_dir() else model_dir / subdir


def parse_seeds(seeds, seed=None) -> list[int]:
    """The episode seeds of a job: ``seeds`` (list or "0,1"), else ``seed``, else [0].

    ``seed`` is the generic single-seed knob the UI sends, where -1 means "random";
    a sim rollout has no random scene, so a negative seed falls back to 0.
    """
    if seeds is None or seeds == "" or seeds == []:
        if seed is None or str(seed).strip() in ("", "-1"):
            return [0]
        seeds = [seed]
    if isinstance(seeds, str):
        seeds = [s.strip() for s in seeds.split(",")]
    if not isinstance(seeds, (list, tuple)):
        raise ValueError("seeds must be a list of integers")
    out = []
    for s in seeds:
        if isinstance(s, bool) or not (str(s).strip().isascii() and str(s).strip().isdecimal()):
            raise ValueError(f"seed {s!r} is not a non-negative integer")
        n = int(s)
        if n > MAX_SEED:
            raise ValueError(f"seed {n} is larger than {MAX_SEED}")
        out.append(n)
    if not 0 < len(out) <= MAX_SEEDS:
        raise ValueError(f"a job runs 1 to {MAX_SEEDS} seeds, got {len(out)}")
    if len(set(out)) != len(out):
        raise ValueError("seeds must not repeat")
    return out


class Praxis1Runner(Runner):
    """Praxis-1 sim rollouts wrapped as a generic async Runner."""

    def setup(self, ckpt_dir: str, device, rank: int, world_size: int) -> None:
        if world_size != 1:
            # One child process owns one GPU; there is nothing for other ranks to do.
            raise RuntimeError("praxis-1 runs on a single GPU; set NPROC_PER_NODE=1")
        self.rank = rank
        self.ckpt_dir = Path(ckpt_dir)
        self.repo = Path(PRAXIS_REPO)
        self.vae = weights_dir(self.ckpt_dir, VAE_SUBDIR) / VAE_FILE
        self.text_encoder = weights_dir(self.ckpt_dir, TEXT_ENCODER)
        self.checkpoints = {name: weights_dir(self.ckpt_dir, rig.checkpoint) for name, rig in RIGS.items()}
        missing = [
            str(p) for p in (
                self.repo / RIGS["droid"].script, self.repo / RIGS["droid"].config,
                self.vae, self.text_encoder / "config.json",
                self.checkpoints["droid"] / "config.json",
            ) if not p.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "praxis-1 is missing files (stage every `hf:` entry of endpoint.yaml): "
                + ", ".join(missing)
            )
        self.available_rigs = {
            name for name, path in self.checkpoints.items() if (path / "config.json").is_file()
        }
        self.cache_dir = Path(os.environ.get("PRAXIS_CACHE_DIR")
                              or Path(tempfile.gettempdir()) / "praxis-cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("praxis-1 ready: rigs %s, upstream %s", sorted(self.available_rigs), self.repo)

    # ------------------------------------------------------------------ job

    def resolve(self, params: dict) -> tuple[str, str, str | None, list[int]]:
        """Validate a job's parameters: (rig, task, prompt or None, seeds)."""
        rig = params.get("rig") or "droid"
        if rig not in RIGS:
            raise ValueError(f"rig must be one of {', '.join(sorted(RIGS))}")
        if rig not in self.available_rigs:
            raise ValueError(f"rig {rig!r} is not staged: no {RIGS[rig].checkpoint}/ in the model dir")
        task = params.get("task") or RIGS[rig].default_task
        if task not in RIGS[rig].tasks:
            raise ValueError(f"task {task!r} is not a {rig} task: {', '.join(RIGS[rig].tasks)}")
        prompt = (params.get("prompt") or "").strip() or None
        if prompt is not None and (len(prompt) > MAX_PROMPT_CHARS or not prompt.isprintable()):
            raise ValueError(f"prompt must be printable text of at most {MAX_PROMPT_CHARS} characters")
        return rig, task, prompt, parse_seeds(params.get("seeds"), params.get("seed"))

    def command(self, rig: str, task: str, prompt: str | None, seeds: list[int], out_dir: Path) -> list[str]:
        """The upstream command line of one job. A list (no shell), one token per value."""
        spec = RIGS[rig]
        ckpt = str(self.checkpoints[rig])
        # Values after `key=` are parsed as YAML; a JSON string is a YAML string, so
        # a path or a prompt can never become a list, a mapping or another key.
        common = [
            f"--set=model.vae_weights={json.dumps(str(self.vae))}",
            f"--set=model.text_encoder={json.dumps(str(self.text_encoder))}",
            # Robot models are baked into the image (PRAXIS_SIM_CACHE); never fetch at run time.
            "--set=scene.download=false",
        ]
        if rig == "droid":
            return [
                sys.executable, "-u", spec.script, f"--config={spec.config}",
                f"--set=model.checkpoint={json.dumps(ckpt)}",
                *common,
                f"--set=task.name={task}",
                f"--set=task.prompt={json.dumps(prompt) if prompt else 'null'}",
                # The task's own time limit, as upstream's suite runner uses.
                "--set=task.time_limit_s=null",
                "--set=viewer.enabled=false",
                f"--set=episodes.seeds={json.dumps(seeds)}",
                f"--set=output.dir={json.dumps(str(out_dir))}",
            ]
        cmd = [
            sys.executable, "-u", spec.script, f"--config={spec.config}",
            f"--checkpoint={ckpt}", f"--out={out_dir}",
            "--units=" + ",".join(f"{task}:{s}" for s in seeds),
            *common,
        ]
        if prompt:
            cmd.append(f"--prompt={prompt}")
        return cmd

    def child_env(self) -> dict:
        env = {k: v for k, v in os.environ.items() if k not in _TORCHRUN_ENV}
        env.update({
            "MUJOCO_GL": "egl",
            "PYTHONUNBUFFERED": "1",
            "PRAXIS_CACHE_DIR": str(self.cache_dir),
            # Every weight is local; a network lookup would only add latency or fail.
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        })
        return env

    def generate(self, **params) -> str:
        rig, task, prompt, seeds = self.resolve(params)
        output_path = Path(params.get("output_path") or Path(tempfile.gettempdir()) / "praxis-1" / "output.mp4")
        # Sidecar directory: <job>/output.mp4 -> <job>/output/{episode_*.mp4, summary.json}
        out_dir = output_path.with_suffix("")
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = self.command(rig, task, prompt, seeds, out_dir)
        log_path = out_dir / "rollout.log"
        timeout = BASE_TIMEOUT_S + PER_EPISODE_TIMEOUT_S * len(seeds)
        logger.info("praxis-1 %s/%s seeds %s -> %s", rig, task, seeds, out_dir)
        with log_path.open("w") as log:
            try:
                proc = subprocess.run(  # nosec B603 - argv list, no shell
                    cmd, cwd=self.repo, env=self.child_env(), stdout=log,
                    stderr=subprocess.STDOUT, timeout=timeout, check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"praxis-1 rollout exceeded {timeout} s; see {log_path}") from exc
        if proc.returncode != 0:
            raise RuntimeError(
                f"praxis-1 rollout failed (exit {proc.returncode}): {_tail(log_path)}"
            )
        episodes = self.collect(rig, task, seeds, out_dir)
        # Keep upstream's own summary (call times, reach/lift metrics) beside ours.
        upstream_summary = out_dir / "summary.json"
        if upstream_summary.exists():
            upstream_summary.replace(out_dir / "upstream_summary.json")
        summary = {
            "model": "praxis-1",
            "rig": rig,
            "task": task,
            "prompt": prompt,
            "seeds": seeds,
            "episodes": len(episodes),
            "successes": sum(bool(e["success"]) for e in episodes),
            "per_episode": episodes,
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
        concat_videos([out_dir / e["video"] for e in episodes], output_path)
        logger.info("praxis-1 %s/%s: %d/%d successes -> %s",
                    rig, task, summary["successes"], len(episodes), output_path)
        return str(output_path)

    @staticmethod
    def collect(rig: str, task: str, seeds: list[int], out_dir: Path) -> list[dict]:
        """Per-seed success and video, from upstream's own outputs, in seed order."""
        if rig == "droid":
            # run_sim.py: <out>/episode_<seed>.mp4 and summary.json["per_episode"].
            upstream = json.loads((out_dir / "summary.json").read_text())
            success = {int(e["seed"]): bool(e["success"]) for e in upstream.get("per_episode", [])}
            video_dir = ""
        else:
            # run_sim_yam.py: <out>/<task>/episode_<seed>.mp4 and one episodes.jsonl line each.
            lines = (out_dir / "episodes.jsonl").read_text().splitlines()
            records = [json.loads(x) for x in lines if x.strip()]
            success = {int(r["seed"]): bool(r["success"]) for r in records if r.get("task") == task}
            video_dir = f"{task}/"
        episodes = []
        for s in seeds:
            video = f"{video_dir}episode_{s:03d}.mp4"
            if s not in success or not (out_dir / video).is_file():
                raise RuntimeError(f"praxis-1 wrote no result for seed {s} in {out_dir}")
            episodes.append({"seed": s, "success": success[s], "video": video})
        return episodes


def concat_videos(videos: list[Path], output: Path) -> None:
    """Join same-size H.264 MP4s into ``output`` without re-encoding."""
    if len(videos) == 1:
        shutil.copyfile(videos[0], output)
        return
    listing = output.with_suffix(".concat.txt")
    # The concat demuxer's own quoting: one 'file' line per input, ' escaped.
    listing.write_text("".join(
        "file '{}'\n".format(str(v.resolve()).replace("'", "'\\''")) for v in videos
    ))
    try:
        subprocess.run(  # nosec B603 - argv list, server-chosen paths
            [_ffmpeg(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
             "-i", str(listing), "-c", "copy", "-movflags", "+faststart", str(output)],
            check=True, capture_output=True, timeout=300,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"joining the episode videos failed: {exc.stderr.decode()[-500:]}") from exc
    finally:
        listing.unlink(missing_ok=True)


def _ffmpeg() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return shutil.which("ffmpeg") or "ffmpeg"


def _tail(path: Path, lines: int = 15) -> str:
    try:
        return " | ".join(path.read_text(errors="replace").strip().splitlines()[-lines:])
    except OSError:
        return f"(no log at {path})"
