# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Praxis-1 cartridge tests — structure, the build-time kernel patch, and the runner.

The runner's GPU work happens in a child process (upstream's sim script), so these
tests replace ``subprocess.run`` with a fake that writes what upstream writes —
episode MP4s, ``summary.json`` / ``episodes.jsonl`` — and check everything the
runner itself does: parameter validation, the exact argv, the child environment,
result collection, the summary, and the joined MP4. No GPU, torch or praxis needed.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CARTRIDGE_DIR = REPO_ROOT / "inference" / "models" / "praxis-1"
INFERENCE_DIR = REPO_ROOT / "inference"
if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))

PRAXIS_SHA = "2042edf6f5512294f9242636ee1ce235153b96f0"


def _stub_torch():
    """Stub torch so lib.runner imports without a GPU (shares the `_wm_stub` marker)."""
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


def _load(name: str, filename: str):
    _stub_torch()
    spec = importlib.util.spec_from_file_location(name, CARTRIDGE_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


runner_mod = _load("praxis1_runner", "runner.py")
patch_mod = _load("praxis1_patch_moe_smem", "patch_moe_smem.py")


# --------------------------------------------------------------------------
# Structure
# --------------------------------------------------------------------------


class TestCartridgeStructure:
    def test_required_files(self):
        for name in ("endpoint.yaml", "runner.py", "requirements.txt", "Dockerfile", "README.md"):
            assert (CARTRIDGE_DIR / name).is_file(), name

    def test_served_by_the_shared_entrypoint(self):
        assert not (CARTRIDGE_DIR / "server.py").exists()
        assert 'ENTRYPOINT ["./lib/serve"]' in (CARTRIDGE_DIR / "Dockerfile").read_text()

    def test_manifest_is_async_single_gpu_80gb(self):
        text = (CARTRIDGE_DIR / "endpoint.yaml").read_text()
        assert re.search(r"^mode:\s*async", text, re.M)
        assert re.search(r"^gpu_count:\s*1\b", text, re.M)
        assert int(re.search(r"^gpu_memory_gb:\s*(\d+)", text, re.M).group(1)) >= 80
        assert int(re.search(r"volume_size:\s*(\d+)", text).group(1)) >= 300

    def test_every_weight_repo_is_pinned_to_a_commit(self):
        text = (CARTRIDGE_DIR / "endpoint.yaml").read_text()
        entries = re.findall(r"- \{ repo: (\S+), subdir: \"([^\"]*)\", revision: ([0-9a-f]{40}) \}", text)
        assert {repo for repo, _, _ in entries} == {
            "runwayml/praxis-1-droid", "runwayml/praxis-1-yam",
            "Wan-AI/Wan2.2-TI2V-5B", "Qwen/Qwen3.5-2B",
        }
        # The runner reads exactly these subdirs.
        subdirs = {subdir for _, subdir, _ in entries}
        assert subdirs == {runner_mod.RIGS["droid"].checkpoint, runner_mod.RIGS["yam"].checkpoint,
                           runner_mod.VAE_SUBDIR, runner_mod.TEXT_ENCODER}

    def test_licence_is_apache_and_unrestricted(self):
        text = (CARTRIDGE_DIR / "endpoint.yaml").read_text()
        assert re.search(r"^license:\s*Apache-2\.0\s*$", text, re.M)
        assert re.search(r"^license_restricted:\s*false", text, re.M)

    def test_no_token_in_the_cartridge(self):
        for path in (p for p in CARTRIDGE_DIR.iterdir() if p.is_file()):
            assert not re.search(r"\bhf_[A-Za-z0-9]{20,}", path.read_text()), path.name

    def test_dockerfile_pins_upstream_and_applies_the_patch(self):
        text = (CARTRIDGE_DIR / "Dockerfile").read_text()
        assert f"ARG PRAXIS_SHA={PRAXIS_SHA}" in text
        assert "checkout --quiet ${PRAXIS_SHA}" in text
        assert "patch_moe_smem.py /opt/praxis-1/praxis/kernels/moe_gemm_triton.py" in text
        assert "ENV NPROC_PER_NODE=1" in text
        # EGL rendering of the sim cameras inside the container.
        assert "NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics" in text
        # Robot models baked in, never fetched at run time.
        assert "ensure_assets" in text

    def test_non_root_user_owns_the_sim_asset_cache(self):
        # Found on the first deploy: ensure_assets() re-hashes the cache at run time
        # as appuser, and some fetched files are 0600 root, so every job failed with
        # PermissionError until the cache was chowned.
        text = (CARTRIDGE_DIR / "Dockerfile").read_text()
        chown = next(line for line in text.splitlines() if "useradd" in line and "chown" in line)
        assert "/opt/praxis-sim-cache" in chown

    def test_image_has_the_two_arm_sim_dependencies(self):
        # Found on the first deploy: rig=yam failed with ModuleNotFoundError: scipy.
        text = (CARTRIDGE_DIR / "Dockerfile").read_text()
        assert '"scipy>=1.10"' in text
        assert "praxis.sim.yam_player" in text  # build-time import check of the yam rig

    def test_endpoint_pins_a_cuda_13_capable_ami(self):
        # The default DLAMI's driver predates CUDA 13 (torch cu130).
        text = (CARTRIDGE_DIR / "endpoint.yaml").read_text()
        assert "ami_ssm_parameter: /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/" in text

    def test_dockerfile_runs_as_non_root(self):
        assert "USER appuser" in (CARTRIDGE_DIR / "Dockerfile").read_text()

    def test_runner_task_lists_match_the_readme(self):
        readme = (CARTRIDGE_DIR / "README.md").read_text()
        for task in (*runner_mod.DROID_TASKS, *runner_mod.YAM_TASKS):
            assert f"`{task}`" in readme, task


# --------------------------------------------------------------------------
# Build-time MoE shared-memory patch
# --------------------------------------------------------------------------

_KERNEL = '''import torch
import triton

DEFAULT_CONFIG = {
    "BM": 128,
    "num_stages": 3,
}


def grouped_mm(x, w, offsets, config=None):
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    return cfg


SWIGLU_CONFIG = {"BM": 128, "num_stages": 3}


def grouped_mm_swiglu(x, w, offsets, config=None):
    cfg = {**SWIGLU_CONFIG, **(config or {})}
    return cfg
'''


def _exec_patched(smem: int) -> dict:
    torch = types.SimpleNamespace(cuda=types.SimpleNamespace(
        is_available=lambda: True,
        get_device_properties=lambda i: types.SimpleNamespace(shared_memory_per_block_optin=smem),
    ))
    ns: dict = {"torch": torch}
    code = patch_mod.patch(_KERNEL).replace("import torch\nimport triton\n", "")
    exec(compile(code, "<patched>", "exec"), ns)  # noqa: S102 - test fixture
    return ns


class TestMoePatch:
    def test_caps_stages_on_small_shared_memory(self):
        ns = _exec_patched(101_376)  # RTX PRO 6000 / L40S class
        assert ns["grouped_mm"](None, None, None)["num_stages"] == 2
        assert ns["grouped_mm_swiglu"](None, None, None)["num_stages"] == 2

    def test_leaves_hopper_untouched(self):
        ns = _exec_patched(232_448)  # H100
        assert ns["grouped_mm"](None, None, None)["num_stages"] == 3
        assert ns["grouped_mm_swiglu"](None, None, None)["num_stages"] == 3

    def test_only_num_stages_changes(self):
        ns = _exec_patched(101_376)
        assert ns["grouped_mm"](None, None, None, {"BM": 64})["BM"] == 64

    def test_refuses_to_patch_twice(self):
        with pytest.raises(ValueError, match="already patched"):
            patch_mod.patch(patch_mod.patch(_KERNEL))

    def test_refuses_when_upstream_moved(self):
        with pytest.raises(ValueError, match="SWIGLU_CONFIG"):
            patch_mod.patch(_KERNEL.replace("cfg = {**SWIGLU_CONFIG", "cfg = {**OTHER_CONFIG"))

    def test_cli_fails_the_build_on_unknown_text(self, tmp_path):
        target = tmp_path / "moe_gemm_triton.py"
        target.write_text("nothing to anchor on\n")
        assert patch_mod.main(["patch_moe_smem.py", str(target)]) == 1
        assert target.read_text() == "nothing to anchor on\n"


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


def _model_dir(root: Path, *, yam: bool = True, nested: bool = False) -> Path:
    """A model dir laid out as endpoint.yaml's `hf:` subdirs (optionally double-nested)."""
    model = root / "model"

    def place(subdir: str) -> Path:
        d = model / subdir / subdir if nested else model / subdir
        d.mkdir(parents=True, exist_ok=True)
        return d

    (place("praxis-1-droid") / "config.json").write_text("{}")
    if yam:
        (place("praxis-1-yam") / "config.json").write_text("{}")
    (place("wan2_2") / "Wan2.2_VAE.pth").write_bytes(b"vae")
    (place("qwen3_5-2b") / "config.json").write_text("{}")
    return model


def _repo(root: Path) -> Path:
    repo = root / "praxis-1"
    for rel in ("scripts/run_sim.py", "scripts/run_sim_yam.py",
                "configs/sim_droid.yaml", "configs/sim_yam.yaml"):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text("")
    return repo


@pytest.fixture
def runner(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_mod, "PRAXIS_REPO", str(_repo(tmp_path)))
    monkeypatch.setenv("PRAXIS_CACHE_DIR", str(tmp_path / "cache"))
    r = runner_mod.create_runner()
    r.setup(str(_model_dir(tmp_path)), "cpu", 0, 1)
    return r


def _arg(cmd: list[str], key: str) -> str:
    """The value of `--set=<key>=...` (or `--<key>=...`) in an argv."""
    for tok in cmd:
        for prefix in (f"--set={key}=", f"--{key}="):
            if tok.startswith(prefix):
                return tok[len(prefix):]
    raise KeyError(key)


class TestSetup:
    def test_finds_both_rigs(self, runner):
        assert runner.available_rigs == {"droid", "yam"}

    def test_yam_is_optional(self, tmp_path, monkeypatch):
        monkeypatch.setattr(runner_mod, "PRAXIS_REPO", str(_repo(tmp_path)))
        r = runner_mod.create_runner()
        r.setup(str(_model_dir(tmp_path, yam=False)), "cpu", 0, 1)
        assert r.available_rigs == {"droid"}
        with pytest.raises(ValueError, match="not staged"):
            r.resolve({"rig": "yam"})

    def test_accepts_double_nested_staging(self, tmp_path, monkeypatch):
        """stage-weights.py --subdir lands files at <model>/<subdir>/<subdir>."""
        monkeypatch.setattr(runner_mod, "PRAXIS_REPO", str(_repo(tmp_path)))
        r = runner_mod.create_runner()
        model = _model_dir(tmp_path, nested=True)
        r.setup(str(model), "cpu", 0, 1)
        assert r.vae == model / "wan2_2" / "wan2_2" / "Wan2.2_VAE.pth"
        assert r.checkpoints["droid"] == model / "praxis-1-droid" / "praxis-1-droid"

    def test_missing_weights_fail_at_boot(self, tmp_path, monkeypatch):
        monkeypatch.setattr(runner_mod, "PRAXIS_REPO", str(_repo(tmp_path)))
        model = _model_dir(tmp_path)
        (model / "wan2_2" / "Wan2.2_VAE.pth").unlink()
        with pytest.raises(FileNotFoundError, match="Wan2.2_VAE.pth"):
            runner_mod.create_runner().setup(str(model), "cpu", 0, 1)

    def test_refuses_multi_rank(self, tmp_path):
        with pytest.raises(RuntimeError, match="single GPU"):
            runner_mod.create_runner().setup(str(tmp_path), "cpu", 0, 8)


class TestResolve:
    def test_defaults(self, runner):
        assert runner.resolve({"prompt": ""}) == ("droid", "lift_cube", None, [0])
        assert runner.resolve({"rig": "yam"}) == ("yam", "dice_in_bag", None, [0])

    @pytest.mark.parametrize("seeds, seed, expected", [
        ([0, 1], None, [0, 1]),
        ("3, 5", None, [3, 5]),
        (None, "7", [7]),
        (None, "-1", [0]),       # the UI's "random"
        ("", None, [0]),
    ])
    def test_seeds(self, runner, seeds, seed, expected):
        assert runner.resolve({"seeds": seeds, "seed": seed})[3] == expected

    @pytest.mark.parametrize("params, match", [
        ({"rig": "aloha"}, "rig must be"),
        ({"task": "lift_cube; rm -rf /"}, "not a droid task"),
        ({"rig": "yam", "task": "lift_cube"}, "not a yam task"),
        ({"seeds": [0, 0]}, "repeat"),
        ({"seeds": "0,\u00b2"}, "not a non-negative integer"),
        ({"seeds": list(range(9))}, "1 to 8"),
        ({"seeds": [-1]}, "non-negative"),
        ({"seeds": [True]}, "non-negative"),
        ({"seeds": [10**7]}, "larger"),
        ({"prompt": "x" * 501}, "prompt"),
        ({"prompt": "line one\nline two"}, "prompt"),
    ])
    def test_rejects(self, runner, params, match):
        with pytest.raises(ValueError, match=match):
            runner.resolve(params)


class TestCommand:
    def test_droid_argv(self, runner, tmp_path):
        cmd = runner.command("droid", "block_in_bowl", None, [0, 1], tmp_path / "out")
        assert cmd[0] == sys.executable and cmd[2] == "scripts/run_sim.py"
        assert _arg(cmd, "config") == "configs/sim_droid.yaml"
        assert _arg(cmd, "task.name") == "block_in_bowl"
        assert _arg(cmd, "task.prompt") == "null"
        assert _arg(cmd, "task.time_limit_s") == "null"
        assert _arg(cmd, "viewer.enabled") == "false"
        assert _arg(cmd, "scene.download") == "false"
        assert json.loads(_arg(cmd, "episodes.seeds")) == [0, 1]
        assert json.loads(_arg(cmd, "model.checkpoint")) == str(runner.checkpoints["droid"])
        assert json.loads(_arg(cmd, "output.dir")) == str(tmp_path / "out")

    def test_droid_prompt_is_one_yaml_string(self, runner, tmp_path):
        import yaml

        hostile = 'Lift it. ", task.name: x, [1] --set=viewer.enabled=true'
        cmd = runner.command("droid", "lift_cube", hostile, [0], tmp_path)
        value = _arg(cmd, "task.prompt")
        # upstream parses the value with yaml.safe_load: it must come back verbatim.
        assert yaml.safe_load(value) == hostile
        assert sum(t.startswith("--set=viewer.enabled=") for t in cmd) == 1

    def test_yam_argv(self, runner, tmp_path):
        cmd = runner.command("yam", "apple_in_bag", "--checkpoint=/etc", [2, 4], tmp_path)
        assert cmd[2] == "scripts/run_sim_yam.py"
        assert _arg(cmd, "units") == "apple_in_bag:2,apple_in_bag:4"
        assert _arg(cmd, "checkpoint") == str(runner.checkpoints["yam"])
        # `--prompt=<value>` is one argparse token, whatever the value starts with.
        assert "--prompt=--checkpoint=/etc" in cmd
        assert sum(t.startswith("--checkpoint=") for t in cmd) == 1

    def test_child_env_drops_torchrun_and_renders_headless(self, runner, monkeypatch):
        for key in ("RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"):
            monkeypatch.setenv(key, "0")
        env = runner.child_env()
        assert not {"RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"} & set(env)
        assert env["MUJOCO_GL"] == "egl"
        assert env["HF_HUB_OFFLINE"] == "1"
        assert env["PRAXIS_CACHE_DIR"] == str(runner.cache_dir)


class FakeChild:
    """Stands in for subprocess.run: plays upstream's sim scripts and ffmpeg."""

    def __init__(self, successes: dict[int, bool], returncode: int = 0, timeout: bool = False):
        self.successes = successes
        self.returncode = returncode
        self.timeout = timeout
        self.calls: list[dict] = []

    def __call__(self, cmd, **kw):
        self.calls.append({"cmd": cmd, **kw})
        if "-f" in cmd and "concat" in cmd:          # ffmpeg join
            listing = Path(cmd[cmd.index("-i") + 1]).read_text()
            parts = re.findall(r"file '(.+)'", listing)
            Path(cmd[-1]).write_bytes(b"".join(Path(p).read_bytes() for p in parts))
            return subprocess.CompletedProcess(cmd, 0, b"", b"")
        if self.timeout:
            raise subprocess.TimeoutExpired(cmd, kw.get("timeout"))
        kw["stdout"].write("loading the model policy\nTraceback: boom\n" if self.returncode else "ok\n")
        if self.returncode:
            return subprocess.CompletedProcess(cmd, self.returncode)
        if cmd[2] == "scripts/run_sim.py":
            out = Path(json.loads(_arg(cmd, "output.dir")))
            per = []
            for s, ok in self.successes.items():
                (out / f"episode_{s:03d}.mp4").write_bytes(f"<ep{s}>".encode())
                per.append({"seed": s, "success": ok, "video": str(out / f"episode_{s:03d}.mp4")})
            (out / "summary.json").write_text(json.dumps({"episodes": len(per), "per_episode": per}))
        else:
            out = Path(_arg(cmd, "out"))
            task = _arg(cmd, "units").split(":")[0]
            (out / task).mkdir(parents=True, exist_ok=True)
            with (out / "episodes.jsonl").open("w") as log:
                for s, ok in self.successes.items():
                    (out / task / f"episode_{s:03d}.mp4").write_bytes(f"<ep{s}>".encode())
                    log.write(json.dumps({"task": task, "seed": s, "success": ok}) + "\n")
            (out / "summary.json").write_text(json.dumps({task: {"successes": 1}}))
        return subprocess.CompletedProcess(cmd, 0)


class TestGenerate:
    def test_droid_job(self, runner, tmp_path, monkeypatch):
        child = FakeChild({0: True, 1: False})
        monkeypatch.setattr(runner_mod.subprocess, "run", child)
        out = tmp_path / "job" / "output.mp4"
        result = runner.generate(task="block_in_bowl", seeds=[0, 1], prompt="", output_path=str(out))

        assert result == str(out)
        assert out.read_bytes() == b"<ep0><ep1>"   # episodes back to back, in seed order
        side = tmp_path / "job" / "output"
        summary = json.loads((side / "summary.json").read_text())
        assert summary["rig"] == "droid" and summary["task"] == "block_in_bowl"
        assert summary["successes"] == 1 and summary["episodes"] == 2
        assert [(e["seed"], e["success"]) for e in summary["per_episode"]] == [(0, True), (1, False)]
        assert (side / "upstream_summary.json").is_file()
        assert (side / "rollout.log").read_text() == "ok\n"

        sim = child.calls[0]
        assert sim["cwd"] == runner.repo and "shell" not in sim
        assert sim["timeout"] == runner_mod.BASE_TIMEOUT_S + 2 * runner_mod.PER_EPISODE_TIMEOUT_S

    def test_yam_job(self, runner, tmp_path, monkeypatch):
        monkeypatch.setattr(runner_mod.subprocess, "run", FakeChild({3: True}))
        out = tmp_path / "job" / "output.mp4"
        runner.generate(rig="yam", task="bell_in_bag", seeds="3", output_path=str(out))
        assert out.read_bytes() == b"<ep3>"
        summary = json.loads((tmp_path / "job" / "output" / "summary.json").read_text())
        assert summary["per_episode"] == [{"seed": 3, "success": True, "video": "bell_in_bag/episode_003.mp4"}]

    def test_child_failure_is_reported_with_the_log_tail(self, runner, tmp_path, monkeypatch):
        monkeypatch.setattr(runner_mod.subprocess, "run", FakeChild({0: True}, returncode=1))
        with pytest.raises(RuntimeError, match="exit 1.*Traceback: boom"):
            runner.generate(output_path=str(tmp_path / "job" / "output.mp4"))

    def test_timeout(self, runner, tmp_path, monkeypatch):
        monkeypatch.setattr(runner_mod.subprocess, "run", FakeChild({0: True}, timeout=True))
        with pytest.raises(RuntimeError, match="exceeded"):
            runner.generate(output_path=str(tmp_path / "job" / "output.mp4"))

    def test_missing_episode_is_an_error(self, runner, tmp_path, monkeypatch):
        monkeypatch.setattr(runner_mod.subprocess, "run", FakeChild({0: True}))  # seed 1 never written
        with pytest.raises(RuntimeError, match="no result for seed 1"):
            runner.generate(seeds=[0, 1], output_path=str(tmp_path / "job" / "output.mp4"))

    def test_invalid_params_never_start_a_child(self, runner, tmp_path, monkeypatch):
        child = FakeChild({0: True})
        monkeypatch.setattr(runner_mod.subprocess, "run", child)
        with pytest.raises(ValueError):
            runner.generate(task="../../etc", output_path=str(tmp_path / "o.mp4"))
        assert child.calls == []
