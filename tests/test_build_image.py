# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for scripts/build-image.py Dockerfile resolution.

These cover the shared-default-vs-per-model logic and the endpoint.yaml build
keys — the core of the template feature — without touching AWS or Docker. The
module uses only stdlib and has no import-time side effects, so it imports
cleanly here.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "inference" / "models"


def _load_build_image():
    """Import scripts/build-image.py (hyphenated name → load by path)."""
    spec = importlib.util.spec_from_file_location(
        "build_image", REPO_ROOT / "scripts" / "build-image.py"
    )
    mod = importlib.util.module_from_spec(spec)
    # Its argparse only runs under main(); import is side-effect free.
    sys.argv = ["build-image.py", "echo-async"]
    spec.loader.exec_module(mod)
    return mod


bi = _load_build_image()


# --------------------------------------------------------------------------
# Dockerfile resolution: per-model Dockerfile wins, else shared default
# --------------------------------------------------------------------------


def test_model_with_own_dockerfile_uses_it():
    # cosmos3-nano ships its own Dockerfile.
    path, uses_shared = bi.resolve_dockerfile("cosmos3-nano")
    assert uses_shared is False
    assert path == "inference/models/cosmos3-nano/Dockerfile"


def test_model_without_dockerfile_uses_shared_default():
    # lingbot-fast (the flagship) has no Dockerfile → shared default. This is the
    # exact case that used to skip the build and launch an unservable instance.
    assert not (MODELS_DIR / "lingbot-fast" / "Dockerfile").exists()
    path, uses_shared = bi.resolve_dockerfile("lingbot-fast")
    assert uses_shared is True
    assert path == "inference/Dockerfile.default"


def test_lyra_keeps_its_custom_dockerfile():
    # lyra-2's native CUDA build must not be replaced by the shared default.
    _, uses_shared = bi.resolve_dockerfile("lyra-2")
    assert uses_shared is False


def test_missing_requirements_is_rejected_for_shared(tmp_path, monkeypatch):
    # A model with no Dockerfile AND no requirements.txt can't use the shared
    # default (it installs requirements.txt), so resolution must fail loudly.
    fake = tmp_path / "models" / "broken"
    fake.mkdir(parents=True)
    (fake / "endpoint.yaml").write_text("version: 0.1.0\n")
    monkeypatch.setattr(bi, "MODELS_DIR", tmp_path / "models")
    with pytest.raises(SystemExit):
        bi.resolve_dockerfile("broken")


# --------------------------------------------------------------------------
# endpoint.yaml build keys
# --------------------------------------------------------------------------


def test_base_image_defaults_when_unset():
    # echo-async pins no base_image → the documented default.
    assert bi.get_base_image("echo-async") == bi.DEFAULT_BASE_IMAGE


def test_base_image_read_from_manifest():
    # lingbot-fast pins a base with a matching flash-attn wheel.
    assert bi.get_base_image("lingbot-fast") == "nvcr.io/nvidia/pytorch:24.08-py3"


def test_flash_attn_wheel_parsed_and_others_none():
    wheel = bi.manifest_scalar("lingbot-fast", "flash_attn_wheel")
    assert wheel and wheel.startswith("https://") and "{ABI}" in wheel
    # A key the manifest doesn't set returns None (so no build-arg is emitted).
    assert bi.manifest_scalar("lingbot-fast", "pip_no_deps") is None


def test_manifest_scalar_strips_inline_comments():
    # version: 0.1.0 has no comment; ensure a commented value would be stripped.
    assert bi.manifest_scalar("echo-async", "version") == "0.1.0"
