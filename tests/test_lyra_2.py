# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the lyra-2 cartridge — offline structure + optional live smoke.

Unit tests run anywhere — they validate the manifest, Dockerfile, and runner
shape. Live tests only run when `--endpoint` is provided (they submit an async
image-to-video job and confirm an MP4 comes back).

Usage:
    # Unit tests (no endpoint needed)
    pytest tests/test_lyra_2.py -k "not Live"

    # Live tests against a deployed endpoint
    pytest tests/test_lyra_2.py --endpoint http://54.123.45.67:8080
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LYRA_DIR = REPO_ROOT / "inference" / "models" / "lyra-2"


@pytest.fixture
def endpoint(request):
    url = request.config.getoption("--endpoint")
    if not url:
        pytest.skip("No --endpoint provided (pass --endpoint or set LINGBOT_ENDPOINT).")
    return url.rstrip("/")


# ---------------------------------------------------------------------------
# Unit tests — repo layout
# ---------------------------------------------------------------------------


class TestRepoLayout:
    def test_dir_exists(self):
        assert LYRA_DIR.is_dir(), f"Missing {LYRA_DIR}"

    def test_required_files(self):
        for name in ("endpoint.yaml", "runner.py", "requirements.txt", "Dockerfile"):
            assert (LYRA_DIR / name).exists(), f"Missing {name}"


class TestEndpointManifest:
    def test_fields(self):
        content = (LYRA_DIR / "endpoint.yaml").read_text()
        assert "nvidia/Lyra-2.0" in content
        # Async batch model, like lingbot-fast / cosmos3-nano.
        assert re.search(r"^mode:\s*async", content, re.M)

    def test_volume_size(self):
        content = (LYRA_DIR / "endpoint.yaml").read_text()
        m = re.search(r"volume_size:\s*(\d+)", content)
        assert m and int(m.group(1)) >= 300, "volume_size should be ≥300 GB for Lyra weights"

    def test_restricted_license_declared(self):
        # Lyra weights are internal-research-only; the manifest must flag it so
        # deploy.sh can gate deployment on acknowledgement.
        content = (LYRA_DIR / "endpoint.yaml").read_text()
        assert re.search(r"^license_restricted:\s*true", content, re.M)
        assert "license_url:" in content


class TestDeployGate:
    def test_deploy_sh_gates_restricted_license(self):
        content = (REPO_ROOT / "deploy.sh").read_text()
        assert "check_license" in content
        # Gate must be invoked from the deploy path, not just defined.
        assert content.count("check_license") >= 2


class TestDockerfile:
    def test_clones_upstream_repo(self):
        content = (LYRA_DIR / "Dockerfile").read_text()
        assert "nv-tlabs/lyra" in content
        assert "LYRA_REPO" in content

    def test_pins_single_rank(self):
        # Lyra is subprocess/single-GPU: it must not launch a torchrun cluster.
        content = (LYRA_DIR / "Dockerfile").read_text()
        assert "NPROC_PER_NODE=1" in content


class TestRunnerShape:
    def test_subclasses_runner_with_hooks(self):
        content = (LYRA_DIR / "runner.py").read_text()
        assert "class LyraRunner(Runner)" in content
        assert "def setup(" in content
        assert "def generate(" in content
        assert "def create_runner(" in content

    def test_loads_upstream_in_process(self):
        # Load-once: imports upstream loaders + helpers, no subprocess/CLI.
        content = (LYRA_DIR / "runner.py").read_text()
        assert "load_model_from_checkpoint" in content
        assert "_generate_one_direction" in content
        assert "subprocess" not in content

    def test_followers_are_noop(self):
        # Only rank 0 drives the subprocess; other ranks must return early.
        content = (LYRA_DIR / "runner.py").read_text()
        assert "self.rank != 0" in content


# ---------------------------------------------------------------------------
# Live tests — hit a running endpoint
# ---------------------------------------------------------------------------


def _requests():
    return pytest.importorskip("requests")


class TestLiveHealth:
    def test_ping(self, endpoint):
        r = _requests().get(f"{endpoint}/ping", timeout=15)
        assert r.status_code == 200


class TestLiveGeneration:
    """Submit an async image-to-video job and download the result."""

    def test_generate_and_download(self, endpoint, tmp_path):
        requests = _requests()
        image_path = LYRA_DIR / "assets" / "sample.png"
        if not image_path.exists():
            pytest.skip("No bundled sample image to submit.")

        with image_path.open("rb") as f:
            r = requests.post(
                f"{endpoint}/generate",
                data={"prompt": "a slow cinematic fly-through of the scene"},
                files={"image": ("sample.png", f, "image/png")},
                timeout=30,
            )
        assert r.status_code == 200, r.text
        job = r.json()
        job_id = job["job_id"]
        assert job["status"] == "queued"

        deadline = time.time() + 1200  # 20 min budget
        final = None
        while time.time() < deadline:
            rs = requests.get(f"{endpoint}/jobs/{job_id}", timeout=10)
            assert rs.status_code == 200
            st = rs.json()
            if st["status"] == "complete":
                final = st
                break
            if st["status"] == "failed":
                pytest.fail(f"Job failed: {st.get('error')}")
            time.sleep(15)
        assert final is not None, "Job did not complete within 20 min"

        rd = requests.get(f"{endpoint}/jobs/{job_id}/output", timeout=60)
        assert rd.status_code == 200
        out = tmp_path / f"{job_id}.mp4"
        out.write_bytes(rd.content)
        assert out.stat().st_size > 10_000, "MP4 too small to be valid"
