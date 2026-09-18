# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for lingbot-fast.

Unit tests run anywhere — they validate manifest + vendored code layout.

Live tests only run when `--endpoint` is provided (or LINGBOT_ENDPOINT env var).
They submit a generate job using example 03 with a short frame count and
confirm an MP4 comes back.

Usage:
    # Unit tests (no endpoint needed)
    pytest tests/test_lingbot_fast.py -k "not live"

    # Live tests against a deployed endpoint
    pytest tests/test_lingbot_fast.py --endpoint http://54.123.45.67:8080
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# pytest config
# ---------------------------------------------------------------------------


@pytest.fixture
def endpoint(request):
    url = request.config.getoption("--endpoint")
    if not url:
        pytest.skip("No --endpoint provided (pass --endpoint or set LINGBOT_ENDPOINT).")
    return url.rstrip("/")


# ---------------------------------------------------------------------------
# Unit tests — repo layout
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
LINGBOT_DIR = REPO_ROOT / "inference" / "models" / "lingbot-fast"


class TestRepoLayout:
    def test_dir_exists(self):
        assert LINGBOT_DIR.is_dir(), f"Missing {LINGBOT_DIR}"

    def test_required_files(self):
        for name in ("endpoint.yaml", "runner.py", "requirements.txt", "README.md"):
            assert (LINGBOT_DIR / name).exists(), f"Missing {name}"

    def test_vendored_wan(self):
        assert (LINGBOT_DIR / "wan" / "image2video_fast.py").exists()
        assert (LINGBOT_DIR / "wan" / "modules" / "model_fast.py").exists()

    def test_examples_present(self):
        for i in ("00", "01", "02", "03", "04", "05"):
            d = LINGBOT_DIR / "examples" / i
            assert d.is_dir(), f"Missing example {i}"
            assert (d / "image.jpg").exists()


class TestEndpointManifest:
    def test_fields(self):
        content = (LINGBOT_DIR / "endpoint.yaml").read_text()
        assert "p5.48xlarge" in content
        assert "robbyant/lingbot-world-base-cam" in content
        assert "robbyant/lingbot-world-fast" in content
        assert re.search(r"^mode:\s*async", content, re.M)

    def test_volume_size(self):
        content = (LINGBOT_DIR / "endpoint.yaml").read_text()
        m = re.search(r"volume_size:\s*(\d+)", content)
        assert m and int(m.group(1)) >= 300, "volume_size should be ≥300 GB for lingbot-fast weights"


class TestDeployCLI:
    def test_cli_exists(self):
        cli_path = REPO_ROOT / "deploy.sh"
        assert cli_path.exists(), "Top-level ./deploy.sh script missing"

    def test_cli_is_executable(self):
        cli_path = REPO_ROOT / "deploy.sh"
        assert os.access(cli_path, os.X_OK), "./deploy.sh should be executable"

    def test_cli_has_subcommands(self):
        content = (REPO_ROOT / "deploy.sh").read_text()
        for cmd in ("status", "destroy", "list", "ui"):
            assert cmd in content, f"CLI missing `{cmd}` subcommand"


# ---------------------------------------------------------------------------
# Live tests — hit a running endpoint
# ---------------------------------------------------------------------------


def _requests():
    requests = pytest.importorskip("requests")
    return requests


class TestLiveHealth:
    def test_health(self, endpoint):
        r = _requests().get(f"{endpoint}/lingbot/health", timeout=15)
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        # p5.48xlarge should report 8-GPU world.
        assert data["world_size"] >= 1

    def test_ping_alias(self, endpoint):
        r = _requests().get(f"{endpoint}/ping", timeout=5)
        assert r.status_code == 200

    def test_examples_listing(self, endpoint):
        r = _requests().get(f"{endpoint}/lingbot/examples", timeout=10)
        assert r.status_code == 200
        data = r.json()
        ids = {e["id"] for e in data["examples"]}
        assert {"00", "03"}.issubset(ids)


class TestLiveGeneration:
    """Short smoke test using bundled example 03 with a low frame count."""

    def test_generate_and_download(self, endpoint, tmp_path):
        requests = _requests()

        # Submit a job via example_id (server already has image + poses).
        r = requests.post(
            f"{endpoint}/lingbot/generate",
            data={
                "prompt": "a serene lakeside scene, cinematic",
                "example_id": "03",
                "frame_num": "21",     # shortest 4n+1 that exercises the pipeline
                "size": "480*832",
                "seed": "42",
            },
            timeout=30,
        )
        assert r.status_code == 200, r.text
        job = r.json()
        job_id = job["job_id"]
        assert job["status"] == "queued"

        # Poll
        deadline = time.time() + 900  # 15 min budget
        final = None
        while time.time() < deadline:
            rs = requests.get(f"{endpoint}/lingbot/status/{job_id}", timeout=10)
            assert rs.status_code == 200
            st = rs.json()
            if st["status"] == "complete":
                final = st
                break
            if st["status"] == "failed":
                pytest.fail(f"Job failed: {st.get('error')}")
            time.sleep(10)
        assert final is not None, "Job did not complete within 15 min"

        # Download result
        rd = requests.get(f"{endpoint}/lingbot/result/{job_id}", timeout=60)
        assert rd.status_code == 200
        assert rd.headers.get("content-type") == "video/mp4"
        out = tmp_path / f"{job_id}.mp4"
        out.write_bytes(rd.content)
        assert out.stat().st_size > 10_000, "MP4 too small to be valid"
