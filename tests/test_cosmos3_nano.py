# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cosmos3-Nano cartridge tests — offline structure + optional live smoke."""

import os
from pathlib import Path

import pytest

CARTRIDGE_DIR = Path(__file__).resolve().parent.parent / "inference" / "models" / "cosmos3-nano"


class TestCartridgeStructure:
    """Verify the cartridge has all required files and correct config."""

    def test_endpoint_yaml_exists(self):
        assert (CARTRIDGE_DIR / "endpoint.yaml").is_file()

    def test_runner_exists(self):
        assert (CARTRIDGE_DIR / "runner.py").is_file()

    def test_uses_shared_serve_entrypoint(self):
        # The cartridge must not ship its own server; it is served by the shared
        # lib/serve entrypoint (which applies the security middleware). A stray
        # server.py would bypass auth/CORS/rate-limiting (R2).
        assert not (CARTRIDGE_DIR / "server.py").exists()
        dockerfile = (CARTRIDGE_DIR / "Dockerfile").read_text()
        assert "./lib/serve" in dockerfile

    def test_requirements_exists(self):
        assert (CARTRIDGE_DIR / "requirements.txt").is_file()

    def test_endpoint_yaml_has_ec2_config(self):
        content = (CARTRIDGE_DIR / "endpoint.yaml").read_text()
        assert "ec2:" in content
        assert "g6e.12xlarge" in content

    def test_endpoint_yaml_has_hf_repo(self):
        content = (CARTRIDGE_DIR / "endpoint.yaml").read_text()
        assert "nvidia/Cosmos3-Nano" in content

    def test_requirements_has_diffusers(self):
        content = (CARTRIDGE_DIR / "requirements.txt").read_text()
        assert "diffusers" in content

    def test_runner_subclasses_runner(self):
        content = (CARTRIDGE_DIR / "runner.py").read_text()
        assert "class Cosmos3NanoPipeline(Runner)" in content
        assert "def setup(" in content
        assert "def generate(" in content


@pytest.mark.skipif(
    not os.environ.get("COSMOS3_ENDPOINT"),
    reason="Set COSMOS3_ENDPOINT=http://host:port to run live tests",
)
class TestLiveEndpoint:
    """Live smoke tests against a running Cosmos3-Nano endpoint."""

    @pytest.fixture
    def endpoint(self):
        return os.environ["COSMOS3_ENDPOINT"]

    def test_health(self, endpoint):
        import requests
        resp = requests.get(f"{endpoint}/cosmos3/health", timeout=10)
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_text_to_video(self, endpoint):
        import requests
        resp = requests.post(
            f"{endpoint}/cosmos3/generate",
            data={"prompt": "A cat walking across a room", "num_frames": 17, "height": 480, "width": 832},
            timeout=600,
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "video/mp4"
        assert len(resp.content) > 10000
