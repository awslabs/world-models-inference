# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The contract every cartridge must satisfy, checked for all of them at once.

The per-cartridge test files pin behaviour that is specific to one model. This
suite is the other half: structural rules that make a cartridge deployable at
all, applied uniformly so the next `inference/models/<id>/` directory gets the
same scrutiny as the existing ones without anyone writing new tests for it.

It also pins the catalogue (frontend/src/data/cartridges.ts) to the manifests:
ids must resolve to endpoint directories, instances must match `ec2.instance`,
and prices must match verified on-demand rates — the catalogue shipped rates
25-100x too high before MR !13 verified them.
"""

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO / "inference" / "models"
CARTRIDGES_TS = REPO / "frontend" / "src" / "data" / "cartridges.ts"

CARTRIDGE_DIRS = sorted(
    d for d in MODELS_DIR.iterdir() if d.is_dir() and not d.name.startswith("_")
)
CARTRIDGE_IDS = [d.name for d in CARTRIDGE_DIRS]

# Verified public us-east-1 on-demand rates (MR !13), as $/sec = hourly / 3600.
# A new instance type must be added here with a rate checked against
# https://aws.amazon.com/ec2/pricing/on-demand/ — that check is the point.
PRICE_PER_SEC = {
    "p5.48xlarge": 0.015289,   # $55.04/hr
    "p5.4xlarge": 0.001911,    # $6.88/hr
    "g6e.12xlarge": 0.002915,  # $10.49/hr
    "g6e.2xlarge": 0.000623,   # $2.24/hr
    "g5.2xlarge": 0.000337,    # $1.212/hr
    "g5.12xlarge": 0.001576,   # $5.672/hr
}

# Known catalogue drift, deferred in MR !13. Shrink these; never grow them
# without a linked follow-up.
CATALOGUE_ONLY_IDS = {
    "vjepa2-ac",      # entry with no matching endpoint directory
}
UNLISTED_DIRS = {
    "echo-realtime",  # never added to cartridges.ts
    "vjepa2",         # listed under the stale id vjepa2-ac (see CATALOGUE_ONLY_IDS)
}


def load_manifest(cartridge: Path) -> dict:
    return yaml.safe_load((cartridge / "endpoint.yaml").read_text())


@pytest.mark.parametrize("cartridge", CARTRIDGE_DIRS, ids=CARTRIDGE_IDS)
class TestEveryCartridge:
    def test_required_files_exist(self, cartridge):
        for name in ("endpoint.yaml", "runner.py", "requirements.txt"):
            assert (cartridge / name).is_file(), f"{cartridge.name} is missing {name}"

    def test_no_private_server(self, cartridge):
        # The shared lib.serve entrypoint is what applies auth, CORS and rate
        # limiting; a cartridge shipping its own server bypasses all of it.
        assert not (cartridge / "server.py").exists()

    def test_dockerfile_uses_shared_entrypoint(self, cartridge):
        dockerfile = cartridge / "Dockerfile"
        if not dockerfile.exists():
            return  # cartridge builds on inference/Dockerfile.default
        content = dockerfile.read_text()
        # Module form (`python -m lib.serve`) or path form (`lib/serve`).
        assert "lib.serve" in content or "lib/serve" in content

    def test_manifest_parses_with_version_and_instance(self, cartridge):
        manifest = load_manifest(cartridge)
        assert manifest.get("version"), "endpoint.yaml needs a version"
        instances = {
            manifest.get("ec2", {}).get("instance"),
            manifest.get("sagemaker", {}).get("instance"),
        } - {None}
        assert instances, "endpoint.yaml needs ec2.instance or sagemaker.instance"

    def test_sagemaker_instance_has_ml_prefix_and_ec2_does_not(self, cartridge):
        manifest = load_manifest(cartridge)
        sm = manifest.get("sagemaker", {}).get("instance")
        ec2 = manifest.get("ec2", {}).get("instance")
        if sm:
            assert sm.startswith("ml."), f"{cartridge.name}: sagemaker instance {sm}"
        if ec2:
            assert not ec2.startswith("ml."), f"{cartridge.name}: ec2 instance {ec2}"

    def test_runner_exports_the_factory(self, cartridge):
        # The shared server imports create_runner by name; without it the
        # container builds fine and dies at boot.
        assert "def create_runner" in (cartridge / "runner.py").read_text()

    def test_realtime_manifests_have_a_streaming_runner(self, cartridge):
        # cdk inferMode(): no sagemaker.output => real-time endpoint. A
        # real-time endpoint whose runner cannot stream accepts a WebSocket
        # and then hangs the session.
        manifest = load_manifest(cartridge)
        is_realtime = "output" not in manifest.get("sagemaker", {})
        if is_realtime:
            assert "def stream(" in (cartridge / "runner.py").read_text(), (
                f"{cartridge.name} is real-time (no sagemaker.output) "
                "but its runner does not define stream()"
            )


def parse_catalogue() -> list[dict]:
    """Pull id/instance/price out of each entry in cartridges.ts.

    Regex, not a TS parser: every entry is a brace-delimited object literal
    with single-quoted strings, and the frontend typechecker (cdk-typecheck CI
    job) already guarantees the file is well-formed TypeScript.
    """
    src = CARTRIDGES_TS.read_text()
    body = src.split("export const CARTRIDGES", 1)[1].split("];", 1)[0]
    entries = []
    for chunk in re.split(r"\n  \{", body)[1:]:
        fields = dict(re.findall(r"(\w+): '?([^',\n]*)'?,", chunk))
        entries.append(fields)
    return entries


CATALOGUE = parse_catalogue()


class TestCatalogueMatchesManifests:
    def test_catalogue_parsed_something(self):
        assert len(CATALOGUE) >= 5
        assert all(e.get("id") and e.get("instance") and e.get("price") for e in CATALOGUE)

    def test_every_catalogue_id_has_an_endpoint_directory(self):
        missing = {e["id"] for e in CATALOGUE} - set(CARTRIDGE_IDS) - CATALOGUE_ONLY_IDS
        assert not missing, f"catalogue entries without inference/models dir: {missing}"

    def test_every_endpoint_directory_is_in_the_catalogue(self):
        missing = set(CARTRIDGE_IDS) - {e["id"] for e in CATALOGUE} - UNLISTED_DIRS
        assert not missing, f"inference/models dirs missing from cartridges.ts: {missing}"

    def test_catalogue_instance_matches_manifest_ec2_instance(self):
        for entry in CATALOGUE:
            cartridge = MODELS_DIR / entry["id"]
            if not cartridge.is_dir():
                continue  # covered (and allowlisted) above
            ec2 = load_manifest(cartridge).get("ec2", {}).get("instance")
            assert entry["instance"] == ec2, (
                f"{entry['id']}: catalogue says {entry['instance']}, "
                f"endpoint.yaml says {ec2}"
            )

    def test_prices_match_verified_on_demand_rates(self):
        for entry in CATALOGUE:
            expected = PRICE_PER_SEC.get(entry["instance"])
            assert expected is not None, (
                f"{entry['id']}: instance {entry['instance']} has no verified "
                "rate in PRICE_PER_SEC — check the AWS pricing page and add it"
            )
            assert float(entry["price"]) == pytest.approx(expected, rel=0.01), (
                f"{entry['id']}: price {entry['price']} != {expected} "
                f"({entry['instance']})"
            )
