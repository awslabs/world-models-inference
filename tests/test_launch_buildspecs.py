# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Launch Stack's catalogue and buildspecs (scripts/launch_buildspecs.py).

The one-click template is published before the account that launches it exists,
so its buildspecs refer to CodeBuild environment variables instead of an account,
Region or bucket. These tests pin that they are otherwise the deploy.sh commands,
that nothing account-specific or CloudFormation-substitutable leaks into them,
and that the licence gate's inputs are carried through. No AWS calls.
"""

import importlib.util
import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lb = _load("launch_buildspecs", "launch_buildspecs.py")
CATALOGUE = {m: lb.describe(m) for m in lb.cartridges()}


def _commands(buildspec: str) -> list:
    spec = json.loads(buildspec)
    return [c for phase in spec["phases"].values() for c in phase["commands"]]


def test_every_cartridge_is_described():
    assert set(CATALOGUE) == set(lb.cartridges())
    assert "_template" not in CATALOGUE


@pytest.mark.parametrize("name", ["lingbot-v2", "lingbot-world-v2-14b"])
def test_lingbot_v2_is_refused(name):
    with pytest.raises(SystemExit, match="never be offered"):
        lb.describe(name)


@pytest.mark.parametrize("model", sorted(CATALOGUE))
def test_licence_fields_follow_the_manifest(model):
    entry = CATALOGUE[model]
    assert entry["license"] and entry["licenseUrl"]
    assert entry["licenseRestricted"] is (lb.bi.manifest_scalar(model, "license_restricted") == "true")


def test_restricted_cartridges_are_flagged():
    restricted = {m for m, e in CATALOGUE.items() if e["licenseRestricted"]}
    assert {"lyra-2", "waypoint-1-5"} <= restricted


@pytest.mark.parametrize("model", sorted(CATALOGUE))
def test_buildspecs_hold_nothing_account_specific(model):
    entry = CATALOGUE[model]
    for spec in [entry["image"], *(w["buildspec"] for w in entry["weights"])]:
        # ${...} would read as a CloudFormation Fn::Sub placeholder in the template.
        assert "${" not in spec
        assert not re.search(r"\b[0-9]{12}\b", spec)
        assert "amazonaws.com" not in spec


@pytest.mark.parametrize("model", sorted(CATALOGUE))
def test_image_buildspec_is_the_deploy_sh_one(model):
    """Same commands as build-image.py, with the registry and Region as variables."""
    launch = CATALOGUE[model]["image"]
    registry, region = "registry.example", "us-east-1"
    expected = _commands(lb.bi.make_buildspec(model, registry, region))
    got = [c.replace("$WM_REGISTRY", registry).replace('"$AWS_REGION"', region) for c in _commands(launch)]
    assert got == expected


@pytest.mark.parametrize("model", sorted(CATALOGUE))
def test_every_weight_source_is_pinned_and_marked(model):
    for w in CATALOGUE[model]["weights"]:
        assert re.fullmatch(r"[0-9a-f]{40}", w["revision"])
        assert w["marker"] == f"_staged/{model}/{w['revision']}"
        cmds = _commands(w["buildspec"])
        assert f"--revision {w['revision']}" in " ".join(cmds)
        assert any(w["marker"] in c and c.startswith(f"echo {w['revision']} |") for c in cmds)


def test_subdir_weights_land_once_under_the_model_prefix():
    """stage-weights.py appends --subdir to the destination and double-nests it."""
    fast = next(w for w in CATALOGUE["lingbot-fast"]["weights"] if w["subdir"])
    sync = next(c for c in _commands(fast["buildspec"]) if c.startswith("aws s3 sync"))
    assert sync == 'aws s3 sync /tmp/weights/ "s3://$ARTIFACTS_BUCKET/lingbot-fast/" --region "$AWS_REGION"'  # nosec B108


def test_hf_entries_match_the_manifest_lines():
    for model, entry in CATALOGUE.items():
        lines = [ln for ln in (lb.MODELS_DIR / model / "endpoint.yaml").read_text().splitlines()
                 if ln.strip().startswith("- { repo:")]
        assert len(entry["weights"]) == len(lines), model


def test_ec2_fields_are_read():
    assert CATALOGUE["lingbot-fast"]["ec2Instance"] == "p5.48xlarge"
    assert CATALOGUE["lingbot-fast"]["ec2VolumeSize"] == 500
    assert CATALOGUE["echo-async"]["ec2VolumeSize"] == 200  # endpoint.yaml default
    assert CATALOGUE["vjepa2"]["modelDataPrefix"] == "vjepa2-vitg-fpc64-384/"
