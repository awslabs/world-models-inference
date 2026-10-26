#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Describe every Launch Stack cartridge, with the buildspecs its build runs.

The one-click Launch Stack (cdk/bin/launch.ts) builds weights and images inside
the customer's account, from a template published before that account exists.
Its buildspecs therefore cannot name the account, region or bucket: they refer
to CodeBuild environment variables instead ($ARTIFACTS_BUCKET, $WM_REGISTRY and
CodeBuild's own $AWS_REGION), which the build orchestration sets at run time.

The commands themselves come from the same functions scripts/stage-weights.py
and scripts/build-image.py use, so the two paths cannot drift apart.

Usage:
  python scripts/launch_buildspecs.py > /tmp/launch-buildspecs.json
"""

import importlib.util
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "inference" / "models"

# LingBot World V2's weights are CC BY-NC-SA (non-commercial). It is not a
# cartridge today; this stops one being added to the public template by accident.
NEVER_OFFERED = re.compile(r"^lingbot-(world-)?v2")

_PREFIX_RE = re.compile(r"^[A-Za-z0-9_-]+(/[A-Za-z0-9_-]+)*/?$")
_HF_ENTRY_RE = re.compile(r"^\s*-\s*\{(.*)\}\s*(#.*)?$")


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sw = _load("stage_weights", "stage-weights.py")
bi = _load("build_image", "build-image.py")


def cartridges() -> list:
    """Every deployable cartridge: inference/models/<id>/endpoint.yaml, minus scaffolds."""
    return sorted(
        p.name for p in MODELS_DIR.iterdir()
        if p.is_dir() and not p.name.startswith("_") and (p / "endpoint.yaml").is_file()
    )


def hf_entries(model: str) -> list:
    """The `hf:` list of endpoint.yaml as (repo, subdir, revision), in file order."""
    entries, in_hf = [], False
    for line in (MODELS_DIR / model / "endpoint.yaml").read_text().splitlines():
        if re.match(r"^hf:\s*(#.*)?$", line):
            in_hf = True
            continue
        if in_hf and line and not line[0].isspace():
            in_hf = False
        m = _HF_ENTRY_RE.match(line) if in_hf else None
        if not m:
            continue
        fields = dict(
            (k.strip(), v.strip().strip('"\''))
            for k, _, v in (f.partition(":") for f in m.group(1).split(","))
        )
        entries.append((fields.get("repo", ""), fields.get("subdir", ""), fields.get("revision", "")))
    return entries


def model_data_prefix(model: str) -> str:
    raw = bi.manifest_scalar(model, "model_data") or ""
    prefix = re.sub(r"^s3://[^/]+/", "", raw)
    return bi.checked(prefix, _PREFIX_RE, f"{model} model_data prefix")


def weights_buildspec(model: str, repo: str, subdir: str, revision: str) -> str:
    sw._checked(revision, sw._SHA_RE, f"{model}: revision of {repo}")
    # The sync copies /tmp/weights/ — which already holds <subdir>/ — to the
    # model's prefix, so the subdir is NOT appended here (stage-weights.py appends
    # it, and the files land one level too deep).
    prefix = model_data_prefix(model).rstrip("/")
    # $VAR rather than ${VAR}: the buildspec sits in a CloudFormation template,
    # where ${...} reads as an Fn::Sub placeholder.
    dest = f'"s3://$ARTIFACTS_BUCKET/{prefix}/"'
    commands = sw.build_commands_with(f"hf:{repo}", subdir, dest, '"$AWS_REGION"', revision)
    # A marker per staged revision, outside the model prefix so the instance
    # never syncs it: a relaunch or stack update finds it and skips the download.
    marker = f'"s3://$ARTIFACTS_BUCKET/{staged_marker(model, revision)}"'
    commands.insert(-1, f'echo {revision} | aws s3 cp - {marker} --region "$AWS_REGION"')
    return json.dumps({"version": "0.2", "phases": {"build": {"commands": commands}}})


def staged_marker(model: str, revision: str) -> str:
    return f"_staged/{model}/{revision}"


def describe(model: str) -> dict:
    if NEVER_OFFERED.match(model):
        sys.exit(f"{model} must never be offered: its weights licence is non-commercial (docs/MODEL_LICENSES.md)")
    bi.checked(model, bi._MODEL_RE, "model id")
    restricted = bi.manifest_scalar(model, "license_restricted")
    if restricted not in ("true", "false"):
        sys.exit(f"{model}: license_restricted must be true or false, got {restricted!r}")
    weights = []
    for repo, subdir, revision in hf_entries(model):
        weights.append({
            "repo": repo,
            "subdir": subdir,
            "revision": revision,
            "marker": staged_marker(model, revision),
            "buildspec": weights_buildspec(model, repo, subdir, revision),
        })
    return {
        "version": bi.get_version(model),
        "ec2Instance": _ec2_field(model, "instance", r"^[a-z][a-z0-9-]*\.[a-z0-9]+$"),
        "ec2VolumeSize": int(_ec2_field(model, "volume_size", r"^[0-9]{2,5}$", default="200")),
        "modelDataPrefix": model_data_prefix(model),
        "license": bi.manifest_scalar(model, "license"),
        "licenseUrl": bi.manifest_scalar(model, "license_url"),
        "licenseRestricted": restricted == "true",
        "weights": weights,
        "image": bi.make_buildspec(model, "$WM_REGISTRY", '"$AWS_REGION"'),
    }


def _ec2_field(model: str, key: str, pattern: str, default: str = "") -> str:
    """A scalar under endpoint.yaml's `ec2:` mapping, shape-checked."""
    text = (MODELS_DIR / model / "endpoint.yaml").read_text()
    block = re.search(r"^ec2:\s*\n((?:[ \t]+.*\n?)*)", text, re.MULTILINE)
    found = re.search(rf"^\s+{key}:\s*([^\s#]+)", block.group(1), re.MULTILINE) if block else None
    value = found.group(1) if found else default
    if not value:
        sys.exit(f"{model}: ec2.{key} required")
    return bi.checked(value, re.compile(pattern), f"{model} ec2.{key}")


def main() -> None:
    json.dump({"models": {m: describe(m) for m in cartridges()}}, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
