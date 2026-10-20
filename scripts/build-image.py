#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build and push a model container image via CodeBuild.

Triggers a CodeBuild build that clones the repo, builds the Dockerfile,
and pushes to ECR. No local Docker needed.

Usage:
  python scripts/build-image.py echo-async
  python scripts/build-image.py vjepa2
  python scripts/build-image.py lingbot-fast --region us-west-2
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "inference" / "models"


# The buildspec below is a shell script that runs on the CodeBuild host under the
# build role, and several of its values come from the cartridge's endpoint.yaml —
# third-party content. Each value is checked against the shape it legitimately
# has before it is interpolated, and shell-quoted as well; validation is the
# gate, quoting the backstop.
_MODEL_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_-]*$")
_TAG_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]{0,127}$")
# A base image must name an explicit version: a tag other than "latest", or a digest.
_IMAGE_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._/-]*"
    r"(:(?!latest$)[A-Za-z0-9_][A-Za-z0-9._-]{0,127}|@sha256:[a-f0-9]{64})$"
)
_WHEEL_RE = re.compile(r"^https://[A-Za-z0-9._~/%+{}:-]+\.whl$")
_FLAG_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_REGION_RE = re.compile(r"^[a-z]{2}(-[a-z]+)+-[0-9]+$")


def checked(value: str, pattern: re.Pattern, what: str) -> str:
    """Return ``value`` if it fully matches ``pattern``; exit otherwise."""
    if not pattern.fullmatch(value):
        sys.exit(f"Invalid {what}: {value!r}")
    return value


def run(cmd, timeout=30):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def ssm_get(name, region):
    r = run(["aws", "ssm", "get-parameter", "--name", name,
             "--query", "Parameter.Value", "--output", "text", "--region", region])
    if r.returncode != 0:
        sys.exit(f"SSM param '{name}' not found. Deploy foundation stack first.")
    return r.stdout.strip()


def get_version(endpoint_id: str) -> str:
    manifest = MODELS_DIR / endpoint_id / "endpoint.yaml"
    if not manifest.exists():
        sys.exit(f"No endpoint.yaml for {endpoint_id}")
    m = re.search(r"^version:\s*(.+)$", manifest.read_text(), re.MULTILINE)
    return checked(m.group(1).strip() if m else "latest", _TAG_RE, "endpoint.yaml version")


# Default base image when a cartridge doesn't pin one in endpoint.yaml. Kept in
# sync with the ARG default in inference/Dockerfile.default.
DEFAULT_BASE_IMAGE = "nvcr.io/nvidia/pytorch:24.01-py3"


def manifest_scalar(endpoint_id: str, key: str):
    """Read a top-level scalar `key: value` from endpoint.yaml, or None."""
    manifest = MODELS_DIR / endpoint_id / "endpoint.yaml"
    m = re.search(rf"^{re.escape(key)}:\s*(.+)$", manifest.read_text(), re.MULTILINE)
    if not m:
        return None
    # Strip inline comments and surrounding quotes.
    return m.group(1).split("#")[0].strip().strip("\"'") or None


def get_base_image(endpoint_id: str) -> str:
    """Read an optional `base_image:` from endpoint.yaml (shared Dockerfile only)."""
    image = manifest_scalar(endpoint_id, "base_image") or DEFAULT_BASE_IMAGE
    return checked(image, _IMAGE_RE, "endpoint.yaml base_image")


def resolve_dockerfile(endpoint_id: str):
    """Pick the model's own Dockerfile if present, else the shared default.

    Returns (dockerfile_path_relative_to_repo_root, uses_shared_default).
    A per-model Dockerfile always wins, so cartridges with genuinely custom
    build steps (native CUDA extensions, upstream clones) keep full control.
    """
    own = MODELS_DIR / endpoint_id / "Dockerfile"
    if own.exists():
        return f"inference/models/{endpoint_id}/Dockerfile", False

    shared = REPO_ROOT / "inference" / "Dockerfile.default"
    if not shared.exists():
        sys.exit(f"No Dockerfile for {endpoint_id} and {shared} is missing")
    if not (MODELS_DIR / endpoint_id / "requirements.txt").exists():
        sys.exit(
            f"{endpoint_id} has no Dockerfile, so the shared default would be used — "
            f"but it also has no requirements.txt. Add one (it may be empty)."
        )
    return "inference/Dockerfile.default", True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="Endpoint ID (directory under inference/models/)")
    parser.add_argument("--region", default=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    args = parser.parse_args()

    endpoint_id = checked(args.model, _MODEL_RE, "model id")
    checked(args.region, _REGION_RE, "--region")
    if not (MODELS_DIR / endpoint_id).is_dir():
        sys.exit(f"No such model directory: inference/models/{endpoint_id}")
    dockerfile, uses_shared = resolve_dockerfile(endpoint_id)

    account = run(["aws", "sts", "get-caller-identity", "--query", "Account", "--output", "text"]).stdout.strip()
    if not account:
        sys.exit("AWS CLI not configured")

    project_name = ssm_get("/world-model/foundation/image-project", args.region)
    version = get_version(endpoint_id)
    repo_name = f"world-model-{endpoint_id}"
    image_uri = f"{account}.dkr.ecr.{args.region}.amazonaws.com/{repo_name}:{version}"

    # Ensure the ECR repo exists with vulnerability scanning on push, so every
    # image this pipeline builds is scanned for known CVEs in its OS and language
    # packages, and with immutable tags, so a pushed version can never be
    # overwritten by a later build (EDSR WM-015). The :buildcache tag is the one
    # exclusion: every build rewrites it, and plain IMMUTABLE would quietly break
    # the layer cache. Both settings are applied to existing repos too.
    mutability = [
        "--image-tag-mutability", "IMMUTABLE_WITH_EXCLUSION",
        "--image-tag-mutability-exclusion-filters", "filterType=WILDCARD,filter=buildcache",
    ]
    r = run(["aws", "ecr", "describe-repositories", "--repository-names", repo_name, "--region", args.region])
    if r.returncode != 0:
        r = run(["aws", "ecr", "create-repository", "--repository-name", repo_name,
                 "--image-scanning-configuration", "scanOnPush=true", *mutability, "--region", args.region])
        if r.returncode != 0:
            sys.exit(f"Failed to create ECR repository {repo_name} (IMMUTABLE_WITH_EXCLUSION needs a "
                     f"recent AWS CLI):\n{r.stderr.strip()}")
    else:
        scan = run(["aws", "ecr", "put-image-scanning-configuration", "--repository-name", repo_name,
                    "--image-scanning-configuration", "scanOnPush=true", "--region", args.region])
        if scan.returncode != 0:
            print(f"  WARNING: could not enable scan-on-push for {repo_name}: {scan.stderr.strip()}")
        mut = run(["aws", "ecr", "put-image-tag-mutability", "--repository-name", repo_name,
                   *mutability, "--region", args.region])
        if mut.returncode != 0:
            sys.exit(f"Failed to make {repo_name} tags immutable (needs a recent AWS CLI):\n{mut.stderr.strip()}")

    # An immutable tag cannot be rebuilt, so fail here rather than after a
    # 30-minute build whose push is refused.
    exists = run(["aws", "ecr", "describe-images", "--repository-name", repo_name,
                  "--image-ids", f"imageTag={version}", "--region", args.region])
    if exists.returncode == 0:
        sys.exit(f"{repo_name}:{version} already exists and tags are immutable. Bump `version:` in "
                 f"inference/models/{endpoint_id}/endpoint.yaml to rebuild.")

    # Upload source (zip the repo)
    import tempfile, zipfile
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        zip_path = tmp.name
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(REPO_ROOT / "inference"):
            dirs[:] = [d for d in dirs if d not in ("__pycache__", ".DS_Store")]
            for f in files:
                if f.endswith(".pyc"):
                    continue
                full = os.path.join(root, f)
                arcname = os.path.relpath(full, REPO_ROOT)
                zf.write(full, arcname)

    bucket = ssm_get("/world-model/foundation/artifacts-bucket", args.region)
    s3_key = f"_builds/{endpoint_id}/source.zip"
    # The source archive can be large (vendored upstream trees, examples), so
    # this upload must NOT use the short default command timeout — a slow link or
    # a big model (lingbot-fast, lyra-2) would otherwise abort mid-upload.
    up = run(["aws", "s3", "cp", zip_path, f"s3://{bucket}/{s3_key}", "--region", args.region],
             timeout=1800)
    if up.returncode != 0:
        os.unlink(zip_path)
        sys.exit(f"Failed to upload build source to s3://{bucket}/{s3_key}:\n{up.stderr}")
    os.unlink(zip_path)

    # Registry-based layer cache: CodeBuild hosts are ephemeral (no local layer
    # cache), so we cache via ECR. Each build pulls the previous image + a
    # dedicated :buildcache tag as cache sources and writes inline cache metadata
    # back. Unchanged early layers (base image, deps, the slow CUDA compiles)
    # become cache hits — only layers at/after the first changed line rebuild.
    cache_tag = f"{account}.dkr.ecr.{args.region}.amazonaws.com/{repo_name}:buildcache"

    # The shared default Dockerfile is parameterised; a per-model Dockerfile is
    # self-contained and takes no build args.
    if uses_shared:
        parts = [
            "--build-arg " + shlex.quote(f"MODEL_ID={endpoint_id}"),
            "--build-arg " + shlex.quote(f"BASE_IMAGE={get_base_image(endpoint_id)}"),
        ]
        # Optional, declared in endpoint.yaml.
        wheel = manifest_scalar(endpoint_id, "flash_attn_wheel")
        if wheel:
            checked(wheel, _WHEEL_RE, "endpoint.yaml flash_attn_wheel")
            parts.append("--build-arg " + shlex.quote(f"FLASH_ATTN_WHEEL={wheel}"))
        no_deps = manifest_scalar(endpoint_id, "pip_no_deps")
        if no_deps is not None:
            checked(no_deps, _FLAG_RE, "endpoint.yaml pip_no_deps")
            parts.append("--build-arg " + shlex.quote(f"PIP_NO_DEPS={no_deps}"))
        build_args = " ".join(parts)
    else:
        build_args = ""

    buildspec = json.dumps({
        "version": "0.2",
        "env": {"variables": {"DOCKER_BUILDKIT": "1"}},
        "phases": {
            "pre_build": {
                "commands": [
                    f"aws ecr get-login-password --region {args.region} | docker login --username AWS --password-stdin {account}.dkr.ecr.{args.region}.amazonaws.com",
                    # Warm the local cache from prior images (ignore failure on first build).
                    f"docker pull {cache_tag} || docker pull {image_uri} || true",
                ]
            },
            "build": {
                "commands": [
                    f"docker build --platform linux/amd64 --provenance=false "
                    f"--build-arg BUILDKIT_INLINE_CACHE=1 "
                    f"{build_args} "
                    f"--cache-from {cache_tag} --cache-from {image_uri} "
                    f"-t {image_uri} -t {cache_tag} "
                    f"-f {dockerfile} .",
                ]
            },
            "post_build": {
                "commands": [
                    f"docker push {image_uri}",
                    # Push the cache tag too so the next build can reuse these layers.
                    f"docker push {cache_tag} || true",
                    f"echo 'Pushed: {image_uri}'",
                ]
            },
        },
    })

    print(f"  Model:   {endpoint_id}")
    print(f"  Image:   {image_uri}")
    if uses_shared:
        print(f"  Build:   {dockerfile} (shared default, base={get_base_image(endpoint_id)})")
    else:
        print(f"  Build:   {dockerfile} (model-specific)")
    print()

    r = run([
        "aws", "codebuild", "start-build",
        "--project-name", project_name,
        "--buildspec-override", buildspec,
        "--source-type-override", "S3",
        "--source-location-override", f"{bucket}/{s3_key}",
        "--region", args.region,
        "--query", "build.id", "--output", "text",
    ])
    if r.returncode != 0:
        sys.exit(f"Failed to start build: {r.stderr}")

    build_id = r.stdout.strip()
    print(f"  Build:   {build_id}")
    print()

    while True:
        time.sleep(15)
        r = run(["aws", "codebuild", "batch-get-builds", "--ids", build_id,
                 "--query", "builds[0].[buildStatus,currentPhase]", "--output", "text",
                 "--region", args.region])
        parts = r.stdout.strip().split()
        status = parts[0] if parts else "UNKNOWN"
        phase = parts[1] if len(parts) > 1 else ""
        sys.stdout.write(f"\r  {status} ({phase})   ")
        sys.stdout.flush()

        if status == "SUCCEEDED":
            print(f"\n\n  Image pushed: {image_uri}")
            return
        if status in ("FAILED", "FAULT", "STOPPED", "TIMED_OUT"):
            print(f"\n\n  Build {status}.")
            print(f"  aws codebuild batch-get-builds --ids {build_id} --query 'builds[0].logs.deepLink' --output text --region {args.region}")
            sys.exit(1)


if __name__ == "__main__":
    main()
