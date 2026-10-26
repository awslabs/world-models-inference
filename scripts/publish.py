#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Publish the one-click Launch Stack template and its assets to S3.

Builds the UI, synthesises cdk/bin/launch.ts, and uploads the template, its
nested templates, Lambda code, the inference/ source and the UI to
s3://<bucket-basename>-<region>/<prefix>/ for each Region, then prints the
CloudFormation console "Launch Stack" link for it.

The bucket must already exist (one per Region; this script never creates or
reconfigures one). For a public Launch Stack its policy must let anyone read
<prefix>/*: see docs/LAUNCH_STACK.md.

Usage:
  python scripts/publish.py --bucket-basename <name> --prefix world-model-inference/v1 --region us-east-1
  python scripts/publish.py --bucket-basename <name> --prefix <p> --region us-east-1 --region us-west-2
  python scripts/publish.py ... --dry-run     # synthesise and list the uploads; no AWS calls
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CDK_DIR = REPO_ROOT / "cdk"
FRONTEND_DIR = REPO_ROOT / "frontend"
STACK = "WorldModelLaunch"
TEMPLATE_NAME = "world-model-launch.template.json"

# <basename>-<region> must be a valid bucket name for the longest Region code
# (14 characters), and the same rule cdk/bin/launch.ts applies.
_BASENAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,40}[a-z0-9]$")
_PREFIX_RE = re.compile(r"^[A-Za-z0-9_-]+(/[A-Za-z0-9_-][A-Za-z0-9._-]*)*$")
_REGION_RE = re.compile(r"^[a-z]{2}(-[a-z]+)+-[0-9]+$")
_STACK_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9-]{0,127}$")
# Fixed timestamp so a re-zip of unchanged content is byte-identical.
_ZIP_DATE = (1980, 1, 1, 0, 0, 0)


def checked(value: str, pattern: re.Pattern, what: str) -> str:
    if not pattern.fullmatch(value):
        sys.exit(f"Invalid {what}: {value!r}")
    return value


def run(cmd, cwd=None, timeout=1800):
    print(f"  $ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=cwd, timeout=timeout)
    if r.returncode != 0:
        sys.exit(f"Command failed ({r.returncode}): {' '.join(cmd)}")


def bucket_for(basename: str, region: str) -> str:
    return f"{basename}-{region}"


def template_url(basename: str, prefix: str, region: str) -> str:
    return f"https://{bucket_for(basename, region)}.s3.{region}.amazonaws.com/{prefix}/{TEMPLATE_NAME}"


def launch_url(basename: str, prefix: str, region: str, stack_name: str) -> str:
    query = urllib.parse.urlencode({"templateURL": template_url(basename, prefix, region), "stackName": stack_name})
    return (f"https://{region}.console.aws.amazon.com/cloudformation/home?region={region}"
            f"#/stacks/create/review?{query}")


def plan_uploads(cdk_out: Path, basename: str, prefix: str, region: str) -> list:
    """(local path, packaging, bucket, key) for every file asset, plus the template itself.

    The asset manifest names each destination as `<basename>-${AWS::Region}`
    under `<prefix>/assets/`; anything else means the synth and this run
    disagree about where the assets live, and the template would point at
    objects that were never uploaded.
    """
    manifest = json.loads((cdk_out / f"{STACK}.assets.json").read_text())
    bucket = bucket_for(basename, region)
    uploads = []
    for asset_id, asset in sorted(manifest.get("files", {}).items()):
        source = asset["source"]
        for dest in asset["destinations"].values():
            dest_bucket = dest["bucketName"].replace("${AWS::Region}", region)
            key = dest["objectKey"]
            if dest_bucket != bucket or not key.startswith(f"{prefix}/assets/"):
                sys.exit(f"Asset {asset_id} is bound for s3://{dest_bucket}/{key}, not s3://{bucket}/{prefix}/assets/")
            uploads.append((cdk_out / source["path"], source.get("packaging", "file"), bucket, key))
    if manifest.get("dockerImages"):
        sys.exit("The launch template must not use Docker image assets: the launching account has no CDK bootstrap.")
    uploads.append((cdk_out / f"{STACK}.template.json", "file", bucket, f"{prefix}/{TEMPLATE_NAME}"))
    return uploads


def zip_dir(src: Path, dest: Path) -> Path:
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(p for p in src.rglob("*") if p.is_file()):
            info = zipfile.ZipInfo(path.relative_to(src).as_posix(), date_time=_ZIP_DATE)
            info.external_attr = (path.stat().st_mode & 0o777) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, path.read_bytes())
    return dest


def stage_image_source(dest: Path) -> None:
    """Copy inference/ the way build-image.py zips it: no caches or bytecode."""
    shutil.copytree(REPO_ROOT / "inference", dest / "inference",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"))


def build_ui(dest: Path, skip_build: bool) -> None:
    if not skip_build:
        if not (FRONTEND_DIR / "node_modules").exists():
            run(["npm", "ci"], cwd=FRONTEND_DIR)
        run(["npm", "run", "build"], cwd=FRONTEND_DIR)
    dist = FRONTEND_DIR / "dist"
    if not (dist / "index.html").exists():
        sys.exit("frontend/dist has no index.html; build the UI first (omit --skip-ui-build).")
    # A config.js copied from public/ (written by ./deploy.sh ui) belongs to
    # someone's local endpoint; the Launch Stack writes its own at deploy time.
    shutil.copytree(dist, dest, ignore=shutil.ignore_patterns("config.js"))


def synth(work: Path, basename: str, prefix: str) -> Path:
    if not (CDK_DIR / "node_modules").exists():
        run(["npm", "ci"], cwd=CDK_DIR)
    catalogue = work / "catalogue.json"
    with catalogue.open("w") as f:
        r = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "launch_buildspecs.py")], stdout=f, timeout=120)
    if r.returncode != 0:
        sys.exit("scripts/launch_buildspecs.py failed")
    out = work / "cdk.out"
    run(["npx", "cdk", "synth", "--quiet",
         "-a", "npx ts-node --prefer-ts-exts bin/launch.ts",
         "-o", str(out),
         "-c", f"publishBucket={basename}",
         "-c", f"publishPrefix={prefix}",
         "-c", f"launchBuildspecs={catalogue}",
         "-c", f"imageSource={work / 'image-source'}",
         "-c", f"uiDist={work / 'ui'}",
         STACK], cwd=CDK_DIR)
    return out


def upload(uploads: list, work: Path, dry_run: bool) -> None:
    zips = work / "zips"
    zips.mkdir(exist_ok=True)
    for path, packaging, bucket, key in uploads:
        if packaging == "zip" and path.is_dir():
            path = zip_dir(path, zips / Path(key).name)
        elif packaging not in ("file", "zip"):
            sys.exit(f"Unsupported asset packaging {packaging!r} for {key}")
        print(f"  {'would upload' if dry_run else 'upload'} {path.name} -> s3://{bucket}/{key}")
        if not dry_run:
            run(["aws", "s3", "cp", str(path), f"s3://{bucket}/{key}", "--only-show-errors"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bucket-basename", required=True,
                        help="Bucket name without the Region; s3://<basename>-<region> must exist")
    parser.add_argument("--prefix", required=True, help="Key prefix, e.g. world-model-inference/v1")
    parser.add_argument("--region", action="append", required=True, help="Region to publish to (repeatable)")
    parser.add_argument("--stack-name", default="WorldModel", help="Default stack name in the Launch Stack link")
    parser.add_argument("--skip-ui-build", action="store_true", help="Use the existing frontend/dist")
    parser.add_argument("--dry-run", action="store_true", help="Synthesise and list uploads; make no AWS calls")
    parser.add_argument("--workdir", help="Keep build outputs here instead of a temporary directory")
    args = parser.parse_args()

    basename = checked(args.bucket_basename, _BASENAME_RE, "--bucket-basename")
    prefix = checked(args.prefix.strip("/"), _PREFIX_RE, "--prefix")
    regions = [checked(r, _REGION_RE, "--region") for r in args.region]
    stack_name = checked(args.stack_name, _STACK_NAME_RE, "--stack-name")

    if not args.dry_run:
        for region in regions:
            bucket = bucket_for(basename, region)
            r = subprocess.run(["aws", "s3api", "head-bucket", "--bucket", bucket, "--region", region],
                               capture_output=True, text=True, timeout=60)
            if r.returncode != 0:
                sys.exit(f"s3://{bucket} is missing or not yours ({r.stderr.strip()}). "
                         f"Create it in {region} first; this script does not create buckets.")

    work = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="wm-publish-"))
    work.mkdir(parents=True, exist_ok=True)
    for sub in ("image-source", "ui", "cdk.out", "zips"):
        shutil.rmtree(work / sub, ignore_errors=True)
    print(f"Working in {work}")

    stage_image_source(work / "image-source")
    build_ui(work / "ui", args.skip_ui_build)
    cdk_out = synth(work, basename, prefix)

    for region in regions:
        print(f"\nPublishing to {region}:")
        upload(plan_uploads(cdk_out, basename, prefix, region), work, args.dry_run)

    print("\nLaunch Stack links:")
    for region in regions:
        print(f"  {region}: {launch_url(basename, prefix, region, stack_name)}")
    print("\nREADME table rows:")
    for region in regions:
        print(f"| {region} | [![Launch Stack](https://s3.amazonaws.com/cloudformation-examples/cloudformation-launch-stack.png)]"
              f"({launch_url(basename, prefix, region, stack_name)}) |")
    if not args.workdir:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    os.environ.pop("CDK_DEFAULT_ACCOUNT", None)  # the template is account-agnostic
    main()
