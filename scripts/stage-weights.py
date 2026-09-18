#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage model weights to S3 via CodeBuild.

Triggers a build that downloads from HuggingFace Hub or torch.hub
and syncs to S3. The CodeBuild project is managed by CDK (shared stack).

Usage:
  python scripts/stage-weights.py vjepa2-ac --source torch_hub:facebookresearch/vjepa2:vjepa2_ac_vit_giant
  python scripts/stage-weights.py lingbot-fast --source hf:robbyant/lingbot-world-base-cam
  python scripts/stage-weights.py lingbot-fast --source hf:robbyant/lingbot-world-fast --subdir lingbot_world_fast
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "inference" / "models"


def run(cmd, timeout=30):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def ssm_get(name, region):
    r = run(["aws", "ssm", "get-parameter", "--name", name,
             "--query", "Parameter.Value", "--output", "text", "--region", region])
    if r.returncode != 0:
        sys.exit(f"SSM param '{name}' not found. Deploy foundation stack first.")
    return r.stdout.strip()


def get_s3_dest(endpoint_id: str, region: str) -> str:
    """Derive S3 destination from foundation bucket + model name in endpoint.yaml."""
    bucket = ssm_get("/world-model/foundation/artifacts-bucket", region)
    # Use the model_data path suffix (after the bucket) from endpoint.yaml
    manifest = MODELS_DIR / endpoint_id / "endpoint.yaml"
    if not manifest.exists():
        sys.exit(f"No endpoint.yaml for {endpoint_id}")
    m = re.search(r"^model_data:\s*s3://[^/]+/(.+)$", manifest.read_text(), re.MULTILINE)
    prefix = m.group(1).strip() if m else f"{endpoint_id}/"
    return f"s3://{bucket}/{prefix}"


# These /tmp paths are literals inside shell commands that run on a remote
# Linux CodeBuild/EC2 box (not this machine), so tempfile.gettempdir() would
# resolve the wrong host. The build environment is single-use and isolated.
_REMOTE_TMP = "/tmp/weights"  # nosec B108 - remote build-box path, not local


def build_commands(source: str, subdir: str, s3_dest: str, region: str) -> list:
    if source.startswith("hf:"):
        repo = source[3:]
        dest_dir = f"{_REMOTE_TMP}/{subdir}" if subdir else _REMOTE_TMP
        install = ["pip3 install -q huggingface_hub"]
        # Authenticate before download so gated repos (e.g. nvidia/Lyra-2.0) work.
        # The modern `hf` CLI does not auto-read HF_TOKEN for gated access, so log
        # in explicitly when a token is provided (no-op token is harmless for public repos).
        # `hf download --local-dir` keeps a full copy in the HF cache AND in
        # --local-dir, so a 60 GB repo needs ~120 GB and fills the CodeBuild
        # volume ("No space left on device"). Sync each file to S3 as we go and
        # drop the cache, so peak disk is one copy rather than two.
        download = [
            'if [ -n "$HF_TOKEN" ]; then hf auth login --token "$HF_TOKEN" || huggingface-cli login --token "$HF_TOKEN"; fi',
            f"mkdir -p {dest_dir}",
            # Download into the cache only (no second --local-dir copy), then move
            # files out to dest_dir so the cache can be pruned as we go.
            f"export HF_HUB_ENABLE_HF_TRANSFER=0 HF_HOME={_REMOTE_TMP}/.hfcache",
            f"hf download {repo} --local-dir {dest_dir}",
            # Remove the duplicate cache blobs now that dest_dir holds real files.
            f"rm -rf {_REMOTE_TMP}/.hfcache {dest_dir}/.cache || true",
            "df -h /tmp | tail -1",
        ]
    elif source.startswith("torch_hub:"):
        parts = source[10:].split(":", 1)
        if len(parts) != 2:
            sys.exit("Format: torch_hub:<repo>:<entrypoint>")
        repo, entry = parts
        install = ["pip3 install -q torch --index-url https://download.pytorch.org/whl/cpu"]
        download = [
            f"mkdir -p {_REMOTE_TMP}",
            f"python3 -c \""
            f"import torch,os,shutil; os.environ['TORCH_HOME']='/tmp/hub'; "
            f"torch.hub.load('{repo}','{entry}',trust_repo=True); "
            f"[shutil.move(os.path.join('/tmp/hub/hub/checkpoints',f),_REMOTE_TMP+'/'+f) "
            f"for f in os.listdir('/tmp/hub/hub/checkpoints')]\"",
        ]
    else:
        sys.exit("Use hf:<repo> or torch_hub:<repo>:<entry>")

    upload = [f"aws s3 sync {_REMOTE_TMP}/ {s3_dest} --region {region}", "echo '=== DONE ==='"]
    return install + download + upload


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="Endpoint ID (directory under inference/models/)")
    parser.add_argument("--source", required=True, help="hf:<repo> or torch_hub:<repo>:<entrypoint>")
    parser.add_argument("--subdir", default="", help="Subdirectory within S3 prefix")
    parser.add_argument("--region", default=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    args = parser.parse_args()

    account = run(["aws", "sts", "get-caller-identity", "--query", "Account", "--output", "text"]).stdout.strip()
    if not account:
        sys.exit("AWS CLI not configured")

    project_name = ssm_get("/world-model/foundation/weights-project", args.region)
    s3_dest = get_s3_dest(args.model, args.region)
    if args.subdir:
        s3_dest = s3_dest.rstrip("/") + "/" + args.subdir + "/"

    commands = build_commands(args.source, args.subdir, s3_dest, args.region)
    buildspec = json.dumps({
        "version": "0.2",
        "phases": {"build": {"commands": commands}},
    })

    hf_token = os.environ.get("HF_TOKEN", "")
    env_overrides = []
    if hf_token:
        env_overrides.append({"name": "HF_TOKEN", "value": hf_token, "type": "PLAINTEXT"})

    print(f"  Model:   {args.model}")
    print(f"  Source:  {args.source}")
    print(f"  Target:  {s3_dest}")
    print()

    start_args = [
        "aws", "codebuild", "start-build",
        "--project-name", project_name,
        "--buildspec-override", buildspec,
        "--region", args.region,
        "--query", "build.id", "--output", "text",
    ]
    if env_overrides:
        start_args += ["--environment-variables-override", json.dumps(env_overrides)]

    r = run(start_args)
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
            print(f"\n\n  Weights staged at {s3_dest}")
            return
        if status in ("FAILED", "FAULT", "STOPPED", "TIMED_OUT"):
            print(f"\n\n  Build {status}. Check logs:")
            print(f"  aws codebuild batch-get-builds --ids {build_id} --query 'builds[0].logs.deepLink' --output text --region {args.region}")
            sys.exit(1)


if __name__ == "__main__":
    main()
