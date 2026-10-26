# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""scripts/publish.py: where the Launch Stack's template and assets go. No AWS calls."""

import importlib.util
import json
import urllib.parse
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("publish", REPO_ROOT / "scripts" / "publish.py")
pub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pub)


def _manifest(tmp_path, bucket="acme-wm-${AWS::Region}", prefix="wm/v1"):
    (tmp_path / "asset.abc").mkdir()
    (tmp_path / "asset.abc" / "index.js").write_text("x")
    (tmp_path / f"{pub.STACK}.template.json").write_text("{}")
    (tmp_path / f"{pub.STACK}.assets.json").write_text(json.dumps({"files": {
        "abc": {"source": {"path": "asset.abc", "packaging": "zip"},
                "destinations": {"d": {"bucketName": bucket, "objectKey": f"{prefix}/assets/abc.zip"}}},
    }}))
    return tmp_path


def test_plan_uploads_targets_the_regional_bucket(tmp_path):
    plan = pub.plan_uploads(_manifest(tmp_path), "acme-wm", "wm/v1", "eu-west-1")
    assert [(p.name, k, b, key) for p, k, b, key in plan] == [
        ("asset.abc", "zip", "acme-wm-eu-west-1", "wm/v1/assets/abc.zip"),
        (f"{pub.STACK}.template.json", "file", "acme-wm-eu-west-1", f"wm/v1/{pub.TEMPLATE_NAME}"),
    ]


def test_plan_uploads_refuses_a_mismatched_destination(tmp_path):
    with pytest.raises(SystemExit, match="bound for"):
        pub.plan_uploads(_manifest(tmp_path, bucket="cdk-hnb659fds-assets-${AWS::AccountId}-${AWS::Region}"),
                         "acme-wm", "wm/v1", "eu-west-1")


def test_launch_url_is_a_console_create_link():
    url = pub.launch_url("acme-wm", "wm/v1", "us-west-2", "WorldModel")
    assert url.startswith("https://us-west-2.console.aws.amazon.com/cloudformation/home?region=us-west-2"
                          "#/stacks/create/review?")
    query = urllib.parse.parse_qs(url.split("?", 2)[2])
    assert query["templateURL"] == [
        f"https://acme-wm-us-west-2.s3.us-west-2.amazonaws.com/wm/v1/{pub.TEMPLATE_NAME}"]
    assert query["stackName"] == ["WorldModel"]


@pytest.mark.parametrize("value, pattern", [
    ("Acme", "_BASENAME_RE"),            # bucket names are lower case
    ("a" * 43, "_BASENAME_RE"),          # no room left for -<region>
    ("acme;id", "_BASENAME_RE"),
    ("wm/../x", "_PREFIX_RE"),
    ("/wm", "_PREFIX_RE"),
    ("us-east-1; id", "_REGION_RE"),
    ("1stack", "_STACK_NAME_RE"),
])
def test_arguments_are_shape_checked(value, pattern):
    with pytest.raises(SystemExit):
        pub.checked(value, getattr(pub, pattern), "value")


def test_zip_is_deterministic(tmp_path):
    src = tmp_path / "src"
    (src / "inference").mkdir(parents=True)
    (src / "inference" / "a.py").write_text("print(1)")
    first = pub.zip_dir(src, tmp_path / "1.zip").read_bytes()
    second = pub.zip_dir(src, tmp_path / "2.zip").read_bytes()
    assert first == second
    assert zipfile.ZipFile(tmp_path / "1.zip").namelist() == ["inference/a.py"]
