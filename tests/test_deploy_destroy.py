# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`./deploy.sh destroy` must only report "nothing to destroy" when the stack is gone.

With expired credentials every aws call failed,
and destroy printed "does not exist ... nothing to destroy" and exited 0 while the
GPU kept running. cmd_destroy is extracted and run against a fake `aws` on PATH
(sourcing deploy.sh would run its dispatch). No AWS calls.
"""

import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DEPLOY_SH = REPO_ROOT / "deploy.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _cmd_destroy() -> str:
    m = re.search(r"^cmd_destroy\(\) \{.*?^\}\n", DEPLOY_SH.read_text(), re.S | re.M)
    assert m, "cmd_destroy not found in deploy.sh"
    return m.group(0)


def _run(tmp_path, describe_stderr: str, describe_rc: int):
    log = tmp_path / "calls.log"
    fake = tmp_path / "aws"
    fake.write_text(f"""#!/bin/bash
echo "$*" >> {log}
if [ "$2" = describe-stacks ]; then echo {describe_stderr!r} >&2; exit {describe_rc}; fi
exit 0
""")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    script = "get_region() { echo us-east-1; }\n" + _cmd_destroy() + "cmd_destroy praxis-1\n"
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                       env={"PATH": f"{tmp_path}:/usr/bin:/bin"})
    return r, (log.read_text() if log.exists() else "")


def test_missing_stack_is_nothing_to_destroy(tmp_path):
    r, calls = _run(tmp_path, "An error occurred (ValidationError): Stack with id WorldModel-praxis-1 does not exist", 254)
    assert r.returncode == 0 and "nothing to destroy" in r.stdout
    assert "delete-stack" not in calls


def test_credential_failure_is_an_error_not_success(tmp_path):
    r, calls = _run(tmp_path, "An error occurred (ExpiredToken) when calling the DescribeStacks operation: The security token included in the request is expired", 255)
    assert r.returncode == 1
    assert "nothing to destroy" not in r.stdout
    assert "nothing was destroyed" in r.stderr and "ExpiredToken" in r.stderr
    assert "delete-stack" not in calls


def test_existing_stack_is_deleted(tmp_path):
    r, calls = _run(tmp_path, "", 0)
    assert r.returncode == 0 and "destroyed" in r.stdout
    assert "delete-stack" in calls and "stack-delete-complete" in calls
