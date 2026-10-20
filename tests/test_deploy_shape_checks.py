# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""deploy.sh validates every SSM value before it reaches a shell, a URL or CDK.

The foundation parameters under /world-model/foundation/* end up in the root boot
script, the client-credentials POST and the token verifier (EDSR WM-004), so
check_shape must reject anything outside each value's legitimate shape. The
helpers are extracted from deploy.sh and run in bash on their own: sourcing the
whole script would run its main dispatch, which deploys. No AWS calls.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DEPLOY_SH = REPO_ROOT / "deploy.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _helpers() -> str:
    """The regex constants and check_shape, exactly as deploy.sh defines them."""
    text = DEPLOY_SH.read_text()
    m = re.search(r"^POOL_RE=.*?^check_shape\(\) \{.*?^\}\n", text, re.S | re.M)
    assert m, "check_shape block not found in deploy.sh"
    return m.group(0)


def _check(value: str, regex_var: str) -> int:
    script = _helpers() + f'check_shape test-param "$1" "${regex_var}"\n'
    return subprocess.run(
        ["bash", "-c", script, "bash", value], capture_output=True, text=True
    ).returncode


INJECTIONS = ['"', ";", "$(id)", "`id`", "\n", "x y", "'"]

GOOD = {
    "POOL_RE": "us-east-1_AbC123xyz",
    "CLIENT_RE": "4abc123def456ghi789jkl0mno",
    "CLIENTS_RE": "4abc123def456,7xyz890uvw",
    "SCOPE_RE": "world-model/invoke",
    "DOMAIN_RE": "https://world-model-example-us-east-1.auth.us-east-1.amazoncognito.com",
}


@pytest.mark.parametrize("regex_var,value", sorted(GOOD.items()))
def test_legitimate_values_pass(regex_var, value):
    assert _check(value, regex_var) == 0


@pytest.mark.parametrize("regex_var", sorted(GOOD))
@pytest.mark.parametrize("bad", INJECTIONS)
def test_shell_metacharacters_rejected(regex_var, bad):
    for value in (GOOD[regex_var] + bad, bad + GOOD[regex_var]):
        assert _check(value, regex_var) == 1, repr(value)


@pytest.mark.parametrize("regex_var", sorted(GOOD))
def test_empty_rejected(regex_var):
    assert _check("", regex_var) == 1


@pytest.mark.parametrize("domain", [
    "https://evil.example",
    "http://world-model.auth.us-east-1.amazoncognito.com",
    "https://world-model.auth.us-east-1.amazoncognito.com.evil.example",
    "https://world-model.auth.us-east-1.amazoncognito.com/path",
])
def test_domain_must_be_a_cognito_hosted_ui(domain):
    """The domain receives the machine client's permanent secret."""
    assert _check(domain, "DOMAIN_RE") == 1
