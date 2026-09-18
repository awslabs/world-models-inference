# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the AWS solution tracking code in the CDK stack descriptions.

The tracking code only registers a deployment if it survives in the synthesised
CloudFormation `Description`, and nothing else in the build would fail if a
refactor dropped it — so assert it here.

A real `cdk synth` needs credentialed VPC/AMI/SSM lookups, which CI does not
have, so these tests read the TypeScript source instead. That is enough to catch
the regression that matters: a stack losing its description, or a new stack
being added without one.
"""

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CDK_LIB = REPO_ROOT / "cdk" / "lib"

# Every stack class in cdk/lib, and the dashboard tag it reports under. All of
# these are reachable from bin/app.ts — see test_every_stack_class_is_reachable,
# which fails if a stack is added to cdk/lib without being wired up.
REACHABLE_STACKS = {
    "shared.ts": "foundation",
    # ModelStack tags by target (ec2 | sagemaker) rather than by model name, so
    # the dashboard entry count stays bounded as models are added.
    "model.ts": None,
}


def _read(name: str) -> str:
    return (CDK_LIB / name).read_text()


# --------------------------------------------------------------------------
# The tracking code itself
# --------------------------------------------------------------------------


def test_solution_id_is_declared():
    """The tracking code must be present and match the uksb-<id> format."""
    src = _read("solution.ts")
    m = re.search(r"""export const SOLUTION_ID = ['"]([^'"]+)['"]""", src)
    assert m, "solution.ts must export SOLUTION_ID"
    assert re.fullmatch(r"uksb-[a-z0-9]+", m.group(1)), (
        f"tracking code {m.group(1)!r} is not in uksb-<id> form"
    )


def test_describe_stack_emits_tracking_code_and_tag():
    """describeStack() must render `<summary> (<id>)(tag:<tag>).`

    The dashboard parses this shape, so the parentheses and trailing period are
    load-bearing, not cosmetic.
    """
    src = _read("solution.ts")
    # Match the returned template literal directly. Matching the function body
    # with a lazy {...} would stop at the first `${...}` interpolation.
    ret = re.search(
        r"export function describeStack\([^)]*\):\s*string\s*\{\s*return\s+`([^`]*)`",
        src,
        re.S,
    )
    assert ret, "describeStack() must return a template literal"
    tpl = ret.group(1)
    assert "${SOLUTION_ID}" in tpl, "description must embed SOLUTION_ID"
    assert "(tag:${tag})" in tpl, "description must embed the per-stack tag"
    assert tpl.rstrip().endswith("."), "the dashboard expects a trailing period"


# --------------------------------------------------------------------------
# Every reachable stack carries a description
# --------------------------------------------------------------------------


@pytest.mark.parametrize("stack_file", sorted(REACHABLE_STACKS))
def test_stack_passes_description_to_super(stack_file):
    """Each deployable stack must call describeStack() in its super() props."""
    src = _read(stack_file)
    assert "describeStack" in src, f"{stack_file} does not import describeStack"
    assert re.search(r"description:\s*describeStack\(", src), (
        f"{stack_file} must pass description: describeStack(...) to super()"
    )


def test_shared_stack_uses_foundation_tag():
    assert "'foundation'," in _read("shared.ts")


def test_model_stack_tags_by_target_not_model_name():
    """Tagging by target keeps the dashboard entry count bounded.

    Tagging by endpointId would create a new entry for every model ever
    deployed from this repo.
    """
    src = _read("model.ts")
    m = re.search(r"description:\s*describeStack\(\s*(.+?),?\s*\),", src, re.S)
    assert m, "model.ts must pass description: describeStack(...) to super()"
    args = m.group(1)

    # describeStack(summary, tag) — the tag is the LAST argument, and it is the
    # value that becomes a dashboard entry. Split on the top-level comma so we
    # inspect the tag specifically rather than the whole call (the summary
    # legitimately mentions endpointId).
    tag_arg = args.rsplit(",", 1)[-1].strip()
    assert "props.target" in tag_arg, (
        f"ModelStack must tag by target; tag argument is {tag_arg!r}"
    )
    assert "endpointId" not in tag_arg, (
        f"ModelStack must not tag by model name — that unbounds the dashboard "
        f"entry count. Tag argument is {tag_arg!r}"
    )


# --------------------------------------------------------------------------
# No orphaned stacks
# --------------------------------------------------------------------------


def test_every_stack_class_is_reachable():
    """Every stack class in cdk/lib must be instantiated in bin/app.ts.

    A stack that is defined but never instantiated cannot be synthesised or
    deployed, so any tracking tag it carries never reports — and `cdk deploy
    <name>` fails with "No stacks match the name(s)". This repo previously
    carried such a stack (`static-site.ts`, removed in this commit) for months
    without anyone noticing.

    If this fails you have either added a stack without wiring it up, or removed
    one without updating REACHABLE_STACKS.
    """
    app = (REPO_ROOT / "cdk" / "bin" / "app.ts").read_text()

    defined = {}
    for ts in sorted(CDK_LIB.glob("*.ts")):
        for cls in re.findall(r"export class (\w+) extends cdk\.Stack", ts.read_text()):
            defined[cls] = ts.name

    assert defined, "found no stack classes in cdk/lib — has the layout changed?"

    orphaned = {
        cls: f for cls, f in defined.items() if not re.search(rf"new {cls}\s*\(", app)
    }
    assert not orphaned, (
        "stack classes defined but never instantiated in bin/app.ts: "
        + ", ".join(f"{c} ({f})" for c, f in sorted(orphaned.items()))
    )

    # And the set of stack-defining files is exactly what these tests cover.
    assert set(defined.values()) == set(REACHABLE_STACKS), (
        f"stack files {sorted(set(defined.values()))} do not match "
        f"REACHABLE_STACKS {sorted(REACHABLE_STACKS)} — update this test module"
    )


def test_cdk_package_scripts_reference_real_stacks():
    """`npm run` scripts must not name a stack that cannot be synthesised."""
    pkg = json.loads((REPO_ROOT / "cdk" / "package.json").read_text())
    app = (REPO_ROOT / "cdk" / "bin" / "app.ts").read_text()

    # Stack ids this app can actually produce. WorldModel-<name> is dynamic, so
    # match on the literal prefix in bin/app.ts rather than a full id.
    known = set(re.findall(r"new \w+\(app,\s*['\"`]([^'\"`]+)", app))
    known |= {m for m in re.findall(r"new \w+\(app,\s*`([^`]+)`", app)}

    for name, cmd in pkg.get("scripts", {}).items():
        m = re.search(r"cdk (?:deploy|destroy)\s+(\S+)", cmd)
        if not m:
            continue
        target = m.group(1)
        assert any(target == k or k.startswith(target) for k in known), (
            f"cdk/package.json script {name!r} targets stack {target!r}, which "
            f"bin/app.ts never creates (it creates: {sorted(known)})"
        )
