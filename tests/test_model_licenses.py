# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every cartridge must declare its weights licence, and the register must agree.

This repo is Apache-2.0; the models it deploys are not. The failure mode this guards
against is silent: a cartridge lands without licence metadata, ``deploy.sh``'s
``check_license`` gate returns early because ``license_restricted`` is absent, and a
non-commercial or research-only model deploys with no notice to the operator. The gate
is opt-in by construction, so the only way to keep it honest is to require the fields.

It also pins the register itself. ``docs/MODEL_LICENSES.md`` is published in a public
repo, so a row that disagrees with the manifest is a licence claim we cannot support.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "inference" / "models"
REGISTER = REPO_ROOT / "docs" / "MODEL_LICENSES.md"

# The scaffold ships deliberate CHANGE-ME placeholders so a new cartridge is prompted
# for its licence; it is never deployed, so it is not held to the same bar.
TEMPLATE = "_template"

REQUIRED_FIELDS = ("license", "license_url", "license_restricted")


def _cartridges() -> list[str]:
    return sorted(
        p.name
        for p in MODELS_DIR.iterdir()
        if p.is_dir() and (p / "endpoint.yaml").is_file() and p.name != TEMPLATE
    )


def _manifest_field(model: str, field: str) -> str | None:
    """Read a top-level scalar, the same way deploy.sh's get_manifest_field does."""
    text = (MODELS_DIR / model / "endpoint.yaml").read_text()
    match = re.search(rf"^{field}:(.*)$", text, re.MULTILINE)
    if match is None:
        return None
    # Strip a YAML inline comment (whitespace then #), not a URL fragment.
    return re.sub(r"\s+#.*$", "", match.group(1)).strip()


def test_there_are_cartridges_to_check() -> None:
    """Guard against the discovery glob silently matching nothing."""
    assert _cartridges(), f"no cartridges discovered under {MODELS_DIR}"


@pytest.mark.parametrize("model", _cartridges())
@pytest.mark.parametrize("field", REQUIRED_FIELDS)
def test_cartridge_declares_licence_field(model: str, field: str) -> None:
    value = _manifest_field(model, field)
    assert value, (
        f"{model}/endpoint.yaml is missing `{field}`. Every cartridge must declare "
        f"its weights licence — see docs/MODEL_LICENSES.md for how to find the real "
        f"one, and note that a Hugging Face `license: other` tag is not a licence."
    )


@pytest.mark.parametrize("model", _cartridges())
def test_licence_is_not_a_placeholder(model: str) -> None:
    value = _manifest_field(model, "license")
    assert value and "CHANGE-ME" not in value, (
        f"{model} still carries the template's CHANGE-ME licence placeholder."
    )


@pytest.mark.parametrize("model", _cartridges())
def test_license_restricted_is_a_bool(model: str) -> None:
    value = _manifest_field(model, "license_restricted")
    assert value in {"true", "false"}, (
        f"{model} has license_restricted={value!r}; deploy.sh compares it to the "
        f'string "true", so anything else silently disables the licence gate.'
    )


@pytest.mark.parametrize("model", _cartridges())
def test_cartridge_appears_in_the_register(model: str) -> None:
    register = REGISTER.read_text()
    assert f"`{model}`" in register, (
        f"{model} is not listed in docs/MODEL_LICENSES.md. The register is published "
        f"in a public repo and is the only place the three licence layers (platform "
        f"code / upstream code / weights) are written down."
    )


@pytest.mark.parametrize("model", _cartridges())
def test_register_states_the_declared_licence(model: str) -> None:
    """The register's row must name the licence the manifest declares.

    Catches the drift that actually happened: the catalogue UI published
    `Apache-2.0` for cosmos3-nano while Hugging Face reported OpenMDW-1.1.
    """
    declared = _manifest_field(model, "license")
    assert declared
    register = REGISTER.read_text()
    row = next(
        (ln for ln in register.splitlines() if ln.startswith("|") and f"`{model}`" in ln),
        None,
    )
    assert row is not None, f"no table row for {model} in docs/MODEL_LICENSES.md"
    assert declared.lower() in row.lower(), (
        f"docs/MODEL_LICENSES.md row for {model} does not mention its declared "
        f"licence {declared!r}:\n  {row}"
    )


def test_restricted_models_are_flagged_as_restricting_you() -> None:
    """A gated model must be visibly gated in the register, not just in YAML."""
    for model in _cartridges():
        if _manifest_field(model, "license_restricted") != "true":
            continue
        row = next(
            ln
            for ln in REGISTER.read_text().splitlines()
            if ln.startswith("|") and f"`{model}`" in ln
        )
        assert "yes" in row.lower(), (
            f"{model} sets license_restricted: true but its register row does not say "
            f"it restricts the reader:\n  {row}"
        )
