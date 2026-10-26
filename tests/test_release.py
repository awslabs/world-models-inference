# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""scripts/release.py: VERSION / pyproject.toml / CHANGELOG.md bookkeeping."""

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("release", REPO_ROOT / "scripts" / "release.py")
rel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rel)

URL = rel.REPO_URL

CHANGELOG = f"""# Changelog

Intro text.

## [Unreleased]

### Fixed

- A fix.

## [0.1.0] - 2026-09-30

### Added

- The first thing.

[Unreleased]: {URL}/compare/v0.1.0...HEAD
[0.1.0]: {URL}/releases/tag/v0.1.0
"""

PYPROJECT = '[project]\nname = "x"\nversion = "0.1.0"\n\n[tool.uv]\npackage = false\n'


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "VERSION").write_text("0.1.0\n")
    (tmp_path / "pyproject.toml").write_text(PYPROJECT)
    (tmp_path / "CHANGELOG.md").write_text(CHANGELOG)
    return tmp_path


# --- the real repository ------------------------------------------------------

def test_repository_release_metadata_is_consistent():
    assert rel.check(REPO_ROOT) == []


def test_repository_current_version_has_release_notes():
    version = (REPO_ROOT / "VERSION").read_text().strip()
    notes = rel.section_body((REPO_ROOT / "CHANGELOG.md").read_text(), version)
    assert notes.strip(), f"CHANGELOG.md section for {version} is empty"


# --- versions -----------------------------------------------------------------

@pytest.mark.parametrize("current, part, expected", [
    ("0.1.0", "patch", "0.1.1"),
    ("0.1.9", "minor", "0.2.0"),
    ("0.9.3", "major", "1.0.0"),
    ("0.1.0", "1.0.0-rc.1", "1.0.0-rc.1"),
    ("1.0.0-rc.1", "1.0.0", "1.0.0"),
])
def test_next_version(current, part, expected):
    assert rel.next_version(current, part) == expected


@pytest.mark.parametrize("current, part", [
    ("0.2.0", "0.1.0"),       # backwards
    ("0.2.0", "0.2.0"),       # same
    ("1.0.0", "1.0.0-rc.1"),  # pre-release of an existing release
    ("0.1.0", "v0.2.0"),      # tag, not a version
    ("0.1.0", "minr"),
])
def test_next_version_rejects(current, part):
    with pytest.raises(rel.ReleaseError):
        rel.next_version(current, part)


@pytest.mark.parametrize("bad", ["1.0", "01.0.0", "1.0.0+build", "v1.0.0", ""])
def test_parse_semver_rejects(bad):
    with pytest.raises(rel.ReleaseError):
        rel.parse_semver(bad)


# --- check --------------------------------------------------------------------

def test_check_passes_and_validates_tag(repo):
    assert rel.check(repo) == []
    assert rel.check(repo, tag="v0.1.0") == []
    assert any("does not match" in p for p in rel.check(repo, tag="v0.2.0"))


def test_check_flags_pyproject_drift(repo):
    (repo / "pyproject.toml").write_text(PYPROJECT.replace("0.1.0", "0.0.9"))
    assert any("pyproject.toml" in p for p in rel.check(repo))


def test_check_flags_missing_section_for_version(repo):
    (repo / "VERSION").write_text("0.2.0\n")
    (repo / "pyproject.toml").write_text(PYPROJECT.replace("0.1.0", "0.2.0"))
    assert any("no section for VERSION 0.2.0" in p for p in rel.check(repo))


def test_check_flags_missing_unreleased(repo):
    (repo / "CHANGELOG.md").write_text(CHANGELOG.replace("## [Unreleased]\n\n### Fixed\n\n- A fix.\n\n", ""))
    assert any("Unreleased" in p for p in rel.check(repo))


def test_check_flags_undated_and_misordered_sections(repo):
    text = CHANGELOG.replace("## [0.1.0] - 2026-09-30", "## [0.1.0]")
    text = text.replace(f"\n[Unreleased]: {URL}", f"\n## [0.2.0] - 2026-10-01\n\n- later\n\n[Unreleased]: {URL}", 1)
    (repo / "CHANGELOG.md").write_text(text)
    problems = rel.check(repo)
    assert any("[0.1.0] has no date" in p for p in problems)
    assert any("newest-first" in p for p in problems)


# --- bump ---------------------------------------------------------------------

def test_bump_moves_unreleased_and_updates_everything(repo):
    assert rel.bump(repo, "minor", "2026-10-08") == ("0.1.0", "0.2.0")

    assert (repo / "VERSION").read_text() == "0.2.0\n"
    assert 'version = "0.2.0"' in (repo / "pyproject.toml").read_text()
    changelog = (repo / "CHANGELOG.md").read_text()
    assert rel.section_body(changelog, "Unreleased").strip() == ""
    assert rel.section_body(changelog, "0.2.0") == "### Fixed\n\n- A fix.\n"
    assert rel.section_body(changelog, "0.1.0") == "### Added\n\n- The first thing.\n"
    assert f"[Unreleased]: {URL}/compare/v0.2.0...HEAD\n" in changelog
    assert f"[0.2.0]: {URL}/compare/v0.1.0...v0.2.0\n" in changelog
    assert f"[0.1.0]: {URL}/releases/tag/v0.1.0\n" in changelog
    assert rel.check(repo, tag="v0.2.0") == []


def test_bump_twice_keeps_history(repo):
    rel.bump(repo, "patch", "2026-10-08")
    changelog = (repo / "CHANGELOG.md").read_text()
    (repo / "CHANGELOG.md").write_text(changelog.replace("## [Unreleased]\n", "## [Unreleased]\n\n- Another.\n", 1))
    rel.bump(repo, "patch", "2026-10-09")
    changelog = (repo / "CHANGELOG.md").read_text()
    assert rel.released_versions(changelog) == ["0.1.2", "0.1.1", "0.1.0"]
    assert f"[0.1.2]: {URL}/compare/v0.1.1...v0.1.2\n" in changelog
    assert rel.check(repo) == []


def test_bump_refuses_empty_unreleased(repo):
    (repo / "CHANGELOG.md").write_text(CHANGELOG.replace("### Fixed\n\n- A fix.\n", "### Fixed\n"))
    with pytest.raises(rel.ReleaseError, match="no entries"):
        rel.bump(repo, "patch", "2026-10-08")
    assert (repo / "VERSION").read_text() == "0.1.0\n"


def test_bump_refuses_placeholder(repo):
    (repo / "CHANGELOG.md").write_text(CHANGELOG.replace("- A fix.", f"- {rel.PLACEHOLDER}: write me."))
    with pytest.raises(rel.ReleaseError, match="placeholder"):
        rel.bump(repo, "patch", "2026-10-08")


def test_bump_adds_links_when_missing(repo):
    (repo / "CHANGELOG.md").write_text(CHANGELOG.split("\n[Unreleased]:")[0] + "\n")
    rel.bump(repo, "patch", "2026-10-08")
    changelog = (repo / "CHANGELOG.md").read_text()
    assert changelog.endswith(f"[Unreleased]: {URL}/compare/v0.1.1...HEAD\n[0.1.1]: {URL}/compare/v0.1.0...v0.1.1\n")


# --- CLI ----------------------------------------------------------------------

def test_cli_notes_accepts_tag_form(repo, capsys):
    assert rel.main(["--root", str(repo), "notes", "v0.1.0"]) == 0
    assert capsys.readouterr().out == "### Added\n\n- The first thing.\n"


def test_cli_check_exit_codes(repo, capsys):
    assert rel.main(["--root", str(repo), "check"]) == 0
    assert rel.main(["--root", str(repo), "check", "--tag", "v9.9.9"]) == 1


def test_cli_bump_dry_run_writes_nothing(repo, capsys):
    before = {p.name: p.read_text() for p in repo.iterdir()}
    assert rel.main(["--root", str(repo), "bump", "major", "--dry-run"]) == 0
    assert capsys.readouterr().out == "1.0.0\n"
    assert {p.name: p.read_text() for p in repo.iterdir()} == before


def test_cli_bump_rejects_bad_date(repo):
    assert rel.main(["--root", str(repo), "bump", "patch", "--date", "08/10/2026"]) == 1
    assert (repo / "VERSION").read_text() == "0.1.0\n"
