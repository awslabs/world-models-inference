#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Release helper: keep VERSION, pyproject.toml and CHANGELOG.md in step.

    python3 scripts/release.py check [--tag vX.Y.Z]
    python3 scripts/release.py bump  major|minor|patch|X.Y.Z [--date YYYY-MM-DD] [--dry-run]
    python3 scripts/release.py notes [X.Y.Z]

`check` is the CI gate: VERSION is valid SemVer, pyproject.toml agrees with it,
CHANGELOG.md has an [Unreleased] section and a section for VERSION, and (with
--tag) the tag is exactly v<VERSION>.

`bump` moves the [Unreleased] entries under a new dated heading, updates the
compare links, writes the new VERSION into VERSION and pyproject.toml, and
prints the commands to tag and publish. It never runs git itself.

`notes` prints one version's changelog section — the body of a release.

Standard library only, so it runs in a bare python image.
"""

import argparse
import datetime
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_URL = "https://github.com/awslabs/world-models-inference"

# SemVer 2.0 core with an optional pre-release (no build metadata: it cannot
# appear in a git tag that sorts sensibly, and we never need it).
SEMVER_RE = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$")
HEADING_RE = re.compile(r"^## \[([^\]]+)\](?: - (\d{4}-\d{2}-\d{2}))?[ \t]*$", re.MULTILINE)
LINK_RE = re.compile(r"^\[([^\]]+)\]: \S+[ \t]*$", re.MULTILINE)
PYPROJECT_VERSION_RE = re.compile(r'^version = "([^"]*)"$', re.MULTILINE)
# Left in an [Unreleased] entry that still needs writing; bump refuses to ship it.
PLACEHOLDER = "TODO(release)"


class ReleaseError(Exception):
    pass


def parse_semver(version):
    m = SEMVER_RE.match(version)
    if not m:
        raise ReleaseError(f"not a SemVer version: {version!r}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)


def semver_key(version):
    major, minor, patch, pre = parse_semver(version)
    # A pre-release sorts before its release. Identifiers compare as strings,
    # which is enough for the rc.N style we use.
    return (major, minor, patch, pre is None, pre or "")


def next_version(current, part):
    if SEMVER_RE.match(part):
        new = part
    else:
        major, minor, patch, _ = parse_semver(current)
        if part == "major":
            new = f"{major + 1}.0.0"
        elif part == "minor":
            new = f"{major}.{minor + 1}.0"
        elif part == "patch":
            new = f"{major}.{minor}.{patch + 1}"
        else:
            raise ReleaseError(f"expected major, minor, patch or X.Y.Z, got {part!r}")
    if semver_key(new) <= semver_key(current):
        raise ReleaseError(f"new version {new} is not greater than {current}")
    return new


def sections(changelog):
    """{name: (heading_start, body_start, body_end)} for each `## [...]` heading."""
    heads = list(HEADING_RE.finditer(changelog))
    links = LINK_RE.search(changelog)
    end_of_sections = links.start() if links else len(changelog)
    out = {}
    for i, m in enumerate(heads):
        body_end = heads[i + 1].start() if i + 1 < len(heads) else end_of_sections
        out[m.group(1)] = (m.start(), m.end(), body_end)
    return out


def section_body(changelog, name):
    spans = sections(changelog)
    if name not in spans:
        raise ReleaseError(f"CHANGELOG.md has no '## [{name}]' section")
    _, start, end = spans[name]
    return changelog[start:end].strip("\n") + "\n"


def released_versions(changelog):
    return [name for name in sections(changelog) if name != "Unreleased"]


def check(root, tag=None):
    """Return a list of problems; empty means consistent."""
    problems = []
    version = (root / "VERSION").read_text().strip()
    try:
        parse_semver(version)
    except ReleaseError as e:
        problems.append(f"VERSION: {e}")

    m = PYPROJECT_VERSION_RE.search((root / "pyproject.toml").read_text())
    if not m:
        problems.append("pyproject.toml: no top-level version line")
    elif m.group(1) != version:
        problems.append(f"pyproject.toml version {m.group(1)} != VERSION {version}")

    changelog = (root / "CHANGELOG.md").read_text()
    spans = sections(changelog)
    if "Unreleased" not in spans:
        problems.append("CHANGELOG.md: missing '## [Unreleased]' section")
    elif list(spans)[0] != "Unreleased":
        problems.append("CHANGELOG.md: '## [Unreleased]' must be the first section")
    if version not in spans:
        problems.append(f"CHANGELOG.md: no section for VERSION {version}")
    released = released_versions(changelog)
    dates = {m.group(1): m.group(2) for m in HEADING_RE.finditer(changelog)}
    for v in released:
        if not SEMVER_RE.match(v):
            problems.append(f"CHANGELOG.md: section [{v}] is not a SemVer version")
        elif not dates[v]:
            problems.append(f"CHANGELOG.md: section [{v}] has no date")
    valid = [v for v in released if SEMVER_RE.match(v)]
    if valid != sorted(valid, key=semver_key, reverse=True):
        problems.append("CHANGELOG.md: released sections are not newest-first")

    if tag is not None and tag != f"v{version}":
        problems.append(f"tag {tag} does not match VERSION (expected v{version})")
    return problems


def bump(root, part, date):
    version_file = root / "VERSION"
    current = version_file.read_text().strip()
    new = next_version(current, part)

    changelog_path = root / "CHANGELOG.md"
    changelog = changelog_path.read_text()
    body = section_body(changelog, "Unreleased")
    if not re.search(r"^\s*- ", body, re.MULTILINE):
        raise ReleaseError("[Unreleased] has no entries — nothing to release")
    if PLACEHOLDER in body:
        raise ReleaseError(f"[Unreleased] still contains a {PLACEHOLDER} placeholder")
    if new in sections(changelog):
        raise ReleaseError(f"CHANGELOG.md already has a section for {new}")

    _, unreleased_body_start, _ = sections(changelog)["Unreleased"]
    changelog = (
        changelog[:unreleased_body_start]
        + f"\n\n## [{new}] - {date}"
        + changelog[unreleased_body_start:]
    )
    changelog = update_links(changelog, current, new)

    pyproject_path = root / "pyproject.toml"
    pyproject, n = PYPROJECT_VERSION_RE.subn(f'version = "{new}"', pyproject_path.read_text(), count=1)
    if n != 1:
        raise ReleaseError("pyproject.toml: no top-level version line")

    changelog_path.write_text(changelog)
    pyproject_path.write_text(pyproject)
    version_file.write_text(new + "\n")
    return current, new


def update_links(changelog, previous, new):
    """Point [Unreleased] at v<new>...HEAD and add a compare link for <new>."""
    unreleased = f"[Unreleased]: {REPO_URL}/compare/v{new}...HEAD"
    release = f"[{new}]: {REPO_URL}/compare/v{previous}...v{new}"
    if re.search(r"^\[Unreleased\]: ", changelog, re.MULTILINE):
        changelog = re.sub(r"^\[Unreleased\]: \S+", lambda _: unreleased + "\n" + release,
                           changelog, count=1, flags=re.MULTILINE)
    else:
        changelog = changelog.rstrip("\n") + f"\n\n{unreleased}\n{release}\n"
    return changelog


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help=argparse.SUPPRESS)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_check = sub.add_parser("check", help="verify VERSION / pyproject / CHANGELOG agree")
    p_check.add_argument("--tag", help="also require this tag to equal v<VERSION>")
    p_bump = sub.add_parser("bump", help="cut a release in the working tree")
    p_bump.add_argument("part", help="major, minor, patch, or an explicit X.Y.Z[-pre]")
    p_bump.add_argument("--date", default=datetime.date.today().isoformat())
    p_bump.add_argument("--dry-run", action="store_true", help="print the new version only")
    p_notes = sub.add_parser("notes", help="print a version's changelog section")
    p_notes.add_argument("version", nargs="?", help="defaults to VERSION")
    args = parser.parse_args(argv)
    root = args.root

    try:
        if args.cmd == "check":
            problems = check(root, args.tag)
            for p in problems:
                print(f"✗ {p}", file=sys.stderr)
            if problems:
                return 1
            print(f"✓ release metadata consistent at {(root / 'VERSION').read_text().strip()}")
        elif args.cmd == "notes":
            version = args.version or (root / "VERSION").read_text().strip()
            sys.stdout.write(section_body((root / "CHANGELOG.md").read_text(), version.removeprefix("v")))
        elif args.cmd == "bump":
            if args.dry_run:
                print(next_version((root / "VERSION").read_text().strip(), args.part))
                return 0
            datetime.date.fromisoformat(args.date)
            previous, new = bump(root, args.part, args.date)
            print(f"✓ {previous} → {new}: VERSION, pyproject.toml and CHANGELOG.md updated\n")
            print("Next — commit this on a release branch, merge it, then tag the merged commit:")
            print(f"  git commit -am 'chore(release): v{new}'")
            print(f"  git tag -a v{new} -m 'v{new}' <merged-commit>")
            print(f"  python3 scripts/release.py notes {new} > release-notes.md")
            print("See docs/RELEASING.md for publishing.")
    except (ReleaseError, ValueError) as e:
        print(f"✗ {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
