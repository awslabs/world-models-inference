#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check third-party software licenses against the distribution policy.

Reads CycloneDX SBOMs (``npm sbom --sbom-format cyclonedx``) and/or
``pip-licenses --format=json`` output and fails if any component carries a
license this accelerator must not ship in a reusable, redistributed base:
strong/network copyleft, non-commercial, or source-available licenses.

It also resolves the packages each cartridge's container installs — from its
requirements files (``--requirements``) and the ``pip install`` lines in its
Dockerfile (``--dockerfiles``) — against PyPI metadata, without installing them.
Packages installed from a URL or VCS (not PyPI) are listed for manual review. With ``--dockerfiles`` it
also inventories each container's base image (``FROM``, plus ``base_image:``
overrides in endpoint.yaml) against the vendor licenses below, and fails on a
base image with no known license. Known vulnerabilities in built images are
covered by ECR scan-on-push (scripts/build-image.py).
Model *weights* licensing is enforced separately by tests/test_model_licenses.py
(see docs/MODEL_LICENSES.md).

Usage:
  python scripts/license-check.py sbom-frontend.cdx.json sbom-cdk.cdx.json \
      --pip-licenses python-licenses.json \
      --requirements inference/lib/requirements.txt inference/models/*/requirements.txt \
      --dockerfiles inference/Dockerfile.default inference/models/*/Dockerfile
"""

import argparse
import json
import re
import shlex
import sys
import time
import urllib.request
from pathlib import Path

# Not permitted in a redistributed base. Matched case-insensitively against the
# SPDX id or license name. LGPL is deliberately NOT here: it is allowed for
# dynamically-linked dependencies, which is how npm/pip packages are consumed.
DENIED = [
    (re.compile(r"\bAGPL", re.I), "network copyleft"),
    (re.compile(r"(?<!L)\bGPL(?!.*(linking|classpath)[- ]exception)", re.I), "strong copyleft"),
    (re.compile(r"\bSSPL", re.I), "source-available"),
    (re.compile(r"\bBUSL|Business Source", re.I), "source-available"),
    (re.compile(r"Commons Clause", re.I), "source-available"),
    (re.compile(r"\bCC-BY-NC|NonCommercial", re.I), "non-commercial"),
    (re.compile(r"\bEUPL|\bOSL-", re.I), "copyleft"),
]


def _cyclonedx_components(path: Path):
    doc = json.loads(path.read_text())
    for comp in doc.get("components", []):
        names = []
        for entry in comp.get("licenses", []) or []:
            lic = entry.get("license") or {}
            names.append(lic.get("id") or lic.get("name") or entry.get("expression") or "")
        yield f"{comp.get('name')}@{comp.get('version')}", [n for n in names if n]


def _pip_components(path: Path):
    for row in json.loads(path.read_text()):
        yield f"{row['Name']}@{row['Version']}", [row.get("License") or ""]


_REQ_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")
_PIN = re.compile(r"==\s*([A-Za-z0-9.+!_-]+)")


def _requirement_names(path: Path):
    """(name, pinned version or None) for each package line in a requirements file."""
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        m = _REQ_NAME.match(line)
        if m:
            pin = _PIN.search(line)
            yield m.group(1), (pin.group(1) if pin else None)


def _pypi_licenses(name: str, version):
    """License strings PyPI publishes for a package (expression, field, classifiers)."""
    url = f"https://pypi.org/pypi/{name}/{version}/json" if version else f"https://pypi.org/pypi/{name}/json"
    with urllib.request.urlopen(url, timeout=20) as resp:  # nosec B310 - fixed https PyPI URL
        info = json.load(resp)["info"]
    found = [info.get("license_expression") or ""]
    field = (info.get("license") or "").strip()
    if field and len(field) < 100:  # some packages paste the whole license text here
        found.append(field)
    found += [c.split("::")[-1].strip() for c in info.get("classifiers") or [] if c.startswith("License ::")]
    return [f for f in found if f and f.upper() not in ("UNKNOWN", "OTHER/PROPRIETARY LICENSE")]


# Base-image registries/repositories and the license their images ship under.
# NVIDIA's images bundle CUDA/cuDNN/NCCL etc. under NVIDIA's terms; those terms
# permit use and deployment of the images (they are pulled from NVIDIA by the
# customer's own build, never redistributed by this repo).
BASE_IMAGE_LICENSES = [
    ("nvcr.io/nvidia/", "NVIDIA Deep Learning Container License",
     "https://developer.download.nvidia.com/licenses/NVIDIA_Deep_Learning_Container_License.pdf"),
    ("nvidia/cuda:", "NVIDIA CUDA Toolkit EULA (container images: NVIDIA Deep Learning Container License)",
     "https://docs.nvidia.com/cuda/eula/index.html"),
    ("python:", "PSF-2.0 (Docker Official Image; OS packages under their own licenses)",
     "https://hub.docker.com/_/python"),
]


def _base_images(dockerfiles):
    """Every concrete base image the containers build FROM, with where it is set."""
    images = {}
    for path in dockerfiles:
        text = path.read_text()
        default = re.search(r"^ARG\s+BASE_IMAGE=(\S+)", text, re.M)
        for m in re.finditer(r"^FROM\s+(\S+)", text, re.M):
            ref = m.group(1)
            if ref == "${BASE_IMAGE}":
                if default:
                    images.setdefault(default.group(1), str(path))
                # Per-cartridge overrides of the shared Dockerfile's base image.
                for manifest in sorted((path.parent / "models").glob("*/endpoint.yaml")):
                    o = re.search(r"^base_image:\s*(\S+)", manifest.read_text(), re.M)
                    if o:
                        images.setdefault(o.group(1).strip("\"'"), str(manifest))
            else:
                images.setdefault(ref, str(path))
    return images


# Packages installed from a URL or VCS rather than PyPI have no PyPI metadata to
# check, so each one is reviewed by hand and recorded here: the install source
# (substring) and its license. An unlisted non-PyPI install fails the check.
# Reviewed licenses still go through the DENIED list like any other.
REVIEWED_NON_PYPI = [
    ("github.com/huggingface/diffusers", "Apache-2.0", "cosmos3-nano"),
    ("github.com/Dao-AILab/flash-attention", "BSD-3-Clause", "flash-attn wheels"),
    ("${URL}", "BSD-3-Clause", "Dockerfile.default flash-attn wheel (FLASH_ATTN_WHEEL)"),
]

# Cartridges outside the reusable-base scope (docs/MODEL_LICENSES.md, "Reusable-base
# scope"). Their requirements and Dockerfiles are not part of what is cleared, so
# they are listed as out of scope rather than checked: waypoint-1-5 installs the
# GPL-3.0 world_engine, lyra-2's weights are internal-research only.
OUT_OF_SCOPE_CARTRIDGES = {"waypoint-1-5", "lyra-2"}


def _out_of_scope(path: Path) -> bool:
    parts = Path(path).parts
    return any(p in OUT_OF_SCOPE_CARTRIDGES for p in parts)


def _reviewed(source: str):
    return next(((lic, note) for key, lic, note in REVIEWED_NON_PYPI if key in source), None)


# pip options that take a value, whose value is therefore not a package.
_PIP_VALUE_OPTS = {"-r", "--requirement", "-c", "--constraint", "-e", "--editable",
                   "-i", "--index-url", "--extra-index-url", "-f", "--find-links",
                   "--retries", "--timeout", "--target", "--prefix", "--platform"}


def _dockerfile_packages(path: Path):
    """Yield (name, pin, source) for each package a Dockerfile ``pip install``s.

    ``source`` is None for a PyPI package, else the URL/VCS it is installed from.
    Requirements files and editable local paths are skipped (they are passed via
    --requirements, or are first-party).
    """
    text = path.read_text().replace("\\\n", " ")
    for line in text.splitlines():
        if not line.lstrip().startswith("RUN"):
            continue
        for m in re.finditer(r"pip3?\s+install\s+([^&;|]*)", line):
            tokens = shlex.split(m.group(1), comments=False, posix=True)
            skip = False
            for tok in tokens:
                if skip:
                    skip = False
                    continue
                if tok in _PIP_VALUE_OPTS:
                    skip = True
                    continue
                if tok.startswith("-"):
                    continue
                if " @ " in tok:
                    name, url = (part.strip() for part in tok.split(" @ ", 1))
                    yield name, None, url
                elif "://" in tok or tok.startswith(("git+", ".", "/", "$")):
                    yield tok.rsplit("/", 1)[-1], None, tok
                else:
                    nm = _REQ_NAME.match(tok)
                    if nm:
                        pin = _PIN.search(tok)
                        yield nm.group(1), (pin.group(1) if pin else None), None


def _requirements_components(paths, dockerfiles=()):
    seen = {}
    skipped = sorted({str(p) for p in [*paths, *dockerfiles] if _out_of_scope(p)})
    for path in skipped:
        print(f"  OUT OF SCOPE {path} (cartridge excluded from the reusable base)")
    paths = [p for p in paths if not _out_of_scope(p)]
    dockerfiles = [p for p in dockerfiles if not _out_of_scope(p)]
    for path in paths:
        for name, version in _requirement_names(path):
            seen.setdefault(name.lower(), (name, version))
    for path in dockerfiles:
        for name, version, source in _dockerfile_packages(path):
            if not source:
                seen.setdefault(name.lower(), (name, version))
            elif _reviewed(source):
                lic, note = _reviewed(source)
                print(f"  REVIEWED {name}: {lic} ({note})")
                yield f"{name} (non-PyPI, reviewed: {note})", [lic]
            else:
                yield f"{name} (non-PyPI, NOT reviewed: {source}, in {path})", ["UNREVIEWED"]
    for name, version in sorted(seen.values()):
        last_error = None
        for attempt in range(3):
            try:
                yield f"{name}@{version or 'latest'} (pypi)", _pypi_licenses(name, version)
                break
            except OSError as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
        else:
            yield f"{name}@{version or 'latest'} (pypi lookup failed: {last_error})", ["LOOKUP-FAILED"]


def classify(licenses):
    """(reason, license) for the first denied license, else None."""
    for lic in licenses:
        # An SPDX "A OR B" expression is acceptable if any alternative is.
        alternatives = [a.strip(" ()") for a in re.split(r"\bOR\b", lic)]
        if all(any(p.search(a) for p, _ in DENIED) for a in alternatives if a):
            for pattern, reason in DENIED:
                if pattern.search(lic):
                    return reason, lic
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sboms", nargs="*", type=Path, help="CycloneDX JSON SBOMs")
    parser.add_argument("--pip-licenses", type=Path, action="append", default=[],
                        help="pip-licenses --format=json output (repeatable)")
    parser.add_argument("--requirements", type=Path, nargs="+", default=[],
                        help="requirements.txt files resolved against PyPI metadata")
    parser.add_argument("--dockerfiles", type=Path, nargs="+", default=[],
                        help="Dockerfiles whose pip installs are resolved the same way")
    args = parser.parse_args()

    components = []
    for sbom in args.sboms:
        components += list(_cyclonedx_components(sbom))
    for pl in args.pip_licenses:
        components += list(_pip_components(pl))
    components += list(_requirements_components(args.requirements, args.dockerfiles))
    if not components:
        print("No components found — nothing was checked.", file=sys.stderr)
        return 2

    denied, unknown, unverified = [], [], []
    for name, licenses in components:
        if licenses and licenses[0] in ("UNREVIEWED", "LOOKUP-FAILED"):
            unverified.append((name, licenses[0]))
            continue
        if not licenses or all(l.strip().upper() in ("", "UNKNOWN", "NOASSERTION") for l in licenses):
            unknown.append(name)
            continue
        hit = classify(licenses)
        if hit:
            denied.append((name, *hit))

    base_failures = []
    images = _base_images(args.dockerfiles)
    if images:
        print(f"Base images ({len(images)}):")
        for ref, where in sorted(images.items()):
            match = next(((lic, url) for prefix, lic, url in BASE_IMAGE_LICENSES if ref.startswith(prefix)), None)
            if match:
                print(f"  {ref}: {match[0]} — {match[1]}  [{where}]")
            else:
                base_failures.append(ref)
                print(f"  UNKNOWN {ref}: no recorded license — add it to BASE_IMAGE_LICENSES  [{where}]")

    print(f"Checked {len(components)} components.")
    for name, why in sorted(unverified):
        print(f"  FAILED  {name}: {'not in REVIEWED_NON_PYPI' if why == 'UNREVIEWED' else 'could not be verified'}")
    for name in sorted(unknown):
        print(f"  REVIEW  {name}: no license declared in package metadata")
    for name, reason, lic in sorted(denied):
        print(f"  DENIED  {name}: {lic} ({reason})")
    if denied:
        print(f"{len(denied)} component(s) carry a license not permitted for distribution.")
        return 1
    if unverified:
        print(f"{len(unverified)} component(s) could not be verified; review them and add to "
              "REVIEWED_NON_PYPI, or retry a failed lookup.")
        return 1
    if base_failures:
        print(f"{len(base_failures)} base image(s) have no recorded license.")
        return 1
    print("No denied licenses." + (f" {len(unknown)} to review by hand." if unknown else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
