#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Merge per-package CycloneDX SBOMs into one SBOM rooted at the deliverable.

`npm sbom` and `uv export` each describe one package (the frontend, the CDK
app, the Python tooling). This joins them under a single root component for the
whole accelerator, so every direct and transitive dependency is reachable from
one root: root -> each package -> its dependency graph.

Usage:
  python scripts/sbom-merge.py sbom/frontend.cdx.json sbom/cdk.cdx.json \
      sbom/python.cdx.json -o sbom/world-model-inference.cdx.json
"""

import argparse
import json
import sys
import tomllib
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ROOT_REF = "world-model-inference"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sboms", nargs="+", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()

    version = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]["version"]
    components: dict[str, dict] = {}
    dependencies: dict[str, set[str]] = {}
    roots: list[str] = []

    for path in args.sboms:
        doc = json.loads(path.read_text())
        top = (doc.get("metadata") or {}).get("component")
        if not top or not top.get("bom-ref"):
            print(f"{path}: no metadata.component to root it under", file=sys.stderr)
            return 2
        roots.append(top["bom-ref"])
        for comp in [top, *doc.get("components", [])]:
            components.setdefault(comp["bom-ref"], comp)
        for dep in doc.get("dependencies", []):
            dependencies.setdefault(dep["ref"], set()).update(dep.get("dependsOn", []))

    dependencies[ROOT_REF] = set(roots)
    merged = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": f"urn:uuid:{uuid.uuid4()}",
        "version": 1,
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "component": {
                "type": "application",
                "bom-ref": ROOT_REF,
                "name": "world-model-inference",
                "version": version,
                "licenses": [{"license": {"id": "Apache-2.0"}}],
            },
        },
        "components": list(components.values()),
        "dependencies": [
            {"ref": ref, "dependsOn": sorted(deps)} for ref, deps in sorted(dependencies.items())
        ],
    }

    reachable, frontier = {ROOT_REF}, [ROOT_REF]
    while frontier:
        for child in dependencies.get(frontier.pop(), ()):
            if child not in reachable:
                reachable.add(child)
                frontier.append(child)
    orphans = set(components) - reachable
    args.output.write_text(json.dumps(merged, indent=2))
    print(f"{len(components)} components from {len(roots)} SBOMs, "
          f"{len(components) - len(orphans)} reachable from {ROOT_REF}.")
    for ref in sorted(orphans):
        print(f"  UNREACHABLE {ref}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
