# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""scripts/license-check.py: the distribution license policy gate used in CI."""

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("license_check", REPO_ROOT / "scripts" / "license-check.py")
lc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lc)


@pytest.mark.parametrize("lic", [
    "MIT", "Apache-2.0", "BSD-3-Clause", "ISC", "LGPL-2.1-or-later", "MPL-2.0",
    "GPL-2.0-only WITH Classpath-exception-2.0", "MIT OR GPL-3.0-only", "Python-2.0",
])
def test_permitted_licenses_pass(lic):
    assert lc.classify([lic]) is None


@pytest.mark.parametrize("lic, reason", [
    ("GPL-3.0-only", "strong copyleft"),
    ("AGPL-3.0-or-later", "network copyleft"),
    ("SSPL-1.0", "source-available"),
    ("BUSL-1.1", "source-available"),
    ("CC-BY-NC-SA-4.0", "non-commercial"),
])
def test_denied_licenses_fail(lic, reason):
    assert lc.classify([lic])[0] == reason


def test_cli_fails_on_a_denied_component(tmp_path, monkeypatch, capsys):
    sbom = tmp_path / "sbom.json"
    sbom.write_text(json.dumps({"components": [
        {"name": "ok", "version": "1", "licenses": [{"license": {"id": "MIT"}}]},
        {"name": "bad", "version": "2", "licenses": [{"license": {"id": "AGPL-3.0-only"}}]},
    ]}))
    monkeypatch.setattr("sys.argv", ["license-check.py", str(sbom)])
    assert lc.main() == 1
    assert "DENIED  bad@2" in capsys.readouterr().out


def test_cli_refuses_an_empty_input(tmp_path, monkeypatch):
    sbom = tmp_path / "sbom.json"
    sbom.write_text(json.dumps({"components": []}))
    monkeypatch.setattr("sys.argv", ["license-check.py", str(sbom)])
    assert lc.main() == 2


def test_requirement_names_handles_pins_extras_urls_and_comments(tmp_path):
    req = tmp_path / "requirements.txt"
    req.write_text(
        "# comment\n"
        "numpy>=1.23.5,<2\n"
        "uvicorn[standard]>=0.27.0\n"
        "diffusers @ git+https://github.com/huggingface/diffusers.git\n"
        "torch==2.4.0  # pinned\n"
        "-r other.txt\n"
        "\n"
    )
    assert list(lc._requirement_names(req)) == [
        ("numpy", None), ("uvicorn", None), ("diffusers", None), ("torch", "2.4.0"),
    ]


def test_every_cartridge_requirements_file_parses():
    for req in (REPO_ROOT / "inference").glob("**/requirements.txt"):
        if "/wan/" in str(req):
            continue
        list(lc._requirement_names(req))


# --------------------------------------------------------------------------
# scripts/sbom-merge.py
# --------------------------------------------------------------------------

_ms = importlib.util.spec_from_file_location("sbom_merge", REPO_ROOT / "scripts" / "sbom-merge.py")
sbom_merge = importlib.util.module_from_spec(_ms)
_ms.loader.exec_module(sbom_merge)


def _bom(root, deps):
    return {
        "metadata": {"component": {"bom-ref": root, "name": root}},
        "components": [{"bom-ref": d, "name": d} for d in deps],
        "dependencies": [{"ref": root, "dependsOn": deps}],
    }


def test_sbom_merge_roots_every_component(tmp_path, monkeypatch):
    a, b, out = tmp_path / "a.json", tmp_path / "b.json", tmp_path / "out.json"
    a.write_text(json.dumps(_bom("frontend", ["react", "shared"])))
    b.write_text(json.dumps(_bom("cdk", ["aws-cdk-lib", "shared"])))
    monkeypatch.setattr("sys.argv", ["sbom-merge.py", str(a), str(b), "-o", str(out)])
    assert sbom_merge.main() == 0
    merged = json.loads(out.read_text())
    assert merged["metadata"]["component"]["bom-ref"] == sbom_merge.ROOT_REF
    root = next(d for d in merged["dependencies"] if d["ref"] == sbom_merge.ROOT_REF)
    assert root["dependsOn"] == ["cdk", "frontend"]
    assert sorted(c["bom-ref"] for c in merged["components"]) == ["aws-cdk-lib", "cdk", "frontend", "react", "shared"]


def test_sbom_merge_rejects_an_unrooted_input(tmp_path, monkeypatch):
    a, out = tmp_path / "a.json", tmp_path / "out.json"
    a.write_text(json.dumps({"components": []}))
    monkeypatch.setattr("sys.argv", ["sbom-merge.py", str(a), "-o", str(out)])
    assert sbom_merge.main() == 2


def test_dockerfile_packages_separates_pypi_from_url_installs(tmp_path):
    df = tmp_path / "Dockerfile"
    df.write_text(
        'RUN pip install --no-cache-dir torch==2.7.1 "uvicorn[standard]" \\\n'
        '    --index-url https://download.pytorch.org/whl/cu128 -r /srv/req.txt\n'
        'RUN pip install "world_engine @ https://github.com/o/w/archive/v1.zip" git+https://github.com/m/MoGe.git\n'
        'RUN pip install --no-build-isolation -e ./local/pkg && echo done\n'
    )
    pkgs = list(lc._dockerfile_packages(df))
    assert ("torch", "2.7.1", None) in pkgs
    assert ("uvicorn", None, None) in pkgs
    assert ("world_engine", None, "https://github.com/o/w/archive/v1.zip") in pkgs
    assert any(src == "git+https://github.com/m/MoGe.git" for _, _, src in pkgs)
    names = [n for n, _, _ in pkgs]
    assert "download.pytorch.org" not in " ".join(names)  # an option value, not a package
    assert "/srv/req.txt" not in names and "./local/pkg" not in names


def test_every_dockerfile_parses():
    for df in [REPO_ROOT / "inference" / "Dockerfile.default", *(REPO_ROOT / "inference" / "models").glob("*/Dockerfile")]:
        list(lc._dockerfile_packages(df))


def test_every_shipped_base_image_has_a_recorded_license():
    dockerfiles = [REPO_ROOT / "inference" / "Dockerfile.default", *(REPO_ROOT / "inference" / "models").glob("*/Dockerfile")]
    images = lc._base_images(dockerfiles)
    assert images, "no base images found"
    for ref in images:
        assert any(ref.startswith(prefix) for prefix, _, _ in lc.BASE_IMAGE_LICENSES), ref


def test_base_images_include_shared_default_and_manifest_overrides():
    images = lc._base_images([REPO_ROOT / "inference" / "Dockerfile.default"])
    assert "nvcr.io/nvidia/pytorch:24.01-py3" in images      # the ARG default
    assert any("endpoint.yaml" in where for where in images.values())  # a base_image override


def test_unreviewed_non_pypi_install_fails(tmp_path, monkeypatch, capsys):
    df = tmp_path / "Dockerfile"
    df.write_text('RUN pip install "evil @ https://example.com/evil.zip"\n')
    sbom = tmp_path / "sbom.json"
    sbom.write_text(json.dumps({"components": [{"name": "ok", "version": "1", "licenses": [{"license": {"id": "MIT"}}]}]}))
    monkeypatch.setattr("sys.argv", ["license-check.py", str(sbom), "--dockerfiles", str(df)])
    assert lc.main() == 1
    assert "not in REVIEWED_NON_PYPI" in capsys.readouterr().out


def test_every_shipped_non_pypi_install_is_reviewed():
    dockerfiles = [REPO_ROOT / "inference" / "Dockerfile.default", *(REPO_ROOT / "inference" / "models").glob("*/Dockerfile")]
    for df in dockerfiles:
        if lc._out_of_scope(df):
            continue
        for name, _, source in lc._dockerfile_packages(df):
            if source:
                assert lc._reviewed(source), f"{name} from {source} in {df} is not in REVIEWED_NON_PYPI"


def test_failed_pypi_lookup_fails_closed(tmp_path, monkeypatch):
    req = tmp_path / "requirements.txt"
    req.write_text("somepkg\n")
    monkeypatch.setattr(lc, "_pypi_licenses", lambda n, v: (_ for _ in ()).throw(OSError("offline")))
    monkeypatch.setattr(lc.time, "sleep", lambda s: None)
    sbom = tmp_path / "sbom.json"
    sbom.write_text(json.dumps({"components": [{"name": "ok", "version": "1", "licenses": [{"license": {"id": "MIT"}}]}]}))
    monkeypatch.setattr("sys.argv", ["license-check.py", str(sbom), "--requirements", str(req)])
    assert lc.main() == 1


def test_out_of_scope_cartridges_are_skipped_and_gpl_in_scope_is_denied(tmp_path, monkeypatch, capsys):
    """world_engine (GPL-3.0) is only acceptable because waypoint-1-5 is out of scope."""
    out = tmp_path / "inference" / "models" / "waypoint-1-5"
    out.mkdir(parents=True)
    (out / "Dockerfile").write_text('RUN pip install "world_engine @ https://github.com/Overworldai/world_engine/archive/x.zip"\n')
    inscope = tmp_path / "inference" / "models" / "some-cartridge"
    inscope.mkdir(parents=True)
    (inscope / "Dockerfile").write_text("RUN pip install gplpkg\n")
    monkeypatch.setattr(lc, "_pypi_licenses", lambda n, v: ["GPL-3.0-only"])
    sbom = tmp_path / "sbom.json"
    sbom.write_text(json.dumps({"components": [{"name": "ok", "version": "1", "licenses": [{"license": {"id": "MIT"}}]}]}))
    monkeypatch.setattr("sys.argv", ["license-check.py", str(sbom), "--dockerfiles", str(out / "Dockerfile"), str(inscope / "Dockerfile")])
    assert lc.main() == 1
    text = capsys.readouterr().out
    assert "OUT OF SCOPE" in text and "waypoint-1-5" in text
    assert "DENIED  gplpkg" in text
