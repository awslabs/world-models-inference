# Releasing

How World Model Inference is versioned, how changes are recorded, and how a release
is cut. The changelog is [CHANGELOG.md](../CHANGELOG.md).

## Versions

### The solution version

The solution has one version, in [`VERSION`](../VERSION), following
[Semantic Versioning 2.0](https://semver.org/spec/v2.0.0.html). `pyproject.toml`
mirrors it; `scripts/release.py check` (run in CI and by the test suite) fails if they
disagree or if `CHANGELOG.md` has no section for it. A release is the annotated git
tag `v<VERSION>`.

What the version covers — the surface a user builds on:

- the `deploy.sh` commands, their flags and environment variables;
- the cartridge contract: the `endpoint.yaml` schema, `create_runner()` and the
  shared serving layer in `inference/lib`;
- the HTTP and WebSocket API an endpoint serves, and its authentication;
- the CDK stacks: names, parameters, outputs and the resources a redeploy replaces;
- which cartridges the catalogue ships.

| Bump | When |
|---|---|
| **MAJOR** | A change to that surface that breaks an existing deployment, client or cartridge |
| **MINOR** | New cartridges, commands, options or API, compatible with what exists |
| **PATCH** | Fixes, security fixes, dependency updates and docs, with no new surface |

While the version is `0.y.z` the surface is still settling, so a MINOR release may
break it. Every breaking change is still called out with a leading **Breaking:** in its
changelog entry, together with what to do about it. Release candidates use a
pre-release suffix (`1.0.0-rc.1`).

`cdk/package.json` and `frontend/package.json` also carry a `version` field. Both are
private, unpublished packages and their versions mean nothing; ignore them.

### Cartridge image versions

Each cartridge's `inference/models/<id>/endpoint.yaml` has its own `version:`, which
is the tag of its container image in Amazon ECR. It is a SemVer of the image, separate
from the solution version, so that releasing the solution does not force every model to
rebuild.

Bump a cartridge's `version:` in the same change that alters anything baked into its
image:

- anything under `inference/models/<id>/`, including its `Dockerfile` and
  `requirements.txt`;
- `inference/Dockerfile.default`, for cartridges that build from it;
- `inference/lib`, which every image contains — so a serving-layer change bumps every
  cartridge.

Treat image tags as immutable: a deploy reuses the image already in ECR at that tag, so
an image change without a bump never reaches a redeploy, and where the repository
enforces tag immutability, pushing a changed image under an existing tag is refused. Use PATCH for fixes and dependency pins, MINOR for new
behaviour (actions, modes, parameters) and MAJOR when the cartridge's request or
response format changes. Name each bumped cartridge with its old and new version in
the changelog entry.

## Recording changes

Every change a user would notice adds an entry under `## [Unreleased]` in
`CHANGELOG.md`, in the same merge request as the change. Use the
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) headings — **Added**,
**Changed**, **Deprecated**, **Removed**, **Fixed**, **Security** — and write the entry
for the person upgrading: what changed, why it matters to them, and anything they have
to do. Tests, CI and refactors with no visible effect need no entry.

An entry that cannot be written yet (for example, while a large change is still in
review) may hold the marker `TODO(release)`; `scripts/release.py bump` refuses to cut a
release while one remains.

## Cutting a release

Releases are cut from `main`.

1. **Prepare.** `main` is green and every merged change since the last release has its
   changelog entry. Decide the bump from the table above.

2. **Bump.** On a release branch:

   ```bash
   python3 scripts/release.py bump minor        # or major, patch, or an explicit X.Y.Z
   ```

   This moves the `[Unreleased]` entries under a new `## [X.Y.Z] - <today>` heading,
   updates the compare links, and writes the version into `VERSION` and
   `pyproject.toml`. Read the section through and tidy it; it becomes the release notes.

3. **Merge.** Open a merge request titled `chore(release): vX.Y.Z` with only those
   files, let CI pass, and merge it.

4. **Tag** the merged commit on `main`, annotated:

   ```bash
   git tag -a vX.Y.Z -m "vX.Y.Z" <merged-commit>
   git push <remote> vX.Y.Z                  # push the one tag; never --tags
   ```

5. **Release.** The tag pipeline re-runs the full CI, checks the tag matches `VERSION`
   (`scripts/release.py check --tag`), takes the notes from the changelog section
   (`scripts/release.py notes`) and creates the release with the merged CycloneDX SBOM
   attached. On GitHub this is `.github/workflows/release.yml`; a pre-release tag
   creates a pre-release.

Maintainers develop in a separate internal repository and publish to the public GitHub
repository by porting content, not history, so a release is tagged in both places under
the same name. In the public repository the tag must name a public commit — the one
that published the release's content.

A fix for an already-released version, when `main` has moved on with changes that should
not ship yet, is cut from a `release/X.Y` branch started at the release tag; the fix
lands on `main` first and is cherry-picked across.

## Release checklist

- [ ] `uv run --group dev python -m pytest tests/ -q` passes, including the licence and
      release-metadata tests.
- [ ] `cd cdk && npx tsc --noEmit` and `cd frontend && npx tsc --noEmit && npm run build`
      are clean.
- [ ] The SBOM job passes the licence policy (`scripts/license-check.py`), and its merged
      SBOM is attached to the release.
- [ ] Every cartridge declares `license`, `license_url` and `license_restricted` in
      `endpoint.yaml`, and [MODEL_LICENSES.md](MODEL_LICENSES.md) is current.
- [ ] No cartridge whose weights licence disallows this use is shipped or listed in the
      catalogue (LingBot World V2 is excluded).
- [ ] Authentication is Amazon Cognito; there is no shared or static token anywhere.
- [ ] Every cartridge whose image content changed has a bumped `version:`, named in the
      changelog.
- [ ] No account IDs, ARNs, internal hostnames, instance IDs, credentials or personal
      email addresses in tracked files, and no measurement evidence
      (see [EVIDENCE.md](EVIDENCE.md)).
- [ ] The changelog section reads as release notes: breaking changes marked, upgrade
      steps given, no `TODO(release)` left.
