# Contributing to World Model Inference

Thank you for your interest in contributing! Bug reports, new cartridges, fixes and
documentation improvements are all welcome. This document explains how to get set
up and what a change needs before it can be merged.

## Reporting bugs

Open an issue with:

- A clear, descriptive title
- Steps to reproduce the problem
- Expected behaviour vs. actual behaviour
- Your environment: OS, AWS CLI / CDK version, model id, instance type, region
- Relevant logs or screenshots — **redact AWS account IDs, ARNs, hostnames and tokens**

## Suggesting features

Open an issue describing the problem you are trying to solve, your proposed
solution, and any alternatives you considered. For a new model, say which one, link
its weights and code licences, and note the GPU it needs.

## Security issues

**Do not open a public issue for a security vulnerability.** Report it through the
[AWS vulnerability reporting page](https://aws.amazon.com/security/vulnerability-reporting/). See
[docs/SECURITY.md](docs/SECURITY.md) for the project's security posture.

## Development setup

Prerequisites: Python 3.12, [uv](https://docs.astral.sh/uv/), Node.js 20+, and the
AWS CLI + CDK if you want to deploy.

```bash
git clone https://github.com/<your-fork>/world-models-inference.git
cd world-models-inference

# Python tests (what CI runs)
uv run --group dev python -m pytest tests/ -q

# Frontend
cd frontend && npm ci && npm run dev

# CDK type-check
cd cdk && npm ci && npx tsc --noEmit
```

## Submitting changes

1. **Fork** the repository and branch from `main` — use descriptive names such as
   `feat/<model-id>-cartridge`, `fix/<issue>` or `docs/<topic>`.
2. **Keep changes focused.** One logical change per pull request; small commits with
   clear messages.
3. **Add or update tests** for any behaviour you change.
4. **Run the checks** below locally.
5. **Add a changelog entry** under `## [Unreleased]` in [CHANGELOG.md](CHANGELOG.md)
   for any change a user would notice, and bump the cartridge's `version:` in
   `endpoint.yaml` if you changed what goes into its image. See
   [docs/RELEASING.md](docs/RELEASING.md).
6. **Open a pull request** that explains what changed, why, and how you tested it
   (for a cartridge: the instance type and measured fps or latency).

### Before you open a pull request

- [ ] `uv run --group dev python -m pytest tests/ -q` passes
- [ ] `cd frontend && npx tsc --noEmit && npm run build` passes (if you touched the frontend)
- [ ] `cd cdk && npx tsc --noEmit` passes (if you touched infrastructure)
- [ ] No AWS account IDs, ARNs, hostnames, instance IDs, keys or tokens in the diff —
      deployment evidence stays out of the repo (see [docs/EVIDENCE.md](docs/EVIDENCE.md))
- [ ] Docs updated if you changed behaviour, flags or `endpoint.yaml` keys

### Adding a model (cartridge)

Follow [docs/ADDING_A_MODEL.md](docs/ADDING_A_MODEL.md). In addition:

- Every cartridge must declare `license`, `license_url` and `license_restricted` in its
  `endpoint.yaml`; `tests/test_model_licenses.py` enforces this.
- Add the model to [docs/MODEL_LICENSES.md](docs/MODEL_LICENSES.md) with both its
  **weights** and **upstream code** licences. Weights that forbid commercial use must
  be gated behind a deploy-time licence acknowledgement.
- Third-party code you vendor must be compatible with Apache 2.0 and attributed in
  [NOTICE](NOTICE).

## Code style

- **Python**: PEP 8, type hints where practical.
- **TypeScript/React**: follow the existing patterns — functional components and hooks.
- **Shell**: `set -euo pipefail`; quote variables.

## Code of conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).

## Licensing

By submitting a pull request, you agree that your contribution is licensed under the
Apache License 2.0, the same licence that covers this project. See [LICENSE](LICENSE).
