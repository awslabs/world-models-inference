# Changelog

All notable changes to World Model Inference are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the solution is versioned
with [Semantic Versioning](https://semver.org/spec/v2.0.0.html) — see
[docs/RELEASING.md](docs/RELEASING.md) for what a version covers and how a release is cut.

Each cartridge's container image carries its own `version:` in
`inference/models/<id>/endpoint.yaml`; when a release rebuilds an image, the entry names
the cartridge and its new image version.

## [Unreleased]

### Added

- **Matrix Game 2 cartridge** (`matrix-game-2`, image `0.1.0`): Skywork's real-time
  world model (MIT code and weights) on the shared serving layer. Validated on a
  `g6e.4xlarge` (1× L40S) at 14.5 fps, 640×352. Deploy with `./deploy.sh matrix-game-2`.
- **One-click Launch Stack**: a single CloudFormation template that builds and deploys a
  chosen cartridge with no local tooling, serves the UI from S3 behind the same
  CloudFront distribution and signs users in with the Cognito Hosted UI. See
  [docs/LAUNCH_STACK.md](docs/LAUNCH_STACK.md).
- **GPU capacity wizard** (`scripts/capacity.py`): finds Regions and instance types with
  free capacity for a cartridge before you deploy, by probing with short-lived targeted
  On-Demand Capacity Reservations that it always cancels. Each cartridge now declares
  `gpu_count` and `gpu_memory_gb` in `endpoint.yaml`. See [docs/CAPACITY.md](docs/CAPACITY.md).
- **Cosmos 3 forward dynamics**: `cosmos3-nano` takes a start image plus an action
  trajectory (`action_mode`, `domain_name`, `view_point`, `raw_actions`,
  `resolution_tier`) and generates the video that follows. Each new input is bounded
  by the `/invocations` allowlist.

### Changed

- **Breaking:** EC2 endpoints use HTTPS by default — through CloudFront, or ACM and
  Route 53 when you have a domain. Plaintext HTTP needs `ALLOW_PLAINTEXT=1`, and the
  local proxy now verifies certificates.
- **Breaking:** a required Cognito scope is mandatory. The server refuses to start
  without `WORLD_MODEL_COGNITO_SCOPE` and rejects tokens that lack it. `deploy.sh` also
  needs the AWS CLI.
- `deploy.sh` passes CDK the same Region it resolves, so a profile Region and
  `AWS_REGION` can no longer split a deployment across Regions. `destroy` waits for the
  stack deletion and fails if it does not finish. With `CAPACITY_BLOCK_ID` set, the
  instance is pinned to the Capacity Block's Availability Zone.
- The catalogue shows prices in $/hr (it rendered most cards as `$0.00/s`), and the
  V-JEPA 2 card matches its cartridge. The docs now match the code: you run
  `npx cdk bootstrap` before `deploy.sh`, and weight sizes and instance prices are
  corrected.
- `CONTRIBUTING.md` is rewritten, and the README lists the contributors.
- Cartridge image versions (the `inference/lib` serving layer changed under all of
  them): `cosmos3-nano`, `echo-async`, `echo-realtime`, `lingbot-fast`, `lyra-2`,
  `matrix-game-3`, `vjepa2` and `waypoint-1-5` go from `0.1.0` to `0.1.1`.

### Fixed

- `waypoint-1-5` installs the shared serving requirements. Without PyJWT its container
  refused to start once Cognito auth was required.
- `cosmos3-nano` image: numpy is held below 2 and only one OpenCV is installed. Either
  clash crashed the image at import.
- `/invocations` returns 400, not 500, for non-string choice values and for numbers too
  large to convert.

### Security

- The development proxy listens on loopback only and gives its token to the local UI
  alone. The server rejects WebSocket connections from origins that are not allowed.
- SSM and CDK context values are validated in `deploy.sh` and CDK and are no longer
  interpolated into the boot script. The container environment is passed as a base64
  `--env-file`, and the model-data URI is checked at boot.
- Build-time installs and Hugging Face weight revisions are pinned (every `hf:` source
  in `endpoint.yaml` names a `revision:`). The weights and image build roles are
  separate. ECR tags are immutable, and deploys go by image digest.
- Dependency updates: urllib3 2.8.0 (GHSA-8988-9cw3-xx77, GHSA-vxq7-64xx-v4gw,
  GHSA-gh4c-6fx4-qh6g), source-map-js 1.2.2 (frontend build), PyJWT 2.15.0,
  Pillow 13.0.0, fsspec 2026.6.0 and anyio 4.14.2.

## [0.1.0] - 2026-09-30

First versioned baseline: the solution as published in the initial open-source release
and its first sync. Reconstructed from the commit history; earlier work was not versioned.

### Added

- Model "cartridges" under `inference/models/<id>/`, each declared by an `endpoint.yaml`
  and served by the shared FastAPI serving layer in `inference/lib` (`create_runner()`
  contract, async `/generate` and real-time WebSocket streaming with an action buffer,
  per-session `runner.reset()`, multi-GPU distributed inference).
- Cartridges: `echo-async` and `echo-realtime` (smoke tests), `lingbot-fast` (8-GPU
  image→video), `cosmos3-nano` (text→video), `vjepa2` (real-time encoder), `lyra-2`
  (image→video), `matrix-game-3` (real-time, wired end to end) and `waypoint-1-5`
  (Overworld Waypoint-1.5-1B real-time, 720p). All cartridge images are at `0.1.0`.
- A shared default Dockerfile and a `_template` cartridge for adding a new model
  (see [docs/ADDING_A_MODEL.md](docs/ADDING_A_MODEL.md)).
- CDK app with independent stacks, named deployments and an SSM registry; image builds
  and weight staging run in CodeBuild; `deploy.sh` drives build, stage and deploy.
- React catalogue UI (`frontend/`) that reaches a deployed endpoint through a Vite
  proxy, with per-cartridge generation profiles.
- Amazon Cognito authentication for the UI and the inference API.
- A per-model licence record ([docs/MODEL_LICENSES.md](docs/MODEL_LICENSES.md)); every
  `endpoint.yaml` declares `license`, `license_url` and `license_restricted`, enforced by
  the test suite, and `waypoint-1-5` deploys only after a licence acknowledgement.
- CI on GitHub Actions and GitLab: Python unit tests, CDK and frontend typechecks, script
  syntax, and CycloneDX SBOMs checked against the distribution licence policy.
- Architecture, endpoints, troubleshooting and security documentation, and a draw.io
  architecture diagram.

### Changed

- Renamed from World Model Accelerator / World Foundry to World Model Inference.

### Removed

- LingBot World V2: its weights licence (CC BY-NC-SA) does not allow this use, so the
  cartridge is excluded and the catalogue does not list it.

### Security

- Inference API: authentication, CORS allowlist, path validation, rate limiting (keyed
  on the last `X-Forwarded-For` hop) and security headers.
- Containers run as non-root; Hugging Face revisions are pinned and pickles load with
  `weights_only`.
- EC2 instances in private subnets with no public IP; ALB ingress restricted to a CIDR;
  IAM and CodeBuild S3 access scoped to project buckets; S3 access logging, lifecycle
  rules and TLS-only bucket policies.
- Remediation of the internal security review findings, including deployment-name
  validation and a mandatory alarm subscriber.

[Unreleased]: https://github.com/awslabs/world-models-inference/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/awslabs/world-models-inference/releases/tag/v0.1.0
