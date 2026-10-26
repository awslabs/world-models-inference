# Launch Stack

The Launch Stack deploys a cartridge from the CloudFormation console with no local
tooling: no `aws` CLI, Node, CDK bootstrap or Docker. It does what `./deploy.sh
<model>` does, inside the account:

```
WorldModelLaunch (parent)
├── Foundation (nested)      VPC, ALB security group, EC2 role, buckets, CodeBuild,
│                            Cognito user pool + first user, SSM registry
└── Model<Cartridge> (nested, only the chosen one is created)
    ├── Build                Step Functions: check the Capacity Block, then in parallel
    │                          stage each pinned weights revision (CodeBuild)
    │                          and build the image (CodeBuild), skipping work already done
    │                        -> signals a WaitCondition (up to 12 h)
    ├── Image digest         read from ECR (immutable tag) after the wait
    ├── Endpoint             launch template (by digest), ASG, ALB, CloudFront
    ├── UI                   S3 bucket served by the same distribution, Cognito Hosted UI client
    └── Desired capacity 1   once, on create
```

It reuses the same constructs, CodeBuild projects and buildspecs as `deploy.sh`
(`SharedStack.define`, `Ec2Inference`, `scripts/stage-weights.py`,
`scripts/build-image.py`), so the two paths cannot drift.

## Parameters

| Parameter | Notes |
|---|---|
| `Model` | Any cartridge in `inference/models/` (LingBot World V2 is refused by the generator and the CDK app). |
| `AcceptModelLicense` | Must be `yes` for a licence-restricted cartridge (`lyra-2`, `waypoint-1-5`); a CloudFormation Rule refuses the stack otherwise. See [Model licences](MODEL_LICENSES.md). |
| `InstanceType` | `model-default` uses the cartridge's `endpoint.yaml`, or pick a GPU type. |
| `AdminEmail` | Invited to Cognito (temporary password by email) and subscribed to scaling-event alerts (confirm the AWS email). |
| `AllowedCidr` | Enforced at the CloudFront edge and on the ALB. `0.0.0.0/0` leaves Cognito sign-in as the only gate. |
| `CapacityBlockId` | Optional. Checked before anything is built (`active` or `scheduled`); the instance is pinned to the block's AZ. |
| `ReuseRetainedBuckets` | `yes` after an earlier launch in the account/Region left its `world-model-*` buckets behind (they are retained on delete). Keeps already-staged weights. |

Every parameter is constrained by `AllowedValues` or `AllowedPattern`. Values that
reach the boot script (the image URI with the built digest, the model data URI) are
base64-encoded into user data and shape-checked on the instance; nothing is
interpolated as shell text.

## What you get

- **UiUrl**: the catalogue UI on the endpoint's own `*.cloudfront.net` name. Sign-in
  is the Cognito Hosted UI (authorization code + PKCE); the browser presents the
  user's access token directly, so there is no dev proxy and no machine token.
- **AutoScalingGroup**: set desired capacity to 0 to stop paying for the GPU, 1 to
  start again.
- **BuildStateMachine**: the Step Functions state machine; open it to follow the
  weights and image builds while the stack is waiting.

## Cost

The same as `./deploy.sh` (see the README's Cost table) plus, while it runs: a NAT
gateway, an ALB and a CloudFront distribution (a few dollars a day together),
CodeBuild minutes for the first build (`X2_LARGE` for weights, `LARGE` for the
image) and S3 storage for the weights. The GPU instance dominates; it starts as
soon as the build finishes.

## Updating and deleting

- **Change the cartridge**: update the stack with a different `Model`. The old
  cartridge's nested stack is deleted and the new one built.
- **A new cartridge version** (a new published template): updating the stack
  re-runs the build for that cartridge only if its version or recipe changed. The
  running instance keeps the old launch template version until it is replaced:
  scale to 0 and back to 1. Image tags are the cartridge `version` and are
  immutable, so a Dockerfile change needs a version bump or the build reuses the
  old image.
- **No GPU capacity**: the stack still reaches `CREATE_COMPLETE`; the Auto Scaling
  group keeps retrying, and its Activity tab shows `InsufficientInstanceCapacity`.
  Update the stack with another `InstanceType`, or with a `CapacityBlockId`. The
  [GPU capacity wizard](CAPACITY.md) finds one that's free and pre-fills the link.
- **Delete**: deletes everything except the retained resources: the artifacts,
  outputs and access-log buckets, the Cognito user pool, and the ECR repository
  (created by the build, outside CloudFormation). Relaunch with
  `ReuseRetainedBuckets=yes` to keep the staged weights.

## Limits

- The VPC spans the Region's first two Availability Zones. A GPU type with capacity
  only in a third AZ cannot be used.
- One Launch Stack per account and Region. Its foundation uses the same fixed names
  as `./deploy.sh`'s `WorldModelFoundation` (roles, CodeBuild projects, SSM
  parameters, Cognito domain), so the two cannot coexist in one Region.
- EC2 only (no SageMaker target), HTTPS through CloudFront only (no custom domain).
- Commercial partition only: the boot check expects `*.amazonaws.com` ECR hosts.
- New ECR repositories are created `IMMUTABLE` (no `buildcache` exclusion), so the
  layer cache is not reused across versions. `build-image.py` upgrades the setting if
  it later touches the repository.
- No Hugging Face token parameter: every shipped weights repository is public. A
  gated cartridge needs one before it can be offered here.

## Publishing

Maintainers publish the template with `scripts/publish.py`. It needs the same
tooling as `deploy.sh` (Python 3, Node, the `aws` CLI) and credentials that can write
to the bucket.

1. Create one bucket per Region, named `<basename>-<region>` (the script never
   creates or reconfigures buckets). For a public Launch Stack, allow anonymous reads
   of the prefix with a bucket policy (and leave Block Public Access off for policies
   on this bucket only):

   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Sid": "PublicReadLaunchStack",
         "Effect": "Allow",
         "Principal": "*",
         "Action": "s3:GetObject",
         "Resource": "arn:aws:s3:::<basename>-<region>/<prefix>/*"
       },
       {
         "Sid": "DenyInsecureTransport",
         "Effect": "Deny",
         "Principal": "*",
         "Action": "s3:*",
         "Resource": ["arn:aws:s3:::<basename>-<region>", "arn:aws:s3:::<basename>-<region>/*"],
         "Condition": { "Bool": { "aws:SecureTransport": "false" } }
       }
     ]
   }
   ```

   The launching account reads the nested templates, Lambda code, UI and
   `inference/` source from this bucket during deployment, so they must stay there
   for as long as anyone may launch, update or roll back that version.

2. Dry run (synthesises and lists the uploads, no AWS calls):

   ```bash
   python scripts/publish.py --bucket-basename <basename> --prefix world-model-inference/v1 \
     --region us-east-1 --region us-west-2 --dry-run
   ```

3. Publish, then paste the printed rows into the README's Launch Stack table:

   ```bash
   python scripts/publish.py --bucket-basename <basename> --prefix world-model-inference/v1 \
     --region us-east-1 --region us-west-2
   ```

Use a new prefix per release (`.../v1`, `.../v2`): assets are content-addressed, but
the parent template's key is fixed, and a stack being updated or rolled back still
needs the assets it was created from.

The script builds `frontend/`, copies `inference/` without caches, generates the
catalogue (`scripts/launch_buildspecs.py`), and synthesises `cdk/bin/launch.ts`
with the CDK default synthesizer pointed at the publish bucket, so no account is
bootstrapped. To synthesise by hand, see the header of `cdk/bin/launch.ts`.
