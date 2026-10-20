# Security

What the default deployment does and does not protect, and what to change before
exposing an endpoint to untrusted networks.

To report a vulnerability, see [CONTRIBUTING.md](../CONTRIBUTING.md).

## What you get by default

`./deploy.sh` is authenticated and network-restricted out of the box:

| Control | Default |
|---|---|
| **Authentication** | Amazon Cognito. A user pool is provisioned in the foundation stack; the server verifies pool-issued **access tokens**. Required by default — a container with no Cognito settings refuses to start |
| **ALB ingress** | Restricted to the deployer's public IP `/32`, detected via `checkip.amazonaws.com` |
| **GPU instance** | Private subnet, no public IP. Reachable only through the ALB (and SSM Session Manager for admin) |
| **Rate limiting** | 60 requests/min per client IP on `/generate` and `/invocations` |
| **CORS** | Deny-by-default — no origins allowed unless you set an allowlist |
| **Security headers** | `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY` |
| **Transport** | ⚠️ **Plain HTTP** unless you supply an ACM certificate |
| **S3 buckets** | Block-all-public-access, SSE-S3 encryption, TLS enforced, access-logged |
| **IAM** | Three scoped roles (EC2, SageMaker, CodeBuild) — no `*FullAccess` managed policies |

There is **no shared secret anywhere in the system**. The container receives only
public identifiers — user pool ID, app client IDs, scope name — so nothing needs
decrypting at boot and the EC2 role no longer has any `ssm:GetParameter` grant.

## Authenticating

Two grant types, both ending in the same access token:

**Automation** — client credentials, no user. This is what `./deploy.sh bench` uses:

```bash
TOKEN=$(./deploy.sh token)          # mints a fresh access token
curl -H "Authorization: Bearer $TOKEN" http://<alb-dns>/health
```

**Browsers** — authorization code + PKCE against the pool's hosted UI. `./deploy.sh ui`
does not need this: its dev proxy mints a machine token and injects the header
server-side, so no token ever reaches the browser bundle.

Send the **access token, not the ID token.** An ID token is signed by the same pool and
would otherwise pass every other check, so the server rejects it explicitly
(`token_use` must be `access`). It also requires the `world-model/invoke` scope, so a
token proves "may run inference" rather than merely "signed in somewhere".

WebSocket clients send the same `Authorization` header. The browser WebSocket API
cannot set headers, so a browser offers the token as a subprotocol instead, after a
fixed marker; the server selects the marker and never echoes the token:

```js
new WebSocket('wss://<alb-dns>/ws', ['world-model.bearer', accessToken])
```

A token is **never accepted in the URL** — on HTTP or WebSocket routes — because a
query string lands in ALB and proxy access logs, browser history and `Referer`
headers. Headers are not logged by the ALB.

Auth is enforced *before* the socket is accepted; failure closes with code 1008.

`/ping` and `/health` are deliberately open so the ALB health check works without
credentials.

## Before production

### 1. Keep TLS on

EC2 endpoints are HTTPS by default through CloudFront, so access tokens never
cross the client's network in plaintext. Never deploy a reachable endpoint with
`ALLOW_PLAINTEXT=1`: anyone on the path could replay a token until it expires.

The CloudFront-to-ALB hop is HTTP on AWS's network, guarded by a secret origin
header. To terminate TLS at the ALB under your own name instead:

```bash
CERTIFICATE_ARN=arn:aws:acm:...:certificate/... ENDPOINT_DOMAIN=wm.example.com \
HOSTED_ZONE_ID=Z... ./deploy.sh <model>
```

The ALB then serves HTTPS on 443 with the `ELBSecurityPolicy-TLS13-1-2-2021-06`
policy, redirects HTTP 80 → 443, and gets a Route 53 alias. The certificate must
be in the ALB's region and cover `ENDPOINT_DOMAIN`. See
[requesting an ACM certificate](https://docs.aws.amazon.com/acm/latest/userguide/gs-acm-request-public.html).

### 2. Set the ingress range deliberately

The default `/32` breaks as soon as your IP changes or a colleague tries the endpoint.

```bash
ALLOWED_CIDR=203.0.113.0/24 ./deploy.sh <model>    # e.g. an office range
ALLOWED_CIDR=0.0.0.0/0      ./deploy.sh <model>    # deliberately world-open
```

Do not use `0.0.0.0/0` without TLS and a token.

### 3. Create real operator accounts

The pool has self-sign-up disabled, so accounts are invited:

```bash
POOL=$(aws ssm get-parameter --name /world-model/foundation/cognito-user-pool-id \
  --query Parameter.Value --output text)
aws cognito-idp admin-create-user --user-pool-id "$POOL" --username you@example.com
```

Per-user identity means you can revoke one person without rotating a secret for
everyone — the thing the previous shared token could not do.

### 4. Tune CORS and rate limits

```bash
WORLD_MODEL_ALLOWED_ORIGINS=https://app.example.com \
WORLD_MODEL_RATE_LIMIT=120 \
  ./deploy.sh <model>
```

`WORLD_MODEL_RATE_LIMIT=0` disables rate limiting entirely.

### 5. Restrict who can change what a deployment runs

Three things in your account decide what runs on the GPU instances, so anyone who can
change them can run code there:

- **The parameters under `/world-model/foundation/*`.** Deployments read the Cognito
  settings, bucket names and network IDs from them.
- **The `world-model-weights` and `world-model-image` CodeBuild projects.** Whoever can
  start a build can override its buildspec, and the build runs as that project's role.
  The image role can push to every cartridge repository.
- **The two build roles** (`WorldModelWeightsBuildRole-<region>`,
  `WorldModelImageBuildRole-<region>`). Whoever can pass them can attach them to a build
  of their own.

The stacks validate everything they read, and the build roles are scoped to the
minimum each build needs. But both of those assume that only your deploy operators
can make these changes, and only your account can enforce that. The stacks can't do
it themselves: the principals involved are your account's own users and roles, SSM
parameters have no resource policies, and CodeBuild project policies only control
sharing with other accounts.

Apply a policy like the one below as a service control policy on the account, or as
a deny policy on every role and user except the operators'. Replace the two
principal patterns with:

- your deploy operators' roles, and
- the CloudFormation execution role that deploys the stacks. This is the CDK default
  shown below, unless you bootstrapped CDK with a different one.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "OnlyDeployersChangeFoundationParameters",
      "Effect": "Deny",
      "Action": [
        "ssm:PutParameter",
        "ssm:DeleteParameter",
        "ssm:DeleteParameters",
        "ssm:LabelParameterVersion"
      ],
      "Resource": "arn:aws:ssm:*:*:parameter/world-model/foundation/*",
      "Condition": {
        "ArnNotLike": {
          "aws:PrincipalArn": [
            "arn:aws:iam::*:role/cdk-*-cfn-exec-role-*",
            "arn:aws:iam::*:role/<your-deploy-operator-role>"
          ]
        }
      }
    },
    {
      "Sid": "OnlyDeployersStartOrChangeBuilds",
      "Effect": "Deny",
      "Action": [
        "codebuild:StartBuild",
        "codebuild:StartBuildBatch",
        "codebuild:RetryBuild",
        "codebuild:RetryBuildBatch",
        "codebuild:UpdateProject"
      ],
      "Resource": [
        "arn:aws:codebuild:*:*:project/world-model-weights",
        "arn:aws:codebuild:*:*:project/world-model-image"
      ],
      "Condition": {
        "ArnNotLike": {
          "aws:PrincipalArn": [
            "arn:aws:iam::*:role/cdk-*-cfn-exec-role-*",
            "arn:aws:iam::*:role/<your-deploy-operator-role>"
          ]
        }
      }
    },
    {
      "Sid": "OnlyCloudFormationPassesBuildRoles",
      "Effect": "Deny",
      "Action": "iam:PassRole",
      "Resource": [
        "arn:aws:iam::*:role/WorldModelWeightsBuildRole-*",
        "arn:aws:iam::*:role/WorldModelImageBuildRole-*"
      ],
      "Condition": {
        "ArnNotLike": {
          "aws:PrincipalArn": "arn:aws:iam::*:role/cdk-*-cfn-exec-role-*"
        }
      }
    }
  ]
}
```

Only CloudFormation needs to pass the build roles: it attaches them when it creates
the projects, and starting a build doesn't pass a role. `deploy.sh` and
`scripts/*.py` never write the foundation parameters. They do start builds, which is
why the operator role is listed in the second statement.

Check the result with the policy simulator. For a role that isn't an operator, each
of these should return `explicitDeny`:

```bash
sim() { aws iam simulate-principal-policy --policy-source-arn <some-other-role-arn> \
  --action-names "$1" --resource-arns "$2" --query 'EvaluationResults[0].EvalDecision' --output text; }
sim ssm:PutParameter     arn:aws:ssm:<region>:<account>:parameter/world-model/foundation/vpc-id
sim codebuild:StartBuild arn:aws:codebuild:<region>:<account>:project/world-model-image
sim iam:PassRole         arn:aws:iam::<account>:role/WorldModelImageBuildRole-<region>
```

This checks the deny-policy form. To check an SCP, make a real call as a
non-operator role instead. For example,
`aws ssm put-parameter --name /world-model/foundation/scp-check --value x --type String`
should fail with `AccessDenied`. If it succeeds, delete the parameter and fix the
policy.

An identity policy can't restrict a principal that holds `AdministratorAccess`,
because Admin can detach it. In accounts where everyone is an administrator, use the
SCP form, or treat the account as single-tenant.

## Revoking access

Tokens are short-lived (one hour) and revocation is enabled on both app clients, so
there is nothing to rotate and no redeploy needed:

```bash
# One user
aws cognito-idp admin-user-global-sign-out --user-pool-id "$POOL" --username you@example.com

# One machine client — rotate its secret
aws cognito-idp update-user-pool-client --user-pool-id "$POOL" \
  --client-id <machine-client-id> --generate-secret
```

Previously this section described regenerating a secret shared by every caller and
restarting the container. That is no longer necessary or possible.

## Known limitations

Things to be aware of rather than surprised by:

- **`WORLD_MODEL_AUTH_MODE=disabled` really does disable authentication.** It is the
  only way to get an unauthenticated endpoint, it must be set deliberately, and the
  server logs a warning on every startup. Without it a container missing its Cognito
  settings refuses to start — which is the point: the previous behaviour was that an
  unset token silently produced an open GPU endpoint.
- **Tokens never travel in a URL.** HTTP and WebSocket routes reject `?token=`, so an
  access token cannot land in ALB/proxy access logs or browser history; browsers
  send it as a WebSocket subprotocol instead (see [Authenticating](#authenticating)).
- **The catalogue UI is local-only.** `./deploy.sh ui` runs vite on `:3000` and injects
  the token server-side in the dev proxy, so the browser only talks to localhost and
  the token never reaches the JS bundle. There is no hosted UI; a CloudFront-hosted SPA
  would use the browser client's authorization-code flow instead, which the pool is
  already configured for.
- **Rate limiting is per-instance and in-memory.** It resets on restart and is not
  shared across instances.
- **Job state is in-memory, and jobs are scoped to their submitter.** A job's status
  and output are returned only to the principal that submitted it (the access
  token's `sub`); anyone else gets 404. The principal is the Cognito user for an
  authorization-code token and the app client for a client-credentials token — so
  every browser behind the `./deploy.sh ui` proxy shares the proxy's machine-client
  principal, and ownership separates distinct users only when each holds their own
  token.
- **Request limits.** `/invocations` takes only documented generation parameters,
  within fixed bounds, and bodies up to 8 MiB (`WORLD_MODEL_MAX_BODY_BYTES`). An input
  image arrives as `image_b64` or a bundled `example_id` and is staged by the server;
  a client never names a file path. `/generate` multipart uploads are not size-capped
  by the application.
- **The vite dev proxy sets `secure: false`**, so it will talk to an ALB with an
  invalid or absent certificate. That is intentional for local development against a
  plain-HTTP ALB; do not reuse that setting in production client code.
