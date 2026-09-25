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

WebSocket clients may pass the token as a query parameter, since the browser WebSocket
API cannot set headers:

```
wss://<alb-dns>/ws?token=<access-token>
```

Prefer the `Authorization` header where your client can set one — a query-string token
lands in ALB access logs. Access tokens expire within the hour, which bounds that
exposure; the shared token this replaced never expired at all.

Auth is enforced *before* the socket is accepted; failure closes with code 1008.

`/ping` and `/health` are deliberately open so the ALB health check works without
credentials.

## Before production

### 1. Enable TLS

The single most important change. Without it access tokens cross the network in
plaintext, where anyone on the path can replay one until it expires.

```bash
CERTIFICATE_ARN=arn:aws:acm:...:certificate/... ./deploy.sh <model>
```

The ALB then serves HTTPS on 443 with the `ELBSecurityPolicy-TLS13-1-2-2021-06`
policy and redirects HTTP 80 → 443. See
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
- **Tokens in WebSocket query strings reach ALB access logs.** Unavoidable for browser
  clients, which cannot set headers. Mitigated by one-hour expiry; use the
  `Authorization` header from any client that can.
- **The catalogue UI is local-only.** `./deploy.sh ui` runs vite on `:3000` and injects
  the token server-side in the dev proxy, so the browser only talks to localhost and
  the token never reaches the JS bundle. There is no hosted UI; a CloudFront-hosted SPA
  would use the browser client's authorization-code flow instead, which the pool is
  already configured for.
- **Rate limiting is per-instance and in-memory.** It resets on restart and is not
  shared across instances.
- **Job state is in-memory.** Job IDs are unguessable UUIDs, but any client with the
  token can fetch any job's output — there is no per-caller ownership check.
- **The vite dev proxy sets `secure: false`**, so it will talk to an ALB with an
  invalid or absent certificate. That is intentional for local development against a
  plain-HTTP ALB; do not reuse that setting in production client code.
