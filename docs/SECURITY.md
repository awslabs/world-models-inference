# Security

What the default deployment does and does not protect, and what to change before
exposing an endpoint to untrusted networks.

To report a vulnerability, see [CONTRIBUTING.md](../CONTRIBUTING.md).

## What you get by default

`./deploy.sh` is authenticated and network-restricted out of the box:

| Control | Default |
|---|---|
| **Bearer token** | Auto-generated on first deploy (32 bytes URL-safe), stored as an SSM `SecureString` at `/world-model/security/api-token` |
| **ALB ingress** | Restricted to the deployer's public IP `/32`, detected via `checkip.amazonaws.com` |
| **GPU instance** | Private subnet, no public IP. Reachable only through the ALB (and SSM Session Manager for admin) |
| **Rate limiting** | 60 requests/min per client IP on `/generate` and `/invocations` |
| **CORS** | Deny-by-default — no origins allowed unless you set an allowlist |
| **Security headers** | `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY` |
| **Transport** | ⚠️ **Plain HTTP** unless you supply an ACM certificate |
| **S3 buckets** | Block-all-public-access, SSE-S3 encryption, TLS enforced, access-logged |
| **IAM** | Three scoped roles (EC2, SageMaker, CodeBuild) — no `*FullAccess` managed policies |

The token value never enters CDK context or the CloudFormation template. On EC2 the
instance reads it from SSM at boot; on SageMaker it is injected via a
`{{resolve:ssm-secure:...}}` dynamic reference.

## Using the token

```bash
TOKEN=$(aws ssm get-parameter --name /world-model/security/api-token \
  --with-decryption --query Parameter.Value --output text)

curl -H "Authorization: Bearer $TOKEN" http://<alb-dns>/health
```

WebSocket clients pass it as a query parameter, since the WebSocket handshake has no
Authorization header:

```
wss://<alb-dns>/ws?token=<token>
```

Auth is enforced *before* the socket is accepted; failure closes with code 1008.

`/ping` and `/health` are deliberately open so the ALB health check works without
credentials.

## Before production

### 1. Enable TLS

The single most important change. Without it the bearer token crosses the network in
plaintext.

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

### 3. Consider stronger auth

A single shared bearer token has no per-user identity, no expiry and no revocation
short of rotating it for everyone. For anything multi-tenant or long-lived, put
[Cognito authentication on the ALB listener](https://docs.aws.amazon.com/elasticloadbalancing/latest/application/listener-authenticate-users.html)
or front the endpoint with an API gateway that issues scoped keys.

### 4. Tune CORS and rate limits

```bash
WORLD_MODEL_ALLOWED_ORIGINS=https://app.example.com \
WORLD_MODEL_RATE_LIMIT=120 \
  ./deploy.sh <model>
```

`WORLD_MODEL_RATE_LIMIT=0` disables rate limiting entirely.

## Rotating the token

```bash
aws ssm put-parameter --name /world-model/security/api-token \
  --type SecureString --value "$(python3 -c 'import secrets;print(secrets.token_urlsafe(32))')" \
  --overwrite
```

The instance reads the token **at boot**, so redeploy or restart the container for a
new value to take effect.

## Known limitations

Things to be aware of rather than surprised by:

- **If `WORLD_MODEL_API_TOKEN` is unset, the API is completely unauthenticated.** The
  server logs a warning on startup. `deploy.sh` always provisions a token, so this
  only happens if you run the container yourself without one.
- **The catalogue UI is local-only.** `./deploy.sh ui` runs vite on `:3000` and injects
  the token server-side in the dev proxy, so the browser only talks to localhost and
  the token never reaches the JS bundle. There is no hosted UI; a CloudFront-hosted SPA
  would need a different auth model.
- **Rate limiting is per-instance and in-memory.** It resets on restart and is not
  shared across instances.
- **Job state is in-memory.** Job IDs are unguessable UUIDs, but any client with the
  token can fetch any job's output — there is no per-caller ownership check.
- **The vite dev proxy sets `secure: false`**, so it will talk to an ALB with an
  invalid or absent certificate. That is intentional for local development against a
  plain-HTTP ALB; do not reuse that setting in production client code.
