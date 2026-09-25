#!/usr/bin/env bash
set -euo pipefail

# deploy.sh — one-command world model deployment.
#
# Builds the container image via CodeBuild (no local Docker needed),
# pushes to ECR, then deploys infrastructure via CDK.
#
# Usage:
#   ./deploy.sh                           # deploy lingbot-fast on EC2 (default)
#   ./deploy.sh cosmos3-nano              # deploy cosmos3-nano on EC2
#   ./deploy.sh cosmos3-nano sagemaker    # deploy to SageMaker
#   ./deploy.sh status                    # show deployments + endpoint URLs
#   ./deploy.sh destroy [model]           # tear down a deployment
#   ./deploy.sh ui [model]               # wire endpoint + launch frontend
#   ./deploy.sh token                     # print a Cognito access token

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
CDK_DIR="$REPO_ROOT/cdk"
FRONTEND_DIR="$REPO_ROOT/frontend"
DEFAULT_MODEL="lingbot-fast"

usage() {
  cat <<EOF
Usage: ./deploy.sh [model] [target]    Deploy a world model (build + infra)
       ./deploy.sh status              Show deployments + endpoint URLs
       ./deploy.sh destroy [model]     Tear down a deployment
       ./deploy.sh ui [model]          Wire endpoint URL + launch frontend
       ./deploy.sh bench [model] [s]   Benchmark a live real-time endpoint
       ./deploy.sh list                List available models
       ./deploy.sh build [model]       Build container image only (no deploy)
       ./deploy.sh token               Print a Cognito access token (expires in 1h)
       ./deploy.sh --help              Show this help

Arguments:
  model   Directory name under inference/models/ (default: $DEFAULT_MODEL)
  target  ec2 | sagemaker (default: ec2)

Examples:
  ./deploy.sh                          # lingbot-fast on EC2
  ./deploy.sh cosmos3-nano             # cosmos3-nano on EC2
  ./deploy.sh cosmos3-nano sagemaker   # cosmos3-nano on SageMaker
  ./deploy.sh destroy cosmos3-nano     # tear down
  ./deploy.sh ui lingbot-fast          # launch UI wired to lingbot-fast endpoint
  ./deploy.sh bench waypoint-1-5 60    # 60s performance report → docs/evidence/
EOF
}

get_region() {
  aws configure get region 2>/dev/null || echo "us-east-1"
}

get_account() {
  aws sts get-caller-identity --query Account --output text 2>/dev/null
}

# --- Authentication ----------------------------------------------------------
#
# The endpoint used to authenticate with a shared static bearer token that this
# script generated into SSM and the app string-compared. That is a bespoke
# credential scheme, which AWS security review classes as custom authentication
# ("a Cognito alternative") — it makes a reusable solution ineligible for an
# Enhanced DSR and forces a full AppSec review. It was also one long-lived secret
# shared by every caller, with no expiry and no per-client revocation.
#
# The foundation stack now provisions a Cognito user pool and publishes its
# non-secret identifiers to SSM. Nothing here is a secret, so unlike the old token
# these can be read with plain get-parameter and passed as CDK context.

COGNITO_POOL_PARAM="/world-model/foundation/cognito-user-pool-id"
COGNITO_CLIENTS_PARAM="/world-model/foundation/cognito-client-ids"
COGNITO_SCOPE_PARAM="/world-model/foundation/cognito-scope"
COGNITO_DOMAIN_PARAM="/world-model/foundation/cognito-domain"
COGNITO_MACHINE_CLIENT_PARAM="/world-model/foundation/cognito-machine-client-id"

# Read one foundation SSM parameter, empty if absent.
read_foundation_param() {
  aws ssm get-parameter --name "$1" --region "$(get_region)" \
    --query Parameter.Value --output text 2>/dev/null || true
}

# Mint a machine access token via the client-credentials grant.
#
# This is how `bench` and CI authenticate: no user, no interactive login, and no
# shared human-held secret. The client secret is read from the Cognito API rather
# than stored in SSM, so there is no plaintext copy of it anywhere.
#
# Printed on stdout. Callers MUST NOT pass it as a command-line argument — it
# would be visible in `ps` to every user on the box; use an env var.
mint_machine_token() {
  local region pool client_id domain secret
  region=$(get_region)
  pool=$(read_foundation_param "$COGNITO_POOL_PARAM")
  client_id=$(read_foundation_param "$COGNITO_MACHINE_CLIENT_PARAM")
  domain=$(read_foundation_param "$COGNITO_DOMAIN_PARAM")
  scope=$(read_foundation_param "$COGNITO_SCOPE_PARAM")
  if [ -z "$pool" ] || [ -z "$client_id" ] || [ -z "$domain" ]; then
    echo "Cognito is not provisioned — run './deploy.sh deploy ...' first." >&2
    return 1
  fi
  secret=$(aws cognito-idp describe-user-pool-client \
    --user-pool-id "$pool" --client-id "$client_id" --region "$region" \
    --query 'UserPoolClient.ClientSecret' --output text 2>/dev/null) || true
  if [ -z "$secret" ] || [ "$secret" = "None" ]; then
    echo "Could not read the machine client secret from Cognito." >&2
    return 1
  fi
  curl -sS -X POST "${domain}/oauth2/token" \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -u "${client_id}:${secret}" \
    -d "grant_type=client_credentials&scope=${scope}" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])'
}

ensure_cdk() {
  if ! command -v npx &>/dev/null; then
    echo "Error: node/npm required. Install from https://nodejs.org" >&2
    exit 1
  fi
  if [ ! -d "$CDK_DIR/node_modules" ]; then
    echo "Installing CDK dependencies..."
    (cd "$CDK_DIR" && npm install --silent)
  fi
}

ensure_foundation() {
  local region
  region=$(get_region)

  ensure_cdk

  # Always run `cdk deploy` for the foundation stack, even when it already
  # exists. CDK is idempotent — it's a no-op when there's no diff — and running
  # it every time ensures changes to the foundation (IAM scoping, the shared
  # API-token SSM read, CodeBuild, etc.) actually reach an existing deployment.
  # Skipping on *_COMPLETE, as we used to, silently stranded those changes.
  echo "Deploying/updating foundation stack (VPC, roles, CodeBuild)..."

  # The artifacts/outputs buckets are RETAIN'd, so they can outlive the stack
  # (prior deploy, or a destroy→redeploy). CDK can't detect that at synth time:
  # creating an existing bucket throws BucketAlreadyExists, importing a missing
  # one fails at deploy. Probe here and tell the stack which path to take.
  local acct import_buckets=false
  acct=$(aws sts get-caller-identity --query Account --output text --region "$region" 2>/dev/null || true)
  if [ -n "$acct" ] && aws s3api head-bucket \
      --bucket "world-model-artifacts-${acct}-${region}" --region "$region" 2>/dev/null; then
    import_buckets=true
  fi

  # ALB ingress CIDR. Default to the deployer's own public IP (/32) so the
  # endpoint is reachable for you but not world-open. Override with
  # ALLOWED_CIDR (e.g. your office range, or 0.0.0.0/0 to expose publicly).
  local allowed_cidr="${ALLOWED_CIDR:-}"
  if [ -z "$allowed_cidr" ]; then
    local myip
    myip=$(curl -s --max-time 5 https://checkip.amazonaws.com 2>/dev/null | tr -d '[:space:]' || true)
    if [ -n "$myip" ]; then
      allowed_cidr="${myip}/32"
      echo "  ALB ingress restricted to your public IP ($allowed_cidr)."
      echo "  Override with ALLOWED_CIDR=<cidr> (0.0.0.0/0 to expose publicly)."
    else
      allowed_cidr="127.0.0.1/32"
      echo "  ⚠  Could not detect your public IP — ALB ingress locked to $allowed_cidr." >&2
      echo "     Set ALLOWED_CIDR=<your-cidr> to reach the endpoint." >&2
    fi
  else
    echo "  ALB ingress restricted to $allowed_cidr (from ALLOWED_CIDR)."
  fi

  (cd "$CDK_DIR" && npx cdk deploy WorldModelFoundation --require-approval never \
    -c "importExistingBuckets=$import_buckets" \
    -c "allowedCidr=$allowed_cidr")
}

get_version() {
  local model="$1"
  grep "^version:" "$REPO_ROOT/inference/models/$model/endpoint.yaml" 2>/dev/null | awk '{print $2}' || echo "latest"
}

# Read a top-level scalar field from a model's endpoint.yaml.
#
# Takes everything after the first colon, not just the first whitespace-delimited
# word: `awk '{print $2}'` turned `license: CC BY-NC-SA 4.0` into `CC`, so the
# licence notice for the one cartridge that most needed it said nothing useful.
get_manifest_field() {
  local model="$1" field="$2"
  grep "^${field}:" "$REPO_ROOT/inference/models/$model/endpoint.yaml" 2>/dev/null \
    | head -1 | cut -d: -f2- | sed -e 's/[[:space:]]\{1,\}#.*$//' -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'
}

# Models with a restricted weights license require explicit acknowledgement
# before deploy. Set LYRA_LICENSE_ACK=1 (or ACCEPT_LICENSE=1) to skip the prompt
# in automation.
check_license() {
  local model="$1"
  [ "$(get_manifest_field "$model" license_restricted)" = "true" ] || return 0

  local name url note
  name=$(get_manifest_field "$model" license)
  url=$(get_manifest_field "$model" license_url)
  # Per-model restriction, because the restrictions genuinely differ: lyra-2 is
  # research-only, LingBot V2 is non-commercial + ShareAlike. The old text asserted
  # "internal R&D only" for every restricted model, which was accurate for exactly
  # one of them and misleading for the rest.
  note=$(get_manifest_field "$model" license_note)
  echo "⚠️  License notice for '$model'" >&2
  echo "    The model weights are under a restricted license: ${name:-restricted}" >&2
  echo "    ${url:-see the endpoint.yaml for this model}" >&2
  echo "    ${note:-Read the license before deploying — it restricts how you may use these weights.}" >&2
  echo "    You are responsible for compliant use in your own account." >&2
  if [ "${ACCEPT_LICENSE:-${LYRA_LICENSE_ACK:-0}}" = "1" ]; then
    echo "    (acknowledged via ACCEPT_LICENSE=1)" >&2
    return 0
  fi
  printf "    Continue? [y/N] " >&2
  read -r reply
  case "$reply" in
    y|Y|yes|YES) return 0 ;;
    *) echo "Aborted: license not acknowledged." >&2; exit 1 ;;
  esac
}

image_exists() {
  local model="$1"
  local region
  region=$(get_region)
  local version
  version=$(get_version "$model")
  local repo="world-model-${model}"

  aws ecr describe-images --repository-name "$repo" --image-ids "imageTag=$version" \
    --region "$region" &>/dev/null
}

cmd_build() {
  local model="${1:-$DEFAULT_MODEL}"
  local region
  region=$(get_region)

  if [ ! -d "$REPO_ROOT/inference/models/$model" ]; then
    echo "Error: no such model directory: inference/models/$model" >&2
    echo "Run './deploy.sh list' to see available models." >&2
    exit 1
  fi
  # A per-model Dockerfile is optional — build-image.py falls back to the shared
  # inference/Dockerfile.default (parameterised via endpoint.yaml).

  ensure_foundation

  echo "Building container image for $model via CodeBuild..."
  echo "  (no local Docker needed — builds in the cloud)"
  echo ""

  python3 "$REPO_ROOT/scripts/build-image.py" "$model" --region "$region"
}

get_endpoint_url() {
  local model="${1:-$DEFAULT_MODEL}"
  local region
  region=$(get_region)

  # Try CloudFormation output (ALB URL). CDK appends a hash to output keys
  # (e.g. EC2EndpointUrl08A2C9CE), so match on the prefix — an exact match
  # silently missed it and fell through to the private IP below.
  local alb_url
  alb_url=$(aws cloudformation describe-stacks --stack-name "WorldModel-$model" \
    --query "Stacks[0].Outputs[?starts_with(OutputKey,'EC2EndpointUrl')].OutputValue | [0]" \
    --output text --region "$region" 2>/dev/null || true)

  if [ -n "$alb_url" ] && [ "$alb_url" != "None" ]; then
    echo "$alb_url"
    return 0
  fi

  # Fallback: try EC2 instance private IP
  local ip
  ip=$(aws ec2 describe-instances \
    --filters "Name=tag:Model,Values=$model" "Name=instance-state-name,Values=running" \
    --query "Reservations[0].Instances[0].PrivateIpAddress" \
    --output text --region "$region" 2>/dev/null || true)

  if [ -n "$ip" ] && [ "$ip" != "None" ] && [ "$ip" != "null" ]; then
    echo "http://$ip:8080"
    return 0
  fi

  # Try SageMaker endpoint
  local sm_name="world-model-${model}"
  local sm_status
  sm_status=$(aws sagemaker describe-endpoint --endpoint-name "$sm_name" \
    --query "EndpointStatus" --output text --region "$region" 2>/dev/null || true)

  if [ "$sm_status" = "InService" ]; then
    echo "sagemaker://$sm_name (region: $region)"
    return 0
  fi

  return 1
}

cmd_list() {
  echo "Available models:"
  for dir in "$REPO_ROOT"/inference/models/*/; do
    id=$(basename "$dir")
    # Skip scaffolds like _template — they aren't deployable.
    case "$id" in _*) continue ;; esac
    if [ -f "$dir/endpoint.yaml" ]; then
      # Every model builds a container; note which ones use a custom Dockerfile
      # versus the shared inference/Dockerfile.default.
      local build_kind=" [shared Dockerfile]"
      [ -f "$dir/Dockerfile" ] && build_kind=" [custom Dockerfile]"
      echo "  $id$build_kind"
    fi
  done
}

cmd_status() {
  local region
  region=$(get_region)
  echo "Active deployments:"
  echo ""

  for dir in "$REPO_ROOT"/inference/models/*/; do
    id=$(basename "$dir")
    if [ -f "$dir/endpoint.yaml" ]; then
      local url
      url=$(get_endpoint_url "$id" 2>/dev/null || true)
      if [ -n "$url" ]; then
        echo "  ✓ $id  →  $url"
      fi
    fi
  done

  echo ""
  echo "CloudFormation stacks:"
  aws cloudformation list-stacks \
    --stack-status-filter CREATE_COMPLETE UPDATE_COMPLETE \
    --query "StackSummaries[?starts_with(StackName,'WorldModel')].{Name:StackName,Status:StackStatus}" \
    --output table --region "$region" 2>/dev/null || echo "  (none)"
}

cmd_destroy() {
  local model="${1:-$DEFAULT_MODEL}"
  ensure_cdk
  echo "Destroying WorldModel-$model..."
  (cd "$CDK_DIR" && npx cdk destroy "WorldModel-$model" --force)
  echo "✓ WorldModel-$model destroyed"
}

cmd_ui() {
  local model="${1:-$DEFAULT_MODEL}"
  local region
  region=$(get_region)

  # Find the endpoint URL
  local url
  url=$(get_endpoint_url "$model" 2>/dev/null || true)

  if [ -z "$url" ]; then
    echo "Warning: no running endpoint found for '$model'."
    echo "The UI will launch in demo mode (no live inference)."
    echo "Deploy first with: ./deploy.sh $model"
    echo ""
    url=""
  else
    echo "Found endpoint: $url"
  fi

  # Write config.js with the endpoint URL
  if [ ! -d "$FRONTEND_DIR/public" ]; then
    echo "Error: frontend/public/ directory not found. Run from the repo root." >&2
    exit 1
  fi

  # Mint a short-lived Cognito access token (client-credentials) so the vite dev
  # proxy can authenticate against a secured endpoint. The token is passed to vite
  # as an env var and injected SERVER-SIDE by the proxy — it is NOT written into
  # config.js, so it never reaches the browser bundle. That property is why the
  # proxy exists, and it is preserved here.
  #
  # The improvement over the old path is the credential: this expires in an hour
  # and is revocable per client, where the previous value was a permanent shared
  # secret read straight out of SSM.
  local api_token=""
  if [ -n "$url" ]; then
    api_token=$(mint_machine_token 2>/dev/null || true)
    if [ -z "$api_token" ]; then
      echo "  ⚠  Could not mint a Cognito token; the proxy will call the endpoint" >&2
      echo "     unauthenticated and a secured endpoint will answer 401." >&2
    fi
  fi

  # With a live endpoint the UI talks to its OWN origin (empty lingbotApiUrl and
  # websocketUrl → relative paths like /generate and /ws), which the vite proxy
  # forwards to the ALB. This sidesteps CORS entirely (browser only ever calls
  # localhost) and keeps the token server-side. ui:'lingbot' renders the World
  # Foundry catalogue, which plays async and real-time cartridges alike.
  local ui_mode="worlds" demo_mode="true"
  if [ -n "$url" ]; then
    ui_mode="lingbot"
    demo_mode="false"
  fi

  # deployedModel names which cartridge this proxy actually serves. The catalogue
  # probes /health and shows a live badge on that one cartridge only — every other
  # card reports its build state, not a deployment it doesn't have.
  local deployed_model=""
  [ -n "$url" ] && deployed_model="$model"

  # Cognito identifiers for the browser sign-in path. These are public values —
  # a pool ID, a public app client ID and a hosted-UI domain — and were empty
  # strings until now, which is why the frontend's Cognito code was dead.
  local cognito_pool_id cognito_browser_client cognito_domain
  cognito_pool_id=$(read_foundation_param "$COGNITO_POOL_PARAM")
  cognito_browser_client=$(read_foundation_param "/world-model/foundation/cognito-browser-client-id")
  cognito_domain=$(read_foundation_param "$COGNITO_DOMAIN_PARAM")

  cat > "$FRONTEND_DIR/public/config.js" <<CONF
window.APP_CONFIG = {
  region: '$region',
  userPoolId: '$cognito_pool_id',
  userPoolClientId: '$cognito_browser_client',
  websocketUrl: '',
  lingbotApiUrl: '',
  environment: 'local',
  cognitoDomain: '$cognito_domain',
  redirectSignIn: 'http://localhost:3000',
  redirectSignOut: 'http://localhost:3000',
  demoMode: $demo_mode,
  deployedModel: '$deployed_model',
  ui: '$ui_mode',
};
CONF

  # Never echo the token itself — only whether one was found.
  local auth_state="none"
  [ -n "$api_token" ] && auth_state="enabled"
  echo "Wrote frontend/public/config.js (endpoint: ${url:-none}, auth: $auth_state)"
  echo ""

  # Install and start
  if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
    echo "Installing frontend dependencies..."
    (cd "$FRONTEND_DIR" && npm install --silent)
  fi

  if [ -n "$url" ]; then
    echo "Starting UI on http://localhost:3000 → proxying to $url (token injected server-side)"
    echo "  (if 3000 is taken vite picks the next free port — use the URL it prints below)"
    (cd "$FRONTEND_DIR" && WM_PROXY_TARGET="$url" WM_PROXY_TOKEN="$api_token" npm run dev)
  else
    echo "Starting UI at http://localhost:3000 (demo mode — no live endpoint)"
    (cd "$FRONTEND_DIR" && npm run dev)
  fi
}

# Benchmark a deployed real-time endpoint and write a performance report.
# Separate command rather than automatic on deploy: it holds the single session
# for its whole duration, so running it unasked would lock out a player.
cmd_bench() {
  local model="${1:-$DEFAULT_MODEL}"
  local seconds="${2:-60}"
  local region
  region=$(get_region)

  local url
  url=$(get_endpoint_url "$model" 2>/dev/null || true)
  if [ -z "$url" ]; then
    echo "Error: no running endpoint for '$model'. Deploy it first: ./deploy.sh $model" >&2
    exit 1
  fi

  # http://host → ws://host/ws (the ALB has no TLS unless CERTIFICATE_ARN was set).
  local ws_url="${url/http:\/\//ws://}"
  ws_url="${ws_url/https:\/\//wss://}"
  ws_url="${ws_url}/ws"

  local token=""
  token=$(aws ssm get-parameter --name "$API_TOKEN_PARAM" --with-decryption \
    --query Parameter.Value --output text --region "$region" 2>/dev/null || true)

  # Seed from a real scene so the numbers describe a session a player would have;
  # without one the model starts from its own prior, which is also valid but
  # makes runs less comparable.
  local seed="$FRONTEND_DIR/public/scenes/mountain-highway.png"
  [ -f "$seed" ] || seed=""
  local out_dir="$REPO_ROOT/docs/evidence/$model"
  mkdir -p "$out_dir"

  echo "Benchmarking $model for ${seconds}s → $ws_url"
  echo "  (holds the single session; a player will be refused until it finishes)"
  python3 "$REPO_ROOT/scripts/benchmark.py" \
    --url "$ws_url" --token "$token" \
    ${seed:+--seed "$seed"} \
    --seconds "$seconds" \
    --out "$out_dir/performance-report.md" \
    --json "$out_dir/performance-report.json"
}

cmd_deploy() {
  local model="${1:-$DEFAULT_MODEL}"
  local target="${2:-ec2}"
  local region
  region=$(get_region)

  if [ ! -d "$REPO_ROOT/inference/models/$model" ]; then
    echo "Error: model '$model' not found under inference/models/" >&2
    echo "Run './deploy.sh list' to see available models." >&2
    exit 1
  fi

  # Restricted-license models require acknowledgement before deploy.
  check_license "$model"

  echo "═══════════════════════════════════════════════════"
  echo "  Deploying: $model → $target"
  echo "  Region:    $region"
  echo "═══════════════════════════════════════════════════"
  echo ""

  # Step 1: Build the container image if needed.
  #
  # EVERY model needs an image: the instance's boot script does an unconditional
  # `docker pull`. Previously this block was gated on the model having its OWN
  # Dockerfile, so a model without one (e.g. lingbot-fast) skipped the build and
  # went straight to launching a GPU instance that could never serve. Models
  # without a Dockerfile now build from the shared inference/Dockerfile.default.
  #
  # We reuse an existing image tag to save a slow CodeBuild run, but that means
  # changes to the inference code (e.g. the security middleware) do NOT reach a
  # redeploy of an already-built tag. Set REBUILD_IMAGE=1 (or bump the version in
  # endpoint.yaml) to force a fresh build.
  if [ "${REBUILD_IMAGE:-0}" != "1" ] && image_exists "$model"; then
    echo "✓ Container image already exists in ECR — skipping build"
    echo "  (set REBUILD_IMAGE=1 to force a rebuild after changing inference code)"
  else
    echo "Step 1/2: Building container image via CodeBuild..."
    cmd_build "$model"
  fi
  echo ""

  # Fail-fast guard: never launch an instance without an image. GPU instances
  # cost up to ~$55/hr, so a missing image must abort here rather than boot a
  # box that fails its `docker pull` and serves nothing.
  if ! image_exists "$model"; then
    echo "Error: no container image in ECR for '$model' after the build step." >&2
    echo "       Refusing to launch a GPU instance that cannot serve." >&2
    echo "       Check the CodeBuild logs above, then retry:" >&2
    echo "         ./deploy.sh build $model" >&2
    exit 1
  fi

  # Step 2: Deploy infrastructure via CDK
  echo "Step 2/2: Deploying infrastructure via CDK..."
  ensure_cdk

  # Foundation stack (idempotent)
  ensure_foundation

  # Authentication: read the Cognito identifiers the foundation stack published.
  # These are public identifiers, not secrets, so they go in CDK context directly.
  local cognito_pool cognito_clients cognito_scope
  cognito_pool=$(read_foundation_param "$COGNITO_POOL_PARAM")
  cognito_clients=$(read_foundation_param "$COGNITO_CLIENTS_PARAM")
  cognito_scope=$(read_foundation_param "$COGNITO_SCOPE_PARAM")

  local ctx_args=(
    -c "model=$model"
    -c "target=$target"
  )

  if [ -n "$cognito_pool" ] && [ -n "$cognito_clients" ]; then
    ctx_args+=(
      -c "authMode=cognito"
      -c "cognitoUserPoolId=$cognito_pool"
      -c "cognitoClientIds=$cognito_clients"
    )
    [ -n "$cognito_scope" ] && ctx_args+=(-c "cognitoScope=$cognito_scope")
    echo "  Auth: Cognito user pool $cognito_pool" >&2
  elif [ "${WORLD_MODEL_AUTH_MODE:-}" = "disabled" ]; then
    # Explicit, and it has to stay explicit. The container's own default is to
    # refuse to start without Cognito, so this is the only way to get an
    # unauthenticated endpoint and it takes a deliberate env var to do it.
    ctx_args+=(-c "authMode=disabled")
    echo "  ⚠  Auth DISABLED (WORLD_MODEL_AUTH_MODE=disabled). Do not expose this." >&2
  else
    echo "ERROR: Cognito is not provisioned, so this deploy would produce an" >&2
    echo "       endpoint the container will refuse to serve. The foundation stack" >&2
    echo "       creates the user pool — re-run after it has deployed, or set" >&2
    echo "       WORLD_MODEL_AUTH_MODE=disabled for local/offline use only." >&2
    exit 1
  fi
  [ -n "${WORLD_MODEL_ALLOWED_ORIGINS:-}" ] && ctx_args+=(-c "allowedOrigins=$WORLD_MODEL_ALLOWED_ORIGINS")
  [ -n "${WORLD_MODEL_RATE_LIMIT:-}" ] && ctx_args+=(-c "rateLimit=$WORLD_MODEL_RATE_LIMIT")
  [ -n "${CERTIFICATE_ARN:-}" ] && ctx_args+=(-c "certificateArn=$CERTIFICATE_ARN")
  if [ -n "${CAPACITY_BLOCK_ID:-}" ]; then
    # Fail fast on a dead reservation. A Capacity Block that has expired (or a
    # typo'd ID) still deploys cleanly, but then EVERY instance launch fails with
    # "Capacity Reservation not valid" and the ASG retries forever with no
    # endpoint — a silent outage. Catch it here instead.
    local cr_state
    cr_state=$(aws ec2 describe-capacity-reservations \
      --capacity-reservation-ids "$CAPACITY_BLOCK_ID" \
      --query "CapacityReservations[0].State" --output text --region "$region" 2>/dev/null || true)
    if [ "$cr_state" != "active" ]; then
      echo "Error: Capacity Reservation $CAPACITY_BLOCK_ID is not usable (state: ${cr_state:-not found})." >&2
      echo "       An expired or missing Capacity Block deploys fine but every instance" >&2
      echo "       launch then fails, leaving the ASG stuck at 0 instances." >&2
      echo "       Either purchase a new Capacity Block and pass its ID, or unset" >&2
      echo "       CAPACITY_BLOCK_ID to launch regular on-demand capacity." >&2
      exit 1
    fi
    ctx_args+=(-c "capacity=$CAPACITY_BLOCK_ID")
  else
    echo "  No CAPACITY_BLOCK_ID set — launching regular on-demand capacity." >&2
    echo "  (Redeploying without it also clears a previously pinned reservation.)" >&2
  fi
  # A Capacity Block is pinned to one AZ, so the ASG must be pinned to the
  # private subnet in that AZ or it will launch in the wrong one and fail.
  [ -n "${CAPACITY_SUBNET_IDS:-}" ] && ctx_args+=(-c "capacitySubnets=$CAPACITY_SUBNET_IDS")
  [ -n "${INSTANCE_TYPE:-}" ] && ctx_args+=(-c "instanceType=$INSTANCE_TYPE")

  if [ -z "${CERTIFICATE_ARN:-}" ] && [ "$target" = "ec2" ]; then
    echo "  ⚠  No CERTIFICATE_ARN set — the ALB will serve plain HTTP. Set CERTIFICATE_ARN" >&2
    echo "     to an ACM cert ARN to enable HTTPS before exposing this to untrusted networks." >&2
  fi

  # Model stack
  (cd "$CDK_DIR" && npx cdk deploy "WorldModel-$model" \
    "${ctx_args[@]}" \
    --require-approval never)

  # The ASG is deployed with a floor of 0 so CloudFormation never blocks a
  # deploy on GPU availability (see cdk/lib/constructs/ec2.ts). Ask for the one
  # instance here instead: set-desired-capacity returns immediately and the ASG
  # retries the launch in the background until capacity appears.
  if [ "$target" = "ec2" ]; then
    aws autoscaling set-desired-capacity \
      --auto-scaling-group-name "world-model-$model" \
      --desired-capacity 1 --region "$region" 2>/dev/null \
      && echo "  ASG world-model-$model set to 1 instance (launches when GPU capacity is available)." \
      || echo "  ⚠  Could not set ASG desired capacity — run: aws autoscaling set-desired-capacity --auto-scaling-group-name world-model-$model --desired-capacity 1" >&2
  fi

  echo ""
  echo "═══════════════════════════════════════════════════"
  echo "  ✓ $model deployed via $target"

  # Show the endpoint URL (may take a moment for instance to get IP)
  sleep 5
  local url
  url=$(get_endpoint_url "$model" 2>/dev/null || true)
  if [ -n "$url" ]; then
    echo "  Endpoint: $url"
    echo ""
    echo "  Next steps:"
    echo "    ./deploy.sh ui $model     # launch UI connected to this endpoint"
    echo "    ./deploy.sh destroy $model # tear down when done"
  else
    echo "  Instance launching... endpoint will be available shortly."
    echo "  Run './deploy.sh status' to check."
  fi
  echo "═══════════════════════════════════════════════════"
}

# Print a fresh Cognito access token for manual calls:
#   curl -H "Authorization: Bearer $(./deploy.sh token)" http://<alb>/health
#
# Uses the client-credentials grant, so it needs no user and no stored secret on
# your side. The token expires within the hour.
cmd_token() {
  mint_machine_token
}

# --- Main ---

case "${1:-}" in
  -h|--help)  usage ;;
  list)       cmd_list ;;
  status)     cmd_status ;;
  build)      shift; cmd_build "$@" ;;
  destroy)    shift; cmd_destroy "$@" ;;
  ui)         shift; cmd_ui "$@" ;;
  bench)      shift; cmd_bench "$@" ;;
  token)      cmd_token ;;
  "")         cmd_deploy "$DEFAULT_MODEL" "ec2" ;;
  *)          cmd_deploy "$@" ;;
esac
