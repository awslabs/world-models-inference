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
       ./deploy.sh list                List available models
       ./deploy.sh build [model]       Build container image only (no deploy)
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
EOF
}

get_region() {
  aws configure get region 2>/dev/null || echo "us-east-1"
}

get_account() {
  aws sts get-caller-identity --query Account --output text 2>/dev/null
}

# SSM parameter name holding the shared inference API token.
API_TOKEN_PARAM="/world-model/security/api-token"

# Ensure a shared API token exists in SSM (SecureString) and print its name.
# Auto-generates one on first deploy so the default path is authenticated
# without the operator having to invent a secret. Re-deploys reuse it. Set
# WORLD_MODEL_API_TOKEN in the environment to pin your own value instead.
ensure_api_token() {
  local region
  region=$(get_region)

  if ! aws ssm get-parameter --name "$API_TOKEN_PARAM" --region "$region" >/dev/null 2>&1; then
    local token="${WORLD_MODEL_API_TOKEN:-}"
    if [ -z "$token" ]; then
      # 32 bytes of URL-safe randomness.
      token=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))" 2>/dev/null \
        || openssl rand -hex 32)
    fi
    aws ssm put-parameter --name "$API_TOKEN_PARAM" --type SecureString \
      --value "$token" --region "$region" \
      --description "Shared bearer token for world-model inference endpoints" >/dev/null
    echo "  Generated inference API token → SSM $API_TOKEN_PARAM" >&2
  fi
  echo "$API_TOKEN_PARAM"
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
get_manifest_field() {
  local model="$1" field="$2"
  grep "^${field}:" "$REPO_ROOT/inference/models/$model/endpoint.yaml" 2>/dev/null | head -1 | awk '{print $2}'
}

# Models with a restricted weights license require explicit acknowledgement
# before deploy. Set LYRA_LICENSE_ACK=1 (or ACCEPT_LICENSE=1) to skip the prompt
# in automation.
check_license() {
  local model="$1"
  [ "$(get_manifest_field "$model" license_restricted)" = "true" ] || return 0

  local name url
  name=$(get_manifest_field "$model" license)
  url=$(get_manifest_field "$model" license_url)
  echo "⚠️  License notice for '$model'" >&2
  echo "    The model weights are under a restricted license: ${name:-restricted}" >&2
  echo "    ${url:-see the endpoint.yaml for this model}" >&2
  echo "    Internal R&D only — not for production, public deployment, or redistribution." >&2
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

  # Fetch the shared API token so the vite dev proxy can authenticate against a
  # secured endpoint (R1). The token is passed to vite as an env var and injected
  # SERVER-SIDE by the proxy — it is NOT written into config.js, so it never
  # reaches the browser bundle. Empty when no token has been provisioned yet.
  local api_token=""
  if [ -n "$url" ]; then
    api_token=$(aws ssm get-parameter --name "$API_TOKEN_PARAM" --with-decryption \
      --query Parameter.Value --output text --region "$region" 2>/dev/null || true)
  fi

  # With a live endpoint the UI talks to its OWN origin (empty lingbotApiUrl →
  # relative paths like /generate), which the vite proxy forwards to the ALB.
  # This sidesteps CORS entirely (browser only ever calls localhost). ui:'lingbot'
  # renders the deployable catalogue/generator rather than the real-time lobby.
  local ui_mode="worlds" demo_mode="true"
  if [ -n "$url" ]; then
    ui_mode="lingbot"
    demo_mode="false"
  fi

  cat > "$FRONTEND_DIR/public/config.js" <<CONF
window.APP_CONFIG = {
  region: '$region',
  userPoolId: '',
  userPoolClientId: '',
  websocketUrl: '',
  lingbotApiUrl: '',
  environment: 'local',
  cognitoDomain: '',
  redirectSignIn: 'http://localhost:3000',
  redirectSignOut: 'http://localhost:3000',
  demoMode: $demo_mode,
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
    echo "Starting UI at http://localhost:3000 → proxying to $url (token injected server-side)"
    (cd "$FRONTEND_DIR" && WM_PROXY_TARGET="$url" WM_PROXY_TOKEN="$api_token" npm run dev)
  else
    echo "Starting UI at http://localhost:3000 (demo mode — no live endpoint)"
    (cd "$FRONTEND_DIR" && npm run dev)
  fi
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

  # Security: ensure a shared API token exists (SSM SecureString), and pass the
  # security-relevant settings into the stack as context. The token itself is
  # never placed in CDK context or the CloudFormation template — only its SSM
  # parameter name is, and the container fetches the value at runtime.
  local token_param
  token_param=$(ensure_api_token)

  local ctx_args=(
    -c "model=$model"
    -c "target=$target"
    -c "apiTokenParam=$token_param"
  )
  [ -n "${WORLD_MODEL_ALLOWED_ORIGINS:-}" ] && ctx_args+=(-c "allowedOrigins=$WORLD_MODEL_ALLOWED_ORIGINS")
  [ -n "${WORLD_MODEL_RATE_LIMIT:-}" ] && ctx_args+=(-c "rateLimit=$WORLD_MODEL_RATE_LIMIT")
  [ -n "${CERTIFICATE_ARN:-}" ] && ctx_args+=(-c "certificateArn=$CERTIFICATE_ARN")
  [ -n "${CAPACITY_BLOCK_ID:-}" ] && ctx_args+=(-c "capacity=$CAPACITY_BLOCK_ID")

  if [ -z "${CERTIFICATE_ARN:-}" ] && [ "$target" = "ec2" ]; then
    echo "  ⚠  No CERTIFICATE_ARN set — the ALB will serve plain HTTP. Set CERTIFICATE_ARN" >&2
    echo "     to an ACM cert ARN to enable HTTPS before exposing this to untrusted networks." >&2
  fi

  # Model stack
  (cd "$CDK_DIR" && npx cdk deploy "WorldModel-$model" \
    "${ctx_args[@]}" \
    --require-approval never)

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

# --- Main ---

case "${1:-}" in
  -h|--help)  usage ;;
  list)       cmd_list ;;
  status)     cmd_status ;;
  build)      shift; cmd_build "$@" ;;
  destroy)    shift; cmd_destroy "$@" ;;
  ui)         shift; cmd_ui "$@" ;;
  "")         cmd_deploy "$DEFAULT_MODEL" "ec2" ;;
  *)          cmd_deploy "$@" ;;
esac
