# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Security controls for the inference server.

Everything here is env-var driven so the same container image runs locked
down in production and open for local/demo use:

  WORLD_MODEL_AUTH_MODE        ``cognito`` (default) or ``disabled``.

                               Authentication is REQUIRED by default. If the
                               Cognito settings below are missing the server
                               REFUSES TO START rather than serving an
                               unauthenticated GPU endpoint — the previous
                               behaviour, where an unset token silently disabled
                               auth, meant a deploy that forgot one env var
                               produced an open endpoint with only a log line to
                               say so.

                               ``disabled`` is an explicit, loudly-logged opt-out
                               for local development. Never use it on a network
                               anyone else can reach.

                               Callers present a Cognito **access token** as
                               ``Authorization: Bearer <token>``, or for
                               WebSockets a ``?token=`` query param. See
                               ``lib/cognito.py`` for what is verified and why it
                               is an access token rather than an ID token.

  WORLD_MODEL_COGNITO_*        User pool, allowed app client IDs and required
                               scope. Documented in ``lib/cognito.py``.

  WORLD_MODEL_ALLOWED_ORIGINS  Comma-separated CORS allowlist (e.g.
                               ``https://d123.cloudfront.net``). If UNSET, no
                               cross-origin browser requests are permitted
                               (deny by default).

  WORLD_MODEL_RATE_LIMIT       Max requests per client IP per 60s window on
                               the expensive inference routes. Default 60.
                               Set to 0 to disable.

Health probes (``/ping``, ``/health``) are always exempt from auth and rate
limiting so load-balancer health checks keep working.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from lib.cognito import AuthError, AuthMisconfigured, CognitoConfig, TokenVerifier

logger = logging.getLogger(__name__)

# Paths that must never require auth / rate limiting (LB health checks).
PUBLIC_PATHS = frozenset({"/ping", "/health"})

# Only the GPU-bound routes are rate limited; status polling is not.
RATE_LIMITED_PREFIXES = ("/generate", "/invocations")

# Example identifiers are single path segments — alphanumerics, dash, underscore.
_EXAMPLE_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")

_DEFAULT_RATE_LIMIT = 60


# =============================================================================
# Configuration
# =============================================================================


class SecurityConfig:
    """Snapshot of security-relevant env vars, resolved once at startup.

    Constructing this raises ``AuthMisconfigured`` when auth is required but not
    configured. That is deliberate: it turns a missing env var into a failed
    deploy instead of an open endpoint.
    """

    def __init__(self) -> None:
        mode = (os.environ.get("WORLD_MODEL_AUTH_MODE") or "cognito").strip().lower()
        if mode not in {"cognito", "disabled"}:
            raise AuthMisconfigured(
                f"WORLD_MODEL_AUTH_MODE={mode!r} is not recognised. Use 'cognito' "
                "(default) or 'disabled'."
            )
        self.auth_mode: str = mode
        self.cognito: Optional[CognitoConfig] = None
        self._verifier: Optional[TokenVerifier] = None

        if mode == "cognito":
            # Both of these raise rather than degrade. from_env() reports which
            # setting is missing; TokenVerifier() fails if PyJWT is absent, which
            # is how a Dockerfile that skips lib/requirements.txt gets caught.
            self.cognito = CognitoConfig.from_env()
            self._verifier = TokenVerifier(self.cognito)

        self.allowed_origins: list[str] = _parse_origins(
            os.environ.get("WORLD_MODEL_ALLOWED_ORIGINS", "")
        )
        self.rate_limit: int = _parse_int(
            os.environ.get("WORLD_MODEL_RATE_LIMIT"), _DEFAULT_RATE_LIMIT
        )

    @property
    def auth_enabled(self) -> bool:
        return self.auth_mode == "cognito"

    @property
    def verifier(self) -> Optional[TokenVerifier]:
        return self._verifier

    def log_summary(self) -> None:
        if self.auth_enabled and self.cognito is not None:
            self.cognito.log_summary()
        else:
            logger.warning(
                "Auth DISABLED via WORLD_MODEL_AUTH_MODE=disabled. The inference "
                "API is UNAUTHENTICATED and anyone who can reach it can run GPU "
                "inference. This is for local development only."
            )
        if self.allowed_origins:
            logger.info("CORS allowlist: %s", ", ".join(self.allowed_origins))
        else:
            logger.info(
                "CORS: no allowlist configured — cross-origin browser requests "
                "are denied. Set WORLD_MODEL_ALLOWED_ORIGINS to permit a frontend."
            )
        if self.rate_limit > 0:
            logger.info("Rate limit: %d req/min per client IP on inference routes.", self.rate_limit)
        else:
            logger.warning("Rate limiting DISABLED (WORLD_MODEL_RATE_LIMIT=0).")


def _parse_origins(raw: str) -> list[str]:
    return [o.strip() for o in raw.split(",") if o.strip()]


def _parse_int(raw: Optional[str], default: int) -> int:
    if raw is None or raw.strip() == "":
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning("Invalid integer %r; using default %d.", raw, default)
        return default


# =============================================================================
# Token extraction / verification
# =============================================================================


def extract_token(auth_header: Optional[str], query_token: Optional[str]) -> Optional[str]:
    """Pull a token from an ``Authorization`` header or ``?token=`` query param.

    Accepts ``Bearer <token>`` (case-insensitive prefix) or a bare token value.
    """
    if auth_header:
        parts = auth_header.split(" ", 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1].strip()
        return auth_header.strip()
    if query_token:
        return query_token.strip()
    return None


def token_valid(config: SecurityConfig, presented: Optional[str]) -> bool:
    """True if ``presented`` is an acceptable Cognito access token.

    Kept as a predicate so the HTTP middleware and the WebSocket handler share one
    decision point. The reason for rejection is logged here rather than returned,
    because it must not reach the client: "token lacks scope X" and "client Y is
    not allowed" are both useful to an attacker enumerating a pool.
    """
    if not config.auth_enabled:
        return True
    verifier = config.verifier
    if verifier is None:  # pragma: no cover - constructor guarantees this
        raise AuthMisconfigured("Auth is enabled but no verifier was built")
    if not presented:
        return False
    try:
        verifier.verify(presented)
        return True
    except AuthError as exc:
        logger.info("Rejected token: %s", exc)
        return False


# =============================================================================
# Path validation (defends against traversal via user-supplied ids)
# =============================================================================


def safe_subdir(base: Path, name: str) -> Path:
    """Resolve ``name`` as a single safe segment under ``base``.

    Rejects anything that is not ``[A-Za-z0-9_-]+`` and bounds-checks the
    resolved path so ``../`` or absolute inputs cannot escape ``base``.
    Raises ``HTTPException(400)`` on invalid input.
    """
    if not name or not _EXAMPLE_ID_RE.match(name):
        raise HTTPException(status_code=400, detail="Invalid identifier")
    base_resolved = base.resolve()
    resolved = (base_resolved / name).resolve()
    if resolved != base_resolved and base_resolved not in resolved.parents:
        raise HTTPException(status_code=400, detail="Invalid path")
    return resolved


# =============================================================================
# Middleware
# =============================================================================


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add conservative security headers to every HTTP response."""

    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("Cache-Control", "no-store")
        return response


class AuthMiddleware(BaseHTTPMiddleware):
    """Require a valid bearer token on all non-public HTTP routes.

    No-op when auth is disabled. WebSocket auth is enforced separately in the
    stream handler (BaseHTTPMiddleware does not see the ``websocket`` scope).
    """

    def __init__(self, app, config: SecurityConfig):
        super().__init__(app)
        self.config = config

    async def dispatch(self, request: Request, call_next):
        if not self.config.auth_enabled or request.method == "OPTIONS":
            return await call_next(request)
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        presented = extract_token(
            request.headers.get("authorization"),
            request.query_params.get("token"),
        )
        if not token_valid(self.config, presented):
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Fixed-window per-IP rate limiting on the expensive inference routes."""

    def __init__(self, app, config: SecurityConfig):
        super().__init__(app)
        self.limit = config.rate_limit
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def _client_key(self, request: Request) -> str:
        # Honour X-Forwarded-For (ALB sets it) but fall back to the peer address.
        # Use the LAST entry, not the first: a client can spoof leading entries,
        # but the ALB appends the real peer IP as the rightmost hop. Taking
        # split(",")[0] would let a caller forge the key and dodge the limit.
        fwd = request.headers.get("x-forwarded-for")
        if fwd:
            hops = [h.strip() for h in fwd.split(",") if h.strip()]
            if hops:
                return hops[-1]
        return request.client.host if request.client else "unknown"

    def _allow(self, key: str, now: float) -> bool:
        window_start = now - 60.0
        with self._lock:
            hits = [t for t in self._hits.get(key, []) if t >= window_start]
            if len(hits) >= self.limit:
                self._hits[key] = hits
                return False
            hits.append(now)
            self._hits[key] = hits
            return True

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if self.limit <= 0 or not path.startswith(RATE_LIMITED_PREFIXES):
            return await call_next(request)
        if not self._allow(self._client_key(request), time.time()):
            return JSONResponse(
                {"detail": "Rate limit exceeded"},
                status_code=429,
                headers={"Retry-After": "60"},
            )
        return await call_next(request)
