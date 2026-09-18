# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Security controls for the inference server.

Everything here is env-var driven so the same container image runs locked
down in production and open for local/demo use:

  WORLD_MODEL_API_TOKEN        Shared bearer token. If set, every request
                               (and every WebSocket) must present it via
                               ``Authorization: Bearer <token>`` (the
                               ``Bearer`` prefix is optional) or, for
                               WebSockets, a ``?token=<token>`` query param.
                               If UNSET, auth is DISABLED and a loud warning
                               is logged at startup.

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
import secrets
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

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
    """Snapshot of security-relevant env vars, resolved once at startup."""

    def __init__(self) -> None:
        self.api_token: Optional[str] = os.environ.get("WORLD_MODEL_API_TOKEN") or None
        self.allowed_origins: list[str] = _parse_origins(
            os.environ.get("WORLD_MODEL_ALLOWED_ORIGINS", "")
        )
        self.rate_limit: int = _parse_int(
            os.environ.get("WORLD_MODEL_RATE_LIMIT"), _DEFAULT_RATE_LIMIT
        )

    @property
    def auth_enabled(self) -> bool:
        return self.api_token is not None

    def log_summary(self) -> None:
        if self.auth_enabled:
            logger.info("Auth ENABLED (WORLD_MODEL_API_TOKEN set).")
        else:
            logger.warning(
                "Auth DISABLED — WORLD_MODEL_API_TOKEN is not set. The inference "
                "API is UNAUTHENTICATED. Set WORLD_MODEL_API_TOKEN before exposing "
                "this endpoint to any untrusted network."
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
    """Constant-time comparison of a presented token against the configured one."""
    if not config.auth_enabled:
        return True
    if not presented:
        return False
    return secrets.compare_digest(presented, config.api_token or "")


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
