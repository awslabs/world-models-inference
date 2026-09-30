# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Amazon Cognito access-token verification.

This replaces a hand-rolled shared bearer token. The distinction matters beyond
taste: AWS security review treats a bespoke credential scheme as *custom
authentication* — "a Cognito alternative" — which makes a reusable solution
ineligible for an Enhanced DSR and requires a full AppSec review. Verifying
tokens issued by a managed identity provider is the opposite: we are an OIDC
resource server doing what the spec says, with no credential scheme of our own.

We verify **access tokens**, not ID tokens. An ID token describes the end user to
the client; an access token authorises an API call and carries the scopes that say
what the caller may do. The previous frontend sent an ID token, which is the usual
mistake. Access tokens from Cognito have no ``aud`` claim — the audience is
``client_id`` — so ``verify_aud`` is off and the client is checked explicitly.

Two grant types are expected, and both land here identically:

* **Browser** — authorization code + PKCE, a real user signs in.
* **Machine** (``deploy.sh bench``, CI) — client credentials, no user. This is
  why the shared static token is not needed for automation any more.

Environment:

  WORLD_MODEL_COGNITO_USER_POOL_ID   e.g. ``eu-west-1_AbC123``. Required unless
                                     auth is explicitly disabled.
  WORLD_MODEL_COGNITO_CLIENT_IDS     Comma-separated allowlist of app client IDs.
                                     Required: a valid token from a client we do
                                     not know is not authorisation.
  WORLD_MODEL_COGNITO_SCOPE          Required scope, e.g. ``world-model/invoke``.
                                     Optional; when set the token must carry it.
  WORLD_MODEL_COGNITO_REGION         Defaults to the pool ID's region prefix,
                                     then AWS_REGION.
  WORLD_MODEL_JWKS_URL               Override the JWKS endpoint. For tests only.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

# PyJWT is a hard requirement whenever auth is on. Import failure must not
# degrade into "serving unauthenticated" — see AuthUnavailable below.
try:  # pragma: no cover - exercised via tests that stub the module
    import jwt
    from jwt import PyJWKClient

    _IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover
    jwt = None  # type: ignore[assignment]
    PyJWKClient = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc


class AuthError(Exception):
    """A token was presented and is not acceptable. Maps to 401."""


class AuthMisconfigured(Exception):
    """Auth is required but not configured. Raised at startup, never per-request.

    Deliberately fatal. The alternative — start up and serve unauthenticated —
    is the failure this module exists to prevent.
    """


@dataclass(frozen=True)
class CognitoConfig:
    user_pool_id: str
    client_ids: frozenset[str]
    region: str
    required_scope: Optional[str] = None
    jwks_url_override: Optional[str] = None
    # Small leeway for clock skew between Cognito and this host. Kept tight:
    # generous skew on `exp` extends the life of a revoked token.
    leeway_seconds: int = 30

    @property
    def issuer(self) -> str:
        return f"https://cognito-idp.{self.region}.amazonaws.com/{self.user_pool_id}"

    @property
    def jwks_url(self) -> str:
        return self.jwks_url_override or f"{self.issuer}/.well-known/jwks.json"

    @classmethod
    def from_env(cls) -> "CognitoConfig":
        pool = (os.environ.get("WORLD_MODEL_COGNITO_USER_POOL_ID") or "").strip()
        if not pool:
            raise AuthMisconfigured(
                "WORLD_MODEL_COGNITO_USER_POOL_ID is not set. Set it to the Cognito "
                "user pool that fronts this endpoint, or set "
                "WORLD_MODEL_AUTH_MODE=disabled to run without authentication "
                "(never on a reachable network)."
            )

        raw_clients = os.environ.get("WORLD_MODEL_COGNITO_CLIENT_IDS", "")
        clients = frozenset(c.strip() for c in raw_clients.split(",") if c.strip())
        if not clients:
            raise AuthMisconfigured(
                "WORLD_MODEL_COGNITO_CLIENT_IDS is not set. Without an allowlist any "
                "app client in the pool could call this endpoint, including ones "
                "created for unrelated applications."
            )

        # A pool ID is "<region>_<id>", so the region is already in hand; AWS_REGION
        # is the fallback for unusual pool IDs rather than the primary source.
        region = (os.environ.get("WORLD_MODEL_COGNITO_REGION") or "").strip()
        if not region:
            region = pool.split("_", 1)[0] if "_" in pool else ""
        if not region:
            region = (os.environ.get("AWS_REGION") or "").strip()
        if not region:
            raise AuthMisconfigured(
                "Could not determine the Cognito region. Set "
                "WORLD_MODEL_COGNITO_REGION or AWS_REGION."
            )

        scope = (os.environ.get("WORLD_MODEL_COGNITO_SCOPE") or "").strip() or None
        override = (os.environ.get("WORLD_MODEL_JWKS_URL") or "").strip() or None
        return cls(
            user_pool_id=pool,
            client_ids=clients,
            region=region,
            required_scope=scope,
            jwks_url_override=override,
        )

    def log_summary(self) -> None:
        logger.info(
            "Auth ENABLED (Cognito). issuer=%s clients=%d scope=%s",
            self.issuer,
            len(self.client_ids),
            self.required_scope or "(none required)",
        )


class TokenVerifier:
    """Verifies Cognito access tokens against the pool's JWKS.

    The signing keys are fetched from Cognito and cached by ``PyJWKClient``.
    Cognito rotates keys, so a cache miss triggers a refetch; that is a network
    call on the request path, which is why the client is created once and shared.
    """

    def __init__(self, config: CognitoConfig, jwk_client: Any = None) -> None:
        if _IMPORT_ERROR is not None or jwt is None:
            # Fail closed, loudly, at construction. A missing crypto dependency
            # must not turn into an endpoint that accepts anything.
            raise AuthMisconfigured(
                "PyJWT (with the [crypto] extra) is required to verify Cognito "
                f"tokens but could not be imported: {_IMPORT_ERROR!r}. It is listed "
                "in inference/lib/requirements.txt; if this container was built "
                "from a Dockerfile that does not install that file, fix the build."
            )
        self.config = config
        self._jwk_client = jwk_client or PyJWKClient(
            config.jwks_url, cache_keys=True, lifespan=3600
        )

    def verify(self, token: str) -> dict[str, Any]:
        """Return the token's claims, or raise ``AuthError``.

        Checks, in order: signature (RS256 against the pool's JWKS), issuer,
        expiry, that this is an *access* token, that it came from a client we
        allow, and that it carries the required scope.
        """
        if not token:
            raise AuthError("No token presented")

        try:
            signing_key = self._jwk_client.get_signing_key_from_jwt(token)
        except Exception as exc:
            # Covers unknown kid, malformed token, and JWKS fetch failure. We
            # cannot distinguish "attacker sent junk" from "Cognito unreachable"
            # here, so log and reject: failing closed on an outage is correct for
            # a GPU endpoint that costs money to run.
            raise AuthError(f"Could not resolve a signing key: {exc}") from exc

        try:
            claims: dict[str, Any] = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                issuer=self.config.issuer,
                leeway=self.config.leeway_seconds,
                options={
                    # Cognito access tokens carry no `aud`; the audience is
                    # `client_id`, checked below. Leaving verify_aud on would
                    # reject every valid access token.
                    "verify_aud": False,
                    "require": ["exp", "iss", "token_use", "client_id"],
                },
            )
        except Exception as exc:
            raise AuthError(f"Token rejected: {exc}") from exc

        token_use = claims.get("token_use")
        if token_use != "access":
            # An ID token is signed by the same pool and would otherwise pass
            # every check above, so this is a real authorisation boundary, not a
            # formality.
            raise AuthError(
                f"Expected an access token, got token_use={token_use!r}. Send the "
                "access token, not the ID token."
            )

        client_id = claims.get("client_id")
        if client_id not in self.config.client_ids:
            raise AuthError(f"Client {client_id!r} is not allowed")

        if self.config.required_scope:
            scopes = _scopes(claims)
            if self.config.required_scope not in scopes:
                raise AuthError(
                    f"Token lacks required scope {self.config.required_scope!r}"
                )

        return claims


def _scopes(claims: dict[str, Any]) -> frozenset[str]:
    """Cognito puts scopes in a space-delimited ``scope`` string."""
    raw = claims.get("scope")
    if isinstance(raw, str):
        return frozenset(s for s in raw.split(" ") if s)
    if isinstance(raw, Iterable) and not isinstance(raw, (bytes, dict)):
        return frozenset(str(s) for s in raw)
    return frozenset()
