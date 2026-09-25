# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for testing against Cognito auth without talking to Cognito.

The server verifies real RS256 signatures against the user pool's JWKS. Tests must
not reach the network, but must still exercise that verification — so these helpers
generate a keypair once, mint properly-signed access tokens with it, and swap the
JWKS client for one that returns the matching public key.

Importantly this patches only the *key source*. Signature checking, issuer, expiry,
``token_use``, client allowlist and scope are all still enforced by the real code
path, so a test that mints a bad token genuinely fails.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

INFERENCE_DIR = Path(__file__).resolve().parents[1] / "inference"
if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))

POOL_ID = "eu-west-1_TestPool"
REGION = "eu-west-1"
BROWSER_CLIENT = "browsertestclient"
MACHINE_CLIENT = "machinetestclient"
SCOPE = "world-model/invoke"
ISSUER = f"https://cognito-idp.{REGION}.amazonaws.com/{POOL_ID}"

_key = None


def signing_key():
    """One RSA keypair for the whole test session — 2048-bit keygen is slow."""
    global _key
    if _key is None:
        from cryptography.hazmat.primitives.asymmetric import rsa

        _key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return _key


class StubJWKClient:
    """Returns one fixed public key, mirroring PyJWKClient's contract."""

    def __init__(self, public_key, *args, **kwargs):
        self._key = public_key

    def get_signing_key_from_jwt(self, token):  # noqa: ARG002 - fixed key by design
        return type("Key", (), {"key": self._key})()


def make_access_token(
    *,
    client_id: str = BROWSER_CLIENT,
    scope: str = SCOPE,
    token_use: str = "access",
    issuer: str = ISSUER,
    expires_in: int = 3600,
    key=None,
) -> str:
    """Mint a signed Cognito-shaped access token."""
    import jwt

    now = int(time.time())
    return jwt.encode(
        {
            "iss": issuer,
            "client_id": client_id,
            "token_use": token_use,
            "scope": scope,
            "iat": now,
            "exp": now + expires_in,
            "sub": "test-user",
        },
        key or signing_key(),
        algorithm="RS256",
    )


def use_cognito(monkeypatch, *, scope: str | None = SCOPE, clients: str | None = None):
    """Configure the process so ``SecurityConfig()`` builds a working verifier.

    Patches the JWKS client rather than the verifier, so every other check in
    ``TokenVerifier.verify`` still runs for real.
    """
    import lib.cognito as cognito_mod

    public = signing_key().public_key()
    monkeypatch.setattr(
        cognito_mod,
        "PyJWKClient",
        lambda *a, **k: StubJWKClient(public),
        raising=True,
    )
    monkeypatch.setenv("WORLD_MODEL_AUTH_MODE", "cognito")
    monkeypatch.setenv("WORLD_MODEL_COGNITO_USER_POOL_ID", POOL_ID)
    monkeypatch.setenv(
        "WORLD_MODEL_COGNITO_CLIENT_IDS",
        clients if clients is not None else f"{BROWSER_CLIENT},{MACHINE_CLIENT}",
    )
    if scope:
        monkeypatch.setenv("WORLD_MODEL_COGNITO_SCOPE", scope)
    else:
        monkeypatch.delenv("WORLD_MODEL_COGNITO_SCOPE", raising=False)


def disable_auth(monkeypatch):
    """The explicit local-development opt-out."""
    monkeypatch.setenv("WORLD_MODEL_AUTH_MODE", "disabled")
    for var in (
        "WORLD_MODEL_COGNITO_USER_POOL_ID",
        "WORLD_MODEL_COGNITO_CLIENT_IDS",
        "WORLD_MODEL_COGNITO_SCOPE",
    ):
        monkeypatch.delenv(var, raising=False)


def requires_jwt():
    """Skip marker for environments without pyjwt[crypto]."""
    pytest.importorskip("jwt", reason="pyjwt[crypto] not installed")
    pytest.importorskip("cryptography", reason="pyjwt[crypto] not installed")
