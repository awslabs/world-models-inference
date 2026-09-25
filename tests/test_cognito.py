# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cognito access-token verification.

These tests sign real RS256 tokens with a generated keypair and hand the verifier a
stub JWKS client, so the signature path is genuinely exercised rather than mocked
away. A test that patches ``verify`` to return True proves nothing about the thing
we care about — that a forged or wrong-shaped token is rejected.

Every rejection case below corresponds to a way an attacker (or an honest mistake)
could otherwise get through: a token signed by the wrong key, an ID token instead
of an access token, a token from a different app client, a missing scope, and an
expired token.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

# Make `import lib.cognito` resolve against the inference package, the same way
# test_security.py does.
INFERENCE_DIR = Path(__file__).resolve().parents[1] / "inference"
if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))

jwt = pytest.importorskip("jwt", reason="pyjwt[crypto] not installed")
pytest.importorskip("cryptography", reason="pyjwt[crypto] not installed")

from cryptography.hazmat.primitives.asymmetric import rsa  # noqa: E402

from lib.cognito import (  # noqa: E402
    AuthError,
    AuthMisconfigured,
    CognitoConfig,
    TokenVerifier,
)

POOL = "eu-west-1_TestPool"
REGION = "eu-west-1"
CLIENT = "browserclient123"
OTHER_CLIENT = "someoneelsesclient"
SCOPE = "world-model/invoke"
ISSUER = f"https://cognito-idp.{REGION}.amazonaws.com/{POOL}"


@pytest.fixture(scope="module")
def keypair():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key, key.public_key()


@pytest.fixture(scope="module")
def other_keypair():
    """A key the pool does not know about — used to forge a token."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key, key.public_key()


class _StubJWKClient:
    """Stands in for PyJWKClient, returning one fixed public key.

    Mirrors the real contract: it returns an object with a ``.key`` attribute, and
    it is the only place the verifier reaches the network in production.
    """

    def __init__(self, public_key):
        self._key = public_key

    def get_signing_key_from_jwt(self, token):  # noqa: ARG002 - fixed key by design
        return type("Key", (), {"key": self._key})()


@pytest.fixture
def config():
    return CognitoConfig(
        user_pool_id=POOL,
        client_ids=frozenset({CLIENT}),
        region=REGION,
        required_scope=SCOPE,
    )


@pytest.fixture
def verifier(config, keypair):
    _, public = keypair
    return TokenVerifier(config, jwk_client=_StubJWKClient(public))


def make_token(
    private_key,
    *,
    token_use: str = "access",
    client_id: str = CLIENT,
    scope: str = SCOPE,
    issuer: str = ISSUER,
    expires_in: int = 3600,
    omit: tuple[str, ...] = (),
) -> str:
    now = int(time.time())
    claims = {
        "iss": issuer,
        "client_id": client_id,
        "token_use": token_use,
        "scope": scope,
        "iat": now,
        "exp": now + expires_in,
        "sub": "user-1",
    }
    for field in omit:
        claims.pop(field, None)
    return jwt.encode(claims, private_key, algorithm="RS256")


# --------------------------------------------------------------------------- #
# Accepting what should be accepted
# --------------------------------------------------------------------------- #


def test_valid_access_token_is_accepted(verifier, keypair):
    private, _ = keypair
    claims = verifier.verify(make_token(private))
    assert claims["client_id"] == CLIENT
    assert claims["token_use"] == "access"


def test_extra_scopes_are_fine_as_long_as_the_required_one_is_present(verifier, keypair):
    private, _ = keypair
    token = make_token(private, scope=f"openid {SCOPE} aws.cognito.signin.user.admin")
    assert verifier.verify(token)["client_id"] == CLIENT


def test_scope_is_optional_when_not_configured(config, keypair):
    """A pool without custom scopes still authenticates."""
    private, public = keypair
    relaxed = CognitoConfig(
        user_pool_id=POOL,
        client_ids=frozenset({CLIENT}),
        region=REGION,
        required_scope=None,
    )
    v = TokenVerifier(relaxed, jwk_client=_StubJWKClient(public))
    assert v.verify(make_token(private, scope="openid"))["sub"] == "user-1"


def test_any_allowlisted_client_is_accepted(keypair):
    """The machine client and the browser client are different app clients."""
    private, public = keypair
    cfg = CognitoConfig(
        user_pool_id=POOL,
        client_ids=frozenset({CLIENT, "machineclient456"}),
        region=REGION,
        required_scope=SCOPE,
    )
    v = TokenVerifier(cfg, jwk_client=_StubJWKClient(public))
    assert v.verify(make_token(private, client_id="machineclient456"))


# --------------------------------------------------------------------------- #
# Rejecting what must be rejected
# --------------------------------------------------------------------------- #


def test_token_signed_by_an_unknown_key_is_rejected(verifier, other_keypair):
    """The core of the whole exercise: a forged signature must not pass."""
    forged_private, _ = other_keypair
    with pytest.raises(AuthError):
        verifier.verify(make_token(forged_private))


def test_id_token_is_rejected(verifier, keypair):
    """An ID token is signed by the same pool, so only token_use stops it.

    The previous frontend sent an ID token. Accepting one would mean any signed-in
    user could call the API regardless of the scopes they were granted.
    """
    private, _ = keypair
    with pytest.raises(AuthError, match="access token"):
        verifier.verify(make_token(private, token_use="id"))


def test_token_from_an_unknown_client_is_rejected(verifier, keypair):
    private, _ = keypair
    with pytest.raises(AuthError, match="not allowed"):
        verifier.verify(make_token(private, client_id=OTHER_CLIENT))


def test_token_without_the_required_scope_is_rejected(verifier, keypair):
    private, _ = keypair
    with pytest.raises(AuthError, match="scope"):
        verifier.verify(make_token(private, scope="openid email"))


def test_expired_token_is_rejected(verifier, keypair):
    private, _ = keypair
    with pytest.raises(AuthError):
        verifier.verify(make_token(private, expires_in=-120))


def test_token_from_another_pool_is_rejected(verifier, keypair):
    """Same key, wrong issuer — guards against pool confusion."""
    private, _ = keypair
    other_issuer = f"https://cognito-idp.{REGION}.amazonaws.com/{REGION}_OtherPool"
    with pytest.raises(AuthError):
        verifier.verify(make_token(private, issuer=other_issuer))


@pytest.mark.parametrize("missing", ["exp", "iss", "token_use", "client_id"])
def test_tokens_missing_required_claims_are_rejected(verifier, keypair, missing):
    private, _ = keypair
    with pytest.raises(AuthError):
        verifier.verify(make_token(private, omit=(missing,)))


def test_unsigned_token_is_rejected(verifier):
    """alg=none is the classic JWT bypass."""
    unsigned = jwt.encode(
        {"iss": ISSUER, "client_id": CLIENT, "token_use": "access", "scope": SCOPE},
        key="",
        algorithm="none",
    )
    with pytest.raises(AuthError):
        verifier.verify(unsigned)


def test_empty_and_garbage_tokens_are_rejected(verifier):
    for bad in ("", "   ", "not-a-jwt", "a.b.c", "Bearer something"):
        with pytest.raises(AuthError):
            verifier.verify(bad)


def test_jwks_fetch_failure_fails_closed(config):
    """If Cognito is unreachable we reject, rather than letting traffic through."""

    class Exploding:
        def get_signing_key_from_jwt(self, token):
            raise RuntimeError("JWKS endpoint unreachable")

    v = TokenVerifier(config, jwk_client=Exploding())
    with pytest.raises(AuthError, match="signing key"):
        v.verify("any.token.here")


# --------------------------------------------------------------------------- #
# Configuration: misconfiguration must be fatal, not silently permissive
# --------------------------------------------------------------------------- #


def test_missing_pool_id_is_fatal(monkeypatch):
    monkeypatch.delenv("WORLD_MODEL_COGNITO_USER_POOL_ID", raising=False)
    monkeypatch.setenv("WORLD_MODEL_COGNITO_CLIENT_IDS", CLIENT)
    with pytest.raises(AuthMisconfigured, match="USER_POOL_ID"):
        CognitoConfig.from_env()


def test_missing_client_allowlist_is_fatal(monkeypatch):
    """A valid token from an unknown client is not authorisation."""
    monkeypatch.setenv("WORLD_MODEL_COGNITO_USER_POOL_ID", POOL)
    monkeypatch.delenv("WORLD_MODEL_COGNITO_CLIENT_IDS", raising=False)
    with pytest.raises(AuthMisconfigured, match="CLIENT_IDS"):
        CognitoConfig.from_env()


def test_region_is_derived_from_the_pool_id(monkeypatch):
    monkeypatch.setenv("WORLD_MODEL_COGNITO_USER_POOL_ID", POOL)
    monkeypatch.setenv("WORLD_MODEL_COGNITO_CLIENT_IDS", f"{CLIENT}, second ")
    monkeypatch.delenv("WORLD_MODEL_COGNITO_REGION", raising=False)
    cfg = CognitoConfig.from_env()
    assert cfg.region == REGION
    assert cfg.issuer == ISSUER
    assert cfg.jwks_url == f"{ISSUER}/.well-known/jwks.json"
    # Whitespace around a comma-separated list must not create phantom clients.
    assert cfg.client_ids == frozenset({CLIENT, "second"})


def test_explicit_region_overrides_the_pool_prefix(monkeypatch):
    monkeypatch.setenv("WORLD_MODEL_COGNITO_USER_POOL_ID", POOL)
    monkeypatch.setenv("WORLD_MODEL_COGNITO_CLIENT_IDS", CLIENT)
    monkeypatch.setenv("WORLD_MODEL_COGNITO_REGION", "us-east-1")
    assert CognitoConfig.from_env().region == "us-east-1"


# --------------------------------------------------------------------------- #
# Fail-closed on a missing dependency
# --------------------------------------------------------------------------- #


def test_missing_pyjwt_is_fatal_rather_than_permissive(config, monkeypatch):
    """An image built without pyjwt must refuse to serve, not serve unguarded.

    This is the shape of a real incident on this repo: an image shipped without the
    security library and kept serving with auth unenforced (PCSR D484282450). The
    dependency now lives in inference/lib/requirements.txt, which every Dockerfile
    installs — and if that ever regresses, construction fails here.
    """
    import lib.cognito as cognito_mod

    monkeypatch.setattr(cognito_mod, "_IMPORT_ERROR", ModuleNotFoundError("no jwt"))
    with pytest.raises(AuthMisconfigured, match="PyJWT"):
        TokenVerifier(config)


def test_security_config_propagates_the_missing_dependency(monkeypatch):
    """The failure must surface at startup through SecurityConfig, not per-request."""
    import lib.cognito as cognito_mod
    from lib.security import SecurityConfig

    monkeypatch.setenv("WORLD_MODEL_AUTH_MODE", "cognito")
    monkeypatch.setenv("WORLD_MODEL_COGNITO_USER_POOL_ID", POOL)
    monkeypatch.setenv("WORLD_MODEL_COGNITO_CLIENT_IDS", CLIENT)
    monkeypatch.setattr(cognito_mod, "_IMPORT_ERROR", ModuleNotFoundError("no jwt"))
    with pytest.raises(AuthMisconfigured, match="PyJWT"):
        SecurityConfig()
