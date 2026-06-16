"""
auth.py — JWT Authentication for Medha API

Supabase uses ES256 (elliptic curve) JWT tokens.
We verify them using Supabase's public JWKS endpoint
instead of the JWT secret string.
"""

import os
import httpx
import jwt

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

security = HTTPBearer()

# Cache the JWKS so we don't fetch it on every request
_jwks_cache: dict | None = None


def _get_jwks() -> dict:
    """
    Fetch Supabase's public JSON Web Key Set (JWKS).
    This contains the public key used to verify ES256 tokens.
    Cached after first fetch.
    """
    global _jwks_cache
    if _jwks_cache:
        return _jwks_cache

    supabase_url = os.getenv("SUPABASE_URL", "").strip()
    jwks_url     = f"{supabase_url}/auth/v1/.well-known/jwks.json"

    try:
        resp = httpx.get(jwks_url, timeout=10)
        resp.raise_for_status()
        _jwks_cache = resp.json()
        return _jwks_cache
    except Exception as e:
        raise RuntimeError(f"Failed to fetch JWKS from Supabase: {e}")


def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    """
    Verify Supabase ES256 JWT and return user_id.
    Raises 401 if token is invalid or expired.
    """
    token = credentials.credentials

    try:
        # Get the public keys from Supabase
        jwks = _get_jwks()

        # Get the key id from token header
        header = jwt.get_unverified_header(token)
        kid    = header.get("kid")

        # Find the matching key in JWKS
        public_key = None
        for key in jwks.get("keys", []):
            if key.get("kid") == kid:
                public_key = jwt.algorithms.ECAlgorithm.from_jwk(key)
                break

        if not public_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="No matching public key found",
            )

        # Verify and decode the token
        payload = jwt.decode(
            token,
            public_key,
            algorithms=["ES256"],
            options={"verify_aud": False},
            leeway=10,
        )

        user_id: str | None = payload.get("sub")

        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token missing user ID",
            )

        return user_id

    except HTTPException:
        raise

    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired — please log in again",
        )

    except jwt.PyJWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}",
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Auth error: {str(e)}",
        )