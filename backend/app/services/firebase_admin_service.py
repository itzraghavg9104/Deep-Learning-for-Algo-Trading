"""
Firebase Admin initialization and token verification helpers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from app.config import settings

try:
    import firebase_admin
    from firebase_admin import auth as firebase_auth
    from firebase_admin import credentials
except Exception:  # pragma: no cover - optional dependency during transition
    firebase_admin = None
    firebase_auth = None
    credentials = None


@dataclass
class FirebasePrincipal:
    uid: str
    email: str
    email_verified: bool
    name: Optional[str] = None


_firebase_initialized = False


def is_firebase_ready() -> bool:
    return (
        settings.FIREBASE_AUTH_ENABLED
        and bool(settings.FIREBASE_SERVICE_ACCOUNT_PATH)
        and firebase_admin is not None
    )


def initialize_firebase_admin() -> bool:
    global _firebase_initialized
    if _firebase_initialized:
        return True
    if not is_firebase_ready():
        return False

    if not firebase_admin._apps:
        cred = credentials.Certificate(settings.FIREBASE_SERVICE_ACCOUNT_PATH)
        firebase_admin.initialize_app(cred, {"projectId": settings.FIREBASE_PROJECT_ID or None})
    _firebase_initialized = True
    return True


def verify_firebase_token(id_token: str) -> FirebasePrincipal:
    if not initialize_firebase_admin():
        raise ValueError("Firebase Admin is not configured")
    decoded: dict[str, Any] = firebase_auth.verify_id_token(id_token)
    uid = decoded.get("uid")
    if not uid:
        raise ValueError("Token missing uid")
    return FirebasePrincipal(
        uid=uid,
        email=(decoded.get("email") or f"{uid}@firebase.local").lower(),
        email_verified=bool(decoded.get("email_verified", False)),
        name=decoded.get("name"),
    )
