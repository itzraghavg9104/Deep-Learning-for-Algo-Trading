"""
Firestore-backed profile/trade persistence with a safe in-memory fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.config import settings
from app.services.demo_store import demo_store
from app.services.firebase_admin_service import initialize_firebase_admin, is_firebase_ready

try:
    from firebase_admin import firestore as admin_firestore
    from google.cloud import firestore
except Exception:  # pragma: no cover - optional dependency
    admin_firestore = None
    firestore = None


@dataclass
class BehaviorProfile:
    user_id: str
    risk_tolerance: float
    risk_category: str
    behavior_array: Dict[str, Any]
    recommendations: Dict[str, Any]
    updated_at: str


class FirestoreStore:
    def __init__(self) -> None:
        self._db = None
        self._enabled = False
        self._init()

    def _init(self) -> None:
        if not is_firebase_ready() or firestore is None or admin_firestore is None:
            self._enabled = False
            return
        if not initialize_firebase_admin():
            self._enabled = False
            return
        self._db = admin_firestore.client(
            database_id=settings.FIRESTORE_DATABASE_ID,
        )
        self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled and self._db is not None

    def _user_ref(self, user_uid: str):
        return self._db.collection("users").document(user_uid)

    def upsert_user(self, user_uid: str, email: str, is_active: bool = True) -> dict:
        if not self.enabled:
            user = demo_store.get_user_by_email(email) or demo_store.create_user(email=email, hashed_password="firebase")
            return {"id": str(user.id), "email": user.email, "is_active": user.is_active}
        payload = {
            "uid": user_uid,
            "email": email.lower(),
            "is_active": is_active,
            "updated_at": firestore.SERVER_TIMESTAMP,
        }
        self._user_ref(user_uid).set(payload, merge=True)
        return {"id": user_uid, "email": email.lower(), "is_active": is_active}

    def get_profile(self, user_uid: str, email: str) -> dict:
        if not self.enabled:
            user = demo_store.get_user_by_email(email) or demo_store.create_user(email=email, hashed_password="firebase")
            profile = demo_store.get_or_create_profile(user.id)
            return {
                "id": str(user.id),
                "email": user.email,
                "risk_profile": {"tolerance": profile.risk_tolerance, "category": profile.risk_category},
                "preferences": {
                    "use_sentiment": profile.use_sentiment,
                    "preferred_timeframe": profile.preferred_timeframe,
                    "symbols": list(profile.symbols),
                },
                "behavior_profile": profile.behavior_profile,
            }

        user_doc = self._user_ref(user_uid).get()
        risk_doc = self._user_ref(user_uid).collection("risk_assessments").document("latest").get()
        pref_doc = self._user_ref(user_uid).collection("preferences").document("latest").get()
        behavior_doc = self._user_ref(user_uid).collection("behavior_profiles").document("latest").get()

        risk_data = risk_doc.to_dict() if risk_doc.exists else {"risk_tolerance": 0.5, "category": "Moderate"}
        pref_data = pref_doc.to_dict() if pref_doc.exists else {
            "use_sentiment": False,
            "preferred_timeframe": "swing",
            "symbols": ["RELIANCE.NS", "TCS.NS", "INFY.NS"],
        }
        behavior_data = behavior_doc.to_dict() if behavior_doc.exists else {}

        return {
            "id": user_uid,
            "email": (user_doc.to_dict() or {}).get("email", email.lower()) if user_doc.exists else email.lower(),
            "risk_profile": {"tolerance": risk_data.get("risk_tolerance", 0.5), "category": risk_data.get("category", "Moderate")},
            "preferences": {
                "use_sentiment": pref_data.get("use_sentiment", False),
                "preferred_timeframe": pref_data.get("preferred_timeframe", "swing"),
                "symbols": pref_data.get("symbols", ["RELIANCE.NS"]),
            },
            "behavior_profile": behavior_data,
        }

    def save_preferences(self, user_uid: str, email: str, preferences: dict) -> dict:
        if not self.enabled:
            user = demo_store.get_user_by_email(email) or demo_store.create_user(email=email, hashed_password="firebase")
            demo_store.update_profile(
                user.id,
                use_sentiment=preferences.get("use_sentiment", False),
                preferred_timeframe=preferences.get("preferred_timeframe", "swing"),
                symbols=tuple(preferences.get("symbols", [])),
            )
            return preferences
        self._user_ref(user_uid).collection("preferences").document("latest").set(
            {
                **preferences,
                "updated_at": firestore.SERVER_TIMESTAMP,
            },
            merge=True,
        )
        return preferences

    def save_risk_assessment(self, user_uid: str, payload: dict) -> dict:
        if not self.enabled:
            return payload
        self._user_ref(user_uid).collection("risk_assessments").document("latest").set(
            {
                **payload,
                "updated_at": firestore.SERVER_TIMESTAMP,
            },
            merge=True,
        )
        return payload

    def save_behavior_profile(self, user_uid: str, payload: dict) -> dict:
        if not self.enabled:
            return payload
        self._user_ref(user_uid).collection("behavior_profiles").document("latest").set(
            {
                **payload,
                "updated_at": firestore.SERVER_TIMESTAMP,
            },
            merge=True,
        )
        return payload

    def get_user_trades(self, user_uid: str, email: str) -> List[dict]:
        if not self.enabled:
            user = demo_store.get_user_by_email(email)
            if not user:
                return []
            return demo_store.get_user_trades(user.id)

        docs = self._user_ref(user_uid).collection("trade_events").order_by("closed_at", direction=firestore.Query.DESCENDING).limit(200).stream()
        return [doc.to_dict() for doc in docs]

    def save_trade_evaluation(self, user_uid: str, payload: dict) -> dict:
        if not self.enabled:
            return payload
        doc_ref = self._user_ref(user_uid).collection("trade_evaluations").document()
        doc_ref.set({**payload, "created_at": firestore.SERVER_TIMESTAMP})
        return {"id": doc_ref.id, **payload}


firestore_store = FirestoreStore()
