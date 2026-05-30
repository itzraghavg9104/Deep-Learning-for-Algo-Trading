"""
Per-user model training orchestration.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

_training_locks: Dict[str, threading.Lock] = {}
_pending_behavior: Dict[str, Dict[str, Any]] = {}
_training_status: Dict[str, Dict[str, Any]] = {}


def _backend_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _user_model_dir(user_id: str) -> Path:
    return _backend_root() / "models" / "users" / user_id


def _run_script(script_rel_path: str, extra_args: list[str]) -> None:
    root = _backend_root()
    script = root / script_rel_path
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{root}:{existing_pythonpath}" if existing_pythonpath else str(root)
    subprocess.run(
        [sys.executable, str(script), *extra_args],
        cwd=str(root),
        env=env,
        check=True,
    )


def _train_user_models(user_id: str, behavior_array: Dict[str, Any]) -> None:
    lock = _training_locks.setdefault(user_id, threading.Lock())
    if not lock.acquire(blocking=False):
        _pending_behavior[user_id] = behavior_array
        _training_status[user_id] = {
            "status": "queued",
            "message": "A training job is already running. Latest reassessment queued.",
            "updated_at": datetime.utcnow().isoformat(),
        }
        logger.info("Training already in progress for user=%s. Queued latest reassessment retrain.", user_id)
        return
    try:
        current_behavior = behavior_array
        while True:
            _training_status[user_id] = {
                "status": "running",
                "message": "PPO training in progress.",
                "updated_at": datetime.utcnow().isoformat(),
            }
            model_dir = _user_model_dir(user_id)
            model_dir.mkdir(parents=True, exist_ok=True)

            user_ppo = model_dir / "ppo_trading_final.zip"
            if user_ppo.exists():
                user_ppo.unlink()

            training_data = _backend_root() / "data" / "training_data.csv"
            if not training_data.exists():
                _run_script("training/download_data.py", [])

            _run_script(
                "training/train_ppo.py",
                [
                    "--model-path",
                    str(model_dir),
                    "--symbol",
                    "ALL",
                    "--behavior-json",
                    json.dumps(current_behavior),
                ],
            )

            metadata = {
                "user_id": user_id,
                "trained_at": datetime.utcnow().isoformat(),
                "behavior_array": current_behavior,
                "model_path": str(user_ppo),
            }
            (model_dir / "meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            _training_status[user_id] = {
                "status": "completed",
                "message": "PPO training completed.",
                "updated_at": datetime.utcnow().isoformat(),
                "model_path": str(user_ppo),
            }
            logger.info("Completed per-user PPO training for user=%s", user_id)

            queued = _pending_behavior.pop(user_id, None)
            if not queued:
                break
            current_behavior = queued
            logger.info("Starting queued reassessment retrain for user=%s", user_id)
    except Exception:
        _training_status[user_id] = {
            "status": "failed",
            "message": "PPO training failed.",
            "updated_at": datetime.utcnow().isoformat(),
        }
        logger.exception("Per-user model training failed for user=%s", user_id)
    finally:
        lock.release()


def trigger_user_retraining(user_id: str, behavior_array: Dict[str, Any]) -> None:
    """
    Fire-and-forget retraining for a user-specific PPO model.
    """
    _training_status[user_id] = {
        "status": "queued",
        "message": "Training request accepted.",
        "updated_at": datetime.utcnow().isoformat(),
    }
    t = threading.Thread(
        target=_train_user_models,
        args=(user_id, behavior_array),
        daemon=True,
    )
    t.start()


def get_user_training_status(user_id: str) -> Dict[str, Any]:
    status = _training_status.get(user_id)
    if status:
        queued = user_id in _pending_behavior
        return {**status, "queued_update_pending": queued}
    return {
        "status": "idle",
        "message": "No training job yet.",
        "updated_at": datetime.utcnow().isoformat(),
        "queued_update_pending": False,
    }
