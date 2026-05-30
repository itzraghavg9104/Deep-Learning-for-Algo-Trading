"""
Model bootstrap utilities.

Ensures required model artifacts exist at startup and trains them when missing.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)


def _backend_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_model_dir() -> Path:
    model_path = Path(settings.MODEL_PATH)
    if model_path.is_absolute():
        return model_path
    return _backend_root() / model_path


def _run_script(script_rel_path: str) -> None:
    backend_root = _backend_root()
    script_path = backend_root / script_rel_path
    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path}")

    logger.info("Running bootstrap script: %s", script_path)
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{backend_root}:{existing_pythonpath}" if existing_pythonpath else str(backend_root)
    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(backend_root),
        env=env,
        check=True,
    )


def ensure_models_ready() -> None:
    """
    Ensure LSTM and PPO model files are present.

    Behavior:
    - If both model files exist: do nothing.
    - If any model is missing and AUTO_TRAIN_IF_MISSING=True:
      1) Download/prepare training data if needed.
      2) Train only the missing models.
    """
    model_dir = _resolve_model_dir()
    lstm_model = model_dir / "lstm_final.pt"
    ppo_model = model_dir / "ppo_trading_final.zip"

    missing_lstm = not lstm_model.exists()
    missing_ppo = not ppo_model.exists()

    if not missing_lstm and not missing_ppo:
        logger.info("All required model artifacts found. Skipping bootstrap training.")
        return

    logger.warning(
        "Missing model artifacts. LSTM missing=%s, PPO missing=%s",
        missing_lstm,
        missing_ppo,
    )

    if not settings.AUTO_TRAIN_IF_MISSING:
        logger.warning(
            "AUTO_TRAIN_IF_MISSING is disabled. Server will run without missing models."
        )
        return

    try:
        training_data = _backend_root() / "data" / "training_data.csv"
        if not training_data.exists():
            _run_script("training/download_data.py")

        if missing_lstm:
            _run_script("training/train_lstm.py")

        if missing_ppo:
            _run_script("training/train_ppo.py")

        logger.info("Model bootstrap completed.")
    except Exception:
        logger.exception("Model bootstrap failed.")
        if settings.AUTO_TRAIN_STRICT:
            raise
