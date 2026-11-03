"""AlphaPulse v2.0 - Model and Data Serialization Utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import torch


def save_model_bundle(
    save_dir: Path | str,
    model: torch.nn.Module,
    scaler: Any,
    meta: dict[str, Any],
    history: dict[str, list[float]] | None = None,
) -> Path:
    """Saves model weights, scaler, metadata and training history into a structured directory."""
    target_path = Path(save_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    # 1. Weights
    torch.save(model.state_dict(), target_path / "model.pt")

    # 2. Scaler
    joblib.dump(scaler, target_path / "scaler.pkl")

    # 3. Metadata (saved as human-readable JSON)
    with open(target_path / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

    # 4. History
    if history is not None:
        with open(target_path / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    return target_path


def load_model_bundle(
    load_dir: Path | str,
    device: torch.device | None = None,
) -> tuple[dict[str, Any], Any, dict[str, Any], dict[str, list[float]] | None]:
    """Loads weights state_dict, scaler, metadata and history from directory."""
    target_path = Path(load_dir)
    if not target_path.exists():
        raise FileNotFoundError(f"Model directory not found: {target_path}")

    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load metadata
    meta_file = target_path / "meta.json"
    if not meta_file.exists():
        raise FileNotFoundError(f"Metadata file missing: {meta_file}")
    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Load weights
    weights_file = target_path / "model.pt"
    if not weights_file.exists():
        raise FileNotFoundError(f"Weights file missing: {weights_file}")
    state_dict = torch.load(weights_file, map_location=dev)

    # Load scaler
    scaler_file = target_path / "scaler.pkl"
    scaler = joblib.load(scaler_file) if scaler_file.exists() else None

    # Load history if present
    history_file = target_path / "history.json"
    history = None
    if history_file.exists():
        with open(history_file, "r", encoding="utf-8") as f:
            history = json.load(f)

    return meta, state_dict, scaler, history

