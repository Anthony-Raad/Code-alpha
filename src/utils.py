"""
Shared helper functions for the music generation project.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MIDI_DIR = DATA_DIR / "midi"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"


def ensure_project_dirs() -> None:
    """Create folders used by the project if they do not already exist."""
    for directory in [MIDI_DIR, PROCESSED_DIR, MODELS_DIR, OUTPUT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def save_pickle(obj: Any, path: Path) -> None:
    """Save a Python object with pickle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(obj, file)


def load_pickle(path: Path) -> Any:
    """Load a Python object from a pickle file."""
    with path.open("rb") as file:
        return pickle.load(file)


def save_json(obj: Any, path: Path) -> None:
    """Save a JSON file with readable indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(obj, file, indent=2)


def load_json(path: Path) -> Any:
    """Load data from a JSON file."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def find_midi_files(directory: Path) -> list[Path]:
    """Return MIDI files from a directory and its subdirectories."""
    extensions = ["*.mid", "*.midi", "*.MID", "*.MIDI"]
    files: list[Path] = []
    for extension in extensions:
        files.extend(directory.rglob(extension))
    return sorted(set(files))
