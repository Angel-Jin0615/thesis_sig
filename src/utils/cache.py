from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_npy_if_exists(path: Path) -> Optional[np.ndarray]:
    if path.exists():
        return np.load(path, allow_pickle=False)
    return None


def save_npy(path: Path, arr: np.ndarray) -> None:
    ensure_dir(path.parent)
    np.save(path, arr, allow_pickle=False)


def ticker_to_filename(ticker: str) -> str:
    return (
        ticker.replace("=", "_")
        .replace("^", "")
        .replace("/", "_")
        .replace(":", "_")
        .replace("-", "_")
    )
