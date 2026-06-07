from __future__ import annotations

from typing import Tuple

import numpy as np


def build_rolling_windows(
    log_prices: np.ndarray, lookback: int, forward: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build rolling windows and future segments.

    Past window at anchor t is X[t-lookback:t] inclusive of t (length lookback+1).
    Future segment is X[t+1:t+forward] - X[t] (length forward).
    """
    n = len(log_prices)
    if n < lookback + forward + 1:
        return (
            np.empty((0, lookback + 1), dtype=float),
            np.empty((0, forward), dtype=float),
            np.empty((0,), dtype=int),
        )

    windows = []
    futures = []
    anchor_idx = []
    for t in range(lookback, n - forward):
        past = log_prices[t - lookback : t + 1]
        future = log_prices[t + 1 : t + 1 + forward] - log_prices[t]
        windows.append(past)
        futures.append(future)
        anchor_idx.append(t)

    return (
        np.asarray(windows, dtype=float),
        np.asarray(futures, dtype=float),
        np.asarray(anchor_idx, dtype=int),
    )


def normalize_windows(windows: np.ndarray) -> np.ndarray:
    """Normalize each window so it starts at zero."""
    if windows.size == 0:
        return windows
    return windows - windows[:, [0]]


def time_augment_windows(normalized_windows: np.ndarray) -> np.ndarray:
    """Convert shape (n_samples, length) to (n_samples, length, 2) with time augmentation."""
    if normalized_windows.size == 0:
        return np.empty((0, 0, 2), dtype=float)

    length = normalized_windows.shape[1]
    t = np.linspace(0.0, 1.0, length, dtype=float)
    t_rep = np.repeat(t[None, :], normalized_windows.shape[0], axis=0)
    return np.stack([t_rep, normalized_windows], axis=2)
