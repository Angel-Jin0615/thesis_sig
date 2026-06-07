from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class HistoricalBootstrapGenerator:
    """
    Unconditional baseline generator:
    sample future segments uniformly from historical library.
    """

    future_segments: np.ndarray
    forward: int
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.future_segments.ndim != 2:
            raise ValueError("future_segments must be 2D: (n_samples, forward)")
        if self.future_segments.shape[0] == 0:
            raise ValueError("future_segments is empty")
        self.rng = np.random.default_rng(self.random_state)

    def generate(self, seed_log_prices: np.ndarray, horizon: int) -> Dict[str, np.ndarray]:
        """
        Generate a synthetic path continuation from seed log-prices.

        `future_segments` are cumulative displacements from segment anchor:
        segment[j] = X_{t+j+1} - X_t, j=0..forward-1
        We append as levels and then derive daily returns via `np.diff`.
        """
        if seed_log_prices.ndim != 1 or seed_log_prices.size == 0:
            raise ValueError("seed_log_prices must be non-empty 1D array")
        if horizon <= 0:
            raise ValueError("horizon must be positive")

        current_path = seed_log_prices.astype(float).copy()
        n_generated = 0
        while n_generated < horizon:
            idx = self.rng.integers(0, self.future_segments.shape[0])
            segment = self.future_segments[idx]
            n_need = min(self.forward, horizon - n_generated)
            new_points = current_path[-1] + segment[:n_need]
            current_path = np.concatenate([current_path, new_points])
            n_generated += n_need

        full_path = current_path
        generated_log_returns = np.diff(full_path[seed_log_prices.size - 1 :])
        return {
            "generated_log_returns": generated_log_returns,
            "increments": generated_log_returns,  # backward-compat alias
            "full_log_path": full_path,
        }
