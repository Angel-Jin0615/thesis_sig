from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.features.signatures import compute_signature_matrix
from src.features.windows import normalize_windows, time_augment_windows


@dataclass
class SignatureBootstrapGenerator:
    """
    Conditional path generator using kNN in signature space:
    signature(past window) -> sample corresponding future segment.
    """

    signature_library: np.ndarray
    future_library: np.ndarray
    lookback: int
    signature_level: int
    k_neighbors: int = 10
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.signature_library.ndim != 2:
            raise ValueError("signature_library must be 2D")
        if self.future_library.ndim != 2:
            raise ValueError("future_library must be 2D")
        if self.signature_library.shape[0] != self.future_library.shape[0]:
            raise ValueError("signature and future libraries must align in first dimension")
        if self.signature_library.shape[0] == 0:
            raise ValueError("empty signature library")

        # Fit scaler on training library only for consistent distance geometry.
        self.scaler = StandardScaler()
        self.signature_library_scaled = self.scaler.fit_transform(self.signature_library)
        self.k = max(1, min(int(self.k_neighbors), self.signature_library.shape[0]))
        self.nn = NearestNeighbors(n_neighbors=self.k, algorithm="auto")
        self.nn.fit(self.signature_library_scaled)
        self.rng = np.random.default_rng(self.random_state)

    def _signature_of_current_window(self, window_log_prices: np.ndarray) -> np.ndarray:
        if window_log_prices.size != self.lookback + 1:
            raise ValueError("window size must be lookback + 1")
        normalized = normalize_windows(window_log_prices.reshape(1, -1))
        time_aug = time_augment_windows(normalized)
        sig = compute_signature_matrix(time_aug, self.signature_level)
        return sig[0]

    def _sample_future_segment(self, current_window: np.ndarray) -> np.ndarray:
        sig = self._signature_of_current_window(current_window)
        sig_scaled = self.scaler.transform(sig.reshape(1, -1))
        _, indices = self.nn.kneighbors(sig_scaled, n_neighbors=self.k)
        neighbors = indices[0]
        chosen = int(self.rng.choice(neighbors))
        return self.future_library[chosen]

    def generate(self, seed_log_prices: np.ndarray, horizon: int) -> Dict[str, np.ndarray]:
        """
        Generate synthetic continuation from seed path.

        seed_log_prices must contain at least lookback+1 points.
        """
        if seed_log_prices.ndim != 1:
            raise ValueError("seed_log_prices must be 1D")
        if seed_log_prices.size < self.lookback + 1:
            raise ValueError("seed_log_prices length must be at least lookback+1")
        if horizon <= 0:
            raise ValueError("horizon must be positive")

        current_path = seed_log_prices.astype(float).copy()
        n_generated = 0
        while n_generated < horizon:
            window = current_path[-(self.lookback + 1) :]
            segment = self._sample_future_segment(window)
            n_need = min(self.future_library.shape[1], horizon - n_generated)
            # segment is cumulative displacements from the current anchor.
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
