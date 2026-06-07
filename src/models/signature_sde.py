from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.linear_model import Ridge


@dataclass
class SignatureSDEPlaceholder:
    """
    Optional placeholder for a signature-conditioned SDE model.

    dX_t = mu(sig_t) dt + sigma(sig_t) dW_t
    with mu and log(sigma) estimated using ridge regression.
    """

    alpha: float = 1.0
    sigma_clip: Tuple[float, float] = (1e-4, 0.2)

    def fit(self, signatures: np.ndarray, next_returns: np.ndarray) -> "SignatureSDEPlaceholder":
        if signatures.ndim != 2:
            raise ValueError("signatures must be 2D")
        if next_returns.ndim != 1:
            raise ValueError("next_returns must be 1D")
        if len(signatures) != len(next_returns):
            raise ValueError("signatures and next_returns length mismatch")

        # Simple proxy: fit drift to returns and log-vol to squared residuals.
        self.mu_model = Ridge(alpha=self.alpha).fit(signatures, next_returns)
        resid = next_returns - self.mu_model.predict(signatures)
        vol_target = np.log(np.sqrt(np.maximum(resid**2, 1e-10)))
        self.log_sigma_model = Ridge(alpha=self.alpha).fit(signatures, vol_target)
        return self

    def simulate(
        self,
        signatures: np.ndarray,
        x0: float,
        dt: float = 1.0 / 252.0,
        random_state: int = 42,
    ) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(random_state)
        mu = self.mu_model.predict(signatures)
        sigma = np.exp(self.log_sigma_model.predict(signatures))
        sigma = np.clip(sigma, self.sigma_clip[0], self.sigma_clip[1])

        increments = mu * dt + sigma * np.sqrt(dt) * rng.standard_normal(size=len(mu))
        path = np.concatenate([[x0], x0 + np.cumsum(increments)])
        return {"increments": increments, "log_path": path}
