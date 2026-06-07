from __future__ import annotations

import numpy as np

try:
    import iisignature
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "iisignature is required for signature extraction. "
        "Install with: pip install iisignature"
    ) from exc


def signature_dimension(path_dim: int, level: int) -> int:
    """Return truncated signature feature dimension."""
    return iisignature.siglength(path_dim, level)


def compute_signature_matrix(time_aug_paths: np.ndarray, level: int) -> np.ndarray:
    """
    Compute truncated signatures for paths.

    Parameters
    ----------
    time_aug_paths : np.ndarray
        Shape (n_samples, path_length, path_dim), usually path_dim=2.
    level : int
        Truncation level.
    """
    if time_aug_paths.ndim != 3:
        raise ValueError("time_aug_paths must be 3D: (n_samples, path_length, path_dim)")
    if time_aug_paths.shape[0] == 0:
        dim = signature_dimension(time_aug_paths.shape[2], level)
        return np.empty((0, dim), dtype=float)

    # `iisignature.prepare` can be unstable on some platforms/builds.
    # Using level directly is slower but robust and mathematically equivalent.
    sigs = [iisignature.sig(path.astype(float), level) for path in time_aug_paths]
    return np.asarray(sigs, dtype=float)


def compute_mean_signature(signature_matrix: np.ndarray) -> np.ndarray:
    if signature_matrix.size == 0:
        return np.empty((0,), dtype=float)
    return signature_matrix.mean(axis=0)
