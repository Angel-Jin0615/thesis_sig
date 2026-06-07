from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.stats import kurtosis, skew
from statsmodels.tsa.stattools import acf

from src.features.signatures import compute_signature_matrix


def max_drawdown_from_returns(log_returns: np.ndarray) -> float:
    if log_returns.size == 0:
        return np.nan
    log_path = np.cumsum(log_returns)
    prices = np.exp(log_path)
    running_max = np.maximum.accumulate(prices)
    drawdowns = prices / running_max - 1.0
    return float(np.min(drawdowns))


def var_es_loss(log_returns: np.ndarray, level: float) -> Dict[str, float]:
    """
    Positive loss convention:
    losses = -returns
    VaR_level = quantile(losses, level)
    ES_level = mean(losses[losses >= VaR_level])
    """
    if log_returns.size == 0:
        return {"var": np.nan, "es": np.nan}
    losses = -log_returns
    var = float(np.quantile(losses, level))
    tail = losses[losses >= var]
    es = float(np.mean(tail)) if tail.size else var
    return {"var": var, "es": es}


def squared_return_acf_values(log_returns: np.ndarray, max_lag: int = 20) -> np.ndarray:
    """ACF of squared daily log returns for lags 1..max_lag (lag 0 removed)."""
    if log_returns.size < 3:
        return np.full(max_lag, np.nan, dtype=float)
    sq = log_returns**2
    nlags = min(max_lag, sq.size - 1)
    vals = acf(sq, nlags=nlags, fft=False)[1:]  # drop lag 0
    if vals.size < max_lag:
        vals = np.concatenate([vals, np.full(max_lag - vals.size, np.nan, dtype=float)])
    return vals


def squared_return_acf_error(
    real_returns_paths: np.ndarray,
    generated_returns_paths: np.ndarray,
    max_lag: int = 20,
) -> float:
    """
    Mean absolute ACF error over lags 1..max_lag.

    ACF is computed per path first, then averaged across paths.
    This avoids flattening multiple paths into one long series and
    prevents artificial path-boundary autocorrelation artifacts.
    """
    real_acfs = np.asarray(
        [squared_return_acf_values(path, max_lag=max_lag) for path in real_returns_paths],
        dtype=float,
    )
    gen_acfs = np.asarray(
        [squared_return_acf_values(path, max_lag=max_lag) for path in generated_returns_paths],
        dtype=float,
    )
    real_mean = np.nanmean(real_acfs, axis=0)
    gen_mean = np.nanmean(gen_acfs, axis=0)
    return float(np.nanmean(np.abs(real_mean - gen_mean)))


def compute_financial_metrics(log_returns: np.ndarray) -> Dict[str, float]:
    if log_returns.size == 0:
        return {
            "mean_return": np.nan,
            "volatility": np.nan,
            "skewness": np.nan,
            "kurtosis": np.nan,
            "max_drawdown": np.nan,
            "var_95": np.nan,
            "var_99": np.nan,
            "es_95": np.nan,
            "es_99": np.nan,
            "acf_squared_lag1": np.nan,
        }

    v95 = var_es_loss(log_returns, 0.95)
    v99 = var_es_loss(log_returns, 0.99)
    acf_vals = squared_return_acf_values(log_returns, max_lag=1)
    acf_sq_lag1 = float(acf_vals[0]) if acf_vals.size else np.nan

    return {
        "mean_return": float(np.mean(log_returns) * 252),
        "volatility": float(np.std(log_returns, ddof=1) * np.sqrt(252)),
        "skewness": float(skew(log_returns, bias=False)),
        "kurtosis": float(kurtosis(log_returns, fisher=True, bias=False)),
        "max_drawdown": max_drawdown_from_returns(log_returns),
        "var_95": v95["var"],
        "var_99": v99["var"],
        "es_95": v95["es"],
        "es_99": v99["es"],
        "acf_squared_lag1": acf_sq_lag1,
    }


def _path_signature_from_returns(paths_returns: np.ndarray, level: int) -> np.ndarray:
    """
    Convert return paths to time-augmented log-price paths and compute signatures.

    paths_returns shape: (n_paths, horizon)
    """
    if paths_returns.ndim != 2:
        raise ValueError("paths_returns must be 2D")
    n_paths, horizon = paths_returns.shape
    log_paths = np.cumsum(paths_returns, axis=1)
    log_paths = np.concatenate([np.zeros((n_paths, 1)), log_paths], axis=1)
    t = np.linspace(0.0, 1.0, horizon + 1, dtype=float)
    t_mat = np.repeat(t[None, :], n_paths, axis=0)
    augmented = np.stack([t_mat, log_paths], axis=2)
    return compute_signature_matrix(augmented, level=level)


def linear_signature_mmd(
    real_paths_returns: np.ndarray,
    generated_paths_returns: np.ndarray,
    signature_level: int,
) -> float:
    """
    Linear-kernel signature MMD:
    || mean(sig(real)) - mean(sig(gen)) ||^2
    """
    if real_paths_returns.size == 0 or generated_paths_returns.size == 0:
        return np.nan
    real_sig = _path_signature_from_returns(real_paths_returns, signature_level)
    gen_sig = _path_signature_from_returns(generated_paths_returns, signature_level)
    diff = real_sig.mean(axis=0) - gen_sig.mean(axis=0)
    return float(np.dot(diff, diff))
