from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.stattools import acf

from src.evaluation.metrics import max_drawdown_from_returns


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _mean_squared_acf(paths_returns: np.ndarray, max_lag: int = 20) -> np.ndarray:
    """
    Compute mean ACF of squared daily log returns across paths for lags 1..max_lag.
    Avoids flattening across paths (which creates artificial boundary jumps).
    """
    arr = np.asarray(paths_returns, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    acf_rows = []
    for path in arr:
        if path.size < 3:
            acf_rows.append(np.full(max_lag, np.nan, dtype=float))
            continue
        nlags = min(max_lag, path.size - 1)
        vals = acf(path**2, nlags=nlags, fft=False)[1:]  # drop lag 0
        if vals.size < max_lag:
            vals = np.concatenate([vals, np.full(max_lag - vals.size, np.nan, dtype=float)])
        acf_rows.append(vals)
    return np.nanmean(np.asarray(acf_rows, dtype=float), axis=0)


def plot_generated_paths(
    real_log_path: np.ndarray,
    hist_paths: np.ndarray,
    sig_paths: np.ndarray,
    output_path: Path,
) -> None:
    _ensure_parent(output_path)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))

    for i in range(min(20, hist_paths.shape[0])):
        ax.plot(hist_paths[i], color="tab:orange", alpha=0.2, linewidth=1)
    for i in range(min(20, sig_paths.shape[0])):
        ax.plot(sig_paths[i], color="tab:green", alpha=0.2, linewidth=1)

    ax.plot(real_log_path, color="black", linewidth=2, label="Real test path")
    ax.plot(hist_paths.mean(axis=0), color="tab:orange", linewidth=2, label="Historical bootstrap mean")
    ax.plot(sig_paths.mean(axis=0), color="tab:green", linewidth=2, label="Signature bootstrap mean")
    ax.set_title("Real vs Generated Log-Price Paths")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Log price")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_return_distribution(
    real_returns: np.ndarray,
    hist_returns: np.ndarray,
    sig_returns: np.ndarray,
    output_path: Path,
) -> None:
    _ensure_parent(output_path)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.kdeplot(real_returns.flatten(), fill=False, label="Real", ax=ax, linewidth=2)
    sns.kdeplot(hist_returns.flatten(), fill=False, label="Historical bootstrap", ax=ax, linewidth=2)
    sns.kdeplot(sig_returns.flatten(), fill=False, label="Signature bootstrap", ax=ax, linewidth=2)
    ax.set_title("Return Distribution: Real vs Generated")
    ax.set_xlabel("Log return")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_squared_return_acf(
    real_returns: np.ndarray,
    hist_returns: np.ndarray,
    sig_returns: np.ndarray,
    output_path: Path,
    max_lag: int = 20,
) -> None:
    _ensure_parent(output_path)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    real_acf = _mean_squared_acf(real_returns, max_lag=max_lag)
    hist_acf = _mean_squared_acf(hist_returns, max_lag=max_lag)
    sig_acf = _mean_squared_acf(sig_returns, max_lag=max_lag)

    lags = np.arange(1, max_lag + 1)
    ax.plot(lags, real_acf, marker="o", label="Real")
    ax.plot(lags, hist_acf, marker="o", label="Historical bootstrap")
    ax.plot(lags, sig_acf, marker="o", label="Signature bootstrap")
    ax.set_title("ACF of Squared Daily Log Returns (Lags 1-20)")
    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_drawdown_distribution(
    real_paths_returns: np.ndarray,
    hist_paths_returns: np.ndarray,
    sig_paths_returns: np.ndarray,
    output_path: Path,
) -> None:
    _ensure_parent(output_path)
    sns.set_theme(style="whitegrid")

    real_dd = [max_drawdown_from_returns(path) for path in real_paths_returns]
    hist_dd = [max_drawdown_from_returns(path) for path in hist_paths_returns]
    sig_dd = [max_drawdown_from_returns(path) for path in sig_paths_returns]

    df = pd.DataFrame(
        {
            "drawdown": real_dd + hist_dd + sig_dd,
            "model": (["Real"] * len(real_dd))
            + (["Historical bootstrap"] * len(hist_dd))
            + (["Signature bootstrap"] * len(sig_dd)),
        }
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x="model", y="drawdown", ax=ax)
    ax.set_title("Max Drawdown Distribution Comparison")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_var_es_comparison(metrics_df: pd.DataFrame, output_path: Path) -> None:
    _ensure_parent(output_path)
    sns.set_theme(style="whitegrid")

    plot_df = metrics_df.melt(
        id_vars=["ticker", "model"],
        value_vars=["var_95", "var_99", "es_95", "es_99"],
        var_name="metric",
        value_name="value",
    )
    plot_df = plot_df.dropna(subset=["value"])
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=plot_df, x="metric", y="value", hue="model", ax=ax, errorbar=None)
    ax.set_title("VaR / ES Comparison (Positive Loss Metrics)")
    ax.set_ylabel("Loss")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_signature_mmd_comparison(mmd_df: pd.DataFrame, output_path: Path) -> None:
    _ensure_parent(output_path)
    sns.set_theme(style="whitegrid")
    plot_df = mmd_df[mmd_df["model"].isin(["historical_bootstrap", "signature_bootstrap"])].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=plot_df, x="ticker", y="signature_mmd", hue="model", ax=ax, errorbar=None)
    ax.set_title("Linear Signature MMD Comparison")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_acf_best_overall_vs_multiday(
    real_returns: np.ndarray,
    historical_returns: np.ndarray,
    signature_overall_returns: np.ndarray,
    signature_multiday_returns: np.ndarray,
    output_path: Path,
    max_lag: int = 20,
) -> None:
    """Compare squared-return ACF (lags 1..20) for real/historical/signature variants."""
    _ensure_parent(output_path)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    real_acf = _mean_squared_acf(real_returns, max_lag=max_lag)
    hist_acf = _mean_squared_acf(historical_returns, max_lag=max_lag)
    sig_overall_acf = _mean_squared_acf(signature_overall_returns, max_lag=max_lag)
    sig_multiday_acf = _mean_squared_acf(signature_multiday_returns, max_lag=max_lag)

    lags = np.arange(1, max_lag + 1)
    ax.plot(lags, real_acf, marker="o", label="Real")
    ax.plot(lags, hist_acf, marker="o", label="Historical bootstrap")
    ax.plot(lags, sig_overall_acf, marker="o", label="Signature bootstrap (best overall)")
    ax.plot(lags, sig_multiday_acf, marker="o", label="Signature bootstrap (best multiday)")
    ax.set_title("ACF of Squared Daily Log Returns: Overall vs Multiday Configs")
    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
