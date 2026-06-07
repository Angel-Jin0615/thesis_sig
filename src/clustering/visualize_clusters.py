from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def save_cluster_figures(
    merged: pd.DataFrame,
    figures_dir: Path,
    cluster_col: str = "cluster",
    sector_col: str = "sector",
) -> Dict[str, Path]:
    """Generate and save required clustering plots."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Path] = {}
    sns.set_theme(style="whitegrid")

    # 1) PCA by signature cluster
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=merged,
        x="pca1",
        y="pca2",
        hue=cluster_col,
        style=cluster_col,
        s=100,
        palette="tab10",
        ax=ax,
    )
    for _, row in merged.iterrows():
        ax.text(row["pca1"], row["pca2"], row["ticker"], fontsize=8, alpha=0.8)
    ax.set_title("PCA of Mean Signature Features by Cluster")
    out["pca_signature_clusters"] = figures_dir / "pca_signature_clusters.png"
    fig.tight_layout()
    fig.savefig(out["pca_signature_clusters"], dpi=150)
    plt.close(fig)

    # 2) PCA by sector labels
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=merged,
        x="pca1",
        y="pca2",
        hue=sector_col,
        s=100,
        palette="tab20",
        ax=ax,
    )
    for _, row in merged.iterrows():
        ax.text(row["pca1"], row["pca2"], row["ticker"], fontsize=8, alpha=0.8)
    ax.set_title("PCA of Mean Signature Features by Sector")
    out["pca_by_sector"] = figures_dir / "pca_by_sector.png"
    fig.tight_layout()
    fig.savefig(out["pca_by_sector"], dpi=150)
    plt.close(fig)

    # 3) Cluster sector composition
    composition = pd.crosstab(merged[cluster_col], merged[sector_col], normalize="index")
    fig, ax = plt.subplots(figsize=(10, 6))
    composition.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_title("Cluster Sector Composition")
    ax.set_ylabel("Proportion")
    ax.legend(title="Sector", bbox_to_anchor=(1.02, 1), loc="upper left")
    out["cluster_sector_composition"] = figures_dir / "cluster_sector_composition.png"
    fig.tight_layout()
    fig.savefig(out["cluster_sector_composition"], dpi=150)
    plt.close(fig)

    # 4) Cluster volatility boxplot
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.boxplot(
        data=merged,
        x=cluster_col,
        y="annualized_volatility",
        ax=ax,
        hue=cluster_col,
        dodge=False,
        legend=False,
        palette="Set2",
    )
    ax.set_title("Annualized Volatility by Cluster")
    out["cluster_volatility"] = figures_dir / "cluster_volatility.png"
    fig.tight_layout()
    fig.savefig(out["cluster_volatility"], dpi=150)
    plt.close(fig)

    # 5) Cluster drawdown boxplot
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.boxplot(
        data=merged,
        x=cluster_col,
        y="max_drawdown",
        ax=ax,
        hue=cluster_col,
        dodge=False,
        legend=False,
        palette="Set3",
    )
    ax.set_title("Max Drawdown by Cluster")
    out["cluster_drawdown"] = figures_dir / "cluster_drawdown.png"
    fig.tight_layout()
    fig.savefig(out["cluster_drawdown"], dpi=150)
    plt.close(fig)

    # 6) Cluster return distribution
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.violinplot(
        data=merged,
        x=cluster_col,
        y="annualized_mean_return",
        ax=ax,
        hue=cluster_col,
        dodge=False,
        legend=False,
        palette="Pastel1",
    )
    ax.set_title("Annualized Mean Log-Return Distribution by Cluster")
    ax.set_ylabel("Annualized Mean Log Return")
    out["cluster_return_distribution"] = figures_dir / "cluster_return_distribution.png"
    fig.tight_layout()
    fig.savefig(out["cluster_return_distribution"], dpi=150)
    plt.close(fig)

    return out
