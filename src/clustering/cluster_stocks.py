from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.clustering.visualize_clusters import save_cluster_figures
from src.utils.cache import load_npy_if_exists, ticker_to_filename
from src.utils.config import ensure_directories_from_config, load_config


SECTOR_MAP = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "NVDA": "Technology",
    "AMZN": "Consumer Discretionary",
    "META": "Communication Services",
    "JPM": "Financials",
    "BAC": "Financials",
    "GS": "Financials",
    "MS": "Financials",
    "XOM": "Energy",
    "CVX": "Energy",
    "JNJ": "Healthcare",
    "PFE": "Healthcare",
    "MRK": "Healthcare",
    "WMT": "Consumer Staples",
    "COST": "Consumer Staples",
    "PG": "Consumer Staples",
    "SPY": "ETF",
    "QQQ": "ETF",
    "IWM": "ETF",
    "EURUSD=X": "FX",
    "GBPUSD=X": "FX",
    "JPY=X": "FX",
    "CAD=X": "FX",
    "AUDUSD=X": "FX",
}


def max_drawdown_from_log_prices(log_prices: np.ndarray) -> float:
    prices = np.exp(log_prices)
    running_max = np.maximum.accumulate(prices)
    drawdown = prices / running_max - 1.0
    return float(drawdown.min())


def compute_financial_summary(processed_file: Path) -> Dict[str, float]:
    df = pd.read_csv(processed_file)
    ret = df["log_return"].dropna().to_numpy(dtype=float)
    log_prices = df["log_price"].to_numpy(dtype=float)
    if ret.size == 0:
        return {
            "annualized_volatility": np.nan,
            "skewness": np.nan,
            "kurtosis": np.nan,
            "max_drawdown": np.nan,
            "average_rolling_volatility": np.nan,
            "daily_mean_return": np.nan,
            "annualized_mean_return": np.nan,
        }

    rolling_vol = pd.Series(ret).rolling(20).std().dropna().to_numpy(dtype=float)
    daily_mean_return = float(np.mean(ret))
    return {
        "annualized_volatility": float(np.std(ret, ddof=1) * np.sqrt(252)),
        "skewness": float(skew(ret, bias=False)),
        "kurtosis": float(kurtosis(ret, fisher=True, bias=False)),
        "max_drawdown": max_drawdown_from_log_prices(log_prices),
        "average_rolling_volatility": float(np.nanmean(rolling_vol) * np.sqrt(252))
        if rolling_vol.size
        else np.nan,
        "daily_mean_return": daily_mean_return,
        "annualized_mean_return": float(daily_mean_return * 252),
    }


def build_mean_signature_matrix(config: Dict) -> pd.DataFrame:
    tickers = list(config.get("tickers", [])) + list(config.get("fx_tickers", []))
    cache_dir = Path(config["cache_dir"])
    lookback = config["lookback"]
    level = config["signature_level"]
    rows: List[Dict] = []

    for ticker in tickers:
        safe = ticker_to_filename(ticker)
        mean_path = (
            cache_dir
            / "mean_signatures"
            / f"{safe}_mean_signature_lookback{lookback}_level{level}.npy"
        )
        mean_sig = load_npy_if_exists(mean_path)
        if mean_sig is None or mean_sig.size == 0:
            print(f"[SKIP] {ticker}: missing mean signature cache")
            continue

        row = {"ticker": ticker, "sector": SECTOR_MAP.get(ticker, "Unknown")}
        for i, value in enumerate(mean_sig):
            row[f"sig_{i}"] = float(value)
        rows.append(row)

    if not rows:
        raise RuntimeError("No mean signatures available. Run build_signature_dataset first.")
    return pd.DataFrame(rows)


def cluster_stocks(config: Dict) -> pd.DataFrame:
    ensure_directories_from_config(config)
    mean_df = build_mean_signature_matrix(config)
    sig_cols = [c for c in mean_df.columns if c.startswith("sig_")]
    X = mean_df[sig_cols].to_numpy(dtype=float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    n_clusters = min(int(config["n_clusters"]), len(mean_df))
    if n_clusters < 2:
        n_clusters = 1
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X_scaled)

    out = mean_df.copy()
    out["cluster"] = labels
    out["pca1"] = X_pca[:, 0]
    out["pca2"] = X_pca[:, 1]

    financial_rows = []
    for ticker in out["ticker"]:
        file_path = Path(config["processed_data_dir"]) / f"{ticker_to_filename(ticker)}.csv"
        stats = compute_financial_summary(file_path) if file_path.exists() else {}
        financial_rows.append(stats)
    financial_df = pd.DataFrame(financial_rows)
    merged = pd.concat([out.reset_index(drop=True), financial_df.reset_index(drop=True)], axis=1)

    tables_dir = Path(config["tables_dir"])
    figures_dir = Path(config["figures_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    membership_path = tables_dir / "cluster_membership.csv"
    merged.to_csv(membership_path, index=False)
    print(f"Saved cluster membership: {membership_path}")

    summary = (
        merged.groupby("cluster")
        .agg(
            n_tickers=("ticker", "count"),
            mean_ann_vol=("annualized_volatility", "mean"),
            mean_skewness=("skewness", "mean"),
            mean_kurtosis=("kurtosis", "mean"),
            mean_max_drawdown=("max_drawdown", "mean"),
            mean_avg_roll_vol=("average_rolling_volatility", "mean"),
            mean_annualized_return=("annualized_mean_return", "mean"),
        )
        .reset_index()
    )
    dominant_sector = (
        merged.groupby("cluster")["sector"]
        .agg(lambda s: s.value_counts().index[0] if not s.empty else "Unknown")
        .reset_index(name="dominant_sector")
    )
    summary = summary.merge(dominant_sector, on="cluster", how="left")
    summary_path = tables_dir / "cluster_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved cluster summary: {summary_path}")

    saved = save_cluster_figures(merged, figures_dir)
    for key, path in saved.items():
        print(f"Saved figure ({key}): {path}")

    return merged


def main() -> None:
    config = load_config()
    cluster_stocks(config)


if __name__ == "__main__":
    main()
