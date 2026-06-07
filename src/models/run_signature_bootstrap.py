from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.models.historical_bootstrap import HistoricalBootstrapGenerator
from src.models.signature_bootstrap import SignatureBootstrapGenerator
from src.utils.cache import load_npy_if_exists, ticker_to_filename
from src.utils.config import ensure_directories_from_config, load_config


def _load_cached_libraries(ticker: str, config: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache_dir = Path(config["cache_dir"])
    safe = ticker_to_filename(ticker)
    lookback = int(config["lookback"])
    level = int(config["signature_level"])
    forward = int(config["forward"])

    sig_path = cache_dir / "signatures" / f"{safe}_lookback{lookback}_level{level}.npy"
    fut_path = cache_dir / "future_segments" / f"{safe}_lookback{lookback}_forward{forward}.npy"
    anc_path = cache_dir / "future_segments" / f"{safe}_anchor_indices_lookback{lookback}.npy"
    sig = load_npy_if_exists(sig_path)
    fut = load_npy_if_exists(fut_path)
    anc = load_npy_if_exists(anc_path)
    if sig is None or fut is None or anc is None:
        raise FileNotFoundError(f"Missing cached signature artifacts for {ticker}")
    return sig, fut, anc


def _build_real_test_windows(test_returns: np.ndarray, horizon: int) -> np.ndarray:
    if len(test_returns) < horizon:
        return np.empty((0, horizon), dtype=float)
    windows = [test_returns[i : i + horizon] for i in range(len(test_returns) - horizon + 1)]
    return np.asarray(windows, dtype=float)


def generate_for_ticker(ticker: str, config: Dict) -> Path:
    lookback = int(config["lookback"])
    forward = int(config["forward"])
    horizon_cfg = int(config["generation_horizon"])
    train_ratio = float(config["train_ratio"])
    n_paths = int(config["n_generated_paths"])

    processed_file = Path(config["processed_data_dir"]) / f"{ticker_to_filename(ticker)}.csv"
    if not processed_file.exists():
        raise FileNotFoundError(f"Processed file missing for {ticker}: {processed_file}")

    df = pd.read_csv(processed_file)
    log_prices = df["log_price"].to_numpy(dtype=float)
    if len(log_prices) < lookback + forward + 50:
        raise ValueError(f"Not enough rows for {ticker}")

    split_idx = int(train_ratio * len(log_prices))
    split_idx = max(split_idx, lookback + 1)
    split_idx = min(split_idx, len(log_prices) - 1)

    seed_start = split_idx - lookback - 1
    if seed_start < 0:
        raise ValueError(f"Insufficient training history for seed window on {ticker}")
    seed = log_prices[seed_start:split_idx]
    if len(seed) != lookback + 1:
        raise ValueError(f"Seed length mismatch for {ticker}: got {len(seed)}")

    available_test = len(log_prices) - split_idx
    horizon = min(horizon_cfg, available_test)
    if horizon <= 0:
        raise ValueError(f"No test horizon available for {ticker}")

    real_future = log_prices[split_idx : split_idx + horizon]
    real_full_path = np.concatenate([seed, real_future])
    test_returns = np.diff(log_prices[split_idx - 1 :])
    real_returns_window = test_returns[:horizon]
    real_return_paths = _build_real_test_windows(test_returns, horizon)

    signatures, futures, anchors = _load_cached_libraries(ticker, config)
    train_mask = anchors + forward <= split_idx - 1
    train_signatures = signatures[train_mask]
    train_futures = futures[train_mask]
    if train_signatures.shape[0] == 0:
        raise ValueError(f"No train windows after chronological split for {ticker}")

    hist_model = HistoricalBootstrapGenerator(
        future_segments=train_futures,
        forward=forward,
        random_state=42,
    )
    sig_model = SignatureBootstrapGenerator(
        signature_library=train_signatures,
        future_library=train_futures,
        lookback=lookback,
        signature_level=int(config["signature_level"]),
        k_neighbors=int(config["k_neighbors"]),
        random_state=42,
    )

    hist_paths_log = []
    sig_paths_log = []
    hist_returns = []
    sig_returns = []
    for i in range(n_paths):
        hist = hist_model.generate(seed, horizon=horizon)
        sig = sig_model.generate(seed, horizon=horizon)
        hist_paths_log.append(hist["full_log_path"])
        sig_paths_log.append(sig["full_log_path"])
        hist_returns.append(hist["generated_log_returns"])
        sig_returns.append(sig["generated_log_returns"])

    hist_paths_log_arr = np.asarray(hist_paths_log, dtype=float)
    sig_paths_log_arr = np.asarray(sig_paths_log, dtype=float)
    hist_returns_arr = np.asarray(hist_returns, dtype=float)
    sig_returns_arr = np.asarray(sig_returns, dtype=float)

    generated_dir = Path(config["cache_dir"]) / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    out_path = generated_dir / f"{ticker_to_filename(ticker)}_generated.npz"
    np.savez(
        out_path,
        ticker=ticker,
        split_idx=split_idx,
        lookback=lookback,
        forward=forward,
        horizon=horizon,
        seed=seed,
        real_full_path=real_full_path,
        real_returns_window=real_returns_window,
        real_return_paths=real_return_paths,
        hist_paths_log=hist_paths_log_arr,
        sig_paths_log=sig_paths_log_arr,
        hist_returns=hist_returns_arr,
        sig_returns=sig_returns_arr,
    )
    return out_path


def run_generation(config: Dict) -> List[Path]:
    ensure_directories_from_config(config)
    targets = ["SPY"]
    fx_tickers = list(config.get("fx_tickers", []))
    if fx_tickers:
        targets.append(fx_tickers[0])

    outputs: List[Path] = []
    failed: List[str] = []
    print(f"Generating scenarios for targets: {targets}")
    for ticker in targets:
        try:
            out_path = generate_for_ticker(ticker, config)
            print(f"[OK] {ticker}: saved generated paths to {out_path}")
            outputs.append(out_path)
        except Exception as exc:
            print(f"[FAIL] {ticker}: {exc}")
            failed.append(ticker)

    if failed:
        print("Failed generation tickers:", ", ".join(failed))
    return outputs


def main() -> None:
    config = load_config()
    run_generation(config)


if __name__ == "__main__":
    main()
