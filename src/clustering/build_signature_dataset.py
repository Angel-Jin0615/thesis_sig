from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.features.signatures import compute_mean_signature, compute_signature_matrix
from src.features.windows import build_rolling_windows, normalize_windows, time_augment_windows
from src.utils.cache import load_npy_if_exists, save_npy, ticker_to_filename
from src.utils.config import ensure_directories_from_config, load_config
from src.utils.parallel import run_parallel


@dataclass
class ProcessResult:
    ticker: str
    status: str
    message: str
    n_samples: int = 0


def _cache_paths(ticker: str, config: Dict) -> Dict[str, Path]:
    cache_dir = Path(config["cache_dir"])
    lookback = config["lookback"]
    forward = config["forward"]
    level = config["signature_level"]
    safe = ticker_to_filename(ticker)
    return {
        "sig": cache_dir / "signatures" / f"{safe}_lookback{lookback}_level{level}.npy",
        "future": cache_dir
        / "future_segments"
        / f"{safe}_lookback{lookback}_forward{forward}.npy",
        "mean": cache_dir
        / "mean_signatures"
        / f"{safe}_mean_signature_lookback{lookback}_level{level}.npy",
        "anchor": cache_dir / "future_segments" / f"{safe}_anchor_indices_lookback{lookback}.npy",
    }


def process_one_ticker(ticker: str, config: Dict) -> ProcessResult:
    """Build or load cached signatures/future segments for one ticker."""
    paths = _cache_paths(ticker, config)
    sig_cached = load_npy_if_exists(paths["sig"])
    fut_cached = load_npy_if_exists(paths["future"])
    mean_cached = load_npy_if_exists(paths["mean"])
    anchor_cached = load_npy_if_exists(paths["anchor"])

    if (
        sig_cached is not None
        and fut_cached is not None
        and mean_cached is not None
        and anchor_cached is not None
    ):
        return ProcessResult(
            ticker=ticker,
            status="loaded_cache",
            message="all cached artifacts found",
            n_samples=int(sig_cached.shape[0]),
        )

    processed_file = Path(config["processed_data_dir"]) / f"{ticker_to_filename(ticker)}.csv"
    if not processed_file.exists():
        return ProcessResult(ticker=ticker, status="failed", message="missing processed CSV")

    try:
        df = pd.read_csv(processed_file)
        if "log_price" not in df.columns:
            return ProcessResult(ticker=ticker, status="failed", message="log_price column missing")

        log_prices = df["log_price"].to_numpy(dtype=float)
        windows, future_segments, anchor_idx = build_rolling_windows(
            log_prices=log_prices,
            lookback=int(config["lookback"]),
            forward=int(config["forward"]),
        )
        if windows.shape[0] == 0:
            return ProcessResult(
                ticker=ticker,
                status="failed",
                message="not enough rows for requested lookback+forward",
            )

        normalized = normalize_windows(windows)
        time_aug = time_augment_windows(normalized)
        signature_matrix = compute_signature_matrix(time_aug, int(config["signature_level"]))
        mean_signature = compute_mean_signature(signature_matrix)

        save_npy(paths["sig"], signature_matrix)
        save_npy(paths["future"], future_segments)
        save_npy(paths["mean"], mean_signature)
        save_npy(paths["anchor"], anchor_idx)

        return ProcessResult(
            ticker=ticker,
            status="processed",
            message="computed and cached",
            n_samples=int(signature_matrix.shape[0]),
        )
    except Exception as exc:  # pragma: no cover
        return ProcessResult(ticker=ticker, status="failed", message=str(exc))


def build_signature_dataset(config: Dict) -> List[ProcessResult]:
    ensure_directories_from_config(config)
    tickers = list(config.get("tickers", [])) + list(config.get("fx_tickers", []))

    def _runner(t: str) -> ProcessResult:
        return process_one_ticker(t, config)

    results = run_parallel(_runner, tickers, n_jobs=int(config.get("n_jobs", -1)))
    failed = [r.ticker for r in results if r.status == "failed"]

    for r in results:
        print(f"[{r.status.upper()}] {r.ticker}: {r.message} (samples={r.n_samples})")
    print(f"Signature dataset complete. Success={len(results) - len(failed)}, Failed={len(failed)}")
    if failed:
        print("Failed tickers:", ", ".join(failed))

    return results


def main() -> None:
    config = load_config()
    build_signature_dataset(config)


if __name__ == "__main__":
    main()
