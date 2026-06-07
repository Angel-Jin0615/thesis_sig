from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.cache import ticker_to_filename
from src.utils.config import ensure_directories_from_config, load_config


def download_one_ticker(
    ticker: str,
    start_date: str,
    end_date: str,
    raw_dir: Path,
) -> Tuple[str, bool, str]:
    """Download one ticker from yfinance and save to raw CSV."""
    out_file = raw_dir / f"{ticker_to_filename(ticker)}.csv"
    try:
        df = yf.download(
            tickers=ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False,
            threads=False,
        )
        if df is None or df.empty:
            return ticker, False, "empty download"

        df.to_csv(out_file)
        return ticker, True, f"saved {out_file.name} ({len(df)} rows)"
    except Exception as exc:  # pragma: no cover
        return ticker, False, str(exc)


def download_universe(config: Dict) -> Tuple[List[str], List[str]]:
    """Download all configured equity and FX tickers."""
    ensure_directories_from_config(config)
    raw_dir = Path(config["raw_data_dir"])
    tickers = list(config.get("tickers", [])) + list(config.get("fx_tickers", []))

    successful: List[str] = []
    failed: List[str] = []
    print(f"Downloading {len(tickers)} tickers from {config['start_date']} to {config['end_date']}")

    for ticker in tickers:
        ticker, ok, msg = download_one_ticker(
            ticker=ticker,
            start_date=config["start_date"],
            end_date=config["end_date"],
            raw_dir=raw_dir,
        )
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {ticker}: {msg}")
        if ok:
            successful.append(ticker)
        else:
            failed.append(ticker)

    print(f"Download complete. Success={len(successful)}, Failed={len(failed)}")
    if failed:
        print("Failed tickers:", ", ".join(failed))
    return successful, failed


def main() -> None:
    config = load_config()
    download_universe(config)


if __name__ == "__main__":
    main()
