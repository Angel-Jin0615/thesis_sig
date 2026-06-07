from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.cache import ticker_to_filename
from src.utils.config import ensure_directories_from_config, load_config


def _flatten_multiindex_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.columns, pd.MultiIndex):
        return df

    flat_cols = []
    for col in df.columns:
        parts = [str(p) for p in col if p and not str(p).startswith("Unnamed")]
        flat_cols.append("_".join(parts) if parts else "unknown")
    df.columns = flat_cols
    return df


def _read_raw_csv(path: Path) -> pd.DataFrame:
    """
    Read raw yfinance CSV robustly, including multi-index header exports.
    """
    df = pd.read_csv(path)
    lower_cols = {c.lower() for c in df.columns}
    if "date" in lower_cols:
        return df

    # Fallback for yfinance multi-index CSV header layout.
    df2 = pd.read_csv(path, header=[0, 1])
    df2 = _flatten_multiindex_columns(df2)
    return df2


def _find_date_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        if "date" in str(col).lower():
            return str(col)
    return str(df.columns[0])


def _find_price_column(df: pd.DataFrame) -> Optional[str]:
    # Prefer adjusted close when available; choose the candidate with most numeric values.
    adj_candidates = [
        c for c in df.columns if "adj close" in str(c).lower() or "adj_close" in str(c).lower()
    ]
    close_candidates = [c for c in df.columns if "close" in str(c).lower()]
    candidates = adj_candidates if adj_candidates else close_candidates
    if not candidates:
        return None

    best_col = None
    best_count = -1
    for col in candidates:
        numeric_count = pd.to_numeric(df[col], errors="coerce").notna().sum()
        if numeric_count > best_count:
            best_count = int(numeric_count)
            best_col = str(col)
    return best_col


def process_one_file(raw_file: Path, output_file: Path) -> Tuple[bool, str]:
    try:
        df = _read_raw_csv(raw_file)
        date_col = _find_date_column(df)
        price_col = _find_price_column(df)
        if price_col is None:
            return False, "price column not found"

        out = pd.DataFrame()
        out["Date"] = pd.to_datetime(df[date_col], errors="coerce")
        out["price"] = pd.to_numeric(df[price_col], errors="coerce")
        out = out.dropna().sort_values("Date").drop_duplicates(subset=["Date"])
        if out.empty:
            return False, "no valid rows after cleaning"

        out["log_price"] = np.log(out["price"])
        out["log_return"] = out["log_price"].diff()
        out = out.dropna().reset_index(drop=True)

        output_file.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_file, index=False)
        return True, f"saved {output_file.name} ({len(out)} rows)"
    except Exception as exc:  # pragma: no cover
        return False, str(exc)


def process_all(config: Dict) -> Tuple[List[str], List[str]]:
    ensure_directories_from_config(config)
    raw_dir = Path(config["raw_data_dir"])
    processed_dir = Path(config["processed_data_dir"])
    tickers = list(config.get("tickers", [])) + list(config.get("fx_tickers", []))

    successful: List[str] = []
    failed: List[str] = []
    for ticker in tickers:
        file_name = f"{ticker_to_filename(ticker)}.csv"
        raw_file = raw_dir / file_name
        processed_file = processed_dir / file_name
        if not raw_file.exists():
            print(f"[FAIL] {ticker}: missing raw file {raw_file.name}")
            failed.append(ticker)
            continue

        ok, msg = process_one_file(raw_file, processed_file)
        print(f"[{'OK' if ok else 'FAIL'}] {ticker}: {msg}")
        if ok:
            successful.append(ticker)
        else:
            failed.append(ticker)

    print(f"Processing complete. Success={len(successful)}, Failed={len(failed)}")
    if failed:
        print("Failed tickers:", ", ".join(failed))
    return successful, failed


def main() -> None:
    config = load_config()
    process_all(config)


if __name__ == "__main__":
    main()
