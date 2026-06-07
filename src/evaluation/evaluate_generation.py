from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.evaluation.metrics import compute_financial_metrics, linear_signature_mmd
from src.evaluation.plots import (
    plot_drawdown_distribution,
    plot_generated_paths,
    plot_return_distribution,
    plot_signature_mmd_comparison,
    plot_squared_return_acf,
    plot_var_es_comparison,
)
from src.utils.config import ensure_directories_from_config, load_config


def _load_generated_files(config: Dict) -> List[Path]:
    generated_dir = Path(config["cache_dir"]) / "generated"
    files = sorted(generated_dir.glob("*_generated.npz"))
    if not files:
        raise FileNotFoundError(
            "No generated files found. Run: python src/models/run_signature_bootstrap.py"
        )
    return files


def evaluate_generated_scenarios(config: Dict) -> pd.DataFrame:
    ensure_directories_from_config(config)
    files = _load_generated_files(config)
    tables_dir = Path(config["tables_dir"])
    figures_dir = Path(config["figures_dir"])

    eval_rows: List[Dict] = []
    mmd_rows: List[Dict] = []
    first_payload = None

    for file_path in files:
        payload = np.load(file_path, allow_pickle=True)
        if first_payload is None:
            first_payload = payload
        ticker = str(payload["ticker"])
        real_paths = payload["real_return_paths"].astype(float)
        hist_returns = payload["hist_returns"].astype(float)
        sig_returns = payload["sig_returns"].astype(float)

        real_path_metrics = [compute_financial_metrics(path) for path in real_paths]
        real_metrics = pd.DataFrame(real_path_metrics).mean(numeric_only=True).to_dict()
        real_metrics.update({"ticker": ticker, "model": "real"})
        eval_rows.append(real_metrics)

        hist_path_metrics = [compute_financial_metrics(path) for path in hist_returns]
        sig_path_metrics = [compute_financial_metrics(path) for path in sig_returns]
        hist_avg = pd.DataFrame(hist_path_metrics).mean(numeric_only=True).to_dict()
        sig_avg = pd.DataFrame(sig_path_metrics).mean(numeric_only=True).to_dict()
        hist_avg.update({"ticker": ticker, "model": "historical_bootstrap"})
        sig_avg.update({"ticker": ticker, "model": "signature_bootstrap"})
        eval_rows.append(hist_avg)
        eval_rows.append(sig_avg)

        mmd_hist = linear_signature_mmd(
            real_paths_returns=real_paths,
            generated_paths_returns=hist_returns,
            signature_level=int(config["signature_level"]),
        )
        mmd_sig = linear_signature_mmd(
            real_paths_returns=real_paths,
            generated_paths_returns=sig_returns,
            signature_level=int(config["signature_level"]),
        )
        mmd_rows.append(
            {"ticker": ticker, "model": "historical_bootstrap", "signature_mmd": mmd_hist}
        )
        mmd_rows.append(
            {"ticker": ticker, "model": "signature_bootstrap", "signature_mmd": mmd_sig}
        )

    eval_df = pd.DataFrame(eval_rows)
    mmd_df = pd.DataFrame(mmd_rows)
    eval_df = eval_df.merge(mmd_df, on=["ticker", "model"], how="left")

    table_path = tables_dir / "generative_model_evaluation.csv"
    eval_df.to_csv(table_path, index=False)
    print(f"Saved evaluation table: {table_path}")

    # Required figures:
    # 1) path comparison
    if first_payload is not None:
        real_log = first_payload["real_full_path"].astype(float)
        hist_paths = first_payload["hist_paths_log"].astype(float)
        sig_paths = first_payload["sig_paths_log"].astype(float)
        plot_generated_paths(
            real_log_path=real_log,
            hist_paths=hist_paths,
            sig_paths=sig_paths,
            output_path=figures_dir / "generated_paths_real_vs_bootstrap.png",
        )

        real_ret_paths = first_payload["real_return_paths"].astype(float)
        hist_ret_paths = first_payload["hist_returns"].astype(float)
        sig_ret_paths = first_payload["sig_returns"].astype(float)
        plot_return_distribution(
            real_returns=real_ret_paths,
            hist_returns=hist_ret_paths,
            sig_returns=sig_ret_paths,
            output_path=figures_dir / "return_distribution_real_vs_generated.png",
        )
        plot_squared_return_acf(
            real_returns=real_ret_paths,
            hist_returns=hist_ret_paths,
            sig_returns=sig_ret_paths,
            output_path=figures_dir / "squared_return_acf_comparison.png",
        )
        plot_drawdown_distribution(
            real_paths_returns=real_ret_paths,
            hist_paths_returns=hist_ret_paths,
            sig_paths_returns=sig_ret_paths,
            output_path=figures_dir / "drawdown_distribution_comparison.png",
        )

    plot_var_es_comparison(eval_df, figures_dir / "var_es_comparison.png")
    plot_signature_mmd_comparison(mmd_df, figures_dir / "signature_mmd_comparison.png")

    print(f"Saved figures under: {figures_dir}")
    return eval_df


def main() -> None:
    config = load_config()
    evaluate_generated_scenarios(config)


if __name__ == "__main__":
    main()
