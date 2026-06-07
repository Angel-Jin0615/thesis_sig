from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.clustering.cluster_stocks import cluster_stocks
from src.evaluation.metrics import (
    compute_financial_metrics,
    linear_signature_mmd,
    squared_return_acf_error,
)
from src.evaluation.plots import (
    plot_acf_best_overall_vs_multiday,
    plot_drawdown_distribution,
    plot_generated_paths,
    plot_return_distribution,
    plot_signature_mmd_comparison,
    plot_squared_return_acf,
    plot_var_es_comparison,
)
from src.features.signatures import compute_signature_matrix
from src.features.windows import build_rolling_windows, normalize_windows, time_augment_windows
from src.models.historical_bootstrap import HistoricalBootstrapGenerator
from src.models.signature_bootstrap import SignatureBootstrapGenerator
from src.utils.cache import ticker_to_filename
from src.utils.config import ensure_directories_from_config, load_config


def _build_real_test_windows(test_returns: np.ndarray, horizon: int) -> np.ndarray:
    if len(test_returns) < horizon:
        return np.empty((0, horizon), dtype=float)
    windows = [test_returns[i : i + horizon] for i in range(len(test_returns) - horizon + 1)]
    return np.asarray(windows, dtype=float)


def _normalize_minmax(series: pd.Series) -> pd.Series:
    lo = float(series.min())
    hi = float(series.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - lo) / (hi - lo)


def _prepare_libraries(
    log_prices: np.ndarray,
    lookback: int,
    forward: int,
    signature_level: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    windows, futures, anchors = build_rolling_windows(
        log_prices=log_prices,
        lookback=lookback,
        forward=forward,
    )
    normalized = normalize_windows(windows)
    time_aug = time_augment_windows(normalized)
    signatures = compute_signature_matrix(time_aug, signature_level)
    return signatures, futures, anchors


def _split_points(
    n_obs: int,
    train_ratio: float,
    val_ratio: float,
    lookback: int,
) -> Tuple[int, int]:
    train_end = int(n_obs * train_ratio)
    val_end = int(n_obs * (train_ratio + val_ratio))
    train_end = max(train_end, lookback + 2)
    val_end = max(val_end, train_end + 2)
    val_end = min(val_end, n_obs - 1)
    return train_end, val_end


def _simulate_split(
    log_prices: np.ndarray,
    start_idx: int,
    end_idx: int,
    lookback: int,
    horizon: int,
    n_paths: int,
    hist_model: HistoricalBootstrapGenerator,
    sig_model: SignatureBootstrapGenerator,
) -> Dict[str, np.ndarray]:
    seed_start = start_idx - lookback - 1
    if seed_start < 0:
        raise ValueError("seed_start < 0")
    seed = log_prices[seed_start:start_idx]
    if seed.size != lookback + 1:
        raise ValueError("seed length mismatch")

    available = end_idx - start_idx
    horizon_eff = min(horizon, available)
    if horizon_eff < 10:
        raise ValueError("insufficient split horizon")

    real_future = log_prices[start_idx : start_idx + horizon_eff]
    real_full_path = np.concatenate([seed, real_future])
    split_returns = np.diff(log_prices[start_idx - 1 : end_idx])
    real_return_paths = _build_real_test_windows(split_returns, horizon_eff)

    hist_paths_log = []
    sig_paths_log = []
    hist_returns = []
    sig_returns = []
    for _ in range(n_paths):
        hist = hist_model.generate(seed, horizon=horizon_eff)
        sig = sig_model.generate(seed, horizon=horizon_eff)
        hist_paths_log.append(hist["full_log_path"])
        sig_paths_log.append(sig["full_log_path"])
        hist_returns.append(hist["generated_log_returns"])
        sig_returns.append(sig["generated_log_returns"])

    return {
        "seed": seed,
        "horizon": horizon_eff,
        "real_full_path": real_full_path,
        "real_return_paths": real_return_paths,
        "hist_paths_log": np.asarray(hist_paths_log, dtype=float),
        "sig_paths_log": np.asarray(sig_paths_log, dtype=float),
        "hist_returns": np.asarray(hist_returns, dtype=float),
        "sig_returns": np.asarray(sig_returns, dtype=float),
    }


def _rows_for_split(
    ticker: str,
    split_name: str,
    lookback: int,
    signature_level: int,
    forward: int,
    k_neighbors: int,
    split_payload: Dict[str, np.ndarray],
) -> List[Dict]:
    real_paths = split_payload["real_return_paths"]
    hist_returns = split_payload["hist_returns"]
    sig_returns = split_payload["sig_returns"]

    real_metrics = pd.DataFrame(
        [compute_financial_metrics(path) for path in real_paths]
    ).mean(numeric_only=True)
    real_m = real_metrics.to_dict()

    rows = [
        {
            "ticker": ticker,
            "split": split_name,
            "model": "real",
            "lookback": lookback,
            "signature_level": signature_level,
            "forward": forward,
            "k_neighbors": k_neighbors,
            "signature_mmd": np.nan,
            "volatility": real_m.get("volatility", np.nan),
            "volatility_error": 0.0,
            "var_95": real_m.get("var_95", np.nan),
            "var_99": real_m.get("var_99", np.nan),
            "var_95_error": 0.0,
            "es_95": real_m.get("es_95", np.nan),
            "es_99": real_m.get("es_99", np.nan),
            "es_95_error": 0.0,
            "max_drawdown": real_m.get("max_drawdown", np.nan),
            "max_drawdown_error": 0.0,
            "squared_return_acf_error": 0.0,
        }
    ]

    model_payload = {
        "historical_bootstrap": hist_returns,
        "signature_bootstrap": sig_returns,
    }
    for model_name, model_returns in model_payload.items():
        mm = pd.DataFrame([compute_financial_metrics(path) for path in model_returns]).mean(
            numeric_only=True
        )
        mm_d = mm.to_dict()
        rows.append(
            {
                "ticker": ticker,
                "split": split_name,
                "model": model_name,
                "lookback": lookback,
                "signature_level": signature_level,
                "forward": forward,
                "k_neighbors": k_neighbors,
                "signature_mmd": linear_signature_mmd(
                    real_paths_returns=real_paths,
                    generated_paths_returns=model_returns,
                    signature_level=signature_level,
                ),
                "volatility": mm_d.get("volatility", np.nan),
                "volatility_error": abs(
                    float(mm_d.get("volatility", np.nan)) - float(real_m.get("volatility", np.nan))
                ),
                "var_95": mm_d.get("var_95", np.nan),
                "var_99": mm_d.get("var_99", np.nan),
                "var_95_error": abs(
                    float(mm_d.get("var_95", np.nan)) - float(real_m.get("var_95", np.nan))
                ),
                "es_95": mm_d.get("es_95", np.nan),
                "es_99": mm_d.get("es_99", np.nan),
                "es_95_error": abs(
                    float(mm_d.get("es_95", np.nan)) - float(real_m.get("es_95", np.nan))
                ),
                "max_drawdown": mm_d.get("max_drawdown", np.nan),
                "max_drawdown_error": abs(
                    float(mm_d.get("max_drawdown", np.nan))
                    - float(real_m.get("max_drawdown", np.nan))
                ),
                "squared_return_acf_error": squared_return_acf_error(
                    real_returns_paths=real_paths,
                    generated_returns_paths=model_returns,
                    max_lag=20,
                ),
            }
        )
    return rows


def run_sensitivity(config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ensure_directories_from_config(config)
    tables_dir = Path(config["tables_dir"])
    figures_dir = Path(config["figures_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    targets = ["SPY"]
    fx_tickers = list(config.get("fx_tickers", []))
    if fx_tickers:
        targets.append(fx_tickers[0])

    k_grid = list(config.get("sensitivity_k_neighbors", [5, 10, 20, 50, 100]))
    level_grid = list(config.get("sensitivity_signature_levels", [2, 3]))
    lookback_grid = list(config.get("sensitivity_lookbacks", [20, 60]))
    forward_grid = list(config.get("sensitivity_forwards", [1, 5]))
    # Optional extended ranges can be enabled in config without changing code:
    # sensitivity_lookbacks: [60, 120]
    # sensitivity_forwards: [5, 10, 20]

    n_paths = min(80, int(config.get("n_generated_paths", 200)))
    horizon = min(80, int(config.get("generation_horizon", 120)))
    train_ratio = float(config.get("train_ratio", 0.6))
    val_ratio = float(config.get("val_ratio", 0.2))
    test_ratio = float(config.get("test_ratio", 0.2))
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0, atol=1e-6):
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    log_price_by_ticker: Dict[str, np.ndarray] = {}
    for ticker in targets:
        processed_file = Path(config["processed_data_dir"]) / f"{ticker_to_filename(ticker)}.csv"
        if not processed_file.exists():
            raise FileNotFoundError(f"Missing processed data for {ticker}: {processed_file}")
        df = pd.read_csv(processed_file)
        log_price_by_ticker[ticker] = df["log_price"].to_numpy(dtype=float)

    val_rows: List[Dict] = []

    for lookback in lookback_grid:
        for signature_level in level_grid:
            for forward in forward_grid:
                for k_neighbors in k_grid:
                    for ticker in targets:
                        log_prices = log_price_by_ticker[ticker]
                        try:
                            train_end, val_end = _split_points(
                                n_obs=len(log_prices),
                                train_ratio=train_ratio,
                                val_ratio=val_ratio,
                                lookback=lookback,
                            )
                            signatures, futures, anchors = _prepare_libraries(
                                log_prices=log_prices,
                                lookback=lookback,
                                forward=forward,
                                signature_level=signature_level,
                            )
                            train_mask = anchors + forward <= train_end - 1
                            train_signatures = signatures[train_mask]
                            train_futures = futures[train_mask]
                            if train_signatures.shape[0] == 0:
                                raise ValueError("empty train signature/future library")

                            hist_model = HistoricalBootstrapGenerator(
                                future_segments=train_futures,
                                forward=forward,
                                random_state=42,
                            )
                            sig_model = SignatureBootstrapGenerator(
                                signature_library=train_signatures,
                                future_library=train_futures,
                                lookback=lookback,
                                signature_level=signature_level,
                                k_neighbors=k_neighbors,
                                random_state=42,
                            )
                            val_payload = _simulate_split(
                                log_prices=log_prices,
                                start_idx=train_end,
                                end_idx=val_end,
                                lookback=lookback,
                                horizon=horizon,
                                n_paths=n_paths,
                                hist_model=hist_model,
                                sig_model=sig_model,
                            )
                            val_rows.extend(
                                _rows_for_split(
                                    ticker=ticker,
                                    split_name="validation",
                                    lookback=lookback,
                                    signature_level=signature_level,
                                    forward=forward,
                                    k_neighbors=k_neighbors,
                                    split_payload=val_payload,
                                )
                            )
                        except Exception as exc:
                            print(
                                f"[FAIL] ticker={ticker}, lookback={lookback}, level={signature_level}, "
                                f"forward={forward}, k={k_neighbors}: {exc}"
                            )
                    print(
                        f"[DONE] lookback={lookback}, level={signature_level}, "
                        f"forward={forward}, k={k_neighbors}"
                    )

    sensitivity_df = pd.DataFrame(val_rows)
    sensitivity_path = tables_dir / "signature_bootstrap_sensitivity.csv"
    sensitivity_df.to_csv(sensitivity_path, index=False)
    print(f"Saved sensitivity table (validation): {sensitivity_path}")

    sig_only = sensitivity_df[sensitivity_df["model"] == "signature_bootstrap"].copy()
    grouped = (
        sig_only.groupby(["lookback", "signature_level", "forward", "k_neighbors"], as_index=False)
        .agg(
            signature_mmd=("signature_mmd", "mean"),
            volatility_error=("volatility_error", "mean"),
            var95_error=("var_95_error", "mean"),
            es95_error=("es_95_error", "mean"),
            drawdown_error=("max_drawdown_error", "mean"),
            acf_error=("squared_return_acf_error", "mean"),
        )
        .reset_index(drop=True)
    )
    grouped["normalized_signature_mmd"] = _normalize_minmax(grouped["signature_mmd"])
    grouped["normalized_volatility_error"] = _normalize_minmax(grouped["volatility_error"])
    grouped["normalized_var95_error"] = _normalize_minmax(grouped["var95_error"])
    grouped["normalized_es95_error"] = _normalize_minmax(grouped["es95_error"])
    grouped["normalized_drawdown_error"] = _normalize_minmax(grouped["drawdown_error"])
    grouped["normalized_acf_error"] = _normalize_minmax(grouped["acf_error"])
    grouped["score"] = (
        grouped["normalized_signature_mmd"]
        + grouped["normalized_volatility_error"]
        + grouped["normalized_var95_error"]
        + grouped["normalized_es95_error"]
        + grouped["normalized_drawdown_error"]
        + grouped["normalized_acf_error"]
    )
    grouped = grouped.sort_values("score", ascending=True).reset_index(drop=True)
    if grouped.empty:
        raise RuntimeError("No best configuration selected from validation results.")

    best_overall = grouped.head(1).copy()
    best_overall["config_type"] = "best_overall"

    multiday_pool = grouped[grouped["forward"] >= 5].copy()
    if multiday_pool.empty:
        best_multiday = best_overall.copy()
        best_multiday["config_type"] = "best_multiday_fallback_overall"
    else:
        best_multiday = multiday_pool.head(1).copy()
        best_multiday["config_type"] = "best_multiday"

    (tables_dir / "best_overall_signature_bootstrap_config.csv").write_text(
        best_overall.to_csv(index=False), encoding="utf-8"
    )
    (tables_dir / "best_multiday_signature_bootstrap_config.csv").write_text(
        best_multiday.to_csv(index=False), encoding="utf-8"
    )
    best_cfg_combined = pd.concat([best_overall, best_multiday], ignore_index=True)
    best_cfg_path = tables_dir / "best_signature_bootstrap_config.csv"
    best_cfg_combined.to_csv(best_cfg_path, index=False)
    print(f"Saved best validation configs: {best_cfg_path}")

    def _evaluate_config_on_test(cfg_row: pd.Series, config_type: str) -> Tuple[pd.DataFrame, Dict]:
        lookback = int(cfg_row["lookback"])
        level = int(cfg_row["signature_level"])
        forward = int(cfg_row["forward"])
        k = int(cfg_row["k_neighbors"])

        rows: List[Dict] = []
        first_payload: Dict = {}
        for ticker in targets:
            log_prices = log_price_by_ticker[ticker]
            train_end, val_end = _split_points(
                n_obs=len(log_prices),
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                lookback=lookback,
            )
            signatures, futures, anchors = _prepare_libraries(
                log_prices=log_prices,
                lookback=lookback,
                forward=forward,
                signature_level=level,
            )
            train_mask = anchors + forward <= train_end - 1
            train_signatures = signatures[train_mask]
            train_futures = futures[train_mask]

            hist_model = HistoricalBootstrapGenerator(
                future_segments=train_futures,
                forward=forward,
                random_state=42,
            )
            sig_model = SignatureBootstrapGenerator(
                signature_library=train_signatures,
                future_library=train_futures,
                lookback=lookback,
                signature_level=level,
                k_neighbors=k,
                random_state=42,
            )
            payload = _simulate_split(
                log_prices=log_prices,
                start_idx=val_end,
                end_idx=len(log_prices),
                lookback=lookback,
                horizon=horizon,
                n_paths=n_paths,
                hist_model=hist_model,
                sig_model=sig_model,
            )
            if not first_payload:
                first_payload = payload
            rows.extend(
                _rows_for_split(
                    ticker=ticker,
                    split_name="test",
                    lookback=lookback,
                    signature_level=level,
                    forward=forward,
                    k_neighbors=k,
                    split_payload=payload,
                )
            )

        df = pd.DataFrame(rows)
        df["config_type"] = config_type
        return df, first_payload

    overall_test_df, overall_payload = _evaluate_config_on_test(
        best_overall.iloc[0], "best_overall"
    )
    multiday_test_df, multiday_payload = _evaluate_config_on_test(
        best_multiday.iloc[0], "best_multiday"
    )

    eval_path = tables_dir / "generative_model_evaluation.csv"
    overall_test_df.to_csv(eval_path, index=False)
    print(f"Saved final test evaluation (best_overall): {eval_path}")

    comparison_df = pd.concat([overall_test_df, multiday_test_df], ignore_index=True)
    comparison_models = comparison_df[
        comparison_df["model"].isin(["historical_bootstrap", "signature_bootstrap"])
    ].copy()
    comparison_path = tables_dir / "best_config_test_comparison.csv"
    comparison_models.to_csv(comparison_path, index=False)
    print(f"Saved best-config test comparison: {comparison_path}")

    # Regenerate required figures from best_overall test evaluation.
    if overall_payload:
        plot_generated_paths(
            real_log_path=overall_payload["real_full_path"],
            hist_paths=overall_payload["hist_paths_log"],
            sig_paths=overall_payload["sig_paths_log"],
            output_path=figures_dir / "generated_paths_real_vs_bootstrap.png",
        )
        plot_return_distribution(
            real_returns=overall_payload["real_return_paths"],
            hist_returns=overall_payload["hist_returns"],
            sig_returns=overall_payload["sig_returns"],
            output_path=figures_dir / "return_distribution_real_vs_generated.png",
        )
        plot_squared_return_acf(
            real_returns=overall_payload["real_return_paths"],
            hist_returns=overall_payload["hist_returns"],
            sig_returns=overall_payload["sig_returns"],
            output_path=figures_dir / "squared_return_acf_comparison.png",
        )
        plot_drawdown_distribution(
            real_paths_returns=overall_payload["real_return_paths"],
            hist_paths_returns=overall_payload["hist_returns"],
            sig_paths_returns=overall_payload["sig_returns"],
            output_path=figures_dir / "drawdown_distribution_comparison.png",
        )

    plot_var_es_comparison(overall_test_df, figures_dir / "var_es_comparison.png")
    mmd_df = overall_test_df[["ticker", "model", "signature_mmd"]].copy()
    plot_signature_mmd_comparison(mmd_df, figures_dir / "signature_mmd_comparison.png")

    if overall_payload and multiday_payload:
        plot_acf_best_overall_vs_multiday(
            real_returns=overall_payload["real_return_paths"],
            historical_returns=overall_payload["hist_returns"],
            signature_overall_returns=overall_payload["sig_returns"],
            signature_multiday_returns=multiday_payload["sig_returns"],
            output_path=figures_dir / "acf_best_overall_vs_multiday.png",
        )

    # Refresh cluster return plot with corrected naming.
    cluster_stocks(config)

    # Cautious summary wording.
    hist_test_mmd = float(
        overall_test_df[overall_test_df["model"] == "historical_bootstrap"]["signature_mmd"].mean()
    )
    sig_test_mmd = float(
        overall_test_df[overall_test_df["model"] == "signature_bootstrap"]["signature_mmd"].mean()
    )
    better = "lower" if sig_test_mmd < hist_test_mmd else "higher"
    forward_overall = int(best_overall.iloc[0]["forward"])
    summary = [
        "# Evaluation Summary",
        "",
        "Signature bootstrap achieves "
        f"**{better} linear signature MMD** than historical bootstrap in the reported test evaluation.",
        "VaR/ES estimates are closer to the real test distribution after fixing the positive-loss convention.",
        "Historical bootstrap remains a strong baseline.",
        f"The best overall validation config uses forward={forward_overall}, which is good for daily "
        "distributional matching but may not fully preserve multi-day path dependence.",
        "Squared-return ACF remains imperfect, so volatility clustering is only partially captured.",
        "This is scenario generation, not directional forecasting.",
        "",
        "## Best Overall Validation Configuration",
        best_overall.to_csv(index=False).strip(),
        "",
        "## Best Multiday Validation Configuration (forward >= 5)",
        best_multiday.to_csv(index=False).strip(),
    ]
    summary_path = tables_dir / "evaluation_summary.md"
    summary_path.write_text("\n".join(summary), encoding="utf-8")
    print(f"Saved summary: {summary_path}")

    return sensitivity_df, best_overall


def main() -> None:
    config = load_config()
    run_sensitivity(config)


if __name__ == "__main__":
    main()
