# Scalable Path-Signature Learning for Equity/FX Clustering and Market Scenario Generation

## 1. Project Motivation
This project implements a research-quality quantitative finance pipeline inspired by **signature-based generative modeling** for financial time series.  
The goal is to represent and compare path dynamics (not just pointwise returns), cluster assets by path behavior, and generate realistic market scenarios using bootstrap baselines and signature conditioning.

## 2. Why Path Signatures in Quant Finance
Path signatures provide a truncated but expressive summary of sequential path shape.  
Compared with plain summary stats, signatures preserve richer temporal structure, especially when we:
- model **log prices / log returns**,
- normalize rolling windows to a common origin,
- and use **time augmentation** `X_t = (t, log_price_t)` before signature extraction.

This supports:
- cross-asset behavior comparison,
- nearest-neighbor retrieval in path space,
- scenario generation conditioned on recent trajectory shape.

## 3. Data
- Source: `yfinance` daily data
- Asset universe:
  - Equities/ETFs: `AAPL, MSFT, NVDA, AMZN, META, JPM, BAC, GS, MS, XOM, CVX, JNJ, PFE, MRK, WMT, COST, PG, SPY, QQQ, IWM`
  - FX: `EURUSD=X, GBPUSD=X, JPY=X, CAD=X, AUDUSD=X`
- Configured period: `2015-01-01` to `2026-05-24` (editable in `configs/config.yaml`)

## 4. Methodology
### 4.1 Preprocessing
- Download raw CSV to `data/raw/`
- Robustly parse single-level and multi-index yfinance headers
- Build processed CSV with:
  - `Date`
  - `price`
  - `log_price`
  - `log_return`

### 4.2 Rolling Windows and Signatures
For each ticker:
- Past window (lookback): `X[t-lookback:t]` on `log_price`
- Normalize each window to start at zero:
  - `X_window_norm = X_window - X_window[0]`
- Time augment:
  - path points become `(u, X_window_norm[u])`, `u ∈ [0, 1]`
- Compute truncated path signatures with `iisignature` (`level=3` by default)
- Cache:
  - signature matrix
  - corresponding future segments `X[t+1:t+forward] - X[t]`
  - mean signature per ticker

### 4.3 Caching and Parallel Processing
- Cache-first logic avoids recomputation.
- Parallel per-ticker processing with `joblib` and configurable `n_jobs`.
- Failures are isolated per ticker and reported without crashing whole runs.

### 4.4 Signature-Based Clustering
- Stock-level representation: mean signature vector
- Standardization: `StandardScaler`
- Visualization: PCA (2D)
- Clustering: `KMeans`
- Financial diagnostics by ticker:
  - annualized volatility
  - skewness
  - kurtosis
  - max drawdown
  - average rolling volatility
  - mean return

### 4.5 Scenario Generation
Two models:
1. **Historical bootstrap (baseline)**: sample future segments unconditionally.
2. **Signature bootstrap**:
   - compute current signature from latest window,
   - find `k` nearest historical signatures,
   - sample one corresponding future segment,
   - append and iterate until horizon is reached.

Generated scenarios are produced for at least:
- `SPY`
- one FX ticker (first entry in `fx_tickers` config)

### 4.6 Evaluation
Chronological train/test split only (no random shuffle, no look-ahead).  
Compare real test data vs generated data on:
- mean return
- volatility
- skewness
- kurtosis
- max drawdown
- VaR 95 / VaR 99
- ES 95 / ES 99
- autocorrelation of squared returns (lag-1)
- linear signature MMD

## 5. Project Structure
The repository follows the requested modular layout:
- `src/data`: download + processing
- `src/features`: rolling windows + signatures
- `src/clustering`: dataset building + clustering + plots
- `src/models`: historical/bootstrap generators + optional signature-SDE placeholder
- `src/evaluation`: metrics + plots + evaluation script
- `run_pipeline.py`: full end-to-end execution

## 6. How to Run
Use your existing conda environment:

```bash
conda activate siggen
```

Step-by-step:

```bash
python src/data/download_data.py
python src/data/process_data.py
python src/clustering/build_signature_dataset.py
python src/clustering/cluster_stocks.py
python src/models/run_signature_bootstrap.py
python src/evaluation/evaluate_generation.py
```

Full pipeline:

```bash
python run_pipeline.py
```

## 7. Outputs
### Tables (`reports/tables/`)
- `cluster_membership.csv`
- `cluster_summary.csv`
- `generative_model_evaluation.csv`

### Figures (`reports/figures/`)
- `pca_signature_clusters.png`
- `pca_by_sector.png`
- `cluster_sector_composition.png`
- `cluster_volatility.png`
- `cluster_drawdown.png`
- `cluster_return_distribution.png`
- `generated_paths_real_vs_bootstrap.png`
- `return_distribution_real_vs_generated.png`
- `squared_return_acf_comparison.png`
- `drawdown_distribution_comparison.png`
- `var_es_comparison.png`
- `signature_mmd_comparison.png`

## 8. Notebooks
- `notebooks/01_data_download_and_processing.ipynb`
- `notebooks/02_signature_feature_extraction.ipynb`
- `notebooks/03_stock_clustering.ipynb`
- `notebooks/04_signature_bootstrap_generation.ipynb`
- `notebooks/05_evaluation_and_results.ipynb`

These are lightweight orchestration notebooks and rely on the modular `src/` implementation.

## 9. Limitations and Academic Warnings
- This project emphasizes **representation and scenario generation**, not directional alpha forecasting.
- Do not claim predictive power without rigorous out-of-sample tests.
- `yfinance` may have missing dates, ticker changes, or occasional download failures.
- Linear signature MMD here is implemented as mean-signature discrepancy in a linear kernel setting.
- The optional `signature_sde.py` is a placeholder extension, not the main production result.

## 10. Future Extensions
- Intraday/tick adaptation with same window/signature pipeline
- Cluster-level pooling for signature bootstrap libraries
- Alternative signature distances/kernels
- More robust stress metrics (tail clustering, regime-conditioned diagnostics)
- Full signature-SDE calibration with stronger regularization and validation
