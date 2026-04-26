# DSE/QF4211 Project

This repository studies whether active cryptocurrency trading strategies can outperform passive buy-and-hold benchmarks after transaction costs and realistic backtesting constraints. The project compares machine-learning, rule-based, and buy-and-hold strategies on hourly Binance spot data for Bitcoin, Ethereum, Ripple, and Solana.

The main research question is:

> To what extent do ML-based trading strategies outperform traditional rule-based strategies and passive buy-and-hold benchmarks in terms of Sharpe Ratio, once transaction costs and backtesting constraints are applied to major crypto assets?

## Reproducibility Overview

The main reproducibility entry point is [`project.ipynb`](project.ipynb). It takes the raw Binance hourly CSV files under [`data/raw/`](data/raw/) and can regenerate the derived data files, trained model artifacts, trading logs, result tables, statistical tests, and output folders used by the project.

The repository includes generated artifacts for inspection, but these should be treated as reproducible outputs of the notebook:

- [`data/engineered_features/`](data/engineered_features/): engineered feature datasets by asset.
- [`data/split_data/`](data/split_data/): chronological train, validation, and test splits.
- [`data/trading_logs/`](data/trading_logs/): standardized per-configuration trading logs and unified experiment summaries.
- [`models/`](models/): fitted ML model artifacts and metadata.
- [`result/`](result/): full generated output from the main experiment pipeline.
- [`report_result/`](report_result/): curated result subset used for the report-facing analysis.

[`result_analysis.ipynb`](result_analysis.ipynb) reproduces the report tables from the generated result files. For simplicity, the relevant analyzed outputs are parked under [`report_result/`](report_result/), while [`result/`](result/) contains the broader full output set from the main pipeline.

## Project Scope

- Assets: `BTC`, `ETH`, `XRP`, `SOL`
- Market data: Binance spot hourly klines
- Main sample in the included splits: `2021-01-01` to `2025-12-31`
- Train/validation/test split: chronological `50% / 25% / 25%`
- Transaction cost: `0.035%` per unit change in position
- Random seed: `42`

Strategy families:

- Buy-and-Hold
- Logistic Regression
- XGBoost
- LightGBM
- GRU
- SMA
- RSI

Feature sets:

- `ohlc_raw`
- `candle_raw`
- `ohlc_extended`
- `candle_extended`

Strategy modes:

- `long_only`
- `short_only`
- `long_short`

Cost regimes:

- `with_cost`
- `no_cost`

## Repository Structure

- [`project.ipynb`](project.ipynb): main end-to-end experiment pipeline.
- [`result_analysis.ipynb`](result_analysis.ipynb): report-table reproduction notebook.
- [`requirements.txt`](requirements.txt): Python dependencies for the notebooks.
- [`data/raw/`](data/raw/): required raw Binance hourly input files.
- [`data/engineered_features/`](data/engineered_features/): generated feature CSVs.
- [`data/split_data/`](data/split_data/): generated chronological train/validation/test splits.
- [`data/trading_logs/`](data/trading_logs/): generated trading logs and unified experiment summary.
- [`models/`](models/): generated trained model artifacts.
- [`result/`](result/): full generated result outputs from the main pipeline.
- [`report_result/`](report_result/): curated outputs used for the report-facing analysis.
- [`2020-2024_regime_sandbox/`](2020-2024_regime_sandbox/): separate robustness rerun for the 2020-2024 regime-shift sample.

## Setup

Create a Python environment and install the required packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Launch Jupyter:

```bash
jupyter notebook
```

The dependencies include the numerical stack, notebook runtime, scikit-learn, XGBoost, LightGBM, TensorFlow, and `requests` for the sandbox raw-data download helper.

## Running the Main Pipeline

Open [`project.ipynb`](project.ipynb) from the repository root and run the notebook sequentially from top to bottom.

Before running, check the configuration cell:

```python
FORCE_DATA_PREP = False
RUN_UNIFIED_GRID = False
GLOBAL_SEED = 42
```

Use:

```python
FORCE_DATA_PREP = True
RUN_UNIFIED_GRID = True
```

to fully regenerate engineered features, train/validation/test splits, model artifacts, trading logs, summary tables, statistical tests, and result outputs from the raw CSV files.

Use:

```python
FORCE_DATA_PREP = False
RUN_UNIFIED_GRID = False
```

for a faster rerun of downstream reporting sections when the generated artifacts already exist.

## Main Pipeline Workflow

[`project.ipynb`](project.ipynb) performs the full workflow:

1. Loads raw hourly Binance CSV files from [`data/raw/`](data/raw/).
2. Standardizes timestamps and OHLCV columns.
3. Builds paper-aligned OHLC, candlestick, raw, and extended feature sets.
4. Creates next-hour return labels for long-side and short-side prediction.
5. Splits each asset chronologically into train, validation, and test sets.
6. Trains and evaluates Logistic Regression, XGBoost, LightGBM, GRU, SMA, RSI, and buy-and-hold strategies.
7. Selects validation thresholds using precision-first feasibility rules.
8. Applies transaction-cost and no-cost backtests.
9. Exports trading logs, trained models, summary CSVs, paper-style tables, statistical tests, and supporting diagnostics.

## Model Artifacts

[`models/`](models/) stores trained model artifacts generated by [`project.ipynb`](project.ipynb), organized by model family, feature set, and asset.

Examples:

- `models/logistic_regression/ohlc_raw/btc/`
- `models/xgboost/candle_extended/eth/`
- `models/lightgbm/ohlc_extended/sol/`
- `models/gru/candle_raw/xrp/`

Typical contents:

- Logistic Regression: `long_model.joblib`, `short_model.joblib`, `scaler.joblib`, `metadata.json`
- XGBoost and LightGBM: `long_model.joblib`, `short_model.joblib`, `metadata.json`
- GRU: `long_model.keras`, `short_model.keras`, `long_scaler.joblib`, `short_scaler.joblib`, `metadata.json`

The long model estimates upward next-hour movement probabilities. The short model estimates downward next-hour movement probabilities. These artifacts are included for inspection and can be regenerated by rerunning the main notebook.

## Result Outputs

The full main-pipeline outputs are under [`result/`](result/).

Top-level summaries:

- `result/summary_all_strategies_raw.csv`: full unified experiment summary before feasibility filtering.
- `result/summary_all_strategies_feasible.csv`: feasible configurations used for the main analysis.


Buy-and-hold outputs:

- `result/buy_and_hold/*_backtest.csv`: per-asset buy-and-hold backtests.
- `result/buy_and_hold/summary_all_assets.csv`: buy-and-hold summary across assets.

Paper-table outputs:

- `result/paper_tables/table_detail_all_configs.csv`: detailed table for all configurations.
- `result/paper_tables/table_detail_feasible_configs.csv`: detailed table for feasible configurations.
- `result/paper_tables/table8_*_feasible.csv`: long-only feasible strategy detail tables.
- `result/paper_tables/table9_*_feasible.csv`: short-only feasible strategy detail tables.
- `result/paper_tables/table_long_short_*_feasible.csv`: long-short feasible strategy detail tables.
- `result/paper_tables/table10_representation_*_feasible.csv`: OHLC versus candlestick representation comparisons.
- `result/paper_tables/table11_feature_type_*_feasible.csv`: raw versus extended feature comparisons.
- `result/paper_tables/table12_model_average_*_feasible.csv`: model-average comparisons.
- `result/paper_tables/table10_feasible_bundle.csv`: combined Table 10-style representation comparisons.
- `result/paper_tables/table11_feasible_bundle.csv`: combined Table 11-style feature-type comparisons.
- `result/paper_tables/table12_feasible_bundle.csv`: combined Table 12-style model-average comparisons.
- `result/paper_tables/best_feature_configs_feasible.csv`: best feasible rows by asset, model, feature set, strategy mode, and cost regime.
- `result/paper_tables/best_model_configs_feasible.csv`: best feasible rows by asset, model, strategy mode, and cost regime.
- `result/paper_tables/best_overall_configs_feasible.csv`: best feasible rows by asset, strategy mode, and cost regime.
- `result/paper_tables/cost_comparison_by_configuration_feasible.csv`: paired no-cost versus with-cost comparison by configuration.
- `result/paper_tables/cost_drag_summary_by_asset_model_strategy_feasible.csv`: transaction-cost drag summary by asset, model, and strategy mode.
- `result/paper_tables/grouped_table_overview_feasible.csv`: combined overview of grouped Table 10/11/12-style outputs.
- `result/paper_tables/best_config_overview_feasible.csv`: representative feasible best-configuration overview.
- `result/paper_tables/infeasible_config_overview.csv`: summary of infeasible configurations excluded from the main analysis.

Statistical-test outputs:

- `result/stat_tests/paper_group_bootstrap_tests_feasible.csv`: grouped bootstrap tests on feasible configurations.
- `result/stat_tests/usefulness_tests_best_overall_feasible.csv`: usefulness tests for best feasible strategies.

## Report Table Reproduction

Run [`result_analysis.ipynb`](result_analysis.ipynb) after the main pipeline outputs exist. This notebook reads generated result files and reproduces the tables used in the report-facing analysis.

The relevant analyzed outputs are also collected under [`report_result/`](report_result/), including:

- buy-and-hold summaries,
- selected paper tables,
- statistical-test outputs,
- regime-comparison outputs.

Users who want the complete generated output set should inspect [`result/`](result/). Users who want the concise report-facing subset should inspect [`report_result/`](report_result/).

## Regime-Shift Sandbox

[`2020-2024_regime_sandbox/`](2020-2024_regime_sandbox/) contains a separate robustness rerun for the 2020-2024 sample window. It follows the same feature-engineering, modelling, backtesting, thresholding, and reporting logic as the main notebook, but writes all generated outputs inside the sandbox directory.

The sandbox notebook is:

- [`2020-2024_regime_sandbox/regime_shift_sandbox.ipynb`](2020-2024_regime_sandbox/regime_shift_sandbox.ipynb)

The sandbox can download monthly Binance spot 1-hour klines and rebuild sandbox-local raw CSVs. Its outputs are intentionally separate from the main 2021-2025 run.

## Reproducibility Notes

- Run notebooks from the repository root unless using the sandbox notebook, which resolves its own sandbox path.
- The main raw input files are expected under [`data/raw/`](data/raw/).
- The notebook seeds Python, NumPy, and TensorFlow with `GLOBAL_SEED = 42`.
- TensorFlow deterministic settings are enabled on a best-effort basis, so GRU results can vary slightly across hardware or backend combinations.
- Binance timestamp units are normalized during preprocessing.
- A full grid rerun can take substantial time because it trains multiple models across assets, feature sets, strategy modes, and cost regimes.
