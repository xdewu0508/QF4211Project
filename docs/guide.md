# QF4211 Project Guide

## 1. Canonical Workflow

- The only authoritative execution pipeline is `project.ipynb`.
- The notebook performs the full workflow end to end:
  - feature engineering from raw Binance hourly data,
  - chronological train / validation / test splitting,
  - unified model and rule-based backtests,
  - export of paper-style tables, statistical tests, plots, and standardized trading logs.
- The repo’s editable written submission is `report/final_report.tex`.

## 2. Repository Structure

- `data/raw/` contains the tracked Binance market inputs needed to rerun the project.
- `data/engineered_features/`, `data/split_data/`, and `data/trading_logs/` are regenerated locally by the notebook and are not tracked in the cleaned repo snapshot.
- `models/` is regenerated locally by the notebook and is not tracked in the cleaned repo snapshot.
- `result/` is regenerated locally by the notebook and is not tracked in the cleaned repo snapshot.
- `docs/Reference/` stores the reference paper and course documents.
- `report/` stores the final LaTeX report source.

## 3. Scope

- Assets: BTC/USDT, ETH/USDT, XRP/USDT, SOL/USDT.
- Frequency: hourly.
- Raw data coverage: January 2021 to December 2025.
- Strategy families:
  - Buy-and-Hold
  - Logistic Regression
  - XGBoost
  - LightGBM
  - GRU
  - SMA
  - RSI
- Feature sets:
  - `ohlc_raw`
  - `candle_raw`
  - `ohlc_extended`
  - `candle_extended`
- Strategy modes:
  - `long_only`
  - `short_only`
  - `long_short`
- Cost regimes:
  - `with_cost`
  - `no_cost`

## 4. Feature Construction

- `ohlc_raw` uses `open`, `high`, `low`, `close`, `volume`, `number_of_trades`, and cyclical hour terms.
- `candle_raw` replaces the raw OHLC block with `close`, `candle_body`, `upper_shadow`, and `lower_shadow`, while retaining `volume`, `number_of_trades`, and cyclical hour terms.
- The paper-aligned extended features add:
  - SMA: `sma_15`, `sma_20`, `sma_25`, `sma_30`
  - RSI: `rsi_15`, `rsi_20`, `rsi_25`, `rsi_30`
  - `macd_line`, `macd_hist`, `wr_14`, `so_14`, `mfi_14`
- Additional engineered features remain in the notebook for broader diagnostics, but the paper-style subsets above drive the main comparisons.
- Targets are based on next-hour open-to-open returns:
  - `label_up = 1(next-hour log return > 0)`
  - `label_down = 1(next-hour log return < 0)`

## 5. Modelling Design

- ML models use two binary classifiers:
  - a long-side classifier trained on `label_up`
  - a short-side classifier trained on `label_down`
- Strategy mappings:
  - `long_only` -> `{0, 1}`
  - `short_only` -> `{0, -1}`
  - `long_short` -> `{0, 1, -1}`
- In `long_short`, if both sides clear their thresholds simultaneously, the notebook chooses the side with the larger threshold margin.
- SMA and RSI are not fixed one-shot baselines. They are tuned on validation data and then sent through the same thresholding path as the ML models.

## 6. Thresholding and Feasibility

- Threshold tuning is precision-first on the validation split.
- A candidate threshold is only considered feasible if it satisfies all three guardrails:
  - at least `20` trades,
  - signal ratio of at least `0.2%`,
  - recall of at least `1%`.
- Among feasible thresholds, the notebook prioritizes:
  - higher validation precision,
  - then higher recall,
  - then higher paper-style Sharpe.
- Main report tables and grouped significance tests are built from feasible configurations.

## 7. Backtest Convention

- Signals are formed using information available at hour `t`.
- If a threshold triggers, the trade enters at the open of hour `t+1`.
- The trade exits at the open of hour `t+2`.
- Transaction cost is `0.035%` per unit change in position.
- A direct flip from `+1` to `-1` or vice versa counts as double turnover.

## 8. Metrics and Statistical Tests

- Main report metric: paper-style Sharpe computed from realized signal returns.
- Additional diagnostics include:
  - cumulative return,
  - annualized Sharpe,
  - maximum drawdown,
  - turnover,
  - number of trades,
  - precision,
  - recall.
- Grouped bootstrap tests follow the paper-oriented comparison structure:
  - `candle` vs `ohlc`
  - `extended` vs `raw`
  - `GRU` vs the remaining ML models
- Usefulness tests also check:
  - whether mean strategy return exceeds zero,
  - whether mean strategy return exceeds buy-and-hold.

## 9. Reproducibility Notes

- `project.ipynb` now seeds Python, NumPy, and TensorFlow from one global seed.
- TensorFlow deterministic settings are enabled on a best-effort basis, but exact deep-learning replication can still vary across hardware / backend combinations.
- Binance raw files may mix millisecond and microsecond timestamps; the notebook now normalizes them row-wise before feature engineering.
- If you want the derived CSVs and report-facing outputs to reflect the latest notebook logic, rerun `project.ipynb` from top to bottom.

## 10. Key Outputs

The following are created automatically when `project.ipynb` is run on a fresh device:

- `data/engineered_features/`
- `data/split_data/`
- `data/trading_logs/`
- `models/`
- `result/`
