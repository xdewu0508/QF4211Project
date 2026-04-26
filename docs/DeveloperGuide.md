# Developer Guide

This guide is for maintainers or collaborators who want the lower-level implementation details behind the project workflow.

## Project Question

The project is built around the following research question:

> Do machine-learning-based cryptocurrency trading strategies still deliver economic value after transaction costs when compared with rule-based strategies and a passive buy-and-hold benchmark?

## Canonical Workflow

- The only authoritative execution pipeline is [`project.ipynb`](../project.ipynb).
- The notebook performs the full workflow end to end:
  - feature engineering from raw Binance hourly data,
  - chronological train / validation / test splitting,
  - unified model and rule-based backtests,
  - export of paper-style tables, statistical tests, plots, and standardized trading logs.
- The synced written submission is [`Final Report Group 3.pdf`](./Final%20Report%20Group%203.pdf).

## Scope

- Assets: `BTC/USDT`, `ETH/USDT`, `XRP/USDT`, `SOL/USDT`
- Frequency: hourly
- Raw data coverage: January 2021 to December 2025
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

## Feature Construction

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

## Modelling Design

- ML models use two binary classifiers:
  - a long-side classifier trained on `label_up`
  - a short-side classifier trained on `label_down`
- Strategy mappings:
  - `long_only` -> `{0, 1}`
  - `short_only` -> `{0, -1}`
  - `long_short` -> `{0, 1, -1}`
- In `long_short`, if both sides clear their thresholds simultaneously, the notebook chooses the side with the larger threshold margin.
- SMA and RSI are validation-tuned and then passed through the same thresholding path as the ML models.

## Thresholding And Feasibility

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

## Backtest Convention

- Signals are formed using information available at hour `t`.
- If a threshold triggers, the trade enters at the open of hour `t+1`.
- The trade exits at the open of hour `t+2`.
- Transaction cost is `0.035%` per unit change in position.
- A direct flip from `+1` to `-1` or vice versa counts as double turnover.

## Metrics And Statistical Tests

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

## Generated Outputs

Running [`project.ipynb`](../project.ipynb) recreates the main generated output folders used throughout the repo:

- `data/engineered_features/`
- `data/split_data/`
- `data/trading_logs/`
- `models/`
- `result/`
- `report_result/`

These outputs are treated as reproducible artifacts of the notebook and can be refreshed by rerunning the pipeline.

## Reproducibility Notes

- The notebook seeds Python, NumPy, and TensorFlow from one global seed.
- TensorFlow deterministic settings are enabled on a best-effort basis, but exact deep-learning replication can still vary across hardware and backend combinations.
- Binance raw files may mix millisecond and microsecond timestamps; the notebook normalizes them row-wise before feature engineering.
- If you change notebook logic and want synchronized outputs, rerun the notebook from top to bottom.
