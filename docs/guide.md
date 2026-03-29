# QF4211 Project Guide

## 0. Overview

- Research question: compare machine-learning trading signals, classic indicator-driven rules, and buy-and-hold across four crypto assets under both `with_cost` and `no_cost` settings.
- Assets: BTC/USDT, ETH/USDT, XRP/USDT, SOL/USDT.
- Data: Binance hourly data, 2021-2025.
- Unified strategy families: Logistic Regression, XGBoost, LightGBM, MLP, LSTM, CNN, GRU, SMA, RSI.
- Main grid dimensions:
  - 4 feature sets: `ohlc_raw`, `candle_raw`, `ohlc_extended`, `candle_extended`
  - 3 strategy modes: `long_only`, `short_only`, `long_short`
  - 2 cost regimes: `with_cost`, `no_cost`
  - 4 assets
- ML models use two binary systems:
  - `label_up` for long signals
  - `label_down` for short signals
- `long_short` combines the two probability systems into `{0, 1, -1}` by comparing which side clears its validation threshold more strongly.
- Labels are based on next-hour open-to-open log returns.
- Realized trading PnL is still compounded from the aligned simple open-to-open return series.
- Optional dependencies are handled explicitly:
  - `lightgbm` controls LightGBM availability
  - `tensorflow` controls MLP/LSTM/CNN/GRU availability
- Parallel sections use a cross-platform worker helper:
  - first try `os.cpu_count()`
  - if needed, fall back to `multiprocessing.cpu_count()`

## 1. Data and Feature Construction

- Load hourly Binance kline data with price, volume, and trade-count fields.
- Create paper-aligned raw feature families:
  - `ohlc_raw`: `open`, `high`, `low`, `close`, `number_of_trades`, `volume`, cyclical hour terms
  - `candle_raw`: `close`, `candle_body`, `upper_shadow`, `lower_shadow`, `number_of_trades`, `volume`, cyclical hour terms
- Create paper-aligned extended indicators:
  - SMA: `sma_15`, `sma_20`, `sma_25`, `sma_30`
  - RSI: `rsi_15`, `rsi_20`, `rsi_25`, `rsi_30`
  - `macd_line`, `macd_hist`, `wr_14`, `so_14`, `mfi_14`
- Keep the broader extra engineered features in the notebook for supplementary analysis, but the main paper-style grid uses the paper-aligned feature-set definitions above.
- Create targets from next-hour open-to-open log return:
  - `label_up = 1(log return > 0)`
  - `label_down = 1(log return < 0)`
- Use chronological splitting: 50% train, 25% validation, 25% test.

## 2. Model Training and Threshold Selection

- ML models train on the train split only.
- Validation predictions are generated separately for long and short classifiers.
- Threshold selection is precision-first:
  - maximize validation precision
  - subject to minimum trade-count, minimum signal-ratio, and minimum recall guardrails
- Guardrails are constraints, not the objective.
- Tie-breakers remain:
  - higher recall
  - higher paper-style Sharpe
- Rule benchmarks:
  - `SMA` uses short/long moving-average windows
  - `RSI` uses selected RSI windows
- For reporting consistency, SMA and RSI are emitted under each feature-set label even though their signals come from their own rule definitions rather than from the ML input matrix.

## 3. Backtest Convention

- Signal is formed at time `t`.
- Trade is evaluated on the aligned next-hour open-to-open return already stored in the row.
- Position sets:
  - `long_only`: `{0, 1}`
  - `short_only`: `{0, -1}`
  - `long_short`: `{0, 1, -1}`
- Transaction cost is `0.035%` per unit change in position.
- Flips from `1` to `-1` or `-1` to `1` count as a double turnover event.

## 4. Metrics

- Paper-style table metrics:
  - Precision
  - Recall
  - Avg return
  - Sharpe
- Here, the notebook’s paper-style Sharpe is computed from signal returns, not from annualized hourly returns.
- The notebook also keeps additional diagnostics such as cumulative return, turnover, number of trades, maximum drawdown, and annualized Sharpe in the raw summary tables.

## 5. Unified Experiment Grid

- The main notebook workflow runs the unified cross-product over:
  - 9 strategy families
  - 4 feature sets
  - 3 strategy modes
  - 2 cost regimes
  - 4 assets
- Tabular ML tasks are parallelized.
- Rule-model tuning tasks are parallelized.
- TensorFlow sequence models are kept sequential for training stability.
- Each completed configuration saves:
  - a standardized trading log under `data/trading_logs/`
  - one row in `data/trading_logs/unified_experiment_summary.csv`

## 6. Reporting Outputs

- Section 10 of the notebook exports paper-style tables modeled on FRL:
  - `Table 8`-style long-only tables
  - `Table 9`-style short-only tables
  - long-short analog tables
  - `Table 10`-style candle vs OHLC summaries
  - `Table 11`-style raw vs extended summaries
  - `Table 12`-style model-average summaries
- These are exported both:
  - per asset
  - for all assets combined
- The notebook also exports:
  - raw full-grid summaries
  - representative best-configuration tables
  - cost-drag comparisons between `with_cost` and `no_cost`
  - equity-curve PNGs with buy-and-hold overlays

## 7. Statistical Tests

- Grouped bootstrap tests follow the paper’s structure more closely:
  - compare `candle` vs `ohlc`
  - compare `extended` vs `raw`
  - use non-overlapping blocks
  - use block length `floor(n^(1/3))`
  - use 10,000 bootstrap iterations
- Additional usefulness tests are included:
  - test whether mean strategy net return is greater than zero
  - test whether mean strategy return exceeds buy-and-hold
- Bootstrap work is parallelized because it is one of the heavier repeated computations in the notebook.

## 8. Supplementary Outputs

- Section 11 bundles grouped paper-style tables across scopes and saves:
  - cost-drag summaries by asset, model, and strategy mode
  - grouped table overviews
  - representative best-configuration overviews
- The older ablation block has been replaced by these paper-aligned supplementary exports.
