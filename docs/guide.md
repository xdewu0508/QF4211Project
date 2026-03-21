# QF4211 Project Guide

## 0. Overview

- Research question: compare ML strategies vs traditional rules and buy-and-hold using Sharpe ratio after transaction costs.
- Assets: BTC/USDT, ETH/USDT, XRP/USDT, SOL/USDT.
- Data: Binance hourly data, 2021-2025.
- Models: Logistic Regression, XGBoost, GRU.
- Benchmarks: Buy-and-Hold, SMA, RSI.
- Split: chronological 50% train / 25% validation / 25% test.

## 1. Build Data and Features

- Load and clean hourly OHLCV + trade count data.
- Create features: OHLC, candlestick features, technical indicators (SMA/RSI/MACD).
- Create next-hour direction label.
- Ensure all features use only past information.

## 2. Train and Tune on Train/Validation

- Train each ML model on train set only.
- Get validation predictions.
- Tune trading thresholds on validation set only (long-only and/or long-short).
- Freeze model settings and thresholds before test.

## 3. Backtest Logic (Notebook Simulation)

- Signal at time `t`, execute at open of `t+1`.
- Hold for one hour (open-to-open return).
- Position values: `1` (long), `0` (flat), `-1` (short).
- Apply transaction cost `0.035%` when position changes.
- Compute net returns = gross returns - transaction costs.

## 4. Evaluate on Test Set

- Run final test once with frozen settings.
- Report metrics: annualized Sharpe (primary), cumulative return, avg return per trade, turnover, max drawdown, precision, recall.
- Compare ML models and benchmarks under the same assumptions.

## 5. Robustness Checks

- Check sensitivity to transaction costs (e.g., 0.02%, 0.035%, 0.05%).
- Check performance by market subperiods if possible.
- Briefly discuss limitations and assumptions.

## 6. Notebook Deliverables

- One clean notebook for full workflow:
  - data preparation
  - feature engineering
  - modeling
  - backtesting
  - results and plots
- Keep outputs reproducible (fixed random seeds, clear parameters, saved result tables).
