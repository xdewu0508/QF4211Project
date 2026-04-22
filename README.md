# QF4211 Project

This repository contains our final comparative backtesting study of machine-learning and rule-based cryptocurrency trading strategies.

## Canonical Entry Points

- Execution pipeline: [`project.ipynb`](./project.ipynb)
- Submission-ready report source: [`report/final_report.tex`](./report/final_report.tex)
- Dependency list: [`requirements.txt`](./requirements.txt)
- Workflow guide: [`docs/guide.md`](./docs/guide.md)

If you only need to reproduce the project, these four files are the right place to start.

## Repository Layout

- [`data/raw/`](./data/raw): tracked Binance hourly input files required for a fresh run
- `data/engineered_features/`, `data/split_data/`, and `data/trading_logs/`: generated locally by the notebook and not tracked in Git
- `models/`: generated model artifacts written locally by the notebook and not tracked in Git
- `result/`: generated tables, tests, and figures written locally by the notebook and not tracked in Git
- [`docs/Reference/`](./docs/Reference): reference paper and course documents used during the project
- [`docs/guide.md`](./docs/guide.md): workflow notes and methodology summary
- [`report/`](./report): final report source

## What `project.ipynb` Does

`project.ipynb` is the single authoritative workflow. It:

1. loads Binance hourly spot data for BTC, ETH, XRP, and SOL,
2. builds the engineered feature sets used in the study,
3. creates the chronological train / validation / test splits,
4. runs the unified backtest grid for ML, rule-based, and buy-and-hold strategies,
5. writes standardized trading logs, summary tables, bootstrap tests, and representative plots.

## Strategy Families

- Buy-and-Hold
- Logistic Regression
- XGBoost
- LightGBM
- GRU
- SMA
- RSI

The main experiment spans:

- 4 feature sets: `ohlc_raw`, `candle_raw`, `ohlc_extended`, `candle_extended`
- 3 strategy modes: `long_only`, `short_only`, `long_short`
- 2 cost regimes: `with_cost`, `no_cost`
- 4 assets: BTC, ETH, XRP, SOL

## Expected Workflow

1. Install the packages in [`requirements.txt`](./requirements.txt).
2. Open [`project.ipynb`](./project.ipynb).
3. Run the notebook from top to bottom.
4. Let the notebook recreate the derived folders and outputs locally.

## Inputs

- Raw hourly Binance data: [`data/raw`](./data/raw)
- Reference paper and course materials: [`docs/Reference`](./docs/Reference)

## Generated Artifacts

These are intentionally not tracked in the cleaned repo snapshot, but they are recreated automatically when you run [`project.ipynb`](./project.ipynb):

- `data/engineered_features/`
- `data/split_data/`
- `data/trading_logs/`
- `models/`
- `result/`

## Reproducibility Notes

- The notebook seeds Python, NumPy, and TensorFlow from one global seed.
- TensorFlow deterministic settings are enabled on a best-effort basis, but deep-learning results can still vary slightly across hardware and backend combinations.
- Binance raw files may mix millisecond and microsecond timestamps; `project.ipynb` normalizes them row-wise before feature engineering.
- On a fresh clone, the notebook recreates the missing generated directories before writing outputs.
- If you want all derived CSVs and result files to reflect the latest notebook logic, rerun [`project.ipynb`](./project.ipynb) from top to bottom.

## Report Materials

- Final LaTeX report with integrated supplementary rebuttal: [`report/final_report.tex`](./report/final_report.tex)
