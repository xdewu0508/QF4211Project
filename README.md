# QF4211 Project

This repository contains the final comparative backtesting study of machine-learning and rule-based trading strategies in cryptocurrency markets.

## Canonical Entry Point

The canonical pipeline is [`project.ipynb`](/Users/wuxuande/PycharmProjects/QF4211Project/project.ipynb). It is the notebook that:

- loads Binance hourly spot data for BTC, ETH, XRP, and SOL,
- builds the engineered feature sets used in the paper,
- creates the chronological train / validation / test split,
- runs the unified experiment grid for ML, rule-based, and buy-and-hold strategies,
- writes the standardized trading logs and summary tables used in the report.

The other notebooks in the repository are exploratory or legacy artifacts and should not be treated as the final source of truth for reported results:

- [`buy_and_hold.ipynb`](/Users/wuxuande/PycharmProjects/QF4211Project/buy_and_hold.ipynb)
- [`feature_engineering.ipynb`](/Users/wuxuande/PycharmProjects/QF4211Project/feature_engineering.ipynb)
- [`train_val_test_split.ipynb`](/Users/wuxuande/PycharmProjects/QF4211Project/train_val_test_split.ipynb)
- [`logistic_regression.ipynb`](/Users/wuxuande/PycharmProjects/QF4211Project/logistic_regression.ipynb)
- [`xgboost.ipynb`](/Users/wuxuande/PycharmProjects/QF4211Project/xgboost.ipynb)
- [`eda_feature_analysis.ipynb`](/Users/wuxuande/PycharmProjects/QF4211Project/eda_feature_analysis.ipynb)

## Expected Workflow

1. Install the dependencies listed in [`requirements.txt`](/Users/wuxuande/PycharmProjects/QF4211Project/requirements.txt).
2. Open [`project.ipynb`](/Users/wuxuande/PycharmProjects/QF4211Project/project.ipynb).
3. Run the notebook from top to bottom.
4. Review the generated outputs in the `data/` and `result/` folders described below.

## Key Inputs and Outputs

### Inputs

- Raw Binance hourly data: [`data/raw`](/Users/wuxuande/PycharmProjects/QF4211Project/data/raw)
- Reference paper and extracted notes: [`docs/Reference`](/Users/wuxuande/PycharmProjects/QF4211Project/docs/Reference) and [`docs/_extracted`](/Users/wuxuande/PycharmProjects/QF4211Project/docs/_extracted)

### Main generated data products

- Engineered features: [`data/engineered_features`](/Users/wuxuande/PycharmProjects/QF4211Project/data/engineered_features)
- Chronological splits: [`data/split_data`](/Users/wuxuande/PycharmProjects/QF4211Project/data/split_data)
- Unified trading-log summary: [`data/trading_logs/unified_experiment_summary.csv`](/Users/wuxuande/PycharmProjects/QF4211Project/data/trading_logs/unified_experiment_summary.csv)

### Report-facing result files

- Feasible configuration summary: [`result/summary_all_strategies_feasible.csv`](/Users/wuxuande/PycharmProjects/QF4211Project/result/summary_all_strategies_feasible.csv)
- Buy-and-hold summary: [`result/buy_and_hold/summary_all_assets.csv`](/Users/wuxuande/PycharmProjects/QF4211Project/result/buy_and_hold/summary_all_assets.csv)
- Paper-style tables: [`result/paper_tables`](/Users/wuxuande/PycharmProjects/QF4211Project/result/paper_tables)
- Statistical-test outputs: [`result/stat_tests`](/Users/wuxuande/PycharmProjects/QF4211Project/result/stat_tests)
- Representative equity-curve plots: [`result/graph/feasible`](/Users/wuxuande/PycharmProjects/QF4211Project/result/graph/feasible)

## Report Materials

- Current report: [`report/DSE_QF4211 Group 3.docx`](/Users/wuxuande/PycharmProjects/QF4211Project/report/DSE_QF4211%20Group%203.docx)
- Peer-review comments: [`report/Peer Review Suggestions.docx`](/Users/wuxuande/PycharmProjects/QF4211Project/report/Peer%20Review%20Suggestions.docx)
- Rebuttal document: [`report/Rebuttal.docx`](/Users/wuxuande/PycharmProjects/QF4211Project/report/Rebuttal.docx)

## Notes on Reproducibility

- The final report is based on the unified workflow in [`project.ipynb`](/Users/wuxuande/PycharmProjects/QF4211Project/project.ipynb), not on the exploratory notebooks.
- The deep-learning component includes a GRU model and may show some run-to-run variation across hardware and software environments.
- The result tree is intentionally comprehensive; when cross-checking the report, start with the files listed in the “Report-facing result files” section above.
