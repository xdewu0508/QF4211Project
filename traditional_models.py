import pandas as pd
import numpy as np


btc_train = pd.read_csv("data/split_data/BTC_train.csv")[["close"]].rename(columns={"close": "Close"})
btc_val   = pd.read_csv("data/split_data/BTC_val.csv")[["close"]].rename(columns={"close": "Close"})
btc_test  = pd.read_csv("data/split_data/BTC_test.csv")[["close"]].rename(columns={"close": "Close"})

eth_train = pd.read_csv("data/split_data/ETH_train.csv")[["close"]].rename(columns={"close": "Close"})
eth_val = pd.read_csv("data/split_data/ETH_val.csv")[["close"]].rename(columns={"close": "Close"})
eth_test = pd.read_csv("data/split_data/ETH_test.csv")[["close"]].rename(columns={"close": "Close"})

sol_train = pd.read_csv("data/split_data/SOL_train.csv")[["close"]].rename(columns={"close": "Close"})
sol_val = pd.read_csv("data/split_data/SOL_val.csv")[["close"]].rename(columns={"close": "Close"})
sol_test = pd.read_csv("data/split_data/SOL_test.csv")[["close"]].rename(columns={"close": "Close"})

xrp_train = pd.read_csv("data/split_data/XRP_train.csv")[["close"]].rename(columns={"close": "Close"})
xrp_val = pd.read_csv("data/split_data/XRP_val.csv")[["close"]].rename(columns={"close": "Close"})
xrp_test = pd.read_csv("data/split_data/XRP_test.csv")[["close"]].rename(columns={"close": "Close"})

import numpy as np
import pandas as pd

def compute_metrics(df):
    df = df.copy()

    # Drop NaNs (from rolling indicators)
    df = df.dropna()

    # =========================
    # Basic checks
    # =========================
    if len(df) == 0:
        return {
            "Cumulative Return": np.nan,
            "Annualized Sharpe": np.nan,
            "Average Return per Trade": np.nan,
            "Turnover": 0,
            "Max Drawdown": np.nan
        }

    # =========================
    # Cumulative Return
    # =========================
    cumulative_return = (1 + df['strategy_returns']).prod() - 1

    # =========================
    # Annualized Sharpe Ratio
    # =========================
    mean_ret = df['strategy_returns'].mean()
    std_ret = df['strategy_returns'].std()

    if std_ret == 0 or np.isnan(std_ret):
        sharpe = 0
    else:
        sharpe = np.sqrt(252) * (mean_ret / std_ret)

    # =========================
    # Turnover
    # =========================
    df['trade'] = df['position'].diff().abs()
    turnover = df['trade'].sum()

    # =========================
    # Maximum Drawdown
    # =========================
    equity_curve = (1 + df['strategy_returns']).cumprod()
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # =========================
    # Average Return per Trade
    # =========================
    trade_returns = []
    in_trade = False
    entry_price = None

    for i in range(len(df)):
        # Enter trade
        if df['trade'].iloc[i] == 1 and df['position'].iloc[i] == 1:
            entry_price = df['Close'].iloc[i]
            in_trade = True

        # Exit trade
        elif df['trade'].iloc[i] == 1 and df['position'].iloc[i] == 0 and in_trade:
            exit_price = df['Close'].iloc[i]
            trade_returns.append((exit_price / entry_price) - 1)
            in_trade = False

    avg_return_per_trade = np.mean(trade_returns) if trade_returns else 0

    return {
        "Cumulative Return": cumulative_return,
        "Annualized Sharpe": sharpe,
        "Average Return per Trade": avg_return_per_trade,
        "Turnover": turnover,
        "Max Drawdown": max_drawdown
    }

def backtest_sma(df, short_window, long_window):
    df = df.copy()

    df['SMA_short'] = df['Close'].rolling(short_window).mean()
    df['SMA_long'] = df['Close'].rolling(long_window).mean()

    df['signal'] = 0
    df.loc[df['SMA_short'] > df['SMA_long'], 'signal'] = 1

    df['position'] = df['signal'].shift(1).fillna(0)

    df['returns'] = df['Close'].pct_change()
    df['strategy_returns'] = df['position'] * df['returns']
    
    metrics = compute_metrics(df)
    
    return df, metrics

def compute_RSI(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def backtest_rsi(df, rsi_window=14, low=30, high=70):
    df = df.copy()

    df['RSI'] = compute_RSI(df['Close'], rsi_window)

    df['signal'] = 0
    df.loc[df['RSI'] < low, 'signal'] = 1
    df.loc[df['RSI'] > high, 'signal'] = 0

    df['position'] = df['signal'].shift(1).fillna(0)

    df['returns'] = df['Close'].pct_change()
    df['strategy_returns'] = df['position'] * df['returns']

    metrics = compute_metrics(df)

    return df, metrics

sma_short_list = [5, 10, 20, 30]
sma_long_list  = [50, 100, 150, 200]

rsi_window_list = [7, 14, 21]
rsi_low_list    = [20, 30, 40]
rsi_high_list   = [60, 70, 80]

best_sma_sharpe = -np.inf
best_sma_params = None

for short in sma_short_list:
    for long in sma_long_list:

        if short >= long:
            continue

        df_train, res_train = backtest_sma(btc_train, short, long)
        df_val, res_val = backtest_sma(btc_val, short, long)

        # filter weak strategies
        if res_train['Annualized Sharpe'] < 0:
            continue

        sharpe = res_val['Annualized Sharpe']

        if sharpe > best_sma_sharpe:
            best_sma_sharpe = sharpe
            best_sma_params = (short, long)

print("Best SMA params:", best_sma_params)


best_rsi_sharpe = -np.inf
best_rsi_params = None

for window in rsi_window_list:
    for low in rsi_low_list:
        for high in rsi_high_list:

            if low >= high:
                continue

            df_train, res_train = backtest_rsi(btc_train, window, low, high)
            df_val, res_val = backtest_rsi(btc_val, window, low, high)
           
            sharpe = res_val['Annualized Sharpe']

            if sharpe > best_rsi_sharpe:
                best_rsi_sharpe = sharpe
                best_rsi_params = (window, low, high)

print("Best RSI params:", best_rsi_params)


# SMA
sma_btc_backtest, sma_test_metrics = backtest_sma(btc_test, *best_sma_params)

# RSI
rsi_btc_backtest, rsi_test_metrics = backtest_rsi(btc_test, *best_rsi_params)

print("BTC SMA Metrics:", sma_test_metrics)
print("BTC RSI Metrics:", rsi_test_metrics)
            
sma_btc_backtest.to_csv("result/traditional_models/sma_btc_backtest.csv", index=False)
rsi_btc_backtest.to_csv("result/traditional_models/rsi_btc_backtest.csv", index=False)

# Using best parameters from BTC to test on other coins

# ETH
sma_eth_backtest, sma_eth_metrics = backtest_sma(eth_test, *best_sma_params)
rsi_eth_backtest, rsi_eth_metrics = backtest_rsi(eth_test, *best_rsi_params)

print("ETH SMA Metrics:", sma_eth_metrics)
print("ETH RSI Metrics:", rsi_eth_metrics)

sma_eth_backtest.to_csv("result/traditional_models/sma_eth_backtest.csv", index=False)
rsi_eth_backtest.to_csv("result/traditional_models/rsi_eth_backtest.csv", index=False)

# SOL
sma_sol_backtest, sma_sol_metrics = backtest_sma(sol_test, *best_sma_params)
rsi_sol_backtest, rsi_sol_metrics = backtest_rsi(sol_test, *best_rsi_params)

print("SOL SMA Metrics:", sma_sol_metrics)
print("SOL RSI Metrics:", rsi_sol_metrics)

sma_sol_backtest.to_csv("result/traditional_models/sma_sol_backtest.csv", index=False)
rsi_sol_backtest.to_csv("result/traditional_models/rsi_sol_backtest.csv", index=False)

# XRP
sma_xrp_backtest, sma_xrp_metrics = backtest_sma(xrp_test, *best_sma_params)
rsi_xrp_backtest, rsi_xrp_metrics = backtest_rsi(xrp_test, *best_rsi_params)

print("XRP SMA Metrics:", sma_xrp_metrics)
print("XRP RSI Metrics:", rsi_xrp_metrics)

sma_xrp_backtest.to_csv("result/traditional_models/sma_xrp_backtest.csv", index=False)
rsi_xrp_backtest.to_csv("result/traditional_models/rsi_xrp_backtest.csv", index=False)

