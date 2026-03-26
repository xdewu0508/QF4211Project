import pandas as pd
import numpy as np

def compute_cumulative_return(returns: pd.Series) -> float:
    """
    Compute total cumulative return from a series of simple returns.
    """
    returns = returns.dropna()
    if len(returns) == 0:
        return np.nan
    return (1 + returns).prod() - 1


def compute_annualized_return(returns: pd.Series, periods_per_year: int = 8760) -> float:
    """
    Compute annualized return from a series of simple hourly returns.
    """
    returns = returns.dropna()
    if len(returns) == 0:
        return np.nan

    cumulative = (1 + returns).prod()
    n_periods = len(returns)

    if cumulative <= 0:
        return np.nan

    return cumulative ** (periods_per_year / n_periods) - 1


def compute_annualized_volatility(returns: pd.Series, periods_per_year: int = 8760) -> float:
    """
    Compute annualized volatility from hourly returns.
    """
    returns = returns.dropna()
    if len(returns) < 2:
        return np.nan
    return returns.std(ddof=1) * np.sqrt(periods_per_year)


def compute_annualized_sharpe(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 8760
) -> float:
    """
    Compute annualized Sharpe ratio using hourly returns.

    Assumes risk_free_rate is annualized (e.g. 0.02 for 2% yearly).
    """
    returns = returns.dropna()
    if len(returns) < 2:
        return np.nan

    rf_per_period = risk_free_rate / periods_per_year
    excess_returns = returns - rf_per_period

    vol = excess_returns.std(ddof=1)
    if vol == 0 or np.isnan(vol):
        return np.nan

    sharpe = excess_returns.mean() / vol
    return sharpe * np.sqrt(periods_per_year)


def compute_max_drawdown(returns: pd.Series) -> float:
    """
    Compute maximum drawdown from a series of simple returns.
    """
    returns = returns.dropna()
    if len(returns) == 0:
        return np.nan

    equity_curve = (1 + returns).cumprod()
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1

    return drawdown.min()


def compute_turnover(position: pd.Series) -> float:
    """
    Compute turnover as the average absolute change in position.

    For buy-and-hold:
    - initial entry is counted separately via num_trades
    - thereafter position stays constant
    """
    position = position.fillna(0)
    position_change = position.diff().abs().fillna(position.iloc[0])
    return position_change.sum()


def compute_num_trades(position: pd.Series) -> int:
    """
    Count number of position changes.
    For buy-and-hold:
    - entering from 0 to 1 at the start counts as 1 trade
    """
    position = position.fillna(0)
    position_change = position.diff().abs().fillna(position.iloc[0])
    return int((position_change > 0).sum())


def compute_trade_returns(net_returns: pd.Series, position: pd.Series) -> list:
    """
    Compute returns per trade by grouping consecutive invested periods.

    For buy-and-hold, there is typically only one trade spanning the full test sample.
    """
    net_returns = net_returns.reset_index(drop=True)
    position = position.reset_index(drop=True)

    trade_returns = []
    in_trade = False
    current_trade_returns = []

    for r, p in zip(net_returns, position):
        if p != 0 and not in_trade:
            in_trade = True
            current_trade_returns = [r]
        elif p != 0 and in_trade:
            current_trade_returns.append(r)
        elif p == 0 and in_trade:
            trade_returns.append((1 + pd.Series(current_trade_returns)).prod() - 1)
            in_trade = False
            current_trade_returns = []

    if in_trade and len(current_trade_returns) > 0:
        trade_returns.append((1 + pd.Series(current_trade_returns)).prod() - 1)

    return trade_returns


def summarize_performance(
    backtest_df: pd.DataFrame,
    return_col: str = "net_return",
    position_col: str = "position",
    periods_per_year: int = 8760,
    risk_free_rate: float = 0.0
) -> pd.Series:
    """
    Generate a full performance summary for the strategy.
    """
    returns = backtest_df[return_col].dropna()
    position = backtest_df[position_col]

    cumulative_return = compute_cumulative_return(returns)
    annualized_return = compute_annualized_return(returns, periods_per_year)
    annualized_volatility = compute_annualized_volatility(returns, periods_per_year)
    annualized_sharpe = compute_annualized_sharpe(returns, risk_free_rate, periods_per_year)
    max_drawdown = compute_max_drawdown(returns)
    turnover = compute_turnover(position)
    num_trades = compute_num_trades(position)

    trade_returns = compute_trade_returns(backtest_df[return_col], backtest_df[position_col])
    avg_return_per_trade = np.mean(trade_returns) if len(trade_returns) > 0 else np.nan

    summary = pd.Series({
        "Cumulative Return": cumulative_return,
        "Annualized Return": annualized_return,
        "Annualized Volatility": annualized_volatility,
        "Annualized Sharpe": annualized_sharpe,
        "Maximum Drawdown": max_drawdown,
        "Turnover": turnover,
        "Number of Trades": num_trades,
        "Average Return per Trade": avg_return_per_trade
    })

    return summary

def compute_directional_metrics(backtest_df: pd.DataFrame) -> pd.Series:
    """
    Optional directional metrics for comparison.
    Since buy-and-hold predicts 'up' every time, this is not a true classifier,
    but the metrics can still be computed for consistency.

    Predicted positive if position == 1
    Actual positive if raw_return > 0
    """
    df = backtest_df.copy()

    df["predicted_positive"] = (df["position"] == 1).astype(int)
    df["actual_positive"] = (df["raw_return"] > 0).astype(int)

    tp = ((df["predicted_positive"] == 1) & (df["actual_positive"] == 1)).sum()
    fp = ((df["predicted_positive"] == 1) & (df["actual_positive"] == 0)).sum()
    fn = ((df["predicted_positive"] == 0) & (df["actual_positive"] == 1)).sum()
    tn = ((df["predicted_positive"] == 0) & (df["actual_positive"] == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else np.nan

    return pd.Series({
        "Precision": precision,
        "Recall": recall,
        "Accuracy": accuracy,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn
    })