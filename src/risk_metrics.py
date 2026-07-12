"""Performance and risk metrics for backtested trading strategies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PerformanceSummary:
    """Container for portfolio performance and risk metrics."""

    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    maximum_drawdown: float
    calmar_ratio: float
    value_at_risk_95: float
    expected_shortfall_95: float
    hit_rate: float
    average_daily_return: float
    best_day: float
    worst_day: float
    number_of_trading_days: int


def _validate_returns(returns: pd.Series, equity: pd.Series, trading_days_per_year: int, confidence_level: float) -> None:
    if not isinstance(returns, pd.Series) or not isinstance(equity, pd.Series):
        raise TypeError("returns and equity must be pandas Series.")

    if len(returns) != len(equity):
        raise ValueError("returns and equity must have the same length.")

    if not np.isfinite(returns.astype(float)).all():
        raise ValueError("returns must contain only finite values.")

    if not np.isfinite(equity.astype(float)).all():
        raise ValueError("equity must contain only finite values.")

    if (equity <= 0).any():
        raise ValueError("equity values must be positive.")

    if trading_days_per_year <= 0:
        raise ValueError("trading_days_per_year must be positive.")

    if not (0 < confidence_level < 1):
        raise ValueError("confidence_level must be between 0 and 1.")

    if len(returns) < 2:
        raise ValueError("At least two observations are required.")


def calculate_total_return(equity: pd.Series) -> float:
    """Calculate total return from an equity curve."""
    if not isinstance(equity, pd.Series):
        raise TypeError("equity must be a pandas Series.")

    if len(equity) < 2:
        raise ValueError("At least two observations are required.")

    if not np.isfinite(equity.astype(float)).all():
        raise ValueError("equity must contain only finite values.")

    if (equity <= 0).any():
        raise ValueError("equity values must be positive.")

    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def calculate_annualized_return(returns: pd.Series, trading_days_per_year: int = 252) -> float:
    """Calculate annualized return from daily returns."""
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series.")

    _validate_returns(returns, pd.Series(np.ones(len(returns)), index=returns.index), trading_days_per_year, 0.95)
    return float((1.0 + returns.mean()) ** trading_days_per_year - 1.0)


def calculate_annualized_volatility(returns: pd.Series, trading_days_per_year: int = 252) -> float:
    """Calculate annualized volatility from daily returns."""
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series.")

    if len(returns) < 2:
        raise ValueError("At least two observations are required.")

    if not np.isfinite(returns.astype(float)).all():
        raise ValueError("returns must contain only finite values.")

    if trading_days_per_year <= 0:
        raise ValueError("trading_days_per_year must be positive.")

    std_daily = returns.std(ddof=0)
    if std_daily == 0:
        return 0.0
    return float(std_daily * np.sqrt(trading_days_per_year))


def calculate_sharpe_ratio(
    returns: pd.Series,
    trading_days_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> float:
    """Calculate the Sharpe ratio using daily net returns."""
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series.")

    if len(returns) < 2:
        raise ValueError("At least two observations are required.")

    if not np.isfinite(returns.astype(float)).all():
        raise ValueError("returns must contain only finite values.")

    if trading_days_per_year <= 0:
        raise ValueError("trading_days_per_year must be positive.")

    if not np.isfinite(risk_free_rate):
        raise ValueError("risk_free_rate must be finite.")

    daily_mean = returns.mean()
    daily_vol = returns.std(ddof=1)
    if daily_vol == 0:
        return 0.0
    return float((daily_mean - risk_free_rate / trading_days_per_year) * np.sqrt(trading_days_per_year) / daily_vol)


def calculate_sortino_ratio(
    returns: pd.Series,
    trading_days_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> float:
    """Calculate the Sortino ratio using downside deviations."""
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series.")

    if len(returns) < 2:
        raise ValueError("At least two observations are required.")

    if not np.isfinite(returns.astype(float)).all():
        raise ValueError("returns must contain only finite values.")

    if trading_days_per_year <= 0:
        raise ValueError("trading_days_per_year must be positive.")

    downside = returns[returns < 0.0]
    if downside.empty:
        return 0.0

    downside_vol = np.sqrt(np.mean(np.square(downside)))
    if downside_vol == 0:
        return 0.0
    return float((returns.mean() - risk_free_rate / trading_days_per_year) * np.sqrt(trading_days_per_year) / downside_vol)


def calculate_drawdown_series(equity: pd.Series) -> pd.Series:
    """Calculate the drawdown series relative to the running maximum."""
    if not isinstance(equity, pd.Series):
        raise TypeError("equity must be a pandas Series.")

    if len(equity) < 2:
        raise ValueError("At least two observations are required.")

    if not np.isfinite(equity.astype(float)).all():
        raise ValueError("equity must contain only finite values.")

    if (equity <= 0).any():
        raise ValueError("equity values must be positive.")

    running_max = equity.cummax()
    return (equity / running_max - 1.0)


def calculate_maximum_drawdown(equity: pd.Series) -> float:
    """Calculate the worst drawdown from an equity curve."""
    return float(calculate_drawdown_series(equity).min())


def calculate_calmar_ratio(returns: pd.Series, equity: pd.Series, trading_days_per_year: int = 252) -> float:
    """Calculate the Calmar ratio as annualized return divided by max drawdown."""
    if not isinstance(returns, pd.Series) or not isinstance(equity, pd.Series):
        raise TypeError("returns and equity must be pandas Series.")

    max_drawdown = calculate_maximum_drawdown(equity)
    if max_drawdown >= 0:
        return 0.0

    annualized_return = calculate_annualized_return(returns, trading_days_per_year=trading_days_per_year)
    return float(annualized_return / abs(max_drawdown))


def calculate_value_at_risk(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """Calculate the historical 95% Value at Risk using the empirical distribution."""
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series.")

    if len(returns) < 2:
        raise ValueError("At least two observations are required.")

    if not np.isfinite(returns.astype(float)).all():
        raise ValueError("returns must contain only finite values.")

    if not (0 < confidence_level < 1):
        raise ValueError("confidence_level must be between 0 and 1.")

    return float(np.quantile(returns.astype(float), 1.0 - confidence_level))


def calculate_expected_shortfall(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """Calculate the historical 95% Expected Shortfall."""
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series.")

    if len(returns) < 2:
        raise ValueError("At least two observations are required.")

    if not np.isfinite(returns.astype(float)).all():
        raise ValueError("returns must contain only finite values.")

    if not (0 < confidence_level < 1):
        raise ValueError("confidence_level must be between 0 and 1.")

    threshold = np.quantile(returns.astype(float), 1.0 - confidence_level)
    tail = returns[returns <= threshold]
    if tail.empty:
        return float(returns.min())
    return float(tail.mean())


def calculate_hit_rate(returns: pd.Series) -> float:
    """Calculate the fraction of positive daily returns."""
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series.")

    if len(returns) < 2:
        raise ValueError("At least two observations are required.")

    if not np.isfinite(returns.astype(float)).all():
        raise ValueError("returns must contain only finite values.")

    return float((returns > 0).mean())


def summarize_performance(backtest_frame: pd.DataFrame, trading_days_per_year: int = 252, risk_free_rate: float = 0.0, confidence_level: float = 0.95) -> PerformanceSummary:
    """Create a performance summary from a backtest DataFrame."""
    if not isinstance(backtest_frame, pd.DataFrame):
        raise TypeError("backtest_frame must be a pandas DataFrame.")

    required_columns = {"net_return", "equity"}
    if not required_columns.issubset(backtest_frame.columns):
        raise ValueError("backtest_frame is missing required columns: net_return, equity")

    returns = backtest_frame["net_return"].astype(float)
    equity = backtest_frame["equity"].astype(float)
    _validate_returns(returns, equity, trading_days_per_year, confidence_level)

    total_return = calculate_total_return(equity)
    annualized_return = calculate_annualized_return(returns, trading_days_per_year=trading_days_per_year)
    annualized_volatility = calculate_annualized_volatility(returns, trading_days_per_year=trading_days_per_year)
    sharpe_ratio = calculate_sharpe_ratio(returns, trading_days_per_year=trading_days_per_year, risk_free_rate=risk_free_rate)
    sortino_ratio = calculate_sortino_ratio(returns, trading_days_per_year=trading_days_per_year, risk_free_rate=risk_free_rate)
    maximum_drawdown = calculate_maximum_drawdown(equity)
    calmar_ratio = calculate_calmar_ratio(returns, equity, trading_days_per_year=trading_days_per_year)
    value_at_risk_95 = calculate_value_at_risk(returns, confidence_level=confidence_level)
    expected_shortfall_95 = calculate_expected_shortfall(returns, confidence_level=confidence_level)
    hit_rate = calculate_hit_rate(returns)

    return PerformanceSummary(
        total_return=total_return,
        annualized_return=annualized_return,
        annualized_volatility=annualized_volatility,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        maximum_drawdown=maximum_drawdown,
        calmar_ratio=calmar_ratio,
        value_at_risk_95=value_at_risk_95,
        expected_shortfall_95=expected_shortfall_95,
        hit_rate=hit_rate,
        average_daily_return=float(returns.mean()),
        best_day=float(returns.max()),
        worst_day=float(returns.min()),
        number_of_trading_days=int(len(returns)),
    )
