"""Benchmark comparison for pairs-trading strategies.

Provides benchmarks including cash, buy-and-hold of a market index, and an
equal-weight buy-and-hold of the two pair constituents.

.. note::

   An equity index is *not* a perfect benchmark for a market-neutral
   pairs-trading strategy.  These benchmarks provide context rather than a
   fully matched risk comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.risk_metrics import (
    calculate_annualized_return,
    calculate_annualized_volatility,
    calculate_maximum_drawdown,
    calculate_sharpe_ratio,
    calculate_total_return,
)


@dataclass(frozen=True)
class BenchmarkResult:
    """Result comparing a strategy to a benchmark over the same period."""

    strategy_return: float
    strategy_annualized_return: float
    strategy_annualized_volatility: float
    strategy_sharpe_ratio: float
    strategy_max_drawdown: float
    strategy_final_equity: float
    benchmark_name: str
    benchmark_return: float
    benchmark_annualized_return: float
    benchmark_annualized_volatility: float
    benchmark_sharpe_ratio: float
    benchmark_max_drawdown: float
    benchmark_final_equity: float

    @property
    def excess_return(self) -> float:
        """Strategy return minus benchmark return."""
        return self.strategy_return - self.benchmark_return


DEFAULT_BENCHMARK_TICKER = "OMX.HE"  # OMX Helsinki 25, configurable


def cash_benchmark_return(
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    annual_rate: float = 0.0,
) -> float:
    """Return the total return of a cash (risk-free) benchmark.

    Parameters
    ----------
    test_start, test_end:
        Period over which to compute cash return.
    annual_rate:
        Annualised cash return rate (e.g. 0.0 for zero return, 0.03 for 3%).

    Returns
    -------
    float
        Total return over the period.
    """
    if not isinstance(test_start, pd.Timestamp) or not isinstance(test_end, pd.Timestamp):
        raise TypeError("test_start and test_end must be pd.Timestamp.")
    if test_end <= test_start:
        raise ValueError("test_end must be after test_start.")

    days = (test_end - test_start).days
    years = days / 365.0
    return (1.0 + annual_rate) ** years - 1.0


def buy_hold_return(
    prices: pd.Series,
) -> float:
    """Calculate the total buy-and-hold return from a price series.

    Parameters
    ----------
    prices:
        Price series aligned to the strategy test period.  The first
        observation is the purchase price, the last is the exit price.

    Returns
    -------
    float
        Total return.
    """
    if not isinstance(prices, pd.Series):
        raise TypeError("prices must be a pandas Series.")
    if len(prices) < 2:
        raise ValueError("At least two observations are required.")
    if (prices <= 0).any():
        raise ValueError("Prices must be positive.")
    return float(prices.iloc[-1] / prices.iloc[0] - 1.0)


def equal_weight_benchmark(
    price_y: pd.Series,
    price_x: pd.Series,
    initial_capital: float,
) -> pd.Series:
    """Calculate an equal-weight buy-and-hold benchmark equity curve.

    50% of capital is allocated to ticker Y and 50% to ticker X at the
    start, held to the end.

    Parameters
    ----------
    price_y, price_x:
        Price series for the two constituents, aligned to common dates.
    initial_capital:
        Starting capital.

    Returns
    -------
    pd.Series
        Daily equity curve.
    """
    if not isinstance(price_y, pd.Series) or not isinstance(price_x, pd.Series):
        raise TypeError("price_y and price_x must be pandas Series.")
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive.")

    aligned = pd.concat(
        [price_y.astype(float).rename("y"), price_x.astype(float).rename("x")],
        axis=1,
    ).dropna()
    if len(aligned) < 2:
        raise ValueError("Not enough overlapping observations.")

    shares_y = (initial_capital * 0.5) / aligned["y"].iloc[0]
    shares_x = (initial_capital * 0.5) / aligned["x"].iloc[0]
    equity = shares_y * aligned["y"] + shares_x * aligned["x"]
    return equity


def market_index_benchmark(
    index_prices: pd.Series,
    initial_capital: float,
) -> pd.Series:
    """Calculate a buy-and-hold market index benchmark equity curve.

    Parameters
    ----------
    index_prices:
        Price series for the market index, aligned to the strategy period.
    initial_capital:
        Starting capital (fully invested).

    Returns
    -------
    pd.Series
        Daily equity curve.
    """
    if not isinstance(index_prices, pd.Series):
        raise TypeError("index_prices must be a pandas Series.")
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive.")
    if (index_prices <= 0).any():
        raise ValueError("Index prices must be positive.")

    shares = initial_capital / index_prices.iloc[0]
    return shares * index_prices


def align_benchmark_dates(
    strategy_equity: pd.Series,
    benchmark_equity: pd.Series,
) -> pd.Series:
    """Align a benchmark equity curve to the strategy test period dates.

    Both series are sorted and only overlapping dates are retained.
    """
    if not isinstance(strategy_equity, pd.Series) or not isinstance(benchmark_equity, pd.Series):
        raise TypeError("Both inputs must be pandas Series.")

    aligned = benchmark_equity.reindex(strategy_equity.index).ffill().bfill()
    if aligned.isna().any():
        raise ValueError("Benchmark data cannot fully cover the strategy period.")
    return aligned


def compare_benchmarks(
    strategy_equity: pd.Series,
    strategy_returns: pd.Series,
    initial_capital: float,
    ticker_y: str,
    ticker_x: str,
    price_y: pd.Series,
    price_x: pd.Series,
    index_prices: pd.Series | None = None,
    cash_rate: float = 0.0,
    trading_days_per_year: int = 252,
) -> pd.DataFrame:
    """Compare the strategy against multiple benchmarks.

    Parameters
    ----------
    strategy_equity:
        Strategy equity curve (from backtest frame).
    strategy_returns:
        Strategy daily net returns.
    initial_capital:
        Starting capital.
    ticker_y, ticker_x:
        Constituent tickers.
    price_y, price_x:
        Constituent price series (full test period).
    index_prices:
        Optional market index price series.
    cash_rate:
        Annualised cash return rate.
    trading_days_per_year:
        Trading days for annualisation.

    Returns
    -------
    pd.DataFrame
        One row per benchmark with columns:
        ``benchmark_name``, ``total_return``, ``annualized_return``,
        ``annualized_volatility``, ``sharpe_ratio``, ``maximum_drawdown``,
        ``final_equity``.
    """
    rows: list[dict] = []

    # Strategy row
    strategy_return = calculate_total_return(strategy_equity)
    strategy_ann_return = calculate_annualized_return(strategy_returns, trading_days_per_year)
    strategy_ann_vol = calculate_annualized_volatility(strategy_returns, trading_days_per_year)
    strategy_sharpe = calculate_sharpe_ratio(strategy_returns, trading_days_per_year)
    strategy_dd = calculate_maximum_drawdown(strategy_equity)

    # --- Cash benchmark ---
    # Cash has a constant equity curve
    cash_equity_daily = pd.Series(initial_capital, index=strategy_equity.index)
    cash_returns = pd.Series(cash_rate / trading_days_per_year, index=strategy_equity.index)

    rows.append(
        _benchmark_row(
            "Cash", cash_equity_daily, cash_returns, initial_capital, trading_days_per_year
        )
    )

    # --- Equal-weight buy-and-hold ---
    try:
        ew_equity = equal_weight_benchmark(price_y, price_x, initial_capital)
        ew_equity_aligned = align_benchmark_dates(strategy_equity, ew_equity)
        ew_returns = ew_equity_aligned.pct_change().fillna(0.0)
        rows.append(_benchmark_row(
            f"Equal-weight {ticker_y}/{ticker_x}",
            ew_equity_aligned, ew_returns, initial_capital, trading_days_per_year,
        ))
    except (ValueError, TypeError):
        pass

    # --- Market index ---
    if index_prices is not None:
        try:
            idx_equity = market_index_benchmark(index_prices, initial_capital)
            idx_equity_aligned = align_benchmark_dates(strategy_equity, idx_equity)
            idx_returns = idx_equity_aligned.pct_change().fillna(0.0)
            rows.append(_benchmark_row(
                "Market index buy-and-hold",
                idx_equity_aligned, idx_returns, initial_capital, trading_days_per_year,
            ))
        except (ValueError, TypeError):
            pass

    # --- Strategy row ---
    rows.append(
        {
            "benchmark_name": "Strategy (pairs trading)",
            "total_return": strategy_return,
            "annualized_return": strategy_ann_return,
            "annualized_volatility": strategy_ann_vol,
            "sharpe_ratio": strategy_sharpe,
            "maximum_drawdown": strategy_dd,
            "final_equity": float(strategy_equity.iloc[-1]),
        }
    )

    result = pd.DataFrame(rows)
    return result


def _benchmark_row(
    name: str,
    equity: pd.Series,
    returns: pd.Series,
    initial_capital: float,
    trading_days_per_year: int,
) -> dict:
    return {
        "benchmark_name": name,
        "total_return": calculate_total_return(equity),
        "annualized_return": calculate_annualized_return(returns, trading_days_per_year),
        "annualized_volatility": calculate_annualized_volatility(returns, trading_days_per_year),
        "sharpe_ratio": calculate_sharpe_ratio(returns, trading_days_per_year),
        "maximum_drawdown": calculate_maximum_drawdown(equity),
        "final_equity": float(equity.iloc[-1]),
    }
