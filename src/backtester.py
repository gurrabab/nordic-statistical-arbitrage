"""Simple backtesting utilities for delayed spread-based strategy signals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestParameters:
    """Configuration for a simple pairs-trading backtest."""

    initial_capital: float
    transaction_cost_bps: float
    slippage_bps: float
    annual_borrow_cost: float
    trading_days_per_year: int = 252

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive.")
        if self.transaction_cost_bps < 0 or self.slippage_bps < 0 or self.annual_borrow_cost < 0:
            raise ValueError("Costs must be non-negative.")
        if self.trading_days_per_year <= 0:
            raise ValueError("trading_days_per_year must be positive.")


def calculate_daily_returns(price_frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily simple returns for each price series."""
    if not isinstance(price_frame, pd.DataFrame):
        raise TypeError("price_frame must be a pandas DataFrame.")

    returns = price_frame.pct_change().dropna()
    columns = [col.replace("price_", "") for col in returns.columns]
    returns = returns.set_axis(columns, axis=1)
    returns = returns.rename(columns=lambda col: f"return_{col.lower()}")
    return returns


def _validate_inputs(
    price_y: pd.Series,
    price_x: pd.Series,
    signal_positions: pd.Series,
    hedge_ratio: float,
    parameters: BacktestParameters,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    if not isinstance(price_y, pd.Series) or not isinstance(price_x, pd.Series) or not isinstance(signal_positions, pd.Series):
        raise TypeError("price_y, price_x, and signal_positions must be pandas Series.")

    if not np.isfinite(hedge_ratio):
        raise ValueError("hedge_ratio must be finite.")

    if not isinstance(parameters, BacktestParameters):
        raise TypeError("parameters must be a BacktestParameters instance.")

    if not isinstance(price_y.index, pd.DatetimeIndex) or not isinstance(price_x.index, pd.DatetimeIndex):
        raise ValueError("price indexes must be datetime-like.")

    aligned = pd.concat(
        [price_y.astype(float).rename("price_y"), price_x.astype(float).rename("price_x"), signal_positions.rename("signal_position")],
        axis=1,
    ).sort_index()
    aligned = aligned.replace([np.inf, -np.inf], np.nan).dropna(how="any")

    if aligned.empty:
        raise ValueError("No overlapping observations remain after alignment.")

    if not np.all(aligned["price_y"] > 0) or not np.all(aligned["price_x"] > 0):
        raise ValueError("Prices must be positive.")

    if not np.isin(aligned["signal_position"].unique(), [-1, 0, 1]).all():
        raise ValueError("positions must only contain -1, 0, or 1.")

    return aligned["price_y"], aligned["price_x"], aligned["signal_position"]


def _weights_for_position(signal_position: int, hedge_ratio: float) -> tuple[float, float]:
    if signal_position == 1:
        weight_y = 1.0 / (1.0 + abs(hedge_ratio))
        weight_x = -hedge_ratio / (1.0 + abs(hedge_ratio))
    elif signal_position == -1:
        weight_y = -1.0 / (1.0 + abs(hedge_ratio))
        weight_x = hedge_ratio / (1.0 + abs(hedge_ratio))
    else:
        return 0.0, 0.0

    return weight_y, weight_x


def run_backtest(
    price_y: pd.Series,
    price_x: pd.Series,
    signal_positions: pd.Series,
    hedge_ratio: float,
    parameters: BacktestParameters,
) -> pd.DataFrame:
    """Run a delayed-position backtest for a spread signal.

    The signal observed at day t is executed at day t+1, so the strategy uses a
    one-day lag between signal generation and portfolio implementation.
    """
    price_y, price_x, signal_positions = _validate_inputs(price_y, price_x, signal_positions, hedge_ratio, parameters)

    returns_y = price_y.pct_change().fillna(0.0)
    returns_x = price_x.pct_change().fillna(0.0)

    frame = pd.DataFrame(
        {
            "price_y": price_y,
            "price_x": price_x,
            "return_y": returns_y,
            "return_x": returns_x,
            "signal_position": signal_positions,
        },
        index=price_y.index,
    )

    frame["executed_position"] = 0
    frame["weight_y"] = 0.0
    frame["weight_x"] = 0.0
    frame["gross_exposure"] = 0.0
    frame["net_exposure"] = 0.0
    frame["gross_return"] = 0.0
    frame["turnover"] = 0.0
    frame["transaction_cost"] = 0.0
    frame["slippage_cost"] = 0.0
    frame["borrow_cost"] = 0.0
    frame["net_return"] = 0.0
    frame["equity"] = float(parameters.initial_capital)
    frame["cumulative_return"] = 0.0

    previous_weights = (0.0, 0.0)
    equity = float(parameters.initial_capital)
    daily_borrow_rate = parameters.annual_borrow_cost / parameters.trading_days_per_year

    for idx in range(1, len(frame)):
        signal_position = int(frame.iloc[idx - 1]["signal_position"])
        frame.at[frame.index[idx], "executed_position"] = signal_position

        weight_y, weight_x = _weights_for_position(signal_position, hedge_ratio)
        gross_exposure = abs(weight_y) + abs(weight_x)
        if gross_exposure > 0:
            weight_y = weight_y / gross_exposure
            weight_x = weight_x / gross_exposure
            gross_exposure = 1.0
        else:
            weight_y = 0.0
            weight_x = 0.0
            gross_exposure = 0.0

        frame.at[frame.index[idx], "weight_y"] = weight_y
        frame.at[frame.index[idx], "weight_x"] = weight_x
        frame.at[frame.index[idx], "gross_exposure"] = gross_exposure
        frame.at[frame.index[idx], "net_exposure"] = weight_y + weight_x

        turnover = abs(weight_y - previous_weights[0]) + abs(weight_x - previous_weights[1])
        transaction_cost = turnover * equity * (parameters.transaction_cost_bps / 10000.0)
        slippage_cost = turnover * equity * (parameters.slippage_bps / 10000.0)
        short_exposure = max(-weight_y, 0.0) + max(-weight_x, 0.0)
        borrow_cost = short_exposure * equity * daily_borrow_rate

        gross_return = weight_y * frame.iloc[idx]["return_y"] + weight_x * frame.iloc[idx]["return_x"]
        equity_next = equity * (1.0 + gross_return) - transaction_cost - slippage_cost - borrow_cost
        net_return = (equity_next / equity) - 1.0 if equity > 0 else 0.0
        equity = equity_next

        frame.at[frame.index[idx], "gross_return"] = gross_return
        frame.at[frame.index[idx], "turnover"] = turnover
        frame.at[frame.index[idx], "transaction_cost"] = transaction_cost
        frame.at[frame.index[idx], "slippage_cost"] = slippage_cost
        frame.at[frame.index[idx], "borrow_cost"] = borrow_cost
        frame.at[frame.index[idx], "net_return"] = net_return
        frame.at[frame.index[idx], "equity"] = equity
        frame.at[frame.index[idx], "cumulative_return"] = (equity / parameters.initial_capital) - 1.0

        previous_weights = (weight_y, weight_x)

    frame.iloc[0, frame.columns.get_loc("equity")] = float(parameters.initial_capital)
    frame.iloc[0, frame.columns.get_loc("cumulative_return")] = 0.0
    return frame
