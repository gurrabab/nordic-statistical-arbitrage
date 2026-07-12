import numpy as np
import pandas as pd
import pytest

from src.backtester import BacktestParameters, calculate_daily_returns, run_backtest


def test_flat_positions_produce_zero_strategy_return() -> None:
    dates = pd.date_range("2020-01-01", periods=3, freq="D")
    price_y = pd.Series([100.0, 101.0, 102.0], index=dates)
    price_x = pd.Series([100.0, 101.0, 102.0], index=dates)
    signal_positions = pd.Series([0, 0, 0], index=dates)

    result = run_backtest(
        price_y=price_y,
        price_x=price_x,
        signal_positions=signal_positions,
        hedge_ratio=1.0,
        parameters=BacktestParameters(initial_capital=1000.0, transaction_cost_bps=5.0, slippage_bps=2.0, annual_borrow_cost=0.02),
    )

    assert result["gross_return"].eq(0.0).all()
    assert result["net_return"].eq(0.0).all()


def test_one_day_execution_delay() -> None:
    dates = pd.date_range("2020-01-01", periods=3, freq="D")
    price_y = pd.Series([100.0, 101.0, 104.0], index=dates)
    price_x = pd.Series([100.0, 100.0, 100.0], index=dates)
    signal_positions = pd.Series([1, 0, 0], index=dates)

    result = run_backtest(
        price_y=price_y,
        price_x=price_x,
        signal_positions=signal_positions,
        hedge_ratio=1.0,
        parameters=BacktestParameters(initial_capital=1000.0, transaction_cost_bps=5.0, slippage_bps=2.0, annual_borrow_cost=0.02),
    )

    assert result.iloc[0]["executed_position"] == 0
    assert result.iloc[1]["executed_position"] == 1
    assert result.iloc[2]["executed_position"] == 0


def test_correct_long_spread_weights() -> None:
    dates = pd.date_range("2020-01-01", periods=2, freq="D")
    price_y = pd.Series([100.0, 101.0], index=dates)
    price_x = pd.Series([100.0, 100.0], index=dates)
    signal_positions = pd.Series([1, 1], index=dates)

    result = run_backtest(
        price_y=price_y,
        price_x=price_x,
        signal_positions=signal_positions,
        hedge_ratio=2.0,
        parameters=BacktestParameters(initial_capital=1000.0, transaction_cost_bps=5.0, slippage_bps=2.0, annual_borrow_cost=0.02),
    )

    assert result.iloc[1]["weight_y"] == pytest.approx(1.0 / 3.0)
    assert result.iloc[1]["weight_x"] == pytest.approx(-2.0 / 3.0)


def test_correct_short_spread_weights() -> None:
    dates = pd.date_range("2020-01-01", periods=2, freq="D")
    price_y = pd.Series([100.0, 101.0], index=dates)
    price_x = pd.Series([100.0, 100.0], index=dates)
    signal_positions = pd.Series([-1, -1], index=dates)

    result = run_backtest(
        price_y=price_y,
        price_x=price_x,
        signal_positions=signal_positions,
        hedge_ratio=2.0,
        parameters=BacktestParameters(initial_capital=1000.0, transaction_cost_bps=5.0, slippage_bps=2.0, annual_borrow_cost=0.02),
    )

    assert result.iloc[1]["weight_y"] == pytest.approx(-1.0 / 3.0)
    assert result.iloc[1]["weight_x"] == pytest.approx(2.0 / 3.0)


def test_gross_exposure_normalization() -> None:
    dates = pd.date_range("2020-01-01", periods=2, freq="D")
    price_y = pd.Series([100.0, 101.0], index=dates)
    price_x = pd.Series([100.0, 100.0], index=dates)
    signal_positions = pd.Series([1, 1], index=dates)

    result = run_backtest(
        price_y=price_y,
        price_x=price_x,
        signal_positions=signal_positions,
        hedge_ratio=0.5,
        parameters=BacktestParameters(initial_capital=1000.0, transaction_cost_bps=5.0, slippage_bps=2.0, annual_borrow_cost=0.02),
    )

    assert result.iloc[1]["gross_exposure"] == pytest.approx(1.0)


def test_transaction_costs_on_entries_and_exits() -> None:
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    price_y = pd.Series([100.0, 101.0, 102.0, 103.0], index=dates)
    price_x = pd.Series([100.0, 100.0, 100.0, 100.0], index=dates)
    signal_positions = pd.Series([0, 1, 0, 0], index=dates)

    result = run_backtest(
        price_y=price_y,
        price_x=price_x,
        signal_positions=signal_positions,
        hedge_ratio=1.0,
        parameters=BacktestParameters(initial_capital=1000.0, transaction_cost_bps=5.0, slippage_bps=2.0, annual_borrow_cost=0.02),
    )

    assert result.iloc[2]["transaction_cost"] > 0.0
    assert result.iloc[3]["transaction_cost"] > 0.0


def test_no_transaction_cost_without_weight_changes() -> None:
    dates = pd.date_range("2020-01-01", periods=3, freq="D")
    price_y = pd.Series([100.0, 101.0, 102.0], index=dates)
    price_x = pd.Series([100.0, 100.0, 100.0], index=dates)
    signal_positions = pd.Series([1, 1, 1], index=dates)

    result = run_backtest(
        price_y=price_y,
        price_x=price_x,
        signal_positions=signal_positions,
        hedge_ratio=1.0,
        parameters=BacktestParameters(initial_capital=1000.0, transaction_cost_bps=5.0, slippage_bps=2.0, annual_borrow_cost=0.02),
    )

    assert result.iloc[2]["transaction_cost"] == pytest.approx(0.0)
    assert result.iloc[2]["slippage_cost"] == pytest.approx(0.0)


def test_borrow_cost_only_on_short_exposure() -> None:
    dates = pd.date_range("2020-01-01", periods=3, freq="D")
    price_y = pd.Series([100.0, 101.0, 102.0], index=dates)
    price_x = pd.Series([100.0, 100.0, 100.0], index=dates)
    signal_positions = pd.Series([0, -1, 0], index=dates)

    result = run_backtest(
        price_y=price_y,
        price_x=price_x,
        signal_positions=signal_positions,
        hedge_ratio=1.0,
        parameters=BacktestParameters(initial_capital=1000.0, transaction_cost_bps=5.0, slippage_bps=2.0, annual_borrow_cost=0.02),
    )

    assert result.iloc[2]["borrow_cost"] > 0.0
    assert result.iloc[0]["borrow_cost"] == pytest.approx(0.0)


def test_invalid_positions_raise_value_error() -> None:
    dates = pd.date_range("2020-01-01", periods=2, freq="D")
    price_y = pd.Series([100.0, 101.0], index=dates)
    price_x = pd.Series([100.0, 100.0], index=dates)
    signal_positions = pd.Series([2, 0], index=dates)

    with pytest.raises(ValueError, match="positions"):
        run_backtest(
            price_y=price_y,
            price_x=price_x,
            signal_positions=signal_positions,
            hedge_ratio=1.0,
            parameters=BacktestParameters(initial_capital=1000.0, transaction_cost_bps=5.0, slippage_bps=2.0, annual_borrow_cost=0.02),
        )


def test_invalid_prices_raise_value_error() -> None:
    dates = pd.date_range("2020-01-01", periods=2, freq="D")
    price_y = pd.Series([100.0, 0.0], index=dates)
    price_x = pd.Series([100.0, 100.0], index=dates)
    signal_positions = pd.Series([0, 0], index=dates)

    with pytest.raises(ValueError, match="positive"):
        run_backtest(
            price_y=price_y,
            price_x=price_x,
            signal_positions=signal_positions,
            hedge_ratio=1.0,
            parameters=BacktestParameters(initial_capital=1000.0, transaction_cost_bps=5.0, slippage_bps=2.0, annual_borrow_cost=0.02),
        )


def test_calculate_daily_returns_matches_expected_values() -> None:
    prices = pd.DataFrame({"y": [100.0, 110.0, 121.0], "x": [100.0, 90.0, 81.0]})

    result = calculate_daily_returns(prices)

    assert result.loc[1, "return_y"] == pytest.approx(0.1)
    assert result.loc[1, "return_x"] == pytest.approx(-0.1)
