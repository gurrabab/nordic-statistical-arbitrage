"""Tests for benchmarks module."""

from __future__ import annotations

import pandas as pd
import pytest

from src.benchmarks import (
    BenchmarkResult,
    align_benchmark_dates,
    buy_hold_return,
    cash_benchmark_return,
    compare_benchmarks,
    equal_weight_benchmark,
    market_index_benchmark,
)

# ---------------------------------------------------------------------------
# cash_benchmark_return
# ---------------------------------------------------------------------------

class TestCashBenchmarkReturn:
    def test_zero_rate(self):
        start = pd.Timestamp("2020-01-01")
        end = pd.Timestamp("2020-12-31")
        ret = cash_benchmark_return(start, end, annual_rate=0.0)
        assert ret == pytest.approx(0.0)

    def test_positive_rate(self):
        start = pd.Timestamp("2020-01-01")
        end = pd.Timestamp("2021-01-01")
        ret = cash_benchmark_return(start, end, annual_rate=0.05)
        assert ret == pytest.approx(0.05, rel=0.01)

    def test_end_before_start_raises(self):
        with pytest.raises(ValueError, match="test_end must be after"):
            cash_benchmark_return(
                pd.Timestamp("2021-01-01"), pd.Timestamp("2020-01-01")
            )

    def test_invalid_types(self):
        with pytest.raises(TypeError):
            cash_benchmark_return("2020-01-01", "2020-12-31")


# ---------------------------------------------------------------------------
# buy_hold_return
# ---------------------------------------------------------------------------

class TestBuyHoldReturn:
    def test_positive_return(self):
        prices = pd.Series([100.0, 110.0, 120.0])
        assert buy_hold_return(prices) == pytest.approx(0.2)

    def test_negative_return(self):
        prices = pd.Series([100.0, 90.0, 80.0])
        assert buy_hold_return(prices) == pytest.approx(-0.2)

    def test_too_few_points(self):
        with pytest.raises(ValueError, match="At least two"):
            buy_hold_return(pd.Series([100.0]))

    def test_non_positive_price(self):
        with pytest.raises(ValueError, match="Prices must be positive"):
            buy_hold_return(pd.Series([100.0, 0.0]))

    def test_wrong_type(self):
        with pytest.raises(TypeError):
            buy_hold_return([100, 110])


# ---------------------------------------------------------------------------
# equal_weight_benchmark
# ---------------------------------------------------------------------------

class TestEqualWeightBenchmark:
    def test_equity_curve_length(self):
        dates = pd.date_range("2020-01-01", periods=5, freq="B")
        py = pd.Series([100, 102, 101, 103, 105], index=dates)
        px = pd.Series([50, 51, 52, 50, 49], index=dates)
        equity = equal_weight_benchmark(py, px, 100_000)
        assert len(equity) == 5
        assert equity.iloc[0] == pytest.approx(100_000.0)

    def test_missing_data_alignment(self):
        dates1 = pd.date_range("2020-01-01", periods=5, freq="B")
        dates2 = pd.date_range("2020-01-03", periods=3, freq="B")
        py = pd.Series([100, 102, 101, 103, 105], index=dates1)
        px = pd.Series([50, 51, 52], index=dates2)
        equity = equal_weight_benchmark(py, px, 100_000)
        assert len(equity) == 3  # only overlapping dates

    def test_wrong_type(self):
        with pytest.raises(TypeError):
            equal_weight_benchmark([1, 2], [3, 4], 100_000)


# ---------------------------------------------------------------------------
# market_index_benchmark
# ---------------------------------------------------------------------------

class TestMarketIndexBenchmark:
    def test_equity_curve(self):
        prices = pd.Series([2000, 2100, 2050, 2200])
        equity = market_index_benchmark(prices, 100_000)
        assert equity.iloc[0] == pytest.approx(100_000.0)
        assert equity.iloc[-1] == pytest.approx(100_000 * 2200 / 2000)

    def test_non_positive(self):
        with pytest.raises(ValueError, match="positive"):
            market_index_benchmark(pd.Series([2000, 0]), 100_000)

    def test_wrong_type(self):
        with pytest.raises(TypeError):
            market_index_benchmark([2000, 2100], 100_000)


# ---------------------------------------------------------------------------
# align_benchmark_dates
# ---------------------------------------------------------------------------

class TestAlignBenchmarkDates:
    def test_alignment(self):
        strat = pd.Series([1, 2, 3], index=pd.date_range("2020-01-01", periods=3, freq="B"))
        bench = pd.Series([10, 20, 30], index=pd.date_range("2020-01-01", periods=3, freq="B"))
        aligned = align_benchmark_dates(strat, bench)
        assert len(aligned) == 3
        assert aligned.iloc[0] == 10.0

    def test_partial_coverage_raises(self):
        strat = pd.Series([1, 2, 3], index=pd.date_range("2020-01-01", periods=3, freq="B"))
        bench = pd.Series([10], index=pd.date_range("2020-01-05", periods=1, freq="B"))
        with pytest.raises(ValueError, match="cannot fully cover"):
            align_benchmark_dates(strat, bench)


# ---------------------------------------------------------------------------
# compare_benchmarks
# ---------------------------------------------------------------------------

class TestCompareBenchmarks:
    def test_basic_comparison(self):
        dates = pd.date_range("2020-01-01", periods=5, freq="B")
        strat_equity = pd.Series([100_000, 100_500, 101_000, 100_800, 101_200], index=dates)
        strat_returns = strat_equity.pct_change().fillna(0.0)
        py = pd.Series([100, 102, 101, 103, 105], index=dates)
        px = pd.Series([50, 51, 52, 50, 49], index=dates)

        result = compare_benchmarks(
            strategy_equity=strat_equity,
            strategy_returns=strat_returns,
            initial_capital=100_000,
            ticker_y="A",
            ticker_x="B",
            price_y=py,
            price_x=px,
        )
        assert "benchmark_name" in result.columns
        assert len(result) >= 2  # Cash + Equal-weight
        assert "Strategy (pairs trading)" in result["benchmark_name"].values
        assert "Cash" in result["benchmark_name"].values


# ---------------------------------------------------------------------------
# BenchmarkResult
# ---------------------------------------------------------------------------

class TestBenchmarkResult:
    def test_excess_return(self):
        result = BenchmarkResult(
            strategy_return=0.10,
            strategy_annualized_return=0.10,
            strategy_annualized_volatility=0.15,
            strategy_sharpe_ratio=0.67,
            strategy_max_drawdown=-0.05,
            strategy_final_equity=110_000,
            benchmark_name="Cash",
            benchmark_return=0.02,
            benchmark_annualized_return=0.02,
            benchmark_annualized_volatility=0.0,
            benchmark_sharpe_ratio=0.0,
            benchmark_max_drawdown=0.0,
            benchmark_final_equity=102_000,
        )
        assert result.excess_return == pytest.approx(0.08)
