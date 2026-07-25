"""Tests for sensitivity module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.sensitivity import (
    generate_parameter_grid,
    parameter_stability_summary,
    run_cost_sensitivity,
    run_parameter_sensitivity,
)

# ---------------------------------------------------------------------------
# generate_parameter_grid
# ---------------------------------------------------------------------------

class TestGenerateParameterGrid:
    def test_default_grid_not_empty(self):
        grid = generate_parameter_grid()
        assert len(grid) > 0
        cols = ["lookback_window", "entry_threshold", "exit_threshold", "stop_threshold"]
        assert all(c in grid.columns for c in cols)

    def test_invalid_combinations_excluded(self):
        """entry must be > exit and stop > entry."""
        grid = generate_parameter_grid(
            lookback_values=[60],
            entry_values=[1.5],
            exit_values=[2.0],  # exit > entry → should be excluded
            stop_values=[3.0],
        )
        assert len(grid) == 0

    def test_all_combinations_valid(self):
        grid = generate_parameter_grid(
            lookback_values=[60, 90],
            entry_values=[2.0],
            exit_values=[0.0, 0.5],
            stop_values=[3.0, 4.0],
        )
        # 2 * 1 * 2 * 2 = 8, but check exit < entry: for entry=2.0, exit=0.0 and 0.5 are ok
        # For each: stop must be > entry (2.0), so stop=3.0 and 4.0 are ok
        assert len(grid) == 8

    def test_deterministic(self):
        g1 = generate_parameter_grid(
            lookback_values=[60, 90], entry_values=[2.0], exit_values=[0.5], stop_values=[3.5]
        )
        g2 = generate_parameter_grid(
            lookback_values=[60, 90], entry_values=[2.0], exit_values=[0.5], stop_values=[3.5]
        )
        pd.testing.assert_frame_equal(g1, g2)

    def test_empty_when_no_valid_params(self):
        grid = generate_parameter_grid(
            lookback_values=[60],
            entry_values=[1.0],
            exit_values=[2.0],  # all exit > entry → empty
            stop_values=[3.0],
        )
        assert len(grid) == 0


# ---------------------------------------------------------------------------
# run_cost_sensitivity
# ---------------------------------------------------------------------------

class TestRunCostSensitivity:
    def test_returns_dataframe(self):
        dates = pd.date_range("2020-01-01", periods=50, freq="B")
        np.random.seed(42)
        py = pd.Series(100 * np.exp(np.cumsum(np.random.randn(50) * 0.01)), index=dates) + 100
        px = pd.Series(50 * np.exp(np.cumsum(np.random.randn(50) * 0.01)), index=dates) + 50
        signal = pd.Series(np.random.choice([-1, 0, 1], size=50), index=dates)

        result = run_cost_sensitivity(
            py, px, signal, hedge_ratio=0.5,
            transaction_cost_values=[0.0, 5.0],
            slippage_values=[0.0],
            borrow_cost_values=[0.0],
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2  # 2 transaction cost values × 1 slippage × 1 borrow
        assert "total_return" in result.columns
        assert "sharpe_ratio" in result.columns
        assert "is_baseline" in result.columns

    def test_higher_costs_never_better(self):
        """Higher costs should not produce higher returns (ceteris paribus)."""
        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        np.random.seed(42)
        py = pd.Series(100 * np.exp(np.cumsum(np.random.randn(100) * 0.01)), index=dates) + 100
        px = pd.Series(50 * np.exp(np.cumsum(np.random.randn(100) * 0.01)), index=dates) + 50
        signal = pd.Series(np.random.choice([-1, 0, 1], size=100), index=dates)

        result = run_cost_sensitivity(
            py, px, signal, hedge_ratio=0.5,
            transaction_cost_values=[0.0, 10.0, 30.0],
            slippage_values=[0.0],
            borrow_cost_values=[0.0],
        )
        returns = result["total_return"].values
        # Returns should be non-increasing as costs increase
        assert list(returns) == sorted(returns, reverse=True)

    def test_zero_costs_match_zero(self):
        dates = pd.date_range("2020-01-01", periods=50, freq="B")
        np.random.seed(42)
        py = pd.Series(100 * np.exp(np.cumsum(np.random.randn(50) * 0.01)), index=dates) + 100
        px = pd.Series(50 * np.exp(np.cumsum(np.random.randn(50) * 0.01)), index=dates) + 50
        signal = pd.Series(np.random.choice([-1, 0, 1], size=50), index=dates)

        result = run_cost_sensitivity(
            py, px, signal, hedge_ratio=0.5,
            transaction_cost_values=[0.0],
            slippage_values=[0.0],
            borrow_cost_values=[0.0],
        )
        assert result["total_cost"].iloc[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# run_parameter_sensitivity
# ---------------------------------------------------------------------------

class TestRunParameterSensitivity:
    def test_returns_dataframe(self):
        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        np.random.seed(42)
        train_y = pd.Series(
            100 * np.exp(np.cumsum(np.random.randn(60) * 0.01)), index=dates[:60]
        ) + 100
        train_x = pd.Series(
            50 * np.exp(np.cumsum(np.random.randn(60) * 0.01)), index=dates[:60]
        ) + 50
        test_y = pd.Series(
            100 * np.exp(np.cumsum(np.random.randn(40) * 0.01)), index=dates[60:]
        ) + 100
        test_x = pd.Series(
            50 * np.exp(np.cumsum(np.random.randn(40) * 0.01)), index=dates[60:]
        ) + 50

        result = run_parameter_sensitivity(
            train_y, train_x, test_y, test_x,
            lookback_values=[60, 90],
            entry_values=[2.0],
            exit_values=[0.5],
            stop_values=[3.5],
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "data_segment" in result.columns
        assert all(result["data_segment"] == "test")


# ---------------------------------------------------------------------------
# parameter_stability_summary
# ---------------------------------------------------------------------------

class TestParameterStabilitySummary:
    def test_summary_fields(self):
        data = pd.DataFrame({
            "sharpe_ratio": [0.5, 1.0, 1.5, -0.2, 0.8],
        })
        summary = parameter_stability_summary(data)
        assert summary["n_combinations"] == 5
        assert summary["n_profitable"] == 4
        assert summary["profitable_proportion"] == pytest.approx(0.8)
        assert summary["median_sharpe"] == pytest.approx(0.8)

    def test_empty_dataframe(self):
        summary = parameter_stability_summary(pd.DataFrame())
        assert summary["n_combinations"] == 0
        assert np.isnan(summary["median_sharpe"])
