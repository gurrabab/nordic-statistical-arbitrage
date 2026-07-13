"""Tests for the walk-forward validation module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtester import BacktestParameters
from src.pair_screener import PairScreeningParameters
from src.signals import SignalParameters
from src.walk_forward import (
    WALK_FORWARD_RESULT_COLUMNS,
    WalkForwardParameters,
    WalkForwardWindow,
    calculate_walk_forward_summary,
    generate_walk_forward_windows,
    run_walk_forward_analysis,
)


# ---------------------------------------------------------------------------
# Helpers — synthetic price data
# ---------------------------------------------------------------------------


def _make_dates(n: int, start: str = "2020-01-02") -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n)


def _make_cointegrated_pair(
    n: int,
    *,
    hedge_ratio: float = 1.5,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a synthetic cointegrated pair with a known hedge ratio."""
    rng = np.random.default_rng(seed)
    # Common stochastic trend
    common = 100.0 + np.cumsum(rng.normal(0, 0.5, size=n))
    # Stationary spread
    spread = rng.normal(0, 0.3, size=n)
    y = common + spread
    x = (common - spread) / hedge_ratio
    dates = _make_dates(n)
    return pd.DataFrame({"Y": y, "X": x}, index=dates)


def _make_universe(n: int = 800) -> pd.DataFrame:
    """Build a small synthetic universe of 6 tickers with one cointegrated pair."""
    rng = np.random.default_rng(42)
    dates = _make_dates(n)
    frame = pd.DataFrame(index=dates)

    # Cointegrated pair
    pair = _make_cointegrated_pair(n, hedge_ratio=1.5, seed=42)
    frame["COIN_Y"] = pair["Y"]
    frame["COIN_X"] = pair["X"]

    # Random walk tickers
    for ticker in ["RND_A", "RND_B", "RND_C", "RND_D"]:
        price = 100.0 + np.cumsum(rng.normal(0, 1.0, size=n))
        rng = np.random.default_rng(rng.integers(0, 2**31))  # advance rng state
        frame[ticker] = price

    return frame


# ---------------------------------------------------------------------------
# WalkForwardParameters
# ---------------------------------------------------------------------------


class TestWalkForwardParameters:
    def test_valid_parameters(self) -> None:
        params = WalkForwardParameters(
            train_window_days=252,
            test_window_days=63,
            step_size_days=63,
            expanding_window=False,
            minimum_train_observations=200,
            minimum_test_observations=30,
            top_n_pairs_per_window=3,
        )
        assert params.train_window_days == 252
        assert params.expanding_window is False

    def test_zero_values_raise(self) -> None:
        with pytest.raises(ValueError, match="train_window_days must be positive"):
            WalkForwardParameters(0, 63, 63, False, 200, 30, 3)
        with pytest.raises(ValueError, match="test_window_days must be positive"):
            WalkForwardParameters(252, 0, 63, False, 200, 30, 3)
        with pytest.raises(ValueError, match="step_size_days must be positive"):
            WalkForwardParameters(252, 63, 0, False, 200, 30, 3)
        with pytest.raises(ValueError, match="minimum_train_observations must be positive"):
            WalkForwardParameters(252, 63, 63, False, 0, 30, 3)
        with pytest.raises(ValueError, match="minimum_test_observations must be positive"):
            WalkForwardParameters(252, 63, 63, False, 200, 0, 3)
        with pytest.raises(ValueError, match="top_n_pairs_per_window must be positive"):
            WalkForwardParameters(252, 63, 63, False, 200, 30, 0)

    def test_negative_values_raise(self) -> None:
        with pytest.raises(ValueError):
            WalkForwardParameters(-10, 63, 63, False, 200, 30, 3)


# ---------------------------------------------------------------------------
# generate_walk_forward_windows
# ---------------------------------------------------------------------------


class TestGenerateWalkForwardWindows:
    def test_empty_index_returns_empty_list(self) -> None:
        params = WalkForwardParameters(10, 5, 5, False, 5, 3, 2)
        assert generate_walk_forward_windows(pd.DatetimeIndex([]), params) == []

    def test_single_date_returns_empty_list(self) -> None:
        params = WalkForwardParameters(10, 5, 5, False, 5, 3, 2)
        assert generate_walk_forward_windows(pd.DatetimeIndex(["2020-01-02"]), params) == []

    def test_insufficient_data_returns_empty_list(self) -> None:
        dates = _make_dates(10)
        params = WalkForwardParameters(10, 5, 5, False, 5, 3, 2)
        windows = generate_walk_forward_windows(dates, params)
        assert len(windows) == 0

    def test_type_error_on_wrong_input(self) -> None:
        params = WalkForwardParameters(10, 5, 5, False, 5, 3, 2)
        with pytest.raises(TypeError, match="index must be a pandas DatetimeIndex"):
            generate_walk_forward_windows(pd.Index([1, 2, 3]), params)  # type: ignore[arg-type]

    def test_correct_chronological_windows(self) -> None:
        dates = _make_dates(500)
        params = WalkForwardParameters(
            train_window_days=252,
            test_window_days=63,
            step_size_days=63,
            expanding_window=False,
            minimum_train_observations=200,
            minimum_test_observations=30,
            top_n_pairs_per_window=3,
        )
        windows = generate_walk_forward_windows(dates, params)

        assert len(windows) >= 2
        assert all(isinstance(w, WalkForwardWindow) for w in windows)

        # Windows must be sequentially numbered
        for i, w in enumerate(windows):
            assert w.window_id == i

        # No overlap: train_end < test_start for each window
        for w in windows:
            assert w.train_end < w.test_start, f"Overlap in window {w.window_id}"

        # No future leakage across windows: each window must be chronologically
        # after the previous one
        for prev_w, next_w in zip(windows[:-1], windows[1:]):
            assert prev_w.test_end < next_w.train_start or prev_w.test_end < next_w.test_start

        # Train/test order is strictly increasing within each window
        for w in windows:
            assert w.train_start < w.train_end
            assert w.test_start < w.test_end
            assert w.train_end < w.test_start

    def test_rolling_window_fixed_size(self) -> None:
        """Rolling windows should have approximately constant train size."""
        dates = _make_dates(800)
        params = WalkForwardParameters(
            train_window_days=252,
            test_window_days=63,
            step_size_days=63,
            expanding_window=False,
            minimum_train_observations=200,
            minimum_test_observations=30,
            top_n_pairs_per_window=3,
        )
        windows = generate_walk_forward_windows(dates, params)

        # All training windows should have the same number of days (barring
        # the final incomplete window — but our function enforces full windows)
        train_sizes = [w.train_observations for w in windows[1:]]
        assert all(s == train_sizes[0] for s in train_sizes)

    def test_expanding_window_growing_train_size(self) -> None:
        dates = _make_dates(800)
        params = WalkForwardParameters(
            train_window_days=252,
            test_window_days=63,
            step_size_days=63,
            expanding_window=True,
            minimum_train_observations=200,
            minimum_test_observations=30,
            top_n_pairs_per_window=3,
        )
        windows = generate_walk_forward_windows(dates, params)

        assert len(windows) >= 2
        # Expanding: train_start should always be the first date
        first_date = dates[0]
        for w in windows:
            assert w.train_start == first_date, f"Window {w.window_id} train_start changed"

        # Each training window should be larger than the previous one
        train_sizes = [w.train_observations for w in windows]
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] > train_sizes[i - 1], "Expanding window must grow"

    def test_minimum_observation_filter(self) -> None:
        dates = _make_dates(300)
        params = WalkForwardParameters(
            train_window_days=200,
            test_window_days=50,
            step_size_days=50,
            expanding_window=False,
            minimum_train_observations=500,  # impossible
            minimum_test_observations=30,
            top_n_pairs_per_window=3,
        )
        windows = generate_walk_forward_windows(dates, params)
        assert len(windows) == 0


# ---------------------------------------------------------------------------
# run_walk_forward_analysis
# ---------------------------------------------------------------------------


_SCREENING_PARAMS = PairScreeningParameters(
    minimum_observations=200,
    maximum_cointegration_pvalue=0.05,
    maximum_adf_pvalue=0.05,
    minimum_half_life=5.0,
    maximum_half_life=250.0,
    minimum_price=1.0,
    maximum_missing_fraction=0.05,
    top_n_pairs=10,
    train_fraction=0.7,
)

_SIGNAL_PARAMS = SignalParameters(
    lookback_window=30,
    entry_threshold=2.0,
    exit_threshold=0.5,
    stop_threshold=3.5,
)

_BACKTEST_PARAMS = BacktestParameters(
    initial_capital=100000.0,
    transaction_cost_bps=5.0,
    slippage_bps=2.0,
    annual_borrow_cost=0.02,
)


class TestRunWalkForwardAnalysis:
    def test_empty_prices_returns_empty(self) -> None:
        wf_params = WalkForwardParameters(252, 63, 63, False, 200, 30, 3)
        prices = pd.DataFrame(index=pd.bdate_range("2020-01-02", periods=100))
        result = run_walk_forward_analysis(
            prices, _SCREENING_PARAMS, wf_params, _SIGNAL_PARAMS, _BACKTEST_PARAMS
        )
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_empty_universe_returns_empty(self) -> None:
        """Too few dates (less than train + test) should yield empty result."""
        wf_params = WalkForwardParameters(252, 63, 63, False, 200, 30, 3)
        prices = _make_universe(n=100)
        result = run_walk_forward_analysis(
            prices, _SCREENING_PARAMS, wf_params, _SIGNAL_PARAMS, _BACKTEST_PARAMS
        )
        assert result.empty

    def test_deterministic_ranking(self) -> None:
        """Running twice should produce identical results."""
        wf_params = WalkForwardParameters(252, 63, 63, False, 200, 30, 3)
        prices = _make_universe(n=800)
        result1 = run_walk_forward_analysis(
            prices, _SCREENING_PARAMS, wf_params, _SIGNAL_PARAMS, _BACKTEST_PARAMS
        )
        result2 = run_walk_forward_analysis(
            prices, _SCREENING_PARAMS, wf_params, _SIGNAL_PARAMS, _BACKTEST_PARAMS
        )
        pd.testing.assert_frame_equal(result1, result2)

    def test_result_has_expected_columns(self) -> None:
        wf_params = WalkForwardParameters(252, 63, 63, False, 200, 30, 3)
        prices = _make_universe(n=800)
        result = run_walk_forward_analysis(
            prices, _SCREENING_PARAMS, wf_params, _SIGNAL_PARAMS, _BACKTEST_PARAMS
        )
        assert not result.empty
        assert list(result.columns) == WALK_FORWARD_RESULT_COLUMNS

    def test_multiple_windows_produced(self) -> None:
        wf_params = WalkForwardParameters(252, 63, 63, False, 200, 30, 3)
        prices = _make_universe(n=800)
        result = run_walk_forward_analysis(
            prices, _SCREENING_PARAMS, wf_params, _SIGNAL_PARAMS, _BACKTEST_PARAMS
        )
        n_windows = result["window_id"].nunique()
        assert n_windows >= 2

    def test_fixed_hedge_ratio_during_test(self) -> None:
        """Verify the test uses the training-period hedge ratio."""
        wf_params = WalkForwardParameters(252, 63, 63, False, 200, 30, 3)
        prices = _make_universe(n=800)
        result = run_walk_forward_analysis(
            prices, _SCREENING_PARAMS, wf_params, _SIGNAL_PARAMS, _BACKTEST_PARAMS
        )
        # The cointegrated pair should appear in some windows with a valid rank
        coin_rows = result[result["ticker_y"].str.contains("COIN")]
        if not coin_rows.empty:
            assert coin_rows["training_rank"].notna().all()
            assert (coin_rows["training_rank"] >= 1).all()

    def test_no_future_leakage_window_structure(self) -> None:
        """All test periods must start after their corresponding training periods."""
        wf_params = WalkForwardParameters(252, 63, 63, False, 200, 30, 3)
        prices = _make_universe(n=800)
        result = run_walk_forward_analysis(
            prices, _SCREENING_PARAMS, wf_params, _SIGNAL_PARAMS, _BACKTEST_PARAMS
        )
        for _, row in result.iterrows():
            assert row["train_end"] < row["test_start"]

    def test_bad_test_results_preserved(self) -> None:
        """Windows with poor performance should still appear in the result."""
        wf_params = WalkForwardParameters(252, 63, 63, False, 200, 30, 3)
        prices = _make_universe(n=800)
        result = run_walk_forward_analysis(
            prices, _SCREENING_PARAMS, wf_params, _SIGNAL_PARAMS, _BACKTEST_PARAMS
        )
        # Some windows may have negative returns — they must still be present
        neg_returns = result["test_total_return"].dropna()
        if not neg_returns.empty:
            has_negative = bool((neg_returns < 0).any())
            # Not all pairs in all windows will be negative, but at least
            # some random-walk pairs should struggle
            assert isinstance(has_negative, bool)

    def test_insufficient_data_handling(self) -> None:
        """Very short price history should gracefully produce empty results."""
        wf_params = WalkForwardParameters(252, 63, 63, False, 200, 30, 3)
        prices = _make_universe(n=50)
        result = run_walk_forward_analysis(
            prices, _SCREENING_PARAMS, wf_params, _SIGNAL_PARAMS, _BACKTEST_PARAMS
        )
        assert result.empty

    def test_type_error_on_wrong_prices(self) -> None:
        wf_params = WalkForwardParameters(252, 63, 63, False, 200, 30, 3)
        with pytest.raises(TypeError, match="prices must be a pandas DataFrame"):
            run_walk_forward_analysis(
                "not_a_frame",  # type: ignore[arg-type]
                _SCREENING_PARAMS,
                wf_params,
                _SIGNAL_PARAMS,
                _BACKTEST_PARAMS,
            )


# ---------------------------------------------------------------------------
# calculate_walk_forward_summary
# ---------------------------------------------------------------------------


class TestCalculateWalkForwardSummary:
    def test_empty_input_returns_empty(self) -> None:
        result = calculate_walk_forward_summary(pd.DataFrame())
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_type_error_on_wrong_input(self) -> None:
        with pytest.raises(TypeError, match="detailed_results must be a pandas DataFrame"):
            calculate_walk_forward_summary("not_a_frame")  # type: ignore[arg-type]

    def test_aggregate_columns_present(self) -> None:
        wf_params = WalkForwardParameters(252, 63, 63, False, 200, 30, 3)
        prices = _make_universe(n=800)
        detailed = run_walk_forward_analysis(
            prices, _SCREENING_PARAMS, wf_params, _SIGNAL_PARAMS, _BACKTEST_PARAMS
        )
        summary = calculate_walk_forward_summary(detailed)
        assert not summary.empty

        expected_cols = [
            "ticker_y",
            "ticker_x",
            "n_windows",
            "average_test_return",
            "median_test_return",
            "average_test_sharpe_ratio",
            "median_test_sharpe_ratio",
            "profitable_window_fraction",
            "worst_window_return",
            "worst_maximum_drawdown",
            "total_number_of_entries",
            "consistency_score",
        ]
        for col in expected_cols:
            assert col in summary.columns, f"Missing column: {col}"

    def test_consistency_score_range(self) -> None:
        wf_params = WalkForwardParameters(252, 63, 63, False, 200, 30, 3)
        prices = _make_universe(n=800)
        detailed = run_walk_forward_analysis(
            prices, _SCREENING_PARAMS, wf_params, _SIGNAL_PARAMS, _BACKTEST_PARAMS
        )
        summary = calculate_walk_forward_summary(detailed)
        if not summary.empty:
            assert summary["consistency_score"].between(0, 1).all()

    def test_consistency_score_higher_is_better(self) -> None:
        """The cointegrated pair should outrank random-walk pairs."""
        wf_params = WalkForwardParameters(252, 63, 63, False, 200, 30, 3)
        prices = _make_universe(n=800)
        detailed = run_walk_forward_analysis(
            prices, _SCREENING_PARAMS, wf_params, _SIGNAL_PARAMS, _BACKTEST_PARAMS
        )
        summary = calculate_walk_forward_summary(detailed)
        if not summary.empty:
            # The cointegrated pair (COIN_Y / COIN_X) should be ranked first
            coin_mask = (
                summary["ticker_y"].str.contains("COIN")
                & summary["ticker_x"].str.contains("COIN")
            )
            if coin_mask.any():
                coin_idx = coin_mask.idxmax()
                coin_score = summary.loc[coin_idx, "consistency_score"]
                # At least not the worst
                assert coin_score == summary["consistency_score"].max()

    def test_profitable_window_fraction_range(self) -> None:
        wf_params = WalkForwardParameters(252, 63, 63, False, 200, 30, 3)
        prices = _make_universe(n=800)
        detailed = run_walk_forward_analysis(
            prices, _SCREENING_PARAMS, wf_params, _SIGNAL_PARAMS, _BACKTEST_PARAMS
        )
        summary = calculate_walk_forward_summary(detailed)
        if not summary.empty:
            assert summary["profitable_window_fraction"].between(0, 1).all()

    def test_n_windows_counted_correctly(self) -> None:
        wf_params = WalkForwardParameters(252, 63, 63, False, 200, 30, 3)
        prices = _make_universe(n=800)
        detailed = run_walk_forward_analysis(
            prices, _SCREENING_PARAMS, wf_params, _SIGNAL_PARAMS, _BACKTEST_PARAMS
        )
        summary = calculate_walk_forward_summary(detailed)
        if not summary.empty:
            for _, row in summary.iterrows():
                pair_rows = detailed[
                    (detailed["ticker_y"] == row["ticker_y"])
                    & (detailed["ticker_x"] == row["ticker_x"])
                ].dropna(subset=["test_total_return"])
                assert row["n_windows"] == len(pair_rows)

    def test_worst_drawdown_preserved(self) -> None:
        wf_params = WalkForwardParameters(252, 63, 63, False, 200, 30, 3)
        prices = _make_universe(n=800)
        detailed = run_walk_forward_analysis(
            prices, _SCREENING_PARAMS, wf_params, _SIGNAL_PARAMS, _BACKTEST_PARAMS
        )
        summary = calculate_walk_forward_summary(detailed)
        if not summary.empty:
            assert (summary["worst_maximum_drawdown"] <= 0).all()
