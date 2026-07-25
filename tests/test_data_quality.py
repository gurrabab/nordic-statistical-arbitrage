"""Tests for data_quality module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_quality import (
    assess_all_tickers,
    assess_ticker_quality,
    compute_all_pair_overlaps,
    compute_pair_overlap,
)

# ---------------------------------------------------------------------------
# assess_ticker_quality
# ---------------------------------------------------------------------------

class TestAssessTickerQuality:
    def test_clean_data_passes(self):
        dates = pd.date_range("2020-01-01", periods=252, freq="B")
        prices = pd.Series(
            100.0 * np.exp(np.cumsum(np.random.randn(252) * 0.01)), index=dates
        ) + 100
        report = assess_ticker_quality(prices, "TEST")
        assert report.passed_quality_filter is True
        assert report.rejection_reason == ""
        assert report.duplicate_dates == 0
        assert report.non_positive_prices == 0
        assert report.constant_price_flag is False

    def test_missing_observations(self):
        dates = pd.date_range("2020-01-01", periods=252, freq="B")
        prices = pd.Series(
            100.0 * np.exp(np.cumsum(np.random.randn(252) * 0.01)), index=dates
        ) + 100
        prices.iloc[50:150] = np.nan  # 100 missing
        report = assess_ticker_quality(prices, "TEST", max_missing_fraction=0.10)
        assert report.missing_rows >= 100
        assert report.missing_fraction >= 0.10
        assert report.passed_quality_filter is False
        assert "missing" in report.rejection_reason

    def test_duplicate_dates(self):
        dates = pd.date_range("2020-01-01", periods=5, freq="B")
        # Create 6 values for 6 indices (5 original + 1 duplicate)
        prices = pd.Series(
            [100, 101, 102, 103, 104, 105],
            index=pd.DatetimeIndex(list(dates) + [dates[-1]]),
        )
        prices = prices.sort_index()
        report = assess_ticker_quality(prices, "TEST")
        assert report.duplicate_dates >= 1

    def test_non_positive_prices(self):
        dates = pd.date_range("2020-01-01", periods=10, freq="B")
        prices = pd.Series([100, 101, 0, -5, 103, 104, 105, 106, 107, 108], index=dates)
        report = assess_ticker_quality(prices, "TEST")
        assert report.non_positive_prices == 2

    def test_constant_price(self):
        dates = pd.date_range("2020-01-01", periods=10, freq="B")
        prices = pd.Series([100.0] * 10, index=dates)
        report = assess_ticker_quality(prices, "TEST")
        assert report.constant_price_flag is True
        assert report.passed_quality_filter is False

    def test_suspicious_returns(self):
        dates = pd.date_range("2020-01-01", periods=10, freq="B")
        prices = pd.Series([100, 100, 200, 100, 100, 100, 100, 100, 100, 100], index=dates)
        report = assess_ticker_quality(prices, "TEST", suspicious_return_threshold=0.10)
        assert report.suspicious_return_count >= 1
        assert report.largest_abs_daily_return >= 0.50

    def test_empty_series(self):
        prices = pd.Series([], dtype=float)
        report = assess_ticker_quality(prices, "EMPTY")
        assert report.passed_quality_filter is False
        assert "empty" in report.rejection_reason

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError):
            assess_ticker_quality([1, 2, 3], "BAD")


# ---------------------------------------------------------------------------
# assess_all_tickers
# ---------------------------------------------------------------------------

class TestAssessAllTickers:
    def test_returns_dataframe(self):
        dates = pd.date_range("2020-01-01", periods=10, freq="B")
        prices = pd.DataFrame({
            "A": np.random.randn(10) * 10 + 100,
            "B": np.random.randn(10) * 10 + 50,
        }, index=dates)
        prices.loc[prices.index[3:6], "B"] = np.nan
        result = assess_all_tickers(prices)
        assert len(result) == 2
        assert list(result["ticker"]) == ["A", "B"]
        assert "passed_quality_filter" in result.columns

    def test_empty_dataframe(self):
        result = assess_all_tickers(pd.DataFrame())
        assert len(result) == 0

    def test_wrong_type(self):
        with pytest.raises(TypeError):
            assess_all_tickers([1, 2, 3])


# ---------------------------------------------------------------------------
# compute_pair_overlap
# ---------------------------------------------------------------------------

class TestComputePairOverlap:
    def test_basic_overlap(self):
        dates = pd.date_range("2020-01-01", periods=10, freq="B")
        prices = pd.DataFrame({
            "A": np.random.randn(10) + 100,
            "B": np.random.randn(10) + 50,
        }, index=dates)
        result = compute_pair_overlap(prices, "A", "B")
        assert result["ticker_y"] == "A"
        assert result["ticker_x"] == "B"
        assert result["overlapping_observations"] == 10

    def test_partial_overlap(self):
        dates1 = pd.date_range("2020-01-01", periods=5, freq="B")
        dates2 = pd.date_range("2020-01-03", periods=5, freq="B")
        prices = pd.DataFrame({
            "A": pd.Series([100] * 5, index=dates1),
            "B": pd.Series([50] * 5, index=dates2),
        })
        result = compute_pair_overlap(prices, "A", "B")
        assert result["overlapping_observations"] == 3

    def test_missing_ticker_raises(self):
        dates = pd.date_range("2020-01-01", periods=5, freq="B")
        prices = pd.DataFrame({"A": range(5)}, index=dates)
        with pytest.raises(ValueError, match="not found"):
            compute_pair_overlap(prices, "A", "B")


# ---------------------------------------------------------------------------
# compute_all_pair_overlaps
# ---------------------------------------------------------------------------

class TestComputeAllPairOverlaps:
    def test_all_pairs(self):
        dates = pd.date_range("2020-01-01", periods=10, freq="B")
        prices = pd.DataFrame({
            "A": range(10), "B": range(10), "C": range(10),
        }, index=dates)
        result = compute_all_pair_overlaps(prices)
        assert len(result) == 3  # AB, AC, BC
        assert list(result.columns) == [
            "ticker_y", "ticker_x", "overlapping_observations",
            "overlap_start", "overlap_end", "overlap_fraction",
        ]

    def test_single_ticker(self):
        dates = pd.date_range("2020-01-01", periods=5, freq="B")
        prices = pd.DataFrame({"A": range(5)}, index=dates)
        result = compute_all_pair_overlaps(prices)
        assert len(result) == 0
