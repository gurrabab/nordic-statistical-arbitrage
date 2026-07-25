import numpy as np
import pandas as pd
import pytest

from src.data_loader import download_ticker_universe
from src.pair_screener import (
    PairScreeningParameters,
    apply_multiple_testing_corrections,
    benjamini_hochberg_correction,
    bonferroni_correction,
    evaluate_top_pairs,
    filter_ticker_universe,
    generate_unique_pairs,
    screen_pairs,
)


def _make_price_frame() -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=320, freq="D")
    rng = np.random.default_rng(7)
    x = np.cumsum(rng.normal(size=320)) + 100.0
    noise = rng.normal(scale=0.05, size=320)
    y = 1.2 + 0.75 * x + noise
    z = np.cumsum(rng.normal(size=320)) + 120.0
    w = np.linspace(50.0, 80.0, 320)
    prices = pd.DataFrame(
        {
            "A": np.maximum(1.0, y),
            "B": np.maximum(1.0, x),
            "C": np.maximum(1.0, z),
            "D": np.maximum(1.0, w),
        },
        index=index,
    )
    return prices


def test_generate_unique_pairs_returns_expected_count_and_no_reversals() -> None:
    tickers = ["A", "B", "C", "D"]

    pairs = generate_unique_pairs(tickers)

    assert len(pairs) == 6
    assert ("B", "A") not in pairs
    assert ("A", "B") in pairs


def test_filter_ticker_universe_removes_unsuitable_tickers() -> None:
    index = pd.date_range("2020-01-01", periods=260, freq="D")
    prices = pd.DataFrame(
        {
            "valid": np.linspace(100.0, 120.0, 260),
            "constant": np.ones(260),
            "negative": -np.linspace(1.0, 260.0, 260),
            "few_obs": np.linspace(50.0, 70.0, 260),
        },
        index=index,
    )
    prices.loc[prices.index[:180], "few_obs"] = np.nan
    prices["few_obs"] = prices["few_obs"].astype(float)

    filtered_prices, reasons = filter_ticker_universe(
        prices,
        PairScreeningParameters(
            minimum_observations=200,
            maximum_cointegration_pvalue=0.05,
            maximum_adf_pvalue=0.05,
            minimum_half_life=10.0,
            maximum_half_life=250.0,
            minimum_price=1.0,
            maximum_missing_fraction=0.05,
            top_n_pairs=5,
            train_fraction=0.7,
        ),
    )

    assert list(filtered_prices.columns) == ["valid"]
    assert reasons.loc["valid", "kept"]
    assert not reasons.loc["constant", "kept"]
    assert not reasons.loc["negative", "kept"]
    assert not reasons.loc["few_obs", "kept"]


def test_screen_pairs_returns_deterministic_ranking_and_training_only_score() -> None:
    prices = _make_price_frame()
    parameters = PairScreeningParameters(
        minimum_observations=200,
        maximum_cointegration_pvalue=0.2,
        maximum_adf_pvalue=0.2,
        minimum_half_life=0.5,
        maximum_half_life=200.0,
        minimum_price=1.0,
        maximum_missing_fraction=0.05,
        top_n_pairs=5,
        train_fraction=0.7,
    )

    results = screen_pairs(prices, parameters)

    assert not results.empty
    assert results["rank"].tolist() == list(range(1, len(results) + 1))
    assert results["score"].is_monotonic_decreasing
    assert "test_total_return" not in results.columns


def test_screen_pairs_records_rejection_reasons() -> None:
    prices = _make_price_frame()
    parameters = PairScreeningParameters(
        minimum_observations=400,
        maximum_cointegration_pvalue=0.05,
        maximum_adf_pvalue=0.05,
        minimum_half_life=10.0,
        maximum_half_life=200.0,
        minimum_price=1.0,
        maximum_missing_fraction=0.05,
        top_n_pairs=5,
        train_fraction=0.7,
    )

    results = screen_pairs(prices, parameters)

    assert results.empty
    assert "rejection_reason" in results.columns


def test_evaluate_top_pairs_uses_fixed_training_hedge_ratio() -> None:
    prices = _make_price_frame()
    parameters = PairScreeningParameters(
        minimum_observations=200,
        maximum_cointegration_pvalue=0.2,
        maximum_adf_pvalue=0.2,
        minimum_half_life=0.5,
        maximum_half_life=200.0,
        minimum_price=1.0,
        maximum_missing_fraction=0.05,
        top_n_pairs=5,
        train_fraction=0.7,
    )

    screening_results = screen_pairs(prices, parameters)
    comparison = evaluate_top_pairs(screening_results, prices, parameters, top_n=2)

    assert not comparison.empty
    assert "training_hedge_ratio" in comparison.columns
    assert comparison["training_hedge_ratio"].notna().all()


def test_empty_valid_pair_result() -> None:
    prices = _make_price_frame()
    parameters = PairScreeningParameters(
        minimum_observations=1000,
        maximum_cointegration_pvalue=0.0,
        maximum_adf_pvalue=0.0,
        minimum_half_life=1000.0,
        maximum_half_life=1000.0,
        minimum_price=1.0,
        maximum_missing_fraction=0.0,
        top_n_pairs=5,
        train_fraction=0.7,
    )

    results = screen_pairs(prices, parameters)

    assert results.empty


def test_partial_ticker_download_failures_are_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_downloader(tickers, *, start=None, end=None, period=None):
        if "A" in tickers:
            raise ValueError("boom")
        return pd.DataFrame({"B": [1.0, 2.0], "C": [3.0, 4.0]})

    prices, report = download_ticker_universe(
        ["A", "B", "C"], start="2020-01-01", downloader=fake_downloader
    )

    assert list(prices.columns) == ["B", "C"]
    assert report.loc["A", "status"] == "failed"
    assert report.loc["B", "status"] == "downloaded"


# ---------------------------------------------------------------------------
# Multiple-testing correction tests
# ---------------------------------------------------------------------------
class TestBonferroniCorrection:
    def test_basic_correction(self):
        pvalues = np.array([0.01, 0.04, 0.06, 0.50])
        adjusted, threshold = bonferroni_correction(pvalues, n_hypotheses=4, alpha=0.05)
        assert adjusted[0] == pytest.approx(0.04)  # 0.01 * 4
        assert adjusted[1] == pytest.approx(0.16)  # 0.04 * 4
        assert adjusted[2] == pytest.approx(0.24)  # 0.06 * 4
        assert threshold == pytest.approx(0.0125)  # 0.05 / 4

    def test_caps_at_one(self):
        adjusted, _ = bonferroni_correction(np.array([0.5, 0.8]), n_hypotheses=4)
        assert adjusted[0] == pytest.approx(1.0)
        assert adjusted[1] == pytest.approx(1.0)

    def test_single_hypothesis(self):
        adjusted, threshold = bonferroni_correction(np.array([0.03]), n_hypotheses=1, alpha=0.05)
        assert adjusted[0] == pytest.approx(0.03)
        assert threshold == pytest.approx(0.05)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="n_hypotheses must be positive"):
            bonferroni_correction(np.array([]), n_hypotheses=0)

    def test_invalid_pvalues_raise(self):
        with pytest.raises(ValueError, match="p-values must be in"):
            bonferroni_correction(np.array([-0.1, 0.5]))


class TestBenjaminiHochbergCorrection:
    def test_basic_correction(self):
        pvalues = np.array([0.01, 0.02, 0.05, 0.20, 0.30])
        adjusted, significant = benjamini_hochberg_correction(pvalues, n_hypotheses=5, alpha=0.05)
        assert len(adjusted) == 5
        # BH adjusted = p * n / rank, adjusted_sorted[0] = 0.01 * 5 / 1 = 0.05
        # adjusted_sorted[1] = 0.02 * 5 / 2 = 0.05
        # adjusted_sorted[2] = 0.05 * 5 / 3 = 0.0833
        assert adjusted[0] == pytest.approx(0.05)
        assert adjusted[1] == pytest.approx(0.05)
        assert significant[0]
        assert significant[1]
        assert not significant[2]  # 0.05 > 0.05 * 3/5 = 0.03

    def test_monotonicity(self):
        """BH-adjusted p-values should be non-decreasing."""
        pvalues = np.array([0.001, 0.01, 0.10, 0.20, 0.50])
        adjusted, _ = benjamini_hochberg_correction(pvalues, n_hypotheses=5, alpha=0.05)
        for i in range(len(adjusted) - 1):
            assert adjusted[i] <= adjusted[i + 1] + 1e-10

    def test_empty_returns_empty(self):
        adjusted, significant = benjamini_hochberg_correction(np.array([]), n_hypotheses=5)
        assert len(adjusted) == 0
        assert len(significant) == 0

    def test_invalid_pvalues_raise(self):
        with pytest.raises(ValueError, match="p-values must be in"):
            benjamini_hochberg_correction(np.array([0.5, 1.5]))


class TestApplyMultipleTestingCorrections:
    def test_adds_correction_columns(self):
        results = pd.DataFrame({
            "cointegration_pvalue": [0.01, 0.04, 0.50],
            "adf_pvalue": [0.02, 0.03, 0.60],
            "passes_filters": [True, True, False],
        })
        corrected = apply_multiple_testing_corrections(results, alpha=0.05)
        assert "cointegration_bonferroni_pvalue" in corrected.columns
        assert "cointegration_bh_pvalue" in corrected.columns
        assert "cointegration_significant_raw" in corrected.columns
        assert "cointegration_significant_bonferroni" in corrected.columns
        assert "cointegration_significant_bh" in corrected.columns
        assert "adf_bonferroni_pvalue" in corrected.columns
        assert "adf_bh_pvalue" in corrected.columns

    def test_empty_dataframe(self):
        empty = pd.DataFrame()
        result = apply_multiple_testing_corrections(empty)
        assert len(result) == 0

    def test_deterministic(self):
        results = pd.DataFrame({
            "cointegration_pvalue": [0.01, 0.05, 0.10],
            "adf_pvalue": [0.02, 0.06, 0.15],
        })
        r1 = apply_multiple_testing_corrections(results.copy(), alpha=0.05)
        r2 = apply_multiple_testing_corrections(results.copy(), alpha=0.05)
        pd.testing.assert_frame_equal(r1, r2)
