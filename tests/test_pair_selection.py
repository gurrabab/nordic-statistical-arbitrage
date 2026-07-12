import numpy as np
import pandas as pd
import pytest

from src.pair_selection import (
    PairAnalysisResult,
    align_price_series,
    analyze_pair,
    estimate_ols_regression,
    estimate_half_life,
    run_adf_test,
    run_cointegration_test,
)


def _create_cointegrated_pair(seed: int = 7) -> tuple[pd.DataFrame, float]:
    rng = np.random.default_rng(seed)
    n = 300
    x = np.cumsum(rng.normal(loc=0.0, scale=1.0, size=n))
    noise = rng.normal(loc=0.0, scale=0.1, size=n)
    y = 1.5 + 0.8 * x + noise
    price_df = pd.DataFrame({"y": y, "x": x}, index=pd.date_range("2000-01-01", periods=n, freq="D"))
    return price_df, 0.8


def test_cointegrated_pair_estimates_hedge_ratio_and_low_pvalues() -> None:
    price_df, true_beta = _create_cointegrated_pair(seed=11)

    result = analyze_pair(price_df, ticker_y="y", ticker_x="x")

    assert isinstance(result, PairAnalysisResult)
    assert result.n_observations >= 60
    assert abs(result.hedge_ratio - true_beta) < 0.2
    assert result.cointegration_pvalue < 0.05
    assert result.adf_pvalue < 0.05


def test_independent_random_walks_are_less_likely_to_be_cointegrated() -> None:
    rng = np.random.default_rng(5)
    n = 300
    x = np.cumsum(rng.normal(size=n))
    y = np.cumsum(rng.normal(size=n))
    price_df = pd.DataFrame({"y": y, "x": x}, index=pd.date_range("2000-01-01", periods=n, freq="D"))

    result = analyze_pair(price_df, ticker_y="y", ticker_x="x")

    assert result.cointegration_pvalue > 0.05


def test_align_price_series_handles_mismatched_dates() -> None:
    index_y = pd.date_range("2020-01-01", periods=80, freq="D")
    index_x = pd.date_range("2020-01-02", periods=80, freq="D")
    y = pd.Series(np.linspace(1.0, 2.0, 80), index=index_y)
    x = pd.Series(np.linspace(1.0, 2.0, 80), index=index_x)

    aligned_y, aligned_x = align_price_series(y, x)

    assert len(aligned_y) == 79
    assert len(aligned_x) == 79
    assert aligned_y.index.equals(aligned_x.index)


def test_missing_values_are_dropped_before_analysis() -> None:
    index = pd.date_range("2020-01-01", periods=80, freq="D")
    y = pd.Series(np.linspace(1.0, 2.0, 80), index=index)
    x = pd.Series(np.linspace(1.0, 2.0, 80), index=index)
    x.iloc[10] = np.nan
    x.iloc[11] = np.nan

    aligned_y, aligned_x = align_price_series(y, x)

    assert len(aligned_y) < 80
    assert len(aligned_y) == len(aligned_x)


def test_constant_series_raise_value_error() -> None:
    y = pd.Series([1.0] * 70, index=pd.date_range("2020-01-01", periods=70, freq="D"))
    x = pd.Series([2.0] * 70, index=y.index)

    with pytest.raises(ValueError, match="constant"):
        align_price_series(y, x)


def test_too_few_observations_raise_value_error() -> None:
    index = pd.date_range("2020-01-01", periods=50, freq="D")
    y = pd.Series(np.linspace(1.0, 2.0, 50), index=index)
    x = pd.Series(np.linspace(1.0, 2.0, 50), index=index)

    with pytest.raises(ValueError, match="60"):
        align_price_series(y, x)


def test_identical_ticker_names_raise_value_error() -> None:
    price_df = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [2.0, 3.0, 4.0]})

    with pytest.raises(ValueError, match="different"):
        analyze_pair(price_df, ticker_y="A", ticker_x="A")


def test_estimate_ols_regression_returns_spread() -> None:
    index = pd.date_range("2020-01-01", periods=70, freq="D")
    y = pd.Series(np.linspace(1.0, 2.0, 70), index=index)
    x = pd.Series(np.linspace(1.0, 2.0, 70), index=index)

    alpha, beta, spread = estimate_ols_regression(y, x)

    assert alpha == pytest.approx(0.0, abs=1e-8)
    assert beta == pytest.approx(1.0, abs=1e-8)
    assert spread.iloc[0] == pytest.approx(0.0, abs=1e-8)


def test_half_life_returns_none_for_non_mean_reverting_series() -> None:
    spread = pd.Series(np.linspace(1.0, 2.0, 70), index=pd.date_range("2020-01-01", periods=70, freq="D"))

    assert estimate_half_life(spread) is None


def test_adf_and_cointegration_functions_return_numeric_outputs() -> None:
    rng = np.random.default_rng(123)
    index = pd.date_range("2020-01-01", periods=80, freq="D")
    x = pd.Series(np.cumsum(rng.normal(size=80)), index=index)
    noise = rng.normal(scale=0.2, size=80)
    y = pd.Series(0.5 + 0.7 * x + noise, index=index)

    statistic, pvalue = run_cointegration_test(y, x)
    adf_statistic, adf_pvalue = run_adf_test(pd.Series(np.linspace(0.1, 0.6, 80), index=index))

    assert np.isfinite(statistic)
    assert np.isfinite(pvalue)
    assert np.isfinite(adf_statistic)
    assert np.isfinite(adf_pvalue)
