"""Utilities for screening a universe of Nordic equities for pairs trading candidates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint

from src.backtester import BacktestParameters, run_backtest
from src.data_loader import download_adjusted_close_prices
from src.pair_selection import align_price_series, estimate_ols_regression
from src.risk_metrics import summarize_performance
from src.signals import SignalParameters, create_signal_frame
from src.validation import estimate_train_test_relationship, split_aligned_prices


# ---------------------------------------------------------------------------
# Multiple-testing corrections
# ---------------------------------------------------------------------------

SCREENING_COLUMNS: list[str] = [
    "ticker_y",
    "ticker_x",
    "alpha",
    "hedge_ratio",
    "cointegration_statistic",
    "cointegration_pvalue",
    "cointegration_bonferroni_pvalue",
    "cointegration_bh_pvalue",
    "cointegration_significant_raw",
    "cointegration_significant_bonferroni",
    "cointegration_significant_bh",
    "adf_statistic",
    "adf_pvalue",
    "adf_bonferroni_pvalue",
    "adf_bh_pvalue",
    "adf_significant_raw",
    "adf_significant_bonferroni",
    "adf_significant_bh",
    "half_life",
    "return_correlation",
    "n_observations",
    "passes_filters",
    "rejection_reason",
    "score",
    "rank",
    "train_period",
]


def bonferroni_correction(
    pvalues: pd.Series | np.ndarray,
    n_hypotheses: int | None = None,
    alpha: float = 0.05,
) -> tuple[np.ndarray, float]:
    """Apply the Bonferroni correction for multiple hypothesis testing.

    Parameters
    ----------
    pvalues:
        Raw p-values.
    n_hypotheses:
        Number of hypotheses.  Defaults to ``len(pvalues)``.
    alpha:
        Family-wise significance level.

    Returns
    -------
    adjusted_pvalues:
        ``min(raw_pvalue * n_hypotheses, 1.0)``.
    adjusted_threshold:
        ``alpha / n_hypotheses``.
    """
    p = np.asarray(pvalues, dtype=float)
    if n_hypotheses is None:
        n_hypotheses = len(p)
    if n_hypotheses <= 0:
        raise ValueError("n_hypotheses must be positive.")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must be in [0, 1].")

    adjusted = np.minimum(p * n_hypotheses, 1.0)
    threshold = alpha / n_hypotheses
    return adjusted, threshold


def benjamini_hochberg_correction(
    pvalues: pd.Series | np.ndarray,
    n_hypotheses: int | None = None,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the Benjamini-Hochberg FDR correction for multiple hypothesis testing.

    Parameters
    ----------
    pvalues:
        Raw p-values.
    n_hypotheses:
        Number of hypotheses.  Defaults to ``len(pvalues)``.
    alpha:
        False discovery rate.

    Returns
    -------
    adjusted_pvalues:
        BH-adjusted p-values (monotone).
    significant:
        Boolean array indicating significance at the given FDR.
    """
    p = np.asarray(pvalues, dtype=float)
    n = n_hypotheses if n_hypotheses is not None else len(p)
    if n <= 0:
        raise ValueError("n_hypotheses must be positive.")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must be in [0, 1].")

    if len(p) == 0:
        return np.array([], dtype=float), np.array([], dtype=bool)

    # Sort p-values, compute BH thresholds, then unsort
    sorted_indices = np.argsort(p)
    sorted_p = p[sorted_indices]
    m = len(sorted_p)
    ranks = np.arange(1, m + 1)
    bh_thresholds = (ranks / n) * alpha
    # Adjusted p-values: cumulative min of (sorted_p * n / rank)
    adjusted_sorted = np.minimum(sorted_p * n / ranks, 1.0)
    # Ensure monotonicity
    for i in range(m - 2, -1, -1):
        adjusted_sorted[i] = min(adjusted_sorted[i], adjusted_sorted[i + 1])

    # Significant if sorted_p <= bh_thresholds (original BH)
    significant_sorted = sorted_p <= bh_thresholds

    # Unsort
    unsort_indices = np.argsort(sorted_indices)
    adjusted = adjusted_sorted[unsort_indices]
    significant = significant_sorted[unsort_indices]
    return adjusted, significant


def apply_multiple_testing_corrections(
    results: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Add multiple-testing correction columns to a pair-screening results DataFrame.

    Adds columns:
    - ``cointegration_bonferroni_pvalue``
    - ``cointegration_bh_pvalue``
    - ``cointegration_significant_raw``
    - ``cointegration_significant_bonferroni``
    - ``cointegration_significant_bh``
    - ``adf_bonferroni_pvalue``
    - ``adf_bh_pvalue``
    - ``adf_significant_raw``
    - ``adf_significant_bonferroni``
    - ``adf_significant_bh``
    """
    if results.empty:
        for col in [
            "cointegration_bonferroni_pvalue",
            "cointegration_bh_pvalue",
            "cointegration_significant_raw",
            "cointegration_significant_bonferroni",
            "cointegration_significant_bh",
            "adf_bonferroni_pvalue",
            "adf_bh_pvalue",
            "adf_significant_raw",
            "adf_significant_bonferroni",
            "adf_significant_bh",
        ]:
            results[col] = np.nan
        return results

    n_hypotheses = len(results)

    raw_coint = results["cointegration_pvalue"].values
    raw_adf = results["adf_pvalue"].values

    # Bonferroni
    bonf_coint, _ = bonferroni_correction(raw_coint, n_hypotheses, alpha)
    bonf_adf, _ = bonferroni_correction(raw_adf, n_hypotheses, alpha)

    # Benjamini-Hochberg
    bh_coint, sig_coint_bh = benjamini_hochberg_correction(raw_coint, n_hypotheses, alpha)
    bh_adf, sig_adf_bh = benjamini_hochberg_correction(raw_adf, n_hypotheses, alpha)

    results["cointegration_bonferroni_pvalue"] = bonf_coint
    results["cointegration_bh_pvalue"] = bh_coint
    results["cointegration_significant_raw"] = raw_coint <= alpha
    results["cointegration_significant_bonferroni"] = raw_coint <= (alpha / n_hypotheses)
    results["cointegration_significant_bh"] = sig_coint_bh

    results["adf_bonferroni_pvalue"] = bonf_adf
    results["adf_bh_pvalue"] = bh_adf
    results["adf_significant_raw"] = raw_adf <= alpha
    results["adf_significant_bonferroni"] = raw_adf <= (alpha / n_hypotheses)
    results["adf_significant_bh"] = sig_adf_bh

    return results


@dataclass(frozen=True)
class PairScreeningParameters:
    """Configuration for screening a universe of equities for pairs-trading candidates."""

    minimum_observations: int
    maximum_cointegration_pvalue: float
    maximum_adf_pvalue: float
    minimum_half_life: float
    maximum_half_life: float
    minimum_price: float
    maximum_missing_fraction: float
    top_n_pairs: int
    train_fraction: float
    alpha: float = 0.05  # significance level for multiple-testing corrections

    def __post_init__(self) -> None:
        if self.minimum_observations <= 0:
            raise ValueError("minimum_observations must be positive.")
        if not 0 <= self.maximum_cointegration_pvalue <= 1:
            raise ValueError("maximum_cointegration_pvalue must be between 0 and 1.")
        if not 0 <= self.maximum_adf_pvalue <= 1:
            raise ValueError("maximum_adf_pvalue must be between 0 and 1.")
        if self.minimum_half_life <= 0 or self.maximum_half_life <= 0:
            raise ValueError("half-life thresholds must be positive.")
        if self.minimum_price <= 0:
            raise ValueError("minimum_price must be positive.")
        if not 0 <= self.maximum_missing_fraction < 1:
            raise ValueError("maximum_missing_fraction must be between 0 and 1.")
        if self.top_n_pairs <= 0:
            raise ValueError("top_n_pairs must be positive.")
        if not 0 < self.train_fraction < 1:
            raise ValueError("train_fraction must be between 0 and 1.")


def download_ticker_universe(
    tickers: list[str],
    *,
    start: str | None = None,
    end: str | None = None,
    period: str | None = None,
    downloader: callable | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download price data for a universe of tickers and return per-ticker success info."""
    if not tickers:
        raise ValueError("tickers must not be empty.")

    data_loader = downloader or download_adjusted_close_prices
    report = pd.DataFrame(index=tickers, columns=["status", "reason"], dtype=object)
    report["status"] = "pending"
    report["reason"] = ""

    downloaded_frames: list[pd.DataFrame] = []
    for ticker in tickers:
        try:
            prices = data_loader([ticker], start=start, end=end, period=period)
        except Exception as exc:  # pragma: no cover - exercised through tests
            report.at[ticker, "status"] = "failed"
            report.at[ticker, "reason"] = str(exc)
            continue

        if not isinstance(prices, pd.DataFrame) or prices.empty:
            report.at[ticker, "status"] = "failed"
            report.at[ticker, "reason"] = "empty download"
            continue

        frame = prices.copy()
        if len(frame.columns) != 1:
            report.at[ticker, "status"] = "failed"
            report.at[ticker, "reason"] = "expected one price column"
            continue

        frame.columns = [ticker]
        downloaded_frames.append(frame)
        report.at[ticker, "status"] = "downloaded"
        report.at[ticker, "reason"] = ""

    if not downloaded_frames:
        return pd.DataFrame(index=pd.DatetimeIndex([])), report

    return pd.concat(downloaded_frames, axis=1).sort_index(), report


def filter_ticker_universe(
    prices: pd.DataFrame,
    parameters: PairScreeningParameters,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter a price frame to include only tickers that pass basic quality checks."""
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame.")
    if not isinstance(parameters, PairScreeningParameters):
        raise TypeError("parameters must be a PairScreeningParameters instance.")

    base_frame = prices.copy()
    base_frame = base_frame.replace([np.inf, -np.inf], np.nan)
    reasons = pd.DataFrame(index=base_frame.columns, columns=["kept", "reason"], dtype=object)
    reasons["kept"] = False
    reasons["reason"] = ""

    filtered_columns: list[str] = []
    for ticker in base_frame.columns:
        series = pd.to_numeric(base_frame[ticker], errors="coerce").astype(float)
        if len(series.dropna()) < parameters.minimum_observations:
            reasons.at[ticker, "reason"] = "insufficient observations"
            continue
        if np.isnan(series).mean() > parameters.maximum_missing_fraction:
            reasons.at[ticker, "reason"] = "excessive missing data"
            continue
        if not np.isfinite(series).all() and series.dropna().shape[0] < parameters.minimum_observations:
            reasons.at[ticker, "reason"] = "non-finite values"
            continue
        if (series <= 0).any():
            reasons.at[ticker, "reason"] = "non-positive prices"
            continue
        if series.dropna().nunique() <= 1:
            reasons.at[ticker, "reason"] = "constant prices"
            continue
        if series.dropna().min() < parameters.minimum_price:
            reasons.at[ticker, "reason"] = "price below minimum"
            continue
        filtered_columns.append(ticker)
        reasons.at[ticker, "kept"] = True

    filtered_prices = base_frame.loc[:, filtered_columns].copy()
    return filtered_prices, reasons


def generate_unique_pairs(tickers: list[str] | tuple[str, ...] | pd.Index) -> list[tuple[str, str]]:
    """Generate all unique unordered ticker pairs for a ticker list."""
    ticker_list = list(tickers)
    if len(ticker_list) != len(set(ticker_list)):
        raise ValueError("Ticker list must not contain duplicates.")
    if len(ticker_list) < 2:
        return []

    pairs: list[tuple[str, str]] = []
    for i, ticker_y in enumerate(ticker_list):
        for ticker_x in ticker_list[i + 1 :]:
            pairs.append((ticker_y, ticker_x))
    return pairs


def _assess_pair(
    y_series: pd.Series,
    x_series: pd.Series,
    parameters: PairScreeningParameters,
) -> dict[str, object]:
    y_aligned, x_aligned = align_price_series(y_series, x_series)
    alpha, hedge_ratio, spread = estimate_ols_regression(y_aligned, x_aligned)

    cointegration_result = coint(y_aligned.astype(float), x_aligned.astype(float))
    cointegration_statistic = float(cointegration_result[0])
    cointegration_pvalue = float(cointegration_result[1])
    adf_statistic, adf_pvalue, *_ = adfuller(spread.astype(float), autolag="AIC")
    half_life = None
    if np.isfinite(hedge_ratio):
        delta_spread = spread.diff().dropna()
        lagged_spread = spread.shift(1).dropna()
        regression_frame = pd.DataFrame({"delta": delta_spread, "lag": lagged_spread})
        regression_frame = regression_frame.dropna()
        if not regression_frame.empty:
            design_matrix = sm.add_constant(regression_frame["lag"], has_constant="add")
            model = sm.OLS(regression_frame["delta"], design_matrix)
            results = model.fit()
            slope = float(results.params.iloc[1])
            if np.isfinite(slope) and slope < 0:
                half_life = float(-np.log(2.0) / slope)

    if len(y_aligned) > 2:
        y_returns = y_aligned.astype(float).pct_change().dropna()
        x_returns = x_aligned.astype(float).pct_change().dropna()
        aligned_returns = pd.concat([y_returns.rename("y"), x_returns.rename("x")], axis=1).dropna()
        if len(aligned_returns) > 2 and aligned_returns["y"].nunique() > 1 and aligned_returns["x"].nunique() > 1:
            return_correlation = float(np.corrcoef(aligned_returns["y"], aligned_returns["x"])[0, 1])
        else:
            return_correlation = np.nan
    else:
        return_correlation = np.nan
    return {
        "alpha": alpha,
        "hedge_ratio": hedge_ratio,
        "cointegration_statistic": float(cointegration_statistic),
        "cointegration_pvalue": float(cointegration_pvalue),
        "adf_statistic": float(adf_statistic),
        "adf_pvalue": float(adf_pvalue),
        "half_life": half_life,
        "return_correlation": return_correlation,
        "n_observations": int(len(y_aligned)),
        "passes_filters": False,
        "rejection_reason": "",
    }


def screen_pairs(
    prices: pd.DataFrame,
    parameters: PairScreeningParameters,
) -> pd.DataFrame:
    """Screen a price frame for valid pairs using only training-period statistics."""
    filtered_prices, _ = filter_ticker_universe(prices, parameters)
    if filtered_prices.empty or len(filtered_prices.columns) < 2:
        return pd.DataFrame(columns=SCREENING_COLUMNS)

    pairs = generate_unique_pairs(filtered_prices.columns)

    results_rows = []
    for ticker_y, ticker_x in pairs:
        pair_result = _assess_pair(filtered_prices[ticker_y], filtered_prices[ticker_x], parameters)
        pair_result["ticker_y"] = ticker_y
        pair_result["ticker_x"] = ticker_x
        pair_result["passes_filters"] = (
            pair_result["n_observations"] >= parameters.minimum_observations
            and pair_result["cointegration_pvalue"] <= parameters.maximum_cointegration_pvalue
            and pair_result["adf_pvalue"] <= parameters.maximum_adf_pvalue
            and pair_result["half_life"] is not None
            and parameters.minimum_half_life <= pair_result["half_life"] <= parameters.maximum_half_life
            and np.isfinite(pair_result["hedge_ratio"])
        )
        if not pair_result["passes_filters"]:
            reasons: list[str] = []
            if pair_result["n_observations"] < parameters.minimum_observations:
                reasons.append("insufficient observations")
            if pair_result["cointegration_pvalue"] > parameters.maximum_cointegration_pvalue:
                reasons.append("cointegration p-value")
            if pair_result["adf_pvalue"] > parameters.maximum_adf_pvalue:
                reasons.append("ADF p-value")
            if pair_result["half_life"] is None or not (parameters.minimum_half_life <= pair_result["half_life"] <= parameters.maximum_half_life):
                reasons.append("half-life")
            if not np.isfinite(pair_result["hedge_ratio"]):
                reasons.append("non-finite hedge ratio")
            pair_result["rejection_reason"] = "; ".join(reasons)
        else:
            pair_result["rejection_reason"] = ""
        results_rows.append(pair_result)

    results = pd.DataFrame(results_rows)
    if results.empty:
        return pd.DataFrame(columns=SCREENING_COLUMNS)

    # Apply multiple-testing corrections to ALL pairs (not just filtered)
    results = apply_multiple_testing_corrections(results, alpha=parameters.alpha)

    valid_results = results[results["passes_filters"]].copy()
    if valid_results.empty:
        empty_out = pd.DataFrame(columns=SCREENING_COLUMNS)
        # Preserve correction columns from full results if any
        if not results.empty:
            empty_out = results.iloc[0:0]
        return empty_out

    valid_results["score"] = (
        -2.0 * np.log10(valid_results["cointegration_pvalue"].replace(0, np.nan) + 1e-12)
        - 1.0 * np.log10(valid_results["adf_pvalue"].replace(0, np.nan) + 1e-12)
        + 0.01 * np.minimum(valid_results["half_life"], 100.0)
        + 0.001 * valid_results["n_observations"]
    )
    valid_results = valid_results.sort_values(["score"], ascending=False)
    valid_results = valid_results.reset_index(drop=True)
    valid_results.insert(0, "rank", range(1, len(valid_results) + 1))
    valid_results["train_period"] = "training"
    return valid_results


def evaluate_top_pairs(
    screening_results: pd.DataFrame,
    prices: pd.DataFrame,
    parameters: PairScreeningParameters,
    *,
    top_n: int | None = None,
) -> pd.DataFrame:
    """Evaluate the top-ranked pairs out-of-sample using fixed training parameters."""
    if not isinstance(screening_results, pd.DataFrame):
        raise TypeError("screening_results must be a pandas DataFrame.")
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame.")
    if not isinstance(parameters, PairScreeningParameters):
        raise TypeError("parameters must be a PairScreeningParameters instance.")

    if screening_results.empty:
        return pd.DataFrame(columns=[
            "rank",
            "ticker_y",
            "ticker_x",
            "training_cointegration_pvalue",
            "training_adf_pvalue",
            "training_half_life",
            "training_score",
            "training_hedge_ratio",
            "test_total_return",
            "test_annualized_return",
            "test_sharpe_ratio",
            "test_maximum_drawdown",
            "test_number_of_entries",
            "test_total_costs",
        ])

    selected_results = screening_results.head(top_n or parameters.top_n_pairs).copy()
    selected_results = selected_results.reset_index(drop=True)

    comparison_rows = []
    for idx, row in selected_results.iterrows():
        ticker_y = row["ticker_y"]
        ticker_x = row["ticker_x"]
        y_series = prices[ticker_y].astype(float)
        x_series = prices[ticker_x].astype(float)
        y_aligned, x_aligned = align_price_series(y_series, x_series)
        train_frame, test_frame, _ = split_aligned_prices(
            pd.concat([y_aligned.rename(ticker_y), x_aligned.rename(ticker_x)], axis=1),
            train_ratio=parameters.train_fraction,
            min_train_observations=max(60, parameters.minimum_observations // 2),
            min_test_observations=max(30, parameters.minimum_observations // 4),
        )
        train_y = train_frame[ticker_y]
        train_x = train_frame[ticker_x]
        test_y = test_frame[ticker_y]
        test_x = test_frame[ticker_x]
        alpha, hedge_ratio, train_spread, test_spread, test_zscore = estimate_train_test_relationship(
            train_y,
            train_x,
            test_y,
            test_x,
            lookback_window=60,
        )
        signal_parameters = SignalParameters(
            lookback_window=60,
            entry_threshold=2.0,
            exit_threshold=0.5,
            stop_threshold=3.5,
        )
        test_signal_frame = create_signal_frame(test_spread, signal_parameters)
        backtest_parameters = BacktestParameters(
            initial_capital=100000.0,
            transaction_cost_bps=5.0,
            slippage_bps=2.0,
            annual_borrow_cost=0.02,
        )
        backtest_frame = run_backtest(
            test_y,
            test_x,
            test_signal_frame["position"],
            hedge_ratio=hedge_ratio,
            parameters=backtest_parameters,
        )
        summary = summarize_performance(backtest_frame)
        comparison_rows.append(
            {
                "rank": idx + 1,
                "ticker_y": ticker_y,
                "ticker_x": ticker_x,
                "training_cointegration_pvalue": row["cointegration_pvalue"],
                "training_adf_pvalue": row["adf_pvalue"],
                "training_half_life": row["half_life"],
                "training_score": row["score"],
                "training_hedge_ratio": hedge_ratio,
                "test_total_return": summary.total_return,
                "test_annualized_return": summary.annualized_return,
                "test_sharpe_ratio": summary.sharpe_ratio,
                "test_maximum_drawdown": summary.maximum_drawdown,
                "test_number_of_entries": int((test_signal_frame["entry_flag"] == True).sum()),
                "test_total_costs": float(backtest_frame["transaction_cost"].sum() + backtest_frame["slippage_cost"].sum()),
            }
        )

    return pd.DataFrame(comparison_rows)
