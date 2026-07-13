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
        return pd.DataFrame(columns=[
            "ticker_y",
            "ticker_x",
            "alpha",
            "hedge_ratio",
            "cointegration_statistic",
            "cointegration_pvalue",
            "adf_statistic",
            "adf_pvalue",
            "half_life",
            "return_correlation",
            "n_observations",
            "passes_filters",
            "rejection_reason",
            "score",
            "rank",
            "train_period",
        ])

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
        return pd.DataFrame(columns=[
            "ticker_y",
            "ticker_x",
            "alpha",
            "hedge_ratio",
            "cointegration_statistic",
            "adf_statistic",
            "cointegration_pvalue",
            "adf_pvalue",
            "half_life",
            "return_correlation",
            "n_observations",
            "passes_filters",
            "rejection_reason",
            "score",
            "rank",
            "train_period",
        ])

    valid_results = results[results["passes_filters"]].copy()
    if valid_results.empty:
        return pd.DataFrame(columns=[
            "ticker_y",
            "ticker_x",
            "alpha",
            "hedge_ratio",
            "cointegration_statistic",
            "adf_statistic",
            "cointegration_pvalue",
            "adf_pvalue",
            "half_life",
            "return_correlation",
            "n_observations",
            "passes_filters",
            "rejection_reason",
            "score",
            "rank",
            "train_period",
        ])

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
