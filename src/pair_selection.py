"""Statistical tools for evaluating whether two price series form a suitable pairs-trading candidate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint


@dataclass
class PairAnalysisResult:
    """Container for the key statistics produced by pair analysis."""

    ticker_y: str
    ticker_x: str
    alpha: float
    hedge_ratio: float
    cointegration_statistic: float
    cointegration_pvalue: float
    adf_statistic: float
    adf_pvalue: float
    half_life: float | None
    n_observations: int


def align_price_series(
    y: pd.Series,
    x: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Align two price series by date and remove incomplete rows.

    The function preserves chronological order, removes rows where either series
    is missing or non-finite, verifies that both series have at least 60
    aligned observations, and checks that neither series is constant.
    """
    if not isinstance(y, pd.Series) or not isinstance(x, pd.Series):
        raise TypeError("Both inputs must be pandas Series objects.")

    y_series = pd.to_numeric(y, errors="coerce").astype(float)
    x_series = pd.to_numeric(x, errors="coerce").astype(float)

    aligned = pd.concat([y_series.rename("y"), x_series.rename("x")], axis=1)
    aligned = aligned.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    aligned = aligned.sort_index()

    if aligned.empty:
        raise ValueError("No overlapping, non-missing observations remain after alignment.")

    if len(aligned) < 60:
        raise ValueError("Fewer than 60 aligned observations are available.")

    if aligned["y"].nunique() <= 1 or aligned["x"].nunique() <= 1:
        raise ValueError("The aligned price series must not be constant.")

    return aligned["y"], aligned["x"]


def estimate_ols_regression(
    y: pd.Series,
    x: pd.Series,
) -> tuple[float, float, pd.Series]:
    """Estimate an OLS regression of the form $y_t = \alpha + \beta x_t + \epsilon_t$."""
    y_aligned, x_aligned = align_price_series(y, x)

    design_matrix = sm.add_constant(x_aligned.astype(float), has_constant="add")
    model = sm.OLS(y_aligned.astype(float), design_matrix)
    results = model.fit()

    alpha = float(results.params.iloc[0])
    hedge_ratio = float(results.params.iloc[1])
    spread = y_aligned - alpha - hedge_ratio * x_aligned

    return alpha, hedge_ratio, spread


def run_cointegration_test(
    y: pd.Series,
    x: pd.Series,
) -> tuple[float, float]:
    """Run the Engle-Granger cointegration test on two price series."""
    y_aligned, x_aligned = align_price_series(y, x)
    statistic, pvalue, _ = coint(y_aligned.astype(float), x_aligned.astype(float))
    return float(statistic), float(pvalue)


def run_adf_test(spread: pd.Series) -> tuple[float, float]:
    """Run the Augmented Dickey-Fuller test on the residual spread."""
    if not isinstance(spread, pd.Series):
        raise TypeError("Spread must be a pandas Series.")

    cleaned = spread.replace([np.inf, -np.inf], np.nan).dropna()

    if cleaned.empty:
        raise ValueError("The spread series is empty after removing invalid values.")

    statistic, pvalue, *_ = adfuller(cleaned.astype(float), autolag="AIC")
    return float(statistic), float(pvalue)


def estimate_half_life(spread: pd.Series) -> float | None:
    """Estimate the half-life of mean reversion for a spread series."""
    if not isinstance(spread, pd.Series):
        raise TypeError("Spread must be a pandas Series.")

    cleaned = spread.replace([np.inf, -np.inf], np.nan).dropna()
    if cleaned.empty:
        raise ValueError("The spread series is empty after removing invalid values.")

    delta_spread = cleaned.diff().dropna()
    spread_lag = cleaned.shift(1).dropna()
    regression_frame = pd.DataFrame({"delta": delta_spread, "lag": spread_lag})
    regression_frame = regression_frame.dropna()

    if regression_frame.empty:
        return None

    design_matrix = sm.add_constant(regression_frame["lag"], has_constant="add")
    model = sm.OLS(regression_frame["delta"], design_matrix)
    results = model.fit()
    slope = float(results.params.iloc[1])

    if not np.isfinite(slope) or slope >= 0:
        return None

    return float(-np.log(2.0) / slope)


def analyze_pair(
    price_df: pd.DataFrame,
    ticker_y: str,
    ticker_x: str,
) -> PairAnalysisResult:
    """Run a complete statistical analysis for a candidate equity pair.

    The function validates the input columns, aligns the two price series by
    date, estimates an OLS spread, runs cointegration and ADF tests, and
    estimates a half-life of mean reversion.
    """
    if not isinstance(price_df, pd.DataFrame):
        raise TypeError("price_df must be a pandas DataFrame.")

    if ticker_y == ticker_x:
        raise ValueError("Ticker names must be different.")

    missing_columns = [ticker for ticker in (ticker_y, ticker_x) if ticker not in price_df.columns]
    if missing_columns:
        raise ValueError(f"Missing ticker columns: {', '.join(missing_columns)}")

    y_series = price_df[ticker_y].astype(float)
    x_series = price_df[ticker_x].astype(float)
    y_aligned, x_aligned = align_price_series(y_series, x_series)

    alpha, hedge_ratio, spread = estimate_ols_regression(y_aligned, x_aligned)
    cointegration_statistic, cointegration_pvalue = run_cointegration_test(y_aligned, x_aligned)
    adf_statistic, adf_pvalue = run_adf_test(spread)
    half_life = estimate_half_life(spread)

    return PairAnalysisResult(
        ticker_y=ticker_y,
        ticker_x=ticker_x,
        alpha=alpha,
        hedge_ratio=hedge_ratio,
        cointegration_statistic=cointegration_statistic,
        cointegration_pvalue=cointegration_pvalue,
        adf_statistic=adf_statistic,
        adf_pvalue=adf_pvalue,
        half_life=half_life,
        n_observations=len(y_aligned),
    )
