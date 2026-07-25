"""Data quality assessment for Nordic equity data used in pairs trading.

Provides per-ticker quality reports and pair-overlap analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

DEFAULT_SUSPICIOUS_RETURN_THRESHOLD: float = 0.20  # 20% daily move


@dataclass(frozen=True)
class TickerQualityReport:
    """Quality report for a single ticker."""

    ticker: str
    first_date: pd.Timestamp | None
    last_date: pd.Timestamp | None
    total_expected_rows: int
    actual_rows: int
    missing_rows: int
    missing_fraction: float
    duplicate_dates: int
    non_positive_prices: int
    constant_price_flag: bool
    largest_abs_daily_return: float
    suspicious_return_count: int
    passed_quality_filter: bool
    rejection_reason: str


def assess_ticker_quality(
    prices: pd.Series,
    ticker: str,
    expected_date_range: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    suspicious_return_threshold: float = DEFAULT_SUSPICIOUS_RETURN_THRESHOLD,
    max_missing_fraction: float = 0.10,
) -> TickerQualityReport:
    """Assess the quality of a single ticker's price series.

    Parameters
    ----------
    prices:
        Price series (daily).
    ticker:
        Ticker name.
    expected_date_range:
        Optional (start, end) tuple for the expected full date range.
        If None, uses the actual date range of ``prices``.
    suspicious_return_threshold:
        Daily return magnitude above which an observation is flagged.
    max_missing_fraction:
        Maximum tolerable fraction of missing observations.

    Returns
    -------
    TickerQualityReport
    """
    if not isinstance(prices, pd.Series):
        raise TypeError("prices must be a pandas Series.")
    if not prices.index.is_monotonic_increasing:
        prices = prices.sort_index()

    first_date = prices.index[0] if len(prices) > 0 else None
    last_date = prices.index[-1] if len(prices) > 0 else None

    if expected_date_range is not None:
        exp_start, exp_end = expected_date_range
        expected_business_days = pd.bdate_range(exp_start, exp_end)
        total_expected = len(expected_business_days)
    else:
        total_expected = len(prices)

    actual = int(prices.count())
    missing = total_expected - actual
    missing_frac = missing / total_expected if total_expected > 0 else 0.0

    # Duplicate dates
    dupes = int(prices.index.duplicated().sum())

    # Non-positive prices
    non_pos = int((prices.dropna() <= 0).sum())

    # Constant price flag
    const_flag = bool(prices.dropna().nunique() <= 1) if len(prices) > 0 else False

    # Extreme returns
    daily_ret = prices.dropna().pct_change().dropna()
    largest_abs_ret = float(daily_ret.abs().max()) if len(daily_ret) > 0 else 0.0
    susp_count = int((daily_ret.abs() > suspicious_return_threshold).sum())

    # Determine pass/fail
    reasons: list[str] = []
    if actual == 0:
        reasons.append("empty series")
    if missing_frac > max_missing_fraction:
        reasons.append(f"missing ({missing_frac:.1%})")
    if non_pos > 0:
        reasons.append(f"{non_pos} non-positive price(s)")
    if const_flag:
        reasons.append("constant price")
    if susp_count > 0:
        reasons.append(f"{susp_count} suspicious return(s)")

    passed = len(reasons) == 0 and actual > 0

    return TickerQualityReport(
        ticker=ticker,
        first_date=first_date,
        last_date=last_date,
        total_expected_rows=total_expected,
        actual_rows=actual,
        missing_rows=missing,
        missing_fraction=missing_frac,
        duplicate_dates=dupes,
        non_positive_prices=non_pos,
        constant_price_flag=const_flag,
        largest_abs_daily_return=largest_abs_ret,
        suspicious_return_count=susp_count,
        passed_quality_filter=passed,
        rejection_reason="; ".join(reasons) if reasons else "",
    )


def assess_all_tickers(
    prices: pd.DataFrame,
    suspicious_return_threshold: float = DEFAULT_SUSPICIOUS_RETURN_THRESHOLD,
    max_missing_fraction: float = 0.10,
) -> pd.DataFrame:
    """Assess quality for every ticker in a price DataFrame.

    Parameters
    ----------
    prices:
        DataFrame with ticker columns.
    suspicious_return_threshold:
        Daily return threshold for flagging.
    max_missing_fraction:
        Maximum tolerable missing observation fraction.

    Returns
    -------
    pd.DataFrame
        One row per ticker with all ``TickerQualityReport`` fields.
    """
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame.")

    expected_range = (prices.index.min(), prices.index.max()) if len(prices) > 0 else None

    reports: list[TickerQualityReport] = []
    for ticker in prices.columns:
        report = assess_ticker_quality(
            prices[ticker],
            ticker=ticker,
            expected_date_range=expected_range,
            suspicious_return_threshold=suspicious_return_threshold,
            max_missing_fraction=max_missing_fraction,
        )
        reports.append(report)

    df = pd.DataFrame([vars(r) for r in reports])
    return df


def compute_pair_overlap(
    prices: pd.DataFrame,
    ticker_y: str,
    ticker_x: str,
) -> dict:
    """Compute the data overlap between two tickers.

    Parameters
    ----------
    prices:
        DataFrame with ticker columns.
    ticker_y, ticker_x:
        Constituent tickers.

    Returns
    -------
    dict
        Keys: ``ticker_y``, ``ticker_x``, ``overlapping_observations``,
        ``overlap_start``, ``overlap_end``, ``overlap_fraction``.
    """
    if ticker_y not in prices.columns or ticker_x not in prices.columns:
        raise ValueError(f"Tickers {ticker_y} or {ticker_x} not found.")

    y_valid = prices[ticker_y].dropna()
    x_valid = prices[ticker_x].dropna()
    overlap = y_valid.index.intersection(x_valid.index)

    n_overlap = len(overlap)
    total = max(len(y_valid), len(x_valid))
    fraction = n_overlap / total if total > 0 else 0.0

    return {
        "ticker_y": ticker_y,
        "ticker_x": ticker_x,
        "overlapping_observations": n_overlap,
        "overlap_start": overlap.min() if n_overlap > 0 else None,
        "overlap_end": overlap.max() if n_overlap > 0 else None,
        "overlap_fraction": fraction,
    }


def compute_all_pair_overlaps(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute pair overlap reports for all unique ticker pairs."""
    pairs: list[dict] = []
    tickers = list(prices.columns)
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            pairs.append(compute_pair_overlap(prices, tickers[i], tickers[j]))
    return pd.DataFrame(pairs)
