"""Walk-forward validation for pairs-trading strategies.

This module implements chronological walk-forward validation to evaluate
whether selected pairs remain robust across multiple sequential out-of-sample
periods without using future information.

Window design
-------------
Each walk-forward window consists of a contiguous training period followed by a
contiguous test period:

   |--------- training ---------|---- test ----|
   t=0                       t=T           t=T+N

- **Rolling window**: The training set has a fixed size and slides forward by
  ``step_size_days`` each iteration.
- **Expanding window**: The training set starts at the earliest available date
  and grows with each iteration.

Anti-leakage rules
------------------
1. Pairs are screened and ranked independently in each training window.
2. The hedge ratio is estimated on training data and locked for the test period.
3. The rolling z-score inside the test window uses only information available
   up to each date.
4. Thresholds for screening are set once and never re-tuned.
5. All window results are saved, including windows with no qualifying pairs
   or negative performance.

Consistency score
-----------------
The consistency score summarises a pair's performance across all windows in
which it was selected.  Four equally-weighted components, each in [0, 1]:

  =================  ====================================================
  Component          Formula
  =================  ====================================================
  Return score       max(0, median_return) / max_median_return
  Sharpe score       max(0, median_sharpe) / max_median_sharpe
  Profitability      profitable_window_fraction
  Drawdown score     clip(1 + worst_drawdown, 0, 1)
  =================  ====================================================

Normalisation uses the maximum observed value across all pairs.
``consistency_score = 0.25 * (return_score + sharpe_score +
profitability + drawdown_score)``
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.backtester import BacktestParameters, run_backtest
from src.pair_screener import (
    PairScreeningParameters,
    screen_pairs,
)
from src.pair_selection import align_price_series, estimate_ols_regression
from src.risk_metrics import summarize_performance
from src.signals import SignalParameters, create_signal_frame


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WalkForwardParameters:
    """Configuration for walk-forward validation.

    Attributes
    ----------
    train_window_days:
        Number of calendar days in each training period.
    test_window_days:
        Number of calendar days in each test period.
    step_size_days:
        Number of calendar days to slide forward between windows.
    expanding_window:
        If True the training window starts at the earliest available date
        and grows.  If False the training window has a fixed size.
    minimum_train_observations:
        Minimum number of rows required in the training period.
    minimum_test_observations:
        Minimum number of rows required in the test period.
    top_n_pairs_per_window:
        Number of top-ranked pairs to evaluate in each window.
    """

    train_window_days: int
    test_window_days: int
    step_size_days: int
    expanding_window: bool
    minimum_train_observations: int
    minimum_test_observations: int
    top_n_pairs_per_window: int

    def __post_init__(self) -> None:
        if self.train_window_days <= 0:
            raise ValueError("train_window_days must be positive.")
        if self.test_window_days <= 0:
            raise ValueError("test_window_days must be positive.")
        if self.step_size_days <= 0:
            raise ValueError("step_size_days must be positive.")
        if self.minimum_train_observations <= 0:
            raise ValueError("minimum_train_observations must be positive.")
        if self.minimum_test_observations <= 0:
            raise ValueError("minimum_test_observations must be positive.")
        if self.top_n_pairs_per_window <= 0:
            raise ValueError("top_n_pairs_per_window must be positive.")


# ---------------------------------------------------------------------------
# Window generation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WalkForwardWindow:
    """Metadata for a single walk-forward window.

    Attributes
    ----------
    window_id:
        Zero-based sequential identifier for the window.
    train_start:
        Earliest date included in the training period.
    train_end:
        Latest date included in the training period.
    test_start:
        Earliest date included in the test period.
    test_end:
        Latest date included in the test period.
    train_observations:
        Number of trading days in the training period.
    test_observations:
        Number of trading days in the test period.
    """

    window_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_observations: int
    test_observations: int


def _find_window_boundaries(
    dates: pd.DatetimeIndex,
    *,
    train_window_days: int,
    test_window_days: int,
) -> list[tuple[int, int, int, int]]:
    """Return a list of (train_start_idx, train_end_idx, test_start_idx,
    test_end_idx) tuples that respect the chronological structure.

    Indices are integer positions into the DatetimeIndex.  Both train_end and
    test_end are inclusive.
    """
    n = len(dates)
    if n < 2:
        return []

    total_days_per_window = train_window_days + test_window_days
    if total_days_per_window > n:
        return []

    boundaries: list[tuple[int, int, int, int]] = []
    train_start = 0

    while True:
        train_end = train_start + train_window_days - 1
        test_start = train_end + 1
        test_end = test_start + test_window_days - 1

        if test_end >= n:
            break

        boundaries.append((train_start, train_end, test_start, test_end))
        train_start += test_window_days

    return boundaries


def _find_window_boundaries_rolling(
    dates: pd.DatetimeIndex,
    *,
    train_window_days: int,
    test_window_days: int,
    step_size_days: int,
) -> list[tuple[int, int, int, int]]:
    """Rolling window variant — the training window slides by step_size_days."""
    n = len(dates)
    if n < 2:
        return []

    boundaries: list[tuple[int, int, int, int]] = []
    train_start = 0

    while True:
        train_end = train_start + train_window_days - 1
        test_start = train_end + 1
        test_end = test_start + test_window_days - 1

        if test_end >= n:
            break

        boundaries.append((train_start, train_end, test_start, test_end))
        train_start += step_size_days

    return boundaries


def _find_window_boundaries_expanding(
    dates: pd.DatetimeIndex,
    *,
    train_window_days: int,
    test_window_days: int,
    step_size_days: int,
) -> list[tuple[int, int, int, int]]:
    """Expanding window variant — training always starts at index 0."""
    n = len(dates)
    if n < 2:
        return []

    boundaries: list[tuple[int, int, int, int]] = []
    min_train = train_window_days
    train_end = min_train - 1

    while True:
        train_end = train_end
        test_start = train_end + 1
        test_end = test_start + test_window_days - 1

        if test_end >= n:
            break

        boundaries.append((0, train_end, test_start, test_end))
        min_train += step_size_days
        train_end = min_train - 1

    return boundaries


def generate_walk_forward_windows(
    index: pd.DatetimeIndex,
    parameters: WalkForwardParameters,
) -> list[WalkForwardWindow]:
    """Generate chronological walk-forward windows from a DatetimeIndex.

    Parameters
    ----------
    index:
        Sorted DatetimeIndex used for positioning windows.
    parameters:
        Walk-forward configuration.

    Returns
    -------
    list[WalkForwardWindow]
        Chronologically ordered windows.  An empty list is returned if the
        index is too short to produce even one full window.

    Notes
    -----
    - The index must be sorted.
    - Test periods start the day after the training period ends (no gap).
    - Windows are non-overlapping in the train/test sense — no future
      information leaks into training.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("index must be a pandas DatetimeIndex.")

    if len(index) < 2:
        return []

    dates = index.sort_values()

    if parameters.expanding_window:
        raw = _find_window_boundaries_expanding(
            dates,
            train_window_days=parameters.train_window_days,
            test_window_days=parameters.test_window_days,
            step_size_days=parameters.step_size_days,
        )
    else:
        raw = _find_window_boundaries_rolling(
            dates,
            train_window_days=parameters.train_window_days,
            test_window_days=parameters.test_window_days,
            step_size_days=parameters.step_size_days,
        )

    windows: list[WalkForwardWindow] = []
    for wid, (ts, te, ss, se) in enumerate(raw):
        train_obs = te - ts + 1
        test_obs = se - ss + 1

        if train_obs < parameters.minimum_train_observations:
            continue
        if test_obs < parameters.minimum_test_observations:
            continue

        windows.append(
            WalkForwardWindow(
                window_id=wid,
                train_start=dates[ts],
                train_end=dates[te],
                test_start=dates[ss],
                test_end=dates[se],
                train_observations=train_obs,
                test_observations=test_obs,
            )
        )

    return windows


# ---------------------------------------------------------------------------
# Result columns
# ---------------------------------------------------------------------------

WALK_FORWARD_RESULT_COLUMNS: list[str] = [
    "window_id",
    "train_start",
    "train_end",
    "test_start",
    "test_end",
    "ticker_y",
    "ticker_x",
    "training_rank",
    "training_score",
    "training_cointegration_pvalue",
    "training_adf_pvalue",
    "training_half_life",
    "test_total_return",
    "test_annualized_return",
    "test_sharpe_ratio",
    "test_maximum_drawdown",
    "test_number_of_entries",
    "test_total_costs",
]

WALK_FORWARD_SUMMARY_COLUMNS: list[str] = [
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


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------


def _build_empty_window_record(
    window: WalkForwardWindow,
    ticker_y: str,
    ticker_x: str,
) -> dict[str, object]:
    """Return a single result dict with performance fields set to NaN."""
    return {
        "window_id": window.window_id,
        "train_start": window.train_start,
        "train_end": window.train_end,
        "test_start": window.test_start,
        "test_end": window.test_end,
        "ticker_y": ticker_y,
        "ticker_x": ticker_x,
        "training_rank": None,
        "training_score": None,
        "training_cointegration_pvalue": None,
        "training_adf_pvalue": None,
        "training_half_life": None,
        "test_total_return": np.nan,
        "test_annualized_return": np.nan,
        "test_sharpe_ratio": np.nan,
        "test_maximum_drawdown": np.nan,
        "test_number_of_entries": 0,
        "test_total_costs": 0.0,
    }


def _build_result_row(
    window: WalkForwardWindow,
    rank: int,
    training_row: pd.Series,
    test_y: pd.Series,
    test_x: pd.Series,
    hedge_ratio: float,
    signal_parameters: SignalParameters,
    backtest_parameters: BacktestParameters,
) -> dict[str, object]:
    """Run the test-period backtest for one pair and return a single result row."""
    test_spread = test_y.astype(float) - hedge_ratio * test_x.astype(float)
    signal_frame = create_signal_frame(test_spread, signal_parameters)
    backtest_frame = run_backtest(
        test_y, test_x, signal_frame["position"],
        hedge_ratio=hedge_ratio,
        parameters=backtest_parameters,
    )
    summary = summarize_performance(backtest_frame)

    return {
        "window_id": window.window_id,
        "train_start": window.train_start,
        "train_end": window.train_end,
        "test_start": window.test_start,
        "test_end": window.test_end,
        "ticker_y": training_row["ticker_y"],
        "ticker_x": training_row["ticker_x"],
        "training_rank": rank,
        "training_score": training_row.get("score"),
        "training_cointegration_pvalue": training_row.get("cointegration_pvalue"),
        "training_adf_pvalue": training_row.get("adf_pvalue"),
        "training_half_life": training_row.get("half_life"),
        "test_total_return": summary.total_return,
        "test_annualized_return": summary.annualized_return,
        "test_sharpe_ratio": summary.sharpe_ratio,
        "test_maximum_drawdown": summary.maximum_drawdown,
        "test_number_of_entries": int((signal_frame["entry_flag"] == True).sum()),
        "test_total_costs": float(
            backtest_frame["transaction_cost"].sum()
            + backtest_frame["slippage_cost"].sum()
        ),
    }


def _estimate_hedge_ratio(
    train_y: pd.Series,
    train_x: pd.Series,
) -> float:
    """Estimate hedge ratio using OLS on training data."""
    y_aligned, x_aligned = align_price_series(train_y, train_x)
    _, hedge_ratio, _ = estimate_ols_regression(y_aligned, x_aligned)
    return hedge_ratio


def run_walk_forward_analysis(
    prices: pd.DataFrame,
    screening_parameters: PairScreeningParameters,
    walk_forward_parameters: WalkForwardParameters,
    signal_parameters: SignalParameters,
    backtest_parameters: BacktestParameters,
) -> pd.DataFrame:
    """Run walk-forward validation across all generated windows.

    For each window the function:

    1. Screens and ranks pairs using **only** training-period prices.
    2. Selects the top ``top_n_pairs_per_window`` by training score.
    3. Estimates the hedge ratio on the training period.
    4. Keeps that hedge ratio fixed during the test period.
    5. Generates z-score signals on the test spread (backward-looking).
    6. Runs the existing backtest on the test period.
    7. Records all performance metrics.

    Pairs that pass screening in one window may differ from those in another
    window — selection is **per window**, which avoids selection bias.

    Parameters
    ----------
    prices:
        Full price history with a DatetimeIndex and ticker columns.
    screening_parameters:
        Pair-screening thresholds (fixed once, never re-tuned).
    walk_forward_parameters:
        Window geometry and pair count.
    signal_parameters:
        Z-score entry/exit thresholds (fixed).
    backtest_parameters:
        Capital, cost and slippage assumptions.

    Returns
    -------
    pd.DataFrame
        One row per pair per window.  Columns are defined in
        ``WALK_FORWARD_RESULT_COLUMNS``.  Returns an empty DataFrame with the
        correct column set when no windows or no pairs are available.
    """
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame.")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices must have a DatetimeIndex.")

    windows = generate_walk_forward_windows(prices.index, walk_forward_parameters)
    if not windows:
        return pd.DataFrame(columns=WALK_FORWARD_RESULT_COLUMNS)

    all_rows: list[dict[str, object]] = []

    for window in windows:
        train_prices = prices.loc[window.train_start : window.train_end]
        test_prices = prices.loc[window.test_start : window.test_end]

        # --- Screen and rank pairs using training data only -----------------
        screening_results = screen_pairs(train_prices, screening_parameters)

        if screening_results.empty:
            # Record a placeholder so no window is silently dropped.
            all_rows.append(
                _build_empty_window_record(window, "", "")
            )
            continue

        top_pairs = screening_results.head(
            walk_forward_parameters.top_n_pairs_per_window
        )

        # --- Evaluate each top pair on the test period ----------------------
        for rank, (_, training_row) in enumerate(top_pairs.iterrows(), start=1):
            ticker_y: str = training_row["ticker_y"]
            ticker_x: str = training_row["ticker_x"]

            if ticker_y not in test_prices.columns or ticker_x not in test_prices.columns:
                all_rows.append(
                    _build_empty_window_record(window, ticker_y, ticker_x)
                )
                continue

            test_y = test_prices[ticker_y].astype(float)
            test_x = test_prices[ticker_x].astype(float)

            # Realign on overlapping test data
            try:
                test_y_aligned, test_x_aligned = align_price_series(test_y, test_x)
            except (ValueError, TypeError):
                all_rows.append(
                    _build_empty_window_record(window, ticker_y, ticker_x)
                )
                continue

            # Estimate hedge ratio on training data (locked for test)
            train_y = train_prices[ticker_y].astype(float)
            train_x = train_prices[ticker_x].astype(float)
            try:
                hedge_ratio = _estimate_hedge_ratio(train_y, train_x)
            except (ValueError, TypeError):
                all_rows.append(
                    _build_empty_window_record(window, ticker_y, ticker_x)
                )
                continue

            if not np.isfinite(hedge_ratio):
                all_rows.append(
                    _build_empty_window_record(window, ticker_y, ticker_x)
                )
                continue

            try:
                row = _build_result_row(
                    window,
                    rank,
                    training_row,
                    test_y_aligned,
                    test_x_aligned,
                    hedge_ratio,
                    signal_parameters,
                    backtest_parameters,
                )
            except (ValueError, TypeError, ZeroDivisionError):
                all_rows.append(
                    _build_empty_window_record(window, ticker_y, ticker_x)
                )
                continue

            all_rows.append(row)

    return pd.DataFrame(all_rows, columns=WALK_FORWARD_RESULT_COLUMNS)


# ---------------------------------------------------------------------------
# Aggregate summaries
# ---------------------------------------------------------------------------


def _normalize_scores(values: pd.Series) -> pd.Series:
    """Min-max normalise a non-negative Series to [0, 1].

    Returns all zeros if the series is constant or zero.
    """
    mx = values.max()
    if mx <= 0:
        return pd.Series(0.0, index=values.index)
    return values / mx


def calculate_walk_forward_summary(
    detailed_results: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate per-window results into per-pair summary statistics.

    Parameters
    ----------
    detailed_results:
        Output of ``run_walk_forward_analysis`` — one row per pair per window.

    Returns
    -------
    pd.DataFrame
        One row per unique pair with aggregate metrics and a consistency score.
        Columns are defined in ``WALK_FORWARD_SUMMARY_COLUMNS``.
    """
    if not isinstance(detailed_results, pd.DataFrame):
        raise TypeError("detailed_results must be a pandas DataFrame.")
    if detailed_results.empty:
        return pd.DataFrame(columns=WALK_FORWARD_SUMMARY_COLUMNS)

    # Filter to rows that have actual performance (non-NaN test return)
    valid = detailed_results.dropna(subset=["test_total_return"]).copy()
    if valid.empty:
        return pd.DataFrame(columns=WALK_FORWARD_SUMMARY_COLUMNS)

    grouped = valid.groupby(["ticker_y", "ticker_x"], sort=False)

    records: list[dict[str, object]] = []
    for (ty, tx), grp in grouped:
        test_returns = grp["test_total_return"].astype(float)
        test_sharpes = grp["test_sharpe_ratio"].astype(float)
        drawdowns = grp["test_maximum_drawdown"].astype(float)

        records.append(
            {
                "ticker_y": ty,
                "ticker_x": tx,
                "n_windows": len(grp),
                "average_test_return": float(test_returns.mean()),
                "median_test_return": float(test_returns.median()),
                "average_test_sharpe_ratio": float(test_sharpes.mean()),
                "median_test_sharpe_ratio": float(test_sharpes.median()),
                "profitable_window_fraction": float((test_returns > 0).mean()),
                "worst_window_return": float(test_returns.min()),
                "worst_maximum_drawdown": float(drawdowns.min()),
                "total_number_of_entries": int(grp["test_number_of_entries"].sum()),
                "consistency_score": 0.0,  # placeholder
            }
        )

    summary = pd.DataFrame(records)

    # --- Consistency score --------------------------------------------------
    # Four equally-weighted components, each normalised to [0, 1].
    median_returns = summary["median_test_return"].clip(lower=0.0)
    median_sharpes = summary["median_test_sharpe_ratio"].clip(lower=0.0)

    return_score = _normalize_scores(median_returns)
    sharpe_score = _normalize_scores(median_sharpes)
    profitability = summary["profitable_window_fraction"]
    drawdown_score = (1.0 + summary["worst_maximum_drawdown"]).clip(lower=0.0, upper=1.0)

    summary["consistency_score"] = (
        0.25 * return_score
        + 0.25 * sharpe_score
        + 0.25 * profitability
        + 0.25 * drawdown_score
    )

    return summary.sort_values("consistency_score", ascending=False).reset_index(drop=True)
