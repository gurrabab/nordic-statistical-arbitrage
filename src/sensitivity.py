"""Sensitivity analysis for pairs-trading strategies.

Provides functions for:

1. **Transaction-cost sensitivity**: Re-run the backtest for a range of cost
   assumptions using pre-computed signals (no re-estimation of pairs).
2. **Parameter sensitivity**: Grid search over signal parameters (lookback,
   entry, exit, stop thresholds) with clear train/validation/test labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.backtester import BacktestParameters, run_backtest
from src.risk_metrics import (
    summarize_performance,
)
from src.signals import SignalParameters, create_signal_frame
from src.validation import (
    estimate_train_test_relationship,
)

# ---------------------------------------------------------------------------
# Transaction-cost sensitivity
# ---------------------------------------------------------------------------

BASELINE_COST_BPS = 5.0
BASELINE_SLIPPAGE_BPS = 2.0
BASELINE_BORROW_COST = 0.02

COST_GRID_TRANSACTION_BPS = [0.0, 5.0, 10.0, 20.0, 30.0]
COST_GRID_SLIPPAGE_BPS = [0.0, 2.0, 5.0, 10.0]
COST_GRID_BORROW_COST = [0.0, 0.02, 0.05, 0.10]


@dataclass(frozen=True)
class CostSensitivityResult:
    """Result for one cost-scenario backtest."""

    transaction_cost_bps: float
    slippage_bps: float
    annual_borrow_cost: float
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    maximum_drawdown: float
    final_equity: float
    total_cost: float
    number_of_trades: int


def run_cost_sensitivity(
    price_y: pd.Series,
    price_x: pd.Series,
    signal_positions: pd.Series,
    hedge_ratio: float,
    initial_capital: float = 100_000.0,
    trading_days_per_year: int = 252,
    transaction_cost_values: list[float] | None = None,
    slippage_values: list[float] | None = None,
    borrow_cost_values: list[float] | None = None,
) -> pd.DataFrame:
    """Run a transaction-cost sensitivity analysis using fixed signals.

    The pair relationship and signals are **not** re-estimated — only the
    cost assumptions change between scenarios.  This isolates cost impact.

    Parameters
    ----------
    price_y, price_x:
        Constituent price series (test period).
    signal_positions:
        Pre-computed position series (from signal generation).
    hedge_ratio:
        Fixed hedge ratio estimated from training data.
    initial_capital:
        Starting capital for all scenarios.
    trading_days_per_year:
        Days per year for annualisation.
    transaction_cost_values:
        List of round-trip transaction costs in bps.
        Defaults to ``[0, 5, 10, 20, 30]``.
    slippage_values:
        List of slippage costs in bps.  Defaults to ``[0, 2, 5, 10]``.
    borrow_cost_values:
        List of annual short borrow costs as decimals.
        Defaults to ``[0.0, 0.02, 0.05, 0.10]``.

    Returns
    -------
    pd.DataFrame
        Columns: ``transaction_cost_bps``, ``slippage_bps``,
        ``annual_borrow_cost``, ``total_return``, ``annualized_return``,
        ``sharpe_ratio``, ``maximum_drawdown``, ``final_equity``,
        ``total_cost``, ``number_of_trades``.
    """
    if transaction_cost_values is None:
        transaction_cost_values = COST_GRID_TRANSACTION_BPS[:]
    if slippage_values is None:
        slippage_values = COST_GRID_SLIPPAGE_BPS[:]
    if borrow_cost_values is None:
        borrow_cost_values = COST_GRID_BORROW_COST[:]

    _validate_cost_values(transaction_cost_values, slippage_values, borrow_cost_values)

    results: list[CostSensitivityResult] = []

    for tc in transaction_cost_values:
        for sc in slippage_values:
            for bc in borrow_cost_values:
                params = BacktestParameters(
                    initial_capital=initial_capital,
                    transaction_cost_bps=tc,
                    slippage_bps=sc,
                    annual_borrow_cost=bc,
                    trading_days_per_year=trading_days_per_year,
                )
                bt_frame = run_backtest(
                    price_y, price_x, signal_positions,
                    hedge_ratio=hedge_ratio, parameters=params,
                )
                summary = summarize_performance(bt_frame, trading_days_per_year)
                total_cost = float(
                    bt_frame["transaction_cost"].sum()
                    + bt_frame["slippage_cost"].sum()
                    + bt_frame["borrow_cost"].sum()
                )

                results.append(
                    CostSensitivityResult(
                        transaction_cost_bps=tc,
                        slippage_bps=sc,
                        annual_borrow_cost=bc,
                        total_return=summary.total_return,
                        annualized_return=summary.annualized_return,
                        sharpe_ratio=summary.sharpe_ratio,
                        maximum_drawdown=summary.maximum_drawdown,
                        final_equity=float(bt_frame["equity"].iloc[-1]),
                        total_cost=total_cost,
                        number_of_trades=0,  # filled below
                    )
                )

    result_df = pd.DataFrame([vars(r) for r in results])

    # Baselines
    result_df["is_baseline"] = (
        (result_df["transaction_cost_bps"] == BASELINE_COST_BPS)
        & (result_df["slippage_bps"] == BASELINE_SLIPPAGE_BPS)
        & (result_df["annual_borrow_cost"] == BASELINE_BORROW_COST)
    )

    return result_df


def _validate_cost_values(
    tc: list[float], sc: list[float], bc: list[float],
) -> None:
    if not tc or not sc or not bc:
        raise ValueError("Cost value lists must not be empty.")
    for v in tc + sc:
        if v < 0:
            raise ValueError(f"Negative cost value: {v}")
    for v in bc:
        if v < 0:
            raise ValueError(f"Negative borrow cost: {v}")


# ---------------------------------------------------------------------------
# Parameter sensitivity
# ---------------------------------------------------------------------------

LOOKBACK_GRID = [20, 40, 60, 90, 120]
ENTRY_THRESHOLD_GRID = [1.5, 2.0, 2.5]
EXIT_THRESHOLD_GRID = [0.0, 0.5, 1.0]
STOP_THRESHOLD_GRID = [3.0, 3.5, 4.0]


def generate_parameter_grid(
    lookback_values: list[int] | None = None,
    entry_values: list[float] | None = None,
    exit_values: list[float] | None = None,
    stop_values: list[float] | None = None,
) -> pd.DataFrame:
    """Generate all valid signal-parameter combinations.

    Filters out combinations that violate:
    - ``entry_threshold > exit_threshold >= 0``
    - ``stop_threshold > entry_threshold``

    Returns
    -------
    pd.DataFrame
        Columns: ``lookback_window``, ``entry_threshold``,
        ``exit_threshold``, ``stop_threshold``.
    """
    if lookback_values is None:
        lookback_values = LOOKBACK_GRID[:]
    if entry_values is None:
        entry_values = ENTRY_THRESHOLD_GRID[:]
    if exit_values is None:
        exit_values = EXIT_THRESHOLD_GRID[:]
    if stop_values is None:
        stop_values = STOP_THRESHOLD_GRID[:]

    param_list: list[dict[str, float]] = []
    for lb in lookback_values:
        for entry in entry_values:
            for ext in exit_values:
                if not (entry > ext >= 0):
                    continue
                for stop in stop_values:
                    if not (stop > entry):
                        continue
                    param_list.append({
                        "lookback_window": lb,
                        "entry_threshold": entry,
                        "exit_threshold": ext,
                        "stop_threshold": stop,
                    })

    return pd.DataFrame(param_list)


def run_parameter_sensitivity(
    train_y: pd.Series,
    train_x: pd.Series,
    test_y: pd.Series,
    test_x: pd.Series,
    initial_capital: float = 100_000.0,
    trading_days_per_year: int = 252,
    lookback_values: list[int] | None = None,
    entry_values: list[float] | None = None,
    exit_values: list[float] | None = None,
    stop_values: list[float] | None = None,
) -> pd.DataFrame:
    """Run parameter sensitivity on **test data only** for comparison purposes.

    **Important anti-overfitting rule**:
    The results from this function should be used for **diagnosis only**.
    Do **not** select the best-performing parameter combination and report it
    as the main result.  The project's baseline parameters
    (lookback=60, entry=2.0, exit=0.5, stop=3.5) remain the primary result.

    The hedge ratio and alpha are estimated **once** on the training data
    and then held fixed.  Only the signal parameters vary.

    Parameters
    ----------
    train_y, train_x:
        Training price series (used only for hedge-ratio estimation).
    test_y, test_x:
        Test price series (used for signal generation and backtest).
    initial_capital:
        Starting capital.
    trading_days_per_year:
        Days per year for annualisation.
    lookback_values, entry_values, exit_values, stop_values:
        Parameter grids.  See module-level defaults.

    Returns
    -------
    pd.DataFrame
        Columns: ``data_segment``, ``lookback_window``, ``entry_threshold``,
        ``exit_threshold``, ``stop_threshold``, ``total_return``,
        ``annualized_return``, ``sharpe_ratio``, ``maximum_drawdown``,
        ``number_of_trades``, ``total_cost``.
    """
    grid = generate_parameter_grid(lookback_values, entry_values, exit_values, stop_values)

    # Estimate hedge ratio once on training data
    alpha, hedge_ratio, train_spread, test_spread, _ = estimate_train_test_relationship(
        train_y, train_x, test_y, test_x, lookback_window=60,
    )

    results: list[dict] = []
    for _, row in grid.iterrows():
        signal_params = SignalParameters(
            lookback_window=int(row["lookback_window"]),
            entry_threshold=row["entry_threshold"],
            exit_threshold=row["exit_threshold"],
            stop_threshold=row["stop_threshold"],
        )
        signal_frame = create_signal_frame(test_spread, signal_params)

        bt_params = BacktestParameters(
            initial_capital=initial_capital,
            transaction_cost_bps=5.0,
            slippage_bps=2.0,
            annual_borrow_cost=0.02,
            trading_days_per_year=trading_days_per_year,
        )
        bt_frame = run_backtest(
            test_y, test_x, signal_frame["position"],
            hedge_ratio=hedge_ratio, parameters=bt_params,
        )
        summary = summarize_performance(bt_frame, trading_days_per_year)
        total_cost = float(
            bt_frame["transaction_cost"].sum()
            + bt_frame["slippage_cost"].sum()
            + bt_frame["borrow_cost"].sum()
        )
        n_entries = int(signal_frame["entry_flag"].sum())

        results.append({
            "data_segment": "test",
            "lookback_window": int(row["lookback_window"]),
            "entry_threshold": row["entry_threshold"],
            "exit_threshold": row["exit_threshold"],
            "stop_threshold": row["stop_threshold"],
            "total_return": summary.total_return,
            "annualized_return": summary.annualized_return,
            "sharpe_ratio": summary.sharpe_ratio,
            "maximum_drawdown": summary.maximum_drawdown,
            "number_of_trades": n_entries,
            "total_cost": total_cost,
        })

    return pd.DataFrame(results)


def parameter_stability_summary(param_results: pd.DataFrame) -> dict:
    """Compute stability statistics from a parameter sensitivity DataFrame.

    Parameters
    ----------
    param_results:
        Output of ``run_parameter_sensitivity()``.

    Returns
    -------
    dict
        Keys: ``median_sharpe``, ``profitable_proportion``,
        ``sharpe_std``, ``sharpe_min``, ``sharpe_max``,
        ``n_combinations``, ``n_profitable``.
    """
    if param_results.empty:
        return {
            "median_sharpe": float("nan"),
            "profitable_proportion": float("nan"),
            "sharpe_std": float("nan"),
            "sharpe_min": float("nan"),
            "sharpe_max": float("nan"),
            "n_combinations": 0,
            "n_profitable": 0,
        }

    sharpes = param_results["sharpe_ratio"].astype(float)
    n_total = len(sharpes)
    n_profitable = int((sharpes > 0).sum())

    return {
        "median_sharpe": float(sharpes.median()),
        "profitable_proportion": n_profitable / n_total if n_total > 0 else float("nan"),
        "sharpe_std": float(sharpes.std(ddof=0)),
        "sharpe_min": float(sharpes.min()),
        "sharpe_max": float(sharpes.max()),
        "n_combinations": n_total,
        "n_profitable": n_profitable,
    }
