"""Trade-level analysis for pairs-trading backtests.

Converts a daily backtest DataFrame into one row per completed trade,
calculates trade-level metrics, and produces a trade summary.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class Direction(Enum):
    """Trade direction relative to the spread."""

    LONG_SPREAD = "long_spread"
    SHORT_SPREAD = "short_spread"


class ExitReason(Enum):
    """Reason a trade was closed."""

    NORMAL_EXIT = "normal_exit"
    STOP_EXIT = "stop_exit"
    END_OF_TEST_PERIOD = "end_of_test_period"


@dataclass
class TradeRecord:
    """One completed (or final incomplete) trade.

    Attributes
    ----------
    pair: ``"{ticker_y}/{ticker_x}"``
    ticker_y, ticker_x: Constituent tickers.
    direction: Whether the trade went long or short the spread.
    entry_date: Date the position was first established.
    exit_date: Date the position was closed (or last data date).
    entry_zscore: Z-score at entry.
    exit_zscore: Z-score at exit.
    entry_equity: Account equity at entry.
    exit_equity: Account equity at exit.
    holding_days: Number of trading days the trade was open.
    gross_return: Gross return over the holding period.
    transaction_cost: Total transaction costs incurred.
    slippage_cost: Total slippage costs incurred.
    borrow_cost: Total borrow costs incurred.
    total_cost: Sum of all costs.
    net_return: Net return (gross - costs).
    maximum_adverse_excursion: Worst percentage return from entry point
        while the trade was open (non-positive).
    maximum_favorable_excursion: Best percentage return from entry point
        while the trade was open (non-negative).
    exit_reason: Why the trade was closed.
    """

    pair: str
    ticker_y: str
    ticker_x: str
    direction: Direction
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_zscore: float
    exit_zscore: float
    entry_equity: float
    exit_equity: float
    holding_days: int
    gross_return: float
    transaction_cost: float
    slippage_cost: float
    borrow_cost: float
    total_cost: float
    net_return: float
    maximum_adverse_excursion: float
    maximum_favorable_excursion: float
    exit_reason: ExitReason


@dataclass
class TradeSummary:
    """Aggregate trade-level statistics."""

    number_of_trades: int
    profitable_trades: int
    losing_trades: int
    win_rate: float
    average_trade_return: float
    median_trade_return: float
    average_winner: float
    average_loser: float
    largest_winner: float
    largest_loser: float
    profit_factor: float
    average_holding_days: float
    median_holding_days: float
    average_total_cost: float
    stop_exit_count: int
    normal_exit_count: int


def extract_trades(
    backtest_frame: pd.DataFrame,
    ticker_y: str,
    ticker_x: str,
    signal_frame: pd.DataFrame | None = None,
) -> list[TradeRecord]:
    """Extract completed and incomplete trades from a backtest DataFrame.

    Parameters
    ----------
    backtest_frame:
        Output of ``run_backtest()``.  Must contain at least the columns
        ``executed_position``, ``equity``, ``gross_return``, ``net_return``,
        ``transaction_cost``, ``slippage_cost``, ``borrow_cost``.
    ticker_y, ticker_x:
        Constituent tickers for pair identification.
    signal_frame:
        Optional signal frame (from ``create_signal_frame``) used to read
        entry/exit z-scores.  If omitted, z-scores are set to NaN.

    Returns
    -------
    list[TradeRecord]
        One entry per completed trade, plus a final entry for any trade
        still open at the end of the data marked with
        ``ExitReason.END_OF_TEST_PERIOD``.
    """
    _validate_backtest_frame(backtest_frame)

    pair = f"{ticker_y}/{ticker_x}"
    trades: list[TradeRecord] = []

    positions = backtest_frame["executed_position"].astype(int).values
    equity = backtest_frame["equity"].astype(float).values
    gross_returns = backtest_frame["gross_return"].astype(float).values
    net_returns = backtest_frame["net_return"].astype(float).values
    tcosts = backtest_frame["transaction_cost"].astype(float).values
    scosts = backtest_frame["slippage_cost"].astype(float).values
    bcosts = backtest_frame["borrow_cost"].astype(float).values
    dates = backtest_frame.index

    # Extract z-scores if signal frame provided
    zscores: pd.Series | None = None
    if signal_frame is not None and "zscore" in signal_frame.columns:
        zscores = signal_frame["zscore"].astype(float).reindex(dates)

    i = 0
    n = len(positions)

    while i < n:
        # Skip flat periods
        if positions[i] == 0:
            i += 1
            continue

        # Entry at i
        entry_idx = i
        entry_position = positions[i]
        entry_date = dates[entry_idx]
        entry_equity = equity[entry_idx]
        entry_zscore = float(zscores.iloc[entry_idx]) if zscores is not None else float("nan")

        direction = Direction.LONG_SPREAD if entry_position == 1 else Direction.SHORT_SPREAD

        # Find exit: position changes or becomes 0 or reverses
        exit_idx: int | None = None
        exit_reason: ExitReason = ExitReason.END_OF_TEST_PERIOD
        exit_date: pd.Timestamp | None = None

        for j in range(i + 1, n):
            if positions[j] != entry_position:
                # Either flat (0) or reversed (opposite sign)
                exit_idx = j
                if positions[j] == 0:
                    # Check if it was a stop
                    if signal_frame is not None and "stop_flag" in signal_frame.columns:
                        # Look at the signal frame's stop flags
                        sig_date = dates[j]
                        if sig_date in signal_frame.index:
                            if signal_frame.loc[sig_date, "stop_flag"]:
                                exit_reason = ExitReason.STOP_EXIT
                            else:
                                exit_reason = ExitReason.NORMAL_EXIT
                        else:
                            exit_reason = ExitReason.NORMAL_EXIT
                    else:
                        exit_reason = ExitReason.NORMAL_EXIT
                else:
                    # Direct reversal: close old trade, open new one
                    exit_reason = ExitReason.NORMAL_EXIT
                break

        if exit_idx is None:
            # Trade still open at end of data
            exit_idx = n - 1
            exit_reason = ExitReason.END_OF_TEST_PERIOD

        exit_date = dates[exit_idx]
        exit_equity = equity[exit_idx]
        exit_zscore = float(zscores.iloc[exit_idx]) if zscores is not None else float("nan")

        holding_days = exit_idx - entry_idx

        # Gross return from entry to exit
        if holding_days > 0:
            gross_return = (equity[exit_idx] / equity[entry_idx]) - 1.0
            # But subtract interest costs for gross return (they're part of net)
            # Actually, gross_return should be the strategy gross return
            # Let's compute it from cumulative gross returns
            gross_return_cum = 1.0
            for k in range(entry_idx + 1, exit_idx + 1):
                gross_return_cum *= 1.0 + gross_returns[k]
            gross_return = gross_return_cum - 1.0
        else:
            gross_return = 0.0

        # Net return from entry to exit
        if holding_days > 0:
            net_return_cum = 1.0
            for k in range(entry_idx + 1, exit_idx + 1):
                net_return_cum *= 1.0 + net_returns[k]
            net_return = net_return_cum - 1.0
        else:
            net_return = 0.0

        # Cost aggregation over the trade
        total_tcost = float(np.sum(tcosts[entry_idx + 1 : exit_idx + 1]))
        total_scost = float(np.sum(scosts[entry_idx + 1 : exit_idx + 1]))
        total_bcost = float(np.sum(bcosts[entry_idx + 1 : exit_idx + 1]))
        total_cost = total_tcost + total_scost + total_bcost

        # MAE/MFE: worst and best percentage return from entry over the trade
        if holding_days > 0:
            trade_equities = equity[entry_idx : exit_idx + 1]
            returns_from_entry = (trade_equities / trade_equities[0]) - 1.0
            mae = float(np.min(returns_from_entry))
            mfe = float(np.max(returns_from_entry))
        else:
            mae = 0.0
            mfe = 0.0

        trades.append(
            TradeRecord(
                pair=pair,
                ticker_y=ticker_y,
                ticker_x=ticker_x,
                direction=direction,
                entry_date=entry_date,
                exit_date=exit_date,
                entry_zscore=entry_zscore,
                exit_zscore=exit_zscore,
                entry_equity=float(entry_equity),
                exit_equity=float(exit_equity),
                holding_days=holding_days,
                gross_return=gross_return,
                transaction_cost=total_tcost,
                slippage_cost=total_scost,
                borrow_cost=total_bcost,
                total_cost=total_cost,
                net_return=net_return,
                maximum_adverse_excursion=mae,
                maximum_favorable_excursion=mfe,
                exit_reason=exit_reason,
            )
        )

        # If reversal, the next position starts a new trade; otherwise advance.
        # END_OF_TEST_PERIOD always terminates processing.
        if exit_reason == ExitReason.END_OF_TEST_PERIOD:
            i = n  # Break out of the while loop
        elif exit_idx < n and positions[exit_idx] != 0:
            # Reversal: the exit_idx IS the new entry
            i = exit_idx
        else:
            i = exit_idx + 1

    return trades


def _validate_backtest_frame(frame: pd.DataFrame) -> None:
    """Ensure the backtest frame has the required columns."""
    required = {
        "executed_position", "equity", "gross_return", "net_return",
        "transaction_cost", "slippage_cost", "borrow_cost",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"backtest_frame missing columns: {missing}")

    # Validate position values
    positions = frame["executed_position"].values
    invalid = positions[~np.isin(positions, [-1, 0, 1])]
    if len(invalid) > 0:
        raise ValueError(
            f"executed_position must be -1, 0, or 1, got {set(invalid)}"
        )


def summarize_trades(trades: list[TradeRecord]) -> TradeSummary:
    """Aggregate a list of completed trades into a ``TradeSummary``.

    Trades marked ``END_OF_TEST_PERIOD`` are included in counts but their
    returns are not used for win/loss classification.
    """
    if not trades:
        return TradeSummary(
            number_of_trades=0,
            profitable_trades=0,
            losing_trades=0,
            win_rate=float("nan"),
            average_trade_return=float("nan"),
            median_trade_return=float("nan"),
            average_winner=float("nan"),
            average_loser=float("nan"),
            largest_winner=float("nan"),
            largest_loser=float("nan"),
            profit_factor=float("nan"),
            average_holding_days=float("nan"),
            median_holding_days=float("nan"),
            average_total_cost=float("nan"),
            stop_exit_count=0,
            normal_exit_count=0,
        )

    n_trades = len(trades)
    stop_exits = sum(1 for t in trades if t.exit_reason == ExitReason.STOP_EXIT)
    normal_exits = sum(1 for t in trades if t.exit_reason == ExitReason.NORMAL_EXIT)

    # Returns for completed trades (exclude end_of_test_period)
    completed = [t for t in trades if t.exit_reason != ExitReason.END_OF_TEST_PERIOD]
    net_returns = np.array([t.net_return for t in completed])

    if len(net_returns) == 0:
        return TradeSummary(
            number_of_trades=n_trades,
            profitable_trades=0,
            losing_trades=0,
            win_rate=float("nan"),
            average_trade_return=float("nan"),
            median_trade_return=float("nan"),
            average_winner=float("nan"),
            average_loser=float("nan"),
            largest_winner=float("nan"),
            largest_loser=float("nan"),
            profit_factor=float("nan"),
            average_holding_days=float(np.mean([t.holding_days for t in trades])),
            median_holding_days=float(np.median([t.holding_days for t in trades])),
            average_total_cost=float(np.mean([t.total_cost for t in trades])),
            stop_exit_count=stop_exits,
            normal_exit_count=normal_exits,
        )

    winners = net_returns[net_returns > 0]
    losers = net_returns[net_returns < 0]
    profitable = len(winners)
    losing = len(losers)
    win_rate = profitable / len(net_returns) if len(net_returns) > 0 else float("nan")

    avg_winner = float(np.mean(winners)) if len(winners) > 0 else float("nan")
    avg_loser = float(np.mean(losers)) if len(losers) > 0 else float("nan")
    largest_winner_val = float(np.max(winners)) if len(winners) > 0 else float("nan")
    largest_loser_val = float(np.min(losers)) if len(losers) > 0 else float("nan")

    # Profit factor = sum(positive) / abs(sum(negative))
    sum_positive = float(np.sum(winners)) if len(winners) > 0 else 0.0
    sum_negative = float(np.sum(losers)) if len(losers) > 0 else 0.0
    if abs(sum_negative) > 0:
        profit_factor = sum_positive / abs(sum_negative)
    elif sum_positive > 0:
        profit_factor = float("inf")
    else:
        profit_factor = float("nan")

    avg_holding = float(np.mean([t.holding_days for t in trades]))
    median_holding = float(np.median([t.holding_days for t in trades]))
    avg_total_cost = float(np.mean([t.total_cost for t in trades]))

    return TradeSummary(
        number_of_trades=n_trades,
        profitable_trades=profitable,
        losing_trades=losing,
        win_rate=win_rate,
        average_trade_return=float(np.mean(net_returns)),
        median_trade_return=float(np.median(net_returns)),
        average_winner=avg_winner,
        average_loser=avg_loser,
        largest_winner=largest_winner_val,
        largest_loser=largest_loser_val,
        profit_factor=profit_factor,
        average_holding_days=avg_holding,
        median_holding_days=median_holding,
        average_total_cost=avg_total_cost,
        stop_exit_count=stop_exits,
        normal_exit_count=normal_exits,
    )


def trades_to_dataframe(trades: list[TradeRecord]) -> pd.DataFrame:
    """Convert a list of TradeRecords to a DataFrame for CSV export."""
    rows = []
    for t in trades:
        rows.append(
            {
                "pair": t.pair,
                "ticker_y": t.ticker_y,
                "ticker_x": t.ticker_x,
                "direction": t.direction.value,
                "entry_date": t.entry_date,
                "exit_date": t.exit_date,
                "entry_zscore": t.entry_zscore,
                "exit_zscore": t.exit_zscore,
                "entry_equity": t.entry_equity,
                "exit_equity": t.exit_equity,
                "holding_days": t.holding_days,
                "gross_return": t.gross_return,
                "transaction_cost": t.transaction_cost,
                "slippage_cost": t.slippage_cost,
                "borrow_cost": t.borrow_cost,
                "total_cost": t.total_cost,
                "net_return": t.net_return,
                "maximum_adverse_excursion": t.maximum_adverse_excursion,
                "maximum_favorable_excursion": t.maximum_favorable_excursion,
                "exit_reason": t.exit_reason.value,
            }
        )
    return pd.DataFrame(rows)
