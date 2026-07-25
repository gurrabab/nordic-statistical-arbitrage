"""Tests for trade_analysis module."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.trade_analysis import (
    Direction,
    ExitReason,
    TradeRecord,
    TradeSummary,
    extract_trades,
    summarize_trades,
    trades_to_dataframe,
)


def _make_backtest_frame(dates, executed_positions, equity_curve=None):
    """Build a minimal backtest frame for trade extraction."""
    n = len(dates)
    df = pd.DataFrame(
        {
            "executed_position": executed_positions,
            "equity": equity_curve if equity_curve is not None else [100_000.0] * n,
            "net_return": [0.0] * n,
            "gross_return": [0.0] * n,
            "transaction_cost": 0.0,
            "slippage_cost": 0.0,
            "borrow_cost": 0.0,
            "signal_position": executed_positions,
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )
    for col in [
        "price_y", "price_x", "return_y", "return_x", "weight_y",
        "weight_x", "gross_exposure", "net_exposure",
        "turnover", "cumulative_return",
    ]:
        if col not in df.columns:
            df[col] = 0.0 if col != "cumulative_return" else 1.0
    return df


class TestExtractTrades:
    def test_no_trades_when_flat(self):
        dates = pd.date_range("2020-01-01", periods=10, freq="B")
        bt = _make_backtest_frame(dates, [0] * 10)
        trades = extract_trades(bt, "A", "B")
        assert len(trades) == 0

    def test_single_long_trade(self):
        dates = pd.date_range("2020-01-01", periods=10, freq="B")
        positions = [0, 0, 1, 1, 1, 1, 1, 0, 0, 0]
        bt = _make_backtest_frame(dates, positions)
        trades = extract_trades(bt, "A", "B")
        assert len(trades) == 1
        t = trades[0]
        assert t.direction == Direction.LONG_SPREAD
        assert t.entry_date == dates[2]
        assert t.exit_date == dates[7]

    def test_single_short_trade(self):
        dates = pd.date_range("2020-01-01", periods=10, freq="B")
        positions = [0, -1, -1, -1, -1, 0, 0, 0, 0, 0]
        bt = _make_backtest_frame(dates, positions)
        trades = extract_trades(bt, "A", "B")
        assert len(trades) == 1
        assert trades[0].direction == Direction.SHORT_SPREAD
        assert trades[0].entry_date == dates[1]
        assert trades[0].exit_date == dates[5]

    def test_reversal_closes_and_opens(self):
        dates = pd.date_range("2020-01-01", periods=10, freq="B")
        positions = [0, 1, 1, -1, -1, -1, 0, 0, 0, 0]
        bt = _make_backtest_frame(dates, positions)
        trades = extract_trades(bt, "A", "B")
        assert len(trades) == 2
        assert trades[0].direction == Direction.LONG_SPREAD
        assert trades[0].exit_reason == ExitReason.NORMAL_EXIT
        assert trades[0].exit_date == dates[3]
        assert trades[1].direction == Direction.SHORT_SPREAD
        assert trades[1].entry_date == dates[3]
        assert trades[1].exit_date == dates[6]

    def test_incomplete_trade_at_end(self):
        dates = pd.date_range("2020-01-01", periods=10, freq="B")
        positions = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        bt = _make_backtest_frame(dates, positions)
        trades = extract_trades(bt, "A", "B")
        assert len(trades) == 1
        assert trades[0].exit_reason == ExitReason.END_OF_TEST_PERIOD

    def test_extract_returns_and_costs(self):
        dates = pd.date_range("2020-01-01", periods=5, freq="B")
        positions = [0, 1, 1, 1, 0]
        equity = [100_000.0, 100_500.0, 101_000.0, 100_800.0, 101_200.0]
        bt = _make_backtest_frame(dates, positions, equity_curve=equity)
        bt["transaction_cost"] = [0.0, 10.0, 0.0, 0.0, 10.0]
        bt["slippage_cost"] = [0.0, 5.0, 0.0, 0.0, 5.0]
        bt["borrow_cost"] = [0.0, 2.0, 2.0, 2.0, 0.0]
        bt["gross_return"] = [0.0, 0.005, 0.005, -0.002, 0.004]
        trades = extract_trades(bt, "A", "B")
        assert len(trades) == 1
        assert trades[0].net_return is not None
        assert trades[0].total_cost > 0

    def test_invalid_position_values_raise(self):
        dates = pd.date_range("2020-01-01", periods=5, freq="B")
        bt = _make_backtest_frame(dates, [0, 2, 0, 0, 0])
        with pytest.raises(ValueError, match="position.*must be.*-1, 0, or 1"):
            extract_trades(bt, "A", "B")

    def test_empty_frame(self):
        bt = _make_backtest_frame([], [])
        trades = extract_trades(bt, "A", "B")
        assert len(trades) == 0


class TestSummarizeTrades:
    def test_basic_summary(self, sample_trades):
        summary = summarize_trades(sample_trades)
        assert isinstance(summary, TradeSummary)
        assert summary.number_of_trades == 3
        assert summary.profitable_trades == 2
        assert summary.losing_trades == 1
        assert summary.win_rate == pytest.approx(2 / 3)

    def test_profit_factor(self, sample_trades):
        summary = summarize_trades(sample_trades)
        assert summary.average_winner > 0
        assert summary.average_loser < 0

    def test_summary_no_trades(self):
        summary = summarize_trades([])
        assert summary.number_of_trades == 0
        assert np.isnan(summary.win_rate)

    def test_avg_holding_period(self, sample_trades):
        summary = summarize_trades(sample_trades)
        assert summary.average_holding_days > 0

    @pytest.fixture
    def sample_trades(self):
        base = datetime(2020, 1, 2)
        return [
            TradeRecord(
                pair="A/B",
                ticker_y="A",
                ticker_x="B",
                entry_date=base,
                exit_date=base + timedelta(days=5),
                direction=Direction.LONG_SPREAD,
                exit_reason=ExitReason.NORMAL_EXIT,
                entry_zscore=2.0,
                exit_zscore=0.5,
                entry_equity=100_000.0,
                exit_equity=105_000.0,
                holding_days=5,
                gross_return=0.05,
                net_return=0.048,
                transaction_cost=15.0,
                slippage_cost=3.0,
                borrow_cost=2.0,
                total_cost=20.0,
                maximum_adverse_excursion=-0.01,
                maximum_favorable_excursion=0.06,
            ),
            TradeRecord(
                pair="A/B",
                ticker_y="A",
                ticker_x="B",
                entry_date=base + timedelta(days=10),
                exit_date=base + timedelta(days=15),
                direction=Direction.SHORT_SPREAD,
                exit_reason=ExitReason.NORMAL_EXIT,
                entry_zscore=-2.0,
                exit_zscore=-0.5,
                entry_equity=105_000.0,
                exit_equity=108_000.0,
                holding_days=5,
                gross_return=0.03,
                net_return=0.029,
                transaction_cost=8.0,
                slippage_cost=1.0,
                borrow_cost=1.0,
                total_cost=10.0,
                maximum_adverse_excursion=-0.02,
                maximum_favorable_excursion=0.04,
            ),
            TradeRecord(
                pair="A/B",
                ticker_y="A",
                ticker_x="B",
                entry_date=base + timedelta(days=20),
                exit_date=base + timedelta(days=25),
                direction=Direction.LONG_SPREAD,
                exit_reason=ExitReason.STOP_EXIT,
                entry_zscore=2.0,
                exit_zscore=3.5,
                entry_equity=108_000.0,
                exit_equity=103_680.0,
                holding_days=5,
                gross_return=-0.04,
                net_return=-0.042,
                transaction_cost=15.0,
                slippage_cost=3.0,
                borrow_cost=2.0,
                total_cost=20.0,
                maximum_adverse_excursion=-0.05,
                maximum_favorable_excursion=0.02,
            ),
        ]


class TestTradesToDataFrame:
    def test_converts_records(self, sample_trades):
        df = trades_to_dataframe(sample_trades)
        assert len(df) == 3
        assert "pair" in df.columns
        assert "direction" in df.columns
        assert "net_return" in df.columns

    def test_empty_list(self):
        df = trades_to_dataframe([])
        assert len(df) == 0

    @pytest.fixture
    def sample_trades(self):
        base = datetime(2020, 1, 2)
        return [
            TradeRecord(
                pair="A/B",
                ticker_y="A",
                ticker_x="B",
                entry_date=base,
                exit_date=base + timedelta(days=5),
                direction=Direction.LONG_SPREAD,
                exit_reason=ExitReason.NORMAL_EXIT,
                entry_zscore=2.0,
                exit_zscore=0.5,
                entry_equity=100_000.0,
                exit_equity=105_000.0,
                holding_days=5,
                gross_return=0.05,
                net_return=0.048,
                transaction_cost=15.0,
                slippage_cost=3.0,
                borrow_cost=2.0,
                total_cost=20.0,
                maximum_adverse_excursion=-0.01,
                maximum_favorable_excursion=0.06,
            ),
            TradeRecord(
                pair="A/B",
                ticker_y="A",
                ticker_x="B",
                entry_date=base + timedelta(days=10),
                exit_date=base + timedelta(days=15),
                direction=Direction.SHORT_SPREAD,
                exit_reason=ExitReason.NORMAL_EXIT,
                entry_zscore=-2.0,
                exit_zscore=-0.5,
                entry_equity=105_000.0,
                exit_equity=108_000.0,
                holding_days=5,
                gross_return=0.03,
                net_return=0.029,
                transaction_cost=8.0,
                slippage_cost=1.0,
                borrow_cost=1.0,
                total_cost=10.0,
                maximum_adverse_excursion=-0.02,
                maximum_favorable_excursion=0.04,
            ),
            TradeRecord(
                pair="A/B",
                ticker_y="A",
                ticker_x="B",
                entry_date=base + timedelta(days=20),
                exit_date=base + timedelta(days=25),
                direction=Direction.LONG_SPREAD,
                exit_reason=ExitReason.STOP_EXIT,
                entry_zscore=2.0,
                exit_zscore=3.5,
                entry_equity=108_000.0,
                exit_equity=103_680.0,
                holding_days=5,
                gross_return=-0.04,
                net_return=-0.042,
                transaction_cost=15.0,
                slippage_cost=3.0,
                borrow_cost=2.0,
                total_cost=20.0,
                maximum_adverse_excursion=-0.05,
                maximum_favorable_excursion=0.02,
            ),
        ]
