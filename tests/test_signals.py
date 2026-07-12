import numpy as np
import pandas as pd
import pytest

from src.signals import SignalParameters, calculate_rolling_zscore, create_signal_frame


def test_calculate_rolling_zscore_matches_expected_values() -> None:
    spread = pd.Series([0.0, 2.0, 4.0, 6.0], index=pd.date_range("2020-01-01", periods=4, freq="D"))
    result = calculate_rolling_zscore(spread, lookback_window=2)

    assert list(result.columns) == ["spread", "rolling_mean", "rolling_std", "zscore"]
    assert result.index.equals(spread.index)
    assert pd.isna(result.loc[spread.index[0], "zscore"])
    assert result.loc[spread.index[1], "rolling_mean"] == pytest.approx(1.0)
    assert result.loc[spread.index[1], "zscore"] == pytest.approx(1.0)


def test_long_entry_generates_long_position() -> None:
    spread = pd.Series([0.0, 0.0, 0.0, 0.0, -100.0], index=pd.date_range("2020-01-01", periods=5, freq="D"))
    parameters = SignalParameters(lookback_window=5, entry_threshold=2.0, exit_threshold=0.5, stop_threshold=3.5)

    frame = create_signal_frame(spread, parameters)

    assert frame.loc[frame.index[4], "position"] == 1
    assert bool(frame.loc[frame.index[4], "entry_flag"]) is True


def test_short_entry_generates_short_position() -> None:
    spread = pd.Series([0.0, 0.0, 0.0, 0.0, 100.0], index=pd.date_range("2020-01-01", periods=5, freq="D"))
    parameters = SignalParameters(lookback_window=5, entry_threshold=2.0, exit_threshold=0.5, stop_threshold=3.5)

    frame = create_signal_frame(spread, parameters)

    assert frame.loc[frame.index[4], "position"] == -1
    assert bool(frame.loc[frame.index[4], "entry_flag"]) is True


def test_positions_persist_until_exit_or_stop() -> None:
    spread = pd.Series([0.0, 0.0, 0.0, 0.0, -100.0, -100.0], index=pd.date_range("2020-01-01", periods=6, freq="D"))
    parameters = SignalParameters(lookback_window=5, entry_threshold=2.0, exit_threshold=0.5, stop_threshold=3.5)

    frame = create_signal_frame(spread, parameters)

    assert frame.loc[frame.index[4], "position"] == 1
    assert frame.loc[frame.index[5], "position"] == 1


def test_normal_exit_closes_position() -> None:
    spread = pd.Series([0.0, 0.0, 0.0, 0.0, -100.0, 0.0], index=pd.date_range("2020-01-01", periods=6, freq="D"))
    parameters = SignalParameters(lookback_window=5, entry_threshold=2.0, exit_threshold=0.5, stop_threshold=3.5)

    frame = create_signal_frame(spread, parameters)

    assert frame.loc[frame.index[4], "position"] == 1
    assert frame.loc[frame.index[5], "position"] == 0
    assert bool(frame.loc[frame.index[5], "exit_flag"]) is True


def test_stop_exit_closes_position() -> None:
    spread = pd.Series([0.0] * 19 + [-100.0, -10000.0], index=pd.date_range("2020-01-01", periods=21, freq="D"))
    parameters = SignalParameters(lookback_window=20, entry_threshold=2.0, exit_threshold=0.5, stop_threshold=3.5)

    frame = create_signal_frame(spread, parameters)

    assert frame.iloc[-2].position == 1
    assert frame.iloc[-1].position == 0
    assert bool(frame.iloc[-1].stop_flag) is True


def test_no_look_ahead_bias() -> None:
    spread = pd.Series([0.0, 0.0, 0.0, 0.0, 100.0], index=pd.date_range("2020-01-01", periods=5, freq="D"))
    parameters = SignalParameters(lookback_window=5, entry_threshold=2.0, exit_threshold=0.5, stop_threshold=3.5)

    frame = create_signal_frame(spread, parameters)

    assert frame.iloc[3].position == 0
    assert frame.iloc[4].position == -1


def test_invalid_parameters_raise_value_error() -> None:
    with pytest.raises(ValueError, match="lookback"):
        SignalParameters(lookback_window=1, entry_threshold=2.0, exit_threshold=0.5, stop_threshold=3.5)

    with pytest.raises(ValueError, match="exit"):
        SignalParameters(lookback_window=2, entry_threshold=1.0, exit_threshold=1.0, stop_threshold=3.5)

    with pytest.raises(ValueError, match="stop"):
        SignalParameters(lookback_window=2, entry_threshold=2.0, exit_threshold=0.5, stop_threshold=2.0)


def test_initial_nan_rows_produce_flat_positions() -> None:
    spread = pd.Series([0.0, 0.0, 0.0, 0.0], index=pd.date_range("2020-01-01", periods=4, freq="D"))
    parameters = SignalParameters(lookback_window=2, entry_threshold=2.0, exit_threshold=0.5, stop_threshold=3.5)

    frame = create_signal_frame(spread, parameters)

    assert frame.iloc[0].position == 0
    assert frame.iloc[1].position == 0
    assert pd.isna(frame.iloc[0].zscore)


def test_zero_rolling_std_produces_nan_zscore() -> None:
    spread = pd.Series([1.0, 1.0, 1.0, 1.0], index=pd.date_range("2020-01-01", periods=4, freq="D"))
    parameters = SignalParameters(lookback_window=2, entry_threshold=2.0, exit_threshold=0.5, stop_threshold=3.5)

    frame = create_signal_frame(spread, parameters)

    assert pd.isna(frame.iloc[1].zscore)
    assert frame.iloc[1].position == 0
