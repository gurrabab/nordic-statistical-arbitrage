"""Utilities for converting a residual spread into discrete trading signals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SignalParameters:
    """Configuration for generating spread-based trading signals."""

    lookback_window: int
    entry_threshold: float
    exit_threshold: float
    stop_threshold: float

    def __post_init__(self) -> None:
        if self.lookback_window < 2:
            raise ValueError("lookback_window must be at least 2.")
        if not (self.entry_threshold > self.exit_threshold >= 0):
            raise ValueError("entry_threshold must be greater than exit_threshold, and exit_threshold must be non-negative.")
        if not (self.stop_threshold > self.entry_threshold):
            raise ValueError("stop_threshold must be greater than entry_threshold.")


def calculate_rolling_zscore(
    spread: pd.Series,
    lookback_window: int,
) -> pd.DataFrame:
    """Compute rolling mean, rolling standard deviation, and z-score for a spread series.

    The calculation is fully backward-looking and uses only observations up to
    the current row, which avoids look-ahead bias.
    """
    if not isinstance(spread, pd.Series):
        raise TypeError("spread must be a pandas Series.")

    if lookback_window < 2:
        raise ValueError("lookback_window must be at least 2.")

    cleaned = spread.astype(float).replace([np.inf, -np.inf], np.nan)
    rolling_mean = cleaned.rolling(window=lookback_window, min_periods=lookback_window).mean()
    rolling_std = cleaned.rolling(window=lookback_window, min_periods=lookback_window).std(ddof=0)

    zscore = pd.Series(np.nan, index=cleaned.index, dtype=float)
    valid_rows = rolling_mean.notna() & rolling_std.notna() & (rolling_std != 0)
    zscore.loc[valid_rows] = (
        (cleaned.loc[valid_rows] - rolling_mean.loc[valid_rows]) / rolling_std.loc[valid_rows]
    )

    return pd.DataFrame(
        {
            "spread": cleaned,
            "rolling_mean": rolling_mean,
            "rolling_std": rolling_std,
            "zscore": zscore,
        },
        index=cleaned.index,
    )


def generate_spread_positions(
    zscore: pd.Series,
    parameters: SignalParameters,
) -> pd.Series:
    """Generate discrete long, short, and flat positions from a z-score series."""
    if not isinstance(zscore, pd.Series):
        raise TypeError("zscore must be a pandas Series.")

    if not isinstance(parameters, SignalParameters):
        raise TypeError("parameters must be a SignalParameters instance.")

    positions = pd.Series(0, index=zscore.index, dtype=int)
    current_position = 0

    for idx, value in zscore.items():
        if pd.isna(value):
            positions.loc[idx] = 0
            continue

        if current_position == 0:
            if value <= -parameters.entry_threshold:
                current_position = 1
                positions.loc[idx] = 1
            elif value >= parameters.entry_threshold:
                current_position = -1
                positions.loc[idx] = -1
            else:
                positions.loc[idx] = 0
        elif current_position == 1:
            if value >= -parameters.exit_threshold:
                current_position = 0
                positions.loc[idx] = 0
            elif value <= -parameters.stop_threshold:
                current_position = 0
                positions.loc[idx] = 0
            else:
                positions.loc[idx] = 1
        else:
            if value <= parameters.exit_threshold:
                current_position = 0
                positions.loc[idx] = 0
            elif value >= parameters.stop_threshold:
                current_position = 0
                positions.loc[idx] = 0
            else:
                positions.loc[idx] = -1

    return positions


def create_signal_frame(
    spread: pd.Series,
    parameters: SignalParameters,
) -> pd.DataFrame:
    """Create a full signal frame with z-scores and position states."""
    if not isinstance(spread, pd.Series):
        raise TypeError("spread must be a pandas Series.")

    if not isinstance(parameters, SignalParameters):
        raise TypeError("parameters must be a SignalParameters instance.")

    zscore_frame = calculate_rolling_zscore(spread, parameters.lookback_window)
    positions = generate_spread_positions(zscore_frame["zscore"], parameters)

    frame = pd.DataFrame(
        {
            "spread": zscore_frame["spread"],
            "rolling_mean": zscore_frame["rolling_mean"],
            "rolling_std": zscore_frame["rolling_std"],
            "zscore": zscore_frame["zscore"],
            "position": positions,
        },
        index=spread.index,
    )

    frame["entry_flag"] = False
    frame["exit_flag"] = False
    frame["stop_flag"] = False

    previous_position = 0
    for idx, row in frame.iterrows():
        zscore_value = row["zscore"]
        current_position = int(row["position"])

        if pd.isna(zscore_value):
            previous_position = current_position
            continue

        if previous_position == 0 and current_position in {1, -1}:
            frame.loc[idx, "entry_flag"] = True
        elif previous_position in {1, -1} and current_position == 0:
            frame.loc[idx, "exit_flag"] = True

        if previous_position in {1, -1} and current_position == 0 and (
            (previous_position == 1 and zscore_value <= -parameters.stop_threshold)
            or (previous_position == -1 and zscore_value >= parameters.stop_threshold)
        ):
            frame.loc[idx, "stop_flag"] = True

        previous_position = current_position

    return frame
