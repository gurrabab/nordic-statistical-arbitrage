"""Utilities for chronological train/test evaluation without look-ahead bias."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.signals import calculate_rolling_zscore


@dataclass(frozen=True)
class TrainTestSplit:
    """Container for chronological train/test split metadata."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_observations: int
    test_observations: int


def split_aligned_prices(
    prices: pd.DataFrame,
    train_ratio: float = 0.7,
    min_train_observations: int = 252,
    min_test_observations: int = 126,
) -> tuple[pd.DataFrame, pd.DataFrame, TrainTestSplit]:
    """Split aligned price data into chronological train and test samples.

    The split is strictly chronological and does not shuffle rows. The test set
    starts immediately after the final training observation, which ensures that
    the out-of-sample period remains untouched during model fitting.
    """
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame.")

    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("prices must have a DatetimeIndex.")

    if prices.empty:
        raise ValueError("prices must not be empty.")

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    if min_train_observations <= 0 or min_test_observations <= 0:
        raise ValueError("minimum observation counts must be positive.")

    sorted_prices = prices.sort_index()
    total_observations = len(sorted_prices)
    if total_observations < min_train_observations + min_test_observations:
        raise ValueError("Not enough observations to create train and test samples.")

    train_size = int(np.floor(total_observations * train_ratio))
    if train_size < min_train_observations:
        train_size = min_train_observations
    if total_observations - train_size < min_test_observations:
        train_size = total_observations - min_test_observations

    if train_size <= 0 or total_observations - train_size <= 0:
        raise ValueError("The train/test split is invalid.")

    train_frame = sorted_prices.iloc[:train_size].copy()
    test_frame = sorted_prices.iloc[train_size:].copy()

    split = TrainTestSplit(
        train_start=train_frame.index[0],
        train_end=train_frame.index[-1],
        test_start=test_frame.index[0],
        test_end=test_frame.index[-1],
        train_observations=len(train_frame),
        test_observations=len(test_frame),
    )

    return train_frame, test_frame, split


def estimate_train_test_relationship(
    train_y: pd.Series,
    train_x: pd.Series,
    test_y: pd.Series,
    test_x: pd.Series,
    lookback_window: int,
) -> tuple[float, float, pd.Series, pd.Series, pd.Series]:
    """Estimate the model on training data and build the test-period spread and z-score.

    The training data is used to estimate alpha and the hedge ratio. These fixed
    estimates are then applied to the test period without re-estimation. The
    rolling z-score for the test period uses only the training history available
    before each test date, which prevents look-ahead bias.
    """
    if not isinstance(train_y, pd.Series) or not isinstance(train_x, pd.Series):
        raise TypeError("train_y and train_x must be pandas Series.")
    if not isinstance(test_y, pd.Series) or not isinstance(test_x, pd.Series):
        raise TypeError("test_y and test_x must be pandas Series.")

    if not isinstance(train_y.index, pd.DatetimeIndex) or not isinstance(test_y.index, pd.DatetimeIndex):
        raise ValueError("price series must have a DatetimeIndex.")

    if lookback_window < 2:
        raise ValueError("lookback_window must be at least 2.")

    train_y = train_y.astype(float)
    train_x = train_x.astype(float)
    test_y = test_y.astype(float)
    test_x = test_x.astype(float)

    if not np.isfinite(train_y).all() or not np.isfinite(train_x).all():
        raise ValueError("training prices must be finite.")
    if not np.isfinite(test_y).all() or not np.isfinite(test_x).all():
        raise ValueError("test prices must be finite.")

    if (train_y <= 0).any() or (train_x <= 0).any() or (test_y <= 0).any() or (test_x <= 0).any():
        raise ValueError("prices must be positive.")

    design_matrix = sm.add_constant(train_x, has_constant="add")
    model = sm.OLS(train_y, design_matrix)
    results = model.fit()
    alpha = float(results.params.iloc[0])
    hedge_ratio = float(results.params.iloc[1])

    train_spread = train_y - alpha - hedge_ratio * train_x
    test_spread = test_y - alpha - hedge_ratio * test_x

    test_zscore_frame = calculate_rolling_zscore(test_spread, lookback_window)
    test_zscore = test_zscore_frame["zscore"]

    return alpha, hedge_ratio, train_spread, test_spread, test_zscore
