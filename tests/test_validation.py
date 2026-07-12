import numpy as np
import pandas as pd
import pytest

from src.validation import estimate_train_test_relationship, split_aligned_prices


def test_split_aligned_prices_is_chronological_and_respects_minimums() -> None:
    index = pd.date_range("2019-01-01", periods=400, freq="D")
    prices = pd.DataFrame(
        {
            "y": np.linspace(100.0, 200.0, 400),
            "x": np.linspace(100.0, 200.0, 400),
        },
        index=index,
    )

    train_prices, test_prices, split = split_aligned_prices(prices, train_ratio=0.7, min_train_observations=200, min_test_observations=100)

    assert len(train_prices) == 280
    assert len(test_prices) == 120
    assert split.train_end < split.test_start
    assert split.train_observations == 280
    assert split.test_observations == 120


def test_estimate_train_test_relationship_uses_training_parameters_for_test_spread() -> None:
    index = pd.date_range("2020-01-01", periods=400, freq="D")
    x = pd.Series(np.linspace(10.0, 20.0, 400), index=index)
    train_y = pd.Series(np.linspace(5.0, 15.0, 300), index=index[:300])
    test_y = pd.Series(np.linspace(15.0, 25.0, 100), index=index[300:])
    train_x = pd.Series(np.linspace(10.0, 20.0, 300), index=index[:300])
    test_x = pd.Series(np.linspace(20.0, 30.0, 100), index=index[300:])

    alpha, hedge_ratio, train_spread, test_spread, test_zscore = estimate_train_test_relationship(
        train_y,
        train_x,
        test_y,
        test_x,
        lookback_window=60,
    )

    assert alpha == pytest.approx(-5.0, abs=1e-8)
    assert hedge_ratio == pytest.approx(1.0, abs=1e-8)
    assert len(train_spread) == 300
    assert len(test_spread) == 100
    assert len(test_zscore) == 100
    assert np.isnan(test_zscore.iloc[:59]).all()
