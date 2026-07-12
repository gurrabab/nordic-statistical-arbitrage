import numpy as np
import pandas as pd
import pytest

from src.risk_metrics import PerformanceSummary, summarize_performance


def test_constant_positive_returns() -> None:
    frame = pd.DataFrame(
        {
            "net_return": [0.01, 0.01, 0.01, 0.01, 0.01],
            "equity": [101.0, 102.01, 103.0301, 104.060401, 105.101005],
        }
    )

    summary = summarize_performance(frame)

    assert summary.total_return > 0.0
    assert summary.annualized_return > 0.0
    assert summary.annualized_volatility == pytest.approx(0.0)
    assert summary.hit_rate == pytest.approx(1.0)


def test_zero_returns() -> None:
    frame = pd.DataFrame({"net_return": [0.0, 0.0, 0.0], "equity": [100.0, 100.0, 100.0]})

    summary = summarize_performance(frame)

    assert summary.total_return == pytest.approx(0.0)
    assert summary.annualized_return == pytest.approx(0.0)
    assert summary.annualized_volatility == pytest.approx(0.0)
    assert summary.sharpe_ratio == pytest.approx(0.0)
    assert summary.sortino_ratio == pytest.approx(0.0)
    assert summary.maximum_drawdown == pytest.approx(0.0)


def test_negative_returns() -> None:
    frame = pd.DataFrame({"net_return": [-0.01, -0.02, -0.03], "equity": [99.0, 97.02, 94.0]})

    summary = summarize_performance(frame)

    assert summary.total_return < 0.0
    assert summary.hit_rate == pytest.approx(0.0)
    assert summary.worst_day == pytest.approx(-0.03)


def test_known_drawdown_example() -> None:
    frame = pd.DataFrame({"net_return": [0.0, -0.1, -0.1111111111, 0.125], "equity": [100.0, 90.0, 80.0, 90.0]})

    summary = summarize_performance(frame)

    assert summary.maximum_drawdown == pytest.approx(-0.2)


def test_sharpe_calculation() -> None:
    returns = pd.Series([0.01, 0.02, 0.03], dtype=float)
    frame = pd.DataFrame({"net_return": returns, "equity": [100.0, 102.0, 105.06]})

    summary = summarize_performance(frame)

    expected = returns.mean() * np.sqrt(252) / returns.std(ddof=1)
    assert summary.sharpe_ratio == pytest.approx(expected)


def test_sortino_calculation() -> None:
    returns = pd.Series([0.01, 0.02, -0.01, 0.03], dtype=float)
    frame = pd.DataFrame({"net_return": returns, "equity": [100.0, 101.0, 100.99, 104.02]})

    summary = summarize_performance(frame)

    downside = returns[returns < 0.0]
    expected_downside = np.sqrt(np.mean(np.square(downside)))
    expected = returns.mean() * np.sqrt(252) / expected_downside if expected_downside > 0 else 0.0
    assert summary.sortino_ratio == pytest.approx(expected)


def test_var_and_expected_shortfall() -> None:
    returns = pd.Series([-0.20, -0.10, 0.0, 0.10, 0.20], dtype=float)
    frame = pd.DataFrame({"net_return": returns, "equity": [100.0, 80.0, 72.0, 79.2, 95.04]})

    summary = summarize_performance(frame)

    expected_var = np.quantile(returns, 0.05)
    expected_es = returns[returns <= expected_var].mean()

    assert summary.value_at_risk_95 == pytest.approx(expected_var)
    assert summary.expected_shortfall_95 == pytest.approx(expected_es)


def test_invalid_input() -> None:
    frame = pd.DataFrame({"net_return": [0.1, np.nan], "equity": [100.0, 100.0]})

    with pytest.raises(ValueError, match="finite"):
        summarize_performance(frame)


def test_missing_required_columns() -> None:
    frame = pd.DataFrame({"net_return": [0.1, 0.2]})

    with pytest.raises(ValueError, match="required"):
        summarize_performance(frame)
