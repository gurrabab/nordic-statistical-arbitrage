"""Entry point for the Nordic statistical arbitrage workflow with backtesting."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.backtester import BacktestParameters, run_backtest
from src.data_loader import download_adjusted_close_prices
from src.pair_selection import align_price_series, analyze_pair, estimate_ols_regression
from src.risk_metrics import summarize_performance
from src.signals import SignalParameters, create_signal_frame
from src.validation import estimate_train_test_relationship, split_aligned_prices


def main() -> None:
    """Download prices, analyze the pair, generate signals, and run a simple backtest."""
    prices = download_adjusted_close_prices(
        ["SEB-A.ST", "SHB-A.ST"],
        start="2020-01-01",
    )

    result = analyze_pair(prices, ticker_y="SEB-A.ST", ticker_x="SHB-A.ST")

    y_series = prices["SEB-A.ST"]
    x_series = prices["SHB-A.ST"]
    y_aligned, x_aligned = align_price_series(y_series, x_series)
    _, hedge_ratio, spread = estimate_ols_regression(y_aligned, x_aligned)

    signal_parameters = SignalParameters(
        lookback_window=60,
        entry_threshold=2.0,
        exit_threshold=0.5,
        stop_threshold=3.5,
    )
    signal_frame = create_signal_frame(spread, signal_parameters)

    backtest_parameters = BacktestParameters(
        initial_capital=100000.0,
        transaction_cost_bps=5.0,
        slippage_bps=2.0,
        annual_borrow_cost=0.02,
    )
    backtest_frame = run_backtest(
        y_aligned,
        x_aligned,
        signal_frame["position"],
        hedge_ratio=hedge_ratio,
        parameters=backtest_parameters,
    )

    aligned_prices = pd.concat([y_aligned.rename("SEB-A.ST"), x_aligned.rename("SHB-A.ST")], axis=1).sort_index()
    train_prices, test_prices, split = split_aligned_prices(
        aligned_prices,
        train_ratio=0.7,
        min_train_observations=252,
        min_test_observations=126,
    )

    train_y = train_prices["SEB-A.ST"]
    train_x = train_prices["SHB-A.ST"]
    test_y = test_prices["SEB-A.ST"]
    test_x = test_prices["SHB-A.ST"]
    alpha, hedge_ratio_train_test, train_spread, test_spread, test_zscore = estimate_train_test_relationship(
        train_y,
        train_x,
        test_y,
        test_x,
        lookback_window=signal_parameters.lookback_window,
    )

    train_signal_frame = create_signal_frame(train_spread, signal_parameters)
    test_signal_frame = create_signal_frame(test_spread, signal_parameters)

    train_backtest = run_backtest(
        train_y,
        train_x,
        train_signal_frame["position"],
        hedge_ratio=hedge_ratio_train_test,
        parameters=backtest_parameters,
    )
    test_backtest = run_backtest(
        test_y,
        test_x,
        test_signal_frame["position"],
        hedge_ratio=hedge_ratio_train_test,
        parameters=backtest_parameters,
    )

    train_summary = summarize_performance(train_backtest)
    test_summary = summarize_performance(test_backtest)

    comparison_frame = pd.DataFrame(
        [
            {
                "period": "train",
                "start": split.train_start,
                "end": split.train_end,
                "observations": split.train_observations,
                "total_return": train_summary.total_return,
                "sharpe": train_summary.sharpe_ratio,
                "max_drawdown": train_summary.maximum_drawdown,
            },
            {
                "period": "test",
                "start": split.test_start,
                "end": split.test_end,
                "observations": split.test_observations,
                "total_return": test_summary.total_return,
                "sharpe": test_summary.sharpe_ratio,
                "max_drawdown": test_summary.maximum_drawdown,
            },
        ]
    )

    print("Pair analysis summary")
    print(f"Ticker Y: {result.ticker_y}")
    print(f"Ticker X: {result.ticker_x}")
    print(f"Alpha: {result.alpha:.6f}")
    print(f"Hedge ratio: {result.hedge_ratio:.6f}")
    print(f"Cointegration statistic: {result.cointegration_statistic:.6f}")
    print(f"Cointegration p-value: {result.cointegration_pvalue:.6f}")
    print(f"ADF statistic: {result.adf_statistic:.6f}")
    print(f"ADF p-value: {result.adf_pvalue:.6f}")
    print(f"Half-life: {result.half_life}")
    print(f"Observations: {result.n_observations}")
    print("Latest signal rows:")
    print(signal_frame.tail(10)[["spread", "zscore", "position", "entry_flag", "exit_flag", "stop_flag"]].to_string())

    performance_summary = summarize_performance(backtest_frame)

    print("Backtest summary")
    print(f"Initial capital: {backtest_parameters.initial_capital:.2f}")
    print(f"Final equity: {backtest_frame['equity'].dropna().iloc[-1]:.2f}")
    print(f"Cumulative return: {backtest_frame['cumulative_return'].dropna().iloc[-1]:.2%}")
    print(f"Number of trading days: {len(backtest_frame)}")
    print(f"Total transaction costs: {backtest_frame['transaction_cost'].sum():.2f}")
    print(f"Total slippage costs: {backtest_frame['slippage_cost'].sum():.2f}")
    print(f"Total borrow costs: {backtest_frame['borrow_cost'].sum():.2f}")
    print("Train/test comparison")
    print(comparison_frame.to_string(index=False))
    print("Performance summary")
    print(f"Total return: {performance_summary.total_return:.2%}")
    print(f"Annualized return: {performance_summary.annualized_return:.2%}")
    print(f"Annualized volatility: {performance_summary.annualized_volatility:.2%}")
    print(f"Sharpe ratio: {performance_summary.sharpe_ratio:.4f}")
    print(f"Sortino ratio: {performance_summary.sortino_ratio:.4f}")
    print(f"Maximum drawdown: {performance_summary.maximum_drawdown:.2%}")
    print(f"Calmar ratio: {performance_summary.calmar_ratio:.4f}")
    print(f"VaR 95%: {performance_summary.value_at_risk_95:.2%}")
    print(f"Expected Shortfall 95%: {performance_summary.expected_shortfall_95:.2%}")
    print(f"Hit rate: {performance_summary.hit_rate:.2%}")
    print("Latest backtest rows:")
    print(backtest_frame.tail(10)[["signal_position", "executed_position", "weight_y", "weight_x", "gross_return", "transaction_cost", "slippage_cost", "borrow_cost", "net_return", "equity"]].to_string())

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(signal_frame.index, signal_frame["zscore"], label="Z-score")
    ax.axhline(2.0, linestyle="--", color="tab:orange", label="Entry threshold")
    ax.axhline(-2.0, linestyle="--", color="tab:orange")
    ax.axhline(0.5, linestyle=":", color="tab:green", label="Exit threshold")
    ax.axhline(-0.5, linestyle=":", color="tab:green")
    ax.axhline(3.5, linestyle="-.", color="tab:red", label="Stop threshold")
    ax.axhline(-3.5, linestyle="-.", color="tab:red")
    ax.set_title("Spread z-score with signal thresholds")
    ax.set_xlabel("Date")
    ax.set_ylabel("Z-score")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "signal_zscore.png")
    plt.close(fig)

    equity_fig, equity_ax = plt.subplots(figsize=(10, 4))
    equity_ax.plot(backtest_frame.index, backtest_frame["equity"], label="Equity")
    equity_ax.set_title("Equity curve")
    equity_ax.set_xlabel("Date")
    equity_ax.set_ylabel("Equity")
    equity_ax.legend()
    equity_fig.tight_layout()
    equity_fig.savefig(output_dir / "equity_curve.png")
    plt.close(equity_fig)

    position_fig, position_ax = plt.subplots(figsize=(10, 4))
    position_ax.plot(backtest_frame.index, backtest_frame["executed_position"], drawstyle="steps-post", label="Executed position")
    position_ax.axhline(0, color="black", linewidth=0.8)
    position_ax.set_title("Executed position over time")
    position_ax.set_xlabel("Date")
    position_ax.set_ylabel("Position")
    position_ax.legend()
    position_fig.tight_layout()
    position_fig.savefig(output_dir / "executed_position.png")
    plt.close(position_fig)

    drawdown_series = (backtest_frame["equity"] / backtest_frame["equity"].cummax() - 1.0)
    drawdown_fig, drawdown_ax = plt.subplots(figsize=(10, 4))
    drawdown_ax.plot(backtest_frame.index, drawdown_series, label="Drawdown")
    drawdown_ax.axhline(0, color="black", linewidth=0.8)
    drawdown_ax.set_title("Drawdown over time")
    drawdown_ax.set_xlabel("Date")
    drawdown_ax.set_ylabel("Drawdown")
    drawdown_ax.legend()
    drawdown_fig.tight_layout()
    drawdown_fig.savefig(output_dir / "drawdown.png")
    plt.close(drawdown_fig)

    returns_hist_fig, returns_hist_ax = plt.subplots(figsize=(10, 4))
    returns_hist_ax.hist(backtest_frame["net_return"].dropna(), bins=30, color="tab:blue", alpha=0.7)
    returns_hist_ax.set_title("Histogram of daily net returns")
    returns_hist_ax.set_xlabel("Daily net return")
    returns_hist_ax.set_ylabel("Frequency")
    returns_hist_fig.tight_layout()
    returns_hist_fig.savefig(output_dir / "daily_return_histogram.png")
    plt.close(returns_hist_fig)

    comparison_frame.to_csv(output_dir / "train_test_comparison.csv", index=False)

    print("Saved charts to results/signal_zscore.png, results/equity_curve.png, results/executed_position.png, results/drawdown.png, and results/daily_return_histogram.png")
    print("Saved train/test comparison to results/train_test_comparison.csv")


if __name__ == "__main__":
    main()
