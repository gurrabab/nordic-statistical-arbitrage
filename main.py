"""Minimal entry point for the Nordic statistical arbitrage project."""

from pathlib import Path

import matplotlib.pyplot as plt

from src.data_loader import download_adjusted_close_prices
from src.pair_selection import align_price_series, analyze_pair, estimate_ols_regression
from src.signals import SignalParameters, create_signal_frame


def main() -> None:
    """Download historical prices, analyze the pair, and print a concise summary."""
    prices = download_adjusted_close_prices(
        ["SEB-A.ST", "SHB-A.ST"],
        start="2019-01-01",
    )

    result = analyze_pair(prices, ticker_y="SEB-A.ST", ticker_x="SHB-A.ST")

    y_series = prices["SEB-A.ST"]
    x_series = prices["SHB-A.ST"]
    y_aligned, x_aligned = align_price_series(y_series, x_series)
    _, _, spread = estimate_ols_regression(y_aligned, x_aligned)

    parameters = SignalParameters(
        lookback_window=60,
        entry_threshold=2.0,
        exit_threshold=0.5,
        stop_threshold=3.5,
    )
    signal_frame = create_signal_frame(spread, parameters)

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

    print("Saved chart to results/signal_zscore.png")


if __name__ == "__main__":
    main()
