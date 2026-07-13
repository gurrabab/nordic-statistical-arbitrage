"""Run walk-forward validation on the Nordic equity universe.

This script downloads the Nordic universe, runs walk-forward validation across
multiple chronological windows, saves detailed and aggregate results to CSV,
and generates diagnostic charts.

Usage
-----
    python walk_forward_analysis.py

Output
------
    results/walk_forward_window_results.csv   — one row per pair per window
    results/walk_forward_pair_summary.csv     — aggregate per-pair statistics
    results/walk_forward_test_return.png      — test return by window
    results/walk_forward_sharpe.png           — Sharpe ratio by window
    results/walk_forward_profitability.png    — profitable-window proportion
    results/walk_forward_drawdown.png         — worst drawdown by pair
    results/walk_forward_selection.png        — selection frequency by pair
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.backtester import BacktestParameters
from src.pair_screener import (
    PairScreeningParameters,
    download_ticker_universe,
)
from src.signals import SignalParameters
from src.universe import NORDIC_UNIVERSE
from src.walk_forward import (
    WalkForwardParameters,
    calculate_walk_forward_summary,
    run_walk_forward_analysis,
)


def main() -> None:
    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------
    screening_parameters = PairScreeningParameters(
        minimum_observations=200,
        maximum_cointegration_pvalue=0.05,
        maximum_adf_pvalue=0.05,
        minimum_half_life=10.0,
        maximum_half_life=250.0,
        minimum_price=1.0,
        maximum_missing_fraction=0.05,
        top_n_pairs=10,
        train_fraction=0.7,
    )

    walk_forward_parameters = WalkForwardParameters(
        train_window_days=504,  # ~2 years of trading days
        test_window_days=63,  # ~3 months
        step_size_days=63,
        expanding_window=False,  # fixed-size rolling windows
        minimum_train_observations=200,
        minimum_test_observations=30,
        top_n_pairs_per_window=3,
    )

    signal_parameters = SignalParameters(
        lookback_window=60,
        entry_threshold=2.0,
        exit_threshold=0.5,
        stop_threshold=3.5,
    )

    backtest_parameters = BacktestParameters(
        initial_capital=100000.0,
        transaction_cost_bps=5.0,
        slippage_bps=2.0,
        annual_borrow_cost=0.02,
    )

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    print("Downloading Nordic universe ...")
    prices, report = download_ticker_universe(NORDIC_UNIVERSE, start="2018-01-01")
    print(report.to_string())
    print(f"Downloaded {len(prices.columns)} tickers, {len(prices)} rows\n")

    # ------------------------------------------------------------------
    # Walk-forward analysis
    # ------------------------------------------------------------------
    print("Running walk-forward validation ...")
    detailed = run_walk_forward_analysis(
        prices,
        screening_parameters,
        walk_forward_parameters,
        signal_parameters,
        backtest_parameters,
    )

    if detailed.empty:
        print("No windows produced — check data availability and parameters.")
        return

    detailed.to_csv(output_dir / "walk_forward_window_results.csv", index=False)
    print(f"Saved {len(detailed)} window-level results to "
          f"{output_dir / 'walk_forward_window_results.csv'}")

    # ------------------------------------------------------------------
    # Aggregate summary
    # ------------------------------------------------------------------
    summary = calculate_walk_forward_summary(detailed)
    summary.to_csv(output_dir / "walk_forward_pair_summary.csv", index=False)
    print(f"Saved {len(summary)} pair summaries to "
          f"{output_dir / 'walk_forward_pair_summary.csv'}")

    print("\nMost consistent pairs (walk-forward):")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)
    print(summary.head(10).to_string(index=False))

    # ------------------------------------------------------------------
    # Charts
    # ------------------------------------------------------------------
    valid = detailed.dropna(subset=["test_total_return"])

    if not valid.empty:
        # --- 1. Test return by window ----------------------------------
        fig1, ax1 = plt.subplots(figsize=(12, 5))
        for (ty, tx), grp in valid.groupby(["ticker_y", "ticker_x"]):
            grp_sorted = grp.sort_values("window_id")
            ax1.plot(
                grp_sorted["window_id"],
                grp_sorted["test_total_return"],
                marker="o",
                linestyle="-",
                label=f"{ty} / {tx}",
            )
        ax1.set_xlabel("Window ID")
        ax1.set_ylabel("Test total return")
        ax1.set_title("Test return by walk-forward window")
        ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
        fig1.tight_layout()
        fig1.savefig(output_dir / "walk_forward_test_return.png", dpi=150)
        plt.close(fig1)

        # --- 2. Sharpe ratio by window ---------------------------------
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        for (ty, tx), grp in valid.groupby(["ticker_y", "ticker_x"]):
            grp_sorted = grp.sort_values("window_id")
            ax2.plot(
                grp_sorted["window_id"],
                grp_sorted["test_sharpe_ratio"],
                marker="s",
                linestyle="-",
                label=f"{ty} / {tx}",
            )
        ax2.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
        ax2.set_xlabel("Window ID")
        ax2.set_ylabel("Test Sharpe ratio")
        ax2.set_title("Test Sharpe ratio by walk-forward window")
        ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
        fig2.tight_layout()
        fig2.savefig(output_dir / "walk_forward_sharpe.png", dpi=150)
        plt.close(fig2)

    if not summary.empty:
        # --- 3. Profitable-window proportion by pair -------------------
        fig3, ax3 = plt.subplots(figsize=(10, max(4, len(summary) * 0.35)))
        labels = summary["ticker_y"] + " / " + summary["ticker_x"]
        colors = [
            "tab:green" if v >= 0.5 else "tab:red"
            for v in summary["profitable_window_fraction"]
        ]
        ax3.barh(labels[::-1], summary["profitable_window_fraction"][::-1],
                 color=colors[::-1])
        ax3.axvline(x=0.5, color="gray", linestyle="--", linewidth=0.8)
        ax3.set_xlabel("Profitable-window fraction")
        ax3.set_title("Proportion of profitable windows by pair")
        fig3.tight_layout()
        fig3.savefig(output_dir / "walk_forward_profitability.png", dpi=150)
        plt.close(fig3)

        # --- 4. Worst drawdown by pair ---------------------------------
        fig4, ax4 = plt.subplots(figsize=(10, max(4, len(summary) * 0.35)))
        ax4.barh(
            labels[::-1],
            summary["worst_maximum_drawdown"][::-1],
            color="tab:orange",
        )
        ax4.set_xlabel("Worst maximum drawdown")
        ax4.set_title("Worst maximum drawdown by pair (across all windows)")
        fig4.tight_layout()
        fig4.savefig(output_dir / "walk_forward_drawdown.png", dpi=150)
        plt.close(fig4)

        # --- 5. Pair selection frequency -------------------------------
        selection_counts = detailed.groupby(["ticker_y", "ticker_x"]).size().sort_values()
        fig5, ax5 = plt.subplots(figsize=(10, max(4, len(selection_counts) * 0.35)))
        selection_labels = selection_counts.index.to_series().apply(
            lambda idx: f"{idx[0]} / {idx[1]}"
        )
        ax5.barh(selection_labels, selection_counts.values, color="tab:blue")
        ax5.set_xlabel("Number of windows")
        ax5.set_title("Pair selection frequency across walk-forward windows")
        fig5.tight_layout()
        fig5.savefig(output_dir / "walk_forward_selection.png", dpi=150)
        plt.close(fig5)

    print("\nCharts saved to results/")


if __name__ == "__main__":
    main()
