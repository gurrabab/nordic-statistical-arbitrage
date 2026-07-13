"""Run a Nordic pair-screening workflow with train/test separation and out-of-sample evaluation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.pair_screener import (
    PairScreeningParameters,
    download_ticker_universe,
    evaluate_top_pairs,
    screen_pairs,
)
from src.universe import NORDIC_UNIVERSE


def main() -> None:
    parameters = PairScreeningParameters(
        minimum_observations=200,
        maximum_cointegration_pvalue=0.05,
        maximum_adf_pvalue=0.05,
        minimum_half_life=10.0,
        maximum_half_life=250.0,
        minimum_price=1.0,
        maximum_missing_fraction=0.05,
        top_n_pairs=5,
        train_fraction=0.7,
    )

    prices, report = download_ticker_universe(NORDIC_UNIVERSE, start="2018-01-01")
    print("Ticker download report")
    print(report.to_string())

    screening_results = screen_pairs(prices, parameters)
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    screening_results.to_csv(output_dir / "pair_screening_results.csv", index=False)

    top_pairs = evaluate_top_pairs(screening_results, prices, parameters, top_n=5)
    top_pairs.to_csv(output_dir / "top_pairs_out_of_sample.csv", index=False)

    print("Top training-ranked pairs")
    print(screening_results.head(10)[["rank", "ticker_y", "ticker_x", "cointegration_pvalue", "adf_pvalue", "half_life", "score"]].to_string(index=False))

    print("Top pairs out-of-sample")
    print(top_pairs.to_string(index=False))

    if not screening_results.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ranking = screening_results.head(10).copy()
        ax.bar(ranking["ticker_y"] + " / " + ranking["ticker_x"], ranking["score"], color="tab:blue")
        ax.set_title("Top 10 pairs ranked by training score")
        ax.set_ylabel("Training score")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(output_dir / "top_10_pairs_training_score.png")
        plt.close(fig)

    if not screening_results.empty and not top_pairs.empty:
        corr_fig, corr_ax = plt.subplots(figsize=(10, 4))
        merged = top_pairs.merge(
            screening_results[["ticker_y", "ticker_x", "cointegration_pvalue"]],
            on=["ticker_y", "ticker_x"],
            how="left",
        )
        corr_ax.scatter(merged["cointegration_pvalue"], merged["test_sharpe_ratio"], alpha=0.7)
        corr_ax.set_xlabel("Training cointegration p-value")
        corr_ax.set_ylabel("Test Sharpe ratio")
        corr_ax.set_title("Training cointegration p-value vs. test Sharpe ratio")
        corr_fig.tight_layout()
        corr_fig.savefig(output_dir / "cointegration_pvalue_vs_test_sharpe.png")
        plt.close(corr_fig)

        return_fig, return_ax = plt.subplots(figsize=(10, 4))
        return_ax.bar(top_pairs["ticker_y"] + " / " + top_pairs["ticker_x"], top_pairs["test_total_return"], color="tab:green")
        return_ax.set_title("Test total return for top evaluated pairs")
        return_ax.set_ylabel("Total return")
        return_ax.tick_params(axis="x", rotation=45)
        return_fig.tight_layout()
        return_fig.savefig(output_dir / "top_pairs_test_total_return.png")
        plt.close(return_fig)

    print(f"Saved screening results to {output_dir / 'pair_screening_results.csv'}")
    print(f"Saved out-of-sample comparison to {output_dir / 'top_pairs_out_of_sample.csv'}")


if __name__ == "__main__":
    main()
