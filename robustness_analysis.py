#!/usr/bin/env python3
"""Comprehensive robustness analysis for the Nordic Statistical Arbitrage project.

Usage
-----
    python robustness_analysis.py [--tickers TICKER_Y TICKER_X ...]
                                  [--start START] [--end END]
                                  [--output-dir OUTPUT_DIR]

This script:
1. Downloads and assesses data quality
2. Runs the baseline pair analysis
3. Screens the universe and runs the baseline backtest
4. Extracts trade-level results
5. Evaluates benchmarks
6. Runs cost sensitivity
7. Runs parameter sensitivity (diagnostic only — no test-period optimisation)
8. Applies multiple-testing corrections
9. Saves all result files to the output directory
10. Prints a concise research summary

.. note::

    The parameter-sensitivity results should be used for **diagnosis only**.
    The baseline parameters remain the project's primary result.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure the project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtester import BacktestParameters, run_backtest  # noqa: E402
from src.benchmarks import compare_benchmarks  # noqa: E402
from src.data_loader import download_adjusted_close_prices  # noqa: E402
from src.data_quality import assess_all_tickers, compute_all_pair_overlaps  # noqa: E402
from src.pair_screener import PairScreeningParameters, screen_pairs  # noqa: E402
from src.pair_selection import align_price_series  # noqa: E402
from src.risk_metrics import summarize_performance  # noqa: E402
from src.sensitivity import (  # noqa: E402
    parameter_stability_summary,
    run_cost_sensitivity,
    run_parameter_sensitivity,
)
from src.signals import SignalParameters, create_signal_frame  # noqa: E402
from src.trade_analysis import extract_trades, summarize_trades, trades_to_dataframe  # noqa: E402
from src.validation import (  # noqa: E402
    estimate_train_test_relationship,
    split_aligned_prices,
)

DEFAULT_TICKERS = ["SEB-A.ST", "SHB-A.ST", "AZN.ST", "ERIC-B.ST", "SAND.ST"]
DEFAULT_START = "2018-01-01"
DEFAULT_END = "2024-12-31"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Comprehensive robustness analysis for Nordic Statistical Arbitrage",
    )
    parser.add_argument(
        "--tickers", nargs="+", default=DEFAULT_TICKERS,
        help=f"Ticker symbols (default: {DEFAULT_TICKERS})",
    )
    parser.add_argument(
        "--start", default=DEFAULT_START,
        help=f"Start date (default: {DEFAULT_START})",
    )
    parser.add_argument(
        "--end", default=DEFAULT_END,
        help=f"End date (default: {DEFAULT_END})",
    )
    parser.add_argument(
        "--output-dir", default="results",
        help="Output directory (default: results/)",
    )
    parser.add_argument(
        "--ticker-y", default="SEB-A.ST",
        help="Primary ticker Y (dependent variable)",
    )
    parser.add_argument(
        "--ticker-x", default="SHB-A.ST",
        help="Primary ticker X (independent variable)",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip data download (use cached CSV files)",
    )
    return parser.parse_args()


def print_separator(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Robustness Analysis — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Tickers: {', '.join(args.tickers)}")
    print(f"Period: {args.start} – {args.end}")
    print(f"Primary pair: {args.ticker_y} / {args.ticker_x}")

    # ------------------------------------------------------------------
    # 1. Data download
    # ------------------------------------------------------------------
    print_separator("1. Data download")
    if args.skip_download:
        csv_path = output_dir / "prices.csv"
        if csv_path.exists():
            prices = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            print(f"Loaded cached data from {csv_path} ({len(prices)} rows)")
        else:
            print(f"No cached data at {csv_path}.  Downloading …")
            prices = download_adjusted_close_prices(args.tickers, start=args.start, end=args.end)
    else:
        prices = download_adjusted_close_prices(args.tickers, start=args.start, end=args.end)

    prices.to_csv(output_dir / "prices.csv")
    print(f"Data shape: {prices.shape}")
    print(f"Date range: {prices.index[0]} – {prices.index[-1]}")

    # ------------------------------------------------------------------
    # 2. Data quality
    # ------------------------------------------------------------------
    print_separator("2. Data quality assessment")
    quality = assess_all_tickers(prices)
    quality.to_csv(output_dir / "data_quality_report.csv", index=False)
    passed = quality["passed_quality_filter"].sum()
    print(f"Tickers passing quality filter: {passed}/{len(quality)}")
    for _, row in quality.iterrows():
        if not row["passed_quality_filter"]:
            print(f"  ⚠ {row['ticker']}: {row['rejection_reason']}")

    overlap = compute_all_pair_overlaps(prices)
    overlap.to_csv(output_dir / "pair_overlap_report.csv", index=False)
    print(f"Pair overlap report: {len(overlap)} pairs")

    # ------------------------------------------------------------------
    # 3. Baseline pair analysis
    # ------------------------------------------------------------------
    print_separator("3. Baseline pair analysis")
    if args.ticker_y not in prices.columns or args.ticker_x not in prices.columns:
        print(f"ERROR: Primary pair not found in data.  Available: {list(prices.columns)}")
        sys.exit(1)

    y_aligned, x_aligned = align_price_series(
        prices[args.ticker_y].astype(float),
        prices[args.ticker_x].astype(float),
    )

    train_frame, test_frame, _ = split_aligned_prices(
        pd.concat([y_aligned.rename(args.ticker_y), x_aligned.rename(args.ticker_x)], axis=1),
        train_ratio=0.7,
    )
    train_y = train_frame[args.ticker_y]
    train_x = train_frame[args.ticker_x]
    test_y = test_frame[args.ticker_y]
    test_x = test_frame[args.ticker_x]

    alpha, hedge_ratio, train_spread, test_spread, test_zscore = (
        estimate_train_test_relationship(train_y, train_x, test_y, test_x, lookback_window=60)
    )
    print(f"Hedge ratio: {hedge_ratio:.4f}")
    print(f"Alpha: {alpha:.6f}")

    # ------------------------------------------------------------------
    # 4. Baseline backtest
    # ------------------------------------------------------------------
    print_separator("4. Baseline backtest")
    signal_params = SignalParameters(
        lookback_window=60, entry_threshold=2.0, exit_threshold=0.5, stop_threshold=3.5,
    )
    signal_frame = create_signal_frame(test_spread, signal_params)

    bt_params = BacktestParameters(
        initial_capital=100_000.0,
        transaction_cost_bps=5.0,
        slippage_bps=2.0,
        annual_borrow_cost=0.02,
    )
    bt_frame = run_backtest(
        test_y, test_x, signal_frame["position"],
        hedge_ratio=hedge_ratio, parameters=bt_params,
    )
    summary = summarize_performance(bt_frame)
    n_entries = int(signal_frame["entry_flag"].sum())

    print(f"Total return: {summary.total_return:.2%}")
    print(f"Annualized return: {summary.annualized_return:.2%}")
    print(f"Sharpe ratio: {summary.sharpe_ratio:.3f}")
    print(f"Max drawdown: {summary.maximum_drawdown:.2%}")
    print(f"Volatility: {summary.annualized_volatility:.2%}")
    print(f"Number of entries: {n_entries}")

    # ------------------------------------------------------------------
    # 5. Trade-level analysis
    # ------------------------------------------------------------------
    print_separator("5. Trade-level analysis")
    trades = extract_trades(bt_frame, args.ticker_y, args.ticker_x, signal_frame=signal_frame)
    if trades:
        trade_summary = summarize_trades(trades)
        trade_df = trades_to_dataframe(trades)
        trade_df.to_csv(output_dir / "trade_log.csv", index=False)
        print(f"Total trades: {trade_summary.number_of_trades}")
        _wr = trade_summary.win_rate
        print(f"Win rate: {_wr:.1%}" if not np.isnan(_wr) else "  Win rate: N/A")
        _pf = trade_summary.profit_factor
        print(f"Profit factor: {_pf:.2f}" if not np.isnan(_pf) else "  Profit factor: N/A")
        _hd = trade_summary.average_holding_days
        print(f"Avg holding days: {_hd:.1f}" if not np.isnan(_hd) else "  Avg holding: N/A")
        print(f"Stop exits: {trade_summary.stop_exit_count}")
    else:
        print("No trades executed.")

    # ------------------------------------------------------------------
    # 6. Benchmark comparison
    # ------------------------------------------------------------------
    print_separator("6. Benchmark comparison")
    benchmark_df = compare_benchmarks(
        strategy_equity=bt_frame["equity"],
        strategy_returns=bt_frame["net_return"],
        initial_capital=100_000.0,
        ticker_y=args.ticker_y,
        ticker_x=args.ticker_x,
        price_y=test_y,
        price_x=test_x,
    )
    benchmark_df.to_csv(output_dir / "benchmark_comparison.csv", index=False)
    print(benchmark_df.to_string(index=False))

    # ------------------------------------------------------------------
    # 7. Cost sensitivity
    # ------------------------------------------------------------------
    print_separator("7. Cost sensitivity analysis")
    cost_result = run_cost_sensitivity(
        test_y, test_x, signal_frame["position"],
        hedge_ratio=hedge_ratio,
        initial_capital=100_000.0,
    )
    cost_result.to_csv(output_dir / "cost_sensitivity.csv", index=False)

    # Check if higher costs always reduce returns
    zero_slip = cost_result[cost_result["slippage_bps"] == 0.0]
    zero_borrow = zero_slip[zero_slip["annual_borrow_cost"] == 0.0]
    zero_slip_borrow = zero_borrow.sort_values("transaction_cost_bps")
    cost_monotonic = zero_slip_borrow["total_return"].is_monotonic_decreasing
    print(f"Returns monotonic with costs: {cost_monotonic}")

    # ------------------------------------------------------------------
    # 8. Parameter sensitivity
    # ------------------------------------------------------------------
    print_separator("8. Parameter sensitivity (diagnostic only)")
    param_result = run_parameter_sensitivity(
        train_y, train_x, test_y, test_x,
        initial_capital=100_000.0,
    )
    param_result.to_csv(output_dir / "parameter_sensitivity.csv", index=False)
    stability = parameter_stability_summary(param_result)
    print(f"Parameter combinations tested: {stability['n_combinations']}")
    print(f"Median Sharpe: {stability['median_sharpe']:.3f}")
    print(f"Profitable proportion: {stability['profitable_proportion']:.1%}")
    print(f"Sharpe range: {stability['sharpe_min']:.2f} – {stability['sharpe_max']:.2f}")
    print(f"Sharpe std: {stability['sharpe_std']:.3f}")

    # ------------------------------------------------------------------
    # 9. Universe screening with multiple-testing corrections
    # ------------------------------------------------------------------
    print_separator("9. Universe screening + multiple-testing corrections")
    screening_params = PairScreeningParameters(
        minimum_observations=200,
        maximum_cointegration_pvalue=0.05,
        maximum_adf_pvalue=0.05,
        minimum_half_life=10.0,
        maximum_half_life=250.0,
        minimum_price=1.0,
        maximum_missing_fraction=0.05,
        top_n_pairs=10,
        train_fraction=0.7,
        alpha=0.05,
    )
    screening_results = screen_pairs(prices, screening_params)
    if not screening_results.empty:
        screening_results.to_csv(output_dir / "multiple_testing.csv", index=False)

        # Column names added by screen_pairs
        # (which calls apply_multiple_testing_corrections internally)
        if "cointegration_significant_raw" in screening_results.columns:
            n_total = len(screening_results)
            raw_sig = int(screening_results["cointegration_significant_raw"].sum())
            bonf_sig = int(screening_results["cointegration_significant_bonferroni"].sum())
            bh_sig = int(screening_results["cointegration_significant_bh"].sum())
            print(f"Pairs tested: {n_total}")
            print(f"Raw significant: {raw_sig}")
            print(f"Bonferroni significant: {bonf_sig}")
            print(f"BH significant (FDR): {bh_sig}")
    else:
        print("No pairs passed screening filters.")

    # ------------------------------------------------------------------
    # 10. Summary
    # ------------------------------------------------------------------
    print_separator("Summary")
    print(f"All results saved to: {output_dir.resolve()}")
    print("\nKey findings:")
    if trades:
        print(f"  - {trade_summary.number_of_trades} trades, "
              f"win rate {trade_summary.win_rate:.1%}, "
              f"Sharpe {summary.sharpe_ratio:.3f}")
    print(f"  - Parameter stability: {stability['profitable_proportion']:.0%} "
          f"of {stability['n_combinations']} combinations profitable")
    print("\n⚠  Reminder: Parameter sensitivity is for diagnosis only.")
    print("   The baseline parameters (lookback=60, entry=2.0, exit=0.5, stop=3.5)")
    print("   remain the primary result of this project.")
    print(f"\nDone.  ({time.strftime('%Y-%m-%d %H:%M:%S')})")


if __name__ == "__main__":
    main()
