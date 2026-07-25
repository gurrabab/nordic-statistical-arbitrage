"""Streamlit dashboard for the Nordic Statistical Arbitrage project.

This dashboard reuses the analytical modules in ``src/`` and adds no
duplicate analytical logic.  See ``README.md`` for methodology details.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit command
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Nordic Statistical Arbitrage",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Import project modules
# ---------------------------------------------------------------------------

from src.backtester import BacktestParameters, run_backtest  # noqa: E402
from src.benchmarks import compare_benchmarks  # noqa: E402
from src.data_quality import assess_all_tickers, compute_all_pair_overlaps  # noqa: E402
from src.pair_screener import (  # noqa: E402
    PairScreeningParameters,
    apply_multiple_testing_corrections,
    download_ticker_universe,
    evaluate_top_pairs,
    screen_pairs,
)
from src.pair_selection import align_price_series, estimate_ols_regression  # noqa: E402
from src.risk_metrics import (  # noqa: E402
    PerformanceSummary,
    summarize_performance,
)
from src.sensitivity import (  # noqa: E402
    parameter_stability_summary,
    run_cost_sensitivity,
    run_parameter_sensitivity,
)
from src.signals import SignalParameters, create_signal_frame  # noqa: E402
from src.trade_analysis import extract_trades, summarize_trades, trades_to_dataframe  # noqa: E402
from src.universe import NORDIC_UNIVERSE  # noqa: E402
from src.validation import (  # noqa: E402
    estimate_train_test_relationship,
    split_aligned_prices,
)
from src.walk_forward import (  # noqa: E402
    WalkForwardParameters,
    calculate_walk_forward_summary,
    generate_walk_forward_windows,
    run_walk_forward_analysis,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TICKER_INFO: dict[str, str] = {
    "SEB-A.ST": "Skandinaviska Enskilda Banken AB",
    "SHB-A.ST": "Svenska Handelsbanken AB",
    "AZN.ST": "AstraZeneca PLC",
    "ERIC-B.ST": "Telefonaktiebolaget LM Ericsson",
    "INDU-C.ST": "Industrivärden AB",
    "SAND.ST": "Sandvik AB",
    "VOLV-B.ST": "Volvo AB",
    "HEXA-B.ST": "Hexagon AB",
    "TEL2-B.ST": "Tele2 AB",
    "SKF-B.ST": "SKF AB",
    "EQNR.OL": "Equinor ASA",
    "DNB.OL": "DNB Bank ASA",
    "ORK.OL": "Orkla ASA",
    "TGS.OL": "TGS ASA",
    "MOWI.OL": "Mowi ASA",
    "NOVO-B.CO": "Novo Nordisk A/S",
    "DSV.CO": "DSV A/S",
    "MAERSK-B.CO": "A.P. Møller-Mærsk A/S",
    "VWS.CO": "Vestas Wind Systems A/S",
    "CARL-B.CO": "Carlsberg A/S",
    "NESTE.HE": "Neste Oyj",
    "OUT1V.HE": "Outokumpu Oyj",
    "KNEBV.HE": "Kone Oyj",
    "WRT1V.HE": "Wärtsilä Oyj",
    "ELISA.HE": "Elisa Oyj",
    "SAMPO.HE": "Sampo Oyj",
    "UPM.HE": "UPM-Kymmene Oyj",
    "FORTUM.HE": "Fortum Oyj",
    "KEMIRA.HE": "Kemira Oyj",
    "NOKIA.HE": "Nokia Oyj",
}

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Cached data functions
# ---------------------------------------------------------------------------

DEFAULT_START = "2018-01-01"
DEFAULT_END = pd.Timestamp.today().strftime("%Y-%m-%d")


@st.cache_data(show_spinner="Downloading ticker universe …")
def cached_download(
    tickers: tuple[str, ...],
    start: str,
    end: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download price data with Streamlit caching."""
    return download_ticker_universe(list(tickers), start=start, end=end)


@st.cache_data(show_spinner="Screening pairs …")
def cached_screen_pairs(
    prices: pd.DataFrame,
    minimum_observations: int,
    maximum_cointegration_pvalue: float,
    maximum_adf_pvalue: float,
    minimum_half_life: float,
    maximum_half_life: float,
    minimum_price: float,
    maximum_missing_fraction: float,
    top_n_pairs: int,
    train_fraction: float,
) -> pd.DataFrame:
    """Run pair screening with caching.  Prices DataFrame is serialised
    via its hash — Streamlit handles this automatically for DataFrames."""
    params = PairScreeningParameters(
        minimum_observations=minimum_observations,
        maximum_cointegration_pvalue=maximum_cointegration_pvalue,
        maximum_adf_pvalue=maximum_adf_pvalue,
        minimum_half_life=minimum_half_life,
        maximum_half_life=maximum_half_life,
        minimum_price=minimum_price,
        maximum_missing_fraction=maximum_missing_fraction,
        top_n_pairs=top_n_pairs,
        train_fraction=train_fraction,
    )
    return screen_pairs(prices, params)


@st.cache_data(show_spinner="Running walk-forward validation …")
def cached_walk_forward(
    prices: pd.DataFrame,
    minimum_observations: int,
    maximum_cointegration_pvalue: float,
    maximum_adf_pvalue: float,
    minimum_half_life: float,
    maximum_half_life: float,
    minimum_price: float,
    maximum_missing_fraction: float,
    top_n_pairs: int,
    train_fraction: float,
    train_window_days: int,
    test_window_days: int,
    step_size_days: int,
    expanding_window: bool,
    minimum_train_observations: int,
    minimum_test_observations: int,
    top_n_pairs_per_window: int,
    lookback_window: int,
    entry_threshold: float,
    exit_threshold: float,
    stop_threshold: float,
    initial_capital: float,
    transaction_cost_bps: float,
    slippage_bps: float,
    annual_borrow_cost: float,
) -> pd.DataFrame:
    screening_params = PairScreeningParameters(
        minimum_observations=minimum_observations,
        maximum_cointegration_pvalue=maximum_cointegration_pvalue,
        maximum_adf_pvalue=maximum_adf_pvalue,
        minimum_half_life=minimum_half_life,
        maximum_half_life=maximum_half_life,
        minimum_price=minimum_price,
        maximum_missing_fraction=maximum_missing_fraction,
        top_n_pairs=top_n_pairs,
        train_fraction=train_fraction,
    )
    wf_params = WalkForwardParameters(
        train_window_days=train_window_days,
        test_window_days=test_window_days,
        step_size_days=step_size_days,
        expanding_window=expanding_window,
        minimum_train_observations=minimum_train_observations,
        minimum_test_observations=minimum_test_observations,
        top_n_pairs_per_window=top_n_pairs_per_window,
    )
    signal_params = SignalParameters(
        lookback_window=lookback_window,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        stop_threshold=stop_threshold,
    )
    bt_params = BacktestParameters(
        initial_capital=initial_capital,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
        annual_borrow_cost=annual_borrow_cost,
    )
    return run_walk_forward_analysis(
        prices, screening_params, wf_params, signal_params, bt_params,
    )


@st.cache_data(show_spinner="Computing pair analysis …")
def cached_pair_analysis(
    prices: pd.DataFrame,
    ticker_y: str,
    ticker_x: str,
) -> dict[str, Any]:
    """Compute alpha, hedge ratio, cointegration, ADF, half-life for one pair."""
    try:
        y_aligned, x_aligned = align_price_series(
            prices[ticker_y].astype(float), prices[ticker_x].astype(float),
        )
    except (ValueError, TypeError) as exc:
        return {"error": str(exc)}

    alpha, hedge_ratio, spread = estimate_ols_regression(y_aligned, x_aligned)

    # Cointegration test
    from statsmodels.tsa.stattools import coint, adfuller  # noqa: PLC0415
    import statsmodels.api as sm  # noqa: PLC0415

    coint_result = coint(y_aligned.astype(float), x_aligned.astype(float))
    cointegration_pvalue = float(coint_result[1])

    adf_stat, adf_pvalue, *_ = adfuller(spread.astype(float), autolag="AIC")

    # Half-life
    half_life: float | None = None
    if np.isfinite(hedge_ratio):
        delta_spread = spread.diff().dropna()
        lagged_spread = spread.shift(1).dropna()
        reg = pd.DataFrame({"delta": delta_spread, "lag": lagged_spread}).dropna()
        if not reg.empty:
            design = sm.add_constant(reg["lag"], has_constant="add")
            model = sm.OLS(reg["delta"], design).fit()
            slope = float(model.params.iloc[1])
            if np.isfinite(slope) and slope < 0:
                half_life = float(-np.log(2.0) / slope)

    # Return correlation
    y_ret = y_aligned.astype(float).pct_change().dropna()
    x_ret = x_aligned.astype(float).pct_change().dropna()
    ret_corr = np.nan
    aligned_ret = pd.concat(
        [y_ret.rename("y"), x_ret.rename("x")], axis=1,
    ).dropna()
    if len(aligned_ret) > 2 and aligned_ret["y"].nunique() > 1 and aligned_ret["x"].nunique() > 1:
        ret_corr = float(np.corrcoef(aligned_ret["y"], aligned_ret["x"])[0, 1])

    return {
        "alpha": alpha,
        "hedge_ratio": hedge_ratio,
        "spread": spread,
        "cointegration_pvalue": cointegration_pvalue,
        "adf_pvalue": adf_pvalue,
        "half_life": half_life,
        "return_correlation": ret_corr,
        "n_observations": len(y_aligned),
        "y_aligned": y_aligned,
        "x_aligned": x_aligned,
    }


@st.cache_data(show_spinner="Running backtest …")
def cached_backtest(
    prices: pd.DataFrame,
    ticker_y: str,
    ticker_x: str,
    hedge_ratio: float,
    lookback_window: int,
    entry_threshold: float,
    exit_threshold: float,
    stop_threshold: float,
    initial_capital: float,
    transaction_cost_bps: float,
    slippage_bps: float,
    annual_borrow_cost: float,
    train_fraction: float,
) -> dict[str, Any]:
    """Run train/test split, estimate relationship, backtest test period."""
    try:
        y_aligned, x_aligned = align_price_series(
            prices[ticker_y].astype(float), prices[ticker_x].astype(float),
        )
    except (ValueError, TypeError) as exc:
        return {"error": str(exc)}

    pair_frame = pd.concat(
        [y_aligned.rename(ticker_y), x_aligned.rename(ticker_x)], axis=1,
    )

    try:
        train_frame, test_frame, split = split_aligned_prices(
            pair_frame,
            train_ratio=train_fraction,
            min_train_observations=max(60, len(pair_frame) // 4),
            min_test_observations=max(30, len(pair_frame) // 8),
        )
    except ValueError as exc:
        return {"error": str(exc)}

    train_y = train_frame[ticker_y]
    train_x = train_frame[ticker_x]
    test_y = test_frame[ticker_y]
    test_x = test_frame[ticker_x]

    signal_params = SignalParameters(
        lookback_window=lookback_window,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        stop_threshold=stop_threshold,
    )
    bt_params = BacktestParameters(
        initial_capital=initial_capital,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
        annual_borrow_cost=annual_borrow_cost,
    )

    alpha, hr, train_spread, test_spread, test_zscore = (
        estimate_train_test_relationship(
            train_y, train_x, test_y, test_x, lookback_window,
        )
    )

    signal_frame = create_signal_frame(test_spread, signal_params)
    bt_frame = run_backtest(
        test_y, test_x, signal_frame["position"],
        hedge_ratio=hr, parameters=bt_params,
    )

    summary = summarize_performance(bt_frame)

    # Full-period signal frame for charts
    full_signal = create_signal_frame(
        pd.concat([train_spread, test_spread]).sort_index(), signal_params,
    )

    return {
        "alpha": alpha,
        "hedge_ratio": hr,
        "train_spread": train_spread,
        "test_spread": test_spread,
        "test_zscore": test_zscore,
        "signal_frame": signal_frame,
        "full_signal_frame": full_signal,
        "backtest_frame": bt_frame,
        "summary": summary,
        "train_frame": train_frame,
        "test_frame": test_frame,
        "split": split,
        "train_prices": pd.concat(
            [train_y.rename(ticker_y), train_x.rename(ticker_x)], axis=1,
        ),
        "test_prices": pd.concat(
            [test_y.rename(ticker_y), test_x.rename(ticker_x)], axis=1,
        ),
    }


# ---------------------------------------------------------------------------
# Plotting helpers (return matplotlib figures)
# ---------------------------------------------------------------------------

STYLE_COLORS = {
    "positive": "#2ecc71",
    "negative": "#e74c3c",
    "neutral": "#95a5a6",
    "primary": "#3498db",
    "secondary": "#9b59b6",
    "text": "#2c3e50",
}


def _style_ax(ax: plt.Axes, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, fontsize=11, fontweight="semibold", color=STYLE_COLORS["text"])
    ax.set_xlabel(xlabel, fontsize=9, color=STYLE_COLORS["text"])
    ax.set_ylabel(ylabel, fontsize=9, color=STYLE_COLORS["text"])
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(True, alpha=0.3)


def plot_normalized_prices(
    test_prices: pd.DataFrame,
    train_prices: pd.DataFrame,
    ticker_y: str,
    ticker_x: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 3.5))
    for col, label, color in [
        (ticker_y, f"{ticker_y} (Y)", STYLE_COLORS["primary"]),
        (ticker_x, f"{ticker_x} (X)", STYLE_COLORS["secondary"]),
    ]:
        full = pd.concat(
            [train_prices[col], test_prices[col]],
        ).sort_index()
        normalized = full / full.iloc[0]
        ax.plot(normalized.index, normalized, label=label, color=color, linewidth=1)
    ax.axvline(
        x=test_prices.index[0], color="gray", linestyle="--", linewidth=0.7,
        label="train/test split",
    )
    _style_ax(ax, "Normalised prices (base = 1.0)", ylabel="Price ratio")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_spread(spread: pd.Series, half_life: float | None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.plot(spread.index, spread.values, color=STYLE_COLORS["primary"], linewidth=0.8)
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    _style_ax(ax, "Residual spread", ylabel="Spread")
    if half_life is not None:
        ax.text(
            0.02, 0.95, f"Half-life: {half_life:.1f} days",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7),
        )
    fig.tight_layout()
    return fig


def plot_zscore(
    signal_frame: pd.DataFrame,
    entry_threshold: float,
    exit_threshold: float,
    stop_threshold: float,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 2.5))
    zscore = signal_frame["zscore"]
    ax.plot(zscore.index, zscore.values, color=STYLE_COLORS["primary"], linewidth=0.8)
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    ax.axhline(y=entry_threshold, color=STYLE_COLORS["positive"], linestyle="--", linewidth=0.7, label=f"entry (±{entry_threshold})")
    ax.axhline(y=-entry_threshold, color=STYLE_COLORS["positive"], linestyle="--", linewidth=0.7)
    ax.axhline(y=exit_threshold, color=STYLE_COLORS["neutral"], linestyle=":", linewidth=0.7, label=f"exit (±{exit_threshold})")
    ax.axhline(y=-exit_threshold, color=STYLE_COLORS["neutral"], linestyle=":", linewidth=0.7)
    ax.axhline(y=stop_threshold, color=STYLE_COLORS["negative"], linestyle="--", linewidth=0.7, label=f"stop (±{stop_threshold})")
    ax.axhline(y=-stop_threshold, color=STYLE_COLORS["negative"], linestyle="--", linewidth=0.7)
    _style_ax(ax, "Z-score with signal thresholds", ylabel="Z-score")
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    return fig


def plot_positions(signal_frame: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 1.8))
    ax.fill_between(
        signal_frame.index, 0, signal_frame["position"],
        where=signal_frame["position"] == 1,
        color=STYLE_COLORS["positive"], alpha=0.6, label="long spread",
    )
    ax.fill_between(
        signal_frame.index, 0, signal_frame["position"],
        where=signal_frame["position"] == -1,
        color=STYLE_COLORS["negative"], alpha=0.6, label="short spread",
    )
    ax.set_ylim(-1.5, 1.5)
    ax.set_yticks([-1, 0, 1])
    _style_ax(ax, "Executed position", ylabel="Position")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


def plot_equity_curve(bt_frame: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.plot(
        bt_frame.index, bt_frame["equity"],
        color=STYLE_COLORS["primary"], linewidth=1,
    )
    _style_ax(ax, "Equity curve", ylabel="Equity")
    fig.tight_layout()
    return fig


def plot_drawdown(bt_frame: pd.DataFrame) -> plt.Figure:
    equity = bt_frame["equity"].astype(float)
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.fill_between(drawdown.index, 0, drawdown.values, color=STYLE_COLORS["negative"], alpha=0.4)
    ax.plot(drawdown.index, drawdown.values, color=STYLE_COLORS["negative"], linewidth=0.7)
    _style_ax(ax, "Drawdown", ylabel="Drawdown")
    fig.tight_layout()
    return fig


def plot_return_histogram(bt_frame: pd.DataFrame) -> plt.Figure:
    returns = bt_frame["net_return"].astype(float)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(returns, bins=60, color=STYLE_COLORS["primary"], alpha=0.7, edgecolor="white", linewidth=0.3)
    ax.axvline(x=0, color=STYLE_COLORS["negative"], linestyle="--", linewidth=0.8)
    _style_ax(ax, "Daily return distribution", xlabel="Daily return", ylabel="Frequency")
    fig.tight_layout()
    return fig


def plot_walk_forward_returns(detailed: pd.DataFrame) -> plt.Figure:
    valid = detailed.dropna(subset=["test_total_return"])
    if valid.empty:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, "No walk-forward data", ha="center", va="center", transform=ax.transAxes)
        return fig
    fig, ax = plt.subplots(figsize=(10, 3.5))
    for (ty, tx), grp in valid.groupby(["ticker_y", "ticker_x"]):
        grp = grp.sort_values("window_id")
        ax.plot(
            grp["window_id"], grp["test_total_return"],
            marker="o", linestyle="-", label=f"{ty} / {tx}", linewidth=1, markersize=4,
        )
    _style_ax(ax, "Walk-forward: test return by window", xlabel="Window ID", ylabel="Test total return")
    ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    return fig


def plot_walk_forward_sharpes(detailed: pd.DataFrame) -> plt.Figure:
    valid = detailed.dropna(subset=["test_sharpe_ratio"])
    if valid.empty:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, "No walk-forward data", ha="center", va="center", transform=ax.transAxes)
        return fig
    fig, ax = plt.subplots(figsize=(10, 3.5))
    for (ty, tx), grp in valid.groupby(["ticker_y", "ticker_x"]):
        grp = grp.sort_values("window_id")
        ax.plot(
            grp["window_id"], grp["test_sharpe_ratio"],
            marker="s", linestyle="-", label=f"{ty} / {tx}", linewidth=1, markersize=4,
        )
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.7)
    _style_ax(ax, "Walk-forward: Sharpe ratio by window", xlabel="Window ID", ylabel="Test Sharpe ratio")
    ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Metric card helper
# ---------------------------------------------------------------------------


def metric_card(
    label: str,
    value: str,
    delta: str | None = None,
    help_text: str | None = None,
) -> None:
    """Render a styled metric card using native Streamlit columns."""
    col = st.columns(1)[0]
    with col:
        st.metric(label=label, value=value, delta=delta, help=help_text)


# ---------------------------------------------------------------------------
# UI — Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("Nordic Statistical Arbitrage")
st.sidebar.markdown("---")

# --- Ticker selection ---
st.sidebar.subheader("Ticker universe")
selected_universe = st.sidebar.multiselect(
    "Select tickers",
    options=sorted(NORDIC_UNIVERSE),
    default=NORDIC_UNIVERSE,
    help="Pairs will be screened from this universe.",
)
if not selected_universe:
    st.sidebar.warning("Select at least 2 tickers.")

st.sidebar.markdown("---")
st.sidebar.subheader("Pair selection")
ticker_y = st.sidebar.selectbox("Ticker Y (dependent)", options=sorted(selected_universe) if selected_universe else [""], index=0)
ticker_x = st.sidebar.selectbox(
    "Ticker X (independent)",
    options=sorted([t for t in (selected_universe or [""]) if t != ticker_y]),
    index=min(1, len(selected_universe) - 1) if len(selected_universe or []) > 1 else 0,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Date range")
start_date = st.sidebar.text_input("Start date", value=DEFAULT_START)
end_date = st.sidebar.text_input("End date", value=DEFAULT_END)

st.sidebar.markdown("---")
st.sidebar.subheader("Signal parameters")
lookback_window = st.sidebar.number_input("Lookback window (days)", min_value=5, value=60, step=5)
entry_threshold = st.sidebar.number_input("Entry threshold (z-score)", min_value=0.5, value=2.0, step=0.1)
exit_threshold = st.sidebar.number_input("Exit threshold (z-score)", min_value=0.0, value=0.5, step=0.1)
stop_threshold = st.sidebar.number_input("Stop threshold (z-score)", min_value=entry_threshold + 0.1, value=max(entry_threshold + 0.1, 3.5), step=0.1)

st.sidebar.markdown("---")
st.sidebar.subheader("Backtest parameters")
initial_capital = st.sidebar.number_input("Initial capital", min_value=1000.0, value=100000.0, step=50000.0, format="%.0f")
transaction_cost_bps = st.sidebar.number_input("Transaction cost (bps)", min_value=0.0, value=5.0, step=1.0)
slippage_bps = st.sidebar.number_input("Slippage (bps)", min_value=0.0, value=2.0, step=1.0)
annual_borrow_cost = st.sidebar.number_input("Annual borrow cost", min_value=0.0, value=0.02, step=0.01, format="%.2f")

st.sidebar.markdown("---")
st.sidebar.subheader("Screening parameters")
train_fraction = st.sidebar.slider("Train fraction", min_value=0.5, max_value=0.95, value=0.7, step=0.05)
screen_min_obs = st.sidebar.number_input("Minimum observations", min_value=60, value=200, step=20)
screen_max_coint_p = st.sidebar.number_input("Max cointegration p-value", min_value=0.001, max_value=1.0, value=0.05, step=0.01, format="%.3f")
screen_max_adf_p = st.sidebar.number_input("Max ADF p-value", min_value=0.001, max_value=1.0, value=0.05, step=0.01, format="%.3f")
screen_min_hl = st.sidebar.number_input("Min half-life (days)", min_value=1.0, value=10.0, step=5.0)
screen_max_hl = st.sidebar.number_input("Max half-life (days)", min_value=screen_min_hl + 1.0, value=250.0, step=10.0)
screen_min_price = st.sidebar.number_input("Minimum price", min_value=0.1, value=1.0, step=1.0)
screen_max_missing = st.sidebar.slider("Max missing fraction", min_value=0.0, max_value=0.5, value=0.05, step=0.05)
screen_top_n = st.sidebar.number_input("Top N pairs (screening)", min_value=1, value=10, step=1)

st.sidebar.markdown("---")
st.sidebar.subheader("Walk-forward settings")
wf_train_days = st.sidebar.number_input("Train window (days)", min_value=60, value=504, step=21)
wf_test_days = st.sidebar.number_input("Test window (days)", min_value=10, value=63, step=7)
wf_step = st.sidebar.number_input("Step size (days)", min_value=1, value=63, step=7)
wf_expanding = st.sidebar.checkbox("Expanding window", value=False)
wf_min_train_obs = st.sidebar.number_input("Min train observations", min_value=30, value=200, step=10)
wf_min_test_obs = st.sidebar.number_input("Min test observations", min_value=5, value=30, step=5)
wf_top_n = st.sidebar.number_input("Top N pairs per window", min_value=1, value=3, step=1)

st.sidebar.markdown("---")
run_button = st.sidebar.button("Run analysis", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# UI — Main panel
# ---------------------------------------------------------------------------

st.title("Nordic Statistical Arbitrage")
st.markdown(
    "A quantitative research pipeline for pairs-trading among Nordic equities. "
    "All methodology is documented in the README."
)

if not run_button:
    st.info("Configure parameters in the sidebar and click **Run analysis**.")
    st.stop()

# =========================================================================
# RUN ANALYSIS
# =========================================================================

with st.status("Running analysis …", expanded=True) as status:
    # --- Step 1: Download ---
    st.write("📥 Downloading ticker universe …")
    try:
        prices, download_report = cached_download(
            tuple(selected_universe), start_date, end_date,
        )
    except Exception as exc:
        st.error(f"Download failed: {exc}")
        st.stop()

    failed_downloads = download_report[download_report["status"] != "downloaded"]
    if not failed_downloads.empty:
        st.warning(f"{len(failed_downloads)} ticker(s) failed to download.")
        with st.expander("Show download failures"):
            st.dataframe(failed_downloads, use_container_width=True)

    if prices.empty:
        st.error("No price data available.  Check ticker symbols and date range.")
        st.stop()

    # --- Step 2: Validate pair ---
    st.write("🔍 Validating pair …")
    if ticker_y not in prices.columns or ticker_x not in prices.columns:
        st.error(f"Selected tickers not available in downloaded data.")
        st.stop()

    pair_result = cached_pair_analysis(prices, ticker_y, ticker_x)
    if "error" in pair_result:
        st.warning(f"Pair analysis error: {pair_result['error']}")
        # Still continue — may have screening results for other pairs

    # --- Step 3: Screen pairs ---
    st.write("📋 Screening pairs …")
    try:
        screening_results = cached_screen_pairs(
            prices,
            minimum_observations=screen_min_obs,
            maximum_cointegration_pvalue=screen_max_coint_p,
            maximum_adf_pvalue=screen_max_adf_p,
            minimum_half_life=screen_min_hl,
            maximum_half_life=screen_max_hl,
            minimum_price=screen_min_price,
            maximum_missing_fraction=screen_max_missing,
            top_n_pairs=screen_top_n,
            train_fraction=train_fraction,
        )
    except Exception as exc:
        st.error(f"Screening error: {exc}")
        screening_results = pd.DataFrame()

    # --- Step 4: Run backtest for selected pair ---
    st.write("📈 Running backtest …")
    bt_result = cached_backtest(
        prices, ticker_y, ticker_x,
        hedge_ratio=pair_result.get("hedge_ratio", 1.0),
        lookback_window=lookback_window,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        stop_threshold=stop_threshold,
        initial_capital=initial_capital,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
        annual_borrow_cost=annual_borrow_cost,
        train_fraction=train_fraction,
    )

    # --- Step 5: Evaluate top pairs OOS ---
    st.write("📊 Evaluating top pairs out-of-sample …")
    oos_results = pd.DataFrame()
    if not screening_results.empty:
        try:
            oos_params = PairScreeningParameters(
                minimum_observations=screen_min_obs,
                maximum_cointegration_pvalue=screen_max_coint_p,
                maximum_adf_pvalue=screen_max_adf_p,
                minimum_half_life=screen_min_hl,
                maximum_half_life=screen_max_hl,
                minimum_price=screen_min_price,
                maximum_missing_fraction=screen_max_missing,
                top_n_pairs=screen_top_n,
                train_fraction=train_fraction,
            )
            oos_results = evaluate_top_pairs(
                screening_results, prices, oos_params, top_n=5,
            )
        except Exception as exc:
            st.warning(f"Out-of-sample evaluation error: {exc}")

    # --- Step 6: Walk-forward validation ---
    st.write("🔄 Running walk-forward validation …")
    wf_detailed = pd.DataFrame()
    wf_summary = pd.DataFrame()
    try:
        wf_detailed = cached_walk_forward(
            prices,
            minimum_observations=screen_min_obs,
            maximum_cointegration_pvalue=screen_max_coint_p,
            maximum_adf_pvalue=screen_max_adf_p,
            minimum_half_life=screen_min_hl,
            maximum_half_life=screen_max_hl,
            minimum_price=screen_min_price,
            maximum_missing_fraction=screen_max_missing,
            top_n_pairs=screen_top_n,
            train_fraction=train_fraction,
            train_window_days=wf_train_days,
            test_window_days=wf_test_days,
            step_size_days=wf_step,
            expanding_window=wf_expanding,
            minimum_train_observations=wf_min_train_obs,
            minimum_test_observations=wf_min_test_obs,
            top_n_pairs_per_window=wf_top_n,
            lookback_window=lookback_window,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            stop_threshold=stop_threshold,
            initial_capital=initial_capital,
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
            annual_borrow_cost=annual_borrow_cost,
        )
        if not wf_detailed.empty:
            wf_summary = calculate_walk_forward_summary(wf_detailed)
    except Exception as exc:
        st.warning(f"Walk-forward error: {exc}")

    status.update(label="Analysis complete", state="complete")

# =========================================================================
# DISPLAY RESULTS
# =========================================================================

TABS = st.tabs([
    "Overview",
    "Statistical analysis",
    "Performance",
    "Charts",
    "Tables",
    "Trade analysis",
    "Benchmarks",
    "Cost sensitivity",
    "Parameter robustness",
    "Multiple testing",
    "Data quality",
])

# -------------------------------------------------------------------------
# TAB 1 — Overview
# -------------------------------------------------------------------------

with TABS[0]:
    st.subheader("Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        ticker_y_name = TICKER_INFO.get(ticker_y, ticker_y)
        ticker_x_name = TICKER_INFO.get(ticker_x, ticker_x)
        st.metric("Selected pair", f"{ticker_y} / {ticker_x}")
        st.caption(f"{ticker_y_name} / {ticker_x_name}")
    with col2:
        st.metric("Data period", f"{prices.index[0].strftime('%Y-%m-%d')} – {prices.index[-1].strftime('%Y-%m-%d')}")
        st.caption(f"{len(prices)} trading days")
    with col3:
        n_screened = len(screening_results) if not screening_results.empty else 0
        total_possible = max(0, len(selected_universe) * (len(selected_universe) - 1) // 2)
        st.metric("Screened pairs", f"{n_screened} / {total_possible}")
        st.caption("Passing filters")
    with col4:
        if "error" not in bt_result:
            smry = bt_result["summary"]
            st.metric("Test Sharpe ratio", f"{smry.sharpe_ratio:.3f}")
            st.caption("Out-of-sample")

    if "error" in bt_result:
        st.warning(f"Backtest could not run for the selected pair: {bt_result['error']}")
    else:
        # Current signal
        signal_frame = bt_result["signal_frame"]
        latest_signal = signal_frame.iloc[-1]
        latest_date = signal_frame.index[-1].strftime("%Y-%m-%d")

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            pos_map = {1: "🟢 Long spread", -1: "🔴 Short spread", 0: "⚪ Flat"}
            pos_text = pos_map.get(int(latest_signal["position"]), "Unknown")
            st.metric("Latest signal", pos_text, help=f"Position as of {latest_date}")
        with col2:
            st.metric("Latest z-score", f"{latest_signal['zscore']:.3f}" if pd.notna(latest_signal["zscore"]) else "N/A")
        with col3:
            n_entries = int((signal_frame["entry_flag"] == True).sum())
            st.metric("Total entries (test)", str(n_entries))

# -------------------------------------------------------------------------
# TAB 2 — Statistical analysis
# -------------------------------------------------------------------------

with TABS[1]:
    st.subheader("Statistical analysis")

    if "error" in pair_result:
        st.warning(f"Could not compute statistics: {pair_result['error']}")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Hedge ratio (β)", f"{pair_result['hedge_ratio']:.4f}")
            st.caption("OLS slope: Y = α + βX + ε")
        with col2:
            st.metric("Alpha (α)", f"{pair_result['alpha']:.4f}")
            st.caption("OLS intercept")
        with col3:
            corr = pair_result.get("return_correlation")
            if corr is not None and np.isfinite(corr):
                st.metric("Return correlation", f"{corr:.4f}")
            else:
                st.metric("Return correlation", "N/A")
            st.caption("Daily return Pearson ρ")

        col1, col2, col3 = st.columns(3)
        with col1:
            cpval = pair_result["cointegration_pvalue"]
            cpval_str = f"{cpval:.6f}" if cpval > 1e-6 else f"{cpval:.2e}"
            st.metric("Cointegration p-value", cpval_str)
            if cpval > 0.05:
                st.warning("⚠ Weak cointegration evidence", icon="⚠️")
        with col2:
            apval = pair_result["adf_pvalue"]
            apval_str = f"{apval:.6f}" if apval > 1e-6 else f"{apval:.2e}"
            st.metric("ADF p-value", apval_str)
            if apval > 0.05:
                st.warning("⚠ Spread may not be stationary", icon="⚠️")
        with col3:
            hl = pair_result.get("half_life")
            if hl is not None:
                st.metric("Half-life (days)", f"{hl:.1f}")
            else:
                st.metric("Half-life (days)", "∞ (non-reverting)")
                st.warning("⚠ Spread not mean-reverting", icon="⚠️")

        st.metric("Aligned observations", str(pair_result["n_observations"]))

        # Warning if statistical evidence is weak
        cpval = pair_result["cointegration_pvalue"]
        apval = pair_result["adf_pvalue"]
        hl = pair_result.get("half_life")
        warnings_list = []
        if cpval > 0.05:
            warnings_list.append(f"Cointegration p-value ({cpval:.4f}) exceeds 0.05 — null hypothesis of no cointegration cannot be rejected.")
        if apval > 0.05:
            warnings_list.append(f"ADF p-value ({apval:.4f}) exceeds 0.05 — spread may have a unit root.")
        if hl is None:
            warnings_list.append("Half-life could not be estimated — spread may not be mean-reverting.")

        if warnings_list:
            st.markdown("---")
            st.error("⚠️ **Statistical concerns**")
            for w in warnings_list:
                st.write(f"- {w}")

# -------------------------------------------------------------------------
# TAB 3 — Performance
# -------------------------------------------------------------------------

with TABS[2]:
    st.subheader("Performance metrics")

    if "error" in bt_result:
        st.warning(f"No performance data: {bt_result['error']}")
    else:
        smry: PerformanceSummary = bt_result["summary"]
        bt_frame = bt_result["backtest_frame"]

        # Row 1
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Final equity", f"${smry.total_return * initial_capital + initial_capital:,.0f}")
        with col2:
            st.metric("Total return", f"{smry.total_return * 100:.2f}%")
        with col3:
            st.metric("Annualized return", f"{smry.annualized_return * 100:.2f}%")
        with col4:
            st.metric("Annualized volatility", f"{smry.annualized_volatility * 100:.2f}%")

        # Row 2
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Sharpe ratio", f"{smry.sharpe_ratio:.3f}")
        with col2:
            st.metric("Sortino ratio", f"{smry.sortino_ratio:.3f}")
        with col3:
            st.metric("Maximum drawdown", f"{smry.maximum_drawdown * 100:.2f}%")
        with col4:
            st.metric("Calmar ratio", f"{smry.calmar_ratio:.3f}")

        # Row 3
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("VaR (95%)", f"{smry.value_at_risk_95 * 100:.3f}%")
        with col2:
            st.metric("Expected shortfall (95%)", f"{smry.expected_shortfall_95 * 100:.3f}%")
        with col3:
            st.metric("Hit rate", f"{smry.hit_rate * 100:.1f}%")
        with col4:
            n_days = smry.number_of_trading_days
            st.metric("Trading days", str(n_days))

# -------------------------------------------------------------------------
# TAB 4 — Charts
# -------------------------------------------------------------------------

with TABS[3]:
    st.subheader("Charts")

    if "error" in bt_result:
        st.warning("No charts available — backtest could not run.")
    else:
        chart_tabs = st.tabs([
            "Prices", "Spread", "Z-score", "Positions",
            "Equity", "Drawdown", "Returns", "WF Returns", "WF Sharpe",
        ])

        with chart_tabs[0]:
            fig = plot_normalized_prices(
                bt_result["test_prices"], bt_result["train_prices"],
                ticker_y, ticker_x,
            )
            st.pyplot(fig)

        with chart_tabs[1]:
            spread = pd.concat([
                bt_result["train_spread"],
                bt_result["test_spread"],
            ]).sort_index()
            fig = plot_spread(spread, pair_result.get("half_life"))
            st.pyplot(fig)

        with chart_tabs[2]:
            full_signal = bt_result["full_signal_frame"]
            fig = plot_zscore(full_signal, entry_threshold, exit_threshold, stop_threshold)
            st.pyplot(fig)

        with chart_tabs[3]:
            fig = plot_positions(bt_result["signal_frame"])
            st.pyplot(fig)

        with chart_tabs[4]:
            fig = plot_equity_curve(bt_result["backtest_frame"])
            st.pyplot(fig)

        with chart_tabs[5]:
            fig = plot_drawdown(bt_result["backtest_frame"])
            st.pyplot(fig)

        with chart_tabs[6]:
            fig = plot_return_histogram(bt_result["backtest_frame"])
            st.pyplot(fig)

        with chart_tabs[7]:
            fig = plot_walk_forward_returns(wf_detailed)
            st.pyplot(fig)

        with chart_tabs[8]:
            fig = plot_walk_forward_sharpes(wf_detailed)
            st.pyplot(fig)

# -------------------------------------------------------------------------
# TAB 5 — Tables
# -------------------------------------------------------------------------

with TABS[4]:
    st.subheader("Tables")

    table_tabs = st.tabs([
        "Signals",
        "Backtest",
        "Screened pairs",
        "OOS comparison",
        "WF results",
        "WF summary",
    ])

    with table_tabs[0]:
        if "error" not in bt_result:
            display_sig = bt_result["signal_frame"][["spread", "zscore", "position", "entry_flag", "exit_flag"]].tail(20)
            st.dataframe(display_sig, use_container_width=True)
            csv = display_sig.to_csv().encode("utf-8")
            st.download_button("📥 Download CSV", csv, "latest_signals.csv", mime="text/csv")
        else:
            st.info("No signal data.")

    with table_tabs[1]:
        if "error" not in bt_result:
            cols = ["equity", "cumulative_return", "net_return", "signal_position", "gross_return", "turnover", "transaction_cost"]
            disp_cols = [c for c in cols if c in bt_result["backtest_frame"].columns]
            display_bt = bt_result["backtest_frame"][disp_cols]
            st.dataframe(display_bt, use_container_width=True)
            csv = display_bt.to_csv().encode("utf-8")
            st.download_button("📥 Download CSV", csv, "backtest_output.csv", mime="text/csv")
        else:
            st.info("No backtest data.")

    with table_tabs[2]:
        if not screening_results.empty:
            display_cols = ["rank", "ticker_y", "ticker_x", "score", "cointegration_pvalue", "adf_pvalue", "half_life", "hedge_ratio", "passes_filters"]
            display_cols = [c for c in display_cols if c in screening_results.columns]
            st.dataframe(screening_results[display_cols], use_container_width=True)
            csv = screening_results.to_csv().encode("utf-8")
            st.download_button("📥 Download CSV", csv, "pair_screening_results.csv", mime="text/csv")
        else:
            st.info("No pairs passed the screening filters.  Try relaxing thresholds.")

    with table_tabs[3]:
        if not oos_results.empty:
            st.dataframe(oos_results, use_container_width=True)
            csv = oos_results.to_csv().encode("utf-8")
            st.download_button("📥 Download CSV", csv, "top_pairs_out_of_sample.csv", mime="text/csv")
        else:
            st.info("No out-of-sample comparison available.")

    with table_tabs[4]:
        if not wf_detailed.empty:
            st.dataframe(wf_detailed, use_container_width=True)
            csv = wf_detailed.to_csv().encode("utf-8")
            st.download_button("📥 Download CSV", csv, "walk_forward_window_results.csv", mime="text/csv")
        else:
            st.info("No walk-forward results available.")

    with table_tabs[5]:
        if not wf_summary.empty:
            st.dataframe(wf_summary, use_container_width=True)
            csv = wf_summary.to_csv().encode("utf-8")
            st.download_button("📥 Download CSV", csv, "walk_forward_pair_summary.csv", mime="text/csv")
        else:
            st.info("No walk-forward summary available.")

# -------------------------------------------------------------------------
# TAB 6 — Trade analysis
# -------------------------------------------------------------------------

with TABS[5]:
    st.subheader("Trade analysis")

    if "error" in bt_result:
        st.warning("No trade data available (backtest did not run).")
    else:
        bt_frame = bt_result["backtest_frame"]

        # Extract trades
        trades = extract_trades(bt_frame, ticker_y, ticker_x, signal_frame=bt_result.get("signal_frame"))
        if trades:
            summary = summarize_trades(trades)
            trade_df = trades_to_dataframe(trades)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total trades", str(summary.number_of_trades))
            col2.metric("Win rate", f"{summary.win_rate:.1%}" if not np.isnan(summary.win_rate) else "N/A")
            col3.metric("Profit factor", f"{summary.profit_factor:.2f}" if not np.isnan(summary.profit_factor) else "N/A")
            col4.metric("Avg holding", f"{summary.average_holding_days:.1f}d" if not np.isnan(summary.average_holding_days) else "N/A")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Avg winner", f"{summary.average_winner:.2%}" if not np.isnan(summary.average_winner) else "N/A")
            col2.metric("Avg loser", f"{summary.average_loser:.2%}" if not np.isnan(summary.average_loser) else "N/A")
            col3.metric("Stop exits", str(summary.stop_exit_count))
            col4.metric("Normal exits", str(summary.normal_exit_count))

            st.markdown("### Trade log")
            st.dataframe(trade_df, use_container_width=True)
            csv = trade_df.to_csv().encode("utf-8")
            st.download_button("📥 Download trade log", csv, "trade_log.csv", mime="text/csv")
        else:
            st.info("No trades were executed during the test period.")


# -------------------------------------------------------------------------
# TAB 7 — Benchmarks
# -------------------------------------------------------------------------

with TABS[6]:
    st.subheader("Benchmark comparison")

    if "error" in bt_result:
        st.warning("No benchmark data (backtest did not run).")
    else:
        bt_frame = bt_result["backtest_frame"]
        test_prices = bt_result["test_prices"]

        st.info(
            "An equity index is **not** a perfect benchmark for a "
            "market-neutral pairs-trading strategy.  These comparisons "
            "provide context, not a fully matched risk comparison."
        )

        benchmark_df = compare_benchmarks(
            strategy_equity=bt_frame["equity"],
            strategy_returns=bt_frame["net_return"],
            initial_capital=initial_capital,
            ticker_y=ticker_y,
            ticker_x=ticker_x,
            price_y=test_prices[ticker_y],
            price_x=test_prices[ticker_x],
            cash_rate=0.0,
        )

        st.dataframe(benchmark_df, use_container_width=True)

        # Highlight strategy vs best benchmark
        non_strategy = benchmark_df[benchmark_df["benchmark_name"] != "Strategy (pairs trading)"]
        if not non_strategy.empty:
            best_sharpe_bench = non_strategy.loc[non_strategy["sharpe_ratio"].idxmax()]
            strategy_row = benchmark_df[benchmark_df["benchmark_name"] == "Strategy (pairs trading)"]
            if not strategy_row.empty:
                strat_sharpe = strategy_row.iloc[0]["sharpe_ratio"]
                st.metric(
                    "Strategy Sharpe vs best benchmark",
                    f"{strat_sharpe:.3f} vs {best_sharpe_bench['sharpe_ratio']:.3f}",
                )

        csv = benchmark_df.to_csv().encode("utf-8")
        st.download_button("📥 Download benchmark comparison", csv, "benchmark_comparison.csv", mime="text/csv")


# -------------------------------------------------------------------------
# TAB 8 — Cost sensitivity
# -------------------------------------------------------------------------

with TABS[7]:
    st.subheader("Cost sensitivity analysis")

    if "error" in bt_result:
        st.warning("No cost sensitivity data (backtest did not run).")
    else:
        test_prices = bt_result["test_prices"]
        signal_frame = bt_result["signal_frame"]
        hedge_ratio_val = bt_result.get("hedge_ratio", 1.0)

        st.markdown(
            "Varying transaction costs, slippage, and borrow costs "
            "while holding the pair relationship and signal positions fixed."
        )

        cost_result = run_cost_sensitivity(
            test_prices[ticker_y],
            test_prices[ticker_x],
            signal_frame["position"],
            hedge_ratio=hedge_ratio_val,
            initial_capital=initial_capital,
        )

        st.dataframe(cost_result, use_container_width=True)

        # Simple plot: return vs transaction cost (zero slippage, zero borrow)
        zero_scenario = cost_result[
            (cost_result["slippage_bps"] == 0.0) & (cost_result["annual_borrow_cost"] == 0.0)
        ]
        if not zero_scenario.empty:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(
                zero_scenario["transaction_cost_bps"],
                zero_scenario["total_return"],
                marker="o", color=STYLE_COLORS["primary"],
            )
            ax.set_xlabel("Transaction cost (bps)")
            ax.set_ylabel("Total return")
            ax.set_title("Return vs transaction cost (zero slippage/borrow)")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        csv = cost_result.to_csv().encode("utf-8")
        st.download_button("📥 Download cost sensitivity", csv, "cost_sensitivity.csv", mime="text/csv")


# -------------------------------------------------------------------------
# TAB 9 — Parameter robustness
# -------------------------------------------------------------------------

with TABS[8]:
    st.subheader("Parameter robustness")

    if "error" in bt_result:
        st.warning("No parameter robustness data (backtest did not run).")
    else:
        st.warning(
            "**Anti-overfitting note**: The baseline parameters "
            f"(lookback={lookback_window}, entry={entry_threshold}, "
            f"exit={exit_threshold}, stop={stop_threshold}) are the "
            "project's primary result.  Do **not** select parameters "
            "based on test-period performance."
        )

        test_prices = bt_result["test_prices"]
        train_prices = bt_result["train_prices"]

        with st.spinner("Running parameter sensitivity (may take a moment) …"):
            param_result = run_parameter_sensitivity(
                train_prices[ticker_y],
                train_prices[ticker_x],
                test_prices[ticker_y],
                test_prices[ticker_x],
                initial_capital=initial_capital,
            )

        stability = parameter_stability_summary(param_result)

        col1, col2, col3 = st.columns(3)
        col1.metric("Median Sharpe", f"{stability['median_sharpe']:.3f}")
        col2.metric("Profitable combos", f"{stability['profitable_proportion']:.1%}")
        col3.metric("Sharpe range", f"{stability['sharpe_min']:.2f} – {stability['sharpe_max']:.2f}")

        st.dataframe(param_result, use_container_width=True)
        csv = param_result.to_csv().encode("utf-8")
        st.download_button("📥 Download parameter sensitivity", csv, "parameter_sensitivity.csv", mime="text/csv")


# -------------------------------------------------------------------------
# TAB 10 — Multiple testing
# -------------------------------------------------------------------------

with TABS[9]:
    st.subheader("Multiple-hypothesis testing corrections")

    if screening_results.empty:
        st.info("No screening results to analyse.")
    else:
        # Apply corrections if not already present
        if "cointegration_bonferroni_pvalue" not in screening_results.columns:
            corrected = apply_multiple_testing_corrections(screening_results.copy())
        else:
            corrected = screening_results

        n_tested = len(corrected)
        st.markdown(f"**Pairs tested**: {n_tested}")

        if "cointegration_significant_raw" in corrected.columns:
            raw_sig = int(corrected["cointegration_significant_raw"].sum())
            bonf_sig = int(corrected["cointegration_significant_bonferroni"].sum())
            bh_sig = int(corrected["cointegration_significant_bh"].sum())

            col1, col2, col3 = st.columns(3)
            col1.metric("Raw significant", str(raw_sig))
            col2.metric("Bonferroni significant", str(bonf_sig))
            col3.metric("BH significant", str(bh_sig))

            st.markdown("### Multiple-testing corrected p-values")
            display_cols = [
                "ticker_y", "ticker_x",
                "cointegration_pvalue", "cointegration_bonferroni_pvalue",
                "cointegration_bh_pvalue", "cointegration_significant_bh",
                "adf_pvalue", "adf_bonferroni_pvalue", "adf_bh_pvalue",
            ]
            display_cols = [c for c in display_cols if c in corrected.columns]
            st.dataframe(corrected[display_cols], use_container_width=True)

        csv = corrected.to_csv().encode("utf-8")
        st.download_button("📥 Download multiple-testing results", csv, "multiple_testing.csv", mime="text/csv")


# -------------------------------------------------------------------------
# TAB 11 — Data quality
# -------------------------------------------------------------------------

with TABS[10]:
    st.subheader("Data quality report")

    st.markdown("Per-ticker quality assessment over the full date range.")

    with st.spinner("Assessing ticker quality …"):
        quality_df = assess_all_tickers(prices)

    if not quality_df.empty:
        passed = quality_df["passed_quality_filter"].sum()
        total = len(quality_df)
        st.metric("Tickers passing quality filter", f"{passed} / {total}")

        st.dataframe(quality_df, use_container_width=True)
        csv = quality_df.to_csv().encode("utf-8")
        st.download_button("📥 Download quality report", csv, "data_quality_report.csv", mime="text/csv")
    else:
        st.info("No ticker quality data.")

    st.markdown("---")
    st.subheader("Pair overlap report")
    with st.spinner("Computing pair overlaps …"):
        overlap_df = compute_all_pair_overlaps(prices)
    if not overlap_df.empty:
        st.dataframe(overlap_df, use_container_width=True)
        csv = overlap_df.to_csv().encode("utf-8")
        st.download_button("📥 Download pair overlap report", csv, "pair_overlap_report.csv", mime="text/csv")
    else:
        st.info("No pair overlap data.")
