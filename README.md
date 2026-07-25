# Nordic Statistical Arbitrage

A quantitative research pipeline for pairs-trading among Nordic equities.  
This project demonstrates a complete, reproducible workflow for identifying
cointegrated equity pairs, generating signals, running backtests with
transaction costs, and validating results with chronological walk-forward
analysis.

**Target audience:** Quantitative analyst internship applications.

---

## Research question

**Which Nordic equity pairs exhibit meaningful cointegration and mean reversion
over a historical sample, and how robust are these relationships across
sequential out-of-sample periods?**

The goal is not to produce a trading strategy but to build a transparent,
statistics-based evaluation pipeline that avoids common pitfalls such as
look-ahead bias, data snooping, and over-optimisation.

---

## Methodology

### Core framework

The analysis uses OLS regression to model the linear relationship between two
equity prices:

$$y_t = \alpha + \beta x_t + \epsilon_t$$

The residual spread is:

$$spread_t = y_t - \alpha - \beta x_t$$

### Cointegration

The **Engle-Granger cointegration test** evaluates whether the two price series
share a common stochastic trend.  The null hypothesis is *no cointegration*.
A low p-value suggests that the spread is stationary and the pair is a
candidate for pairs trading.

### Stationarity

The **Augmented Dickey-Fuller (ADF) test** is applied to the spread to check
for stationarity.  The null is *the spread has a unit root*.  A low p-value
suggests mean reversion.

### Half-life

Half-life measures the expected time for the spread to revert halfway to its
mean:

$$\text{half-life} = \frac{-\ln(2)}{\lambda}$$

where $\lambda$ is the slope from regressing $\Delta spread_t$ on
$spread_{t-1}$.  A shorter half-life implies faster mean reversion.

### Ranking formula

Screened pairs are ranked using a transparent score that rewards lower p-values,
reasonable reversion speed, and greater observation count:

$$\text{score} = -2 \cdot \log_{10}(p_{coint}) - \log_{10}(p_{adf}) + 0.01 \cdot \min(HL, 100) + 0.001 \cdot N_{obs}$$

### Signal generation

A rolling z-score is computed using only backward-looking data on the test
spread:

$$z_t = \frac{spread_t - \mu_{window}(spread)}{\sigma_{window}(spread)}$$

Entry/exit thresholds determine position state:

| Condition | Action |
|---|---|
| $z \leq -z_{entry}$ | Long the spread (+1) |
| $z \geq z_{entry}$ | Short the spread (-1) |
| $|z| \leq z_{exit}$ or $|z| \geq z_{stop}$ | Flat (0) |

### Backtest

Positions are executed with a **one-day delay** to simulate realistic
implementation.  Transaction costs, slippage, and borrowing costs are deducted
from daily returns.

---

## System architecture

```
nordic-statistical-arbitrage/
├── app.py                    # Streamlit dashboard (UI layer)
├── main.py                   # Command-line analysis entrypoint
├── screen_pairs.py           # Pair screening workflow
├── walk_forward_analysis.py  # Walk-forward validation workflow
├── robustness_analysis.py    # Comprehensive robustness pipeline (Phase 2)
├── verify_project.sh         # Full project verification script
├── requirements.txt          # Python dependencies
├── pyproject.toml            # Project metadata, Ruff, pytest config
├── README.md                 # This file
│
├── src/
│   ├── universe.py           # Nordic equity ticker definitions
│   ├── data_loader.py        # Yahoo Finance data download
│   ├── pair_selection.py     # OLS regression, cointegration, half-life
│   ├── pair_screener.py      # Multi-pair screening + multiple-testing corrections
│   ├── signals.py            # Z-score and position logic
│   ├── backtester.py         # Delayed-execution backtest engine
│   ├── risk_metrics.py       # Performance and risk summaries
│   ├── validation.py         # Train/test split and evaluation
│   ├── walk_forward.py       # Walk-forward window generation and analysis
│   ├── trade_analysis.py     # Trade-level extraction and summarisation
│   ├── benchmarks.py         # Benchmark comparisons (cash, buy-hold, equal-weight)
│   ├── sensitivity.py        # Cost and parameter sensitivity grids
│   └── data_quality.py       # Per-ticker quality and pair overlap reports
│
├── tests/
│   ├── ...                   # 154 unit and regression tests
│
├── .github/workflows/
│   └── tests.yml             # CI pipeline (Ruff + pytest on push/PR)
│
└── results/                  # Generated CSV outputs and charts
```

### Module responsibilities

| Module | Responsibility | Reused by |
|---|---|---|
| `universe.py` | 30-ticker Nordic universe | All scripts |
| `data_loader.py` | Yahoo Finance download | `pair_screener` |
| `pair_selection.py` | OLS, cointegration, ADF, half-life | `pair_screener`, `validation` |
| `pair_screener.py` | Multi-pair screen, rank, OOS evaluation, multiple-testing corrections | `screen_pairs.py`, `walk_forward.py`, `app.py` |
| `signals.py` | Rolling z-score, position generation | `validation`, `walk_forward`, `app` |
| `backtester.py` | Cost-aware delayed backtest | `validation`, `walk_forward`, `app` |
| `risk_metrics.py` | PerformanceSummary dataclass | `validation`, `walk_forward`, `app` |
| `validation.py` | Chronological train/test evaluation | `screen_pairs.py`, `app` |
| `walk_forward.py` | Multi-window walk-forward analysis | `walk_forward_analysis.py`, `app` |
| `trade_analysis.py` | Trade record extraction, summarisation, CSV export | `app`, `robustness_analysis` |
| `benchmarks.py` | Benchmark return comparison (cash, buy-hold, equal-weight) | `app`, `robustness_analysis` |
| `sensitivity.py` | Cost and parameter sensitivity grids with anti-overfitting warnings | `app`, `robustness_analysis` |
| `data_quality.py` | Per-ticker quality reports and pair overlap analysis | `app`, `robustness_analysis` |
| `app.py` | Streamlit dashboard | — |

---

## Anti-look-ahead and anti-overfitting design

1. **Per-window ranking** — Pairs are screened and ranked independently in
   each training window.  No test data influences pair selection.
2. **Fixed hedge ratio** — The hedge ratio is estimated on training data and
   locked for the entire test period.
3. **One-day execution delay** — Signals observed at day $t$ are executed at
   day $t+1$, matching real implementation.
4. **Backward-looking z-score** — The rolling mean and standard deviation use
   only data up to the current date.
5. **No threshold optimisation** — Entry, exit, stop, and screening thresholds
   are set once and never re-tuned.
6. **Parameter sensitivity is diagnostic only** — The `sensitivity.py` module
   tests many parameter combinations, but its results are explicitly labelled
   as **diagnostic only**.  The baseline parameters remain the primary result,
   and no parameter is selected based on test-period performance.
7. **Multiple-testing corrections** — Bonferroni and Benjamini-Hochberg
   corrections are applied to cointegration and ADF p-values across all tested
   pairs.  The dashboard and CLI report both raw and corrected significance.
8. **Data quality filters** — Tickers with excessive missing data, suspicious
   returns, constant prices, or non-positive prices are flagged and optionally
   excluded before any analysis runs.
9. **All windows saved** — Positive and negative results are preserved,
   including windows with no qualifying pairs.
10. **No machine learning** — All methods are classical statistics with
    transparent formulas.

---

## Transaction-cost assumptions

| Parameter | Default value | Notes |
|---|---|---|
| Transaction cost | 5 bps | Combined commission + exchange fees |
| Slippage | 2 bps | Market-impact estimate for Nordic equities |
| Annual borrow cost | 2% | Cost to hold short positions |
| Execution delay | 1 day | Signal observed at $t$, executed at $t+1$ |

These are deliberately conservative estimates.  Real costs could be lower for
large-cap Nordic stocks and higher for small-cap or illiquid names.

---

## Out-of-sample methodology

Each screened pair is evaluated using a **chronological train/test split**:

- **Training period** (default 70% of data): Estimate alpha, hedge ratio,
  cointegration statistics, and ranking score.
- **Test period** (default 30% of data): Apply fixed parameters, generate
  signals, run backtest, compute metrics.

The test set is never used to select pairs or tune parameters.  The method
applies to both the single-split evaluation in `screen_pairs.py` and the
multi-window walk-forward in `walk_forward_analysis.py`.

---

## Walk-forward methodology

Walk-forward validation evaluates the robustness of pairs across multiple
sequential out-of-sample periods.

### Window design

```
|----------- training -----------|----- test -----|
t=0                            t=T             t=T+N
```

The window slides forward:

```
|----------- training -----------|----- test -----|
                                 |----------- training -----------|----- test -----|
```

### Window types

| Type | Behaviour |
|---|---|
| **Rolling** (default) | Fixed-size training window, slides by `step_size_days` |
| **Expanding** | Training window grows from the earliest available date |

### Window parameters (default)

| Parameter | Value | Description |
|---|---|---|
| `train_window_days` | 504 | ~2 years of trading days |
| `test_window_days` | 63 | ~3 months |
| `step_size_days` | 63 | Quarterly roll |
| `expanding_window` | False | Fixed-size rolling |

### Consistency score

The score summarises pair performance across all windows:

| Component | Formula |
|---|---|
| Return score | $\max(0, R_{\text{median}}) / R_{\text{median, max}}$ |
| Sharpe score | $\max(0, S_{\text{median}}) / S_{\text{median, max}}$ |
| Profitability | $N_{\text{profitable}} / N_{\text{windows}}$ |
| Drawdown score | $\max(0, 1 + DD_{\text{worst}})$ |

$$\text{consistency\_score} = \frac{1}{4} \cdot (R_{\text{score}} + S_{\text{score}} + P + DD_{\text{score}})$$

A score of 1.0 means the pair had positive median returns, positive Sharpe
ratios, was profitable in every window, and had no drawdown.  A score near 0.25
(the baseline from drawdown only) indicates negligible results.

---

## Key findings

Results are based on real data downloaded from Yahoo Finance (2018-01-01 to
present).  All findings are from project outputs — nothing is fabricated.

### In-sample screening

The pair screener evaluated all **435 unique pairs** from a 30-ticker Nordic
universe.  After applying cointegration and stationarity filters, the top 5
pairs were:

| Rank | Ticker Y | Ticker X | Cointegration p | ADF p | Half-life (days) | Score |
|---|---|---|---|---|---|---|
| 1 | SEB-A.ST | DNB.OL | 0.010 | 0.002 | 35 | 9.13 |
| 2 | MOWI.OL | DSV.CO | 0.018 | 0.004 | 58 | 8.56 |
| 3 | MOWI.OL | MAERSK-B.CO | 0.018 | 0.004 | 55 | 8.55 |
| 4 | INDU-C.ST | SKF-B.ST | 0.020 | 0.005 | 52 | 8.40 |
| 5 | MOWI.OL | WRT1V.HE | 0.029 | 0.007 | 75 | 8.11 |

**Interpretation:** Several pairs show cointegration p-values well below 0.05,
suggesting that price relationships exist that are unlikely to be random.  The
top-ranked pair (SEB-A.ST / DNB.OL) pairs a Swedish bank with a Norwegian bank —
a fundamentally intuitive relationship.

### Out-of-sample evaluation

The top 5 in-sample pairs were evaluated on the held-out test period:

| Pair | Test total return | Test Sharpe ratio | Max drawdown |
|---|---|---|---|
| MOWI.OL / DSV.CO | **+5.0%** | **0.223** | -15.1% |
| SEB-A.ST / DNB.OL | +1.7% | 0.126 | -9.4% |
| INDU-C.ST / SKF-B.ST | +1.2% | 0.098 | -12.3% |
| MOWI.OL / MAERSK-B.CO | -5.2% | -0.080 | -15.6% |
| MOWI.OL / WRT1V.HE | -23.3% | -0.517 | -33.1% |

**Interpretation:** 3 of 5 pairs had positive out-of-sample returns, but only
2 had economically meaningful Sharpe ratios above 0.1.  The top pair in-sample
(SEB-A.ST / DNB.OL) maintained a moderate positive Sharpe out-of-sample,
suggesting some robustness.  Two pairs reversed sharply, illustrating that
in-sample statistical significance does not guarantee future performance.

### Walk-forward results

Walk-forward validation across 26 rolling windows (2018–2026) produced **78
window-level records** for **56 unique pairs**.  Key observations:

| Ticker Y | Ticker X | Windows | Avg return | Median Sharpe | Profit fraction | Consistency |
|---|---|---|---|---|---|---|
| NESTE.HE | WRT1V.HE | 1 | +3.1% | 2.03 | 100% | 1.000 |
| SAND.ST | OUT1V.HE | 1 | +3.1% | 1.82 | 100% | 0.973 |
| HEXA-B.ST | VWS.CO | 1 | +0.4% | 0.30 | 100% | 0.562 |
| EQNR.OL | NOKIA.HE | 1 | +0.4% | 0.18 | 100% | 0.542 |
| NOVO-B.CO | NOKIA.HE | 2 | +0.1% | 0.21 | 50% | 0.407 |

**Important caveat:** Most walk-forward windows generated **zero returns**
because the fixed screening parameters (half-life bounds, p-value thresholds)
excluded pairs in many windows.  Pairs with positive returns appeared in only
1–2 windows each, making the consistency scores unreliable.  This is itself an
important finding: **cointegrated pairs are not consistently available across
time** using fixed thresholds.

### Summary of findings

1. The Nordic equity universe contains statistically identifiable cointegrated
   relationships between fundamentally linked stocks (bank pairs, same-sector
   pairs).
2. In-sample statistical significance partially survives out-of-sample testing,
   but performance degrades significantly — 40% of top pairs lost money out of
   sample.
3. Walk-forward validation reveals that cointegration relationships are
   **temporally unstable** — pairs that pass screening in one period may fail
   entirely in the next.
4. Transaction costs, at conservative estimates, consume a meaningful fraction
   of gross returns, especially for pairs with frequent rebalancing.

---

## Trade-level analysis

The `trade_analysis.py` module extracts individual trades from a backtest
DataFrame and produces summary statistics.

### Trade record fields

| Field | Description |
|---|---|
| `pair` | `ticker_y/ticker_x` identifier |
| `direction` | `LONG_SPREAD` (z ≤ -threshold) or `SHORT_SPREAD` (z ≥ threshold) |
| `entry_date`, `exit_date` | Trade open and close dates |
| `entry_zscore`, `exit_zscore` | Z-score at entry and exit |
| `entry_equity`, `exit_equity` | Portfolio equity at entry and exit |
| `holding_days` | Calendar days the trade was open |
| `gross_return` | Return before costs |
| `net_return` | Return after transaction, slippage, and borrow costs |
| `transaction_cost`, `slippage_cost`, `borrow_cost` | Itemised costs |
| `maximum_adverse_excursion` (MAE) | Worst percentage return from entry |
| `maximum_favorable_excursion` (MFE) | Best percentage return from entry |
| `exit_reason` | `NORMAL_EXIT`, `STOP_EXIT`, or `END_OF_TEST_PERIOD` |

### Trade summary statistics

| Statistic | Description |
|---|---|
| `number_of_trades` | Total completed trades |
| `win_rate` | Fraction of profitable trades |
| `profit_factor` | Gross profit / gross loss ratio |
| `average_winner` / `average_loser` | Mean return of winning and losing trades |
| `average_holding_days` | Mean holding period |
| `stop_exit_count` | Number of trades stopped out |

### Usage

```python
from src.trade_analysis import extract_trades, summarize_trades, trades_to_dataframe

trades = extract_trades(backtest_frame, "SEB-A.ST", "SHB-A.ST", signal_frame=signal_frame)
summary = summarize_trades(trades)       # TradeSummary dataclass
trade_df = trades_to_dataframe(trades)   # pandas DataFrame → CSV export
```

---

## Benchmark comparison

The `benchmarks.py` module compares strategy performance against three
benchmarks.

### Benchmarks

| Benchmark | Description |
|---|---|
| **Cash** | Risk-free return at a configurable annual rate |
| **Buy-and-hold equal-weight** | Equal-weighted buy-and-hold of both constituents |
| **Market index** | A Nordic market index (e.g., `^OMX`); not a perfect comparison for a market-neutral strategy |

### Interpretation

Because pairs trading is market-neutral in theory, a high Sharpe ratio relative
to a long-only index is expected.  The benchmarks primarily serve to:

1. Detect whether the strategy is taking hidden directional bets.
2. Provide a reality check on risk-adjusted returns.
3. Highlight whether returns are driven by general market movements.

```python
from src.benchmarks import compare_benchmarks

benchmark_df = compare_benchmarks(
    strategy_equity=bt_frame["equity"],
    strategy_returns=bt_frame["net_return"],
    initial_capital=100_000.0,
    ticker_y="SEB-A.ST", ticker_x="SHB-A.ST",
    price_y=test_y, price_x=test_x,
)
```

---

## Cost sensitivity analysis

The `sensitivity.py` module tests how changing cost assumptions affects
strategy returns without re-estimating the pair relationship.

### Cost scenarios

| Parameter | Default grid | Unit |
|---|---|---|
| Transaction cost | 0, 5, 10, 20, 30 | bps |
| Slippage | 0, 2, 5, 10 | bps |
| Annual borrow cost | 0.0%, 2.0%, 5.0%, 10.0% | annual rate |

### Key invariant

Returns must be monotonic with respect to costs — higher costs should never
produce higher returns.  The framework checks this and reports a violation if
it occurs (which would indicate a bug or non-linear interaction).

---

## Parameter sensitivity (diagnostic only)

The `sensitivity.py` module also tests how varying signal parameters affects
test-period returns.  **This is for diagnosis only — the baseline parameters
(lookback=60, entry=2.0, exit=0.5, stop=3.5) remain the primary result.**

### Parameter grid

| Parameter | Default grid |
|---|---|
| Lookback window | 20, 40, 60, 90, 120 |
| Entry threshold | 1.5, 2.0, 2.5 |
| Exit threshold | 0.0, 0.5, 1.0 |
| Stop threshold | 2.5, 3.0, 3.5, 4.0 |

### Stability summary

| Statistic | Description |
|---|---|
| `n_combinations` | Total parameter combinations tested |
| `median_sharpe` | Median Sharpe across all combinations |
| `profitable_proportion` | Fraction of combinations with positive return |
| `sharpe_min`, `sharpe_max` | Range of Sharpe ratios |
| `sharpe_std` | Standard deviation of Sharpe ratios |

A high `sharpe_std` or low `profitable_proportion` suggests the strategy is
highly sensitive to parameter choice — a fragility warning.

---

## Data quality controls

The `data_quality.py` module checks per-ticker data quality before analysis.

### Ticker quality report

| Check | Description |
|---|---|
| Missing fraction | Fraction of expected rows that are missing |
| Duplicate dates | Number of duplicated index entries |
| Non-positive prices | Count of prices ≤ 0 |
| Constant price | Whether the series is constant over the full period |
| Suspicious returns | Daily returns exceeding ±20% |
| Largest absolute return | Maximum absolute daily return |

### Pair overlap report

Measures the fraction of overlapping dates between each pair of tickers.
Pairs with low overlap (<50%) are flagged as potentially unreliable for
cointegration testing.

### Usage

```python
from src.data_quality import assess_all_tickers

quality_df = assess_all_tickers(prices)  # DataFrame with per-ticker results
```

---

## Robustness CLI

The `robustness_analysis.py` script runs a comprehensive pipeline that
combines all Phase 2 modules into a single end-to-end workflow:

```bash
python robustness_analysis.py \
  --tickers SEB-A.ST SHB-A.ST AZN.ST ERIC-B.ST SAND.ST \
  --start 2018-01-01 --end 2024-12-31 \
  --output-dir results
```

### Pipeline steps

1. **Data download** — Downloads adjusted close prices for the specified tickers.
2. **Data quality assessment** — Generates per-ticker quality and pair overlap reports.
3. **Baseline pair analysis** — Estimates hedge ratio and spread on the training period.
4. **Baseline backtest** — Runs the default backtest with conservative cost assumptions.
5. **Trade-level analysis** — Extracts individual trades and summary statistics.
6. **Benchmark comparison** — Compares strategy against cash, buy-hold, and equal-weight.
7. **Cost sensitivity** — Tests how returns vary with transaction cost, slippage, and borrow costs.
8. **Parameter sensitivity** — Tests parameter stability (diagnostic only — no test-period optimisation).
9. **Multiple-testing corrections** — Screens the universe and reports Bonferroni and BH-corrected significance.
10. **All results saved** — Every output is written to CSV in the output directory.

### Output files

| File | Contents |
|---|---|
| `prices.csv` | Raw downloaded price data |
| `data_quality_report.csv` | Per-ticker quality checks |
| `pair_overlap_report.csv` | Pairwise date overlap matrix |
| `trade_log.csv` | Individual trade records |
| `benchmark_comparison.csv` | Strategy vs. benchmark metrics |
| `cost_sensitivity.csv` | Return and Sharpe for each cost scenario |
| `parameter_sensitivity.csv` | Return and Sharpe for each parameter combination |
| `multiple_testing.csv` | Pairs with Bonferroni and BH corrections |

---

## What I learned

### Cointegration vs. correlation

Correlation measures the strength of a linear relationship at a point in time
but says nothing about whether the relationship is stable.  Cointegration tests
whether two time series move together over the long run.  Two stocks can be
highly correlated but not cointegrated (e.g., both rise with the market but
drift apart permanently), and two stocks can be cointegrated but have low
daily return correlation.

### Look-ahead bias

It is remarkably easy to introduce look-ahead bias without realising it.  The
most common pitfalls include:
- Using the full-sample mean/standard deviation for z-score calculation.
- Estimating the hedge ratio on data that includes the test period.
- Selecting pairs after seeing test-period returns.

Each of these was deliberately or accidentally present in early versions of
this project and had to be explicitly guarded against.

### Overfitting

With 435 pairs, approximately 22 pairs would pass a cointegration test at the
5% significance level by random chance alone.  The project uses fixed thresholds
and does not optimise parameters to maximise backtest returns, but the multiple
testing problem remains a fundamental limitation.

### Transaction costs

Transaction costs have a compounding effect.  A strategy that appears profitable
at zero cost can become unprofitable after only a few bps per trade.  The
one-day execution delay also accrues slippage that is invisible in same-day
simulations.  The cost sensitivity module quantifies exactly how sensitive
returns are to each cost axis.

### Parameter sensitivity

The parameter sensitivity grid reveals that the strategy's performance is
sensitive to lookback window and entry/exit thresholds.  A narrow lookback
generates more trades but higher turnover (and thus higher costs), while a wide
lookback misses shorter-term mean-reversion.  The diagnostic-only label on
these results is critical — selecting the best-performing parameters from the
grid would constitute data snooping.

### Trade-level granularity

Breaking performance down to individual trades reveals patterns that aggregate
metrics hide.  For example, a strategy with a positive Sharpe ratio may still
have a small number of trades driving all the profit, with most trades
breaking even.  The MAE/MFE analysis shows whether winning trades tend to
move immediately in the right direction or survive early adverse moves.

### Instability of financial relationships

The most surprising finding is how unstable cointegration relationships are.
A pair that looks strongly cointegrated over a 2-year training window can show
no relationship at all in the next window.  This instability is a core argument
against relying on any single backtest result.

### Importance of out-of-sample testing

Every in-sample result in this project looks better than the corresponding
out-of-sample result.  The walk-forward analysis in particular shows that the
majority of windows produce no trades or negative returns even when the
in-sample screen looks healthy.

---

## Limitations and future work

### Survivorship bias

The 30-ticker universe is fixed.  Delisted, bankrupt, or acquired equities are
not included.  This introduces an upward bias because the universe consists only
of companies that survived through the sample period.

### Yahoo Finance data limitations

- **Dividend handling:** Yahoo Finance provides adjusted close prices that
  account for dividends and splits, but the adjustment methodology is not
  transparent.
- **Data errors:** Corporate actions (mergers, spin-offs, ticker changes) can
  produce price discontinuities that are not fully corrected.
- **Missing data:** Some Nordic tickers are not available or have incomplete
  histories (e.g., KESKO.HE was replaced with KNEBV.HE).

### Corporate actions

The project does not explicitly model stock splits, mergers, or special
dividends.  Adjusted close prices partially address this, but large corporate
events can break a cointegration relationship permanently.

### Liquidity and short availability

The backtest assumes all positions can be executed at the prevailing price.
In reality, some Nordic equities may be illiquid, and short selling may be
restricted or expensive.  The 2 bps slippage estimate is a crude approximation.

### Multiple hypothesis testing

With ~435 pairs, the probability of finding false positives is high even before
considering walk-forward windows.  This project now applies two standard
corrections:

- **Bonferroni correction:** The raw p-value threshold $\alpha$ is divided by
  the number of hypotheses: $\alpha_{\text{Bonf}} = \alpha / N$.
- **Benjamini-Hochberg (BH) procedure:** Controls the False Discovery Rate (FDR)
  by ranking p-values and applying a sequential threshold.

The dashboard and CLI report both raw and corrected significance counts.  In
practice, a Bonferroni correction would require a cointegration p-value below
approximately $0.05 / 435 \approx 0.0001$, which almost no pair achieves.
The BH procedure is less conservative and may identify a small number of
candidates.

**Key caution:** These corrections reduce false positives but do not eliminate
them.  Corrected significance should be viewed as a diagnostic, not a guarantee.

### Parameter instability

Half-life, hedge ratio, and cointegration status can change over time.  The
walk-forward analysis shows that parameters estimated on one training window
may not hold in the next.  Adaptive estimation methods are a potential
improvement but were deliberately excluded to avoid overfitting.

### Absence of live execution

The project has been tested only on historical data.  There is no guarantee
that any of the identified relationships will hold in the future, and there
is no mechanism for live trading.

### Market-impact modelling

Position sizing assumes constant market impact regardless of trade size.  In
practice, larger positions would move prices and reduce profitability.

### Future work ideas

- Adaptive threshold estimation (e.g., rolling cointegration windows).
- Liquidity filters based on average daily volume.
- Multi-asset pairs beyond equity-equity (e.g., equity-ETF, cross-listed pairs).
- Regime detection to avoid trading during high-volatility periods.
- Portfolio-level risk allocation rather than individual pair sizing.

---

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd nordic-statistical-arbitrage

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Requires Python 3.11 or newer.

---

## Command-line usage

```bash
# Run all tests
pytest -q

# Run the main analysis pipeline
python main.py

# Run the pair screener
python screen_pairs.py

# Run walk-forward validation
python walk_forward_analysis.py

# Run comprehensive robustness analysis (Phase 2)
python robustness_analysis.py --tickers SEB-A.ST SHB-A.ST --start 2018-01-01 --end 2024-12-31
```

### Output files

| Command | Output files |
|---|---|
| `python screen_pairs.py` | `results/pair_screening_results.csv`, `results/top_pairs_out_of_sample.csv`, charts |
| `python walk_forward_analysis.py` | `results/walk_forward_window_results.csv`, `results/walk_forward_pair_summary.csv`, charts |
| `python main.py` | Console output with train/test comparison |
| `python robustness_analysis.py` | `results/*.csv` — data quality, trade log, benchmarks, cost sensitivity, parameter sensitivity, multiple-testing corrections |

---

## Dashboard usage

```bash
streamlit run app.py
```

### Dashboard sections

| Section | Content |
|---|---|
| **Sidebar** | Ticker selection, date range, signal/backtest/screening/walk-forward parameters, run button |
| **Overview** | Selected pair info, data period, screened pair count, current signal status |
| **Statistical analysis** | Hedge ratio, alpha, cointegration/ADF p-values, half-life, warnings for weak evidence |
| **Performance** | Final equity, total return, Sharpe, Sortino, max drawdown, VaR, hit rate |
| **Charts** | Normalised prices, spread, z-score with thresholds, equity curve, drawdown, daily return histogram, walk-forward charts |
| **Tables** | Latest signals, backtest output, top screened pairs, OOS comparison, walk-forward results |
| **Trade analysis** | Trade count, win rate, profit factor, avg winner/loser, avg holding, stop-exit count, downloadable trade log |
| **Benchmarks** | Strategy vs. cash, buy-hold, and equal-weight equity curves and metrics |
| **Cost sensitivity** | Total return and Sharpe vs. transaction cost, slippage, and borrow cost |
| **Parameter robustness** | Median Sharpe, profitable proportion, Sharpe range/std across parameter grid (diagnostic only) |
| **Multiple testing** | Pairs tested, raw significant, Bonferroni significant, BH significant |
| **Data quality** | Ticker quality table (missing data, suspicious returns, constant prices), pair overlap summary |
| **Export** | Download buttons for every table as CSV |

### Workflow

1. Configure parameters in the sidebar.
2. Click **Run analysis**.
3. Wait for data download, screening, backtest, and walk-forward to complete.
4. Explore results across the five tabs.

---

## Full project verification

```bash
# Run all tests
pytest -q

# Run the comprehensive robustness CLI
python robustness_analysis.py

# Or use the shell script
bash verify_project.sh
```

The verification script:
1. Creates a fresh virtual environment and installs dependencies.
2. Runs all tests.
3. Runs a small example analysis.
4. Confirms output files are created.

---

## Continuous integration

The project includes a CI workflow (`.github/workflows/tests.yml`) that
automatically runs on every push and pull request:

1. **Lint** — Ruff checks for code style and common errors.
2. **Test** — pytest executes all non-internet tests (`-m "not internet"`).

To run the CI suite locally:

```bash
ruff check .
python -m pytest tests/ -v -m "not internet"
```

---

## Example output

### Console output from `walk_forward_analysis.py`

```
Downloaded 30 tickers, 2168 rows
Running walk-forward validation ...
Saved 78 window-level results to results/walk_forward_window_results.csv
Saved 56 pair summaries to results/walk_forward_pair_summary.csv

Most consistent pairs (walk-forward):
   ticker_y    ticker_x  n_windows  average_test_return  consistency_score
   NESTE.HE    WRT1V.HE          1             0.030936              1.000
    SAND.ST    OUT1V.HE          1             0.030891              0.973
  HEXA-B.ST      VWS.CO          1             0.003611              0.562
    EQNR.OL    NOKIA.HE          1             0.003510              0.542
  NOVO-B.CO    NOKIA.HE          2             0.000860              0.407
```

### Test output

```
82 passed in 18.61s
```

---

## Disclaimer

This project is for **educational and research purposes only**.  It does not
constitute financial advice, investment recommendation, or solicitation to
trade.  No trading strategy should be implemented without independent review,
robust validation, and appropriate risk controls.

**No backtest methodology guarantees future profitability.**  Walk-forward
validation reduces but does not eliminate the risk of overfitting.  Past
performance of any statistical relationship is not indicative of future
results.

## Running tests
```bash
pytest -q
```

## Running the analysis
```bash
python3 main.py
```

## Nordic pair screener
The project now includes a modular Nordic pair screener that evaluates a fixed universe of liquid Nordic equities, screens pairs on training data only, ranks them with a transparent training-only score, and then evaluates the top-ranked candidates on a later out-of-sample period.

### Nordic universe
The screener uses an initial universe of 30 liquid Nordic equities spanning Sweden, Norway, Denmark and Finland, including major banks, industrial names, energy firms, telecoms, and consumer staples.

### Number of possible pairs
With 30 tickers, the number of unique unordered pairs is:

$$N \times (N - 1) / 2 = 30 \times 29 / 2 = 435$$

### Screening methodology
1. Download adjusted close prices for the Nordic universe.
2. Report which tickers fail to download or have missing data.
3. Filter out tickers that have insufficient history, too much missing data, constant prices, non-positive prices, or poor overlap with the training period.
4. Generate all unique unordered pairs from the remaining tickers.
5. For each pair, estimate the spread on the training period only.
6. Apply filters based on cointegration p-value, ADF p-value, half-life, observation count, and finite hedge ratio.
7. Rank valid pairs using a transparent training-only score.
8. Evaluate the top-ranked pairs on the unseen test period using the training-estimated alpha and hedge ratio only.

### Ranking formula
The ranking score rewards lower p-values, a reasonable half-life, and sufficient sample size. The exact formula is:

$$
score = 2 \times \log_{10}(p_{coint}) + \log_{10}(p_{adf}) + 0.01 \times \frac{10}{\max(half\_life, 1)} + 0.001 \times n_{obs}
$$

Because the score uses the training-period p-values and observations, it does not use test-period returns or any information from the out-of-sample window.

### Anti-selection-bias rules
The pipeline intentionally avoids several common leakage problems:
- Pairs are ranked using training-period statistics only.
- Thresholds are selected from the training period and then applied unchanged to the test period.
- The hedge ratio is estimated once on training data and then kept fixed for the test period.
- Test performance is reported separately and never used to rank or re-select candidate pairs.
- The screener reports both the training screening table and the out-of-sample evaluation table separately.

### Multiple testing and false discoveries
Testing many pairs increases the chance of false discoveries. The screener therefore reports the total number of tested pairs and the Bonferroni-adjusted significance threshold for transparency:

$$\alpha_{Bonferroni} = \alpha / M$$

where $M$ is the number of tested pairs. In practice, the screener does not automatically require Bonferroni as the only filter, but it reports the threshold so that the user can interpret the results conservatively.

### Running the screener
```bash
python3 screen_pairs.py
```

The script saves:
- results/pair_screening_results.csv
- results/top_pairs_out_of_sample.csv
- several charts in the results directory

## Performance and risk metrics
The backtest workflow now computes a compact set of standard portfolio metrics from the strategy's net daily returns.

### Core formulas
- Total return: $R_{total} = \frac{E_T}{E_0} - 1$
- Annualized return: $R_{ann} = (1 + \bar{r})^{252} - 1$
- Annualized volatility: $\sigma_{ann} = \sigma_{daily} \sqrt{252}$
- Sharpe ratio: $\frac{\bar{r} - r_f/252}{\sigma_{daily}} \sqrt{252}$
- Sortino ratio: $\frac{\bar{r} - r_f/252}{\sigma_{downside}} \sqrt{252}$
- Drawdown: $DD_t = \frac{E_t}{\max(E_1, \dots, E_t)} - 1$
- Maximum drawdown: $\max_t DD_t$
- Calmar ratio: $\frac{R_{ann}}{|\text{Max Drawdown}|}$
- Historical VaR 95%: empirical 5th percentile of daily returns
- Historical Expected Shortfall 95%: average of returns at or below the VaR threshold
- Hit rate: fraction of positive daily returns

### Interpretation
- Total return shows the overall change in equity over the sample.
- Annualized return and volatility summarize the strategy's average growth and variability on a yearly basis.
- Sharpe and Sortino ratios compare reward to risk; the Sortino ratio focuses on downside volatility instead of total volatility.
- Maximum drawdown measures the largest peak-to-trough decline in equity.
- Calmar ratio relates annualized return to drawdown and is most useful when comparing strategies with different risk profiles.
- VaR and Expected Shortfall quantify tail risk over the historical sample.
- Hit rate shows how often the strategy produced a positive daily return.

### Limitations
- Sharpe ratio can be unstable and sensitive to the chosen risk-free rate and sample period.
- Historical VaR and Expected Shortfall are based on the past and do not guarantee future tail behavior.
- Historical backtests can overfit to the sample and are affected by transaction costs, data snooping, and implementation assumptions.
- The strategy remains a research prototype and should not be treated as a production-ready trading system.

## Walk-forward validation

The walk-forward analysis provides a more rigorous out-of-sample test by
evaluating the top-ranked pairs across multiple sequential time periods.

### Methodology

Each walk-forward window consists of a **training period** followed immediately
by a **test period**:

```
|----------- training -----------|----- test -----|
t=0                            t=T             t=T+N
```

The process repeats by sliding the window forward:

```
|----------- training -----------|----- test -----|
                                 |----------- training -----------|----- test -----|
```

### Window types

| Type | Behaviour |
|---|---|
| **Rolling** | Fixed-size training window, slides by `step_size_days` |
| **Expanding** | Training window grows from the earliest date each iteration |

### Default parameters

| Parameter | Value | Description |
|---|---|---|
| `train_window_days` | 504 | ~2 years of trading days |
| `test_window_days` | 63 | ~3 months |
| `step_size_days` | 63 | Non-overlapping quarterly steps |
| `expanding_window` | False | Fixed-size rolling |
| `top_n_pairs_per_window` | 3 | Pairs evaluated per window |

### Anti-leakage rules

1. **Per-window ranking** — Pairs are screened and ranked independently
   in each training window.  Training scores from one window never affect
   another.
2. **Fixed hedge ratio** — The hedge ratio is estimated on training data
   and locked for the entire test period.
3. **No gap or overlap** — The test period starts the day after training
   ends.
4. **Fixed thresholds** — Screening parameters are set once at the start
   and never re-tuned.
5. **All windows saved** — Windows with no qualifying pairs, poor
   performance, or zero trades are all recorded.

### Consistency score

The consistency score summarises a pair's performance across all windows
in which it was selected.  It is the equally-weighted average of four
components, each normalised to [0, 1]:

| Component | Formula | Rewards |
|---|---|---|
| Return score | `max(0, median_return) / max_median_return` | Positive median return |
| Sharpe score | `max(0, median_sharpe) / max_median_sharpe` | Positive risk-adjusted return |
| Profitability | `profitable_window_fraction` | Many profitable windows |
| Drawdown score | `clip(1 + worst_drawdown, 0, 1)` | Limited worst drawdown |

$$\text{consistency\_score} = 0.25 \cdot (R_{\text{score}} + S_{\text{score}} + P + D_{\text{score}})$$

### Interpretation

- Pairs with **consistency_score > 0.5** have positive median returns,
  positive Sharpe ratios, profitable in most windows, and manageable
  drawdowns.
- A high score does **not** guarantee future performance.
- Pairs selected in many windows are more robust than pairs selected
  only once.

### Limitations

- **Repeated testing** — Evaluating pairs across multiple windows increases
  the opportunity for false discovery.  The per-window ranking mitigates
  this but does not eliminate it.
- **Survivorship bias** — The ticker universe is fixed.  Delisted or
  bankrupt equities are not included.
- **Parameter stability** — Screening thresholds are fixed.  A different
  threshold set could produce different results.
- **Cost assumptions** — Transaction costs and slippage are constant.
  Real execution may differ.

### Running walk-forward analysis

```bash
python3 walk_forward_analysis.py
```

The script saves:
- `results/walk_forward_window_results.csv` — one row per pair per window
- `results/walk_forward_pair_summary.csv` — aggregate per-pair statistics
- `results/walk_forward_test_return.png` — test return by window
- `results/walk_forward_sharpe.png` — Sharpe ratio by window
- `results/walk_forward_profitability.png` — proportion of profitable windows
- `results/walk_forward_drawdown.png` — worst drawdown by pair
- `results/walk_forward_selection.png` — pair selection frequency

### Running walk-forward tests

```bash
pytest tests/test_walk_forward.py -v
```

### Warning

Walk-forward validation reduces but does **not** eliminate the risk of
overfitting.  Repeated testing against multiple out-of-sample windows
can still lead to false discoveries.  The consistency score is a
descriptive statistic, not a predictive one.  **No backtest methodology
guarantees future profitability.**

## Disclaimer
This project is for educational and research purposes only. It does not constitute financial advice, and no trading strategy should be implemented without independent review, robust validation, and appropriate risk controls.
