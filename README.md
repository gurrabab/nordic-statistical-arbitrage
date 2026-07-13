# Nordic Statistical Arbitrage

## Project objective
This project explores pairs trading opportunities among Nordic equities by combining statistical arbitrage ideas with financial time-series analysis. The initial focus is on identifying cointegrated and mean-reverting pairs using publicly available market data and documenting the workflow in a way that is suitable for a quantitative analyst internship application.

## Research question
Which Nordic equity pairs exhibit meaningful cointegration and mean reversion over a historical sample, and how should such relationships be evaluated before any trading decision is considered?

## Proposed methodology
The project will follow a structured, research-first workflow:
1. Download adjusted closing prices for selected Nordic equities.
2. Validate data quality, including missing observations and duplicate ticker inputs.
3. Evaluate candidate pairs using cointegration and stationarity diagnostics.
4. Prepare a reproducible analysis pipeline for future signal development and risk assessment.

The core statistical framework uses an OLS regression with an intercept:

$$y_t = \alpha + \beta x_t + \epsilon_t$$

The spread is defined as the residual from this regression:

$$spread_t = y_t - \alpha - \beta x_t$$

The analysis then evaluates two hypotheses:
- Engle-Granger cointegration test: the null hypothesis is that the two price series are not cointegrated.
- ADF test on the spread: the null hypothesis is that the spread contains a unit root and is therefore not stationary.

The hedge ratio, $\beta$, is interpreted as the estimated slope that relates the dependent series to the independent series. In a pairs-trading context, it is the position size that neutralizes the common market exposure between the two instruments.

Half-life describes the expected time for the spread to move halfway back toward its mean. A shorter half-life suggests faster mean reversion, while a non-positive slope implies that the spread is not estimated to revert.

No machine learning methods are planned for this initial step. The work is intentionally limited to transparent, statistics-based analysis.

## Project structure
- data/: raw or processed market data files (ignored by Git when downloaded locally).
- notebooks/: exploratory analysis notebooks.
- results/: generated tables, plots, and report outputs.
- src/: modular Python package for data loading, pair selection, signals, backtesting, and risk metrics.
- tests/: regression and unit tests for the core analysis logic.

## Installation
Use Python 3.11 or newer.

```bash
cd nordic-statistical-arbitrage
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

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
