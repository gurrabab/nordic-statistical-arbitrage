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

## Disclaimer
This project is for educational and research purposes only. It does not constitute financial advice, and no trading strategy should be implemented without independent review, robust validation, and appropriate risk controls.
