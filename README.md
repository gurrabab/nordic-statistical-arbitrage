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

## Disclaimer
This project is for educational and research purposes only. It does not constitute financial advice, and no trading strategy should be implemented without independent review, robust validation, and appropriate risk controls.
