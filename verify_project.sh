#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# verify_project.sh — Full project verification script
#
# Usage:  bash verify_project.sh
#
# This script:
#   1. Creates a fresh virtual environment and installs dependencies.
#   2. Runs all tests.
#   3. Runs a small example analysis (pair screening with 6 tickers).
#   4. Confirms output files are created.
#   5. Prints a summary.
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo " Nordic Statistical Arbitrage — Verify"
echo "========================================"

# ---- Step 0: Clean any previous verification venv ----
VENV_DIR=".venv_verify"
if [ -d "$VENV_DIR" ]; then
    echo "[1/5] Removing previous verification environment ..."
    rm -rf "$VENV_DIR"
fi

# ---- Step 1: Create virtual environment and install dependencies ----
echo "[1/5] Creating virtual environment and installing dependencies ..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

echo "[2/5] Dependencies installed successfully."

# ---- Step 2: Run all tests ----
echo "[3/5] Running all tests ..."
python -m pytest -v --tb=short
echo "[3/5] All tests passed."

# ---- Step 3: Run small analysis ----
echo "[4/5] Running minimal pair-screening analysis (6 tickers, 2 years) ..."
python -c "
from src.pair_screener import download_ticker_universe, screen_pairs, PairScreeningParameters

tickers = ['NOVO-B.CO', 'DSV.CO', 'AZN.ST', 'ERIC-B.ST', 'SAND.ST', 'VOLV-B.ST']
prices, report = download_ticker_universe(tickers, start='2020-01-01', end='2022-01-01')
print(f'Downloaded {len(prices)} rows for {len(prices.columns)} tickers')

params = PairScreeningParameters(
    minimum_observations=100,
    maximum_cointegration_pvalue=0.05,
    maximum_adf_pvalue=0.05,
    minimum_half_life=5.0,
    maximum_half_life=250.0,
    minimum_price=1.0,
    maximum_missing_fraction=0.1,
    top_n_pairs=5,
    train_fraction=0.7,
)
results = screen_pairs(prices, params)
print(f'Screened {len(results)} qualifying pairs')
if not results.empty:
    print(results[['ticker_y', 'ticker_x', 'cointegration_pvalue', 'adf_pvalue', 'score']].to_string(index=False))
else:
    print('No pairs passed the screening filters.')
"
echo "[4/5] Analysis completed."

# ---- Step 4: Confirm output file structure ----
echo "[5/5] Validating project structure ..."
EXPECTED_FILES=(
    "app.py"
    "main.py"
    "screen_pairs.py"
    "walk_forward_analysis.py"
    "requirements.txt"
    "README.md"
    "src/__init__.py"
    "src/universe.py"
    "src/data_loader.py"
    "src/pair_selection.py"
    "src/pair_screener.py"
    "src/signals.py"
    "src/backtester.py"
    "src/risk_metrics.py"
    "src/validation.py"
    "src/walk_forward.py"
    "tests/test_walk_forward.py"
)
MISSING=0
for f in "${EXPECTED_FILES[@]}"; do
    if [ ! -f "$f" ]; then
        echo "  MISSING: $f"
        MISSING=$((MISSING + 1))
    fi
done
if [ "$MISSING" -eq 0 ]; then
    echo "  All expected files present."
else
    echo "  $MISSING expected file(s) missing."
fi

# ---- Summary ----
echo ""
echo "========================================"
echo " Verification complete"
echo "========================================"
echo "  Virtual environment: $VENV_DIR"
echo "  Missing files:       $MISSING"
echo ""
echo " To launch the dashboard:"
echo "   source $VENV_DIR/bin/activate"
echo "   streamlit run app.py"
echo "========================================"

# Clean up venv
rm -rf "$VENV_DIR"
