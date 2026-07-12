"""Utilities for downloading and validating financial price data."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import pandas as pd
import yfinance as yf


def download_adjusted_close_prices(
    tickers: Sequence[str] | Iterable[str],
    *,
    start: str | None = None,
    end: str | None = None,
    period: str | None = None,
) -> pd.DataFrame:
    """Download adjusted close prices for one or more tickers.

    Parameters
    ----------
    tickers:
        Sequence of ticker symbols to download.
    start:
        Optional start date forwarded to yfinance.
    end:
        Optional end date forwarded to yfinance.
    period:
        Optional period forwarded to yfinance.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by date with one column per requested ticker.

    Raises
    ------
    TypeError
        If yfinance does not return a pandas DataFrame.
    ValueError
        If tickers are invalid, downloaded data is empty, adjusted close
        prices are unavailable, requested tickers are missing, or the data
        contains missing observations.
    """
    ticker_list = [ticker.strip() for ticker in tickers]

    if not ticker_list:
        raise ValueError("At least one ticker must be provided.")

    if any(not ticker for ticker in ticker_list):
        raise ValueError("Ticker symbols must not be empty.")

    if len(ticker_list) != len(set(ticker_list)):
        raise ValueError("Duplicate tickers are not allowed.")

    data = yf.download(
        tickers=ticker_list,
        start=start,
        end=end,
        period=period,
        progress=False,
        auto_adjust=False,
        threads=True,
    )

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame from the download.")

    if data.empty:
        raise ValueError("Downloaded data is empty.")

    prices = _extract_adjusted_close_prices(
        data=data,
        ticker_list=ticker_list,
    )

    missing_tickers = [
        ticker
        for ticker in ticker_list
        if ticker not in prices.columns
    ]

    if missing_tickers:
        raise ValueError(
            "Downloaded data is missing requested tickers: "
            + ", ".join(missing_tickers)
        )

    prices = prices.loc[:, ticker_list].copy()

    if prices.isna().any().any():
        missing_counts = prices.isna().sum()
        affected_tickers = missing_counts[missing_counts > 0]

        details = ", ".join(
            f"{ticker}: {count}"
            for ticker, count in affected_tickers.items()
        )

        raise ValueError(
            "Downloaded data contains missing observations"
            f" ({details})."
        )

    return prices


def _extract_adjusted_close_prices(
    data: pd.DataFrame,
    ticker_list: list[str],
) -> pd.DataFrame:
    """Extract adjusted close prices from yfinance output.

    Supports both common MultiIndex layouts:

    - ("Adj Close", "SEB-A.ST")
    - ("SEB-A.ST", "Adj Close")

    It also supports the flat-column structure returned for a single ticker.
    """
    if isinstance(data.columns, pd.MultiIndex):
        return _extract_from_multiindex(data)

    return _extract_from_flat_columns(
        data=data,
        ticker_list=ticker_list,
    )


def _extract_from_multiindex(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Extract adjusted close prices from MultiIndex columns."""
    for level_number in range(data.columns.nlevels):
        level_values = data.columns.get_level_values(level_number)

        if "Adj Close" in level_values:
            prices = data.xs(
                "Adj Close",
                axis=1,
                level=level_number,
            ).copy()

            if isinstance(prices, pd.Series):
                prices = prices.to_frame()

            if isinstance(prices.columns, pd.MultiIndex):
                prices.columns = [
                    _find_ticker_in_column(column)
                    for column in prices.columns
                ]

            return prices

    raise ValueError(
        "Downloaded data does not contain adjusted close prices."
    )


def _extract_from_flat_columns(
    data: pd.DataFrame,
    ticker_list: list[str],
) -> pd.DataFrame:
    """Extract adjusted close prices from non-MultiIndex columns."""
    if "Adj Close" not in data.columns:
        raise ValueError(
            "Downloaded data does not contain adjusted close prices."
        )

    if len(ticker_list) != 1:
        raise ValueError(
            "Expected MultiIndex columns when downloading multiple tickers."
        )

    prices = data[["Adj Close"]].copy()
    prices.columns = ticker_list

    return prices


def _find_ticker_in_column(
    column: object,
) -> str:
    """Return the ticker-like value from a remaining column label."""
    if isinstance(column, tuple):
        non_empty_values = [
            str(value)
            for value in column
            if value is not None and str(value)
        ]

        if not non_empty_values:
            raise ValueError(
                "Could not determine ticker from downloaded columns."
            )

        return non_empty_values[-1]

    return str(column)