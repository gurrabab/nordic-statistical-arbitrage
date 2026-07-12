import pandas as pd
import pytest

from src.data_loader import download_adjusted_close_prices


class DummyDownloadResult(dict):
    """Simple helper class used to mimic yfinance output in tests."""


def test_download_adjusted_close_prices_returns_expected_frame(monkeypatch):
    frame = pd.DataFrame(
        {
            ("SEB-A.ST", "Adj Close"): [100.0, 101.0, 102.0],
            ("SHB-A.ST", "Adj Close"): [90.0, 91.0, 92.0],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )

    def fake_download(*args, **kwargs):
        return frame

    monkeypatch.setattr("src.data_loader.yf.download", fake_download)

    result = download_adjusted_close_prices(["SEB-A.ST", "SHB-A.ST"], period="1mo")

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["SEB-A.ST", "SHB-A.ST"]
    assert result.shape == (3, 2)


def test_download_adjusted_close_prices_rejects_duplicates(monkeypatch):
    monkeypatch.setattr("src.data_loader.yf.download", lambda *args, **kwargs: pd.DataFrame())

    with pytest.raises(ValueError, match="Duplicate"):
        download_adjusted_close_prices(["SEB-A.ST", "SEB-A.ST"])


def test_download_adjusted_close_prices_rejects_empty_download(monkeypatch):
    monkeypatch.setattr("src.data_loader.yf.download", lambda *args, **kwargs: pd.DataFrame())

    with pytest.raises(ValueError, match="empty"):
        download_adjusted_close_prices(["SEB-A.ST", "SHB-A.ST"])


def test_download_adjusted_close_prices_rejects_missing_observations(monkeypatch):
    frame = pd.DataFrame(
        {
            ("SEB-A.ST", "Adj Close"): [100.0, None, 102.0],
            ("SHB-A.ST", "Adj Close"): [90.0, 91.0, 92.0],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )

    monkeypatch.setattr("src.data_loader.yf.download", lambda *args, **kwargs: frame)

    with pytest.raises(ValueError, match="missing"):
        download_adjusted_close_prices(["SEB-A.ST", "SHB-A.ST"])
