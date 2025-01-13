import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent)) # ChatGPT
import pytest
from app.src.question.text_input import DataFetcher
import pandas as pd


class MockAgg:
    """Mock class for polygon.io aggregates"""
    def __init__(self, close, timestamp):
        self.close = close
        self.timestamp = timestamp


@pytest.fixture
def data_fetcher():
    """Fixture providing DataFetcher instance"""
    return DataFetcher()


@pytest.fixture
def mock_aggs():
    """Fixture providing mock aggregate data"""
    return [
        MockAgg(100.0, 1704067200000),
        MockAgg(101.0, 1704153600000),
        MockAgg(102.0, 1704240000000),
    ]


def test_process_data_valid(data_fetcher, mock_aggs):
    """Test processing of valid aggregate data"""
    result = data_fetcher.process_data(mock_aggs)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(mock_aggs)
    assert 'Close' in result.columns
    assert list(result['Close']) == [100.0, 101.0, 102.0]
    assert isinstance(result.index, pd.DatetimeIndex)


def test_get_data_multiple_tickers(data_fetcher, mock_aggs, mocker):
    """Test fetching data for multiple tickers"""
    tickers = ['AAPL', 'C:XAUUSD']
    mocker.patch.object(
        data_fetcher, 'fetch_aggregates', return_value=mock_aggs)

    result = data_fetcher.get_data(tickers, days=30)

    assert isinstance(result, dict)
    assert set(result.keys()) == set(tickers)
    for ticker in tickers:
        assert isinstance(result[ticker], pd.DataFrame)
        assert not result[ticker].empty
        assert 'Close' in result[ticker].columns


def test_get_data_empty_response(data_fetcher, mocker):
    """Test handling of empty data response with invalid ticker"""
    mocker.patch.object(data_fetcher, 'fetch_aggregates', return_value=[])

    result = data_fetcher.get_data(['PEPEMEMECOIN'])
    assert isinstance(result, dict)
    assert len(result) == 0
