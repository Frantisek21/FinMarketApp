import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent)) # ChatGPT
import pytest
from app.src.question.text_input import DataFetcher
from app.src.question.multi_choice import CorrelationAnalysis
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def sample_dataframes():
    """Fixture providing sample price data for both stock and hedge."""
    dates = pd.date_range(start='2024-01-01', periods=10, freq='D')

    np.random.seed(69)
    base_returns = np.random.normal(0, 0.01, 10)
    stock_returns = base_returns + np.random.normal(0, 0.005, 10)
    gold_returns = base_returns * 0.5 + np.random.normal(0, 0.005, 10)

    stock_prices = 100 * np.exp(np.cumsum(stock_returns))
    gold_prices = 1800 * np.exp(np.cumsum(gold_returns))

    return {
        'AAPL': pd.DataFrame({'Close': stock_prices}, index=dates),
        'C:XAUUSD': pd.DataFrame({'Close': gold_prices}, index=dates)
    }


def test_correlation_analysis(sample_dataframes):
    """Test correlation analysis functionality."""
    analyzer = CorrelationAnalysis()

    # Manually
    stock_df = sample_dataframes['AAPL']
    gold_df = sample_dataframes['C:XAUUSD']

    stock_returns = np.log(
        stock_df['Close'] / stock_df['Close'].shift(1)).dropna()
    gold_returns = np.log(
        gold_df['Close'] / gold_df['Close'].shift(1)).dropna()

    expected_corr = stock_returns.corr(gold_returns)

    try:
        analyzer.analyze_correlation(
            sample_dataframes,
            'AAPL',
            'C:XAUUSD',
            'orange',
            rolling_window=5
        )

        assert 0 < expected_corr <= 1, (
            f"Expected positive correlation, got {expected_corr}"
        )

    except Exception as e:
        pytest.fail(f"Correlation analysis failed: {e}")


def test_data_fetcher_date_range():
    """Test date range calculation for data fetching."""
    fetcher = DataFetcher()
    days = 30

    result = fetcher.get_data(['AAPL'], days=days)

    start_date = datetime.today() - timedelta(days=days)
    end_date = datetime.today()

    # Verify correct date range (GPT helped with generating this section)
    assert (end_date - start_date).days == days

    if result:
        assert isinstance(result, dict)
        assert all(isinstance(
            df, pd.DataFrame) for df in result.values())
        if 'AAPL' in result and not result['AAPL'].empty:
            date_range = result['AAPL'].index.max() - result['AAPL'].index.min()
            assert date_range.days <= days, (
                f"Date range {date_range.days} exceeds specified {days} days"
            )
