import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent)) # ChatGPT
import pytest
from app.src.question.multi_choice import GarchAnalysis
import pandas as pd
import numpy as np


@pytest.fixture
def sample_price_data():
    """
    Fixture providing sample price data for volatility analysis.
    """
    np.random.seed(69)
    n_points = 100

    # Generate prices with small variations for controlled log returns
    prices = 100 + np.cumsum(np.random.normal(loc=0, scale=0.1, size=n_points))
    dates = pd.date_range(start="2024-01-01", periods=n_points, freq="D")
    return pd.DataFrame({'Close': prices}, index=dates)


@pytest.fixture
def garch_analyzer():
    """Fixture providing GarchAnalysis instance."""
    return GarchAnalysis()


def test_log_returns_scaling_small_mean(garch_analyzer, sample_price_data):
    """
    Test that log returns are scaled when the mean is below the threshold.
    """
    scale_threshold = 0.001
    expected_factor = 1000

    # Adjust data to create very small log returns
    sample_price_data['Close'] *= 0.001

    # Calculate log returns
    log_returns = np.log(
        sample_price_data['Close'] /
        sample_price_data['Close'].shift(1)
    ).dropna()

    # Call the preprocessing method
    processed_returns, scale_factor = garch_analyzer.preprocess_log_returns(
        log_returns, scale_threshold=scale_threshold
    )

    # Assertions
    assert scale_factor == expected_factor, \
        f"Expected scale factor {expected_factor}, got {scale_factor}."
    assert abs(processed_returns.mean()) >= scale_threshold, \
        f"Processed mean {processed_returns.mean()} is below " \
        f"threshold after scaling."


def test_log_returns_scaling_large_mean(garch_analyzer, sample_price_data):
    """
    Test that log returns are not scaled when the mean is above the threshold.
    """
    scale_threshold = 0.001
    expected_factor = 1

    # Adjust data to create large log returns
    sample_price_data['Close'] += np.arange(len(sample_price_data)) * 10

    # Calculate log returns
    log_returns = np.log(
        sample_price_data['Close'] /
        sample_price_data['Close'].shift(1)
    ).dropna()

    assert abs(log_returns.mean()) >= scale_threshold, \
        f"Mean of log returns {log_returns.mean()} is below " \
        f"the threshold {scale_threshold}."

    # Call the preprocessing method
    processed_returns, scale_factor = garch_analyzer.preprocess_log_returns(
        log_returns, scale_threshold=scale_threshold
    )

    # Assertions
    assert scale_factor == expected_factor, \
        f"Expected scale factor {expected_factor}, got {scale_factor}."
    assert processed_returns.equals(log_returns), \
        "Processed returns should be unchanged for large mean values."


@pytest.mark.parametrize('p_max,q_max', [
    (1, 1),
    (2, 2),
    (3, 3)
])
def test_garch_parameter_search(garch_analyzer,
                                sample_price_data,
                                p_max,
                                q_max):
    """
    Test GARCH parameter search with different max orders.
    And ensure the chosen parameter values are within the expected range.
    """
    log_returns = np.log(
        sample_price_data['Close'] /
        sample_price_data['Close'].shift(1)
    ).dropna()

    p, q = garch_analyzer.find_best_garch_params(log_returns)

    assert isinstance(p, int), f"Expected integer for p, got {type(p)}."
    assert isinstance(q, int), f"Expected integer for q, got {type(q)}."
    assert 1 <= p <= p_max, f"p ({p}) not in range [1, {p_max}]."
    assert 1 <= q <= q_max, f"q ({q}) not in range [1, {q_max}]."


def test_volatility_analysis_integration(garch_analyzer, sample_price_data):
    """
    Integration test for complete volatility analysis.
    Ensure the analysis completes without errors.
    """
    try:
        garch_analyzer.analyze_volatility(sample_price_data)
        assert True
    except Exception as e:
        pytest.fail(f"Volatility analysis failed with error: {e}")
