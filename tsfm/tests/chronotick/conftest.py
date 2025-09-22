"""
Pytest configuration and shared fixtures for ChronoTick tests.
"""

import pytest
import numpy as np
import tempfile
import yaml
from pathlib import Path
import sys

# Add the chronotick_inference package to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_offset_data():
    """Create sample clock offset data for testing."""
    np.random.seed(42)  # For reproducible tests
    
    t = np.arange(300)  # 5 minutes of data
    
    # Simulate realistic clock offset with multiple components
    linear_drift = 1e-6 * t  # Linear drift
    oscillations = 2e-6 * np.sin(2 * np.pi * t / 60)  # 1-minute oscillations
    noise = np.random.normal(0, 1e-7, len(t))  # Random noise
    
    offset = linear_drift + oscillations + noise
    return offset.astype(np.float64)


@pytest.fixture  
def sample_system_metrics():
    """Create sample system metrics data for testing."""
    np.random.seed(42)
    
    n_samples = 300
    
    return {
        'cpu_usage': np.random.uniform(20, 80, n_samples),
        'memory_usage': np.random.uniform(30, 90, n_samples), 
        'temperature': np.random.uniform(60, 75, n_samples),
        'voltage': np.random.normal(3.3, 0.05, n_samples),
        'frequency': np.random.normal(2.4e9, 1e7, n_samples),
        'disk_io': np.random.exponential(100, n_samples),
        'network_io': np.random.exponential(50, n_samples)
    }


@pytest.fixture
def minimal_config():
    """Create minimal configuration for testing."""
    return {
        'short_term': {
            'model_name': 'chronos',
            'device': 'cpu',
            'enabled': True,
            'inference_interval': 1.0,
            'prediction_horizon': 5,
            'context_length': 100,
            'max_uncertainty': 0.1,
            'model_params': {}
        },
        'long_term': {
            'model_name': 'timesfm',
            'device': 'cpu', 
            'enabled': True,
            'inference_interval': 30.0,
            'prediction_horizon': 60,
            'context_length': 300,
            'model_params': {}
        },
        'fusion': {
            'enabled': True,
            'method': 'inverse_variance',
            'uncertainty_threshold': 0.05,
            'fallback_weights': {'short_term': 0.7, 'long_term': 0.3}
        },
        'preprocessing': {
            'outlier_removal': {'enabled': False, 'method': 'iqr', 'threshold': 1.5},
            'missing_value_handling': {'enabled': False, 'method': 'interpolate'},
            'normalization': {'enabled': False, 'method': 'zscore'}
        },
        'covariates': {
            'enabled': True,
            'variables': ['cpu_usage', 'temperature', 'memory_usage'],
            'future_variables': []
        },
        'performance': {
            'max_memory_mb': 512,
            'model_timeout': 5.0,
            'cache_size': 10
        },
        'logging': {
            'level': 'INFO',
            'log_predictions': False,
            'log_uncertainty': False,
            'log_fusion_weights': False
        },
        'clock': {
            'frequency_type': 'second',
            'frequency_code': 9
        }
    }


@pytest.fixture
def temp_config_file(minimal_config):
    """Create a temporary configuration file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(minimal_config, f)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def mock_tsfm_factory():
    """Create a mock TSFM factory for testing."""
    from unittest.mock import Mock
    
    # Create mock factory
    mock_factory = Mock()
    
    # Create mock models
    mock_short_model = Mock()
    mock_long_model = Mock()
    
    # Setup model loading behavior
    def mock_load_model(model_name, **kwargs):
        if model_name == 'chronos':
            return mock_short_model
        elif model_name == 'timesfm':
            return mock_long_model
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    mock_factory.load_model.side_effect = mock_load_model
    mock_factory.unload_model.return_value = None
    
    # Setup model result mocking
    mock_result = Mock()
    mock_result.predictions = np.array([1e-5, 2e-5, 3e-5, 4e-5, 5e-5])
    mock_result.quantiles = {
        '0.1': np.array([0.5e-5, 1.5e-5, 2.5e-5, 3.5e-5, 4.5e-5]),
        '0.9': np.array([1.5e-5, 2.5e-5, 3.5e-5, 4.5e-5, 5.5e-5])
    }
    mock_result.metadata = {'model_name': 'test_model'}
    
    mock_short_model.forecast.return_value = mock_result
    mock_short_model.forecast_with_covariates.return_value = mock_result
    mock_short_model.health_check.return_value = {'status': 'loaded'}
    
    mock_long_model.forecast.return_value = mock_result
    mock_long_model.forecast_with_covariates.return_value = mock_result
    mock_long_model.health_check.return_value = {'status': 'loaded'}
    
    # Setup factory helper methods
    mock_factory.create_frequency_info.return_value = Mock(freq_str='S', freq_value=9)
    mock_factory.create_covariates_input.return_value = Mock()
    
    return mock_factory


@pytest.fixture
def mock_prediction_result():
    """Create a mock prediction result for testing."""
    from chronotick_inference import PredictionResult, ModelType
    import time
    
    return PredictionResult(
        predictions=np.array([1e-5, 2e-5, 3e-5]),
        uncertainty=np.array([0.1e-5, 0.2e-5, 0.3e-5]),
        quantiles={
            '0.1': np.array([0.8e-5, 1.8e-5, 2.8e-5]),
            '0.9': np.array([1.2e-5, 2.2e-5, 3.2e-5])
        },
        confidence=0.85,
        model_type=ModelType.SHORT_TERM,
        timestamp=time.time(),
        inference_time=0.1,
        metadata={'test': True}
    )


# Test markers for different test categories
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests"
    )
    config.addinivalue_line(
        "markers", "slow: Slow running tests"
    )
    config.addinivalue_line(
        "markers", "gpu: Tests requiring GPU"
    )
    config.addinivalue_line(
        "markers", "network: Tests requiring network access"
    )


# Test helper functions
def assert_valid_offset_data(data):
    """Assert that offset data is valid."""
    assert isinstance(data, np.ndarray)
    assert data.dtype in [np.float32, np.float64]
    assert len(data) > 0
    assert np.all(np.isfinite(data))


def assert_valid_system_metrics(metrics):
    """Assert that system metrics dictionary is valid."""
    assert isinstance(metrics, dict)
    assert len(metrics) > 0
    
    for name, values in metrics.items():
        assert isinstance(name, str)
        assert isinstance(values, np.ndarray)
        assert len(values) > 0
        assert np.all(np.isfinite(values))


def assert_valid_prediction_result(result):
    """Assert that a prediction result is valid."""
    from chronotick_inference import PredictionResult
    
    assert isinstance(result, PredictionResult)
    assert isinstance(result.predictions, np.ndarray)
    assert len(result.predictions) > 0
    assert np.all(np.isfinite(result.predictions))
    assert result.confidence >= 0.0
    assert result.confidence <= 1.0
    assert result.timestamp > 0
    assert result.inference_time >= 0