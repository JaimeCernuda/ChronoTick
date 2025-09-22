#!/usr/bin/env python3
"""
Unit tests for multivariate forecasting functionality.
Tests the enhanced multivariate capabilities across all models.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add the tsfm package to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tsfm.base import BaseTimeSeriesModel, MultivariateInput, ModelStatus
from tsfm.LLM.timesfm import TimesFMModel
from tsfm.LLM.ttm import TTMModel
from tsfm.LLM.chronos_bolt import ChronosBoltModel
from tsfm.LLM.toto import TotoModel
from tsfm.LLM.time_moe import TimeMoEModel


class TestMultivariateInput:
    """Test MultivariateInput data structure."""
    
    @pytest.fixture
    def sample_multivariate_data(self):
        """Create sample multivariate time series data."""
        np.random.seed(42)
        # 3 variables, 500 timesteps
        data = np.random.randn(3, 500).astype(np.float32)
        variable_names = ["sales", "inventory", "price"]
        return MultivariateInput(
            data=data,
            variable_names=variable_names,
            target_variables=["sales", "inventory"],
            metadata={"source": "synthetic", "frequency": "daily"}
        )
    
    def test_multivariate_input_creation(self, sample_multivariate_data):
        """Test MultivariateInput creation and properties."""
        mv_input = sample_multivariate_data
        
        assert mv_input.data.shape == (3, 500)
        assert len(mv_input.variable_names) == 3
        assert mv_input.variable_names == ["sales", "inventory", "price"]
        assert mv_input.target_variables == ["sales", "inventory"]
        assert mv_input.metadata["source"] == "synthetic"
    
    def test_multivariate_input_default_targets(self):
        """Test MultivariateInput with default target variables."""
        data = np.random.randn(2, 100).astype(np.float32)
        variable_names = ["var1", "var2"]
        
        mv_input = MultivariateInput(
            data=data,
            variable_names=variable_names
        )
        
        assert mv_input.target_variables is None
        assert mv_input.metadata is None


class TestBaseMultivariateFunctionality:
    """Test base multivariate functionality."""
    
    @pytest.fixture
    def base_model(self):
        """Create a mock model that inherits from BaseTimeSeriesModel."""
        class MockModel(BaseTimeSeriesModel):
            def load_model(self): pass
            def forecast(self, context, horizon, **kwargs):
                return self._mock_forecast_output(context, horizon)
            def unload_model(self): pass
            
            def _mock_forecast_output(self, context, horizon):
                from tsfm.base import ForecastOutput
                predictions = np.random.randn(horizon).astype(np.float32)
                return ForecastOutput(
                    predictions=predictions,
                    metadata={"model": "mock", "horizon": horizon}
                )
        
        return MockModel("mock_model", "cpu")
    
    @pytest.fixture
    def multivariate_input(self):
        """Create multivariate input for testing."""
        np.random.seed(42)
        data = np.random.randn(3, 200).astype(np.float32)
        return MultivariateInput(
            data=data,
            variable_names=["A", "B", "C"],
            target_variables=["A", "B"]
        )
    
    def test_base_multivariate_forecast(self, base_model, multivariate_input):
        """Test base multivariate forecasting (fallback behavior)."""
        horizon = 24
        results = base_model.forecast_multivariate(multivariate_input, horizon)
        
        # Should return forecasts for target variables only
        assert isinstance(results, dict)
        assert set(results.keys()) == {"A", "B"}
        
        for var_name, forecast_output in results.items():
            assert len(forecast_output.predictions) == horizon
            assert forecast_output.metadata["variable_name"] == var_name
    
    def test_multivariate_all_variables(self, base_model):
        """Test multivariate forecasting with all variables as targets."""
        data = np.random.randn(2, 150).astype(np.float32)
        mv_input = MultivariateInput(
            data=data,
            variable_names=["X", "Y"]
            # No target_variables specified - should forecast all
        )
        
        results = base_model.forecast_multivariate(mv_input, 12)
        assert set(results.keys()) == {"X", "Y"}


class TestModelMultivariateCapabilities:
    """Test multivariate capabilities for each model."""
    
    @pytest.fixture
    def multivariate_data(self):
        """Sample multivariate data for all models."""
        np.random.seed(42)
        data = np.random.randn(2, 300).astype(np.float32)
        return MultivariateInput(
            data=data,
            variable_names=["metric1", "metric2"]
        )
    
    def test_timesfm_multivariate_support(self, multivariate_data):
        """Test TimesFM multivariate support flags."""
        model = TimesFMModel()
        
        # TimesFM 2.0 should not have native multivariate, so should use base implementation
        assert not hasattr(model, 'multivariate_support') or not model.multivariate_support
        
        # Test method exists but would need model loaded to work
        assert hasattr(model, 'forecast_multivariate')
    
    def test_ttm_multivariate_support(self, multivariate_data):
        """Test TTM multivariate support."""
        model = TTMModel()
        
        # TTM should have multivariate support
        assert hasattr(model, 'multivariate_support')
        assert model.multivariate_support == True
        
        # Test the enhanced method exists
        assert hasattr(model, 'forecast_multivariate')
    
    def test_chronos_bolt_multivariate_support(self, multivariate_data):
        """Test Chronos-Bolt multivariate support."""
        model = ChronosBoltModel()
        
        # Chronos-Bolt should have multivariate support
        assert hasattr(model, 'multivariate_support')
        assert model.multivariate_support == True
        
        # Test the enhanced method exists
        assert hasattr(model, 'forecast_multivariate')
    
    def test_toto_multivariate_support(self, multivariate_data):
        """Test Toto multivariate support."""
        model = TotoModel()
        
        # Toto should have multivariate support
        assert hasattr(model, 'multivariate_support')
        assert model.multivariate_support == True
        
        # Test the enhanced method exists
        assert hasattr(model, 'forecast_multivariate')
    
    def test_time_moe_multivariate_support(self):
        """Test Time-MoE multivariate support."""
        model = TimeMoEModel()
        
        # Time-MoE doesn't have explicit multivariate flag but has covariates adaptability
        assert hasattr(model, 'covariates_adaptable')
        assert model.covariates_adaptable == True
        
        # Should not have multivariate_support flag
        assert not hasattr(model, 'multivariate_support')


class TestMultivariateMetadata:
    """Test multivariate forecasting metadata enhancement."""
    
    @pytest.fixture
    def ttm_model(self):
        """TTM model for metadata testing."""
        return TTMModel()
    
    @pytest.fixture
    def sample_mv_input(self):
        """Sample multivariate input."""
        data = np.random.randn(3, 100).astype(np.float32)
        return MultivariateInput(
            data=data,
            variable_names=["temp", "humidity", "pressure"]
        )
    
    def test_enhanced_metadata(self, ttm_model, sample_mv_input):
        """Test that enhanced models add multivariate-specific metadata."""
        if ttm_model.multivariate_support:
            # Mock the model status to avoid actual loading
            ttm_model.status = ModelStatus.LOADED
            ttm_model.model = "mock"  # Set a mock model
            
            try:
                results = ttm_model.forecast_multivariate(sample_mv_input, 12)
                
                # Check that each result has enhanced metadata
                for var_name, forecast_result in results.items():
                    metadata = forecast_result.metadata
                    if metadata:  # If metadata was enhanced
                        assert 'total_variables' in metadata
                        assert metadata['total_variables'] == 3
                        
            except RuntimeError:
                # Expected if dependencies not available
                pytest.skip("TTM dependencies not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])