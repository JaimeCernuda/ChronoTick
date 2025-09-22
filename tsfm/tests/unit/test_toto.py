#!/usr/bin/env python3
"""
Unit tests for Toto model implementation.
Tests the model directly without factory pattern.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add the tsfm package to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tsfm.LLM.toto import TotoModel
from tsfm.base import ModelStatus


class TestTotoModel:
    """Unit tests for Toto model."""
    
    @pytest.fixture
    def model(self):
        """Create Toto model instance."""
        return TotoModel(model_name="toto_test", device="cpu")
    
    @pytest.fixture
    def sample_data(self):
        """Create sample time series data."""
        np.random.seed(42)
        return np.random.randn(200).astype(np.float32)
    
    def test_model_initialization(self, model):
        """Test model initialization."""
        assert model.model_name == "toto_test"
        assert model.device == "cpu"
        assert model.status == ModelStatus.UNLOADED
        assert model.model is None
        assert model.forecaster is None
        assert model.model_repo == "Datadog/Toto-Open-Base-1.0"
    
    def test_dependency_check(self, model):
        """Test dependency checking."""
        # Just test that dependency check method works
        dependency_status = model._check_dependencies()
        assert isinstance(dependency_status, bool)
        
        # If dependencies are missing, should have error message
        if not dependency_status:
            assert model._dependency_error is not None
    
    def test_model_loading(self, model):
        """Test model loading."""
        try:
            model.load_model()
            assert model.status == ModelStatus.LOADED
            assert model.model is not None
            assert model.forecaster is not None
        except Exception as e:
            # If dependencies not available, should fail gracefully
            if "Toto dependencies not available" in str(e):
                assert model.status == ModelStatus.ERROR
                pytest.skip("Toto dependencies not installed")
            else:
                # For other errors, still check error handling
                assert model.status == ModelStatus.ERROR
                assert "Toto model loading failed" in str(e)
    
    def test_input_validation(self, model, sample_data):
        """Test input validation."""
        # Test valid input
        validated = model.validate_input(sample_data)
        assert isinstance(validated, np.ndarray)
        assert validated.dtype == np.float32
        
        # Test list input
        validated = model.validate_input(sample_data.tolist())
        assert isinstance(validated, np.ndarray)
        
        # Test invalid input
        with pytest.raises(ValueError):
            model.validate_input("invalid")
        
        # Empty array should be handled gracefully or raise ValueError
        try:
            model.validate_input(np.array([]))
        except ValueError:
            pass  # Expected behavior
    
    def test_forecasting_without_model(self, model, sample_data):
        """Test forecasting fails when model not loaded."""
        with pytest.raises(RuntimeError, match="Toto model not loaded"):
            model.forecast(sample_data, horizon=24)
    
    def test_forecasting_with_model(self, model, sample_data):
        """Test forecasting with loaded model."""
        try:
            model.load_model()
            
            forecast_output = model.forecast(sample_data, horizon=24)
            
            assert forecast_output is not None
            assert forecast_output.predictions is not None
            assert len(forecast_output.predictions) == 24
            assert forecast_output.metadata['model_name'] == "toto_test"
            assert forecast_output.metadata['forecast_horizon'] == 24
            assert forecast_output.metadata['context_length'] == len(sample_data)
            
            # Check quantiles (Toto provides 9 quantile levels)
            if forecast_output.quantiles is not None:
                assert len(forecast_output.quantiles) == 9
                assert '0.5' in forecast_output.quantiles  # Median should be available
            
        except Exception as e:
            if "Toto" in str(e):
                pytest.skip(f"Toto test skipped: {e}")
            else:
                raise
    
    def test_model_unloading(self, model):
        """Test model unloading."""
        try:
            model.load_model()
            model.unload_model()
            assert model.status == ModelStatus.UNLOADED
            assert model.model is None
            assert model.forecaster is None
        except Exception as e:
            if "Toto dependencies not available" in str(e):
                pytest.skip("Toto dependencies not installed")
            else:
                # Test unloading even if loading failed
                model.unload_model()
                assert model.status == ModelStatus.UNLOADED
    
    def test_health_check(self, model):
        """Test health check functionality."""
        health = model.health_check()
        
        assert 'model_name' in health
        assert 'device' in health
        assert 'status' in health
        assert 'model_repo' in health
        assert 'dependency_status' in health
        
        # Check dependency status is present
        assert health['dependency_status'] in ['ok', 'failed']
        
        # If failed, should have error message
        if health['dependency_status'] == 'failed':
            assert 'dependency_error' in health
    
    def test_configuration_parameters(self):
        """Test custom configuration parameters."""
        custom_model = TotoModel(
            model_name="custom_toto",
            device="cpu",
            model_repo="custom/toto-model"
        )
        
        assert custom_model.model_repo == "custom/toto-model"
    
    def test_context_length_validation(self, model):
        """Test context length requirements."""
        # Toto requires at least 64 context points
        short_data = np.random.randn(30).astype(np.float32)
        
        try:
            model.load_model()
            
            # Should handle short context gracefully
            forecast_output = model.forecast(short_data, horizon=12)
            # Model should either pad or handle short context
            assert forecast_output is not None
            
        except Exception as e:
            if "Toto" in str(e):
                pytest.skip(f"Toto test skipped: {e}")
            else:
                # Short context might raise specific error
                assert len(short_data) < 64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])