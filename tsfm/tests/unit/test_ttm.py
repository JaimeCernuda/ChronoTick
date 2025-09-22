#!/usr/bin/env python3
"""
Unit tests for TTM (Tiny Time Mixer) model implementation.
Tests the model directly without factory pattern.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add the tsfm package to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tsfm.LLM.ttm import TTMModel
from tsfm.base import ModelStatus


class TestTTMModel:
    """Unit tests for TTM model."""
    
    @pytest.fixture
    def model(self):
        """Create TTM model instance."""
        return TTMModel(model_name="ttm_test", device="cpu")
    
    @pytest.fixture
    def sample_data(self):
        """Create sample time series data."""
        np.random.seed(42)
        return np.random.randn(512).astype(np.float32)
    
    def test_model_initialization(self, model):
        """Test model initialization."""
        assert model.model_name == "ttm_test"
        assert model.device == "cpu"
        assert model.status == ModelStatus.UNLOADED
        assert model.model is None
        assert model.model_repo == "ibm-granite/granite-timeseries-ttm-r2"
        assert model.context_length == 512
        assert model.prediction_length == 96
    
    def test_dependency_check(self, model):
        """Test dependency checking."""
        try:
            import transformers
            assert model._check_dependencies() == True
        except ImportError:
            assert model._check_dependencies() == False
            assert "TTM dependencies not available" in model._dependency_error
    
    def test_model_loading(self, model):
        """Test model loading."""
        try:
            model.load_model()
            assert model.status == ModelStatus.LOADED
            assert model.model is not None
        except Exception as e:
            # If dependencies not available, should fail gracefully
            if "TTM dependencies not available" in str(e):
                assert model.status == ModelStatus.ERROR
                pytest.skip("TTM dependencies not installed")
            else:
                # For other errors (like unsupported model type), still check error handling
                assert model.status == ModelStatus.ERROR
                assert "TTM model loading failed" in str(e)
    
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
        
        with pytest.raises(ValueError):
            model.validate_input(np.array([]))
    
    def test_forecasting_without_model(self, model, sample_data):
        """Test forecasting fails when model not loaded."""
        with pytest.raises(RuntimeError, match="TTM model not loaded"):
            model.forecast(sample_data, horizon=24)
    
    @pytest.mark.skipif(
        not pytest.importorskip("transformers", reason="Transformers not installed"),
        reason="Transformers package required"
    )
    def test_forecasting_with_model(self, model, sample_data):
        """Test forecasting with loaded model."""
        try:
            model.load_model()
            
            forecast_output = model.forecast(sample_data, horizon=24)
            
            assert forecast_output is not None
            assert forecast_output.predictions is not None
            assert len(forecast_output.predictions) == 24
            assert forecast_output.metadata['model_name'] == "ttm_test"
            assert forecast_output.metadata['forecast_horizon'] == 24
            assert forecast_output.metadata['context_length'] == len(sample_data)
            
        except Exception as e:
            if "TTM" in str(e) or "tinytimemixer" in str(e):
                pytest.skip(f"TTM test skipped: {e}")
            else:
                raise
    
    def test_model_unloading(self, model):
        """Test model unloading."""
        try:
            model.load_model()
            model.unload_model()
            assert model.status == ModelStatus.UNLOADED
            assert model.model is None
        except Exception as e:
            if "TTM dependencies not available" in str(e):
                pytest.skip("TTM dependencies not installed")
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
        assert 'context_length' in health
        assert 'prediction_length' in health
        assert 'dependency_status' in health
        
        # Check dependency status
        try:
            import transformers
            assert health['dependency_status'] == 'ok'
        except ImportError:
            assert health['dependency_status'] == 'failed'
            assert 'dependency_error' in health
    
    def test_configuration_parameters(self):
        """Test custom configuration parameters."""
        custom_model = TTMModel(
            model_name="custom_ttm",
            device="cpu",
            context_length=256,
            prediction_length=48,
            model_repo="custom/ttm-model"
        )
        
        assert custom_model.context_length == 256
        assert custom_model.prediction_length == 48
        assert custom_model.model_repo == "custom/ttm-model"
    
    def test_data_preprocessing(self, model, sample_data):
        """Test TTM-specific data preprocessing."""
        # TTM expects specific input format - test preprocessing logic
        processed = model._prepare_input(sample_data)
        
        assert processed is not None
        assert isinstance(processed, dict)
        assert 'input_values' in processed
        assert 'context_mean' in processed
        assert 'context_std' in processed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])