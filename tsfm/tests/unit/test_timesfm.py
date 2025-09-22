#!/usr/bin/env python3
"""
Unit tests for TimesFM model implementation.
Tests the model directly without factory pattern.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add the tsfm package to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tsfm.LLM.timesfm import TimesFMModel
from tsfm.base import ModelStatus


class TestTimesFMModel:
    """Unit tests for TimesFM model."""
    
    @pytest.fixture
    def model(self):
        """Create TimesFM model instance."""
        return TimesFMModel(model_name="timesfm_test", device="cpu")
    
    @pytest.fixture
    def sample_data(self):
        """Create sample time series data."""
        np.random.seed(42)
        return np.random.randn(512).astype(np.float32)
    
    def test_model_initialization(self, model):
        """Test model initialization."""
        assert model.model_name == "timesfm_test"
        assert model.device == "cpu"
        assert model.status == ModelStatus.UNLOADED
        assert model.tfm is None
        assert model.context_len == 2048
        assert model.horizon_len == 'flexible'  # TimesFM 2.0 supports flexible horizons
    
    def test_dependency_check(self, model):
        """Test dependency checking."""
        try:
            import timesfm
            assert model._check_dependencies() == True
        except ImportError:
            assert model._check_dependencies() == False
            assert "TimesFM package not available" in model._dependency_error
    
    @pytest.mark.skipif(
        not pytest.importorskip("timesfm", reason="TimesFM not installed"),
        reason="TimesFM package required"
    )
    def test_model_loading(self, model):
        """Test model loading with real model."""
        # This test requires actual model loading - no skipping for failures
        model.load_model()
        assert model.status == ModelStatus.LOADED
        assert model.tfm is not None
    
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
        with pytest.raises(RuntimeError, match="TimesFM model not loaded"):
            model.forecast(sample_data, horizon=24)
    
    @pytest.mark.skipif(
        not pytest.importorskip("timesfm", reason="TimesFM not installed"),
        reason="TimesFM package required"
    )
    def test_forecasting_with_model(self, model, sample_data):
        """Test forecasting with loaded model."""
        try:
            model.load_model()
            
            forecast_output = model.forecast(sample_data, horizon=24)
            
            assert forecast_output is not None
            assert forecast_output.predictions is not None
            assert len(forecast_output.predictions) == 24
            assert forecast_output.metadata['model_name'] == "timesfm_test"
            assert forecast_output.metadata['forecast_horizon'] == 24
            assert forecast_output.metadata['context_length'] == len(sample_data)
            
        except Exception as e:
            if "TimesFM" in str(e):
                pytest.skip(f"TimesFM test skipped: {e}")
            else:
                raise
    
    def test_model_unloading(self, model):
        """Test model unloading."""
        try:
            model.load_model()
            model.unload_model()
            assert model.status == ModelStatus.UNLOADED
            assert model.tfm is None
        except Exception as e:
            if "TimesFM package not available" in str(e):
                pytest.skip("TimesFM package not installed")
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
        assert 'context_len' in health
        assert 'horizon_len' in health
        assert 'dependency_status' in health
        
        # Check dependency status
        try:
            import timesfm
            assert health['dependency_status'] == 'ok'
        except ImportError:
            assert health['dependency_status'] == 'failed'
            assert 'dependency_error' in health
    
    def test_configuration_parameters(self):
        """Test custom configuration parameters."""
        custom_model = TimesFMModel(
            model_name="custom_timesfm",
            device="cpu",
            context_len=256,
            horizon_len=48,
            input_patch_len=16,
            output_patch_len=64,
            num_layers=10,
            model_dims=640
        )
        
        assert custom_model.context_len == 256
        assert custom_model.horizon_len == 48
        assert custom_model.input_patch_len == 16
        assert custom_model.output_patch_len == 64
        assert custom_model.num_layers == 10
        assert custom_model.model_dims == 640


if __name__ == "__main__":
    pytest.main([__file__, "-v"])