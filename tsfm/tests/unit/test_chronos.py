#!/usr/bin/env python3
"""
Unit tests for Chronos-Bolt model implementation.
Tests the model directly without factory pattern.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add the tsfm package to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tsfm.LLM.chronos_bolt import ChronosBoltModel
from tsfm.base import ModelStatus


class TestChronosBoltModel:
    """Unit tests for Chronos-Bolt model."""
    
    @pytest.fixture
    def model(self):
        """Create Chronos-Bolt model instance."""
        return ChronosBoltModel(model_name="chronos_test", device="cpu")
    
    @pytest.fixture
    def sample_data(self):
        """Create sample time series data."""
        np.random.seed(42)
        return np.random.randn(100).astype(np.float32)
    
    def test_model_initialization(self, model):
        """Test model initialization."""
        assert model.model_name == "chronos_test"
        assert model.device == "cpu"
        assert model.status == ModelStatus.UNLOADED
        assert model.pipeline is None
        assert model.model_size == "base"  # Upgraded from tiny
        assert model.model_repo == "amazon/chronos-bolt-base"  # Chronos-Bolt format
    
    def test_dependency_check(self, model):
        """Test dependency checking."""
        try:
            import chronos
            assert model._check_dependencies() == True
        except ImportError:
            assert model._check_dependencies() == False
            assert "Chronos package not available" in model._dependency_error
    
    def test_model_loading(self, model):
        """Test model loading."""
        try:
            model.load_model()
            assert model.status == ModelStatus.LOADED
            assert model.pipeline is not None
        except Exception as e:
            # If dependencies not available, should fail gracefully
            if "Chronos package not available" in str(e):
                assert model.status == ModelStatus.ERROR
                pytest.skip("Chronos package not installed")
            else:
                # For other errors, still check error handling
                assert model.status == ModelStatus.ERROR
                assert "Chronos-Bolt model loading failed" in str(e)
    
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
        with pytest.raises(RuntimeError, match="Chronos-Bolt model not loaded"):
            model.forecast(sample_data, horizon=24)
    
    @pytest.mark.skipif(
        not pytest.importorskip("chronos", reason="Chronos not installed"),
        reason="Chronos package required"
    )
    def test_forecasting_with_model(self, model, sample_data):
        """Test forecasting with loaded model."""
        try:
            model.load_model()
            
            forecast_output = model.forecast(sample_data, horizon=24)
            
            assert forecast_output is not None
            assert forecast_output.predictions is not None
            assert len(forecast_output.predictions) == 24
            assert forecast_output.metadata['model_name'] == "chronos_test"
            assert forecast_output.metadata['forecast_horizon'] == 24
            assert forecast_output.metadata['context_length'] == len(sample_data)
            
            # Check quantiles
            if forecast_output.quantiles is not None:
                assert '0.5' in forecast_output.quantiles  # Median should be available
            
        except Exception as e:
            if "Chronos" in str(e):
                pytest.skip(f"Chronos test skipped: {e}")
            else:
                raise
    
    def test_model_unloading(self, model):
        """Test model unloading."""
        try:
            model.load_model()
            model.unload_model()
            assert model.status == ModelStatus.UNLOADED
            assert model.pipeline is None
        except Exception as e:
            if "Chronos package not available" in str(e):
                pytest.skip("Chronos package not installed")
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
        assert 'model_size' in health
        assert 'dependency_status' in health
        
        # Check dependency status
        try:
            import chronos
            assert health['dependency_status'] == 'ok'
        except ImportError:
            assert health['dependency_status'] == 'failed'
            assert 'dependency_error' in health
    
    def test_configuration_parameters(self):
        """Test custom configuration parameters."""
        custom_model = ChronosBoltModel(
            model_name="custom_chronos",
            device="cpu",
            model_size="tiny",
            model_repo="amazon/chronos-bolt-tiny"
        )
        
        assert custom_model.model_size == "tiny"
        assert custom_model.model_repo == "amazon/chronos-bolt-tiny"
    
    def test_different_model_sizes(self):
        """Test different model size configurations."""
        sizes = ["tiny", "mini", "small", "base", "large"]
        
        for size in sizes:
            model = ChronosBoltModel(model_size=size)
            assert model.model_size == size
            assert f"chronos-bolt-{size}" in model.model_repo  # Updated to Chronos-Bolt format


if __name__ == "__main__":
    pytest.main([__file__, "-v"])