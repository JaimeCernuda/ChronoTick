#!/usr/bin/env python3
"""
Unit tests for Time-MoE model implementation.
Tests the model directly without factory pattern.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add the tsfm package to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tsfm.LLM.time_moe import TimeMoEModel
from tsfm.base import ModelStatus


class TestTimeMoEModel:
    """Unit tests for Time-MoE model."""
    
    @pytest.fixture
    def model(self):
        """Create Time-MoE model instance."""
        return TimeMoEModel(model_name="time_moe_test", device="cpu")
    
    @pytest.fixture
    def sample_data(self):
        """Create sample time series data."""
        np.random.seed(42)
        return np.random.randn(512).astype(np.float32)
    
    def test_model_initialization(self, model):
        """Test model initialization."""
        assert model.model_name == "time_moe_test"
        assert model.device == "cpu"
        assert model.status == ModelStatus.UNLOADED
        assert model.model is None
        assert model.model_repo == "Maple728/TimeMoE-200M"
        assert model.variant == "200M"
        assert model.max_context_length == 4096
        assert model.prediction_length == 96
    
    def test_dependency_check(self, model):
        """Test dependency checking."""
        try:
            import transformers
            # Check version compatibility
            from transformers import __version__
            version_parts = __version__.split('.')
            major, minor = int(version_parts[0]), int(version_parts[1])
            
            if major > 4 or (major == 4 and minor >= 49):
                # Version too high for Time-MoE
                assert model._check_dependencies() == False
                assert "requires transformers==4.40.1" in model._dependency_error
            else:
                assert model._check_dependencies() == True
        except ImportError:
            assert model._check_dependencies() == False
            assert "Required dependencies not available" in model._dependency_error
    
    def test_version_compatibility_check(self, model):
        """Test transformers version compatibility checking."""
        try:
            import transformers
            from transformers import __version__
            
            # Test dependency checking which includes version logic
            result = model._check_dependencies()
            
            version_parts = __version__.split('.')
            major, minor = int(version_parts[0]), int(version_parts[1])
            
            if major == 4 and minor == 40:
                assert result == True
            else:
                assert result == False
                assert "requires transformers==4.40.1" in model._dependency_error
                
        except ImportError:
            pytest.skip("Transformers not installed")
    
    def test_model_loading(self, model):
        """Test model loading."""
        try:
            model.load_model()
            assert model.status == ModelStatus.LOADED
            assert model.model is not None
            # Time-MoE doesn't use a tokenizer, just check model is loaded
        except Exception as e:
            # If dependencies not available or version incompatible, should fail gracefully
            if "Time-MoE requires transformers==4.40.1" in str(e):
                assert model.status == ModelStatus.ERROR
                pytest.skip("Time-MoE requires transformers==4.40.1")
            elif "Transformers not available" in str(e):
                assert model.status == ModelStatus.ERROR
                pytest.skip("Transformers not installed")
            else:
                # For other errors, still check error handling
                assert model.status == ModelStatus.ERROR
                assert "Time-MoE model loading failed" in str(e)
    
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
        with pytest.raises(RuntimeError, match="Time-MoE model not loaded"):
            model.forecast(sample_data, horizon=24)
    
    def test_forecasting_with_version_incompatibility(self, model, sample_data):
        """Test forecasting with version incompatibility."""
        try:
            import transformers
            from transformers import __version__
            
            if not model._check_dependencies():
                with pytest.raises(RuntimeError, match="Time-MoE requires transformers==4.40.1"):
                    model.forecast(sample_data, horizon=24)
            else:
                # If version is compatible, test actual forecasting
                model.load_model()
                forecast_output = model.forecast(sample_data, horizon=24)
                
                assert forecast_output is not None
                assert forecast_output.predictions is not None
                assert len(forecast_output.predictions) == 24
                assert forecast_output.metadata['model_name'] == "time_moe_test"
                
        except ImportError:
            pytest.skip("Transformers not installed")
        except Exception as e:
            if "Time-MoE requires transformers==4.40.1" in str(e):
                pytest.skip("Time-MoE version incompatibility expected")
            else:
                raise
    
    def test_model_unloading(self, model):
        """Test model unloading."""
        try:
            if model._check_dependencies():
                model.load_model()
                model.unload_model()
                assert model.status == ModelStatus.UNLOADED
                assert model.model is None
                assert model.tokenizer is None
            else:
                # Test unloading even if loading failed
                model.unload_model()
                assert model.status == ModelStatus.UNLOADED
        except Exception as e:
            if "Time-MoE requires transformers==4.40.1" in str(e):
                pytest.skip("Time-MoE version incompatibility")
            else:
                model.unload_model()
                assert model.status == ModelStatus.UNLOADED
    
    def test_health_check(self, model):
        """Test health check functionality."""
        health = model.health_check()
        
        assert 'model_name' in health
        assert 'device' in health
        assert 'status' in health
        assert 'model_repo' in health
        assert 'max_context_length' in health
        assert 'prediction_length' in health
        assert 'dependency_status' in health
        # Check basic health status attributes
        
        # Check dependency status
        try:
            import transformers
            if model._check_dependencies():
                assert health['dependency_status'] == 'ok'
            else:
                assert health['dependency_status'] == 'failed'
        except ImportError:
            assert health['dependency_status'] == 'failed'
            assert 'dependency_error' in health
    
    def test_configuration_parameters(self):
        """Test custom configuration parameters."""
        custom_model = TimeMoEModel(
            model_name="custom_time_moe",
            device="cpu",
            max_context_length=256,
            prediction_length=48,
            model_repo="custom/time-moe-model"
        )
        
        assert custom_model.max_context_length == 256
        assert custom_model.prediction_length == 48
        assert custom_model.model_repo == "custom/time-moe-model"
    
    def test_sequence_length_handling(self, model):
        """Test sequence length requirements."""
        # Test different input lengths
        short_data = np.random.randn(100).astype(np.float32)
        long_data = np.random.randn(1000).astype(np.float32)
        
        # Model should handle different sequence lengths
        validated_short = model.validate_input(short_data)
        validated_long = model.validate_input(long_data)
        
        assert isinstance(validated_short, np.ndarray)
        assert isinstance(validated_long, np.ndarray)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])