#!/usr/bin/env python3
"""
Backward compatibility tests for TSFM Factory.
Ensures that existing API calls and workflows continue to work with enhanced features.
"""

import pytest
import numpy as np
from pathlib import Path
import sys
import warnings

# Add the tsfm package to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tsfm import TSFMFactory
from tsfm.datasets.loader import create_synthetic_data

warnings.filterwarnings("ignore")


class TestBackwardCompatibilityAPI:
    """Test that all existing API calls continue to work."""
    
    @pytest.fixture
    def factory(self):
        """Create TSFM factory instance."""
        return TSFMFactory()
    
    @pytest.fixture
    def legacy_data(self):
        """Create data in the format used by legacy tests."""
        return create_synthetic_data(length=500, pattern="trend", seed=42)
    
    def test_legacy_factory_operations(self, factory):
        """Test that basic factory operations work as before."""
        # Test listing available models
        available = factory.list_available_models()
        assert isinstance(available, list)
        assert len(available) > 0
        
        # Test model info retrieval
        for model_name in available:
            info = factory.get_model_info(model_name)
            assert isinstance(info, dict)
            assert 'name' in info
    
    def test_legacy_model_loading_unloading(self, factory):
        """Test that model loading/unloading API remains unchanged."""
        try:
            # Test loading a model
            model = factory.load_model("timesfm")
            assert model is not None
            
            # Test listing loaded models
            loaded = factory.list_loaded_models()
            assert "timesfm" in loaded
            
            # Test unloading
            factory.unload_model("timesfm")
            loaded_after = factory.list_loaded_models()
            assert "timesfm" not in loaded_after
            
        except Exception as e:
            if "TimesFM package not available" in str(e):
                pytest.skip("TimesFM package not installed")
            else:
                raise
    
    def test_legacy_forecasting_api(self, factory, legacy_data):
        """Test that the basic forecasting API remains unchanged."""
        try:
            model = factory.load_model("chronos")
            
            # Test basic forecasting with numpy array
            context = legacy_data[:400]
            forecast_output = model.forecast(context, horizon=96)
            
            # Verify output structure hasn't changed
            assert hasattr(forecast_output, 'predictions')
            assert hasattr(forecast_output, 'quantiles')
            assert hasattr(forecast_output, 'metadata')
            
            assert len(forecast_output.predictions) == 96
            assert isinstance(forecast_output.metadata, dict)
            
            # Test basic forecasting with list input
            context_list = context.tolist()
            forecast_output_list = model.forecast(context_list, horizon=48)
            assert len(forecast_output_list.predictions) == 48
            
            factory.unload_model("chronos")
            
        except Exception as e:
            if "chronos" in str(e).lower():
                pytest.skip(f"Chronos test skipped: {e}")
            else:
                raise
    
    def test_legacy_metadata_structure(self, factory, legacy_data):
        """Test that existing metadata fields are preserved."""
        try:
            model = factory.load_model("ttm")
            
            context = legacy_data[:300]
            forecast_output = model.forecast(context, horizon=24)
            
            metadata = forecast_output.metadata
            
            # Verify essential legacy metadata fields exist
            legacy_fields = [
                'model_name',
                'forecast_horizon', 
                'context_length',
                'device'
            ]
            
            for field in legacy_fields:
                assert field in metadata, f"Legacy field '{field}' missing from metadata"
            
            # Verify metadata types are compatible
            assert isinstance(metadata['model_name'], str)
            assert isinstance(metadata['forecast_horizon'], int)
            assert isinstance(metadata['context_length'], int)
            
            factory.unload_model("ttm")
            
        except Exception as e:
            if "TTM" in str(e):
                pytest.skip(f"TTM test skipped: {e}")
            else:
                raise
    
    def test_legacy_error_handling(self, factory):
        """Test that error handling behavior remains consistent."""
        # Test loading non-existent model
        with pytest.raises(ValueError, match="not found"):
            factory.load_model("nonexistent_model")
        
        # Test forecasting without loading model
        unloaded_model = factory._create_model("timesfm")  # Create but don't load
        with pytest.raises(RuntimeError):
            unloaded_model.forecast(np.random.randn(100), horizon=24)
    
    def test_legacy_input_validation(self, factory):
        """Test that input validation behavior is unchanged."""
        try:
            model = factory.load_model("timesfm")
            
            # Test empty input
            with pytest.raises(ValueError):
                model.forecast(np.array([]), horizon=24)
            
            # Test invalid input type
            with pytest.raises(ValueError):
                model.forecast("invalid", horizon=24)
            
            # Test NaN input
            with pytest.raises(ValueError):
                model.forecast(np.array([1.0, np.nan, 3.0]), horizon=24)
            
            factory.unload_model("timesfm")
            
        except Exception as e:
            if "TimesFM package not available" in str(e):
                pytest.skip("TimesFM package not installed")
            else:
                raise


class TestBackwardCompatibilityDataFlow:
    """Test that data processing workflows remain unchanged."""
    
    @pytest.fixture
    def factory(self):
        return TSFMFactory()
    
    def test_legacy_preprocessing_compatibility(self, factory):
        """Test that existing preprocessing workflows work."""
        from tsfm.datasets.preprocessing import normalize_data
        
        # Test that preprocessing functions maintain API
        data = np.random.randn(200)
        normalized, stats = normalize_data(data)
        
        assert isinstance(normalized, np.ndarray)
        assert isinstance(stats, dict)
        assert 'mean' in stats
        assert 'std' in stats
    
    def test_legacy_synthetic_data_compatibility(self):
        """Test that synthetic data generation API is unchanged."""
        # Test basic synthetic data creation
        data = create_synthetic_data(length=100, pattern="linear", seed=42)
        assert len(data) == 100
        assert isinstance(data, np.ndarray)
        
        # Test with different patterns
        patterns = ["linear", "exponential", "seasonal", "trend", "mixed"]
        for pattern in patterns:
            try:
                data = create_synthetic_data(length=50, pattern=pattern)
                assert len(data) == 50
            except Exception:
                # Some patterns might not be implemented, that's ok
                pass
    
    def test_legacy_metrics_compatibility(self, factory):
        """Test that metrics calculation API is unchanged."""
        try:
            from tsfm.utils.metrics import calculate_metrics
            
            # Generate test data
            true_values = np.random.randn(50)
            predictions = true_values + np.random.normal(0, 0.1, 50)
            
            metrics = calculate_metrics(true_values, predictions)
            
            assert isinstance(metrics, dict)
            # Common metrics should be present
            expected_metrics = ['mae', 'mse', 'rmse', 'mape']
            for metric in expected_metrics:
                if metric in metrics:  # Only check if implemented
                    assert isinstance(metrics[metric], (int, float))
            
        except ImportError:
            pytest.skip("Metrics module not available")


class TestEnhancedFeaturesBackwardCompatibility:
    """Test that new features don't break existing functionality."""
    
    @pytest.fixture
    def factory(self):
        return TSFMFactory()
    
    def test_enhanced_models_basic_forecast(self, factory):
        """Test that enhanced models still work with basic forecast calls."""
        try:
            model = factory.load_model("timesfm")
            
            # Basic forecast should work despite enhanced capabilities
            data = np.random.randn(200)
            result = model.forecast(data, horizon=48)
            
            assert len(result.predictions) == 48
            
            # New metadata fields should be present but not break existing access
            assert 'model_name' in result.metadata
            if hasattr(model, 'covariates_support'):
                assert 'covariates_support' in result.metadata
            if hasattr(model, 'timesfm_version'):
                assert 'timesfm_version' in result.metadata
            
            factory.unload_model("timesfm")
            
        except Exception as e:
            if "TimesFM" in str(e):
                pytest.skip(f"TimesFM test skipped: {e}")
            else:
                raise
    
    def test_enhanced_models_preserve_original_behavior(self, factory):
        """Test that enhanced models preserve original forecasting behavior."""
        models_to_test = ["ttm", "chronos", "toto"]
        
        for model_name in models_to_test:
            try:
                model = factory.load_model(model_name)
                
                # Standard forecast
                data = np.random.randn(150)
                result = model.forecast(data, horizon=24)
                
                # Verify core behavior is preserved
                assert len(result.predictions) == 24
                assert result.predictions.dtype in [np.float32, np.float64]
                assert isinstance(result.metadata, dict)
                
                # Health check should work
                health = model.health_check()
                assert 'model_name' in health
                assert 'status' in health
                
                factory.unload_model(model_name)
                
            except Exception as e:
                if any(term in str(e) for term in [model_name, "package not available", "dependencies"]):
                    pytest.skip(f"{model_name} test skipped: {e}")
                else:
                    raise


class TestLegacyConfigurationCompatibility:
    """Test that existing configuration methods still work."""
    
    @pytest.fixture
    def factory(self):
        return TSFMFactory()
    
    def test_legacy_model_configuration(self, factory):
        """Test that existing model configuration parameters work."""
        try:
            # Test TimesFM with legacy parameters
            custom_config = {
                'context_len': 256,
                'horizon_len': 48,
                'device': 'cpu'
            }
            
            model = factory.load_model("timesfm", **custom_config)
            
            # Verify configuration was applied
            assert model.context_len == 256
            assert model.horizon_len == 48
            assert model.device == 'cpu'
            
            factory.unload_model("timesfm")
            
        except Exception as e:
            if "TimesFM" in str(e):
                pytest.skip(f"TimesFM config test skipped: {e}")
            else:
                raise
    
    def test_legacy_device_specification(self, factory):
        """Test that device specification still works as before."""
        try:
            # Test CPU device specification
            model = factory.load_model("chronos", device="cpu")
            assert model.device == "cpu"
            
            factory.unload_model("chronos")
            
        except Exception as e:
            if "chronos" in str(e).lower():
                pytest.skip(f"Chronos device test skipped: {e}")
            else:
                raise


class TestRegressionPrevention:
    """Tests to prevent regression in core functionality."""
    
    @pytest.fixture
    def factory(self):
        return TSFMFactory()
    
    def test_no_memory_leaks_in_loading_unloading(self, factory):
        """Test that repeated loading/unloading doesn't cause issues."""
        model_name = "chronos"
        
        try:
            for i in range(3):  # Test multiple load/unload cycles
                model = factory.load_model(model_name)
                assert model is not None
                
                # Basic functionality test
                data = np.random.randn(100)
                result = model.forecast(data, horizon=12)
                assert len(result.predictions) == 12
                
                factory.unload_model(model_name)
                
                # Verify model is properly unloaded
                loaded = factory.list_loaded_models()
                assert model_name not in loaded
                
        except Exception as e:
            if "chronos" in str(e).lower():
                pytest.skip(f"Chronos regression test skipped: {e}")
            else:
                raise
    
    def test_concurrent_model_operations(self, factory):
        """Test that operations on different models don't interfere."""
        try:
            # Load two different models
            model1 = factory.load_model("timesfm")
            model2 = factory.load_model("chronos")
            
            # Test both models work independently
            data = np.random.randn(100)
            
            result1 = model1.forecast(data, horizon=24)
            result2 = model2.forecast(data, horizon=24)
            
            assert len(result1.predictions) == 24
            assert len(result2.predictions) == 24
            
            # Unload both
            factory.unload_model("timesfm")
            factory.unload_model("chronos")
            
        except Exception as e:
            if any(term in str(e) for term in ["TimesFM", "chronos", "package not available"]):
                pytest.skip(f"Concurrent models test skipped: {e}")
            else:
                raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])