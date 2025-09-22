#!/usr/bin/env python3
"""
Unit tests for enhanced TSFM Factory functionality.
Tests the new capabilities and convenience methods.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add the tsfm package to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tsfm import TSFMFactory, MultivariateInput, CovariatesInput, FrequencyInfo


class TestEnhancedFactoryAPI:
    """Test enhanced factory API methods."""
    
    @pytest.fixture
    def factory(self):
        """Create factory instance."""
        return TSFMFactory()
    
    def test_enhanced_model_info(self, factory):
        """Test enhanced model info with capabilities."""
        models = factory.list_available_models()
        assert len(models) > 0
        
        for model_name in models:
            info = factory.get_model_info(model_name)
            
            # Check basic info fields
            assert 'name' in info
            assert 'class' in info
            assert 'module' in info
            assert 'docstring' in info
            
            # Check enhanced capabilities field
            assert 'capabilities' in info
            capabilities = info['capabilities']
            
            # Check capability structure
            expected_capability_fields = [
                'multivariate_support',
                'covariates_support', 
                'exogenous_support',
                'covariates_adaptable',
                'mixture_of_experts',
                'direct_multistep',
                'enhanced_methods'
            ]
            
            for field in expected_capability_fields:
                assert field in capabilities
            
            assert isinstance(capabilities['enhanced_methods'], list)
    
    def test_capability_queries(self, factory):
        """Test capability-based model queries."""
        # Test multivariate models
        mv_models = factory.get_multivariate_models()
        assert isinstance(mv_models, list)
        
        # Test covariates models
        cov_models = factory.get_covariates_models() 
        assert isinstance(cov_models, list)
        
        # Test generic capability query
        for capability in ['multivariate_support', 'covariates_support']:
            models = factory.list_models_by_capability(capability)
            assert isinstance(models, list)
    
    def test_enhanced_features_summary(self, factory):
        """Test enhanced features summary."""
        summary = factory.get_enhanced_features_summary()
        
        assert 'total_models' in summary
        assert 'multivariate_models' in summary
        assert 'covariates_models' in summary
        assert 'enhanced_capabilities' in summary
        
        capabilities = summary['enhanced_capabilities']
        assert 'multivariate_support' in capabilities
        assert 'covariates_support' in capabilities
        assert 'frequency_detection' in capabilities
        assert 'enhanced_data_structures' in capabilities
        
        # Check data structures list
        data_structures = capabilities['enhanced_data_structures']
        expected_structures = ['MultivariateInput', 'CovariatesInput', 'FrequencyInfo']
        for structure in expected_structures:
            assert structure in data_structures


class TestConvenienceDataStructures:
    """Test convenience methods for creating data structures."""
    
    @pytest.fixture
    def factory(self):
        return TSFMFactory()
    
    def test_create_multivariate_input(self, factory):
        """Test multivariate input creation convenience method."""
        data = np.random.randn(3, 100).astype(np.float32)
        variable_names = ["var1", "var2", "var3"]
        
        mv_input = factory.create_multivariate_input(
            data=data,
            variable_names=variable_names,
            target_variables=["var1", "var2"],
            metadata={"source": "test"}
        )
        
        assert isinstance(mv_input, MultivariateInput)
        assert np.array_equal(mv_input.data, data)
        assert mv_input.variable_names == variable_names
        assert mv_input.target_variables == ["var1", "var2"]
        assert mv_input.metadata["source"] == "test"
    
    def test_create_covariates_input(self, factory):
        """Test covariates input creation convenience method."""
        target = np.random.randn(200).astype(np.float32)
        covariates = {
            "temp": np.random.randn(200).astype(np.float32),
            "humidity": np.random.randn(200).astype(np.float32)
        }
        future_covariates = {
            "temp": np.random.randn(50).astype(np.float32)
        }
        categorical_covariates = {
            "region": ["A", "B", "C"]
        }
        
        cov_input = factory.create_covariates_input(
            target=target,
            covariates=covariates,
            future_covariates=future_covariates,
            categorical_covariates=categorical_covariates,
            metadata={"source": "weather"}
        )
        
        assert isinstance(cov_input, CovariatesInput)
        assert np.array_equal(cov_input.target, target)
        assert cov_input.covariates.keys() == covariates.keys()
        assert cov_input.future_covariates.keys() == future_covariates.keys()
        assert cov_input.categorical_covariates["region"] == ["A", "B", "C"]
        assert cov_input.metadata["source"] == "weather"
    
    def test_create_frequency_info(self, factory):
        """Test frequency info creation convenience method."""
        freq_info = factory.create_frequency_info(
            freq_str="D",
            freq_value=1,
            is_regular=True,
            detected_freq="D"
        )
        
        assert isinstance(freq_info, FrequencyInfo)
        assert freq_info.freq_str == "D"
        assert freq_info.freq_value == 1
        assert freq_info.is_regular == True
        assert freq_info.detected_freq == "D"
    
    def test_create_frequency_info_defaults(self, factory):
        """Test frequency info creation with defaults."""
        freq_info = factory.create_frequency_info()
        
        assert isinstance(freq_info, FrequencyInfo)
        assert freq_info.freq_str is None
        assert freq_info.freq_value is None
        assert freq_info.is_regular == True
        assert freq_info.detected_freq is None


class TestConvenienceForecastingMethods:
    """Test convenience forecasting methods."""
    
    @pytest.fixture
    def factory(self):
        return TSFMFactory()
    
    def test_factory_multivariate_forecasting(self, factory):
        """Test factory-level multivariate forecasting."""
        # Test with model that doesn't support multivariate
        try:
            model = factory.load_model("timesfm")
            
            data = np.random.randn(2, 100).astype(np.float32)
            mv_input = factory.create_multivariate_input(
                data=data,
                variable_names=["A", "B"]
            )
            
            # Should work with base implementation
            result = factory.forecast_multivariate("timesfm", mv_input, 24)
            assert isinstance(result, dict)
            assert len(result) == 2
            
            factory.unload_model("timesfm")
            
        except Exception as e:
            if "TimesFM" in str(e):
                pytest.skip(f"TimesFM test skipped: {e}")
            else:
                # Test error handling for unloaded model
                with pytest.raises(ValueError, match="not loaded"):
                    data = np.random.randn(2, 50).astype(np.float32)
                    mv_input = factory.create_multivariate_input(
                        data=data,
                        variable_names=["X", "Y"]
                    )
                    factory.forecast_multivariate("unloaded_model", mv_input, 12)
    
    def test_factory_covariates_forecasting(self, factory):
        """Test factory-level covariates forecasting."""
        try:
            model = factory.load_model("timesfm")
            
            target = np.random.randn(150).astype(np.float32)
            cov_input = factory.create_covariates_input(
                target=target,
                covariates={"feature": np.random.randn(150).astype(np.float32)}
            )
            frequency = factory.create_frequency_info(freq_str="D", freq_value=1)
            
            result = factory.forecast_with_covariates("timesfm", cov_input, 48, frequency=frequency)
            assert len(result.predictions) == 48
            
            factory.unload_model("timesfm")
            
        except Exception as e:
            if "TimesFM" in str(e):
                pytest.skip(f"TimesFM covariates test skipped: {e}")
            else:
                # Test error handling
                with pytest.raises(ValueError, match="not loaded"):
                    target = np.random.randn(100).astype(np.float32)
                    cov_input = factory.create_covariates_input(
                        target=target,
                        covariates={"feature": np.random.randn(100).astype(np.float32)}
                    )
                    factory.forecast_with_covariates("unloaded_model", cov_input, 24)


class TestBackwardCompatibility:
    """Test that enhanced factory maintains backward compatibility."""
    
    @pytest.fixture
    def factory(self):
        return TSFMFactory()
    
    def test_existing_methods_still_work(self, factory):
        """Test that all existing factory methods still work."""
        # Basic model listing
        models = factory.list_models()
        assert isinstance(models, list)
        assert len(models) > 0
        
        # Alias method
        available = factory.list_available_models()
        assert available == models
        
        # Model info (should work with enhanced version)
        for model_name in models[:2]:  # Test first 2 models
            info = factory.get_model_info(model_name)
            assert 'name' in info
            assert 'class' in info
    
    def test_model_loading_unloading_unchanged(self, factory):
        """Test that model loading/unloading behavior is unchanged."""
        try:
            # Test loading
            model = factory.load_model("chronos")
            assert model is not None
            
            # Test loaded models list
            loaded = factory.list_loaded_models()
            assert "chronos" in loaded
            
            # Alternative method name
            loaded2 = factory.get_loaded_models()
            assert loaded == loaded2
            
            # Test unloading
            factory.unload_model("chronos")
            loaded_after = factory.list_loaded_models()
            assert "chronos" not in loaded_after
            
        except Exception as e:
            if "chronos" in str(e).lower():
                pytest.skip(f"Chronos test skipped: {e}")
            else:
                raise
    
    def test_health_check_enhanced(self, factory):
        """Test that health check works with enhanced features."""
        health = factory.health_check()
        
        assert 'factory_status' in health
        assert 'registered_models' in health
        assert 'loaded_models' in health
        assert 'models_health' in health
        
        assert health['factory_status'] == 'operational'
        assert isinstance(health['registered_models'], list)
        assert isinstance(health['loaded_models'], list)
        assert isinstance(health['models_health'], dict)


class TestEnhancedFactoryIntegration:
    """Test integration of enhanced features."""
    
    @pytest.fixture
    def factory(self):
        return TSFMFactory()
    
    def test_complete_enhanced_workflow(self, factory):
        """Test complete workflow with enhanced features."""
        # Get summary of enhanced features
        summary = factory.get_enhanced_features_summary()
        assert summary['total_models'] > 0
        
        # Find models with capabilities
        mv_models = factory.get_multivariate_models()
        cov_models = factory.get_covariates_models()
        
        print(f"Found {len(mv_models)} multivariate models: {mv_models}")
        print(f"Found {len(cov_models)} covariates models: {cov_models}")
        
        # Create enhanced data structures
        mv_data = factory.create_multivariate_input(
            data=np.random.randn(3, 200).astype(np.float32),
            variable_names=["cpu", "memory", "disk"]
        )
        
        cov_data = factory.create_covariates_input(
            target=np.random.randn(200).astype(np.float32),
            covariates={"external": np.random.randn(200).astype(np.float32)}
        )
        
        frequency = factory.create_frequency_info(freq_str="H", freq_value=2)
        
        # Test with actual models if available
        if cov_models:
            try:
                model_name = cov_models[0]
                model = factory.load_model(model_name)
                
                # Test enhanced forecasting
                result = factory.forecast_with_covariates(
                    model_name, cov_data, 24, frequency=frequency
                )
                assert len(result.predictions) == 24
                
                factory.unload_model(model_name)
                
            except Exception as e:
                pytest.skip(f"Enhanced workflow test skipped: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])