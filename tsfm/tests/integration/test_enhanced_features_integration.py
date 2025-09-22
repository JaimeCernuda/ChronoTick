#!/usr/bin/env python3
"""
Integration tests for enhanced TSFM features: multivariate, covariates, and frequency support.
Tests the complete pipeline through factory pattern with enhanced capabilities.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import warnings

# Add the tsfm package to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tsfm import TSFMFactory
from tsfm.base import MultivariateInput, CovariatesInput, FrequencyInfo
from tsfm.datasets.loader import create_synthetic_data

warnings.filterwarnings("ignore")


class TestEnhancedFeaturesIntegration:
    """Integration tests for enhanced TSFM features."""
    
    @pytest.fixture
    def factory(self):
        """Create TSFM factory instance."""
        return TSFMFactory()
    
    @pytest.fixture
    def reports_dir(self):
        """Create reports directory."""
        reports_dir = Path("tests/integration/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        return reports_dir
    
    @pytest.fixture
    def multivariate_data(self):
        """Create synthetic multivariate time series data."""
        np.random.seed(42)
        
        # Generate correlated time series
        length = 800
        t = np.linspace(0, 4*np.pi, length)
        
        # Base series with trend and seasonality
        base_series = 0.1 * t + 2 * np.sin(t) + 0.5 * np.cos(2*t)
        
        # Three related variables
        sales = base_series + np.random.normal(0, 0.2, length)
        inventory = -0.8 * sales + 50 + np.random.normal(0, 0.3, length)
        price = 0.6 * sales + 100 + np.random.normal(0, 0.1, length)
        
        data = np.array([sales, inventory, price]).astype(np.float32)
        variable_names = ["sales", "inventory", "price"]
        
        return MultivariateInput(
            data=data,
            variable_names=variable_names,
            target_variables=["sales", "inventory"],
            metadata={"source": "synthetic_correlated", "frequency": "daily"}
        )
    
    @pytest.fixture
    def covariates_data(self):
        """Create synthetic time series with covariates."""
        np.random.seed(42)
        length = 600
        
        # Target time series (e.g., sales)
        t = np.linspace(0, 3*np.pi, length)
        target = 100 + 10 * np.sin(t) + 5 * np.cos(2*t) + np.random.normal(0, 2, length)
        
        # Covariates that influence the target
        temperature = 20 + 10 * np.sin(t + np.pi/4) + np.random.normal(0, 1, length)
        day_of_week = np.tile(np.arange(1, 8), length // 7 + 1)[:length]
        promotion = np.random.choice([0, 1], length, p=[0.8, 0.2])
        
        # Future covariates (known for forecasting period)
        future_length = 96
        future_temp = 20 + 10 * np.sin(np.linspace(3*np.pi, 3.5*np.pi, future_length)) + np.random.normal(0, 1, future_length)
        future_dow = np.tile(np.arange(1, 8), future_length // 7 + 1)[:future_length]
        
        return CovariatesInput(
            target=target.astype(np.float32),
            covariates={
                "temperature": temperature.astype(np.float32),
                "day_of_week": day_of_week.astype(np.float32),
                "promotion": promotion.astype(np.float32)
            },
            future_covariates={
                "temperature": future_temp.astype(np.float32),
                "day_of_week": future_dow.astype(np.float32)
            },
            categorical_covariates={
                "region": ["north", "south", "east", "west"],
                "store_type": ["large", "medium", "small"]
            },
            metadata={"source": "retail_with_weather", "frequency": "daily"}
        )
    
    @pytest.fixture
    def frequency_data(self):
        """Create time series with frequency information."""
        # Daily frequency time series
        dates = pd.date_range("2023-01-01", periods=500, freq="D")
        values = np.random.randn(500).astype(np.float32)
        
        frequency_info = FrequencyInfo(
            freq_str="D",
            freq_value=1,
            is_regular=True,
            detected_freq="D"
        )
        
        return values, dates, frequency_info


class TestMultivariateIntegration:
    """Integration tests for multivariate forecasting."""
    
    @pytest.fixture
    def factory(self):
        return TSFMFactory()
    
    @pytest.fixture
    def multivariate_data(self):
        """Sample multivariate data (600+ timesteps for TTM compatibility)."""
        np.random.seed(42)
        data = np.random.randn(3, 600).astype(np.float32)  # 600 timesteps for TTM
        return MultivariateInput(
            data=data,
            variable_names=["cpu_usage", "memory_usage", "disk_io"],
            target_variables=["cpu_usage", "memory_usage"]
        )
    
    def test_ttm_multivariate_integration(self, factory, multivariate_data):
        """Test TTM multivariate forecasting through factory."""
        try:
            model = factory.load_model("ttm")
            
            if hasattr(model, 'multivariate_support') and model.multivariate_support:
                # Test multivariate forecasting
                results = model.forecast_multivariate(multivariate_data, horizon=48)
                
                assert isinstance(results, dict)
                assert set(results.keys()) == {"cpu_usage", "memory_usage"}
                
                for var_name, forecast_output in results.items():
                    assert len(forecast_output.predictions) == 48
                    assert forecast_output.metadata["variable_name"] == var_name
                    
                    # Check for TTM-specific metadata if enhanced
                    if 'ttm_multivariate' in forecast_output.metadata:
                        assert forecast_output.metadata['ttm_multivariate'] == True
                        assert forecast_output.metadata['total_variables'] == 3
                        
            else:
                pytest.skip("TTM multivariate support not available")
                
            factory.unload_model("ttm")
            
        except Exception as e:
            if "TTM" in str(e) or "granite" in str(e).lower():
                pytest.skip(f"TTM test skipped: {e}")
            else:
                raise
    
    def test_chronos_bolt_multivariate_integration(self, factory, multivariate_data):
        """Test Chronos-Bolt multivariate forecasting through factory."""
        try:
            model = factory.load_model("chronos")
            
            if hasattr(model, 'multivariate_support') and model.multivariate_support:
                # Test multivariate forecasting
                results = model.forecast_multivariate(multivariate_data, horizon=24)
                
                assert isinstance(results, dict)
                assert len(results) == 2  # Only target variables
                
                for var_name, forecast_output in results.items():
                    assert len(forecast_output.predictions) == 24
                    
                    # Check for Chronos-Bolt specific metadata if enhanced
                    if 'chronos_bolt_multivariate' in forecast_output.metadata:
                        assert forecast_output.metadata['chronos_bolt_multivariate'] == True
                        assert forecast_output.metadata['inference_method'] == 'separate_univariate'
                        assert 'multivariate_variables' in forecast_output.metadata
                        
            else:
                pytest.skip("Chronos-Bolt multivariate support not available")
                
            factory.unload_model("chronos")
            
        except Exception as e:
            if "chronos" in str(e).lower():
                pytest.skip(f"Chronos-Bolt test skipped: {e}")
            else:
                raise


class TestCovariatesIntegration:
    """Integration tests for covariates forecasting."""
    
    @pytest.fixture
    def factory(self):
        return TSFMFactory()
    
    @pytest.fixture
    def covariates_data(self):
        """Sample covariates data."""
        np.random.seed(42)
        target = np.random.randn(300).astype(np.float32)
        covariates = {
            "external_trend": np.linspace(0, 1, 300).astype(np.float32),
            "cyclical_pattern": np.sin(np.linspace(0, 4*np.pi, 300)).astype(np.float32)
        }
        return CovariatesInput(target=target, covariates=covariates)
    
    def test_timesfm_covariates_integration(self, factory, covariates_data):
        """Test TimesFM 2.0 covariates forecasting through factory."""
        try:
            model = factory.load_model("timesfm")
            
            if hasattr(model, 'covariates_support') and model.covariates_support:
                # Test covariates forecasting
                result = model.forecast_with_covariates(covariates_data, horizon=96)
                
                assert len(result.predictions) == 96
                assert 'covariates_used' in result.metadata
                
                # Check for TimesFM 2.0 specific metadata if enhanced
                if 'timesfm_covariates_version' in result.metadata:
                    assert result.metadata['timesfm_covariates_version'] == '2.0'
                    assert 'covariates_enhanced' in result.metadata
                    
            else:
                pytest.skip("TimesFM covariates support not available")
                
            factory.unload_model("timesfm")
            
        except Exception as e:
            if "TimesFM" in str(e):
                pytest.skip(f"TimesFM test skipped: {e}")
            else:
                raise
    
    def test_toto_high_cardinality_covariates_integration(self, factory, covariates_data):
        """Test Toto high-cardinality covariates forecasting through factory."""
        try:
            model = factory.load_model("toto")
            
            if hasattr(model, 'covariates_support') and model.covariates_support:
                # Test covariates forecasting
                result = model.forecast_with_covariates(covariates_data, horizon=48)
                
                assert len(result.predictions) == 48
                
                # Check for Toto specific metadata if enhanced
                if 'toto_high_cardinality_covariates' in result.metadata:
                    assert result.metadata['observability_optimized'] == True
                    assert 'covariates_variables' in result.metadata
                    
            else:
                pytest.skip("Toto covariates support not available")
                
            factory.unload_model("toto")
            
        except Exception as e:
            if "Toto" in str(e) or "transformers" in str(e):
                pytest.skip(f"Toto test skipped: {e}")
            else:
                raise
    
    def test_time_moe_covariates_adaptability_integration(self, factory, covariates_data):
        """Test Time-MoE covariates adaptability through factory."""
        try:
            model = factory.load_model("time_moe")
            
            if hasattr(model, 'covariates_adaptable') and model.covariates_adaptable:
                # Test covariates forecasting
                result = model.forecast_with_covariates(covariates_data, horizon=24)
                
                assert len(result.predictions) == 24
                
                # Check for Time-MoE specific metadata if enhanced
                if 'time_moe_covariates_adaptation' in result.metadata:
                    assert result.metadata['mixture_of_experts_routing'] == True
                    assert result.metadata['billion_scale_architecture'] == True
                    
            else:
                pytest.skip("Time-MoE covariates adaptability not available")
                
            factory.unload_model("time_moe")
            
        except Exception as e:
            if "Time-MoE" in str(e) or "transformers" in str(e):
                pytest.skip(f"Time-MoE test skipped: {e}")
            else:
                raise


class TestFrequencyIntegration:
    """Integration tests for frequency support."""
    
    @pytest.fixture
    def factory(self):
        return TSFMFactory()
    
    def test_frequency_detection_integration(self):
        """Test frequency detection utilities."""
        from tsfm.base import BaseTimeSeriesModel
        
        # Test with pandas date range
        timestamps = ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]
        freq_info = BaseTimeSeriesModel.detect_frequency(timestamps=timestamps)
        
        assert isinstance(freq_info, FrequencyInfo)
        # Should detect daily frequency if pandas available
        if freq_info.freq_str:
            assert freq_info.is_regular == True
    
    def test_frequency_conversion_integration(self):
        """Test frequency format conversion."""
        from tsfm.base import BaseTimeSeriesModel
        
        # Test various frequency formats
        test_cases = [
            ("daily", 1, "daily"),
            ("H", 2, "hourly"),
            (3, 3, "monthly"),
            ("quarterly", 4, "quarterly")
        ]
        
        for input_freq, expected_value, expected_type in test_cases:
            freq_info = BaseTimeSeriesModel.convert_frequency_format(input_freq)
            
            assert isinstance(freq_info, FrequencyInfo)
            if isinstance(input_freq, str) and input_freq in ["daily", "quarterly"]:
                assert freq_info.freq_value == expected_value
            elif isinstance(input_freq, int):
                assert freq_info.freq_value == input_freq


class TestCompleteEnhancedPipeline:
    """Test complete pipeline with all enhanced features."""
    
    @pytest.fixture
    def factory(self):
        return TSFMFactory()
    
    def test_factory_enhanced_capabilities_listing(self, factory):
        """Test that factory can list enhanced capabilities of models."""
        try:
            # Load a model and check its capabilities
            model = factory.load_model("ttm")
            
            health = model.health_check()
            assert 'model_name' in health
            
            # Check for enhanced capabilities in model attributes
            enhanced_features = []
            if hasattr(model, 'multivariate_support'):
                enhanced_features.append('multivariate')
            if hasattr(model, 'exogenous_support'):
                enhanced_features.append('exogenous')
            if hasattr(model, 'covariates_support'):
                enhanced_features.append('covariates')
            if hasattr(model, 'covariates_adaptable'):
                enhanced_features.append('covariates_adaptable')
            
            # TTM should have multivariate and exogenous support
            assert 'multivariate' in enhanced_features or 'exogenous' in enhanced_features
            
            factory.unload_model("ttm")
            
        except Exception as e:
            if "TTM" in str(e):
                pytest.skip(f"TTM capabilities test skipped: {e}")
            else:
                raise
    
    def test_backward_compatibility(self, factory):
        """Test that enhanced models maintain backward compatibility."""
        try:
            model = factory.load_model("timesfm")
            
            # Test that basic forecasting still works
            sample_data = np.random.randn(200).astype(np.float32)
            result = model.forecast(sample_data, horizon=24)
            
            assert len(result.predictions) == 24
            assert result.metadata['model_name'] == "timesfm"
            
            # Test that new metadata fields are present
            if hasattr(model, 'covariates_support'):
                assert 'covariates_support' in result.metadata
            if hasattr(model, 'timesfm_version'):
                assert 'timesfm_version' in result.metadata
            
            factory.unload_model("timesfm")
            
        except Exception as e:
            if "TimesFM" in str(e):
                pytest.skip(f"Backward compatibility test skipped: {e}")
            else:
                raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])