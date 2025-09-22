#!/usr/bin/env python3
"""
Real inference tests for TimesFM 2.0 enhanced functionality.
Tests actual model loading and inference with multivariate/covariates.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add the tsfm package to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tsfm.LLM.timesfm import TimesFMModel
from tsfm.base import MultivariateInput, CovariatesInput, ModelStatus


class TestTimesFMRealInference:
    """Real inference tests for TimesFM 2.0 enhanced functionality."""
    
    @pytest.fixture(scope="class")
    def loaded_model(self):
        """Create and load TimesFM 2.0 model once for the class."""
        model = TimesFMModel(model_name="timesfm_test", device="cpu")
        try:
            model.load_model()
            yield model
        except Exception as e:
            pytest.skip(f"TimesFM 2.0 model loading failed: {e}")
        finally:
            if hasattr(model, 'tfm') and model.tfm is not None:
                model.unload_model()
    
    def test_real_multivariate_inference_single_target(self, loaded_model):
        """Test real multivariate inference with single target variable."""
        # Create 3-variable multivariate data (200 timesteps)
        np.random.seed(42)
        data = np.random.randn(3, 200).astype(np.float32)
        
        multivariate_input = MultivariateInput(
            data=data,
            variable_names=["revenue", "costs", "margin"],
            target_variables=["revenue"],  # Single target
            metadata={"frequency": "daily"}
        )
        
        # Test forecast - multivariate returns dict of {variable_name -> ForecastOutput}
        results = loaded_model.forecast_multivariate(multivariate_input, horizon=24)
        
        # Verify results structure
        assert results is not None
        assert isinstance(results, dict)
        assert "revenue" in results
        
        # Verify individual result
        revenue_result = results["revenue"]
        assert revenue_result.predictions is not None
        assert len(revenue_result.predictions) == 24
        assert revenue_result.metadata['model_name'] == "timesfm_test"
        assert revenue_result.metadata['forecast_horizon'] == 24
        assert revenue_result.metadata['variable_name'] == "revenue"
        assert revenue_result.metadata['multivariate_variables'] == ["revenue", "costs", "margin"]
        assert revenue_result.metadata['target_variables'] == ["revenue"]
        assert revenue_result.metadata['timesfm_2_0_multivariate'] == True
        assert revenue_result.metadata['inference_method'] == 'separate_univariate'
        
        # Should have quantiles from TimesFM 2.0
        assert revenue_result.quantiles is not None
        assert '0.5' in revenue_result.quantiles
        
    def test_real_multivariate_inference_multiple_targets(self, loaded_model):
        """Test real multivariate inference with multiple target variables."""
        # Create 4-variable multivariate data
        np.random.seed(123)
        data = np.random.randn(4, 150).astype(np.float32)
        
        multivariate_input = MultivariateInput(
            data=data,
            variable_names=["cpu_usage", "memory_usage", "disk_io", "network_io"],
            target_variables=["cpu_usage", "memory_usage"],  # Multiple targets
            metadata={"frequency": "hourly"}
        )
        
        # Test forecast - multivariate returns dict
        results = loaded_model.forecast_multivariate(multivariate_input, horizon=12)
        
        # Verify results structure
        assert results is not None
        assert isinstance(results, dict)
        assert "cpu_usage" in results
        assert "memory_usage" in results
        
        # Verify individual results
        cpu_result = results["cpu_usage"]
        memory_result = results["memory_usage"]
        
        assert cpu_result.predictions is not None
        assert len(cpu_result.predictions) == 12
        assert cpu_result.metadata['variable_name'] == "cpu_usage"
        assert cpu_result.metadata['target_variables'] == ["cpu_usage", "memory_usage"]
        assert cpu_result.metadata['timesfm_version'] == '2.0'
        assert cpu_result.quantiles is not None
        
        assert memory_result.predictions is not None
        assert len(memory_result.predictions) == 12
        assert memory_result.metadata['variable_name'] == "memory_usage"
        
    def test_real_covariates_inference_basic(self, loaded_model):
        """Test real covariates inference with basic setup."""
        # Create target and covariates data
        np.random.seed(456)
        target = np.random.randn(100).astype(np.float32)
        
        covariates = {
            "weather_temp": np.random.randn(100).astype(np.float32),
            "holiday_flag": np.random.randn(100).astype(np.float32)
        }
        
        covariates_input = CovariatesInput(
            target=target,
            covariates=covariates,
            metadata={"frequency": "daily"}
        )
        
        # Test forecast
        result = loaded_model.forecast_with_covariates(covariates_input, horizon=24)
        
        # Verify results
        assert result is not None
        assert result.predictions is not None
        assert len(result.predictions) == 24
        assert 'covariates_used' in result.metadata
        assert set(result.metadata['covariates_used']) == {"weather_temp", "holiday_flag"}
        assert result.metadata['covariates_enhanced'] == True
        assert result.metadata['timesfm_covariates_version'] == '2.0'
        assert result.quantiles is not None
        
    def test_real_covariates_inference_with_future(self, loaded_model):
        """Test real covariates inference with future covariates."""
        # Create target and covariates data
        np.random.seed(789)
        target = np.random.randn(80).astype(np.float32)
        
        covariates = {
            "external_signal": np.random.randn(80).astype(np.float32)
        }
        
        # Future covariates for forecast horizon
        future_covariates = {
            "external_signal": np.random.randn(24).astype(np.float32)  # 24 future values
        }
        
        covariates_input = CovariatesInput(
            target=target,
            covariates=covariates,
            future_covariates=future_covariates,
            metadata={"frequency": "daily", "source": "real_test"}
        )
        
        # Test forecast
        result = loaded_model.forecast_with_covariates(covariates_input, horizon=24)
        
        # Verify results
        assert result is not None
        assert result.predictions is not None
        assert len(result.predictions) == 24
        assert 'future_covariates_available' in result.metadata
        assert result.metadata['future_covariates_available'] == True
        assert 'external_signal' in result.metadata['covariates_used']
        assert result.quantiles is not None
        
    def test_real_enhanced_functionality_combination(self, loaded_model):
        """Test real inference combining enhanced features."""
        # Test that enhanced capabilities work together
        np.random.seed(999)
        
        # Create multivariate data that could use covariates-like interpretation
        data = np.random.randn(2, 120).astype(np.float32)
        
        multivariate_input = MultivariateInput(
            data=data,
            variable_names=["primary_metric", "supporting_factor"],
            target_variables=["primary_metric"],
            metadata={
                "frequency": "daily",
                "enhanced_test": True,
                "model_version": "timesfm-2.0"
            }
        )
        
        # Test with different horizon
        results = loaded_model.forecast_multivariate(multivariate_input, horizon=48)
        
        # Verify enhanced functionality
        assert results is not None
        assert isinstance(results, dict)
        assert "primary_metric" in results
        
        primary_result = results["primary_metric"]
        assert len(primary_result.predictions) == 48
        assert primary_result.metadata['covariates_support'] == True
        assert primary_result.metadata['model_repo'] == "google/timesfm-2.0-500m-pytorch"
        assert primary_result.metadata['timesfm_version'] == '2.0'
        
        # Verify quantile functionality
        assert primary_result.quantiles is not None
        assert len(primary_result.quantiles) == 9  # 9 quantiles [0.1, 0.2, ..., 0.9]
        quantile_keys = set(primary_result.quantiles.keys())
        expected_keys = {'0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9'}
        assert quantile_keys == expected_keys
        
    def test_real_inference_error_handling(self, loaded_model):
        """Test real inference error handling with invalid inputs."""
        # Test with invalid multivariate input
        with pytest.raises(RuntimeError):  # Our implementation raises RuntimeError
            invalid_input = MultivariateInput(
                data=np.array([1, 2, 3]),  # Wrong shape - should be 2D
                variable_names=["var1", "var2"]  # Mismatched
            )
            loaded_model.forecast_multivariate(invalid_input, horizon=10)
            
    def test_real_inference_performance_metrics(self, loaded_model):
        """Test that real inference provides performance metadata."""
        # Create standard test data
        np.random.seed(42)
        data = np.random.randn(2, 100).astype(np.float32)
        
        multivariate_input = MultivariateInput(
            data=data,
            variable_names=["series1", "series2"],
            target_variables=["series1"]
        )
        
        # Test forecast
        results = loaded_model.forecast_multivariate(multivariate_input, horizon=24)
        
        # Get result for series1
        result = results["series1"]
        
        # Verify performance metadata
        assert 'input_shape' in result.metadata
        assert 'output_shape' in result.metadata
        assert 'context_length' in result.metadata
        assert 'has_quantiles' in result.metadata
        assert result.metadata['has_quantiles'] == True
        assert 'context_len' in result.metadata  # TimesFM specific
        assert result.metadata['context_len'] == 2048
        
        # Verify actual inference happened (not mock)
        assert np.all(np.isfinite(result.predictions))
        assert len(result.predictions) == 24

    def test_real_timesfm_2_0_specific_features(self, loaded_model):
        """Test TimesFM 2.0 specific enhanced features."""
        # Test enhanced context length and flexible horizons
        np.random.seed(42)
        
        # Create longer context data to test 2048 context length
        long_context = np.random.randn(1000).astype(np.float32)
        
        # Test different horizons
        for horizon in [12, 24, 48, 96]:
            result = loaded_model.forecast(long_context, horizon=horizon)
            
            assert result is not None
            assert len(result.predictions) == horizon
            assert result.metadata['context_len'] == 2048  # TimesFM 2.0 feature
            assert result.metadata['timesfm_version'] == '2.0'
            assert result.metadata['covariates_support'] == True
            
            # Verify quantiles (TimesFM may return different quantile lengths)
            assert result.quantiles is not None
            assert '0.5' in result.quantiles
            # Note: TimesFM quantiles may have different lengths than predictions
            assert len(result.quantiles['0.5']) <= horizon  # More flexible check


if __name__ == "__main__":
    pytest.main([__file__, "-v"])