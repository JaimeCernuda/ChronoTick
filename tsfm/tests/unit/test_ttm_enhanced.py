#!/usr/bin/env python3
"""
Real inference tests for TTM enhanced functionality.
Tests actual model loading and inference with multivariate/exogenous variables.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add the tsfm package to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tsfm.LLM.ttm import TTMModel
from tsfm.base import MultivariateInput, CovariatesInput, ModelStatus


class TestTTMRealInference:
    """Real inference tests for TTM enhanced functionality."""
    
    @pytest.fixture(scope="class")
    def loaded_model(self):
        """Create and load TTM model once for the class."""
        model = TTMModel(model_name="ttm_test", device="cpu")
        try:
            model.load_model()
            yield model
        except Exception as e:
            pytest.skip(f"TTM model loading failed: {e}")
        finally:
            if hasattr(model, 'model') and model.model is not None:
                model.unload_model()
    
    def test_real_multivariate_inference_single_target(self, loaded_model):
        """Test real multivariate inference with single target variable."""
        # Create 3-variable multivariate data (600 timesteps - TTM needs 512+ minimum)
        np.random.seed(42)
        data = np.random.randn(3, 600).astype(np.float32)
        
        multivariate_input = MultivariateInput(
            data=data,
            variable_names=["load", "temperature", "capacity"],
            target_variables=["load"],  # Single target
            metadata={"frequency": "hourly", "source": "grid_data"}
        )
        
        # Test forecast - multivariate returns dict of {variable_name -> ForecastOutput}
        results = loaded_model.forecast_multivariate(multivariate_input, horizon=24)
        
        # Verify results structure
        assert results is not None
        assert isinstance(results, dict)
        assert "load" in results
        
        # Verify individual result
        load_result = results["load"]
        assert load_result.predictions is not None
        assert len(load_result.predictions) == 24
        assert load_result.metadata['model_name'] == "ttm_test"
        assert load_result.metadata['forecast_horizon'] == 24
        assert load_result.metadata['variable_name'] == "load"
        assert load_result.metadata['ttm_multivariate'] == True
        assert load_result.metadata['channel_mixing_ready'] == True
        assert load_result.metadata['total_variables'] == 3
        assert load_result.metadata['ttm_architecture'] == 'channel_mixing'
        
    def test_real_multivariate_inference_multiple_targets(self, loaded_model):
        """Test real multivariate inference with multiple target variables."""
        # Create 4-variable multivariate data (600 timesteps for TTM)
        np.random.seed(123)
        data = np.random.randn(4, 600).astype(np.float32)
        
        multivariate_input = MultivariateInput(
            data=data,
            variable_names=["cpu", "memory", "disk", "network"],
            target_variables=["cpu", "memory"],  # Multiple targets
            metadata={"frequency": "minute", "source": "monitoring"}
        )
        
        # Test forecast - multivariate returns dict
        results = loaded_model.forecast_multivariate(multivariate_input, horizon=12)
        
        # Verify results structure
        assert results is not None
        assert isinstance(results, dict)
        assert "cpu" in results
        assert "memory" in results
        
        # Verify individual results
        cpu_result = results["cpu"]
        memory_result = results["memory"]
        
        assert cpu_result.predictions is not None
        assert len(cpu_result.predictions) == 12
        assert cpu_result.metadata['variable_name'] == "cpu"
        assert cpu_result.metadata['ttm_multivariate'] == True
        assert cpu_result.metadata['channel_mixing_ready'] == True
        
        assert memory_result.predictions is not None
        assert len(memory_result.predictions) == 12
        assert memory_result.metadata['variable_name'] == "memory"
        
    def test_real_exogenous_inference_basic(self, loaded_model):
        """Test real exogenous variables inference with basic setup."""
        # Create target and exogenous data (600 timesteps for TTM)
        np.random.seed(456)
        target = np.random.randn(600).astype(np.float32)
        
        covariates = {
            "outdoor_temp": np.random.randn(600).astype(np.float32),
            "occupancy": np.random.randn(600).astype(np.float32)
        }
        
        covariates_input = CovariatesInput(
            target=target,
            covariates=covariates,
            metadata={"frequency": "hourly", "building": "office_A"}
        )
        
        # Test forecast
        result = loaded_model.forecast_with_covariates(covariates_input, horizon=24)
        
        # Verify results
        assert result is not None
        assert result.predictions is not None
        assert len(result.predictions) == 24
        assert 'ttm_exogenous_infusion' in result.metadata
        assert result.metadata['ttm_exogenous_infusion'] == True
        assert 'exogenous_variables' in result.metadata
        assert set(result.metadata['exogenous_variables']) == {"outdoor_temp", "occupancy"}
        assert result.metadata['channel_mixing_enabled'] == True
        
    def test_real_exogenous_inference_with_future(self, loaded_model):
        """Test real exogenous inference with future covariates."""
        # Create target and covariates data (600 timesteps for TTM)
        np.random.seed(789)
        target = np.random.randn(600).astype(np.float32)
        
        covariates = {
            "price_signal": np.random.randn(600).astype(np.float32)
        }
        
        # Future covariates for forecast horizon
        future_covariates = {
            "price_signal": np.random.randn(24).astype(np.float32)  # 24 future values
        }
        
        covariates_input = CovariatesInput(
            target=target,
            covariates=covariates,
            future_covariates=future_covariates,
            metadata={"frequency": "hourly", "market": "energy"}
        )
        
        # Test forecast
        result = loaded_model.forecast_with_covariates(covariates_input, horizon=24)
        
        # Verify results
        assert result is not None
        assert result.predictions is not None
        assert len(result.predictions) == 24
        assert 'ttm_exogenous_infusion' in result.metadata
        assert result.metadata['ttm_exogenous_infusion'] == True
        assert 'exogenous_variables' in result.metadata
        assert result.metadata['exogenous_variables'] == ["price_signal"]
        
    def test_real_enhanced_functionality_combination(self, loaded_model):
        """Test real inference combining enhanced features."""
        # Test that enhanced capabilities work together
        np.random.seed(999)
        
        # Create multivariate data that could use exogenous-like interpretation (600 timesteps for TTM)
        data = np.random.randn(2, 600).astype(np.float32)
        
        multivariate_input = MultivariateInput(
            data=data,
            variable_names=["energy_demand", "weather_factor"],
            target_variables=["energy_demand"],
            metadata={
                "frequency": "hourly",
                "enhanced_test": True,
                "model_version": "ttm-r2"
            }
        )
        
        # Test with different horizon
        results = loaded_model.forecast_multivariate(multivariate_input, horizon=48)
        
        # Verify enhanced functionality
        assert results is not None
        assert isinstance(results, dict)
        assert "energy_demand" in results
        
        demand_result = results["energy_demand"]
        assert len(demand_result.predictions) == 48
        assert demand_result.metadata['multivariate_support'] == True
        assert demand_result.metadata['exogenous_support'] == True
        assert demand_result.metadata['model_repo'] == "ibm-granite/granite-timeseries-ttm-r2"
        
    def test_real_inference_error_handling(self, loaded_model):
        """Test real inference error handling with invalid inputs."""
        # Test with invalid multivariate input
        with pytest.raises(IndexError):  # Base implementation raises IndexError for wrong shape
            invalid_input = MultivariateInput(
                data=np.array([1, 2, 3]),  # Wrong shape - should be 2D
                variable_names=["var1", "var2"]  # Mismatched
            )
            loaded_model.forecast_multivariate(invalid_input, horizon=10)
            
    def test_real_inference_performance_metrics(self, loaded_model):
        """Test that real inference provides performance metadata."""
        # Create standard test data (600 timesteps for TTM)
        np.random.seed(42)
        data = np.random.randn(2, 600).astype(np.float32)
        
        multivariate_input = MultivariateInput(
            data=data,
            variable_names=["metric_a", "metric_b"],
            target_variables=["metric_a"]
        )
        
        # Test forecast
        results = loaded_model.forecast_multivariate(multivariate_input, horizon=24)
        
        # Get result for metric_a
        result = results["metric_a"]
        
        # Verify performance metadata
        assert 'input_shape' in result.metadata
        assert 'output_shape' in result.metadata
        assert 'context_length' in result.metadata
        assert result.metadata['ttm_multivariate'] == True
        assert result.metadata['multivariate_support'] == True
        
        # Verify actual inference happened (not mock)
        assert np.all(np.isfinite(result.predictions))
        assert len(result.predictions) == 24

    def test_real_ttm_specific_features(self, loaded_model):
        """Test TTM specific enhanced features."""
        # Test TTM's MLP-mixer architecture and exogenous support
        np.random.seed(42)
        
        # Create context data that meets TTM's minimum requirements (600 timesteps)
        context = np.random.randn(600).astype(np.float32)
        
        # Test different horizons with TTM
        for horizon in [12, 24, 48]:
            result = loaded_model.forecast(context, horizon=horizon)
            
            assert result is not None
            assert len(result.predictions) == horizon
            assert result.metadata['multivariate_support'] == True
            assert result.metadata['exogenous_support'] == True
            assert result.metadata['model_repo'] == "ibm-granite/granite-timeseries-ttm-r2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])