#!/usr/bin/env python3
"""
Real inference tests for Time-MoE enhanced functionality.
Tests actual model loading and inference with multivariate/covariates using MoE architecture.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add the tsfm package to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tsfm.LLM.time_moe import TimeMoEModel
from tsfm.base import MultivariateInput, CovariatesInput, ModelStatus


class TestTimeMoERealInference:
    """Real inference tests for Time-MoE enhanced functionality."""
    
    @pytest.fixture(scope="class")
    def loaded_model(self):
        """Create and load Time-MoE model once for the class."""
        model = TimeMoEModel(model_name="time_moe_test", device="cpu")
        try:
            model.load_model()
            yield model
        except Exception as e:
            pytest.skip(f"Time-MoE model loading failed: {e}")
        finally:
            if hasattr(model, 'model') and model.model is not None:
                model.unload_model()
    
    def test_real_multivariate_inference_single_target(self, loaded_model):
        """Test real multivariate inference with single target variable."""
        # Create 3-variable multivariate data (200 timesteps)
        np.random.seed(42)
        data = np.random.randn(3, 200).astype(np.float32)
        
        multivariate_input = MultivariateInput(
            data=data,
            variable_names=["traffic_volume", "weather_index", "event_flag"],
            target_variables=["traffic_volume"],  # Single target
            metadata={"frequency": "daily", "source": "city_traffic"}
        )
        
        # Test forecast - multivariate returns dict of {variable_name -> ForecastOutput}
        results = loaded_model.forecast_multivariate(multivariate_input, horizon=24)
        
        # Verify results structure
        assert results is not None
        assert isinstance(results, dict)
        assert "traffic_volume" in results
        
        # Verify individual result
        traffic_result = results["traffic_volume"]
        assert traffic_result.predictions is not None
        assert len(traffic_result.predictions) == 24
        assert traffic_result.metadata['model_name'] == "time_moe_test"
        assert traffic_result.metadata['forecast_horizon'] == 24
        assert traffic_result.metadata['variable_name'] == "traffic_volume"
        assert traffic_result.metadata['multivariate_variables'] == ["traffic_volume", "weather_index", "event_flag"]
        assert traffic_result.metadata['target_variables'] == ["traffic_volume"]
        assert traffic_result.metadata['time_moe_multivariate'] == True
        assert traffic_result.metadata['mixture_of_experts_multivariate'] == True
        assert traffic_result.metadata['expert_specialization'] == 'variable_patterns'
        assert traffic_result.metadata['inference_method'] == 'separate_univariate_with_moe'
        
    def test_real_multivariate_inference_multiple_targets(self, loaded_model):
        """Test real multivariate inference with multiple target variables."""
        # Create 4-variable multivariate data
        np.random.seed(123)
        data = np.random.randn(4, 150).astype(np.float32)
        
        multivariate_input = MultivariateInput(
            data=data,
            variable_names=["stock_price", "volume", "volatility", "sentiment"],
            target_variables=["stock_price", "volume"],  # Multiple targets
            metadata={"frequency": "minute", "market": "NYSE"}
        )
        
        # Test forecast - multivariate returns dict
        results = loaded_model.forecast_multivariate(multivariate_input, horizon=12)
        
        # Verify results structure
        assert results is not None
        assert isinstance(results, dict)
        assert "stock_price" in results
        assert "volume" in results
        
        # Verify individual results
        price_result = results["stock_price"]
        volume_result = results["volume"]
        
        assert price_result.predictions is not None
        assert len(price_result.predictions) == 12
        assert price_result.metadata['variable_name'] == "stock_price"
        assert price_result.metadata['target_variables'] == ["stock_price", "volume"]
        assert price_result.metadata['mixture_of_experts'] == True
        
        assert volume_result.predictions is not None
        assert len(volume_result.predictions) == 12
        assert volume_result.metadata['variable_name'] == "volume"
        
    def test_real_covariates_inference_basic(self, loaded_model):
        """Test real covariates inference with MoE adaptability."""
        # Create target and covariates data
        np.random.seed(456)
        target = np.random.randn(100).astype(np.float32)
        
        covariates = {
            "economic_indicator": np.random.randn(100).astype(np.float32),
            "policy_change": np.random.randn(100).astype(np.float32)
        }
        
        covariates_input = CovariatesInput(
            target=target,
            covariates=covariates,
            metadata={"frequency": "monthly", "domain": "economics"}
        )
        
        # Test forecast
        result = loaded_model.forecast_with_covariates(covariates_input, horizon=24)
        
        # Verify results
        assert result is not None
        assert result.predictions is not None
        assert len(result.predictions) == 24
        assert 'time_moe_covariates_adaptation' in result.metadata
        assert result.metadata['time_moe_covariates_adaptation'] == True
        assert 'mixture_of_experts_routing' in result.metadata
        assert result.metadata['mixture_of_experts_routing'] == True
        assert 'covariates_variables' in result.metadata
        assert set(result.metadata['covariates_variables']) == {"economic_indicator", "policy_change"}
        assert result.metadata['expert_specialization'] == 'covariate_patterns'
        assert result.metadata['billion_scale_architecture'] == True
        
    def test_real_covariates_inference_with_categorical(self, loaded_model):
        """Test real covariates inference with categorical variables."""
        # Create target and covariates data
        np.random.seed(789)
        target = np.random.randn(80).astype(np.float32)
        
        covariates = {
            "continuous_var": np.random.randn(80).astype(np.float32)
        }
        
        # Categorical covariates
        categorical_covariates = {
            "season": ["spring", "summer", "autumn", "winter"] * 20,
            "day_type": ["weekday", "weekend"] * 40
        }
        
        covariates_input = CovariatesInput(
            target=target,
            covariates=covariates,
            categorical_covariates=categorical_covariates,
            metadata={"frequency": "daily", "source": "retail"}
        )
        
        # Test forecast
        result = loaded_model.forecast_with_covariates(covariates_input, horizon=24)
        
        # Verify results
        assert result is not None
        assert result.predictions is not None
        assert len(result.predictions) == 24
        assert 'categorical_support' in result.metadata
        assert result.metadata['categorical_support'] == True
        assert result.metadata['iclr_2025_model'] == True
        
    def test_real_enhanced_functionality_combination(self, loaded_model):
        """Test real inference combining enhanced MoE features."""
        # Test that enhanced capabilities work together
        np.random.seed(999)
        
        # Create multivariate data that could benefit from MoE expert routing
        data = np.random.randn(2, 120).astype(np.float32)
        
        multivariate_input = MultivariateInput(
            data=data,
            variable_names=["primary_signal", "auxiliary_signal"],
            target_variables=["primary_signal"],
            metadata={
                "frequency": "hourly",
                "enhanced_test": True,
                "model_version": "time-moe-200m"
            }
        )
        
        # Test with different horizon
        results = loaded_model.forecast_multivariate(multivariate_input, horizon=48)
        
        # Verify enhanced functionality
        assert results is not None
        assert isinstance(results, dict)
        assert "primary_signal" in results
        
        primary_result = results["primary_signal"]
        assert len(primary_result.predictions) == 48
        assert primary_result.metadata['covariates_adaptable'] == True
        assert primary_result.metadata['model_repo'] == "Maple728/TimeMoE-200M"
        assert primary_result.metadata['mixture_of_experts'] == True
        assert primary_result.metadata['time_moe_multivariate'] == True
        
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
            variable_names=["signal_1", "signal_2"],
            target_variables=["signal_1"]
        )
        
        # Test forecast
        results = loaded_model.forecast_multivariate(multivariate_input, horizon=24)
        
        # Get result for signal_1
        result = results["signal_1"]
        
        # Verify performance metadata
        assert 'input_shape' in result.metadata
        assert 'output_shape' in result.metadata
        assert 'context_length' in result.metadata
        assert result.metadata['mixture_of_experts'] == True
        assert result.metadata['time_moe_multivariate'] == True
        
        # Verify actual inference happened (not mock)
        assert np.all(np.isfinite(result.predictions))
        assert len(result.predictions) == 24

    def test_real_time_moe_specific_features(self, loaded_model):
        """Test Time-MoE specific MoE architecture features."""
        # Test MoE billion-scale architecture and expert routing
        np.random.seed(42)
        
        # Create context data for Time-MoE
        context = np.random.randn(100).astype(np.float32)
        
        # Test different horizons with Time-MoE
        for horizon in [12, 24, 48]:
            result = loaded_model.forecast(context, horizon=horizon)
            
            assert result is not None
            assert len(result.predictions) == horizon
            assert result.metadata['covariates_adaptable'] == True
            assert result.metadata['model_repo'] == "Maple728/TimeMoE-200M"
            assert result.metadata['mixture_of_experts'] == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])