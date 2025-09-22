#!/usr/bin/env python3
"""
Real inference tests for Toto enhanced functionality.
Tests actual model loading and inference with multivariate/covariates for observability data.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add the tsfm package to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tsfm.LLM.toto import TotoModel
from tsfm.base import MultivariateInput, CovariatesInput, ModelStatus


class TestTotoRealInference:
    """Real inference tests for Toto enhanced functionality."""
    
    @pytest.fixture(scope="class")
    def loaded_model(self):
        """Create and load Toto model once for the class."""
        model = TotoModel(model_name="toto_test", device="cpu")
        try:
            model.load_model()
            yield model
        except Exception as e:
            pytest.skip(f"Toto model loading failed: {e}")
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
            variable_names=["cpu_utilization", "memory_usage", "error_rate"],
            target_variables=["cpu_utilization"],  # Single target
            metadata={"frequency": "minute", "source": "observability"}
        )
        
        # Test forecast - multivariate returns dict of {variable_name -> ForecastOutput}
        results = loaded_model.forecast_multivariate(multivariate_input, horizon=24)
        
        # Verify results structure
        assert results is not None
        assert isinstance(results, dict)
        assert "cpu_utilization" in results
        
        # Verify individual result
        cpu_result = results["cpu_utilization"]
        assert cpu_result.predictions is not None
        assert len(cpu_result.predictions) == 24
        assert cpu_result.metadata['model_name'] == "toto_test"
        assert cpu_result.metadata['forecast_horizon'] == 24
        assert cpu_result.metadata['variable_name'] == "cpu_utilization"
        assert cpu_result.metadata['toto_multivariate'] == True
        assert cpu_result.metadata['observability_attention'] == True
        assert cpu_result.metadata['total_variables'] == 3
        assert cpu_result.metadata['toto_architecture'] == 'observability_specialized'
        
    def test_real_multivariate_inference_multiple_targets(self, loaded_model):
        """Test real multivariate inference with multiple target variables."""
        # Create 4-variable multivariate data
        np.random.seed(123)
        data = np.random.randn(4, 150).astype(np.float32)
        
        multivariate_input = MultivariateInput(
            data=data,
            variable_names=["latency", "throughput", "error_count", "queue_depth"],
            target_variables=["latency", "throughput"],  # Multiple targets
            metadata={"frequency": "second", "service": "api_gateway"}
        )
        
        # Test forecast - multivariate returns dict
        results = loaded_model.forecast_multivariate(multivariate_input, horizon=12)
        
        # Verify results structure
        assert results is not None
        assert isinstance(results, dict)
        assert "latency" in results
        assert "throughput" in results
        
        # Verify individual results
        latency_result = results["latency"]
        throughput_result = results["throughput"]
        
        assert latency_result.predictions is not None
        assert len(latency_result.predictions) == 12
        assert latency_result.metadata['variable_name'] == "latency"
        assert latency_result.metadata['toto_multivariate'] == True
        assert latency_result.metadata['observability_attention'] == True
        
        assert throughput_result.predictions is not None
        assert len(throughput_result.predictions) == 12
        assert throughput_result.metadata['variable_name'] == "throughput"
        
    def test_real_covariates_inference_basic(self, loaded_model):
        """Test real covariates inference with observability context."""
        # Create target and covariates data
        np.random.seed(456)
        target = np.random.randn(100).astype(np.float32)
        
        covariates = {
            "deployment_events": np.random.randn(100).astype(np.float32),
            "traffic_pattern": np.random.randn(100).astype(np.float32)
        }
        
        covariates_input = CovariatesInput(
            target=target,
            covariates=covariates,
            metadata={"frequency": "minute", "environment": "production"}
        )
        
        # Test forecast
        result = loaded_model.forecast_with_covariates(covariates_input, horizon=24)
        
        # Verify results
        assert result is not None
        assert result.predictions is not None
        assert len(result.predictions) == 24
        assert 'toto_high_cardinality_covariates' in result.metadata
        assert result.metadata['toto_high_cardinality_covariates'] == True
        assert 'covariates_variables' in result.metadata
        assert set(result.metadata['covariates_variables']) == {"deployment_events", "traffic_pattern"}
        assert result.metadata['observability_optimized'] == True
        assert result.metadata['decoder_transformer_architecture'] == True
        
    def test_real_covariates_inference_with_future(self, loaded_model):
        """Test real covariates inference with future observability data."""
        # Create target and covariates data
        np.random.seed(789)
        target = np.random.randn(80).astype(np.float32)
        
        covariates = {
            "scheduled_events": np.random.randn(80).astype(np.float32)
        }
        
        # Future covariates for forecast horizon
        future_covariates = {
            "scheduled_events": np.random.randn(24).astype(np.float32)  # 24 future values
        }
        
        covariates_input = CovariatesInput(
            target=target,
            covariates=covariates,
            future_covariates=future_covariates,
            metadata={"frequency": "minute", "service": "monitoring"}
        )
        
        # Test forecast
        result = loaded_model.forecast_with_covariates(covariates_input, horizon=24)
        
        # Verify results
        assert result is not None
        assert result.predictions is not None
        assert len(result.predictions) == 24
        assert 'toto_high_cardinality_covariates' in result.metadata
        assert result.metadata['toto_high_cardinality_covariates'] == True
        assert 'covariates_variables' in result.metadata
        assert result.metadata['covariates_variables'] == ["scheduled_events"]
        
    def test_real_enhanced_functionality_combination(self, loaded_model):
        """Test real inference combining enhanced observability features."""
        # Test that enhanced capabilities work together
        np.random.seed(999)
        
        # Create multivariate observability data
        data = np.random.randn(2, 120).astype(np.float32)
        
        multivariate_input = MultivariateInput(
            data=data,
            variable_names=["response_time", "system_load"],
            target_variables=["response_time"],
            metadata={
                "frequency": "minute",
                "enhanced_test": True,
                "model_version": "toto-datadog",
                "observability_domain": True
            }
        )
        
        # Test with different horizon
        results = loaded_model.forecast_multivariate(multivariate_input, horizon=48)
        
        # Verify enhanced functionality
        assert results is not None
        assert isinstance(results, dict)
        assert "response_time" in results
        
        response_result = results["response_time"]
        assert len(response_result.predictions) == 48
        assert response_result.metadata['multivariate_support'] == True
        assert response_result.metadata['model_repo'] == "Datadog/Toto-Open-Base-1.0"
        assert response_result.metadata['observability_attention'] == True
        assert response_result.metadata['toto_architecture'] == 'observability_specialized'
        
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
        # Create standard test data
        np.random.seed(42)
        data = np.random.randn(2, 100).astype(np.float32)
        
        multivariate_input = MultivariateInput(
            data=data,
            variable_names=["metric_1", "metric_2"],
            target_variables=["metric_1"]
        )
        
        # Test forecast
        results = loaded_model.forecast_multivariate(multivariate_input, horizon=24)
        
        # Get result for metric_1
        result = results["metric_1"]
        
        # Verify performance metadata
        assert 'input_shape' in result.metadata
        assert 'output_shape' in result.metadata
        assert 'context_length' in result.metadata
        assert result.metadata['toto_multivariate'] == True
        assert result.metadata['observability_attention'] == True
        
        # Verify actual inference happened (not mock)
        assert np.all(np.isfinite(result.predictions))
        assert len(result.predictions) == 24

    def test_real_toto_specific_features(self, loaded_model):
        """Test Toto specific observability features."""
        # Test Toto's observability specialization
        np.random.seed(42)
        
        # Create context data typical of observability metrics
        context = np.random.randn(100).astype(np.float32)
        
        # Test different horizons with Toto
        for horizon in [12, 24, 48]:
            result = loaded_model.forecast(context, horizon=horizon)
            
            assert result is not None
            assert len(result.predictions) == horizon
            assert result.metadata['multivariate_support'] == True
            assert result.metadata['model_repo'] == "Datadog/Toto-Open-Base-1.0"
            assert result.metadata['covariates_support'] == True
            assert 'quantile_levels' in result.metadata


if __name__ == "__main__":
    pytest.main([__file__, "-v"])