#!/usr/bin/env python3
"""
Real inference tests for Chronos-Bolt enhanced functionality.
Tests actual model loading and inference with multivariate/covariates.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add the tsfm package to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tsfm.LLM.chronos_bolt import ChronosBoltModel
from tsfm.base import MultivariateInput, CovariatesInput, ModelStatus


class TestChronosBoltRealInference:
    """Real inference tests for Chronos-Bolt enhanced functionality."""
    
    @pytest.fixture(scope="class")
    def loaded_model(self):
        """Create and load Chronos-Bolt model once for the class."""
        model = ChronosBoltModel(model_name="chronos_test", device="cpu")
        try:
            model.load_model()
            yield model
        except Exception as e:
            pytest.skip(f"Chronos-Bolt model loading failed: {e}")
        finally:
            if hasattr(model, 'pipeline') and model.pipeline is not None:
                model.unload_model()
    
    def test_real_multivariate_inference_single_target(self, loaded_model):
        """Test real multivariate inference with single target variable."""
        # Create 3-variable multivariate data (200 timesteps)
        np.random.seed(42)
        data = np.random.randn(3, 200).astype(np.float32)
        
        multivariate_input = MultivariateInput(
            data=data,
            variable_names=["sales", "inventory", "price"],
            target_variables=["sales"],  # Single target
            metadata={"frequency": "daily"}
        )
        
        # Test forecast - multivariate returns dict of {variable_name -> ForecastOutput}
        results = loaded_model.forecast_multivariate(multivariate_input, horizon=24)
        
        # Verify results structure
        assert results is not None
        assert isinstance(results, dict)
        assert "sales" in results
        
        # Verify individual result
        sales_result = results["sales"]
        assert sales_result.predictions is not None
        assert len(sales_result.predictions) == 24
        assert sales_result.metadata['model_name'] == "chronos_test"
        assert sales_result.metadata['forecast_horizon'] == 24
        assert sales_result.metadata['variable_name'] == "sales"
        assert sales_result.metadata['multivariate_variables'] == ["sales", "inventory", "price"]
        assert sales_result.metadata['target_variables'] == ["sales"]
        
        # Should have quantiles from Chronos-Bolt
        assert sales_result.quantiles is not None
        assert '0.5' in sales_result.quantiles
        
    def test_real_multivariate_inference_multiple_targets(self, loaded_model):
        """Test real multivariate inference with multiple target variables."""
        # Create 4-variable multivariate data
        np.random.seed(123)
        data = np.random.randn(4, 150).astype(np.float32)
        
        multivariate_input = MultivariateInput(
            data=data,
            variable_names=["demand", "supply", "price", "competition"],
            target_variables=["demand", "supply"],  # Multiple targets
            metadata={"frequency": "hourly"}
        )
        
        # Test forecast - multivariate returns dict
        results = loaded_model.forecast_multivariate(multivariate_input, horizon=12)
        
        # Verify results structure
        assert results is not None
        assert isinstance(results, dict)
        assert "demand" in results
        assert "supply" in results
        
        # Verify individual results
        demand_result = results["demand"]
        supply_result = results["supply"]
        
        assert demand_result.predictions is not None
        assert len(demand_result.predictions) == 12
        assert demand_result.metadata['variable_name'] == "demand"
        assert demand_result.metadata['target_variables'] == ["demand", "supply"]
        assert demand_result.quantiles is not None
        
        assert supply_result.predictions is not None
        assert len(supply_result.predictions) == 12
        assert supply_result.metadata['variable_name'] == "supply"
        
    def test_real_covariates_inference_basic(self, loaded_model):
        """Test real covariates inference with basic setup."""
        # Create target and covariates data
        np.random.seed(456)
        target = np.random.randn(100).astype(np.float32)
        
        covariates = {
            "temperature": np.random.randn(100).astype(np.float32),
            "humidity": np.random.randn(100).astype(np.float32)
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
        assert 'covariates_available' in result.metadata
        assert set(result.metadata['covariates_available']) == {"temperature", "humidity"}
        assert result.quantiles is not None
        
    def test_real_covariates_inference_with_future(self, loaded_model):
        """Test real covariates inference with future covariates."""
        # Create target and covariates data
        np.random.seed(789)
        target = np.random.randn(80).astype(np.float32)
        
        covariates = {
            "weather": np.random.randn(80).astype(np.float32)
        }
        
        # Future covariates for forecast horizon
        future_covariates = {
            "weather": np.random.randn(24).astype(np.float32)  # 24 future values
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
        assert result.metadata['future_covariates_available'] == ["weather"]
        assert result.quantiles is not None
        
    def test_real_enhanced_functionality_combination(self, loaded_model):
        """Test real inference combining enhanced features."""
        # Test that enhanced capabilities work together
        np.random.seed(999)
        
        # Create multivariate data that could use covariates-like interpretation
        data = np.random.randn(2, 120).astype(np.float32)
        
        multivariate_input = MultivariateInput(
            data=data,
            variable_names=["main_metric", "external_factor"],
            target_variables=["main_metric"],
            metadata={
                "frequency": "daily",
                "enhanced_test": True,
                "model_version": "chronos-bolt"
            }
        )
        
        # Test with different horizon
        results = loaded_model.forecast_multivariate(multivariate_input, horizon=48)
        
        # Verify enhanced functionality
        assert results is not None
        assert isinstance(results, dict)
        assert "main_metric" in results
        
        main_result = results["main_metric"]
        assert len(main_result.predictions) == 48
        assert main_result.metadata['multivariate_support'] == True
        assert main_result.metadata['model_repo'] == "amazon/chronos-bolt-base"
        assert main_result.metadata['prediction_method'] == 'quantile_median'
        
        # Verify quantile functionality
        assert main_result.quantiles is not None
        assert len(main_result.quantiles) == 9  # 9 quantiles [0.1, 0.2, ..., 0.9]
        quantile_keys = set(main_result.quantiles.keys())
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
            variable_names=["metric1", "metric2"],
            target_variables=["metric1"]
        )
        
        # Test forecast
        results = loaded_model.forecast_multivariate(multivariate_input, horizon=24)
        
        # Get result for metric1
        result = results["metric1"]
        
        # Verify performance metadata
        assert 'input_shape' in result.metadata
        assert 'output_shape' in result.metadata
        assert 'context_length' in result.metadata
        assert 'has_quantiles' in result.metadata
        assert result.metadata['has_quantiles'] == True
        
        # Verify actual inference happened (not mock)
        assert np.all(np.isfinite(result.predictions))
        assert len(result.predictions) == 24


if __name__ == "__main__":
    pytest.main([__file__, "-v"])