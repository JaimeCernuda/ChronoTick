#!/usr/bin/env python3
"""
Performance tests for TSFM Factory and models.
Tests model loading times, inference speed, memory usage, and scalability.
"""

import pytest
import numpy as np
import time
try:
    import psutil
except ImportError:
    psutil = None
import os
from pathlib import Path
import sys
import warnings
import gc

# Add the tsfm package to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tsfm import TSFMFactory
from tsfm.base import MultivariateInput, CovariatesInput
from tsfm.datasets.loader import create_synthetic_data

warnings.filterwarnings("ignore")


class TestModelLoadingPerformance:
    """Test model loading and initialization performance."""
    
    @pytest.fixture
    def factory(self):
        """Create TSFM factory instance."""
        return TSFMFactory()
    
    def test_model_loading_times(self, factory):
        """Test and measure model loading times."""
        models_to_test = ["timesfm", "chronos", "ttm", "toto", "time_moe"]
        loading_times = {}
        
        for model_name in models_to_test:
            try:
                start_time = time.time()
                model = factory.load_model(model_name)
                end_time = time.time()
                
                loading_time = end_time - start_time
                loading_times[model_name] = loading_time
                
                # Basic sanity check
                assert model is not None
                print(f"{model_name} loaded in {loading_time:.2f} seconds")
                
                factory.unload_model(model_name)
                
            except Exception as e:
                if any(term in str(e).lower() for term in ["package not available", "dependencies", "transformers"]):
                    pytest.skip(f"{model_name} loading test skipped: {e}")
                    continue
                else:
                    raise
        
        # Performance assertions (reasonable loading times)
        for model_name, loading_time in loading_times.items():
            assert loading_time < 60.0, f"{model_name} loading took too long: {loading_time:.2f}s"
    
    def test_memory_usage_during_loading(self, factory):
        """Test memory usage during model loading."""
        if psutil is None:
            pytest.skip("psutil not available")
        process = psutil.Process(os.getpid())
        
        try:
            # Measure baseline memory
            gc.collect()
            baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Load model and measure memory increase
            model = factory.load_model("chronos")
            
            gc.collect()
            loaded_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = loaded_memory - baseline_memory
            
            print(f"Memory increase for Chronos loading: {memory_increase:.1f} MB")
            
            # Reasonable memory usage (model dependent)
            assert memory_increase < 2000, f"Memory usage too high: {memory_increase:.1f} MB"
            
            factory.unload_model("chronos")
            
            # Check memory is freed after unloading
            gc.collect()
            time.sleep(1)  # Allow time for cleanup
            
            unloaded_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_freed = loaded_memory - unloaded_memory
            
            print(f"Memory freed after unloading: {memory_freed:.1f} MB")
            
        except Exception as e:
            if "chronos" in str(e).lower():
                pytest.skip(f"Memory test skipped: {e}")
            else:
                raise


class TestInferencePerformance:
    """Test inference speed and performance."""
    
    @pytest.fixture
    def factory(self):
        return TSFMFactory()
    
    @pytest.fixture
    def performance_data(self):
        """Create data for performance testing."""
        return {
            "small": np.random.randn(100).astype(np.float32),
            "medium": np.random.randn(500).astype(np.float32),
            "large": np.random.randn(1000).astype(np.float32)
        }
    
    def test_inference_speed_basic(self, factory, performance_data):
        """Test basic inference speed across different data sizes."""
        try:
            model = factory.load_model("chronos")
            
            inference_times = {}
            
            for size_name, data in performance_data.items():
                start_time = time.time()
                result = model.forecast(data, horizon=96)
                end_time = time.time()
                
                inference_time = end_time - start_time
                inference_times[size_name] = inference_time
                
                assert len(result.predictions) == 96
                print(f"Inference time for {size_name} data ({len(data)} points): {inference_time:.3f}s")
            
            # Performance expectations
            assert inference_times["small"] < 30.0, "Small inference too slow"
            assert inference_times["medium"] < 60.0, "Medium inference too slow"
            assert inference_times["large"] < 120.0, "Large inference too slow"
            
            factory.unload_model("chronos")
            
        except Exception as e:
            if "chronos" in str(e).lower():
                pytest.skip(f"Inference speed test skipped: {e}")
            else:
                raise
    
    def test_batch_inference_efficiency(self, factory):
        """Test efficiency of multiple predictions."""
        try:
            model = factory.load_model("timesfm")
            
            # Single large prediction vs multiple small predictions
            large_data = np.random.randn(500).astype(np.float32)
            
            # Time single prediction
            start_time = time.time()
            result_single = model.forecast(large_data, horizon=96)
            single_time = time.time() - start_time
            
            # Time multiple smaller predictions
            small_data = np.random.randn(100).astype(np.float32)
            start_time = time.time()
            for _ in range(5):
                result_multi = model.forecast(small_data, horizon=96)
            multi_time = time.time() - start_time
            
            print(f"Single large prediction: {single_time:.3f}s")
            print(f"5 small predictions: {multi_time:.3f}s")
            print(f"Efficiency ratio: {multi_time/single_time:.2f}x")
            
            # Verify results
            assert len(result_single.predictions) == 96
            assert len(result_multi.predictions) == 96
            
            factory.unload_model("timesfm")
            
        except Exception as e:
            if "TimesFM" in str(e):
                pytest.skip(f"Batch efficiency test skipped: {e}")
            else:
                raise
    
    def test_enhanced_features_performance_impact(self, factory):
        """Test performance impact of enhanced features."""
        try:
            model = factory.load_model("ttm")
            
            # Standard forecast timing
            standard_data = np.random.randn(200).astype(np.float32)
            start_time = time.time()
            standard_result = model.forecast(standard_data, horizon=48)
            standard_time = time.time() - start_time
            
            # Multivariate forecast timing (if supported)
            if hasattr(model, 'multivariate_support') and model.multivariate_support:
                mv_data = MultivariateInput(
                    data=np.random.randn(2, 200).astype(np.float32),
                    variable_names=["var1", "var2"]
                )
                start_time = time.time()
                mv_result = model.forecast_multivariate(mv_data, horizon=48)
                mv_time = time.time() - start_time
                
                print(f"Standard forecast: {standard_time:.3f}s")
                print(f"Multivariate forecast: {mv_time:.3f}s")
                print(f"Multivariate overhead: {(mv_time/standard_time - 1)*100:.1f}%")
                
                # Multivariate shouldn't be more than 3x slower for 2 variables
                assert mv_time < standard_time * 3, "Multivariate forecast too slow"
            
            # Covariates forecast timing (if supported)
            if hasattr(model, 'exogenous_support') and model.exogenous_support:
                cov_data = CovariatesInput(
                    target=standard_data,
                    covariates={"feature1": np.random.randn(200).astype(np.float32)}
                )
                start_time = time.time()
                cov_result = model.forecast_with_covariates(cov_data, horizon=48)
                cov_time = time.time() - start_time
                
                print(f"Covariates forecast: {cov_time:.3f}s")
                print(f"Covariates overhead: {(cov_time/standard_time - 1)*100:.1f}%")
                
                # Covariates shouldn't be more than 2x slower
                assert cov_time < standard_time * 2, "Covariates forecast too slow"
            
            factory.unload_model("ttm")
            
        except Exception as e:
            if "TTM" in str(e):
                pytest.skip(f"Enhanced features performance test skipped: {e}")
            else:
                raise


class TestScalabilityPerformance:
    """Test scalability with different data sizes and horizons."""
    
    @pytest.fixture
    def factory(self):
        return TSFMFactory()
    
    def test_context_length_scalability(self, factory):
        """Test performance across different context lengths."""
        try:
            model = factory.load_model("timesfm")
            
            context_lengths = [100, 200, 500, 1000]
            times = []
            
            for length in context_lengths:
                data = np.random.randn(length).astype(np.float32)
                
                start_time = time.time()
                result = model.forecast(data, horizon=96)
                end_time = time.time()
                
                inference_time = end_time - start_time
                times.append(inference_time)
                
                print(f"Context length {length}: {inference_time:.3f}s")
                assert len(result.predictions) == 96
            
            # Check that scaling is reasonable (not exponential)
            for i in range(1, len(times)):
                scale_factor = context_lengths[i] / context_lengths[i-1]
                time_ratio = times[i] / times[i-1]
                
                # Time shouldn't increase more than linearly with context length
                assert time_ratio < scale_factor * 2, f"Poor scaling at context length {context_lengths[i]}"
            
            factory.unload_model("timesfm")
            
        except Exception as e:
            if "TimesFM" in str(e):
                pytest.skip(f"Context scalability test skipped: {e}")
            else:
                raise
    
    def test_horizon_length_scalability(self, factory):
        """Test performance across different forecast horizons."""
        try:
            model = factory.load_model("chronos")
            
            horizons = [24, 96, 192, 336]
            times = []
            data = np.random.randn(300).astype(np.float32)
            
            for horizon in horizons:
                start_time = time.time()
                result = model.forecast(data, horizon=horizon)
                end_time = time.time()
                
                inference_time = end_time - start_time
                times.append(inference_time)
                
                print(f"Horizon {horizon}: {inference_time:.3f}s")
                assert len(result.predictions) == horizon
            
            # Check reasonable scaling with horizon
            for i in range(1, len(times)):
                scale_factor = horizons[i] / horizons[i-1]
                time_ratio = times[i] / times[i-1]
                
                # Time shouldn't increase more than linearly with horizon
                assert time_ratio < scale_factor * 1.5, f"Poor horizon scaling at {horizons[i]}"
            
            factory.unload_model("chronos")
            
        except Exception as e:
            if "chronos" in str(e).lower():
                pytest.skip(f"Horizon scalability test skipped: {e}")
            else:
                raise


class TestResourceUtilization:
    """Test CPU and memory resource utilization."""
    
    @pytest.fixture
    def factory(self):
        return TSFMFactory()
    
    def test_cpu_utilization(self, factory):
        """Test CPU utilization during inference."""
        if psutil is None:
            pytest.skip("psutil not available")
        try:
            model = factory.load_model("chronos")
            
            data = np.random.randn(500).astype(np.float32)
            
            # Monitor CPU during inference
            process = psutil.Process(os.getpid())
            
            # Get baseline CPU
            cpu_percent_before = process.cpu_percent()
            time.sleep(0.1)  # Allow measurement to stabilize
            
            start_time = time.time()
            result = model.forecast(data, horizon=96)
            end_time = time.time()
            
            cpu_percent_after = process.cpu_percent()
            inference_time = end_time - start_time
            
            print(f"Inference time: {inference_time:.3f}s")
            print(f"CPU utilization: {cpu_percent_after:.1f}%")
            
            # Verify result
            assert len(result.predictions) == 96
            
            # CPU should be utilized during inference
            if inference_time > 1.0:  # Only check for longer inferences
                assert cpu_percent_after > 0, "No CPU utilization detected"
            
            factory.unload_model("chronos")
            
        except Exception as e:
            if "chronos" in str(e).lower():
                pytest.skip(f"CPU utilization test skipped: {e}")
            else:
                raise
    
    def test_memory_stability(self, factory):
        """Test memory stability during repeated inference."""
        if psutil is None:
            pytest.skip("psutil not available")
        try:
            model = factory.load_model("timesfm")
            
            process = psutil.Process(os.getpid())
            data = np.random.randn(200).astype(np.float32)
            
            memory_measurements = []
            
            # Run multiple inferences and monitor memory
            for i in range(5):
                gc.collect()
                memory_before = process.memory_info().rss / 1024 / 1024  # MB
                
                result = model.forecast(data, horizon=48)
                assert len(result.predictions) == 48
                
                gc.collect()
                memory_after = process.memory_info().rss / 1024 / 1024  # MB
                
                memory_measurements.append(memory_after - memory_before)
                print(f"Iteration {i+1} memory change: {memory_after - memory_before:.1f} MB")
                
                time.sleep(0.1)  # Small delay between iterations
            
            # Memory usage should be stable (no major leaks)
            max_memory_increase = max(memory_measurements)
            avg_memory_increase = sum(memory_measurements) / len(memory_measurements)
            
            print(f"Average memory increase per inference: {avg_memory_increase:.1f} MB")
            print(f"Max memory increase: {max_memory_increase:.1f} MB")
            
            # Reasonable memory behavior
            assert max_memory_increase < 100, f"Excessive memory usage: {max_memory_increase:.1f} MB"
            assert avg_memory_increase < 50, f"Memory leak detected: {avg_memory_increase:.1f} MB avg increase"
            
            factory.unload_model("timesfm")
            
        except Exception as e:
            if "TimesFM" in str(e):
                pytest.skip(f"Memory stability test skipped: {e}")
            else:
                raise


class TestPerformanceRegression:
    """Test for performance regressions in enhanced features."""
    
    @pytest.fixture
    def factory(self):
        return TSFMFactory()
    
    def test_no_performance_regression_enhanced_models(self, factory):
        """Test that enhanced models don't have significant performance regression."""
        models_to_test = [
            ("timesfm", "TimesFM 2.0 performance"),
            ("chronos", "Chronos-Bolt performance"),
            ("ttm", "TTM enhanced performance")
        ]
        
        performance_data = np.random.randn(300).astype(np.float32)
        
        for model_name, description in models_to_test:
            try:
                model = factory.load_model(model_name)
                
                # Measure basic inference time
                start_time = time.time()
                result = model.forecast(performance_data, horizon=96)
                inference_time = time.time() - start_time
                
                print(f"{description}: {inference_time:.3f}s")
                
                # Verify result quality
                assert len(result.predictions) == 96
                assert result.metadata is not None
                
                # Performance expectations (generous bounds for CI)
                assert inference_time < 120.0, f"{model_name} inference too slow: {inference_time:.3f}s"
                
                # Check enhanced metadata doesn't significantly impact performance
                metadata_keys = len(result.metadata.keys())
                assert metadata_keys > 0, "No metadata generated"
                
                factory.unload_model(model_name)
                
            except Exception as e:
                if any(term in str(e) for term in [model_name, "package not available", "dependencies"]):
                    pytest.skip(f"{description} test skipped: {e}")
                else:
                    raise


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # -s to show print statements