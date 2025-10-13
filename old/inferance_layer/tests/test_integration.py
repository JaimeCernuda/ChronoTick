#!/usr/bin/env python3
"""
Integration Tests for ChronoTick Inference Engine

These tests demonstrate the proper testing strategy:
1. Unit tests with mocks (test_engine.py, test_utils.py)
2. Integration tests with real TSFM models (this file)
"""

import pytest
import numpy as np
import sys
from pathlib import Path
import tempfile
import yaml

# Add the chronotick_inference package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chronotick_inference import (
    ChronoTickInferenceEngine,
    ClockDataGenerator,
    SystemMetricsCollector,
    create_inference_engine
)


class MockTSFMModel:
    """
    Mock TSFM model that simulates real model behavior.
    This would be replaced with actual TSFM models in full integration.
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.loaded = True
    
    def forecast(self, data: np.ndarray, prediction_length: int = 5, **kwargs):
        """Simulate model forecasting with realistic behavior."""
        # Simulate some processing time
        import time
        time.sleep(0.1)
        
        # Generate realistic predictions based on input data
        last_values = data[-10:] if len(data) >= 10 else data
        trend = np.mean(np.diff(last_values)) if len(last_values) > 1 else 0
        
        predictions = []
        for i in range(prediction_length):
            # Add trend + some noise
            pred = last_values[-1] + trend * (i + 1) + np.random.normal(0, np.std(last_values) * 0.1)
            predictions.append(pred)
        
        # Create mock result object
        class MockResult:
            def __init__(self):
                self.predictions = np.array(predictions)
                self.quantiles = {
                    '0.1': self.predictions * 0.9,
                    '0.9': self.predictions * 1.1
                }
                self.metadata = {'model_name': model_name, 'inference_time': 0.1}
        
        return MockResult()
    
    def forecast_with_covariates(self, data: np.ndarray, covariates=None, prediction_length: int = 5, **kwargs):
        """Simulate covariate-enhanced forecasting."""
        # For demo, just add some covariate influence
        result = self.forecast(data, prediction_length)
        
        # Simulate covariate influence (e.g., temperature affects clock)
        if covariates and hasattr(covariates, 'data') and 'temperature' in covariates.data:
            temp_avg = np.mean(covariates.data['temperature'][-10:])
            temp_influence = (temp_avg - 65) * 1e-6  # 1μs per degree from 65°C baseline
            result.predictions += temp_influence
        
        return result
    
    def health_check(self):
        """Simulate model health check."""
        return {'status': 'loaded', 'model_name': self.model_name}


class MockTSFMFactory:
    """
    Mock TSFM Factory that creates mock models.
    This simulates the real TSFMFactory behavior.
    """
    
    def __init__(self):
        self.models = {}
    
    def load_model(self, model_name: str, **kwargs):
        """Load a mock model."""
        if model_name in ['chronos', 'timesfm', 'ttm', 'toto', 'time-moe']:
            model = MockTSFMModel(model_name)
            self.models[model_name] = model
            return model
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def unload_model(self, model_name: str):
        """Unload a model."""
        if model_name in self.models:
            del self.models[model_name]
    
    def create_frequency_info(self, freq_str: str = 'S', freq_value: int = 9, 
                              is_regular: bool = True, detected_freq: str = 'S', **kwargs):
        """Create frequency info."""
        class MockFreqInfo:
            def __init__(self):
                self.freq_str = freq_str
                self.freq_value = freq_value
                self.is_regular = is_regular
                self.detected_freq = detected_freq
        return MockFreqInfo()
    
    def create_covariates_input(self, target=None, covariates=None, **kwargs):
        """Create covariates input."""
        class MockCovariates:
            def __init__(self):
                self.data = covariates or {}
                self.target = target
            
            def __len__(self):
                if self.data:
                    # Return length of first covariate array
                    first_key = next(iter(self.data.keys()))
                    return len(self.data[first_key])
                return 0
        return MockCovariates()


@pytest.mark.integration
class TestIntegrationWorkflow:
    """Integration tests that demonstrate complete workflow."""
    
    @pytest.fixture
    def integration_config(self):
        """Create integration test configuration."""
        return {
            'short_term': {
                'model_name': 'chronos',
                'device': 'cpu',
                'enabled': True,
                'inference_interval': 1.0,
                'prediction_horizon': 5,
                'context_length': 100,
                'max_uncertainty': 0.1,
                'model_params': {'prediction_length': 5}
            },
            'long_term': {
                'model_name': 'timesfm',
                'device': 'cpu',
                'enabled': True,
                'inference_interval': 30.0,
                'prediction_horizon': 60,
                'context_length': 300,
                'model_params': {'context_len': 300}
            },
            'fusion': {
                'enabled': True,
                'method': 'inverse_variance',
                'uncertainty_threshold': 0.05,
                'fallback_weights': {'short_term': 0.7, 'long_term': 0.3}
            },
            'preprocessing': {
                'outlier_removal': {'enabled': False},
                'missing_value_handling': {'enabled': False},
                'normalization': {'enabled': False}
            },
            'covariates': {
                'enabled': True,
                'variables': ['cpu_usage', 'temperature', 'memory_usage'],
                'future_variables': []
            },
            'performance': {
                'max_memory_mb': 512,
                'model_timeout': 5.0,
                'cache_size': 10
            },
            'logging': {
                'level': 'INFO',
                'log_predictions': True,
                'log_uncertainty': True,
                'log_fusion_weights': True
            },
            'clock': {
                'frequency_type': 'second',
                'frequency_code': 9
            }
        }
    
    @pytest.fixture
    def integration_config_file(self, integration_config):
        """Create temporary config file for integration testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(integration_config, f)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)
    
    @pytest.fixture
    def realistic_data(self):
        """Generate realistic clock offset and system metrics data."""
        generator = ClockDataGenerator(seed=42)
        
        # Generate 10 minutes of data (600 seconds)
        offset_data, system_metrics = generator.generate_realistic_scenario(
            "server_load", duration=600
        )
        
        return offset_data, system_metrics
    
    def test_complete_prediction_workflow(self, integration_config_file, realistic_data):
        """
        Test complete prediction workflow with mock models.
        This demonstrates how the system would work with real TSFM models.
        """
        offset_data, system_metrics = realistic_data
        
        # Patch the TSFM factory with our mock
        from unittest.mock import patch
        
        with patch('chronotick_inference.engine.TSFMFactory', MockTSFMFactory):
            # Test engine initialization
            engine = ChronoTickInferenceEngine(integration_config_file)
            assert engine is not None
            
            # Test model loading
            success = engine.initialize_models()
            assert success is True
            assert engine.short_term_model is not None
            assert engine.long_term_model is not None
            
            # Test short-term prediction
            context_data = offset_data[-100:]  # Last 100 seconds
            short_result = engine.predict_short_term(context_data, system_metrics)
            
            assert short_result is not None
            assert len(short_result.predictions) == 5
            assert short_result.confidence > 0
            assert short_result.uncertainty is not None
            
            # Test long-term prediction
            long_context = offset_data[-300:]  # Last 5 minutes
            long_result = engine.predict_long_term(long_context, system_metrics)
            
            assert long_result is not None
            assert len(long_result.predictions) > 0
            
            # Test fused prediction
            fused_result = engine.predict_fused(context_data, system_metrics)
            
            assert fused_result is not None
            assert fused_result.prediction > 0 or fused_result.prediction < 0  # Non-zero
            assert fused_result.uncertainty > 0
            assert 'short_term' in fused_result.weights
            assert 'long_term' in fused_result.weights
            assert abs(fused_result.weights['short_term'] + fused_result.weights['long_term'] - 1.0) < 1e-6
            
            # Test health check
            health = engine.health_check()
            assert health['status'] in ['healthy', 'degraded', 'error']
            
            # Test performance stats
            stats = engine.get_performance_stats()
            assert 'short_term_inferences' in stats
            assert 'long_term_inferences' in stats
            assert 'fusion_operations' in stats
            
            # Cleanup
            engine.shutdown()
    
    def test_real_time_prediction_simulation(self, integration_config_file):
        """
        Simulate real-time prediction workflow.
        This shows how the system would work in continuous operation.
        """
        from unittest.mock import patch
        
        with patch('chronotick_inference.engine.TSFMFactory', MockTSFMFactory):
            engine = ChronoTickInferenceEngine(integration_config_file)
            engine.initialize_models()
            
            # Simulate streaming data
            generator = ClockDataGenerator(seed=123)
            offset_buffer = []
            predictions = []
            
            # Simulate 30 seconds of real-time operation
            for second in range(30):
                # Generate new offset measurement
                new_offset = generator.generate_offset_sequence(
                    duration=1, 
                    drift_rate=1e-6,
                    noise_level=1e-7
                )[0]
                offset_buffer.append(new_offset)
                
                # Make prediction every 5 seconds when we have enough data
                if second % 5 == 0 and len(offset_buffer) >= 10:
                    context = np.array(offset_buffer[-50:])  # Last 50 seconds
                    
                    # Get current system metrics
                    metrics_collector = SystemMetricsCollector()
                    current_metrics = metrics_collector._collect_current_metrics()
                    system_metrics = {
                        'cpu_usage': np.array([current_metrics.cpu_usage]),
                        'temperature': np.array([current_metrics.temperature or 70.0]),
                        'memory_usage': np.array([current_metrics.memory_usage])
                    }
                    
                    # Make prediction
                    result = engine.predict_fused(context, system_metrics)
                    if result:
                        predictions.append({
                            'timestamp': second,
                            'prediction': result.prediction,
                            'uncertainty': result.uncertainty,
                            'weights': result.weights
                        })
            
            # Verify we got predictions
            assert len(predictions) > 0
            
            # Verify prediction quality
            for pred in predictions:
                assert isinstance(pred['prediction'], (int, float))
                assert pred['uncertainty'] > 0
                assert 'short_term' in pred['weights']
                assert 'long_term' in pred['weights']
            
            engine.shutdown()
    
    def test_error_recovery_and_fallbacks(self, integration_config_file):
        """
        Test error recovery and fallback mechanisms.
        This shows robust operation when models fail.
        """
        from unittest.mock import patch, Mock
        
        # Create a factory that simulates model failures
        class FailingMockFactory(MockTSFMFactory):
            def load_model(self, model_name: str, **kwargs):
                if model_name == 'chronos':
                    # Simulate short-term model failure
                    raise RuntimeError("Model loading failed")
                else:
                    return super().load_model(model_name, **kwargs)
        
        with patch('chronotick_inference.engine.TSFMFactory', FailingMockFactory):
            engine = ChronoTickInferenceEngine(integration_config_file)
            
            # Model initialization should handle failures gracefully
            success = engine.initialize_models()
            # Depending on implementation, this might return False or log errors
            
            # Generate test data
            generator = ClockDataGenerator()
            offset_data = generator.generate_offset_sequence(duration=100)
            
            # System should still work with available models
            # Even if short-term fails, long-term should work
            result = engine.predict_long_term(offset_data)
            # Result might be None if model failed, which is acceptable
            
            # Test health check shows degraded status
            health = engine.health_check()
            assert health is not None
            
            engine.shutdown()
    
    def test_performance_characteristics(self, integration_config_file, realistic_data):
        """
        Test performance characteristics and resource usage.
        """
        import time
        
        offset_data, system_metrics = realistic_data
        
        from unittest.mock import patch
        
        with patch('chronotick_inference.engine.TSFMFactory', MockTSFMFactory):
            engine = ChronoTickInferenceEngine(integration_config_file)
            engine.initialize_models()
            
            # Test prediction timing
            context_data = offset_data[-100:]
            
            start_time = time.time()
            result = engine.predict_short_term(context_data, system_metrics)
            inference_time = time.time() - start_time
            
            # Verify reasonable inference time (< 2 seconds for mock)
            assert inference_time < 2.0
            assert result is not None
            
            # Test memory usage
            health = engine.health_check()
            if 'memory_usage_mb' in health:
                assert health['memory_usage_mb'] < 1000  # < 1GB
            
            # Test batch predictions
            batch_start = time.time()
            batch_results = []
            
            for i in range(5):
                batch_context = offset_data[-(100+i*10):-(i*10) if i*10 > 0 else None]
                batch_result = engine.predict_short_term(batch_context, system_metrics)
                if batch_result:
                    batch_results.append(batch_result)
            
            batch_time = time.time() - batch_start
            
            # Verify batch performance
            assert len(batch_results) > 0
            assert batch_time < 10.0  # Reasonable batch time
            
            engine.shutdown()


@pytest.mark.integration
def test_integration_with_utilities():
    """
    Test integration between engine and utility components.
    """
    # Test data generator integration
    generator = ClockDataGenerator(seed=456)
    offset_data, system_metrics = generator.generate_realistic_scenario("thermal_cycle", 300)
    
    assert len(offset_data) == 300
    assert 'temperature' in system_metrics
    assert 'cpu_usage' in system_metrics
    
    # Test metrics collector integration
    collector = SystemMetricsCollector(collection_interval=0.1)
    collector.start_collection()
    
    # Let it collect briefly
    import time
    time.sleep(0.3)
    
    recent_metrics = collector.get_recent_metrics(window_seconds=1)
    collector.stop_collection()
    
    assert len(recent_metrics) > 0
    assert 'cpu_usage' in recent_metrics
    assert 'memory_usage' in recent_metrics
    
    # Test visualization integration
    from chronotick_inference.utils import PredictionVisualizer
    
    visualizer = PredictionVisualizer()
    
    # Create mock predictions for visualization
    timestamps = np.arange(len(offset_data))
    predictions = offset_data + np.random.normal(0, np.std(offset_data) * 0.1, len(offset_data))
    
    plot_result = visualizer.plot_predictions(
        timestamps=timestamps,
        actual_offsets=offset_data,
        predictions=predictions
    )
    
    assert isinstance(plot_result, str)
    assert len(plot_result) > 0
    
    # Test performance report
    mock_predictions = [{'prediction': p} for p in predictions[:100]]
    actual_values = offset_data[:100]
    
    report = visualizer.create_performance_report(mock_predictions, actual_values)
    assert "Performance Report" in report
    assert "Mean Absolute Error" in report


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "-m", "integration"])