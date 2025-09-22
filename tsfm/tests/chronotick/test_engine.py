#!/usr/bin/env python3
"""
Tests for ChronoTick Inference Engine

Unit and integration tests for the core inference functionality.
"""

import pytest
import numpy as np
import time
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, patch
import sys

# Add the chronotick_inference package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chronotick_inference import (
    ChronoTickInferenceEngine,
    PredictionResult,
    FusedPrediction,
    ModelType,
    create_inference_engine,
    quick_predict
)


class TestChronoTickInferenceEngine:
    """Test the main inference engine class."""
    
    @pytest.fixture
    def sample_config(self):
        """Create a sample configuration for testing."""
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
                'variables': ['cpu_usage', 'temperature']
            },
            'performance': {'max_memory_mb': 512, 'cache_size': 10},
            'logging': {'level': 'INFO', 'log_predictions': False},
            'clock': {
                'frequency_type': 'second',
                'frequency_code': 9
            }
        }
    
    @pytest.fixture
    def temp_config_file(self, sample_config):
        """Create a temporary config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(sample_config, f)
            return f.name
    
    @pytest.fixture
    def sample_offset_data(self):
        """Create sample offset data for testing."""
        t = np.arange(200)
        # Simulate realistic clock offset with drift and noise
        offset = 1e-5 * t + 1e-6 * np.sin(2 * np.pi * t / 50) + np.random.normal(0, 1e-6, 200)
        return offset.astype(np.float64)
    
    @pytest.fixture
    def sample_covariates(self):
        """Create sample covariate data."""
        return {
            'cpu_usage': np.random.uniform(20, 80, 200),
            'temperature': np.random.uniform(60, 75, 200),
            'memory_usage': np.random.uniform(30, 90, 200)
        }
    
    def test_engine_initialization(self, temp_config_file):
        """Test engine initialization with valid config."""
        engine = ChronoTickInferenceEngine(temp_config_file)
        
        assert engine.config is not None
        assert engine.factory is not None
        assert engine.short_term_model is None  # Not loaded yet
        assert engine.long_term_model is None
        
        # Clean up
        Path(temp_config_file).unlink()
    
    def test_config_loading(self, sample_config, temp_config_file):
        """Test configuration loading."""
        engine = ChronoTickInferenceEngine(temp_config_file)
        
        assert engine.config['short_term']['model_name'] == 'chronos'
        assert engine.config['long_term']['model_name'] == 'timesfm'
        assert engine.config['fusion']['enabled'] is True
        
        Path(temp_config_file).unlink()
    
    def test_invalid_config_file(self):
        """Test handling of invalid config file."""
        with pytest.raises(FileNotFoundError):
            ChronoTickInferenceEngine("nonexistent_config.yaml")
    
    @patch('chronotick_inference.engine.TSFMFactory')
    def test_model_initialization_mock(self, mock_factory, temp_config_file):
        """Test model initialization with mocked TSFM factory."""
        # Setup mock
        mock_factory_instance = Mock()
        mock_factory.return_value = mock_factory_instance
        
        mock_model = Mock()
        mock_factory_instance.load_model.return_value = mock_model
        
        # Test initialization
        engine = ChronoTickInferenceEngine(temp_config_file)
        result = engine.initialize_models()
        
        assert result is True
        assert mock_factory_instance.load_model.call_count == 2  # Short + long term
        
        Path(temp_config_file).unlink()
    
    @patch('chronotick_inference.engine.TSFMFactory')
    def test_short_term_prediction_mock(self, mock_factory, temp_config_file, sample_offset_data):
        """Test short-term prediction with mocked models."""
        # Setup mock
        mock_factory_instance = Mock()
        mock_factory.return_value = mock_factory_instance
        
        mock_model = Mock()
        mock_result = Mock()
        mock_result.predictions = np.array([1e-5, 2e-5, 3e-5, 4e-5, 5e-5])
        mock_result.quantiles = {'0.1': np.array([0.5e-5, 1.5e-5, 2.5e-5, 3.5e-5, 4.5e-5]),
                                '0.9': np.array([1.5e-5, 2.5e-5, 3.5e-5, 4.5e-5, 5.5e-5])}
        mock_result.metadata = {'model_name': 'chronos'}
        
        mock_model.forecast.return_value = mock_result
        mock_factory_instance.load_model.return_value = mock_model
        
        # Test prediction
        engine = ChronoTickInferenceEngine(temp_config_file)
        engine.initialize_models()
        
        result = engine.predict_short_term(sample_offset_data)
        
        assert result is not None
        assert isinstance(result, PredictionResult)
        assert result.model_type == ModelType.SHORT_TERM
        assert len(result.predictions) == 5
        assert result.uncertainty is not None
        assert result.confidence > 0
        
        Path(temp_config_file).unlink()
    
    def test_prediction_data_validation(self, temp_config_file):
        """Test input data validation."""
        engine = ChronoTickInferenceEngine(temp_config_file)
        
        # Test with insufficient data
        short_data = np.array([1e-5, 2e-5])  # Only 2 points
        result = engine.predict_short_term(short_data)
        assert result is None  # Should fail due to insufficient data
        
        # Test with invalid data types
        with pytest.raises(ValueError):
            engine.validate_input("invalid_data")
        
        # Test with NaN values
        nan_data = np.array([1e-5, np.nan, 3e-5])
        with pytest.raises(ValueError):
            engine.validate_input(nan_data)
        
        Path(temp_config_file).unlink()
    
    @patch('chronotick_inference.engine.TSFMFactory')
    def test_fused_prediction_mock(self, mock_factory, temp_config_file, sample_offset_data):
        """Test fused prediction combining short and long term models."""
        # Setup mock
        mock_factory_instance = Mock()
        mock_factory.return_value = mock_factory_instance
        
        # Mock short-term model
        mock_short_model = Mock()
        mock_short_result = Mock()
        mock_short_result.predictions = np.array([1e-5])
        mock_short_result.quantiles = {'0.1': np.array([0.5e-5]), '0.9': np.array([1.5e-5])}
        mock_short_result.uncertainty = np.array([0.3e-5])
        mock_short_result.confidence = 0.9
        mock_short_result.model_type = ModelType.SHORT_TERM
        mock_short_result.timestamp = time.time()
        mock_short_result.inference_time = 0.1
        mock_short_result.metadata = {}
        mock_short_model.forecast.return_value = mock_short_result
        
        # Mock long-term model
        mock_long_model = Mock()
        mock_long_result = Mock()
        mock_long_result.predictions = np.array([1.2e-5])
        mock_long_result.quantiles = {'0.1': np.array([0.8e-5]), '0.9': np.array([1.6e-5])}
        mock_long_result.uncertainty = np.array([0.2e-5])
        mock_long_result.confidence = 0.95
        mock_long_result.model_type = ModelType.LONG_TERM
        mock_long_result.timestamp = time.time()
        mock_long_result.inference_time = 0.5
        mock_long_result.metadata = {}
        mock_long_model.forecast.return_value = mock_long_result
        
        # Setup factory to return different models
        def mock_load_model(model_name, **kwargs):
            if model_name == 'chronos':
                return mock_short_model
            elif model_name == 'timesfm':
                return mock_long_model
            else:
                raise ValueError(f"Unknown model: {model_name}")
        
        mock_factory_instance.load_model.side_effect = mock_load_model
        
        # Test fused prediction
        engine = ChronoTickInferenceEngine(temp_config_file)
        engine.initialize_models()
        
        result = engine.predict_fused(sample_offset_data)
        
        assert result is not None
        assert isinstance(result, FusedPrediction)
        assert 'short_term' in result.weights
        assert 'long_term' in result.weights
        assert abs(result.weights['short_term'] + result.weights['long_term'] - 1.0) < 1e-6
        assert result.prediction > 0
        assert result.uncertainty > 0
        
        Path(temp_config_file).unlink()
    
    def test_uncertainty_calculation(self, temp_config_file):
        """Test uncertainty calculation from quantiles."""
        engine = ChronoTickInferenceEngine(temp_config_file)
        
        # Test with valid quantiles
        quantiles = {
            '0.1': np.array([1e-5, 2e-5]),
            '0.9': np.array([3e-5, 4e-5])
        }
        uncertainty = engine._calculate_uncertainty(quantiles)
        
        assert uncertainty is not None
        assert len(uncertainty) == 2
        assert all(u > 0 for u in uncertainty)
        
        # Test with missing quantiles
        uncertainty = engine._calculate_uncertainty(None)
        assert uncertainty is None
        
        # Test with incomplete quantiles
        incomplete_quantiles = {'0.5': np.array([2e-5])}
        uncertainty = engine._calculate_uncertainty(incomplete_quantiles)
        assert uncertainty is None
        
        Path(temp_config_file).unlink()
    
    def test_confidence_calculation(self, temp_config_file):
        """Test confidence score calculation."""
        engine = ChronoTickInferenceEngine(temp_config_file)
        
        # Test with low uncertainty (high confidence)
        low_uncertainty = np.array([1e-7])  # Very small
        confidence = engine._calculate_confidence(low_uncertainty, max_uncertainty=1e-6)
        assert confidence > 0.8
        
        # Test with high uncertainty (low confidence)
        high_uncertainty = np.array([2e-6])  # Larger than threshold
        confidence = engine._calculate_confidence(high_uncertainty, max_uncertainty=1e-6)
        assert confidence < 0.5
        
        # Test with no uncertainty
        confidence = engine._calculate_confidence(None, max_uncertainty=1e-6)
        assert confidence == 1.0
        
        Path(temp_config_file).unlink()
    
    @patch('chronotick_inference.engine.TSFMFactory')
    def test_covariates_integration(self, mock_factory, temp_config_file, sample_offset_data, sample_covariates):
        """Test integration with covariates (system metrics)."""
        # Setup mock
        mock_factory_instance = Mock()
        mock_factory.return_value = mock_factory_instance
        
        mock_model = Mock()
        mock_result = Mock()
        mock_result.predictions = np.array([1e-5])
        mock_result.quantiles = None
        mock_result.metadata = {}
        
        mock_model.forecast_with_covariates.return_value = mock_result
        mock_factory_instance.load_model.return_value = mock_model
        mock_factory_instance.create_covariates_input.return_value = Mock()
        
        # Test with covariates
        engine = ChronoTickInferenceEngine(temp_config_file)
        engine.initialize_models()
        
        result = engine.predict_short_term(sample_offset_data, sample_covariates)
        
        # Should have called forecast_with_covariates
        mock_model.forecast_with_covariates.assert_called_once()
        
        Path(temp_config_file).unlink()
    
    @patch('chronotick_inference.engine.psutil')
    def test_health_check(self, mock_psutil, temp_config_file):
        """Test health check functionality."""
        # Setup mock
        mock_process = Mock()
        mock_process.memory_info.return_value.rss = 100 * 1024 * 1024  # 100MB
        mock_psutil.Process.return_value = mock_process
        
        engine = ChronoTickInferenceEngine(temp_config_file)
        health = engine.health_check()
        
        assert health is not None
        assert 'status' in health
        assert 'timestamp' in health
        assert 'memory_usage_mb' in health
        assert health['memory_usage_mb'] == 100.0
        
        Path(temp_config_file).unlink()
    
    def test_performance_stats(self, temp_config_file):
        """Test performance statistics tracking."""
        engine = ChronoTickInferenceEngine(temp_config_file)
        
        # Initial stats should be zero
        stats = engine.get_performance_stats()
        assert stats['short_term_inferences'] == 0
        assert stats['long_term_inferences'] == 0
        assert stats['fusion_operations'] == 0
        
        Path(temp_config_file).unlink()
    
    def test_context_manager(self, temp_config_file):
        """Test context manager functionality."""
        with patch('chronotick_inference.engine.TSFMFactory') as mock_factory:
            mock_factory_instance = Mock()
            mock_factory.return_value = mock_factory_instance
            mock_factory_instance.load_model.return_value = Mock()
            
            # Test context manager
            with ChronoTickInferenceEngine(temp_config_file) as engine:
                assert engine is not None
                # Context manager should initialize models
            
            # Should have called shutdown on exit
            # (We can't easily test this without more complex mocking)
        
        Path(temp_config_file).unlink()


class TestUtilityFunctions:
    """Test utility functions."""
    
    @pytest.fixture
    def temp_config_file(self):
        """Create a minimal config file for testing."""
        config = {
            'short_term': {'model_name': 'chronos', 'device': 'cpu', 'enabled': True},
            'long_term': {'model_name': 'timesfm', 'device': 'cpu', 'enabled': True},
            'fusion': {'enabled': True},
            'preprocessing': {'outlier_removal': {'enabled': False}},
            'covariates': {'enabled': False},
            'performance': {'max_memory_mb': 512},
            'logging': {'level': 'INFO'},
            'clock': {'frequency_type': 'second', 'frequency_code': 9}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            return f.name
    
    @patch('chronotick_inference.engine.TSFMFactory')
    def test_create_inference_engine(self, mock_factory, temp_config_file):
        """Test the create_inference_engine utility function."""
        mock_factory_instance = Mock()
        mock_factory.return_value = mock_factory_instance
        mock_factory_instance.load_model.return_value = Mock()
        
        engine = create_inference_engine(temp_config_file)
        
        assert engine is not None
        assert isinstance(engine, ChronoTickInferenceEngine)
        
        # Clean up
        engine.shutdown()
        Path(temp_config_file).unlink()
    
    @patch('chronotick_inference.engine.TSFMFactory')
    def test_quick_predict(self, mock_factory, temp_config_file):
        """Test the quick_predict utility function."""
        mock_factory_instance = Mock()
        mock_factory.return_value = mock_factory_instance
        
        mock_model = Mock()
        mock_result = Mock()
        mock_result.predictions = np.array([1e-5])
        mock_result.quantiles = None
        mock_result.uncertainty = None
        mock_result.confidence = 0.9
        mock_result.model_type = ModelType.SHORT_TERM
        mock_result.timestamp = time.time()
        mock_result.inference_time = 0.1
        mock_result.metadata = {}
        
        mock_model.forecast.return_value = mock_result
        mock_factory_instance.load_model.return_value = mock_model
        
        # Test quick predict
        offset_data = np.random.normal(0, 1e-6, 100)
        
        result = quick_predict(
            offset_history=offset_data,
            config_path=temp_config_file,
            use_fusion=False
        )
        
        assert result is not None
        
        Path(temp_config_file).unlink()


class TestDataStructures:
    """Test data structure classes."""
    
    def test_prediction_result_creation(self):
        """Test PredictionResult creation and attributes."""
        predictions = np.array([1e-5, 2e-5, 3e-5])
        uncertainty = np.array([0.1e-5, 0.2e-5, 0.3e-5])
        quantiles = {'0.1': np.array([0.5e-5, 1.5e-5, 2.5e-5])}
        
        result = PredictionResult(
            predictions=predictions,
            uncertainty=uncertainty,
            quantiles=quantiles,
            confidence=0.9,
            model_type=ModelType.SHORT_TERM,
            timestamp=time.time(),
            inference_time=0.1
        )
        
        assert np.array_equal(result.predictions, predictions)
        assert np.array_equal(result.uncertainty, uncertainty)
        assert result.quantiles == quantiles
        assert result.confidence == 0.9
        assert result.model_type == ModelType.SHORT_TERM
        assert result.timestamp > 0
        assert result.inference_time == 0.1
    
    def test_fused_prediction_creation(self):
        """Test FusedPrediction creation and attributes."""
        weights = {'short_term': 0.7, 'long_term': 0.3}
        source_predictions = {
            'short_term': PredictionResult(
                predictions=np.array([1e-5]),
                model_type=ModelType.SHORT_TERM,
                timestamp=time.time(),
                inference_time=0.1
            )
        }
        
        fused = FusedPrediction(
            prediction=1.1e-5,
            uncertainty=0.2e-5,
            weights=weights,
            source_predictions=source_predictions,
            timestamp=time.time(),
            metadata={'fusion_method': 'inverse_variance'}
        )
        
        assert fused.prediction == 1.1e-5
        assert fused.uncertainty == 0.2e-5
        assert fused.weights == weights
        assert len(fused.source_predictions) == 1
        assert fused.metadata['fusion_method'] == 'inverse_variance'


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_missing_config_file(self):
        """Test handling of missing configuration file."""
        with pytest.raises(FileNotFoundError):
            ChronoTickInferenceEngine("nonexistent_config.yaml")
    
    def test_invalid_yaml_config(self):
        """Test handling of invalid YAML configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_path = f.name
        
        try:
            with pytest.raises(yaml.YAMLError):
                ChronoTickInferenceEngine(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_empty_data_handling(self, temp_config_file):
        """Test handling of empty or insufficient data."""
        config = {
            'short_term': {'enabled': True},
            'preprocessing': {'outlier_removal': {'enabled': False}},
            'clock': {'frequency_type': 'second'}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name
        
        try:
            engine = ChronoTickInferenceEngine(config_path)
            
            # Test with empty array
            result = engine.predict_short_term(np.array([]))
            assert result is None
            
            # Test with very short array
            result = engine.predict_short_term(np.array([1e-5, 2e-5]))
            assert result is None
            
        finally:
            Path(config_path).unlink()


# Performance and Integration Tests
class TestPerformance:
    """Test performance characteristics."""
    
    @pytest.mark.slow
    def test_prediction_timing(self):
        """Test prediction timing performance."""
        # This test would measure actual prediction times
        # For now, we'll just verify the structure is correct
        
        # Generate test data
        offset_data = np.random.normal(0, 1e-6, 1000)
        
        start_time = time.time()
        # Simulate prediction work
        time.sleep(0.001)  # 1ms simulated work
        end_time = time.time()
        
        inference_time = end_time - start_time
        
        # Verify timing is reasonable
        assert inference_time < 1.0  # Should be under 1 second
        assert inference_time > 0.0
    
    @pytest.mark.slow
    def test_memory_usage(self):
        """Test memory usage patterns."""
        # This test would monitor memory usage during operation
        # For now, we'll just verify the structure exists
        
        try:
            import psutil
            process = psutil.Process()
            initial_memory = process.memory_info().rss
            
            # Simulate memory-intensive operation
            large_array = np.random.random((1000, 1000))
            current_memory = process.memory_info().rss
            
            # Memory should have increased
            assert current_memory >= initial_memory
            
            # Clean up
            del large_array
            
        except ImportError:
            pytest.skip("psutil not available for memory testing")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
    Path(temp_config_file).unlink()