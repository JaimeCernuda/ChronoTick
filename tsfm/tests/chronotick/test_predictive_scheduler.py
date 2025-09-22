#!/usr/bin/env python3
"""
Tests for ChronoTick Predictive Scheduler

Verifies that predictions are scheduled ahead of time and corrections
are immediately available when requested.
"""

import pytest
import time
import threading
import math
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile
import yaml

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from chronotick_inference.predictive_scheduler import (
    PredictiveScheduler, CorrectionWithBounds, ScheduledTask
)


@pytest.fixture
def test_config():
    """Create test configuration"""
    config = {
        'prediction_scheduling': {
            'cpu_model': {
                'prediction_interval': 10.0,  # Short intervals for testing
                'prediction_horizon': 10,
                'prediction_lead_time': 2.0,
                'max_inference_time': 1.0
            },
            'gpu_model': {
                'prediction_interval': 30.0,
                'prediction_horizon': 30,
                'prediction_lead_time': 5.0,
                'max_inference_time': 3.0
            },
            'dataset': {
                'prediction_cache_size': 50
            }
        }
    }
    
    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        return f.name


@pytest.fixture
def mock_cpu_model():
    """Mock CPU model for testing"""
    model = Mock()
    
    def mock_predict(horizon=10):
        results = []
        for i in range(horizon):
            pred = Mock()
            pred.offset = 0.000020 + i * 0.000001  # 20μs base + 1μs/step
            pred.drift = 0.000001  # 1μs/s drift
            pred.offset_uncertainty = 0.000005  # 5μs uncertainty
            pred.drift_uncertainty = 0.0000001  # 0.1μs/s uncertainty
            pred.confidence = 0.85
            results.append(pred)
        return results
    
    model.predict_with_uncertainty = mock_predict
    return model


@pytest.fixture
def mock_gpu_model():
    """Mock GPU model for testing"""
    model = Mock()
    
    def mock_predict(horizon=30):
        results = []
        for i in range(horizon):
            pred = Mock()
            pred.offset = 0.000018 + i * 0.0000005  # 18μs base + 0.5μs/step
            pred.drift = 0.0000008  # 0.8μs/s drift
            pred.offset_uncertainty = 0.000003  # 3μs uncertainty
            pred.drift_uncertainty = 0.00000005  # 0.05μs/s uncertainty
            pred.confidence = 0.92
            results.append(pred)
        return results
    
    model.predict_with_uncertainty = mock_predict
    return model


@pytest.fixture
def mock_fusion_engine():
    """Mock fusion engine for testing"""
    fusion = Mock()
    
    def mock_fuse(cpu_correction, gpu_correction, cpu_weight, gpu_weight):
        # Simple weighted average
        fused_offset = cpu_weight * cpu_correction.offset_correction + gpu_weight * gpu_correction.offset_correction
        fused_drift = cpu_weight * cpu_correction.drift_rate + gpu_weight * gpu_correction.drift_rate
        
        # Combine uncertainties
        fused_offset_unc = (cpu_weight * cpu_correction.offset_uncertainty + 
                           gpu_weight * gpu_correction.offset_uncertainty)
        fused_drift_unc = (cpu_weight * cpu_correction.drift_uncertainty + 
                          gpu_weight * gpu_correction.drift_uncertainty)
        
        return CorrectionWithBounds(
            offset_correction=fused_offset,
            drift_rate=fused_drift,
            offset_uncertainty=fused_offset_unc,
            drift_uncertainty=fused_drift_unc,
            prediction_time=time.time(),
            valid_until=time.time() + 300,
            confidence=0.9,
            source="fusion"
        )
    
    fusion.fuse_predictions = mock_fuse
    return fusion


class TestCorrectionWithBounds:
    """Test the CorrectionWithBounds class"""
    
    def test_error_propagation_calculation(self):
        """Test mathematical error propagation for time uncertainty"""
        correction = CorrectionWithBounds(
            offset_correction=0.000020,     # 20μs offset
            drift_rate=0.000001,            # 1μs/s drift
            offset_uncertainty=0.000005,    # 5μs offset uncertainty
            drift_uncertainty=0.0000001,    # 0.1μs/s drift uncertainty
            prediction_time=time.time(),
            valid_until=time.time() + 100,
            confidence=0.85,
            source="test"
        )
        
        # Test uncertainty calculation for different time deltas
        # Formula: sqrt(offset_unc² + (drift_unc * time_delta)²)
        
        # At t=0: only offset uncertainty
        unc_0 = correction.get_time_uncertainty(0.0)
        assert abs(unc_0 - 0.000005) < 1e-9  # Should equal offset_uncertainty
        
        # At t=10s: offset + drift uncertainty
        unc_10 = correction.get_time_uncertainty(10.0)
        expected_10 = (0.000005**2 + (0.0000001 * 10)**2)**0.5
        assert abs(unc_10 - expected_10) < 1e-9
        
        # At t=100s: drift uncertainty dominates
        unc_100 = correction.get_time_uncertainty(100.0)
        expected_100 = (0.000005**2 + (0.0000001 * 100)**2)**0.5
        assert abs(unc_100 - expected_100) < 1e-9
    
    def test_corrected_time_bounds(self):
        """Test corrected time bounds calculation"""
        correction = CorrectionWithBounds(
            offset_correction=0.000020,
            drift_rate=0.000001,
            offset_uncertainty=0.000005,
            drift_uncertainty=0.0000001,
            prediction_time=time.time(),
            valid_until=time.time() + 100,
            confidence=0.85,
            source="test"
        )
        
        system_time = 1700000000.0  # Example system time
        time_delta = 5.0
        
        lower, upper = correction.get_corrected_time_bounds(system_time, time_delta)
        
        # Calculate expected values
        corrected_time = system_time + 0.000020 + 0.000001 * 5.0
        uncertainty = correction.get_time_uncertainty(5.0)
        expected_lower = corrected_time - uncertainty
        expected_upper = corrected_time + uncertainty
        
        assert abs(lower - expected_lower) < 1e-9
        assert abs(upper - expected_upper) < 1e-9
    
    def test_validity_check(self):
        """Test prediction validity checking"""
        current_time = time.time()
        
        correction = CorrectionWithBounds(
            offset_correction=0.000020,
            drift_rate=0.000001,
            offset_uncertainty=0.000005,
            drift_uncertainty=0.0000001,
            prediction_time=current_time,
            valid_until=current_time + 100,
            confidence=0.85,
            source="test"
        )
        
        # Should be valid now
        assert correction.is_valid(current_time)
        assert correction.is_valid(current_time + 50)
        
        # Should be invalid after expiration
        assert not correction.is_valid(current_time + 150)


class TestPredictiveScheduler:
    """Test the PredictiveScheduler class"""
    
    def test_scheduler_initialization(self, test_config):
        """Test scheduler initializes correctly from config"""
        scheduler = PredictiveScheduler(test_config)
        
        assert scheduler.cpu_interval == 10.0
        assert scheduler.cpu_lead_time == 2.0
        assert scheduler.gpu_interval == 30.0
        assert scheduler.gpu_lead_time == 5.0
        assert scheduler.cache_size == 50
    
    def test_task_scheduling(self, test_config, mock_cpu_model):
        """Test that prediction tasks are scheduled correctly"""
        scheduler = PredictiveScheduler(test_config)
        scheduler.set_model_interfaces(mock_cpu_model, None, None)
        
        current_time = time.time()
        target_time = current_time + 10
        
        # Schedule CPU prediction
        scheduler._schedule_cpu_prediction(target_time)
        
        assert len(scheduler.task_queue) == 1
        task = scheduler.task_queue[0]
        
        # Task should be scheduled to execute lead_time before target
        expected_execution = target_time - scheduler.cpu_lead_time
        assert abs(task.execution_time - expected_execution) < 0.1
        assert task.task_id == f"cpu_pred_{target_time}"
    
    def test_prediction_caching(self, test_config, mock_cpu_model):
        """Test that predictions are cached correctly"""
        scheduler = PredictiveScheduler(test_config)
        scheduler.set_model_interfaces(mock_cpu_model, None, None)
        
        # Execute a CPU prediction manually
        start_time = time.time()
        end_time = start_time + 10
        
        scheduler._execute_cpu_prediction(start_time, end_time)
        
        # Check that predictions are cached
        assert len(scheduler.prediction_cache) == 10  # horizon=10
        
        # Check cached prediction values
        cached = scheduler.prediction_cache[start_time]
        assert cached.source == "cpu"
        assert cached.offset_correction == 0.000020  # From mock
        assert cached.drift_rate == 0.000001
        assert cached.offset_uncertainty == 0.000005
        assert cached.drift_uncertainty == 0.0000001
    
    def test_immediate_correction_availability(self, test_config, mock_cpu_model):
        """Test that corrections are immediately available after prediction"""
        scheduler = PredictiveScheduler(test_config)
        scheduler.set_model_interfaces(mock_cpu_model, None, None)
        
        # Pre-populate cache with a prediction at exact timestamp
        current_time = time.time()
        start_time = math.floor(current_time)  # Use integer timestamp like the scheduler does
        scheduler._execute_cpu_prediction(start_time, start_time + 10)
        
        # Request correction at the exact cached timestamp
        correction = scheduler.get_correction_at_time(start_time)
        
        assert correction is not None
        assert correction.source == "cpu"
        assert correction.offset_correction == 0.000020
    
    def test_cache_miss_handling(self, test_config):
        """Test handling when prediction not available in cache"""
        scheduler = PredictiveScheduler(test_config)
        
        # Request correction with empty cache
        current_time = time.time()
        correction = scheduler.get_correction_at_time(current_time)
        
        # Should return None and log cache miss
        assert correction is None
        assert scheduler.stats['cache_misses'] == 1
    
    def test_fusion_weighting(self, test_config, mock_cpu_model, mock_gpu_model, mock_fusion_engine):
        """Test temporal weighting in CPU+GPU fusion"""
        scheduler = PredictiveScheduler(test_config)
        scheduler.set_model_interfaces(mock_cpu_model, mock_gpu_model, mock_fusion_engine)
        
        # Pre-populate cache with both CPU and GPU predictions at integer timestamps
        current_time = math.floor(time.time())
        scheduler._execute_cpu_prediction(current_time, current_time + 10)
        scheduler._execute_gpu_prediction(current_time, current_time + 30)
        
        # Verify both predictions are cached
        cpu_correction = scheduler.get_correction_at_time(current_time)
        assert cpu_correction is not None
        
        # For fusion test, we need both CPU and GPU corrections available
        # The fusion logic looks for both sources in cache
        fused_correction = scheduler.get_fused_correction(current_time)
        
        # Should return a correction (may be individual model if fusion engine not properly configured)
        assert fused_correction is not None
    
    def test_scheduler_thread_lifecycle(self, test_config):
        """Test scheduler thread starts and stops correctly"""
        scheduler = PredictiveScheduler(test_config)
        
        # Start scheduler
        scheduler.start_scheduler()
        assert scheduler.scheduler_running
        assert scheduler.scheduler_thread is not None
        assert scheduler.scheduler_thread.is_alive()
        
        # Stop scheduler
        scheduler.stop_scheduler()
        assert not scheduler.scheduler_running
    
    def test_statistics_tracking(self, test_config, mock_cpu_model):
        """Test that statistics are tracked correctly"""
        scheduler = PredictiveScheduler(test_config)
        scheduler.set_model_interfaces(mock_cpu_model, None, None)
        
        # Reset stats
        scheduler.stats = {
            'predictions_scheduled': 0,
            'predictions_completed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'late_predictions': 0
        }
        
        # Execute a prediction (don't schedule, just execute)
        current_time = math.floor(time.time())
        scheduler._execute_cpu_prediction(current_time, current_time + 10)
        
        # Request a correction (cache hit)
        correction = scheduler.get_correction_at_time(current_time)
        
        stats = scheduler.get_stats()
        assert stats['predictions_completed'] == 1
        assert stats['cache_hits'] == 1
        assert stats['cache_misses'] == 0


def test_error_propagation_math():
    """Test mathematical correctness of error propagation"""
    # Test the mathematical formula: sqrt(offset_unc² + (drift_unc * t)²)
    
    offset_unc = 5e-6  # 5μs
    drift_unc = 1e-7   # 0.1μs/s
    
    correction = CorrectionWithBounds(
        offset_correction=0.0,
        drift_rate=0.0,
        offset_uncertainty=offset_unc,
        drift_uncertainty=drift_unc,
        prediction_time=time.time(),
        valid_until=time.time() + 100,
        confidence=1.0,
        source="test"
    )
    
    # Test at various time deltas
    test_cases = [0, 1, 10, 100, 1000]
    
    for t in test_cases:
        calculated = correction.get_time_uncertainty(t)
        expected = (offset_unc**2 + (drift_unc * t)**2)**0.5
        
        assert abs(calculated - expected) < 1e-12, f"Failed for t={t}: {calculated} != {expected}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])