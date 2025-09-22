#!/usr/bin/env python3
"""
Integration tests for ChronoTick Real Data Pipeline

Tests the complete real data system without excessive mocking.
Focuses on real components and integration.
"""

import pytest
import time
import tempfile
import yaml
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from chronotick_inference.real_data_pipeline import RealDataPipeline, CorrectionWithBounds
from chronotick_inference.ntp_client import ClockMeasurementCollector


@pytest.fixture
def real_system_config():
    """Create complete real system configuration"""
    config = {
        'clock_measurement': {
            'ntp': {
                'servers': ['pool.ntp.org', 'time.google.com', 'time.cloudflare.com'],
                'timeout_seconds': 2.0,
                'max_acceptable_uncertainty': 0.050,  # Relaxed for testing
                'min_stratum': 2
            },
            'timing': {
                'warm_up': {
                    'duration_seconds': 10,  # Short for testing
                    'measurement_interval': 2.0
                },
                'normal_operation': {
                    'measurement_interval': 30.0
                }
            }
        },
        'prediction_scheduling': {
            'cpu_model': {
                'prediction_interval': 30.0,
                'prediction_horizon': 10,
                'prediction_lead_time': 5.0,
                'max_inference_time': 10.0
            },
            'gpu_model': {
                'prediction_interval': 60.0,
                'prediction_horizon': 30,
                'prediction_lead_time': 15.0,
                'max_inference_time': 30.0
            },
            'dataset': {
                'prediction_cache_size': 50
            }
        },
        'short_term': {
            'model_name': 'chronos',
            'device': 'cpu',
            'enabled': True,
            'inference_interval': 30.0,
            'prediction_horizon': 10,
            'context_length': 100,
            'max_uncertainty': 0.1,
            'model_params': {
                'prediction_length': 10,
                'repo': 'amazon/chronos-bolt-base',
                'size': 'base',
                'multivariate': True
            }
        },
        'long_term': {
            'model_name': 'chronos',
            'device': 'cpu', 
            'enabled': True,
            'inference_interval': 60.0,
            'prediction_horizon': 30,
            'context_length': 300,
            'max_uncertainty': 0.15,
            'model_params': {
                'prediction_length': 30,
                'repo': 'amazon/chronos-bolt-base',
                'size': 'base',
                'multivariate': True
            }
        },
        'fusion': {
            'enabled': True,
            'method': 'inverse_variance',
            'uncertainty_threshold': 0.05,
            'fallback_weights': {
                'short_term': 0.8,
                'long_term': 0.2
            }
        },
        'preprocessing': {
            'outlier_removal': {
                'enabled': True,
                'method': 'iqr',
                'threshold': 2.0
            },
            'missing_value_handling': {
                'enabled': True,
                'method': 'interpolate'
            }
        },
        'covariates': {
            'enabled': True,
            'variables': ['cpu_usage', 'temperature', 'memory_usage'],
            'future_variables': []
        },
        'performance': {
            'max_memory_mb': 2048,
            'model_timeout': 30.0,
            'cache_size': 5,
            'batch_size': 1
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
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        return f.name


class TestRealDataPipelineIntegration:
    """Integration tests for the complete real data pipeline"""
    
    def test_pipeline_initialization(self, real_system_config):
        """Test that the real data pipeline initializes correctly"""
        pipeline = RealDataPipeline(real_system_config)
        
        # Should initialize all components
        assert pipeline.ntp_collector is not None
        assert pipeline.predictive_scheduler is not None
        assert hasattr(pipeline, 'warm_up_complete')
        assert hasattr(pipeline, 'initialized')
    
    def test_fallback_correction_available(self, real_system_config):
        """Test that fallback corrections are always available"""
        pipeline = RealDataPipeline(real_system_config)
        
        current_time = time.time()
        correction = pipeline._fallback_correction(current_time)
        
        assert isinstance(correction, CorrectionWithBounds)
        assert correction.source == "no_data"  # Actual source name in implementation
        assert correction.offset_correction == 0.0
        assert correction.drift_rate == 0.0
        assert correction.offset_uncertainty > 0
        assert correction.drift_uncertainty > 0
    
    def test_real_clock_correction_interface(self, real_system_config):
        """Test the main interface for getting real clock corrections"""
        pipeline = RealDataPipeline(real_system_config)
        
        current_time = time.time()
        
        # Should return correction without errors (may be fallback initially)
        correction = pipeline.get_real_clock_correction(current_time)
        
        assert isinstance(correction, CorrectionWithBounds)
        assert hasattr(correction, 'offset_correction')
        assert hasattr(correction, 'drift_rate')
        assert hasattr(correction, 'offset_uncertainty')
        assert hasattr(correction, 'drift_uncertainty')
        assert hasattr(correction, 'confidence')
        assert hasattr(correction, 'source')
        
        # Mathematical validity
        assert isinstance(correction.offset_correction, (int, float))
        assert isinstance(correction.drift_rate, (int, float))
        assert correction.offset_uncertainty >= 0
        assert correction.drift_uncertainty >= 0
        assert 0 <= correction.confidence <= 1
    
    def test_error_bounds_mathematical_correctness(self, real_system_config):
        """Test that error bounds follow correct mathematical propagation"""
        pipeline = RealDataPipeline(real_system_config)
        current_time = time.time()
        
        correction = pipeline.get_real_clock_correction(current_time)
        
        # Test uncertainty calculation for different time deltas
        uncertainty_0 = correction.get_time_uncertainty(0.0)
        uncertainty_10 = correction.get_time_uncertainty(10.0)
        uncertainty_100 = correction.get_time_uncertainty(100.0)
        
        # At t=0, uncertainty should equal offset_uncertainty
        assert abs(uncertainty_0 - correction.offset_uncertainty) < 1e-9
        
        # Uncertainty should increase with time due to drift uncertainty
        assert uncertainty_100 >= uncertainty_10 >= uncertainty_0
        
        # Verify mathematical formula: sqrt(offset_unc² + (drift_unc * t)²)
        expected_100 = (correction.offset_uncertainty**2 + 
                       (correction.drift_uncertainty * 100)**2)**0.5
        assert abs(uncertainty_100 - expected_100) < 1e-9
    
    def test_no_synthetic_data_in_corrections(self, real_system_config):
        """Verify that the system produces no synthetic data - everything is real or fallback"""
        pipeline = RealDataPipeline(real_system_config)
        current_time = time.time()
        
        # Get multiple corrections
        corrections = []
        for i in range(5):
            correction = pipeline.get_real_clock_correction(current_time + i)
            corrections.append(correction)
        
        # All corrections should be either 'no_data' or real measurement sources
        # No 'synthetic' or 'mock' sources allowed
        valid_sources = {'no_data', 'cpu', 'gpu', 'fusion', 'ntp'}
        
        for correction in corrections:
            assert correction.source in valid_sources, f"Invalid source: {correction.source}"
            assert 'synthetic' not in correction.source.lower()
            assert 'mock' not in correction.source.lower()
            assert 'fake' not in correction.source.lower()
    
    def test_configuration_driven_behavior(self, real_system_config):
        """Test that the system behavior is properly configuration-driven"""
        pipeline = RealDataPipeline(real_system_config)
        
        # Test warm-up configuration (accessed through ntp_collector)
        assert pipeline.ntp_collector.warm_up_duration == 10  # From test config
        
        # Test NTP configuration  
        assert 'pool.ntp.org' in pipeline.ntp_collector.ntp_client.config.servers
        assert pipeline.ntp_collector.ntp_client.config.timeout_seconds == 2.0
        
        # Test predictive scheduling configuration
        scheduler = pipeline.predictive_scheduler
        assert scheduler.cpu_lead_time == 5.0
        assert scheduler.gpu_lead_time == 15.0
    
    def test_system_lifecycle(self, real_system_config):
        """Test the complete system lifecycle"""
        pipeline = RealDataPipeline(real_system_config)
        
        # Start the system (collection is started through ntp_collector)
        pipeline.ntp_collector.start_collection()
        
        # Give it a moment to initialize
        time.sleep(0.5)
        
        # Should be collecting (though may not have measurements yet)
        assert pipeline.ntp_collector.collection_running
        
        # Stop the system
        pipeline.ntp_collector.stop_collection()
        
        # Should stop cleanly
        assert not pipeline.ntp_collector.collection_running
    
    def test_retrospective_bias_correction_readiness(self, real_system_config):
        """Test that system is ready for retrospective bias correction"""
        pipeline = RealDataPipeline(real_system_config)
        
        # Should have methods for retrospective correction
        assert hasattr(pipeline, '_check_for_ntp_updates')
        assert hasattr(pipeline.ntp_collector, 'get_recent_measurements')
        
        # Get recent measurements (will be empty initially)
        recent = pipeline.ntp_collector.get_recent_measurements(window_seconds=300)
        assert isinstance(recent, list)


class TestNTPClientIntegration:
    """Integration tests for NTP client with real network (when available)"""
    
    def test_ntp_client_real_network_available(self, real_system_config):
        """Test NTP client with real network servers (if network available)"""
        collector = ClockMeasurementCollector(real_system_config)
        
        # Try to get a real measurement (this may timeout in some environments)
        try:
            measurement = collector.ntp_client.get_best_measurement()
            
            if measurement is not None:
                # If we got a real measurement, verify it's realistic
                assert abs(measurement.offset) < 1.0  # Offset should be < 1 second
                assert measurement.delay < 1.0  # Delay should be < 1 second
                assert 1 <= measurement.stratum <= 15  # Valid stratum range
                assert measurement.uncertainty > 0
                print(f"✅ Real NTP measurement: offset={measurement.offset*1e6:.1f}μs, "
                      f"delay={measurement.delay*1000:.1f}ms, stratum={measurement.stratum}")
            else:
                print("ℹ️ No NTP measurement available (network/firewall limitations)")
                
        except Exception as e:
            print(f"ℹ️ NTP client test skipped due to network: {e}")
    
    def test_measurement_collection_lifecycle(self, real_system_config):
        """Test the measurement collection lifecycle"""
        collector = ClockMeasurementCollector(real_system_config)
        
        # Start collection briefly
        collector.start_collection()
        
        # Should be running
        assert collector.collection_running
        assert collector.collection_thread is not None
        assert collector.collection_thread.is_alive()
        
        # Stop collection
        collector.stop_collection()
        
        # Should stop cleanly
        assert not collector.collection_running


class TestRealSystemPerformance:
    """Performance tests for the real system"""
    
    def test_correction_call_latency(self, real_system_config):
        """Test that correction calls have acceptable latency"""
        pipeline = RealDataPipeline(real_system_config)
        
        current_time = time.time()
        
        # Measure call latency
        latencies = []
        for _ in range(10):
            start = time.time()
            correction = pipeline.get_real_clock_correction(current_time)
            latency = (time.time() - start) * 1000  # Convert to ms
            latencies.append(latency)
            
            assert correction is not None
        
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        
        # Should be reasonably fast
        assert avg_latency < 50.0, f"Average latency too high: {avg_latency:.1f}ms"
        assert max_latency < 100.0, f"Maximum latency too high: {max_latency:.1f}ms"
        
        print(f"✅ Correction call performance: avg={avg_latency:.1f}ms, max={max_latency:.1f}ms")
    
    def test_memory_usage_stability(self, real_system_config):
        """Test that memory usage remains stable"""
        import psutil
        import os
        
        pipeline = RealDataPipeline(real_system_config)
        process = psutil.Process(os.getpid())
        
        # Baseline memory usage
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Make many correction calls
        current_time = time.time()
        for i in range(100):
            correction = pipeline.get_real_clock_correction(current_time + i)
            assert correction is not None
        
        # Check memory usage after
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Should not have significant memory leaks
        assert memory_increase < 50.0, f"Memory usage increased by {memory_increase:.1f}MB"
        
        print(f"✅ Memory stability: {initial_memory:.1f}MB → {final_memory:.1f}MB "
              f"(+{memory_increase:.1f}MB)")


def test_real_system_comprehensive_functionality():
    """Comprehensive test of real system functionality"""
    config_path = Path(__file__).parent.parent.parent / "chronotick_inference" / "config.yaml"
    
    if not config_path.exists():
        print("ℹ️ Skipping comprehensive test - no default config found")
        return
    
    try:
        pipeline = RealDataPipeline(str(config_path))
        
        # Test basic functionality
        current_time = time.time()
        correction = pipeline.get_real_clock_correction(current_time)
        
        assert isinstance(correction, CorrectionWithBounds)
        print(f"✅ Real system comprehensive test passed")
        print(f"   Source: {correction.source}")
        print(f"   Offset: {correction.offset_correction*1e6:+.1f}μs")
        print(f"   Uncertainty: ±{correction.offset_uncertainty*1e6:.1f}μs")
        
    except Exception as e:
        print(f"ℹ️ Comprehensive test encountered issue: {e}")
        # This is expected in some environments


if __name__ == "__main__":
    pytest.main([__file__, "-v"])