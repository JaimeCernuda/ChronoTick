#!/usr/bin/env python3
"""
Tests for ChronoTick Inference Utilities

Unit tests for utility classes including data generation, metrics collection, and visualization.
"""

import pytest
import numpy as np
import time
import threading
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add the chronotick_inference package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chronotick_inference.utils import (
    ClockDataGenerator,
    SystemMetricsCollector,
    PredictionVisualizer,
    SystemMetrics,
    create_test_data,
    simulate_real_time_collection
)


class TestClockDataGenerator:
    """Test the clock data generator utility."""
    
    def test_generator_initialization(self):
        """Test generator initialization."""
        generator = ClockDataGenerator(base_frequency=2.4e9, seed=42)
        
        assert generator.base_frequency == 2.4e9
    
    def test_basic_offset_generation(self):
        """Test basic offset sequence generation."""
        generator = ClockDataGenerator(seed=42)
        
        offset_data = generator.generate_offset_sequence(
            duration=100,
            sampling_rate=1.0,
            drift_rate=1e-6,
            noise_level=1e-7
        )
        
        assert len(offset_data) == 100
        assert offset_data.dtype == np.float64
        assert np.all(np.isfinite(offset_data))
        
        # Check that drift is present (should be increasing trend)
        trend = np.polyfit(np.arange(len(offset_data)), offset_data, 1)[0]
        assert trend > 0  # Positive drift
    
    def test_offset_generation_parameters(self):
        """Test offset generation with different parameters."""
        generator = ClockDataGenerator(seed=123)
        
        # Test with oscillations
        offset_data = generator.generate_offset_sequence(
            duration=50,
            oscillation_period=10.0,
            oscillation_amplitude=1e-5
        )
        
        assert len(offset_data) == 50
        
        # Should have oscillatory component
        # Simple check: standard deviation should be reasonable
        assert np.std(offset_data) > 1e-6
    
    def test_system_correlation_generation(self):
        """Test offset generation correlated with system metrics."""
        generator = ClockDataGenerator(seed=456)
        
        duration = 60
        system_metrics = {
            'temperature': np.linspace(60, 80, duration),  # Increasing temperature
            'cpu_usage': np.random.uniform(20, 80, duration),
            'voltage': np.full(duration, 3.3)
        }
        
        correlations = {
            'temperature': 2e-6,  # 2μs per degree
            'cpu_usage': 1e-7     # 0.1μs per percent
        }
        
        offset_data = generator.generate_with_system_correlation(
            duration=duration,
            system_metrics=system_metrics,
            correlations=correlations
        )
        
        assert len(offset_data) == duration
        assert np.all(np.isfinite(offset_data))
        
        # With increasing temperature, offset should generally increase
        # (due to positive correlation)
        correlation_coef = np.corrcoef(system_metrics['temperature'], offset_data)[0, 1]
        assert correlation_coef > 0.1  # Should be positively correlated
    
    def test_realistic_scenarios(self):
        """Test realistic scenario generation."""
        generator = ClockDataGenerator(seed=789)
        
        scenarios = ["server_load", "thermal_cycle", "network_spike"]
        
        for scenario in scenarios:
            offset_data, metrics = generator.generate_realistic_scenario(
                scenario=scenario,
                duration=120
            )
            
            assert len(offset_data) == 120
            assert isinstance(metrics, dict)
            
            # Check required metrics
            required_metrics = ['cpu_usage', 'temperature', 'memory_usage', 'voltage']
            for metric in required_metrics:
                assert metric in metrics
                assert len(metrics[metric]) == 120
                assert np.all(np.isfinite(metrics[metric]))
            
            # Check metric ranges
            assert np.all(metrics['cpu_usage'] >= 0)
            assert np.all(metrics['cpu_usage'] <= 100)
            assert np.all(metrics['memory_usage'] >= 0)
            assert np.all(metrics['memory_usage'] <= 100)
            assert np.all(metrics['temperature'] > 30)  # Reasonable temperature
            assert np.all(metrics['temperature'] < 100)
    
    def test_invalid_scenario(self):
        """Test handling of invalid scenario names."""
        generator = ClockDataGenerator()
        
        with pytest.raises(ValueError, match="Unknown scenario"):
            generator.generate_realistic_scenario("invalid_scenario")
    
    def test_reproducibility(self):
        """Test that same seed produces same results."""
        seed = 999
        
        generator1 = ClockDataGenerator(seed=seed)
        data1 = generator1.generate_offset_sequence(duration=50)
        
        generator2 = ClockDataGenerator(seed=seed)
        data2 = generator2.generate_offset_sequence(duration=50)
        
        np.testing.assert_array_equal(data1, data2)


class TestSystemMetricsCollector:
    """Test the system metrics collector."""
    
    def test_collector_initialization(self):
        """Test collector initialization."""
        collector = SystemMetricsCollector(collection_interval=0.5)
        
        assert collector.collection_interval == 0.5
        assert len(collector.metrics_history) == 0
        assert collector.collection_thread is None
        assert not collector.stop_event.is_set()
    
    @patch('chronotick_inference.utils.psutil')
    def test_metrics_collection(self, mock_psutil):
        """Test metrics collection with mocked psutil."""
        # Setup mocks
        mock_psutil.cpu_percent.return_value = 45.2
        
        mock_memory = Mock()
        mock_memory.percent = 67.8
        mock_psutil.virtual_memory.return_value = mock_memory
        
        mock_psutil.sensors_temperatures.return_value = {
            'cpu_thermal': [Mock(current=72.5)]
        }
        
        mock_psutil.getloadavg.return_value = [1.2, 1.1, 1.0]
        
        mock_disk = Mock()
        mock_disk.read_bytes = 1000
        mock_disk.write_bytes = 500
        mock_psutil.disk_io_counters.return_value = mock_disk
        
        mock_net = Mock()
        mock_net.bytes_sent = 2000
        mock_net.bytes_recv = 3000
        mock_psutil.net_io_counters.return_value = mock_net
        
        # Test collection
        collector = SystemMetricsCollector()
        metrics = collector._collect_current_metrics()
        
        assert isinstance(metrics, SystemMetrics)
        assert metrics.cpu_usage == 45.2
        assert metrics.memory_usage == 67.8
        assert metrics.temperature == 72.5
        assert metrics.load_average == 1.2
        assert metrics.disk_io == 1500  # read + write
        assert metrics.network_io == 5000  # sent + recv
        assert metrics.timestamp > 0
    
    def test_collection_start_stop(self):
        """Test starting and stopping collection."""
        collector = SystemMetricsCollector(collection_interval=0.1)
        
        # Start collection
        collector.start_collection()
        assert collector.collection_thread is not None
        assert collector.collection_thread.is_alive()
        
        # Let it collect briefly
        time.sleep(0.2)
        
        # Stop collection
        collector.stop_collection()
        assert collector.stop_event.is_set()
        
        # Thread should stop
        time.sleep(0.1)
        assert not collector.collection_thread.is_alive()
    
    @patch('chronotick_inference.utils.psutil')
    def test_recent_metrics_retrieval(self, mock_psutil):
        """Test retrieval of recent metrics."""
        # Setup mock
        mock_psutil.cpu_percent.return_value = 50.0
        mock_memory = Mock()
        mock_memory.percent = 70.0
        mock_psutil.virtual_memory.return_value = mock_memory
        mock_psutil.sensors_temperatures.return_value = {}
        mock_psutil.getloadavg.side_effect = AttributeError()  # Not available
        mock_psutil.disk_io_counters.side_effect = AttributeError()
        mock_psutil.net_io_counters.side_effect = AttributeError()
        
        collector = SystemMetricsCollector()
        
        # Manually add some metrics
        current_time = time.time()
        for i in range(5):
            metrics = SystemMetrics(
                timestamp=current_time - (4-i),  # 4 seconds ago to now
                cpu_usage=50.0 + i,
                memory_usage=70.0 + i
            )
            collector.metrics_history.append(metrics)
        
        # Get recent metrics (last 3 seconds)
        recent = collector.get_recent_metrics(window_seconds=3)
        
        assert 'cpu_usage' in recent
        assert 'memory_usage' in recent
        assert len(recent['cpu_usage']) == 3  # Last 3 metrics
        assert len(recent['memory_usage']) == 3
        
        # Check values
        np.testing.assert_array_equal(recent['cpu_usage'], [52.0, 53.0, 54.0])
        np.testing.assert_array_equal(recent['memory_usage'], [72.0, 73.0, 74.0])
    
    def test_metrics_summary(self):
        """Test metrics summary generation."""
        collector = SystemMetricsCollector()
        
        # Test with no data
        summary = collector.get_metrics_summary()
        assert summary['status'] == 'no_data'
        
        # Add some metrics
        current_time = time.time()
        for i in range(10):
            metrics = SystemMetrics(
                timestamp=current_time - (9-i),
                cpu_usage=40.0 + i,
                memory_usage=60.0 + i
            )
            collector.metrics_history.append(metrics)
        
        # Get summary
        summary = collector.get_metrics_summary()
        
        assert summary['status'] == 'collecting'
        assert summary['total_samples'] == 10
        assert summary['cpu_usage']['current'] == 49.0  # Last value
        assert summary['cpu_usage']['avg'] > 40.0
        assert summary['memory_usage']['current'] == 69.0


class TestPredictionVisualizer:
    """Test the prediction visualizer."""
    
    def test_visualizer_initialization(self):
        """Test visualizer initialization."""
        visualizer = PredictionVisualizer()
        
        assert visualizer.plot_history == []
    
    def test_text_plot_creation(self):
        """Test text-based plot creation."""
        visualizer = PredictionVisualizer()
        
        # Create test data
        timestamps = np.arange(20)
        actual = np.sin(timestamps * 0.5) * 1e-5
        predicted = actual + np.random.normal(0, 1e-6, len(actual))
        
        # Create text plot
        plot_text = visualizer._create_text_plot(timestamps, actual, predicted)
        
        assert isinstance(plot_text, str)
        assert "Clock Offset Predictions" in plot_text
        assert "Legend: A = Actual, P = Predicted" in plot_text
        assert len(plot_text.split('\n')) > 10  # Should have multiple lines
    
    def test_text_plot_edge_cases(self):
        """Test text plot with edge cases."""
        visualizer = PredictionVisualizer()
        
        # Test with identical values
        timestamps = np.arange(5)
        identical_values = np.full(5, 1e-5)
        
        plot_text = visualizer._create_text_plot(timestamps, identical_values, identical_values)
        assert "no variation to plot" in plot_text
    
    @patch('chronotick_inference.utils.plt', create=True)
    def test_plot_with_matplotlib(self, mock_plt):
        """Test plotting with matplotlib available."""
        # Setup mock
        mock_figure = Mock()
        mock_plt.figure.return_value = mock_figure
        mock_plt.savefig.return_value = None
        
        visualizer = PredictionVisualizer()
        
        # Test data
        timestamps = np.arange(10)
        actual = np.random.normal(0, 1e-5, 10)
        predicted = actual + np.random.normal(0, 1e-6, 10)
        uncertainties = np.full(10, 2e-6)
        
        result = visualizer.plot_predictions(
            timestamps=timestamps,
            actual_offsets=actual,
            predictions=predicted,
            uncertainties=uncertainties
        )
        
        assert "Plot saved to" in result
        mock_plt.figure.assert_called_once()
        mock_plt.savefig.assert_called_once()
    
    def test_performance_report_creation(self):
        """Test performance report creation."""
        visualizer = PredictionVisualizer()
        
        # Create test prediction data
        predictions = [
            {'prediction': 1e-5, 'model_type': 'short_term'},
            {'prediction': 1.2e-5, 'model_type': 'short_term'},
            {'prediction': 0.9e-5, 'model_type': 'long_term'},
            {'prediction': 1.1e-5, 'model_type': 'fused'}
        ]
        
        actual_values = [1.05e-5, 1.15e-5, 0.95e-5, 1.08e-5]
        
        report = visualizer.create_performance_report(predictions, actual_values)
        
        assert isinstance(report, str)
        assert "Performance Report" in report
        assert "Total Predictions: 4" in report
        assert "Mean Absolute Error" in report
        assert "Root Mean Square Error" in report
        assert "Model Usage:" in report
        assert "short_term: 2 predictions" in report
    
    def test_performance_report_empty_data(self):
        """Test performance report with empty data."""
        visualizer = PredictionVisualizer()
        
        report = visualizer.create_performance_report([], [])
        assert "No data available" in report
    
    def test_performance_report_metrics_calculation(self):
        """Test accuracy of performance metrics calculation."""
        visualizer = PredictionVisualizer()
        
        # Known test case
        predictions = [{'prediction': 1e-5}, {'prediction': 2e-5}]
        actual_values = [1.1e-5, 1.9e-5]  # Errors: 0.1e-5, -0.1e-5
        
        report = visualizer.create_performance_report(predictions, actual_values)
        
        # MAE should be 0.1e-5 = 0.1 μs
        assert "0.100 μs" in report  # MAE
        
        # RMSE should be 0.1e-5 = 0.1 μs (since errors are symmetric)
        # Max error should be 0.1e-5 = 0.1 μs
        assert "Maximum Error: 0.100 μs" in report


class TestUtilityFunctions:
    """Test standalone utility functions."""
    
    def test_create_test_data(self):
        """Test create_test_data function."""
        offset_data, system_metrics = create_test_data(duration=60)
        
        assert len(offset_data) == 60
        assert isinstance(system_metrics, dict)
        
        # Check that we get expected metrics
        expected_metrics = ['cpu_usage', 'temperature', 'memory_usage', 'voltage']
        for metric in expected_metrics:
            assert metric in system_metrics
            assert len(system_metrics[metric]) == 60
    
    @patch('chronotick_inference.utils.time.sleep')
    def test_simulate_real_time_collection(self, mock_sleep):
        """Test simulate_real_time_collection function."""
        # Mock sleep to avoid actual delays
        mock_sleep.return_value = None
        
        collector = simulate_real_time_collection(duration=1)
        
        assert isinstance(collector, SystemMetricsCollector)
        assert collector.collection_thread is not None
        
        # Clean up
        collector.stop_collection()


class TestSystemMetricsDataClass:
    """Test the SystemMetrics dataclass."""
    
    def test_system_metrics_creation(self):
        """Test SystemMetrics creation with all fields."""
        timestamp = time.time()
        
        metrics = SystemMetrics(
            timestamp=timestamp,
            cpu_usage=45.2,
            memory_usage=67.8,
            temperature=72.5,
            voltage=3.31,
            frequency=2.4e9,
            disk_io=1500,
            network_io=5000,
            load_average=1.2
        )
        
        assert metrics.timestamp == timestamp
        assert metrics.cpu_usage == 45.2
        assert metrics.memory_usage == 67.8
        assert metrics.temperature == 72.5
        assert metrics.voltage == 3.31
        assert metrics.frequency == 2.4e9
        assert metrics.disk_io == 1500
        assert metrics.network_io == 5000
        assert metrics.load_average == 1.2
    
    def test_system_metrics_optional_fields(self):
        """Test SystemMetrics with only required fields."""
        timestamp = time.time()
        
        metrics = SystemMetrics(
            timestamp=timestamp,
            cpu_usage=50.0,
            memory_usage=70.0
        )
        
        assert metrics.timestamp == timestamp
        assert metrics.cpu_usage == 50.0
        assert metrics.memory_usage == 70.0
        assert metrics.temperature is None
        assert metrics.voltage is None
        assert metrics.frequency is None


class TestThreadSafety:
    """Test thread safety of utility classes."""
    
    def test_metrics_collector_thread_safety(self):
        """Test that metrics collector is thread-safe."""
        collector = SystemMetricsCollector(collection_interval=0.05)
        
        # Start collection
        collector.start_collection()
        
        # Access metrics from multiple threads
        results = []
        
        def read_metrics():
            try:
                for _ in range(5):
                    metrics = collector.get_recent_metrics(window_seconds=1)
                    results.append(len(metrics))
                    time.sleep(0.01)
            except Exception as e:
                results.append(f"error: {e}")
        
        # Start multiple reader threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=read_metrics)
            threads.append(thread)
            thread.start()
        
        # Wait for threads to complete
        for thread in threads:
            thread.join(timeout=2.0)
        
        # Stop collection
        collector.stop_collection()
        
        # Check that no errors occurred
        errors = [r for r in results if isinstance(r, str) and r.startswith('error')]
        assert len(errors) == 0, f"Thread safety errors: {errors}"


class TestErrorHandling:
    """Test error handling in utility classes."""
    
    def test_generator_invalid_parameters(self):
        """Test generator with invalid parameters."""
        generator = ClockDataGenerator()
        
        # Test with negative duration
        with pytest.raises((ValueError, TypeError)):
            generator.generate_offset_sequence(duration=-10)
    
    @patch('chronotick_inference.utils.psutil')
    def test_metrics_collector_psutil_errors(self, mock_psutil):
        """Test metrics collector when psutil operations fail."""
        # Setup mock to raise exceptions
        mock_psutil.cpu_percent.side_effect = OSError("Permission denied")
        mock_psutil.virtual_memory.side_effect = OSError("Access denied")
        mock_psutil.sensors_temperatures.side_effect = AttributeError("Not supported")
        
        collector = SystemMetricsCollector()
        
        # Should handle errors gracefully
        metrics = collector._collect_current_metrics()
        
        # Should still create metrics object with timestamp
        assert isinstance(metrics, SystemMetrics)
        assert metrics.timestamp > 0
        # Other fields might be None or default values
    
    def test_visualizer_matplotlib_import_error(self):
        """Test visualizer when matplotlib is not available."""
        visualizer = PredictionVisualizer()
        
        # Test that it falls back to text plot when matplotlib not available
        timestamps = np.arange(10)
        actual = np.random.normal(0, 1e-5, 10)
        predicted = actual + np.random.normal(0, 1e-6, 10)
        
        # This should not raise an error even if matplotlib is not available
        result = visualizer.plot_predictions(timestamps, actual, predicted)
        
        # Should return some kind of result (either plot path or text plot)
        assert isinstance(result, str)
        assert len(result) > 0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])