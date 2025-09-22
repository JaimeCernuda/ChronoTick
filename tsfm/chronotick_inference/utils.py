"""
ChronoTick Inference Layer Utilities

Utility classes for data generation, system metrics collection, and visualization.
"""

import time
import threading
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import psutil
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """Container for system metrics at a point in time."""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    temperature: Optional[float] = None
    voltage: Optional[float] = None
    frequency: Optional[float] = None
    disk_io: Optional[float] = None
    network_io: Optional[float] = None
    load_average: Optional[float] = None


class ClockDataGenerator:
    """
    Generates synthetic clock offset data for testing and simulation.
    
    This class creates realistic clock drift patterns including:
    - Linear drift (temperature, aging effects)
    - Periodic oscillations (crystal oscillator effects)
    - Random walk noise
    - Sudden jumps (thermal events)
    """
    
    def __init__(self, base_frequency: float = 1e9, seed: Optional[int] = None):
        """
        Initialize the clock data generator.
        
        Args:
            base_frequency: Base clock frequency in Hz
            seed: Random seed for reproducible data
        """
        self.base_frequency = base_frequency
        if seed is not None:
            np.random.seed(seed)
    
    def generate_offset_sequence(self,
                                duration: int,
                                sampling_rate: float = 1.0,
                                drift_rate: float = 1e-6,
                                noise_level: float = 1e-6,
                                oscillation_period: float = 300.0,
                                oscillation_amplitude: float = 5e-6,
                                jump_probability: float = 0.001,
                                jump_magnitude: float = 1e-4) -> np.ndarray:
        """
        Generate a sequence of clock offset measurements.
        
        Args:
            duration: Duration in seconds
            sampling_rate: Samples per second
            drift_rate: Linear drift rate (seconds per second)
            noise_level: Random noise standard deviation
            oscillation_period: Period of oscillations in seconds
            oscillation_amplitude: Amplitude of oscillations in seconds
            jump_probability: Probability of sudden jumps per sample
            jump_magnitude: Magnitude of sudden jumps in seconds
            
        Returns:
            Array of offset values in seconds
        """
        n_samples = int(duration * sampling_rate)
        t = np.arange(n_samples) / sampling_rate
        
        # Linear drift component
        linear_drift = drift_rate * t
        
        # Oscillatory component (crystal behavior)
        oscillations = oscillation_amplitude * np.sin(2 * np.pi * t / oscillation_period)
        
        # Random walk noise
        noise = np.cumsum(np.random.normal(0, noise_level, n_samples))
        
        # Sudden jumps (thermal events, etc.)
        jumps = np.zeros(n_samples)
        jump_times = np.random.random(n_samples) < jump_probability
        jumps[jump_times] = np.random.normal(0, jump_magnitude, np.sum(jump_times))
        cumulative_jumps = np.cumsum(jumps)
        
        # Combine all components
        offset = linear_drift + oscillations + noise + cumulative_jumps
        
        return offset.astype(np.float64)
    
    def generate_with_system_correlation(self,
                                       duration: int,
                                       system_metrics: Dict[str, np.ndarray],
                                       correlations: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        Generate clock offset data correlated with system metrics.
        
        Args:
            duration: Duration in seconds
            system_metrics: Dictionary of system metric arrays
            correlations: Correlation coefficients for each metric
            
        Returns:
            Array of offset values correlated with system metrics
        """
        if correlations is None:
            correlations = {
                'temperature': 2e-6,    # 2μs per degree
                'cpu_usage': 1e-7,      # 0.1μs per percent
                'voltage': -5e-5,       # -50μs per volt
                'frequency': -1e-15     # -1ns per Hz
            }
        
        # Start with base offset sequence
        base_offset = self.generate_offset_sequence(duration)
        
        # Add correlated components
        correlated_offset = base_offset.copy()
        
        for metric_name, metric_values in system_metrics.items():
            if metric_name in correlations:
                correlation = correlations[metric_name]
                # Normalize metric to zero mean
                normalized_metric = metric_values - np.mean(metric_values)
                # Add correlated component
                correlated_offset += correlation * normalized_metric
        
        return correlated_offset
    
    def generate_realistic_scenario(self, 
                                   scenario: str = "server_load",
                                   duration: int = 3600) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Generate realistic scenarios for testing.
        
        Args:
            scenario: Scenario type ('server_load', 'thermal_cycle', 'network_spike')
            duration: Duration in seconds
            
        Returns:
            Tuple of (offset_data, system_metrics)
        """
        t = np.arange(duration)
        
        if scenario == "server_load":
            # Simulates server under varying load
            cpu_usage = 30 + 40 * np.sin(2 * np.pi * t / 1800) + np.random.normal(0, 5, duration)
            cpu_usage = np.clip(cpu_usage, 0, 100)
            
            temperature = 65 + 0.2 * cpu_usage + np.random.normal(0, 1, duration)
            memory_usage = 40 + 20 * np.sin(2 * np.pi * t / 3600) + np.random.normal(0, 3, duration)
            voltage = 3.3 + np.random.normal(0, 0.05, duration)
            
        elif scenario == "thermal_cycle":
            # Simulates thermal cycling
            temperature = 50 + 20 * np.sin(2 * np.pi * t / 7200) + np.random.normal(0, 2, duration)
            cpu_usage = 25 + 15 * np.sin(2 * np.pi * t / 1800) + np.random.normal(0, 3, duration)
            memory_usage = 35 + np.random.normal(0, 2, duration)
            voltage = 3.3 - 0.01 * (temperature - 65) + np.random.normal(0, 0.02, duration)
            
        elif scenario == "network_spike":
            # Simulates network load spikes
            cpu_usage = np.full(duration, 20.0)
            # Add random spikes
            spike_times = np.random.random(duration) < 0.001
            cpu_usage[spike_times] += np.random.uniform(40, 80, np.sum(spike_times))
            cpu_usage = np.clip(cpu_usage, 0, 100)
            
            temperature = 60 + 0.15 * cpu_usage + np.random.normal(0, 1, duration)
            memory_usage = 30 + 0.3 * cpu_usage + np.random.normal(0, 2, duration)
            voltage = 3.3 + np.random.normal(0, 0.03, duration)
            
        else:
            raise ValueError(f"Unknown scenario: {scenario}")
        
        # Create system metrics
        system_metrics = {
            'cpu_usage': cpu_usage,
            'temperature': temperature,
            'memory_usage': np.clip(memory_usage, 0, 100),
            'voltage': voltage,
            'frequency': np.full(duration, 2.4e9) + np.random.normal(0, 1e7, duration),
            'disk_io': np.random.exponential(10, duration),
            'network_io': np.random.exponential(50, duration)
        }
        
        # Generate correlated offset data
        offset_data = self.generate_with_system_correlation(duration, system_metrics)
        
        return offset_data, system_metrics


class SystemMetricsCollector:
    """
    Collects real-time system metrics for use as covariates in clock prediction.
    
    This class provides a thread-safe way to collect system performance metrics
    that can influence clock behavior, such as CPU usage, temperature, and voltage.
    """
    
    def __init__(self, collection_interval: float = 1.0):
        """
        Initialize the metrics collector.
        
        Args:
            collection_interval: Interval between collections in seconds
        """
        self.collection_interval = collection_interval
        self.metrics_history: List[SystemMetrics] = []
        self.collection_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        
    def start_collection(self):
        """Start collecting metrics in a background thread."""
        if self.collection_thread and self.collection_thread.is_alive():
            logger.warning("Metrics collection already running")
            return
        
        self.stop_event.clear()
        self.collection_thread = threading.Thread(target=self._collection_loop)
        self.collection_thread.daemon = True
        self.collection_thread.start()
        logger.info("Started system metrics collection")
    
    def stop_collection(self):
        """Stop collecting metrics."""
        self.stop_event.set()
        if self.collection_thread:
            self.collection_thread.join(timeout=5.0)
        logger.info("Stopped system metrics collection")
    
    def _collection_loop(self):
        """Main collection loop running in background thread."""
        while not self.stop_event.is_set():
            try:
                metrics = self._collect_current_metrics()
                
                with self.lock:
                    self.metrics_history.append(metrics)
                    # Keep only recent history (last hour)
                    if len(self.metrics_history) > 3600:
                        self.metrics_history = self.metrics_history[-3600:]
                
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
            
            self.stop_event.wait(self.collection_interval)
    
    def _collect_current_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        timestamp = time.time()
        
        # CPU usage
        cpu_usage = psutil.cpu_percent(interval=0.1)
        
        # Memory usage
        memory_info = psutil.virtual_memory()
        memory_usage = memory_info.percent
        
        # Temperature (if available)
        temperature = None
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                # Get CPU temperature if available
                for name, entries in temps.items():
                    if 'cpu' in name.lower() or 'core' in name.lower():
                        if entries:
                            temperature = entries[0].current
                            break
        except (AttributeError, OSError):
            pass
        
        # Load average
        load_average = None
        try:
            load_average = psutil.getloadavg()[0]  # 1-minute load average
        except (AttributeError, OSError):
            pass
        
        # Disk I/O
        disk_io = None
        try:
            disk_stats = psutil.disk_io_counters()
            if disk_stats:
                disk_io = disk_stats.read_bytes + disk_stats.write_bytes
        except (AttributeError, OSError):
            pass
        
        # Network I/O
        network_io = None
        try:
            net_stats = psutil.net_io_counters()
            if net_stats:
                network_io = net_stats.bytes_sent + net_stats.bytes_recv
        except (AttributeError, OSError):
            pass
        
        return SystemMetrics(
            timestamp=timestamp,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            temperature=temperature,
            load_average=load_average,
            disk_io=disk_io,
            network_io=network_io
        )
    
    def get_recent_metrics(self, window_seconds: int = 300) -> Dict[str, np.ndarray]:
        """
        Get recent metrics as arrays suitable for model input.
        
        Args:
            window_seconds: Time window in seconds
            
        Returns:
            Dictionary of metric arrays
        """
        with self.lock:
            if not self.metrics_history:
                return {}
            
            cutoff_time = time.time() - window_seconds
            recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
            
            if not recent_metrics:
                return {}
            
            metrics_dict = {}
            
            # Extract each metric type
            metrics_dict['cpu_usage'] = np.array([m.cpu_usage for m in recent_metrics])
            metrics_dict['memory_usage'] = np.array([m.memory_usage for m in recent_metrics])
            
            # Optional metrics (may be None)
            if any(m.temperature is not None for m in recent_metrics):
                temps = [m.temperature if m.temperature is not None else 0.0 for m in recent_metrics]
                metrics_dict['temperature'] = np.array(temps)
            
            if any(m.load_average is not None for m in recent_metrics):
                loads = [m.load_average if m.load_average is not None else 0.0 for m in recent_metrics]
                metrics_dict['load_average'] = np.array(loads)
            
            return metrics_dict
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary statistics of collected metrics."""
        with self.lock:
            if not self.metrics_history:
                return {"status": "no_data"}
            
            recent_metrics = self.metrics_history[-60:]  # Last minute
            
            return {
                "status": "collecting",
                "total_samples": len(self.metrics_history),
                "recent_samples": len(recent_metrics),
                "cpu_usage": {
                    "current": recent_metrics[-1].cpu_usage,
                    "avg": np.mean([m.cpu_usage for m in recent_metrics]),
                    "max": np.max([m.cpu_usage for m in recent_metrics])
                },
                "memory_usage": {
                    "current": recent_metrics[-1].memory_usage,
                    "avg": np.mean([m.memory_usage for m in recent_metrics])
                },
                "collection_interval": self.collection_interval
            }


class PredictionVisualizer:
    """
    Visualizes prediction results and model performance.
    
    This class provides methods to create plots and visualizations
    of clock predictions, uncertainties, and model performance.
    """
    
    def __init__(self):
        """Initialize the visualizer."""
        self.plot_history = []
    
    def plot_predictions(self,
                        timestamps: np.ndarray,
                        actual_offsets: np.ndarray,
                        predictions: np.ndarray,
                        uncertainties: Optional[np.ndarray] = None,
                        title: str = "Clock Offset Predictions") -> str:
        """
        Create a plot of predictions vs actual values.
        
        Args:
            timestamps: Time points
            actual_offsets: Actual measured offsets
            predictions: Predicted offsets
            uncertainties: Prediction uncertainties (optional)
            title: Plot title
            
        Returns:
            String representation of the plot (text-based)
        """
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 6))
            
            # Plot actual vs predicted
            plt.plot(timestamps, actual_offsets * 1e6, 'b-', label='Actual', alpha=0.7)
            plt.plot(timestamps, predictions * 1e6, 'r-', label='Predicted', alpha=0.8)
            
            # Add uncertainty bands if available
            if uncertainties is not None:
                upper = (predictions + uncertainties) * 1e6
                lower = (predictions - uncertainties) * 1e6
                plt.fill_between(timestamps, lower, upper, alpha=0.3, color='red', label='Uncertainty')
            
            plt.xlabel('Time (seconds)')
            plt.ylabel('Clock Offset (microseconds)')
            plt.title(title)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save plot
            plot_path = f"prediction_plot_{int(time.time())}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return f"Plot saved to {plot_path}"
            
        except ImportError:
            # Fallback to text-based visualization
            return self._create_text_plot(timestamps, actual_offsets, predictions)
    
    def _create_text_plot(self,
                         timestamps: np.ndarray,
                         actual: np.ndarray,
                         predicted: np.ndarray,
                         width: int = 80,
                         height: int = 20) -> str:
        """Create a simple text-based plot."""
        # Convert to microseconds for better readability
        actual_us = actual * 1e6
        predicted_us = predicted * 1e6
        
        # Find data range
        all_values = np.concatenate([actual_us, predicted_us])
        min_val, max_val = np.min(all_values), np.max(all_values)
        
        if max_val == min_val:
            return "All values are identical - no variation to plot"
        
        # Create plot grid
        plot_lines = []
        
        # Header
        plot_lines.append(f"Clock Offset Predictions (μs)")
        plot_lines.append(f"Range: {min_val:.3f} to {max_val:.3f} μs")
        plot_lines.append("=" * width)
        
        # Sample points for plotting
        n_points = min(width - 10, len(actual_us))
        indices = np.linspace(0, len(actual_us) - 1, n_points).astype(int)
        
        for i in range(height):
            line = ""
            threshold = min_val + (max_val - min_val) * (height - i - 1) / height
            
            for j, idx in enumerate(indices):
                if abs(actual_us[idx] - threshold) < (max_val - min_val) / height:
                    line += "A"  # Actual
                elif abs(predicted_us[idx] - threshold) < (max_val - min_val) / height:
                    line += "P"  # Predicted
                else:
                    line += " "
            
            plot_lines.append(f"{threshold:6.3f} |{line}")
        
        plot_lines.append("=" * width)
        plot_lines.append("Legend: A = Actual, P = Predicted")
        
        return "\n".join(plot_lines)
    
    def create_performance_report(self,
                                 predictions: List[Dict],
                                 actual_values: List[float]) -> str:
        """
        Create a text-based performance report.
        
        Args:
            predictions: List of prediction dictionaries
            actual_values: List of actual offset values
            
        Returns:
            Formatted performance report
        """
        if not predictions or not actual_values:
            return "No data available for performance report"
        
        # Calculate metrics
        pred_values = [p.get('prediction', 0) for p in predictions]
        errors = np.array(actual_values) - np.array(pred_values)
        
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors ** 2))
        max_error = np.max(np.abs(errors))
        
        # Format report
        report = [
            "=== ChronoTick Prediction Performance Report ===",
            f"Total Predictions: {len(predictions)}",
            f"Mean Absolute Error: {mae * 1e6:.3f} μs",
            f"Root Mean Square Error: {rmse * 1e6:.3f} μs", 
            f"Maximum Error: {max_error * 1e6:.3f} μs",
            "",
            "Model Usage:",
        ]
        
        # Count model usage
        model_counts = {}
        for pred in predictions:
            model_type = pred.get('model_type', 'unknown')
            model_counts[model_type] = model_counts.get(model_type, 0) + 1
        
        for model, count in model_counts.items():
            percentage = (count / len(predictions)) * 100
            report.append(f"  {model}: {count} predictions ({percentage:.1f}%)")
        
        return "\n".join(report)


# Example usage and testing utilities
def create_test_data(duration: int = 3600) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Create test data for demonstration purposes."""
    generator = ClockDataGenerator(seed=42)
    return generator.generate_realistic_scenario("server_load", duration)


def simulate_real_time_collection(duration: int = 60) -> SystemMetricsCollector:
    """Simulate real-time metrics collection for testing."""
    collector = SystemMetricsCollector(collection_interval=1.0)
    collector.start_collection()
    
    # Let it collect for a short time
    time.sleep(min(duration, 5))  # Don't wait too long in tests
    
    return collector