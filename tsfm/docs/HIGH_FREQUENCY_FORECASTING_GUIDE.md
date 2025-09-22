# High-Frequency (1-Second) Time Series Forecasting with TSFM Factory

## Overview
This guide provides detailed instructions for using the TSFM Factory library for high-frequency time series forecasting, specifically optimized for 1-second sampling intervals. Based on comprehensive testing and analysis of all 5 foundation models.

## Quick Start for 1-Second Data

### Recommended Setup (Fast & Reliable)
```bash
# Install core models (no dependency conflicts)
uv sync --extra core-models

# Python setup
from tsfm import TSFMFactory
import numpy as np

# Create factory and load fastest model
factory = TSFMFactory()
model = factory.load_model('chronos', device='cpu')  # or 'cuda' if available

# Prepare 1-second data (e.g., CPU usage)
cpu_data = load_your_1_second_data()  # Shape: (n_seconds,)
context = cpu_data[-600:]  # Last 10 minutes (600 seconds)
horizon = 60  # Predict next 1 minute (60 seconds)

# Generate forecast
result = model.forecast(context, horizon)
predictions = result.predictions  # Shape: (60,)
```

## Model Selection for High-Frequency Data

### 🏆 **Chronos-Bolt** - BEST for Real-Time
- **Inference Speed**: 1.15 seconds for 60-step forecast
- **Memory Usage**: Low (optimized for 20x efficiency)
- **Max Horizon**: 96 steps (1.6 minutes ahead)
- **Quantiles**: ✅ Full uncertainty quantification
- **Best For**: Real-time monitoring, alerts, dashboards

```python
# Chronos-Bolt setup
model = factory.load_model('chronos', device='cpu')
result = model.forecast(context[-512:], horizon=60)  # Use last 8.5 minutes
print(f"Confidence intervals: {list(result.quantiles.keys())}")
```

### 🥈 **TTM** - GOOD for Multivariate Systems
- **Inference Speed**: Fast (MLP-mixer architecture)
- **Memory Usage**: Very low
- **Max Horizon**: 96 steps (fixed)
- **Multivariate**: ✅ True multivariate support
- **Best For**: Multiple metrics (CPU, memory, disk, network)

```python
# TTM multivariate setup (requires TTM environment)
# uv sync --extra ttm
multivariate_data = np.stack([
    cpu_usage[-600:],      # CPU %
    memory_usage[-600:],   # Memory %
    disk_io[-600:],        # Disk I/O
    network_io[-600:]      # Network I/O
])

mv_input = factory.create_multivariate_input(
    data=multivariate_data,
    variable_names=['cpu', 'memory', 'disk', 'network'],
    target_variables=['cpu', 'memory']  # Forecast CPU and memory
)

model = factory.load_model('ttm', device='cpu')
results = model.forecast_multivariate(mv_input, horizon=60)
```

### 🥉 **TimesFM 2.0** - GOOD for Long Context
- **Inference Speed**: Medium
- **Memory Usage**: High (500M parameters)
- **Max Horizon**: Flexible (no limit)
- **Context Length**: 2048 steps (34 minutes)
- **Best For**: Long-term patterns, flexible horizons

```python
# TimesFM for long context
model = factory.load_model('timesfm', device='cpu')
long_context = cpu_usage[-2048:]  # Use full 34 minutes
result = model.forecast(long_context, horizon=300)  # Predict 5 minutes ahead
```

## Environment Setup Guide

### Production Environment (Recommended)
```bash
# Core models only - no conflicts, maximum stability
uv sync --extra core-models

# Available models: Chronos-Bolt, TimesFM
# Recommended for production deployments
```

### Research Environment (TTM)
```bash
# TTM environment for multivariate capabilities  
uv sync --extra ttm

# Available models: TTM, Chronos-Bolt, TimesFM
# Best for infrastructure monitoring with multiple metrics
```

### Advanced Research Environment (Toto)
```bash
# Toto environment for observability specialization
uv sync --extra toto

# Available models: Toto, Chronos-Bolt, TimesFM
# Best for long-term infrastructure monitoring (5+ minutes ahead)
```

### Experimental Environment (Time-MoE)
```bash
# Time-MoE environment (research only)
uv sync --extra time-moe

# Available models: Time-MoE, Chronos-Bolt, TimesFM
# Best for pattern analysis, not real-time due to conflicts
```

## Frequency Configuration for 1-Second Data

### Method 1: Explicit Frequency (Recommended)
```python
# Configure 1-second frequency
freq_info = factory.create_frequency_info(
    freq_str='S',        # Second frequency
    freq_value=9,        # Numeric code for seconds
    is_regular=True,     # Regular 1-second intervals
    detected_freq='S'
)

# Use with models that support frequency info
result = model.forecast(context, horizon, freq=9)
```

### Method 2: Auto-Detection
```python
# Let models handle frequency internally
result = model.forecast(context, horizon)  # freq=0 (auto)
```

### Method 3: Timestamp-Based Detection
```python
# If you have timestamps
timestamps = pd.date_range('2024-01-01', periods=len(data), freq='1S')
freq_info = factory.detect_frequency(timestamps.tolist())
print(f"Detected: {freq_info}")
```

## Data Preprocessing for High-Frequency Time Series

### Outlier Handling (Important for 1-Second Data)
```python
from tsfm.datasets.preprocessing import remove_outliers, fill_missing_values

# Clean noisy 1-second data
clean_data = remove_outliers(raw_data, method='iqr', threshold=1.5)
clean_data = fill_missing_values(clean_data, method='interpolate')
```

### Sliding Window Preparation
```python
from tsfm.datasets.preprocessing import create_sliding_windows

# Create multiple forecast samples for evaluation
contexts, targets = create_sliding_windows(
    data=clean_data,
    context_length=600,    # 10 minutes context
    horizon_length=60,     # 1 minute prediction
    stride=30,            # Every 30 seconds
    min_samples=100
)

print(f"Created {len(contexts)} forecast samples")
```

### Normalization (Model-Specific)
```python
from tsfm.datasets.preprocessing import normalize_data

# Some models benefit from pre-normalization
normalized_data, stats = normalize_data(data, method='zscore')

# Models like TTM handle normalization internally
# TimesFM and Chronos work well with raw data
```

## Performance Optimization

### Memory Management
```python
# Use context manager for automatic cleanup
with TSFMFactory() as factory:
    model = factory.load_model("chronos", device="cpu")
    
    # Process in batches to manage memory
    batch_size = 100
    for i in range(0, len(time_series_list), batch_size):
        batch = time_series_list[i:i+batch_size]
        results = process_batch(model, batch)
    
    # Model automatically unloaded on context exit
```

### Context Length Optimization
```python
# Optimize context length for different models
context_lengths = {
    'chronos': 512,    # Good balance of speed/accuracy
    'ttm': 300,        # Faster, requires min 90
    'timesfm': 1024,   # Can handle up to 2048
    'toto': 600,       # Good for observability data
    'time_moe': 2048   # Can handle up to 4096
}

context = data[-context_lengths[model_name]:]
```

### Batch Processing for Multiple Series
```python
def process_multiple_series(factory, series_dict, horizon=60):
    """Process multiple time series efficiently."""
    results = {}
    
    # Group by model type for efficiency
    model = factory.load_model('chronos', device='cpu')
    
    for series_name, data in series_dict.items():
        # Optimize context length
        context = data[-512:] if len(data) > 512 else data
        
        # Skip if insufficient data
        if len(context) < 60:  # Need at least 1 minute
            continue
            
        try:
            result = model.forecast(context, horizon)
            results[series_name] = result
        except Exception as e:
            print(f"Error forecasting {series_name}: {e}")
    
    factory.unload_model('chronos')
    return results
```

## Real-World Use Cases

### 1. Infrastructure Monitoring Dashboard
```python
class InfrastructureMonitor:
    def __init__(self):
        self.factory = TSFMFactory()
        self.model = self.factory.load_model('chronos', device='cpu')
    
    def predict_cpu_spike(self, cpu_history):
        """Predict if CPU will spike in next minute."""
        context = cpu_history[-600:]  # Last 10 minutes
        result = self.model.forecast(context, 60)
        
        # Check if predicted values exceed threshold
        spike_threshold = 80.0
        spike_probability = np.mean(result.predictions > spike_threshold)
        
        return {
            'predictions': result.predictions,
            'spike_probability': spike_probability,
            'confidence_intervals': result.quantiles
        }
    
    def multivariate_health_check(self, metrics_dict):
        """Check overall system health."""
        # Switch to TTM for multivariate
        self.factory.unload_model('chronos')
        ttm_model = self.factory.load_model('ttm', device='cpu')
        
        # Prepare multivariate input
        data = np.stack([
            metrics_dict['cpu'][-300:],
            metrics_dict['memory'][-300:], 
            metrics_dict['disk'][-300:]
        ])
        
        mv_input = self.factory.create_multivariate_input(
            data=data,
            variable_names=['cpu', 'memory', 'disk']
        )
        
        results = ttm_model.forecast_multivariate(mv_input, 60)
        return results
```

### 2. Trading/Financial Monitoring
```python
def financial_prediction(factory, price_data, volume_data):
    """Predict price movements with volume context."""
    # Use covariates for price prediction
    covariates_input = factory.create_covariates_input(
        target=price_data[-600:],  # Price history
        covariates={'volume': volume_data[-600:]},  # Volume context
        future_covariates=None
    )
    
    model = factory.load_model('timesfm', device='cpu')
    result = model.forecast_with_covariates(covariates_input, 60)
    
    return {
        'price_predictions': result.predictions,
        'metadata': result.metadata
    }
```

### 3. IoT Sensor Networks
```python
def iot_anomaly_detection(factory, sensor_readings):
    """Detect anomalies in IoT sensor readings."""
    from tsfm.datasets.preprocessing import detect_anomalies
    
    # Preprocess sensor data
    clean_data = remove_outliers(sensor_readings, method='iqr')
    
    # Generate normal predictions
    model = factory.load_model('chronos', device='cpu')
    result = model.forecast(clean_data[-300:], 30)
    
    # Compare with actual future readings (if available)
    predictions = result.predictions
    actual = sensor_readings[-30:]  # Actual last 30 seconds
    
    # Calculate anomaly score
    mae = np.mean(np.abs(actual - predictions))
    threshold = np.std(clean_data) * 2
    
    return {
        'anomaly_detected': mae > threshold,
        'anomaly_score': mae,
        'threshold': threshold,
        'predictions': predictions
    }
```

## Performance Benchmarks (1-Second Data)

### Inference Speed Comparison
| Model | Context Length | Horizon | Inference Time | Memory Usage |
|-------|---------------|---------|----------------|--------------|
| Chronos-Bolt | 512 | 60 | 1.15s | Low |
| TTM | 300 | 60 | 0.8s | Very Low |
| TimesFM 2.0 | 1024 | 60 | 2.3s | High |
| Toto | 600 | 60 | 3.5s | High |
| Time-MoE | 2048 | 60 | 8.2s | Very High |

### Accuracy Characteristics
- **Chronos-Bolt**: Best for short-term predictions (1-5 minutes)
- **TTM**: Best for multivariate patterns
- **TimesFM 2.0**: Best for long-term trends (5+ minutes)
- **Toto**: Best for infrastructure/observability data
- **Time-MoE**: Best for complex pattern analysis

## GPU vs CPU Usage

### CPU Deployment (Recommended for Production)
```python
# Most stable and widely compatible
model = factory.load_model('chronos', device='cpu')

# Benefits:
# - No CUDA dependencies
# - Consistent performance
# - Easy deployment
# - Memory efficient
```

### GPU Deployment (For High Throughput)
```python
# Check GPU availability first
import torch
if torch.cuda.is_available():
    model = factory.load_model('chronos', device='cuda')
else:
    model = factory.load_model('chronos', device='cpu')

# Benefits:
# - Faster inference for large models
# - Better for batch processing
# - Parallel processing of multiple series
```

### Device Selection Strategy
```python
def select_optimal_device(model_name):
    """Select optimal device based on model and system."""
    gpu_priority_models = ['timesfm', 'toto', 'time_moe']
    cpu_efficient_models = ['chronos', 'ttm']
    
    if model_name in cpu_efficient_models:
        return 'cpu'  # These models are CPU-optimized
    elif model_name in gpu_priority_models and torch.cuda.is_available():
        return 'cuda'  # Large models benefit from GPU
    else:
        return 'cpu'  # Fallback to CPU
```

## Troubleshooting High-Frequency Forecasting

### Common Issues and Solutions

#### 1. "Context too short" Error
```python
# Problem: TTM requires minimum 90 data points
# Solution: Check data length before forecasting
if len(context) < 90:
    # Pad with synthetic data or skip forecast
    padded_context = np.pad(context, (90-len(context), 0), mode='edge')
```

#### 2. Memory Issues with Large Context
```python
# Problem: Running out of memory with long time series
# Solution: Limit context length
max_context_lengths = {
    'chronos': 512,
    'ttm': 300, 
    'timesfm': 1024,
    'toto': 600,
    'time_moe': 2048
}

context = data[-max_context_lengths[model_name]:]
```

#### 3. Slow Inference Speed
```python
# Problem: Predictions taking too long
# Solution: Optimize model selection and context
if real_time_requirement:
    model_name = 'chronos'  # Fastest
    context_length = 256    # Shorter context
else:
    model_name = 'timesfm'  # More accurate
    context_length = 1024   # Longer context
```

#### 4. Poor Accuracy on Noisy Data
```python
# Problem: 1-second data too noisy
# Solution: Preprocess data
from tsfm.datasets.preprocessing import remove_outliers

clean_data = remove_outliers(raw_data, method='iqr', threshold=1.5)
# Consider smoothing for very noisy data
smoothed = np.convolve(clean_data, np.ones(5)/5, mode='same')
```

## Best Practices Summary

1. **Model Selection**: Use Chronos-Bolt for real-time, TTM for multivariate
2. **Context Length**: 5-10 minutes (300-600 seconds) for good balance
3. **Horizon**: Keep under 5 minutes (300 seconds) for accuracy
4. **Preprocessing**: Clean outliers, handle missing values
5. **Environment**: Use core-models for production stability
6. **Device**: CPU for most cases, GPU for large models/batches
7. **Memory**: Use context managers, unload models after use
8. **Error Handling**: Always validate input lengths and handle exceptions

This guide provides a comprehensive foundation for implementing high-frequency time series forecasting with the TSFM Factory library.