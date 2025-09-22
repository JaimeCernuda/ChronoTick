# TSFM Factory: Comprehensive Usage Guide

## Library Overview

The TSFM (Time Series Foundation Models) Factory is a unified Python library providing access to 5 state-of-the-art time series forecasting foundation models through a consistent interface. This guide covers everything from basic setup to advanced usage patterns.

## Table of Contents
1. [Installation & Environment Setup](#installation--environment-setup)
2. [Basic Usage Patterns](#basic-usage-patterns)
3. [Model-Specific Usage](#model-specific-usage)
4. [Advanced Features](#advanced-features)
5. [Data Handling](#data-handling)
6. [Performance Optimization](#performance-optimization)
7. [Error Handling & Debugging](#error-handling--debugging)
8. [Production Deployment](#production-deployment)

## Installation & Environment Setup

### Quick Start Installation
```bash
# Clone the repository
git clone <repository-url>
cd tsfm

# Install core models (recommended for most users)
uv sync --extra core-models

# Available models: Chronos-Bolt, TimesFM
# No dependency conflicts, maximum stability
```

### Environment-Specific Installations

#### TTM Environment (Multivariate Support)
```bash
uv sync --extra ttm
# Models: TTM, Chronos-Bolt, TimesFM
# transformers==4.38.0 (exact version required)
```

#### Toto Environment (Observability Data)
```bash
uv sync --extra toto  
# Models: Toto, Chronos-Bolt, TimesFM
# transformers>=4.52.0 (latest features)
```

#### Time-MoE Environment (Research)
```bash
uv sync --extra time-moe
# Models: Time-MoE, Chronos-Bolt, TimesFM
# transformers==4.40.1 (specific compatibility)
```

#### Development Environment
```bash
uv sync --extra dev --extra test
# Includes all development tools: black, ruff, mypy, pytest
```

### Dependency Conflict Resolution
The library uses **mutually exclusive** environments due to transformers version conflicts:

| Environment | Transformers Version | Available Models | Best For |
|------------|---------------------|------------------|----------|
| core-models | 4.52.1 | Chronos, TimesFM | Production |
| ttm | 4.38.0 | TTM, Chronos, TimesFM | Multivariate |
| toto | ≥4.52.0 | Toto, Chronos, TimesFM | Observability |
| time-moe | 4.40.1 | Time-MoE, Chronos, TimesFM | Research |

## Basic Usage Patterns

### 1. Simple Forecasting
```python
import numpy as np
from tsfm import TSFMFactory

# Create sample time series data
data = np.random.randn(1000).cumsum()

# Initialize factory and load model
factory = TSFMFactory()
model = factory.load_model('chronos', device='cpu')

# Generate forecast
context = data[-100:]  # Last 100 points
horizon = 24          # Predict 24 steps ahead
result = model.forecast(context, horizon)

print(f"Predictions: {result.predictions}")
print(f"Has quantiles: {result.quantiles is not None}")
print(f"Metadata: {result.metadata}")

# Clean up
factory.unload_model('chronos')
```

### 2. Context Manager Usage (Recommended)
```python
# Automatic resource management
with TSFMFactory() as factory:
    model = factory.load_model('timesfm', device='cpu')
    result = model.forecast(data[-200:], 48)
    # Model automatically unloaded on exit
```

### 3. Model Comparison
```python
def compare_models(data, context_length=100, horizon=24):
    """Compare different models on the same data."""
    results = {}
    
    with TSFMFactory() as factory:
        # Test available models
        for model_name in factory.list_models():
            try:
                model = factory.load_model(model_name, device='cpu')
                result = model.forecast(data[-context_length:], horizon)
                results[model_name] = {
                    'predictions': result.predictions,
                    'has_quantiles': result.quantiles is not None,
                    'inference_time': result.metadata.get('inference_time', 'N/A')
                }
                factory.unload_model(model_name)
            except Exception as e:
                results[model_name] = f"Error: {e}"
    
    return results
```

## Model-Specific Usage

### TimesFM 2.0 (Google) - Long Context & Flexible Horizons
```python
# TimesFM excels at long context and flexible prediction lengths
with TSFMFactory() as factory:
    model = factory.load_model('timesfm', device='cpu')
    
    # Use long context (up to 2048 points)
    long_context = data[-1500:]
    
    # Flexible horizon lengths
    short_forecast = model.forecast(long_context, 12)
    medium_forecast = model.forecast(long_context, 96) 
    long_forecast = model.forecast(long_context, 200)
    
    # With frequency information
    result = model.forecast(long_context, 48, freq=9)  # 9 = seconds
```

### TTM (IBM) - True Multivariate Support
```python
# TTM Environment: uv sync --extra ttm
with TSFMFactory() as factory:
    model = factory.load_model('ttm', device='cpu')
    
    # Multivariate time series
    multivariate_data = np.random.randn(5, 500)  # 5 variables, 500 timesteps
    
    mv_input = factory.create_multivariate_input(
        data=multivariate_data,
        variable_names=['cpu', 'memory', 'disk', 'network', 'temp'],
        target_variables=['cpu', 'memory']  # Only forecast these
    )
    
    results = model.forecast_multivariate(mv_input, horizon=24)
    
    for var_name, forecast in results.items():
        print(f"{var_name}: {forecast.predictions[:5]}...")
```

### Chronos-Bolt (Amazon) - Ultra-Fast Inference
```python
# Chronos-Bolt: Best for real-time applications
import time

with TSFMFactory() as factory:
    model = factory.load_model('chronos', device='cpu')
    
    start_time = time.time()
    result = model.forecast(data[-512:], 60)
    inference_time = time.time() - start_time
    
    print(f"Inference time: {inference_time:.3f} seconds")
    print(f"Quantile levels: {list(result.quantiles.keys())}")
    
    # Access different confidence levels
    lower_bound = result.quantiles['0.1']  # 10th percentile
    median = result.quantiles['0.5']       # Median prediction
    upper_bound = result.quantiles['0.9']  # 90th percentile
```

### Toto (Datadog) - Observability Specialization
```python
# Toto Environment: uv sync --extra toto
# Specialized for infrastructure/observability time series
with TSFMFactory() as factory:
    model = factory.load_model('toto', device='cpu')
    
    # High-cardinality covariates (common in observability)
    covariates_input = factory.create_covariates_input(
        target=cpu_usage_data,
        covariates={
            'memory_usage': memory_data,
            'disk_io': disk_data,
            'network_io': network_data
        },
        categorical_covariates={
            'service': ['web', 'api', 'db'],
            'region': ['us-east', 'us-west', 'eu']
        }
    )
    
    # Longer prediction horizons (Toto supports up to 336 steps)
    result = model.forecast_with_covariates(covariates_input, horizon=300)
```

### Time-MoE (Mixture of Experts) - Pattern Specialization
```python
# Time-MoE Environment: uv sync --extra time-moe
# Best for complex pattern analysis, not real-time
with TSFMFactory() as factory:
    model = factory.load_model('time_moe', device='cpu')
    
    # Very long context (up to 4096 points)
    very_long_context = data[-3000:]
    
    # Mixture-of-experts automatically routes to specialized experts
    result = model.forecast(very_long_context, horizon=96)
    
    # Check expert specialization info in metadata
    print(f"MoE metadata: {result.metadata.get('mixture_of_experts')}")
```

## Advanced Features

### 1. Enhanced Data Structures

#### Multivariate Input
```python
# Complex multivariate setup
multivariate_data = np.stack([
    temperature_data,
    humidity_data, 
    pressure_data,
    wind_speed_data
])

mv_input = factory.create_multivariate_input(
    data=multivariate_data,
    variable_names=['temp', 'humidity', 'pressure', 'wind'],
    target_variables=['temp', 'pressure'],  # Only forecast these
    metadata={
        'location': 'Station_A',
        'measurement_interval': '1_hour',
        'quality_score': 0.95
    }
)
```

#### Covariates Input  
```python
# Complex covariates setup
covariates_input = factory.create_covariates_input(
    target=sales_data,
    covariates={
        'price': price_data,
        'promotion': promotion_data,
        'weather': weather_data
    },
    future_covariates={
        'planned_promotions': future_promotions,
        'weather_forecast': weather_forecast
    },
    categorical_covariates={
        'day_of_week': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
        'season': ['Spring', 'Summer', 'Fall', 'Winter']
    },
    metadata={
        'store_id': 'Store_123',
        'category': 'Electronics'
    }
)
```

#### Frequency Information
```python
# Comprehensive frequency setup
freq_info = factory.create_frequency_info(
    freq_str='H',           # Hourly frequency
    freq_value=2,           # Numeric code for hourly
    is_regular=True,        # Regular intervals
    detected_freq='H'       # Auto-detected frequency
)

# Use frequency in forecasting
result = model.forecast_with_covariates(
    covariates_input, 
    horizon=24, 
    frequency=freq_info
)
```

### 2. Model Capability Queries
```python
# Discover model capabilities
factory = TSFMFactory()

# List models by capability
print("Multivariate models:", factory.get_multivariate_models())
print("Covariates models:", factory.get_covariates_models())

# Get detailed model information
for model_name in factory.list_models():
    info = factory.get_model_info(model_name)
    capabilities = info['capabilities']
    
    print(f"\n{model_name.upper()}:")
    print(f"  Multivariate: {capabilities['multivariate_support']}")
    print(f"  Covariates: {capabilities['covariates_support']}")
    print(f"  Direct Multistep: {capabilities['direct_multistep']}")
    print(f"  MoE: {capabilities['mixture_of_experts']}")
```

### 3. Enhanced Features Summary
```python
# Get comprehensive feature overview
features = factory.get_enhanced_features_summary()
print(f"Total models: {features['total_models']}")
print(f"Multivariate capable: {features['multivariate_models']}")
print(f"Enhanced capabilities: {features['enhanced_capabilities']}")
```

## Data Handling

### 1. Data Loading and Preprocessing
```python
from tsfm.datasets.loader import DatasetLoader, create_synthetic_data
from tsfm.datasets.preprocessing import (
    normalize_data, remove_outliers, fill_missing_values,
    create_sliding_windows, detect_anomalies
)

# Load data
loader = DatasetLoader()
available_datasets = loader.list_available_datasets()
print(f"Available datasets: {available_datasets}")

# Or create synthetic data
synthetic_data = create_synthetic_data(
    length=2000,
    pattern='mixed',  # 'linear', 'seasonal', 'mixed', 'ett', 'random'
    noise_level=0.1,
    seed=42
)

# Preprocess data
clean_data = remove_outliers(synthetic_data, method='iqr', threshold=1.5)
filled_data = fill_missing_values(clean_data, method='interpolate')
normalized_data, norm_stats = normalize_data(filled_data, method='zscore')

# Create training windows
contexts, targets = create_sliding_windows(
    data=normalized_data,
    context_length=100,
    horizon_length=24,
    stride=1,
    min_samples=50
)

print(f"Created {len(contexts)} training samples")
```

### 2. ETT Dataset Example
```python
from tsfm.datasets.loader import load_ett_data

# Load ETT (Electricity Transforming Temperature) data
ett_data = load_ett_data(variant='ETTh1', column='OT')
print(f"ETT data shape: {ett_data.shape}")

# Use for forecasting
with TSFMFactory() as factory:
    model = factory.load_model('timesfm', device='cpu')
    result = model.forecast(ett_data[-500:], 96)
```

### 3. Anomaly Detection
```python
# Detect anomalies in time series
anomalies = detect_anomalies(
    data=time_series_data,
    window_size=50,
    threshold=3.0
)

print(f"Found {np.sum(anomalies)} anomalies")
anomaly_indices = np.where(anomalies)[0]
print(f"Anomaly indices: {anomaly_indices[:10]}...")
```

## Performance Optimization

### 1. Memory Management
```python
import gc

class OptimizedForecaster:
    def __init__(self, model_name='chronos'):
        self.factory = TSFMFactory()
        self.model_name = model_name
        self.model = None
    
    def __enter__(self):
        self.model = self.factory.load_model(self.model_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.model:
            self.factory.unload_model(self.model_name)
        gc.collect()  # Force garbage collection
    
    def batch_forecast(self, data_list, horizon=24):
        """Process multiple time series efficiently."""
        results = []
        for data in data_list:
            try:
                result = self.model.forecast(data, horizon)
                results.append(result)
            except Exception as e:
                results.append(f"Error: {e}")
        return results

# Usage
with OptimizedForecaster('chronos') as forecaster:
    results = forecaster.batch_forecast([data1, data2, data3])
```

### 2. Model Selection Strategy
```python
def select_optimal_model(data_characteristics):
    """Select best model based on data characteristics."""
    length = data_characteristics['length']
    frequency = data_characteristics['frequency']
    multivariate = data_characteristics['multivariate']
    real_time = data_characteristics['real_time_requirement']
    
    if real_time and not multivariate:
        return 'chronos'  # Fastest inference
    elif multivariate:
        return 'ttm'      # True multivariate support
    elif length > 1000:
        return 'timesfm'  # Long context handling
    elif frequency == 'irregular':
        return 'timesfm'  # Flexible frequency handling
    else:
        return 'chronos'  # Default fast option

# Example usage
characteristics = {
    'length': 500,
    'frequency': 'regular',
    'multivariate': False,
    'real_time_requirement': True
}

optimal_model = select_optimal_model(characteristics)
print(f"Recommended model: {optimal_model}")
```

### 3. Context Length Optimization
```python
# Model-specific optimal context lengths
OPTIMAL_CONTEXTS = {
    'chronos': 512,    # Balance of speed and accuracy
    'ttm': 300,        # Fast inference, requires min 90
    'timesfm': 1024,   # Can handle up to 2048
    'toto': 600,       # Good for observability patterns
    'time_moe': 2048   # Can handle up to 4096
}

def optimize_context(data, model_name):
    """Optimize context length for model."""
    optimal_length = OPTIMAL_CONTEXTS.get(model_name, 512)
    return data[-optimal_length:] if len(data) > optimal_length else data
```

## Error Handling & Debugging

### 1. Common Error Patterns
```python
def robust_forecast(factory, model_name, data, horizon, **kwargs):
    """Robust forecasting with comprehensive error handling."""
    try:
        # Validate inputs
        if len(data) < 10:
            raise ValueError(f"Data too short: {len(data)} points")
        
        if horizon <= 0:
            raise ValueError(f"Invalid horizon: {horizon}")
        
        # Check model-specific requirements
        if model_name == 'ttm' and len(data) < 90:
            print(f"Warning: TTM requires 90+ points, padding data")
            data = np.pad(data, (90 - len(data), 0), mode='edge')
        
        # Load model with error handling
        try:
            model = factory.load_model(model_name, device='cpu')
        except RuntimeError as e:
            print(f"Model loading failed: {e}")
            print(f"Trying CPU fallback...")
            model = factory.load_model(model_name, device='cpu')
        
        # Optimize context length
        context = optimize_context(data, model_name)
        
        # Generate forecast
        result = model.forecast(context, horizon, **kwargs)
        
        # Validate output
        if result.predictions is None or len(result.predictions) == 0:
            raise RuntimeError("Model returned empty predictions")
        
        return result
        
    except Exception as e:
        print(f"Forecasting failed: {e}")
        return None
    finally:
        # Cleanup
        try:
            factory.unload_model(model_name)
        except:
            pass
```

### 2. Health Checking
```python
def system_health_check(factory):
    """Comprehensive system health check."""
    health_report = factory.health_check()
    
    print("=== TSFM Factory Health Report ===")
    print(f"Factory Status: {health_report['factory_status']}")
    print(f"Registered Models: {health_report['registered_models']}")
    print(f"Loaded Models: {health_report['loaded_models']}")
    
    # Check individual model health
    for model_name in health_report['registered_models']:
        try:
            model = factory.create_model(model_name, device='cpu')
            model_health = model.health_check()
            print(f"\n{model_name.upper()}:")
            print(f"  Status: {model_health['status']}")
            print(f"  Dependencies: {model_health.get('dependency_status', 'unknown')}")
            if 'dependency_error' in model_health:
                print(f"  Error: {model_health['dependency_error']}")
        except Exception as e:
            print(f"\n{model_name.upper()}: ERROR - {e}")
```

### 3. Debug Mode
```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('tsfm')

# Debug model loading
with TSFMFactory() as factory:
    logger.info("Loading model for debugging...")
    model = factory.load_model('chronos', device='cpu')
    
    logger.info("Generating test forecast...")
    test_data = np.random.randn(100)
    result = model.forecast(test_data, 10)
    
    logger.info(f"Forecast successful: {result.predictions.shape}")
```

## Production Deployment

### 1. Production-Ready Configuration
```python
# production_config.py
import os
from pathlib import Path

class ProductionConfig:
    # Environment
    ENVIRONMENT = os.getenv('TSFM_ENV', 'core-models')
    
    # Model settings
    DEFAULT_MODEL = os.getenv('TSFM_DEFAULT_MODEL', 'chronos')
    DEFAULT_DEVICE = os.getenv('TSFM_DEVICE', 'cpu')
    
    # Performance settings
    MAX_CONTEXT_LENGTH = int(os.getenv('TSFM_MAX_CONTEXT', '512'))
    MAX_HORIZON_LENGTH = int(os.getenv('TSFM_MAX_HORIZON', '96'))
    
    # Logging
    LOG_LEVEL = os.getenv('TSFM_LOG_LEVEL', 'INFO')
    
    # Data directories
    DATA_DIR = Path(os.getenv('TSFM_DATA_DIR', './data'))
    MODEL_CACHE_DIR = Path(os.getenv('TSFM_CACHE_DIR', './cache'))

# production_forecaster.py
class ProductionForecaster:
    def __init__(self, config=None):
        self.config = config or ProductionConfig()
        self.factory = TSFMFactory()
        self.models = {}
        
        # Pre-load commonly used models
        self._warm_up_models()
    
    def _warm_up_models(self):
        """Pre-load models for faster response."""
        try:
            self.models[self.config.DEFAULT_MODEL] = self.factory.load_model(
                self.config.DEFAULT_MODEL, 
                device=self.config.DEFAULT_DEVICE
            )
        except Exception as e:
            print(f"Warning: Could not load default model: {e}")
    
    def forecast(self, data, horizon=None, model_name=None):
        """Production forecast with validation and fallbacks."""
        # Input validation
        horizon = horizon or 24
        model_name = model_name or self.config.DEFAULT_MODEL
        
        if len(data) > self.config.MAX_CONTEXT_LENGTH:
            data = data[-self.config.MAX_CONTEXT_LENGTH:]
        
        if horizon > self.config.MAX_HORIZON_LENGTH:
            horizon = self.config.MAX_HORIZON_LENGTH
        
        # Get or load model
        if model_name not in self.models:
            self.models[model_name] = self.factory.load_model(
                model_name, device=self.config.DEFAULT_DEVICE
            )
        
        model = self.models[model_name]
        
        # Generate forecast
        result = model.forecast(data, horizon)
        
        # Add production metadata
        result.metadata.update({
            'production_version': '1.0',
            'environment': self.config.ENVIRONMENT,
            'model_version': model_name,
            'timestamp': pd.Timestamp.now().isoformat()
        })
        
        return result
    
    def health_check(self):
        """Production health check endpoint."""
        return {
            'status': 'healthy',
            'loaded_models': list(self.models.keys()),
            'config': {
                'environment': self.config.ENVIRONMENT,
                'default_model': self.config.DEFAULT_MODEL,
                'device': self.config.DEFAULT_DEVICE
            }
        }
    
    def cleanup(self):
        """Cleanup resources."""
        for model_name in list(self.models.keys()):
            self.factory.unload_model(model_name)
        self.models.clear()
```

### 2. API Integration Example
```python
from flask import Flask, jsonify, request
import numpy as np

app = Flask(__name__)
forecaster = ProductionForecaster()

@app.route('/forecast', methods=['POST'])
def forecast_endpoint():
    try:
        data = request.json
        
        # Validate input
        if 'data' not in data:
            return jsonify({'error': 'Missing data field'}), 400
        
        time_series = np.array(data['data'])
        horizon = data.get('horizon', 24)
        model_name = data.get('model', None)
        
        # Generate forecast
        result = forecaster.forecast(time_series, horizon, model_name)
        
        # Return response
        response = {
            'predictions': result.predictions.tolist(),
            'horizon': horizon,
            'model': model_name,
            'metadata': result.metadata
        }
        
        if result.quantiles:
            response['quantiles'] = {
                k: v.tolist() for k, v in result.quantiles.items()
            }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_endpoint():
    return jsonify(forecaster.health_check())

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
```

### 3. Docker Deployment
```dockerfile
# Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install UV
RUN pip install uv

# Copy requirements
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --extra core-models

# Copy application
COPY . .

# Expose port
EXPOSE 5000

# Run application
CMD ["uv", "run", "python", "production_api.py"]
```

### 4. Monitoring and Metrics
```python
from tsfm.utils.metrics import MetricsCalculator
import time

class ProductionMonitor:
    def __init__(self):
        self.metrics_calc = MetricsCalculator()
        self.performance_log = []
    
    def monitor_forecast(self, data, horizon, model_name):
        """Monitor forecast performance."""
        start_time = time.time()
        
        # Generate forecast
        forecaster = ProductionForecaster()
        result = forecaster.forecast(data, horizon, model_name)
        
        inference_time = time.time() - start_time
        
        # Log performance
        self.performance_log.append({
            'timestamp': pd.Timestamp.now(),
            'model': model_name,
            'data_length': len(data),
            'horizon': horizon,
            'inference_time': inference_time,
            'memory_usage': self._get_memory_usage()
        })
        
        return result
    
    def _get_memory_usage(self):
        """Get current memory usage."""
        try:
            import psutil
            return psutil.Process().memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            return None
    
    def get_performance_report(self):
        """Generate performance report."""
        if not self.performance_log:
            return {'message': 'No performance data available'}
        
        df = pd.DataFrame(self.performance_log)
        
        return {
            'total_forecasts': len(df),
            'average_inference_time': df['inference_time'].mean(),
            'model_usage': df['model'].value_counts().to_dict(),
            'memory_usage': {
                'average': df['memory_usage'].mean() if 'memory_usage' in df else None,
                'peak': df['memory_usage'].max() if 'memory_usage' in df else None
            }
        }
```

This comprehensive guide covers all aspects of using the TSFM Factory library from basic usage to advanced production deployment patterns.