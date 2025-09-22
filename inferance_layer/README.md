# ChronoTick Inference Layer

This module provides the forecasting engine for the ChronoTick system, implementing both short-term and long-term models with optional fusion capabilities using the TSFM Factory library.

## Overview

The ChronoTick Inference Layer is designed to predict clock drift patterns using state-of-the-art time series foundation models. It supports:

- **Short-term forecasting**: High-frequency, low-latency predictions for immediate clock correction
- **Long-term forecasting**: Lower-frequency, higher-accuracy predictions for stable drift estimation  
- **Model fusion**: Combines predictions using inverse-variance weighting for optimal accuracy
- **Covariates support**: Incorporates system metrics (CPU, temperature, voltage) as exogenous variables

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Short-term    │    │    Long-term     │    │     Fusion      │
│     Model       │───▶│      Model       │───▶│    Engine       │
│ (CPU, 1-5sec)   │    │ (CPU/GPU, 5min)  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TSFM Factory Interface                       │
│  ┌───────────┐ ┌──────────┐ ┌──────────────┐ ┌──────────────┐  │
│  │ Chronos   │ │   TTM    │ │   TimesFM    │ │     Toto     │  │
│  │  -Bolt    │ │          │ │     2.0      │ │              │  │
│  └───────────┘ └──────────┘ └──────────────┘ └──────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Configuration

The system is configured via `config.yaml`:

### Model Selection
```yaml
short_term:
  model_name: "chronos"  # Fast, CPU-optimized for real-time
  device: "cpu"
  prediction_horizon: 5  # seconds

long_term:
  model_name: "timesfm"  # Accurate, supports longer context
  device: "cpu"          # or "cuda" if available
  prediction_horizon: 300  # 5 minutes
```

### Fusion Configuration  
```yaml
fusion:
  enabled: true
  method: "inverse_variance"  # Weight by prediction uncertainty
  uncertainty_threshold: 0.05  # Switch to long-term if short-term uncertain
```

### Covariates (System Metrics)
```yaml
covariates:
  enabled: true
  variables:
    - "cpu_usage"
    - "temperature" 
    - "voltage"
    - "frequency"
```

## Usage

### Basic Usage
```python
from chronotick_inference import ChronoTickInferenceEngine
import numpy as np

# Load historical offset data (1-second sampling)
offset_history = np.array([...])  # Clock offset measurements

# Initialize engine
with ChronoTickInferenceEngine("config.yaml") as engine:
    # Short-term prediction (next 5 seconds)
    short_result = engine.predict_short_term(offset_history[-300:])
    print(f"Next offset: {short_result.predictions[0]:.6f}s")
    
    # Long-term prediction (next 5 minutes)
    long_result = engine.predict_long_term(offset_history[-3600:])
    
    # Fused prediction (optimal combination)
    fused_result = engine.predict_fused(offset_history[-600:])
    print(f"Fused prediction: {fused_result.prediction:.6f}s")
    print(f"Weights: {fused_result.weights}")
```

### With System Metrics (Covariates)
```python
# Include system metrics for better predictions
covariates = {
    'cpu_usage': cpu_measurements,     # CPU utilization %
    'temperature': temp_measurements,  # System temperature
    'voltage': voltage_measurements,   # System voltage
    'frequency': freq_measurements     # CPU frequency
}

with ChronoTickInferenceEngine() as engine:
    result = engine.predict_fused(offset_history, covariates)
    print(f"Prediction with covariates: {result.prediction:.6f}s")
```

### Quick Prediction Function
```python
from chronotick_inference import quick_predict

# Simple one-line prediction
result = quick_predict(
    offset_history=offset_data,
    use_fusion=True,
    covariates=system_metrics
)
```

## Model Recommendations

### For Real-time Applications (< 1 second latency)
- **Short-term**: Chronos-Bolt (fastest inference, uncertainty quantification)
- **Long-term**: TimesFM 2.0 (flexible horizons, good accuracy)
- **Environment**: `core-models` (no dependency conflicts)

### For High-precision Applications  
- **Short-term**: TTM (true multivariate support, very fast)
- **Long-term**: Toto (specialized for observability data)
- **Environment**: `ttm` or `toto` (model-specific environments)

### Performance Characteristics

| Model | Inference Time | Memory | Accuracy | Best For |
|-------|---------------|---------|----------|----------|
| Chronos-Bolt | ~1.2s | Low | High | Real-time, uncertainty |
| TTM | ~0.8s | Very Low | High | Multivariate systems |
| TimesFM 2.0 | ~2.3s | High | Very High | Long-term trends |
| Toto | ~3.5s | High | High | Infrastructure monitoring |

## Environment Setup

The inference layer supports multiple TSFM environments:

```bash
# Recommended for production (stable, no conflicts)
cd ../tsfm && uv sync --extra core-models

# For multivariate capabilities  
cd ../tsfm && uv sync --extra ttm

# For observability specialization
cd ../tsfm && uv sync --extra toto
```

## Data Requirements

### Minimum Data Requirements
- **Short-term model**: 10+ data points (10 seconds)
- **Long-term model**: 60+ data points (1 minute)
- **Optimal performance**: 300+ data points (5+ minutes)

### Sampling Frequency
- Designed for 1-second sampling intervals
- Configurable via `config.yaml` frequency settings
- Models automatically handle frequency encoding

### Data Format
```python
# Historical offset data (seconds from true time)
offset_history = np.array([
    0.000123,   # t-2: 123 microseconds offset
    0.000145,   # t-1: 145 microseconds offset  
    0.000167,   # t-0: 167 microseconds offset
    # ... more measurements
])

# System metrics (optional, for covariates)
covariates = {
    'cpu_usage': np.array([45.2, 47.1, 48.5, ...]),     # CPU %
    'temperature': np.array([68.1, 68.3, 68.5, ...]),   # Celsius
    'voltage': np.array([3.31, 3.30, 3.29, ...]),       # Volts
    'frequency': np.array([2.4e9, 2.5e9, 2.4e9, ...])   # Hz
}
```

## Output Format

### Prediction Results
```python
@dataclass
class PredictionResult:
    predictions: np.ndarray          # Point forecasts
    uncertainty: np.ndarray          # Uncertainty estimates (σ)
    quantiles: Dict[str, np.ndarray] # Confidence intervals (10%, 90%)
    confidence: float                # Overall confidence score
    model_type: ModelType           # SHORT_TERM or LONG_TERM
    timestamp: float                # When prediction was made
    inference_time: float           # How long inference took
    metadata: Dict[str, Any]        # Model-specific information
```

### Fused Predictions
```python
@dataclass  
class FusedPrediction:
    prediction: float                      # Combined point forecast
    uncertainty: float                     # Combined uncertainty
    weights: Dict[str, float]              # Model weights used
    source_predictions: Dict[str, PredictionResult]  # Original predictions
    timestamp: float                       # When fusion was performed
    metadata: Dict[str, Any]               # Fusion metadata
```

## Integration with ChronoTick

The inference layer is designed to integrate seamlessly with the ChronoTick system:

### Continuous Correction Engine
```python
# In ChronoTick's correction loop
with ChronoTickInferenceEngine() as engine:
    while system_running:
        # Get recent offset measurements
        recent_offsets = collect_recent_offsets(window=300)  # 5 minutes
        
        # Get system metrics
        metrics = collect_system_metrics()
        
        # Predict next offset
        prediction = engine.predict_fused(recent_offsets, metrics)
        
        # Apply correction
        apply_offset_correction(prediction.prediction)
        
        # Wait for next sampling interval
        time.sleep(1.0)
```

### Retrospective Correction
```python
# When external sync event occurs (NTP/PTP)
def on_sync_event(true_offset, timestamp):
    # Update prediction history with true offset
    engine.apply_retrospective_correction(true_offset, timestamp)
    
    # Optionally retrain models if large error detected
    if abs(prediction_error) > threshold:
        engine.retrain_models()
```

## Performance Monitoring

### Health Checks
```python
health = engine.health_check()
print(f"Status: {health['status']}")  # healthy, degraded, error
print(f"Memory: {health['memory_usage_mb']:.1f} MB")
print(f"Models: {health['models']}")
```

### Performance Statistics
```python
stats = engine.get_performance_stats()
print(f"Short-term inferences: {stats['short_term_inferences']}")
print(f"Average inference time: {stats['average_inference_time']:.3f}s")
print(f"Fusion operations: {stats['fusion_operations']}")
```

## Error Handling

The system includes comprehensive error handling:

- **Model loading failures**: Automatic fallback to available models
- **Prediction failures**: Graceful degradation with error logging
- **Memory constraints**: Automatic context length limiting
- **Invalid data**: Preprocessing and validation checks

## Thread Safety

The inference engine is designed for multi-threaded use:

- Thread-safe prediction methods using RLock
- Separate model instances for concurrent access
- Atomic operations for performance statistics
- Safe shutdown procedures

## Extending the System

### Adding New Models
1. Add model configuration to `config.yaml`
2. Update model initialization in `initialize_models()`
3. Extend fusion logic if needed

### Custom Fusion Methods
1. Implement new fusion algorithm in `_fuse_predictions()`
2. Add configuration option in `config.yaml`
3. Update fusion selection logic

### Additional Covariates
1. Add new variables to `config.yaml` covariates section
2. Update data collection to include new metrics
3. Models will automatically incorporate them