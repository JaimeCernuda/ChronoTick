# TSFM Factory: Deep Model Analysis

## Overview
This document provides a comprehensive analysis of all 5 time series foundation models available in the TSFM Factory library, their capabilities, dependencies, and optimal use cases for high-frequency forecasting.

## Model Specifications & Capabilities

### 1. TimesFM 2.0 (Google Research)
**Repository**: `google/timesfm-2.0-500m-pytorch`
**Architecture**: 500M parameter transformer-based foundation model
**Key Upgrades**: Enhanced from original TimesFM with 4x longer context and dynamic covariates

#### Technical Specifications:
- **Context Length**: 2048 (4x increase from v1)
- **Horizon**: Flexible (no longer limited to 96)
- **Model Dimensions**: 1280
- **Layers**: 50 layers
- **Input Patch Length**: 32
- **Output Patch Length**: 128
- **Covariates Support**: ✅ Dynamic covariates
- **Multivariate Support**: ⚠️ Limited (separate univariate forecasting)

#### Dependency Requirements:
```bash
pip install timesfm>=1.0.0
```

#### Performance Characteristics:
- **Strength**: Long context understanding, flexible horizons
- **Best For**: Long-term forecasting, irregular time series
- **Limitation**: Primarily univariate, covariates support under development

---

### 2. TTM (Tiny Time Mixer - IBM Granite)
**Repository**: `ibm-granite/granite-timeseries-ttm-r2`
**Architecture**: Lightweight MLP-mixer architecture with channel mixing
**Key Feature**: True multivariate support with exogenous variable infusion

#### Technical Specifications:
- **Context Length**: 512 (minimum required: 90)
- **Prediction Length**: 96 (fixed)
- **Multivariate Support**: ✅ True multivariate via channel mixing
- **Exogenous Support**: ✅ Dedicated mixer blocks for covariates
- **Architecture**: MLP-mixer with channel and temporal mixing

#### Dependency Requirements:
```bash
pip install transformers==4.38.0  # Exact version required
pip install granite-tsfm  # Preferred implementation
# Fallback: transformers with trust_remote_code=True
```

#### Performance Characteristics:
- **Strength**: Efficient multivariate processing, exogenous variable integration
- **Best For**: Multivariate time series, systems with external factors
- **Limitation**: Fixed prediction length, requires minimum context

---

### 3. Chronos-Bolt (Amazon)
**Repository**: `amazon/chronos-bolt-base`
**Architecture**: Enhanced T5 encoder-decoder with probabilistic forecasting
**Key Feature**: 250x faster inference, 20x memory efficiency

#### Technical Specifications:
- **Model Sizes**: tiny, small, base (default: base)
- **Prediction Length**: 96
- **Samples**: 20 (for probabilistic forecasting)
- **Direct Multistep**: ✅ 250x faster than iterative
- **Quantile Forecasting**: ✅ [0.1, 0.2, ..., 0.9]
- **Multivariate Support**: ⚠️ Limited (separate univariate)

#### Dependency Requirements:
```bash
pip install chronos-forecasting>=1.0.0
```

#### Performance Characteristics:
- **Strength**: Ultra-fast inference, probabilistic outputs, memory efficient
- **Best For**: Real-time forecasting, uncertainty quantification
- **Limitation**: Primarily univariate, limited covariates support

---

### 4. Toto (Datadog)
**Repository**: `Datadog/Toto-Open-Base-1.0`
**Architecture**: 151M parameter decoder-only transformer
**Training Data**: 2+ trillion observability time series data points

#### Technical Specifications:
- **Prediction Length**: 336 (longest among all models)
- **Samples**: 256 (high sample count for uncertainty)
- **Samples per Batch**: 256
- **Multivariate Support**: ✅ Enhanced attention for observability data
- **Covariates Support**: ✅ High-cardinality covariates specialization
- **Training Scale**: 2+ trillion data points from observability systems

#### Dependency Requirements:
```bash
pip install transformers>=4.52.0  # Latest transformers required
pip install toto-ts>=0.1.0
```

#### Performance Characteristics:
- **Strength**: Long-term forecasting, observability data, high-cardinality covariates
- **Best For**: Infrastructure monitoring, APM data, long horizons
- **Limitation**: Requires latest transformers, memory intensive

---

### 5. Time-MoE (Mixture of Experts)
**Repository**: `Maple728/TimeMoE-200M`
**Architecture**: Billion-scale MoE with auto-regressive forecasting
**Key Feature**: Mixture-of-experts for specialized pattern learning

#### Technical Specifications:
- **Variant**: 200M (upgraded from 50M)
- **Max Context Length**: 4096 (longest context)
- **Prediction Length**: 96
- **MoE Architecture**: ✅ Specialized experts for different patterns
- **Covariates Adaptable**: ✅ Architecture ready for covariates
- **Auto-regressive**: ✅ Arbitrary prediction horizons

#### Dependency Requirements:
```bash
pip install transformers==4.40.1  # Specific version for compatibility
# CRITICAL: Incompatible with transformers>=4.49 due to DynamicCache API changes
```

#### Performance Characteristics:
- **Strength**: Extremely long context, expert specialization, flexible horizons
- **Best For**: Complex patterns, long-term dependencies, variable-length forecasting
- **Limitation**: Strict transformers version dependency, high memory usage

## Model Comparison Matrix

| Feature | TimesFM 2.0 | TTM | Chronos-Bolt | Toto | Time-MoE |
|---------|-------------|-----|--------------|------|----------|
| **Context Length** | 2048 | 512 | Variable | Variable | 4096 |
| **Max Horizon** | Flexible | 96 | 96 | 336 | Flexible |
| **True Multivariate** | ❌ | ✅ | ❌ | ✅ | ⚠️ |
| **Covariates** | ✅ | ✅ | ❌ | ✅ | ✅ |
| **Quantiles** | ✅ | ❌ | ✅ | ✅ | ❌ |
| **GPU Support** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Inference Speed** | Medium | Fast | Ultra-Fast | Medium | Slow |
| **Memory Usage** | High | Low | Low | High | Very High |

## Dependency Environment Matrix

### Critical Dependency Conflicts
The library manages three **mutually exclusive** environments due to transformers version conflicts:

#### Environment 1: TTM
```bash
uv sync --extra ttm
```
- **transformers==4.38.0** (exact version for TinyTimeMixer architecture)
- **Models Available**: TTM, Chronos, TimesFM
- **Conflicts With**: Time-MoE, Toto

#### Environment 2: Time-MoE
```bash
uv sync --extra time-moe
```
- **transformers==4.40.1** (required for Time-MoE)
- **Models Available**: Time-MoE, Chronos, TimesFM
- **Conflicts With**: TTM, Toto

#### Environment 3: Toto
```bash
uv sync --extra toto
```
- **transformers>=4.52.0** (latest features for Toto)
- **Models Available**: Toto, Chronos, TimesFM
- **Conflicts With**: TTM, Time-MoE

#### Environment 4: Core Models (No Conflicts)
```bash
uv sync --extra core-models
```
- **Models Available**: Chronos, TimesFM
- **Advantage**: No transformers conflicts
- **Use Case**: Production deployments without experimental models

## High-Frequency (1-Second) Forecasting Analysis

### Model Suitability for 1-Second Data:

#### ✅ **Recommended Models**:

1. **Chronos-Bolt** - BEST CHOICE
   - **Pros**: 250x faster inference, optimized for real-time
   - **Cons**: Limited to 96-step horizon (1.6 minutes ahead)
   - **Frequency Handling**: Excellent with freq=0 (unknown frequency)

2. **TTM** - GOOD CHOICE
   - **Pros**: Fast inference, handles short contexts well
   - **Cons**: Minimum 90 data points required (1.5 minutes of history)
   - **Frequency Handling**: Good with proper normalization

3. **TimesFM 2.0** - ACCEPTABLE
   - **Pros**: Flexible horizons, good pattern recognition
   - **Cons**: Higher memory usage, slower inference
   - **Frequency Handling**: Excellent with dynamic frequency detection

#### ⚠️ **Conditional Models**:

4. **Toto** - GOOD FOR LONG-TERM
   - **Pros**: 336-step prediction (5.6 minutes ahead)
   - **Cons**: Slow inference, high memory usage
   - **Use Case**: Infrastructure monitoring with 1-second metrics

5. **Time-MoE** - RESEARCH USE
   - **Pros**: 4096 context (1.1 hours of history)
   - **Cons**: Very slow, high memory, dependency conflicts
   - **Use Case**: Deep pattern analysis, not real-time

### Frequency Configuration for 1-Second Data:

```python
# Option 1: Automatic detection (recommended)
frequency_info = factory.create_frequency_info(
    freq_str='S',        # Second frequency
    freq_value=9,        # Numeric code for seconds
    is_regular=True,
    detected_freq='S'
)

# Option 2: Unknown frequency (works with all models)
freq = 0  # Let models handle frequency internally
```

## Performance Optimization for High-Frequency Data

### Memory Management:
```python
# Use context manager for automatic cleanup
with TSFMFactory() as factory:
    model = factory.load_model("chronos", device="cuda")
    # Forecasting code here
    # Model automatically unloaded on exit
```

### Batch Processing:
```python
# Process multiple series efficiently
results = {}
for series_name, data in time_series_dict.items():
    # Truncate to manageable size for memory
    if len(data) > 2048:
        data = data[-2048:]  # Keep recent 2048 seconds (34 minutes)
    
    result = factory.forecast_multivariate(
        multivariate_input, 
        horizon=60  # Predict 1 minute ahead
    )
    results[series_name] = result
```

### Device Optimization:
```python
# GPU setup for high-frequency inference
factory = TSFMFactory()
model = factory.load_model("chronos", device="cuda")

# CPU fallback for memory constraints
model = factory.load_model("ttm", device="cpu")
```

## Recommended Workflows

### Real-Time 1-Second Forecasting:
1. **Environment**: Core models (`uv sync --extra core-models`)
2. **Model**: Chronos-Bolt for speed
3. **Context**: Last 512-1024 seconds (8-17 minutes)
4. **Horizon**: 60-300 seconds (1-5 minutes)
5. **Device**: GPU for speed, CPU for stability

### Infrastructure Monitoring:
1. **Environment**: Toto (`uv sync --extra toto`)
2. **Model**: Toto for observability specialization
3. **Context**: Last 1800 seconds (30 minutes)
4. **Horizon**: 300-1800 seconds (5-30 minutes)
5. **Covariates**: System metrics, alerts, events

### Research/Analysis:
1. **Environment**: Time-MoE (`uv sync --extra time-moe`)
2. **Model**: Time-MoE for deep pattern analysis
3. **Context**: Last 3600-4096 seconds (1+ hour)
4. **Horizon**: Variable based on analysis needs
5. **Use Case**: Pattern discovery, long-term trend analysis