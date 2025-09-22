# ChronoTick Configuration Guide

## 🎯 Overview

ChronoTick provides multiple optimized configurations for different hardware setups and model combinations. Choose the configuration that best matches your deployment scenario.

## 📋 Available Configurations

### **CPU-Only Configurations**

#### **1. cpu_only_chronos.yaml**
- **Models**: Chronos-Bolt (short + long term)
- **Device**: CPU only
- **Memory**: ~2GB
- **Use Case**: Basic deployments, development, low-resource environments
- **Strengths**: Minimal resource usage, fast startup
- **Inference Time**: ~100-200ms

#### **2. cpu_only_ttm.yaml**
- **Models**: TTM (Tiny Time Mixer) (short + long term)
- **Device**: CPU only  
- **Memory**: ~1GB
- **Use Case**: Memory-constrained environments, edge deployment
- **Strengths**: Very efficient, good for longer horizons
- **Inference Time**: ~80-150ms

### **GPU-Only Configurations**

#### **3. gpu_only_timesfm.yaml**
- **Models**: TimesFM 2.0 (short + long term)
- **Device**: CUDA GPU
- **Memory**: ~8GB GPU memory
- **Use Case**: High-accuracy requirements, long-term predictions
- **Strengths**: Excellent for long horizons, covariates support
- **Inference Time**: ~50-100ms

#### **4. gpu_only_toto.yaml**
- **Models**: Toto transformer (short + long term)
- **Device**: CUDA GPU
- **Memory**: ~6GB GPU memory
- **Use Case**: Fast GPU inference, good balance of speed/accuracy
- **Strengths**: Efficient attention mechanism, future covariates
- **Inference Time**: ~40-80ms

### **Hybrid Configurations**

#### **5. hybrid_chronos_ttm.yaml**
- **Models**: Chronos (short-term) + TTM (long-term)
- **Device**: CPU only
- **Memory**: ~3GB
- **Use Case**: Best CPU performance, complementary model strengths
- **Strengths**: Optimized for each prediction horizon
- **Inference Time**: ~120-250ms

#### **6. hybrid_timesfm_toto.yaml**
- **Models**: Toto (short-term) + TimesFM (long-term)
- **Device**: CUDA GPU
- **Memory**: ~10GB GPU memory
- **Use Case**: High-performance GPU deployment
- **Strengths**: Speed + accuracy combination, advanced features
- **Inference Time**: ~60-120ms

### **Advanced Configuration**

#### **7. multi_device_full.yaml**
- **Models**: Chronos (CPU short-term) + TimesFM (GPU long-term)
- **Devices**: CPU + GPU hybrid
- **Memory**: ~12GB total (4GB CPU + 8GB GPU)
- **Use Case**: Maximum performance, high-availability systems
- **Features**: Load balancing, failover, advanced monitoring
- **Inference Time**: ~30-80ms

## 🔧 Configuration Selector Tool

Use the included configuration selector to automatically choose the best setup:

### **Automatic Hardware Detection & Recommendation**
```bash
# Detect hardware and get recommendation
uv run python chronotick_inference/config_selector.py --recommend

# Example output:
# Hardware Detection:
#   CPU Cores: 16
#   GPU Available: True
#   GPU Memory: 24 GB
# 
# Recommended Configuration: hybrid_timesfm_toto
# Reason: High-performance GPU setup with TimesFM + Toto
```

### **List All Available Configurations**
```bash
uv run python chronotick_inference/config_selector.py --list
```

### **Select and Apply Configuration**
```bash
# Apply recommended configuration
uv run python chronotick_inference/config_selector.py --select hybrid_timesfm_toto

# Apply specific configuration
uv run python chronotick_inference/config_selector.py --select cpu_only_chronos
```

### **Validate Configuration**
```bash
uv run python chronotick_inference/config_selector.py --validate config.yaml
```

## 📊 Performance Comparison

| Configuration | Memory | Inference Time | Accuracy | Best For |
|---------------|--------|----------------|----------|----------|
| cpu_only_chronos | 2GB | 150ms | Good | Development, basic deployments |
| cpu_only_ttm | 1GB | 120ms | Good | Edge devices, memory-limited |
| gpu_only_timesfm | 8GB | 75ms | Excellent | Long-term accuracy |
| gpu_only_toto | 6GB | 60ms | Very Good | Balanced performance |
| hybrid_chronos_ttm | 3GB | 180ms | Very Good | CPU optimization |
| hybrid_timesfm_toto | 10GB | 90ms | Excellent | GPU optimization |
| multi_device_full | 12GB | 50ms | Excellent | Production systems |

## 🎛️ Configuration Parameters Explained

### **Model Settings**
```yaml
short_term:
  model_name: chronos          # Model type: chronos, ttm, timesfm, toto
  device: cpu                  # Device: cpu, cuda
  prediction_horizon: 5        # How many steps to predict
  context_length: 100          # Historical data points to use
  max_uncertainty: 0.1         # Maximum acceptable uncertainty
```

### **Fusion Settings**
```yaml
fusion:
  method: inverse_variance     # Fusion method: inverse_variance, weighted_average
  uncertainty_threshold: 0.05  # Threshold for fusion decisions
  fallback_weights:            # Weights when uncertainty too high
    short_term: 0.7
    long_term: 0.3
```

### **Performance Tuning**
```yaml
performance:
  max_memory_mb: 2048         # Memory limit
  model_timeout: 10.0         # Timeout for inference
  cache_size: 10              # Number of models to keep in cache
  batch_size: 4               # Batch size for inference
```

## 🚀 Deployment Scenarios

### **Development & Testing**
```bash
# Quick start for development
uv run python chronotick_inference/config_selector.py --select cpu_only_chronos
uv run python examples/basic_usage.py
```

### **Edge Deployment**
```bash
# Memory-efficient for edge devices
uv run python chronotick_inference/config_selector.py --select cpu_only_ttm
```

### **Production Server (GPU)**
```bash
# High-performance production setup
uv run python chronotick_inference/config_selector.py --select hybrid_timesfm_toto
```

### **High-Availability System**
```bash
# Full-featured with redundancy
uv run python chronotick_inference/config_selector.py --select multi_device_full
```

## 🔄 TSFM Environment Mapping

### **Core Models Environment** (Recommended)
- **chronos**: ✅ Available
- **timesfm**: ✅ Available
- Compatible configurations: `cpu_only_chronos`, `gpu_only_timesfm`, `multi_device_full`

### **TTM Environment**
- **ttm**: ✅ Available
- Compatible configurations: `cpu_only_ttm`, `hybrid_chronos_ttm`
- Install: `uv sync --extra ttm`

### **Toto Environment**
- **toto**: ✅ Available
- Compatible configurations: `gpu_only_toto`, `hybrid_timesfm_toto`
- Install: `uv sync --extra toto`

## 🎯 Quick Selection Guide

### **Choose Based on Hardware:**
- **No GPU, <4GB RAM**: `cpu_only_ttm`
- **No GPU, 4-8GB RAM**: `cpu_only_chronos` or `hybrid_chronos_ttm`
- **GPU with 4-6GB VRAM**: `gpu_only_toto`
- **GPU with 8GB+ VRAM**: `gpu_only_timesfm` or `hybrid_timesfm_toto`
- **CPU + GPU, 12GB+ total**: `multi_device_full`

### **Choose Based on Requirements:**
- **Fastest inference**: `gpu_only_toto`
- **Best accuracy**: `gpu_only_timesfm` or `hybrid_timesfm_toto`
- **Lowest memory**: `cpu_only_ttm`
- **Best balance**: `hybrid_chronos_ttm` (CPU) or `hybrid_timesfm_toto` (GPU)
- **Production ready**: `multi_device_full`

## 🔧 Custom Configuration

To create a custom configuration:

1. Copy an existing configuration:
   ```bash
   cp chronotick_inference/configs/cpu_only_chronos.yaml my_custom_config.yaml
   ```

2. Modify parameters as needed

3. Validate your configuration:
   ```bash
   uv run python chronotick_inference/config_selector.py --validate my_custom_config.yaml
   ```

4. Apply your configuration:
   ```bash
   cp my_custom_config.yaml chronotick_inference/config.yaml
   ```

The configuration system provides maximum flexibility for your specific ChronoTick deployment needs!