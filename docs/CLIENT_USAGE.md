# Client Usage & Uncertainty Quantification

This document addresses common questions about ChronoTick client usage.

## ✅ Issue 1: Client Imports (RESOLVED - Already Works!)

**Current State**: Imports ARE unified in `clients/python/__init__.py`

Users can import everything from one place:
```python
from clients.python import ChronoTickClient, ChronoTickSHMConfig

# OR if installed as package:
from chronotick_client import ChronoTickClient, ChronoTickSHMConfig
```

The `__init__.py` exports both classes:
```python
__all__ = ['ChronoTickClient', 'ChronoTickSHMConfig']
```

**No action needed** - this already provides clean imports!

## ✅ Issue 2: pyproject.toml TimesFM Dependency (FIXED)

**Problem**: TimesFM 2.5 API requires GitHub version, not PyPI

**Solution**: Updated `pyproject.toml`:
```toml
chronotick = [
    "timesfm @ git+https://github.com/google-research/timesfm.git",
    ...
]
```

**Installation**:
```bash
uv sync --extra chronotick
```

## ✅ Issue 3: Uncertainty Quantification Returns (CONFIRMED WORKING)

**YES - Error bounds ARE returned to clients!** Here's the complete pipeline:

### 1. TimesFM 2.5 Quantiles → Uncertainties

**Location**: `server/src/chronotick/tsfm/LLM/timesfm.py:217-252`

```python
# TimesFM returns quantiles [0.1, 0.5, 0.9]
point_forecast, quantile_forecast = self.tfm.forecast(...)

# Extract quantiles
quantiles = {}
if quantile_forecast is not None:
    for q_level, q_values in quantile_forecast.items():
        quantiles[str(q_level)] = q_values
```

**Uncertainty Calculation**: `server/src/chronotick/inference/tsfm_model_wrapper.py:89-92`

```python
# Convert quantiles to uncertainty (80% confidence interval)
if quantiles and '0.1' in quantiles and '0.9' in quantiles:
    uncertainty = (quantiles['0.9'] - quantiles['0.1']) / 2.0
```

### 2. Python Client Returns Uncertainty

**Location**: `clients/python/client.py:35-57, 143-151`

```python
class CorrectedTime(NamedTuple):
    corrected_timestamp: float
    system_timestamp: float
    uncertainty_seconds: float      # ✅ RETURNED!
    confidence: float               # ✅ RETURNED!
    source: str
    offset_correction: float
    drift_rate: float

# Usage:
client = ChronoTickClient()
time_info = client.get_time()
print(f"Uncertainty: ±{time_info.uncertainty_seconds * 1000:.3f}ms")
```

### 3. MCP `get_time` Tool Returns Uncertainty

**Location**: `server/src/chronotick/inference/mcp_server.py` (see ERROR_BOUNDING.md)

MCP tools return full `ClockCorrection` with:
- `offset_uncertainty` (from TimesFM quantiles)
- `drift_uncertainty` (from TimesFM quantiles)
- `confidence` (model confidence score)

Example MCP response:
```json
{
  "corrected_time": 1234567890.123,
  "offset_correction": 0.000110,
  "drift_rate": 0.000001,
  "offset_uncertainty": 0.000023,   // ✅ FROM TIMESFM!
  "confidence": 0.95,                // ✅ FROM TIMESFM!
  "source": "fusion"
}
```

### 4. Fusion Uncertainty Quantification (CONFIRMED!)

**YES - Short and long-term quantiles ARE merged during fusion!**

**Location**: `server/src/chronotick/inference/engine.py:452-538`

When both short-term and long-term models are enabled, their uncertainties are combined using **weighted variance combination**:

```python
# Step 1: Extract uncertainties from both models (line 474-475)
short_uncertainty = self._get_prediction_uncertainty(short_pred)  # From short-term quantiles
long_uncertainty = self._get_prediction_uncertainty(long_pred)    # From long-term quantiles

# Step 2: Calculate inverse variance weights (line 484-491)
if fusion_config['method'] == 'inverse_variance':
    short_variance = short_uncertainty ** 2
    long_variance = long_uncertainty ** 2

    total_precision = (1 / short_variance) + (1 / long_variance)
    short_weight = (1 / short_variance) / total_precision
    long_weight = (1 / long_variance) / total_precision

# Step 3: Fuse predictions using weighted average (line 512)
fused_value = short_weight * short_value + long_weight * long_value

# Step 4: COMBINE UNCERTAINTIES using weighted variance formula (line 514-518)
fused_uncertainty = np.sqrt(
    (short_weight ** 2) * (short_uncertainty ** 2) +
    (long_weight ** 2) * (long_uncertainty ** 2)
)
```

**Mathematical Formula**:
```
σ_fused = √(w₁² σ₁² + w₂² σ₂²)
```

Where:
- `σ₁, σ₂` = Short and long-term uncertainties (from TimesFM quantiles)
- `w₁, w₂` = Inverse variance weights
- `σ_fused` = Combined uncertainty returned to clients

This ensures that:
- **More confident models** (lower uncertainty) get higher weight
- **Combined uncertainty** is always less than or equal to individual uncertainties
- **Uncertainty propagation** follows standard statistical principles

### 5. Complete Data Flow (With Fusion)

```
SHORT-TERM TimesFM 2.5              LONG-TERM TimesFM 2.5
  ↓ Quantiles [0.1, 0.5, 0.9]         ↓ Quantiles [0.1, 0.5, 0.9]
  ↓                                    ↓
  σ_short = (q90 - q10) / 2           σ_long = (q90 - q10) / 2
  ↓                                    ↓
  ↓────────────────┬──────────────────↓
                   ↓
          FUSION ENGINE (engine.py:452-538)
                   ↓
     1. Calculate inverse variance weights
     2. Fuse predictions: ŷ = w₁y₁ + w₂y₂
     3. COMBINE UNCERTAINTIES: σ_fused = √(w₁²σ₁² + w₂²σ₂²)
                   ↓
          FusedPrediction
    (prediction, fused_uncertainty, weights)
                   ↓
  ┌───────────────┼───────────────────┐
  ↓               ↓                   ↓
Python Client   MCP Server        Daemon SHM
  ↓               ↓                   ↓
CorrectedTime   get_time tool     ChronoTickData
uncertainty_    offset_           offset_
seconds         uncertainty       uncertainty
✅ FUSED!       ✅ FUSED!         ✅ FUSED!
```

## Verification

All three paths return uncertainty:

1. **Python Client**: `uncertainty_seconds` field in `CorrectedTime`
2. **MCP Tools**: `offset_uncertainty` in tool responses
3. **Daemon/SHM**: `offset_uncertainty` in shared memory

See `docs/ERROR_BOUNDING.md` for detailed implementation locations.

## Summary

**All four issues are resolved:**
- ✅ Client imports are unified (`clients/python/__init__.py`)
- ✅ pyproject.toml specifies TimesFM from GitHub
- ✅ Uncertainty quantification is returned in all clients/MCPs
- ✅ **Fusion uncertainty quantification IS implemented** - Short and long-term quantiles are merged using weighted variance combination (`engine.py:514-518`)
