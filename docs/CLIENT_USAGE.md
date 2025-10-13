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

### 4. Complete Data Flow

```
TimesFM 2.5 forecast()
  ↓ Returns quantiles: [0.1, 0.5, 0.9]
  ↓
tsfm_model_wrapper.predict_with_uncertainty()
  ↓ Calculates uncertainty = (q90 - q10) / 2
  ↓
ClockCorrection dataclass
  ↓ Contains: offset_uncertainty, drift_uncertainty, confidence
  ↓
┌─────────────────────┬──────────────────────┐
│                     │                      │
Python Client       MCP Server            Daemon SHM
│                     │                      │
CorrectedTime.      get_time tool        ChronoTickData
uncertainty_seconds   offset_uncertainty    offset_uncertainty
✅ RETURNED!          ✅ RETURNED!          ✅ STORED!
```

## Verification

All three paths return uncertainty:

1. **Python Client**: `uncertainty_seconds` field in `CorrectedTime`
2. **MCP Tools**: `offset_uncertainty` in tool responses
3. **Daemon/SHM**: `offset_uncertainty` in shared memory

See `docs/ERROR_BOUNDING.md` for detailed implementation locations.

## Summary

**All three issues are resolved:**
- ✅ Client imports are unified (`clients/python/__init__.py`)
- ✅ pyproject.toml specifies TimesFM from GitHub
- ✅ Uncertainty quantification is returned in all clients/MCPs
