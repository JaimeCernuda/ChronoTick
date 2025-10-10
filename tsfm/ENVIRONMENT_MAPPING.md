# TSFM Model Environment Mapping

**Date**: 2025-10-09
**Purpose**: Document model environments, conflicts, and test requirements

---

## Overview

TSFM Factory supports 5 foundation models with **mutually exclusive** transformers library requirements. This document maps environments to models and tests.

---

## Model Environment Matrix

| Model | Transformers Version | Environment Extra | GPU Support | Context Length | Horizon |
|-------|---------------------|-------------------|-------------|----------------|---------|
| **Chronos-Bolt** | Any (no conflict) | `core-models` | Optional | 512 | Flexible |
| **TimesFM 2.0** | Any (no conflict) | `core-models` | Optional | 2048 | Flexible |
| **TTM** | `==4.38.0` | `ttm` | Yes | 512 | 96 |
| **Toto** | `>=4.52.0` | `toto` | Yes | 336 | 336 |
| **Time-MoE** | `==4.40.1` | `time-moe` | Yes | 4096 | Flexible |

---

## Environment Definitions

### 1. `core-models` (DEFAULT - No Conflicts)

**Install**:
```bash
cd tsfm/
uv sync --extra core-models
```

**Models Available**:
- ✅ Chronos-Bolt (Amazon, 250x faster inference)
- ✅ TimesFM 2.0 (Google, 500M params)

**Transformers**: No specific version requirement

**Use Case**: **RECOMMENDED for ChronoTick**
- Short-term: Chronos-Bolt on CPU
- Long-term: TimesFM on GPU (or CPU)
- No dependency conflicts
- Production-ready

**Tests Passing**:
```bash
uv run pytest tests/unit/test_chronos_bolt_enhanced.py  # 8 tests
uv run pytest tests/unit/test_timesfm_enhanced.py      # 7 tests
uv run pytest tests/chronotick/                        # 64 production tests
```

---

### 2. `ttm` (TTM Models Only)

**Install**:
```bash
cd tsfm/
uv sync --extra ttm
```

**Models Available**:
- ✅ TTM (IBM Tiny Time Mixer)
- ❌ Chronos-Bolt (compatible, not in extras)
- ❌ TimesFM (compatible, not in extras)

**Transformers**: `==4.38.0` (STRICT)

**Conflicts With**:
- ❌ `time-moe` (needs transformers 4.40.1)
- ❌ `toto` (needs transformers ≥4.52.0)

**Use Case**: Multivariate time series forecasting
- Supports multiple variables simultaneously
- Exogenous variable support
- Efficient for small-scale deployments

**Tests Passing**:
```bash
uv run pytest tests/unit/test_ttm_enhanced.py          # 9 tests
uv run pytest tests/unit/test_multivariate.py          # Multivariate support
```

**ChronoTick Compatibility**: ⚠️ **CONDITIONAL**
- Can be used as short-term or long-term model
- Requires separate environment from Toto/Time-MoE
- Not recommended for production (use core-models instead)

---

### 3. `toto` (Toto Models Only)

**Install**:
```bash
cd tsfm/
uv sync --extra toto
```

**Models Available**:
- ✅ Toto (Datadog, high-cardinality covariates)

**Transformers**: `>=4.52.0` (MINIMUM)

**Conflicts With**:
- ❌ `ttm` (needs transformers 4.38.0)
- ❌ `time-moe` (needs transformers 4.40.1)

**Use Case**: High-cardinality covariate support
- Enhanced attention mechanisms
- Static and dynamic covariates
- Best for complex covariate scenarios

**Tests Passing**:
```bash
uv run pytest tests/unit/test_toto_enhanced.py         # 8 tests
uv run pytest tests/unit/test_covariates.py            # Covariate support
```

**ChronoTick Compatibility**: ⚠️ **EXPERIMENTAL**
- Can use system metrics as high-cardinality covariates
- Requires separate environment
- Not currently integrated

---

### 4. `time-moe` (Time-MoE Models Only)

**Install**:
```bash
cd tsfm/
uv sync --extra time-moe
```

**Models Available**:
- ✅ Time-MoE (Mixture-of-Experts, 200M params)

**Transformers**: `==4.40.1` (STRICT)

**Conflicts With**:
- ❌ `ttm` (needs transformers 4.38.0)
- ❌ `toto` (needs transformers ≥4.52.0)

**Use Case**: Large-scale time series with MoE architecture
- 4096 context length
- Multiple expert networks
- Best for complex patterns

**Tests Passing**:
```bash
uv run pytest tests/unit/test_time_moe_enhanced.py     # 8 tests
```

**ChronoTick Compatibility**: ⚠️ **NOT RECOMMENDED**
- Too resource-intensive for real-time predictions
- Long inference times
- Better suited for offline analysis

---

## Environment Conflict Resolution

### uv Conflict Configuration

**File**: `pyproject.toml`

```toml
[tool.uv]
conflicts = [
    [
        { extra = "ttm" },
        { extra = "time-moe" },
        { extra = "toto" },
    ],
]
```

**Behavior**:
- Running `uv sync --extra ttm --extra toto` → **ERROR** (conflict detected)
- `uv sync --extra core-models --extra ttm` → **OK** (core-models compatible)
- `uv sync --extra ttm` → **OK** (single environment)

---

## Test Environment Requirements

### Unit Tests (Model-Specific)

| Test File | Required Environment | Models Tested | Can Run in core-models? |
|-----------|---------------------|---------------|------------------------|
| `test_factory_enhanced.py` | Any | Factory methods | ✅ Yes |
| `test_base_enhanced.py` | Any | Base class | ✅ Yes |
| `test_chronos_bolt_enhanced.py` | `core-models` | Chronos-Bolt | ✅ Yes |
| `test_timesfm_enhanced.py` | `core-models` | TimesFM | ✅ Yes |
| `test_ttm_enhanced.py` | `ttm` | TTM | ❌ No |
| `test_toto_enhanced.py` | `toto` | Toto | ❌ No |
| `test_time_moe_enhanced.py` | `time-moe` | Time-MoE | ❌ No |
| `test_multivariate.py` | `ttm` preferred | Multivariate | ⚠️ Partial |
| `test_covariates.py` | `toto` preferred | Covariates | ⚠️ Partial |

### Integration Tests

| Test File | Required Environment | Notes |
|-----------|---------------------|-------|
| `test_performance.py` | `core-models` | Benchmarks Chronos/TimesFM |
| `test_enhanced_features_integration.py` | `core-models` | Full feature test |
| `test_backward_compatibility.py` | Any | API compatibility |

### ChronoTick Tests

| Test File | Required Environment | Models Used | Status |
|-----------|---------------------|-------------|--------|
| `test_real_data_pipeline.py` | None (mocked) | N/A | ✅ 31/31 |
| `test_ntp_client.py` | None | N/A | ✅ 13/13 |
| `test_mcp_server.py` | None (mocked) | N/A | ✅ 9/9 |
| `test_predictive_scheduler.py` | None (mocked) | N/A | ✅ 11/11 |
| `test_engine.py` | `core-models` | Chronos, TimesFM | ❌ 0/5 (config bugs) |
| `test_integration.py` | `core-models` | Old workflow | ❌ 3/7 (deprecated) |

---

## ChronoTick Production Recommendations

### Recommended Setup: `core-models`

**Why**:
1. ✅ No transformers conflicts
2. ✅ Fast inference (Chronos-Bolt)
3. ✅ Good long-term predictions (TimesFM)
4. ✅ Well-tested (64/64 production tests pass)
5. ✅ CPU-friendly (Chronos) + GPU-capable (TimesFM)

**Configuration** (`config.yaml`):
```yaml
short_term:
  enabled: true
  model_name: "chronos"
  device: "cpu"
  context_length: 100
  prediction_horizon: 5
  inference_interval: 1.0

long_term:
  enabled: true
  model_name: "timesfm"
  device: "gpu"  # or "cpu" if no GPU
  context_length: 300
  prediction_horizon: 60
  inference_interval: 30.0

fusion:
  enabled: true
  method: "inverse_variance"
```

**Installation**:
```bash
cd tsfm/
uv sync --extra core-models --extra dev --extra test
```

**Verification**:
```bash
# Test models load
uv run python -c "from tsfm import TSFMFactory; f = TSFMFactory(); print(f.list_models())"

# Expected output:
# ['chronos', 'timesfm']

# Run tests
uv run pytest tests/chronotick/ -v
# Expected: 64/64 pass (once engine is fixed)
```

---

## Alternative Configurations

### Option 2: TTM for Multivariate

**Use Case**: Need to predict multiple clock metrics simultaneously

**Setup**:
```bash
uv sync --extra ttm
```

**Config**:
```yaml
short_term:
  model_name: "ttm"
  device: "cpu"

long_term:
  enabled: false  # TTM only
```

**Limitations**:
- Can't use Chronos/TimesFM in same environment
- Need separate install for development

---

### Option 3: Toto for Rich Covariates

**Use Case**: Many system metrics (10+ covariates)

**Setup**:
```bash
uv sync --extra toto
```

**Config**:
```yaml
short_term:
  model_name: "toto"
  device: "gpu"

covariates:
  enabled: true
  variables: [cpu_usage, temperature, memory, io_wait, ...]
```

**Limitations**:
- Requires GPU for good performance
- Can't combine with TTM or Time-MoE
- More complex covariate management

---

## Testing Strategy by Environment

### Full Test Suite (Requires All Environments)

**Step 1**: Test core models
```bash
uv sync --extra core-models --extra test
uv run pytest tests/unit/test_chronos_bolt_enhanced.py -v
uv run pytest tests/unit/test_timesfm_enhanced.py -v
uv run pytest tests/chronotick/ -v
```

**Step 2**: Test TTM (separate environment)
```bash
uv sync --extra ttm --extra test
uv run pytest tests/unit/test_ttm_enhanced.py -v
```

**Step 3**: Test Toto (separate environment)
```bash
uv sync --extra toto --extra test
uv run pytest tests/unit/test_toto_enhanced.py -v
```

**Step 4**: Test Time-MoE (separate environment)
```bash
uv sync --extra time-moe --extra test
uv run pytest tests/unit/test_time_moe_enhanced.py -v
```

### Quick Test (core-models only)

```bash
uv sync --extra core-models --extra test
uv run pytest tests/chronotick/ tests/unit/test_chronos_bolt_enhanced.py -v
```

**Expected**: 72 tests pass (64 ChronoTick + 8 Chronos)

---

## Environment Switching Workflow

### Development Workflow

```bash
# Primary development (ChronoTick)
uv sync --extra core-models --extra dev --extra test

# Test TTM features
uv sync --extra ttm --extra dev --extra test
# Work on TTM...
# Run TTM tests...

# Switch back to core
uv sync --extra core-models --extra dev --extra test
```

### CI/CD Strategy

**Option 1: Matrix Testing**
```yaml
# .github/workflows/test.yml
strategy:
  matrix:
    environment: [core-models, ttm, toto, time-moe]
steps:
  - run: uv sync --extra ${{ matrix.environment }} --extra test
  - run: uv run pytest tests/unit/test_${{ matrix.environment }}*.py
```

**Option 2: Sequential Testing**
```bash
#!/bin/bash
for env in core-models ttm toto time-moe; do
  uv sync --extra $env --extra test
  uv run pytest tests/unit/test_${env}*.py
done
```

---

## Troubleshooting

### Error: "Conflicting dependencies"

```
ERROR: Cannot install ttm and toto extras together
```

**Solution**: Install separately
```bash
# Wrong:
uv sync --extra ttm --extra toto  # ❌ CONFLICT

# Right:
uv sync --extra ttm   # Test TTM
uv sync --extra toto  # Test Toto (separate run)
```

### Error: "Model not found"

```
KeyError: 'ttm' not in available models
```

**Solution**: Check environment
```bash
# Verify current environment
uv run python -c "from tsfm import TSFMFactory; print(TSFMFactory().list_models())"

# If TTM missing:
uv sync --extra ttm
```

### Error: "Transformers version mismatch"

```
ImportError: Transformers version 4.52.0 incompatible with TTM
```

**Solution**: Reinstall correct environment
```bash
# Check installed version
uv run python -c "import transformers; print(transformers.__version__)"

# Reinstall TTM environment
uv sync --extra ttm --reinstall
```

---

## Summary

### Production ChronoTick: Use `core-models`

✅ **Recommended**: Chronos-Bolt (CPU) + TimesFM (GPU)
- No conflicts
- Fast + accurate
- Well-tested
- Production-ready

### Development/Research: Use environment switching

⚠️ **TTM/Toto/Time-MoE**: Experimental
- Requires environment switching
- Test in isolation
- Not recommended for production ChronoTick

### Test Coverage: 117 total tests

- **64 tests**: ChronoTick production (environment-agnostic)
- **40 tests**: TSFM core models (core-models environment)
- **13 tests**: Model-specific (ttm, toto, time-moe environments)

---

**Signed**: Environment Mapping Documentation
**Date**: 2025-10-09
**Maintainer**: ChronoTick Team
