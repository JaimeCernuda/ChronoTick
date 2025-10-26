# Experiment-14 Implementation Plan: Drift Prediction & Enhanced Uncertainty

**Status**: Implementation in progress
**Date**: 2025-10-25
**Goal**: Add drift prediction via TimesFM batch forecasting and improve uncertainty quantification

---

## Architecture Overview

### Current State (Experiment-13)
- **Offset Prediction**: TimesFM predicts clock offset (2 models: short-term 5s, long-term 60s)
- **Drift Tracking**: Measured retrospectively from offset changes
- **Uncertainty**: Arbitrary `offset * 0.1` formula
- **Formulas**: Fix1 (offset-based) and Fix2 (NTP-anchored)

### Target State (Experiment-14)
- **Dual Prediction**: TimesFM predicts BOTH offset and drift in parallel (batch mode)
- **Drift Dataset**: Collect drift history during warmup alongside offset history
- **Proper Uncertainty**: Use TimesFM quantile predictions (10th-90th percentiles)
- **Additional Formulas**: Fix3 (predicted drift) and Fix4 (hybrid)

---

## TimesFM Batch Architecture

### Verified Capability
From `tsfm/test_batch_forecasting.py`:
```python
point_forecast, quantile_forecast = model.forecast(
    horizon=5,
    inputs=[
        offset_history,  # Series 1: offset predictions
        drift_history,   # Series 2: drift predictions
    ]
)
# Returns:
# point_forecast: (2, 5) - [offset_pred, drift_pred]
# quantile_forecast: (2, 5, 10) - quantiles for both series
```

### Implementation Strategy
- **2 TimesFM instances** (not 4):
  - Short-term: Batch predicts `[offset, drift]` with 5-second horizon
  - Long-term: Batch predicts `[offset, drift]` with 60-second horizon
- **Independent predictions**: Each series predicted independently (not multivariate)
- **Shared warmup**: Single warmup period collects both offset AND drift history

---

## Implementation Tasks

### 1. NTP Client: Drift Dataset Collection
**File**: `server/src/chronotick/inference/ntp_client.py`

**Changes**:
- Add `drift_history` list alongside `offset_history`
- Calculate drift from consecutive NTP measurements
- Formula: `drift_us_per_s = (offset_i - offset_(i-1)) / time_delta_seconds * 1e6`
- Store in `ClockMeasurementCollector` for warmup

**Note**: Drift is in μs/s (microseconds per second). No unit change needed - floating point precision is sufficient.

---

### 2. TSFM Model Wrapper: Batch Prediction
**File**: `server/src/chronotick/inference/tsfm_model_wrapper.py`

**Current**:
```python
# Single-series prediction
prediction_result = self.predict_method(offset_history, covariates)
# Returns offset predictions only
```

**Target**:
```python
# Batch prediction of [offset, drift]
prediction_result = self.predict_method(
    inputs=[offset_history, drift_history],
    covariates=covariates
)
# Returns:
# - offset_predictions: (horizon,)
# - drift_predictions: (horizon,)
# - offset_quantiles: (horizon, 10)
# - drift_quantiles: (horizon, 10)
```

**Changes**:
- Modify `_get_offset_history()` → `_get_prediction_inputs()` to return both offset and drift
- Update `predict_with_uncertainty()` to handle batch outputs
- Extract quantiles for BOTH offset and drift
- Populate `PredictionWithUncertainty` with proper drift values and uncertainties

---

### 3. Inference Engine: Batch Forecasting
**File**: `server/src/chronotick/inference/engine.py`

**Changes**:
- Update `predict_short_term()` and `predict_long_term()` to accept batch inputs
- Call TimesFM with `inputs=[offset_history, drift_history]`
- Parse batch outputs: split into offset and drift results
- Extract quantiles for uncertainty calculation

---

### 4. Real Data Pipeline: Multiple Correction Formulas
**File**: `server/src/chronotick/inference/real_data_pipeline.py`

**Current Formulas**:
- **Fix1**: `system_time + offset + drift * elapsed` (tied to system clock)
- **Fix2**: `ntp_anchor + elapsed + drift * elapsed` (NTP-anchored)

**New Formulas**:
- **Fix3**: `ntp_anchor + elapsed + predicted_drift * elapsed`
  - Uses TimesFM predicted drift instead of retrospective drift
  - Tests if ML can predict drift better than measuring it

- **Fix4**: `ntp_anchor + elapsed + (offset - offset_at_ntp) + predicted_drift * elapsed`
  - Hybrid approach: Combines offset delta with predicted drift
  - Similar to Proposal 1 from experiment-13, but with predicted drift

**Implementation**:
- Add `calculate_fix3()` and `calculate_fix4()` methods
- Track `predicted_drift` from TimesFM alongside `measured_drift`
- Calculate all 4 formulas in parallel for comparison

---

### 5. Uncertainty Calculation: TimesFM Quantiles
**File**: `server/src/chronotick/inference/tsfm_model_wrapper.py`

**Current** (WRONG):
```python
offset_uncertainty = abs(offset) * 0.1  # Arbitrary!
```

**Target** (CORRECT):
```python
# Extract from TimesFM quantile predictions
p10 = offset_quantiles[step, 1]  # 10th percentile
p90 = offset_quantiles[step, 9]  # 90th percentile

# 80% confidence interval half-width
offset_uncertainty = (p90 - p10) / 2

# Store quantiles for full distribution
quantiles = {
    'p10': p10,
    'p50': offset_quantiles[step, 5],  # median
    'p90': p90
}
```

**Apply to BOTH**:
- Offset uncertainty from offset quantiles
- Drift uncertainty from drift quantiles

---

### 6. CSV Output Schema Updates
**File**: `scripts/client_driven_validation_v3.py` (or create v4)

**Current Columns**:
```
timestamp, system_time, ntp_time, offset_ms, drift_rate_us_per_s,
chronotick_time_fix1, chronotick_time_fix2, ...
```

**New Columns**:
```
..., predicted_drift_short, predicted_drift_long, predicted_drift_fused,
drift_uncertainty_us_per_s, offset_uncertainty_ms,
chronotick_time_fix3, chronotick_time_fix4,
fix3_error_ms, fix4_error_ms
```

---

### 7. MCP Interface Updates
**File**: `tsfm/chronotick_mcp.py` (if MCP server exists)

**Enhancements**:
- Return `predicted_drift` in `get_time` response
- Include `drift_uncertainty` in bounds
- Add `get_time_with_future_uncertainty` support for drift projection

---

## Configuration Files

### Experiment-14 Configs
Create platform-specific configs:
- `configs/config_experiment14_wsl.yaml`
- `configs/config_experiment14_homelab.yaml`
- `configs/config_experiment14_ares.yaml`

**Key Parameters**:
```yaml
experiment:
  name: "experiment-14"
  description: "Drift prediction with TimesFM batch forecasting"

warmup:
  duration_seconds: 300  # 5 minutes
  collect_drift: true
  min_samples_offset: 100
  min_samples_drift: 50

prediction:
  enable_drift_prediction: true
  use_batch_forecasting: true
  uncertainty_method: "timesfm_quantiles"  # not "offset_multiplier"

formulas:
  enabled: ["fix1", "fix2", "fix3", "fix4"]

output:
  include_drift_predictions: true
  include_quantiles: true
  include_all_formulas: true
```

---

## Testing Plan

### Phase 1: Local Testing (WSL2)
1. Run 5-minute test with warmup
2. Verify batch forecasting works
3. Check drift predictions are reasonable
4. Validate quantile-based uncertainty

### Phase 2: Platform Deployment
1. Deploy to homelab (unsynchronized)
2. Deploy to ARES cluster (synchronized)
3. Run overnight (8+ hours)

### Phase 3: Analysis
1. Compare Fix1 vs Fix2 vs Fix3 vs Fix4
2. Analyze uncertainty bounds (% of true time contained)
3. Evaluate predicted drift vs measured drift accuracy

---

## Success Criteria

1. **Batch Forecasting**: TimesFM successfully predicts both offset and drift
2. **Realistic Uncertainty**: Quantile-based bounds contain true time ~90% of time
3. **Drift Prediction Accuracy**: Predicted drift MAE < 10 μs/s
4. **Formula Comparison**: Identify best formula (likely Fix2 or Fix3)
5. **Data Quality**: Complete CSV with all new columns for full analysis

---

## Rollback Plan

If Experiment-14 fails:
- Keep Experiment-13 code in separate branch
- Can revert to offset-only prediction
- Maintain backward compatibility in configs
