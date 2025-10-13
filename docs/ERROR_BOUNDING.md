# Error Bounding and Uncertainty Quantification

**ChronoTick provides comprehensive uncertainty quantification for all time corrections.**

## Overview

Every time correction returned by ChronoTick includes:
- **Offset uncertainty** (`offset_uncertainty`): Error bound on the clock offset
- **Drift uncertainty** (`drift_uncertainty`): Error bound on the clock drift rate
- **Confidence score** (`confidence`): Overall prediction confidence (0-1)
- **Source indicator** (`source`): Which model/fusion provided the correction

## Implementation Locations

Error bounding is implemented across 6 core files:

### 1. **engine.py** (`server/src/chronotick/inference/engine.py`)
- `ClockCorrection` dataclass defines the uncertainty fields
- All corrections return offset/drift uncertainty

### 2. **daemon.py** (`server/src/chronotick/inference/daemon.py`)
- Daemon serves corrections with full uncertainty info
- Returns tuple: `(offset, drift, offset_unc, drift_unc, confidence, source)`

### 3. **tsfm_model_wrapper.py** (`server/src/chronotick/inference/tsfm_model_wrapper.py`)
- `predict_with_uncertainty()` method returns quantile-based uncertainties
- TimesFM 2.5 provides [0.1, 0.5, 0.9] quantiles
- Uncertainty = (q90 - q10) / 2 (80% confidence interval)

### 4. **predictive_scheduler.py** (`server/src/chronotick/inference/predictive_scheduler.py`)
- Schedules predictions with uncertainty propagation
- Caches corrections with full uncertainty metadata
- Fusion combines uncertainties via inverse-variance weighting

### 5. **real_data_pipeline.py** (`server/src/chronotick/inference/real_data_pipeline.py`)
- Returns `ClockCorrection` with all uncertainty fields populated
- NTP corrections include measurement uncertainty
- ML predictions include model uncertainty

### 6. **logging/dataset_correction_logger.py** (`server/src/chronotick/inference/logging/dataset_correction_logger.py`)
- Logs all corrections with uncertainty bounds
- CSV output includes: offset, drift, offset_unc, drift_unc, confidence
- Client predictions logged separately for analysis

## Usage Example

```python
from chronotick.inference.real_data_pipeline import RealDataPipeline

pipeline = RealDataPipeline("config.yaml")
correction = pipeline.get_real_clock_correction(time.time())

# Access uncertainty bounds
print(f"Offset: {correction.offset_correction:.3f}s ± {correction.offset_uncertainty:.3f}s")
print(f"Drift: {correction.drift_rate:.6f} ± {correction.drift_uncertainty:.6f}")
print(f"Confidence: {correction.confidence:.2%}")
print(f"Source: {correction.source}")
```

## Uncertainty Sources

1. **NTP Measurements**: Network delay, stratum, round-trip time
2. **ML Predictions**: Model quantiles (TimesFM 2.5: q10, q50, q90)
3. **Fusion**: Inverse-variance weighted combination
4. **Backtracking**: Interpolation uncertainty from NTP boundaries

## Configuration

Control uncertainty behavior via `config_enhanced_features.yaml`:

```yaml
prediction_scheduling:
  ntp_correction:
    offset_uncertainty_ms: 1.0    # Default NTP offset uncertainty
    drift_uncertainty_us_per_s: 100.0  # Default NTP drift uncertainty
```

## Validation

Run validation test to see uncertainty in action:

```bash
uv run python scripts/test_with_visualization_data.py backtracking \
  --config configs/config_enhanced_features.yaml \
  --duration 900 --interval 10
```

Results include CSV with full uncertainty quantification for every prediction.
