# 8-Hour Test Degradation: Root Cause Analysis

## Executive Summary

TimesFM performance degrades catastrophically after ~4 hours, with prediction errors growing from 55ms (hours 0-2) to 275ms (hours 6-8). **The model is OVERPREDICTING by 2-3x**, not underpredicting. When NTP says "you need 120ms correction", TimesFM predicts "you need 500ms correction", causing massive overcorrection that accumulates error faster than the system clock.

## Diagnostic Findings

### Performance Degradation Timeline
- **Hours 0-2**: Predicted 142.5ms vs Truth 130.4ms (1.09x ratio) ✓ **GOOD**
- **Hours 2-4**: Predicted 123.6ms vs Truth 111.2ms (1.11x ratio) ✓ **GOOD**
- **Hours 4-6**: Predicted 249.8ms vs Truth 135.8ms (1.84x ratio) ⚠️ **DIVERGING**
- **Hours 6-8**: Predicted 373.8ms vs Truth 113.8ms (3.28x ratio) ❌ **CATASTROPHIC**

### Key Observations
1. **Mean prediction error**: 146.0ms absolute (93.4ms signed)
2. **Accumulated error at 8hr**: ChronoTick 21,400ms vs System Clock 18,900ms
3. **Dataset growth**: 0 → 2800+ measurements over 8 hours
4. **NTP uncertainties**: 99.4% < 10ms (suspiciously low, suggests overconfidence)

## Root Causes

### 1. Dataset Contamination Feedback Loop (PRIMARY CAUSE)

**Location**: `real_data_pipeline.py:1378-1385`

```python
self.dataset_manager.add_prediction(
    timestamp=current_time,
    offset=correction.offset_correction,  # ← BAD PREDICTION GETS STORED
    drift=correction.drift_rate,
    source=correction.source,
    uncertainty=correction.offset_uncertainty,
    confidence=correction.confidence
)
```

**The Problem**:
- Every TimesFM prediction gets added to the dataset, even bad ones
- When TimesFM predicts 400ms (when truth is 120ms), that 400ms gets stored
- Next prediction uses a dataset contaminated with the 400ms error
- This creates **positive feedback**: bad predictions → bad training data → worse predictions
- Over 8 hours, contamination accumulates and dominates the dataset

**Evidence**:
- Dataset grows from 0 to 2800+ measurements
- Prediction errors compound over time
- No filtering or quality control on predictions before adding to dataset

### 2. Unbounded Dataset Growth

**Location**: `real_data_pipeline.py:995-999`

```python
def get_recent_measurements(self, window_seconds: int = None):
    if window_seconds is None:
        # No time filtering - return ALL accumulated measurements
        # This prevents dataset from being artificially truncated over time
        measurements = [(ts, data['offset']) for ts, data in self.measurement_dataset.items()]
```

**The Problem**:
- Dataset grows without limit (2800+ entries by hour 8)
- Comment says "prevents artificial truncation" but causes contamination accumulation
- TimesFM context windows are only 100 (short-term) and 300 (long-term)
- Feeding 2800 measurements to a model expecting 100-300 may cause issues
- Old contaminated predictions never expire

### 3. No Prediction Magnitude Bounds

**Location**: `engine.py:197-289` (predict functions)

**The Problem**:
- No sanity checks on prediction magnitudes
- If TimesFM predicts 700ms, it's accepted without question
- No capping, no outlier rejection, no bounds checking
- Allows runaway predictions to contaminate dataset

### 4. Backtracking Correction May Amplify Errors

**Location**: `real_data_pipeline.py:771-917`

```python
def _apply_backtracking_correction(self, start_time, end_time, error, interval_duration):
    """REPLACE predictions with interpolated NTP ground truth"""
    # Replaces ALL predictions with linear interpolation
    ntp_interpolated = start_ntp_offset + alpha * (end_ntp_offset - start_ntp_offset)
    self.measurement_dataset[timestamp]['offset'] = ntp_interpolated
```

**The Problem**:
- Assumes linear interpolation is correct, but real drift may be nonlinear
- If NTP measurement has errors, interpolation spreads that error across entire interval
- Aggressive replacement may overcorrect and create new training bias
- Combined with contamination feedback, this can amplify rather than dampen errors

### 5. Preprocessing Not Applied to Predictions

**Location**: `config_test_drift_aware.yaml:77-87`

```yaml
preprocessing:
  outlier_removal:
    enabled: true
    method: iqr
    threshold: 2.0
```

**The Problem**:
- Outlier removal only applies to INPUT data fed to models
- Predictions added to dataset bypass outlier filtering
- Bad predictions don't get caught and removed before contaminating dataset

## Proposed Solutions

### Solution 1: Dataset Sliding Window (HIGH IMPACT)

**What**: Cap dataset size using sliding window instead of unlimited growth

**Implementation**:
```python
# In DatasetManager.add_prediction()
def add_prediction(self, timestamp, offset, drift, source, uncertainty, confidence):
    # Add prediction
    self.measurement_dataset[int(timestamp)] = {...}

    # NEW: Enforce sliding window
    MAX_DATASET_SIZE = 1000  # Keep last 1000 measurements
    if len(self.measurement_dataset) > MAX_DATASET_SIZE:
        # Remove oldest entries
        oldest_keys = sorted(self.measurement_dataset.keys())[:100]
        for key in oldest_keys:
            del self.measurement_dataset[key]
```

**Benefits**:
- Prevents ancient contamination from persisting
- Keeps dataset size manageable (1000 vs 2800+)
- Old bad predictions eventually expire
- More responsive to recent corrections

### Solution 2: Adaptive Prediction Magnitude Capping (HIGH IMPACT)

**What**: Cap predictions relative to last NTP measurement + time elapsed (not absolute)

**Implementation**:
```python
# In RealDataPipeline._get_predictive_correction()
def _get_predictive_correction(self, current_time):
    correction = self.predictive_scheduler.get_fused_correction(current_time)

    if correction:
        # NEW: Adaptive cap based on last NTP + time elapsed
        time_since_ntp = current_time - self.last_ntp_time if self.last_ntp_time else 0
        last_ntp_magnitude = abs(self.last_ntp_offset) if self.last_ntp_offset is not None else 0.1

        # Allow growth based on: last NTP + accumulated drift + uncertainty buffer
        MAX_DRIFT_RATE = 0.005  # 5ms/s max reasonable drift
        UNCERTAINTY_BUFFER = 0.050  # 50ms buffer for measurement uncertainty
        max_allowed_magnitude = last_ntp_magnitude + (time_since_ntp * MAX_DRIFT_RATE) + UNCERTAINTY_BUFFER

        # Apply adaptive cap
        if abs(correction.offset_correction) > max_allowed_magnitude:
            logger.warning(f"Capping prediction from {correction.offset_correction*1000:.1f}ms to {max_allowed_magnitude*1000:.1f}ms "
                          f"(last_ntp={last_ntp_magnitude*1000:.1f}ms, age={time_since_ntp:.0f}s)")
            correction.offset_correction = np.sign(correction.offset_correction) * max_allowed_magnitude
            correction.confidence *= 0.5  # Reduce confidence for capped predictions
```

**Benefits**:
- Allows legitimate large corrections when drift accumulates
- Prevents runaway predictions (500ms when truth is 120ms)
- Adapts to system behavior (more permissive when NTP is old)
- Still aggressive when NTP is recent (tight control)

### Solution 3: Confidence-Based Prediction Capping (MEDIUM IMPACT)

**What**: Cap low-confidence predictions more aggressively (don't skip - maintains TimesFM spacing!)

**Implementation**:
```python
# In RealDataPipeline._get_predictive_correction()
def _apply_confidence_based_capping(self, correction, last_ntp_magnitude):
    """Cap predictions based on confidence - ALWAYS add to maintain spacing for TimesFM"""

    if correction.confidence >= 0.8:
        # High confidence: Use as-is (already went through adaptive cap)
        return correction

    elif correction.confidence >= 0.5:
        # Medium confidence: Cap to 1.5x last NTP magnitude
        cap = last_ntp_magnitude * 1.5
        if abs(correction.offset_correction) > cap:
            logger.info(f"Capping medium-confidence ({correction.confidence:.2f}) prediction "
                       f"from {correction.offset_correction*1000:.1f}ms to {cap*1000:.1f}ms")
            correction.offset_correction = np.sign(correction.offset_correction) * cap

    else:
        # Low confidence: Cap to 1.2x last NTP magnitude
        cap = last_ntp_magnitude * 1.2
        if abs(correction.offset_correction) > cap:
            logger.info(f"Capping low-confidence ({correction.confidence:.2f}) prediction "
                       f"from {correction.offset_correction*1000:.1f}ms to {cap*1000:.1f}ms")
            correction.offset_correction = np.sign(correction.offset_correction) * cap

    # ALWAYS add to dataset (maintains even spacing for TimesFM)
    return correction
```

**Benefits**:
- Maintains evenly-spaced data required by TimesFM (freq=9 seconds assumption)
- Reduces contamination from uncertain predictions while keeping them in dataset
- Progressive capping based on confidence level
- No gaps in training data

### Solution 4: Prediction Damping Factor (MEDIUM IMPACT)

**What**: Apply only partial correction instead of full prediction

**Implementation**:
```python
# In RealDataPipeline._get_predictive_correction()
DAMPING_FACTOR = 0.85  # Apply 85% of predicted correction

if correction:
    # Apply damping to prediction
    correction.offset_correction *= DAMPING_FACTOR
    correction.drift_rate *= DAMPING_FACTOR
    # Scale uncertainty proportionally
    correction.offset_uncertainty *= DAMPING_FACTOR
```

**Benefits**:
- Provides headroom for prediction errors
- Prevents overcorrection
- Still provides significant improvement over system clock
- Simple and mathematically sound

### Solution 5: Dataset Outlier Cleaning (MEDIUM IMPACT)

**What**: Apply outlier removal to dataset after adding predictions

**Implementation**:
```python
# In DatasetManager.add_prediction()
def add_prediction(self, timestamp, offset, drift, source, uncertainty, confidence):
    # Add prediction
    self.measurement_dataset[int(timestamp)] = {...}

    # NEW: Clean outliers periodically
    if len(self.measurement_dataset) % 100 == 0:  # Every 100 measurements
        self._clean_outliers_from_dataset()

def _clean_outliers_from_dataset(self):
    """Remove outlier predictions using IQR method"""
    offsets = np.array([d['offset'] for d in self.measurement_dataset.values()])
    Q1, Q3 = np.percentile(offsets, [25, 75])
    IQR = Q3 - Q1
    lower_bound = Q1 - 2.0 * IQR
    upper_bound = Q3 + 2.0 * IQR

    # Remove outliers
    for ts in list(self.measurement_dataset.keys()):
        if not (lower_bound <= self.measurement_dataset[ts]['offset'] <= upper_bound):
            if self.measurement_dataset[ts]['source'].startswith('prediction_'):
                del self.measurement_dataset[ts]
                logger.info(f"Removed outlier prediction at t={ts}")
```

**Benefits**:
- Catches and removes contamination after it enters dataset
- Uses same IQR method as preprocessing
- Only removes predictions, not NTP measurements (preserves ground truth)

## Recommended Implementation Strategy

### Phase 1: Core Fixes (Implement Tonight)
1. ✅ **Solution 1**: Dataset Sliding Window (1000 measurements)
2. ✅ **Solution 2**: Prediction Magnitude Capping (300ms)
3. ✅ **Solution 3**: Confidence-Based Dataset Addition (threshold=0.7)

### Phase 2: Enhancement (If Phase 1 Insufficient)
4. Solution 4: Prediction Damping Factor (0.85)
5. Solution 5: Dataset Outlier Cleaning

## Configuration Changes

Add to `config_test_drift_aware.yaml`:

```yaml
prediction_scheduling:
  dataset:
    max_history_size: 1000  # Enforce sliding window

  adaptive_capping:
    max_drift_rate: 0.005  # 5ms/s maximum reasonable system drift
    uncertainty_buffer: 0.050  # 50ms buffer for measurement uncertainty

  confidence_capping:
    high_confidence_threshold: 0.8  # Use prediction as-is above this
    medium_confidence_threshold: 0.5  # Medium: cap to 1.5x last NTP
    medium_confidence_multiplier: 1.5
    low_confidence_multiplier: 1.2  # Low: cap to 1.2x last NTP

  # Optional enhancements (Phase 2)
  prediction_damping_factor: 0.85  # Apply 85% of prediction (optional)
  outlier_cleaning_enabled: false  # Clean contamination periodically (optional)
  outlier_cleaning_interval: 100  # Clean every N measurements (optional)
```

## Expected Improvements

With these changes:
- **Hours 0-2**: Should remain good (1.09x → ~1.05x)
- **Hours 2-4**: Should remain good (1.11x → ~1.08x)
- **Hours 4-6**: Should prevent divergence (1.84x → ~1.15x)
- **Hours 6-8**: Should prevent catastrophic failure (3.28x → ~1.25x)

Target: **Maintain <1.5x ratio throughout 8-hour test**

## Testing Plan

1. Apply Phase 1 fixes
2. Launch 8-hour overnight test
3. Monitor prediction magnitude over time
4. Compare to current baseline (21,400ms ChronoTick vs 18,900ms System)
5. Success criteria: ChronoTick < 15,000ms accumulated error (better than system clock)

## Files to Modify

1. `server/src/chronotick/inference/real_data_pipeline.py` - Core changes
2. `configs/config_test_drift_aware.yaml` - Add new parameters
3. `scripts/test_3min_validation_v2.py` - Might need update for new config
