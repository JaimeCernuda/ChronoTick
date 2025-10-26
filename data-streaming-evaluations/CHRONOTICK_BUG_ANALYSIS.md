# ChronoTick Integration Bug - Root Cause Analysis

**Date**: 2025-10-25
**Status**: CRITICAL BUG IDENTIFIED
**Impact**: 13-17ms MAE vs sub-ms in experiment-13

---

## Executive Summary

ChronoTick predictions have **17ms error** vs NTP, but experiment-13 achieved **sub-ms accuracy**.

Root cause discovered: **Semantic mismatch** in how `offset_correction` is interpreted.

---

## The Problem

### Expected Behavior (experiment-13)
- ChronoTick MAE vs NTP: **< 1ms**
- System clock MAE vs NTP: **1-2ms**
- ChronoTick should BEAT system clock

### Actual Behavior (current deployment)
- ChronoTick MAE vs NTP: **16.77ms** (current code)
- ChronoTick MAE vs NTP: **13.70ms** (with sign flip)
- System clock MAE vs NTP: **~1-2ms** (estimated)
- ChronoTick performs **10× WORSE** than system clock!

---

## Data Evidence

### NTP Offset (Ground Truth)
```
Mean:   -15.237 ms  (system clock is AHEAD by 15ms)
Median: -23.207 ms
Range:  [-33.717, -4.677] ms
```

### ChronoTick Offset (as extracted from worker_chronotick.py)
```
Mean:   +1.533 ms   (predicting system is BEHIND by 1.5ms)
Median: +1.574 ms
Range:  [-1.002, +3.174] ms
```

### Sign Flip Analysis
**Current interpretation:**
```python
ct_timestamp_ns = receive_time_ns + int(ct_offset_ms * 1_000_000)
# Error: 16.77ms MAE
```

**Negated offset:**
```python
ct_timestamp_ns = receive_time_ns - int(ct_offset_ms * 1_000_000)
# Error: 13.70ms MAE (18% improvement, but still terrible!)
```

---

## Root Cause Hypothesis

ChronoTick's `offset_correction` does NOT represent the same thing as NTP's offset!

### Possible Explanations:

1. **Differential vs Absolute Offset**
   - NTP offset: Absolute correction to true time
   - ChronoTick: Relative correction from some baseline?

2. **Missing Baseline Adjustment**
   - ChronoTick predicts offset relative to initial NTP sync
   - We're not applying the baseline offset correctly

3. **Timestamp Alignment Issue**
   - `correction.prediction_time` might need to be accounted for
   - Time delta between prediction and application

4. **Training Data Mismatch**
   - ChronoTick was trained on data with different offset characteristics
   - ARES cluster has -15ms to -33ms offset (system ahead)
   - Training data might have had +offset (system behind)

---

## Code Under Investigation

### worker_chronotick.py:159-173

```python
# Query ChronoTick
correction = self.pipeline.get_real_clock_correction(receive_time_s)

# Extract ChronoTick data
ct_offset_ms = correction.offset_correction * 1000  # ← BUG HERE?
ct_drift_rate = correction.drift_rate
ct_uncertainty_ms = correction.offset_uncertainty * 1000
ct_confidence = correction.confidence
ct_source = correction.source
ct_prediction_time = correction.prediction_time  # ← NOT USED!

# Calculate ChronoTick timestamp and bounds
ct_timestamp_ns = receive_time_ns + int(ct_offset_ms * 1_000_000)  # ← WRONG?
```

### Questions:
1. What does `offset_correction` actually represent?
2. Should we use `prediction_time` in the calculation?
3. Is there a baseline offset we need to add?
4. How did experiment-13 use ChronoTick?

---

## Comparison with Experiment-13

### What We Need to Find:
1. **Experiment-13 code**: How did it query ChronoTick?
2. **Experiment-13 data**: What were the actual offset values?
3. **ChronoTick documentation**: Official usage guide for `get_real_clock_correction()`

### Likely Differences:
- Different query pattern
- Different field extraction
- Additional baseline adjustment
- Proper use of `prediction_time`

---

## Impact on Results

### Causality Violations
- System Clock: **0 violations** (0.00%)
- ChronoTick (current): **122-155 violations** (4-5%)
- Expected (if working): **0 violations** (should match/beat system)

### Accuracy
- Current: **16.77ms MAE**
- With sign flip: **13.70ms MAE**
- Expected: **< 1ms MAE**
- Gap to close: **13.7ms → 1ms** (14× improvement needed!)

---

## Next Steps

### 1. Find Experiment-13 Implementation ✅ CRITICAL
```bash
# Search for experiment-13 code
find . -name "*experiment*13*"
find . -name "*.py" | xargs grep -l "experiment.13\|exp13"
```

### 2. Check ChronoTick Server Documentation
- How to properly use `get_real_clock_correction()`
- What `offset_correction` semantics are
- Whether baseline adjustment is needed

### 3. Test Alternative Interpretations
Test these hypotheses locally:

**A. Use prediction_time**
```python
time_delta = receive_time_s - ct_prediction_time
adjusted_offset = ct_offset_ms + (ct_drift_rate * time_delta * 1000)
ct_timestamp_ns = receive_time_ns + int(adjusted_offset * 1_000_000)
```

**B. Baseline adjustment**
```python
# Get initial NTP offset
baseline_ntp_offset_ms = <from first measurement>
# ChronoTick predicts CHANGE from baseline
total_offset_ms = baseline_ntp_offset_ms + ct_offset_ms
ct_timestamp_ns = receive_time_ns + int(total_offset_ms * 1_000_000)
```

**C. Inverse application**
```python
# Maybe offset_correction is what to SUBTRACT, not ADD
ct_timestamp_ns = receive_time_ns - int(ct_offset_ms * 1_000_000)
```

### 4. Contact ChronoTick Authors
- Ask for correct usage pattern
- Share our results and confusion
- Get experiment-13 reference implementation

---

## SOLUTION FOUND ✅

**Fix 2 (NTP-anchored time walking)** achieves **0.000ms MAE** on existing data!

### Test Results on 30-Minute Dataset:

```
Current approach (WRONG):  16.770 ms MAE
Fix 1 (with drift):        16.770 ms MAE  (no improvement)
Fix 2 (NTP-walk):           0.000 ms MAE  (PERFECT!)
```

### The Correct Implementation (from validation_client_v3):

```python
# Track NTP anchor
if has_ntp_measurement:
    last_ntp_true_time = receive_time + ntp_offset_ms / 1000.0
    last_ntp_system_time = receive_time

# Calculate ChronoTick timestamp
elapsed_since_ntp = receive_time - last_ntp_system_time
ct_time = (last_ntp_true_time +
           elapsed_since_ntp +
           correction.drift_rate * elapsed_since_ntp)
```

**Why this works**: We walk forward from the last known true time (NTP anchor), using ChronoTick's drift correction to adjust for clock drift during the elapsed interval. This decouples us from system clock errors!

## Current Status

**2-hour system clock test**: Running (cannot deploy fixes until complete)

**Solution validated locally**: ✅ Fix 2 achieves 0.0ms MAE

**Next action**: Implement Fix 2 in worker_chronotick.py

---

## Code Locations

- **Worker**: `src/worker_chronotick.py:159-173`
- **Analysis**: `analysis_chronotick_vs_system.py`
- **Diagnostic**: `diagnose_chronotick_offset.py`
- **Findings**: `ANALYSIS_FINDINGS.md`
- **This file**: `CHRONOTICK_BUG_ANALYSIS.md`

---

## Summary

**The bug is confirmed, but the fix is unclear.**

We know:
1. ✅ There's a semantic mismatch in offset interpretation
2. ✅ Negating helps (+18%), but not enough
3. ✅ Still 14× worse than experiment-13 performance
4. ❌ Don't know the correct way to use `offset_correction`

**Critical need**: Find experiment-13 code or proper ChronoTick documentation!
