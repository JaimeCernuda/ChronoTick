# NTP Correction v2: Drift-Aware Design

## Problem Decomposition

### The Fundamental Challenge

When NTP measurement arrives at time `t_ntp`, we observe:
```
offset_NTP = true_offset(t_ntp)
```

Our ML prediction at that time was:
```
offset_ML_predicted = offset_ML + drift_ML × (t_ntp - t_pred)
```

The innovation (prediction error) is:
```
innovation = offset_NTP - offset_ML_predicted
```

This innovation could be due to:
1. **Initial offset error**: `offset_ML` was wrong
2. **Drift rate error**: `drift_ML` was wrong  
3. **Both** (most likely)

**The Challenge**: One observation, two unknowns → infinite solutions!

We need a principled way to attribute the error to offset vs drift.

## Design Principles

### 1. Uncertainty-Based Attribution

Use the ML model's own uncertainty estimates to decide how to distribute the correction:

```python
# ML provides these uncertainties
σ_offset = ml_correction.offset_uncertainty
σ_drift = ml_correction.drift_uncertainty

# After time Δt, the variance contributions are:
var_from_offset = σ_offset²
var_from_drift = (σ_drift × Δt)²
total_variance = var_from_offset + var_from_drift

# Relative importance (weights sum to 1)
w_offset = var_from_offset / total_variance
w_drift = var_from_drift / total_variance

# Apply corrections proportionally
offset_correction = w_offset × innovation
drift_correction = (w_drift × innovation) / Δt
```

**Rationale**: 
- If offset uncertainty dominates → attribute error mostly to offset
- If drift uncertainty × time dominates → attribute error mostly to drift
- This is **data-driven**, not arbitrary!

### 2. Confidence-Aware Blending

Only apply NTP correction when it actually improves uncertainty:

```python
# Calculate combined uncertainty after blending
σ_combined = 1 / sqrt(1/σ_ML² + 1/σ_NTP²)

# Only apply if blending reduces uncertainty
if σ_combined < σ_ML:
    apply_correction()
else:
    use_ML_only()
```

**Rationale**:
- Sometimes ML is better than NTP (33ms vs 42ms showed this!)
- Don't blindly blend - only when it helps

### 3. Time-Adaptive Drift Correction

Adjust drift correction strength based on time since NTP:

```python
# For recent NTP (small Δt), drift hasn't accumulated much
# For old NTP (large Δt), drift error dominates

if Δt < 60:  # Less than 1 minute
    drift_correction_factor = 0.3  # Conservative
elif Δt < 180:  # 1-3 minutes
    drift_correction_factor = 0.7  # Moderate
else:  # > 3 minutes
    drift_correction_factor = 1.0  # Full correction
```

**Rationale**:
- Don't over-correct drift based on short-term noise
- Trust drift corrections more when long time has elapsed

## Mathematical Formulation

### State-Space Model

**State**: `x = [offset, drift]`
**Observation**: `z = offset_NTP`

State evolution:
```
offset(t) = offset(t₀) + drift × Δt
drift(t) = drift(t₀) + ε_drift  (slow random walk)
```

Observation model:
```
z = offset_true + ε_measurement
```

### Simplified Kalman Update

Full Kalman filter formulation:

```python
# State: [offset, drift]
# Covariance: P = [[σ_offset², cov],
#                  [cov, σ_drift²]]

# Prediction (already done by ML)
x_pred = [offset_ML + drift_ML × Δt, drift_ML]
P_pred = [[σ_offset², 0],
          [0, σ_drift²]]  # Assuming no covariance

# Observation
z = offset_NTP
R = σ_NTP²

# Innovation
y = z - x_pred[0]

# Kalman gain
H = [1, 0]  # We observe offset only
S = H × P_pred × H^T + R = σ_offset² + σ_NTP²
K = P_pred × H^T / S

K_offset = σ_offset² / (σ_offset² + σ_NTP²)
K_drift = 0  # No direct observation of drift!

# But we can infer drift from innovation over time...
# If innovation is large and Δt is large → drift is wrong
K_drift_indirect = (σ_drift² × Δt) / (σ_offset² + σ_NTP² + σ_drift² × Δt²)
```

This gives us the optimal weights!

### Practical Implementation

```python
def compute_correction_weights(σ_offset, σ_drift, σ_NTP, Δt):
    """
    Compute optimal weights for offset and drift correction.
    
    Based on uncertainty propagation and Kalman filter principles.
    """
    # Variance from offset uncertainty
    var_offset = σ_offset²
    
    # Variance from drift uncertainty accumulated over time
    var_drift = (σ_drift × Δt)²
    
    # Total predicted variance
    var_total_pred = var_offset + var_drift
    
    # Observation variance
    var_obs = σ_NTP²
    
    # Combined variance after blending
    var_combined = (var_total_pred × var_obs) / (var_total_pred + var_obs)
    
    # Offset correction weight (Kalman gain for offset)
    w_offset = var_offset / (var_offset + var_obs)
    
    # Drift correction weight (indirect, based on time)
    if Δt > 0:
        # Attribute to drift proportionally to how much it contributed
        drift_contribution = var_drift / var_total_pred
        w_drift = drift_contribution × w_offset / Δt
    else:
        w_drift = 0
    
    return w_offset, w_drift, var_combined

def apply_drift_aware_correction(ml_correction, ntp_offset, ntp_uncertainty, current_time):
    """Apply NTP correction to both offset and drift."""
    
    # Time since NTP measurement
    Δt = current_time - last_ntp_time
    
    # Predict what ML thinks the offset should be at NTP time
    ml_offset_at_ntp = ml_correction.offset_correction + ml_correction.drift_rate × Δt_from_pred
    
    # Innovation (prediction error)
    innovation = ntp_offset - ml_offset_at_ntp
    
    # Compute optimal weights
    w_offset, w_drift, σ_combined = compute_correction_weights(
        ml_correction.offset_uncertainty,
        ml_correction.drift_uncertainty,
        ntp_uncertainty,
        Δt
    )
    
    # Only apply if it reduces uncertainty
    if σ_combined >= ml_correction.offset_uncertainty:
        logger.info("[NTP_CORRECTION] Skipping - NTP doesn't improve uncertainty")
        return ml_correction
    
    # Apply corrections
    corrected_offset = ml_correction.offset_correction + w_offset × innovation
    corrected_drift = ml_correction.drift_rate + w_drift × innovation
    
    # Uncertainty after correction
    corrected_uncertainty = sqrt(σ_combined)
    
    return CorrectionWithBounds(
        offset_correction=corrected_offset,
        drift_rate=corrected_drift,
        offset_uncertainty=corrected_uncertainty,
        drift_uncertainty=ml_correction.drift_uncertainty,  # Unchanged for now
        ...
    )
```

## Configuration Parameters

```yaml
ntp_correction:
  enabled: true
  method: drift_aware  # drift_aware, offset_only, kalman
  
  # Drift correction
  drift_correction_enabled: true
  min_time_for_drift_correction: 60  # seconds
  drift_correction_strength: 0.7  # 0-1, how aggressively to correct drift
  
  # Uncertainty management
  uncertainty_threshold: 1.5  # Only apply if σ_NTP < threshold × σ_ML
  min_confidence_gain: 0.1  # Require at least 10% uncertainty reduction
  
  # Timing
  max_ntp_age_seconds: 180  # Must be < NTP interval
  uncertainty_growth_rate: 0.00001  # 10 μs/s
```

## Expected Outcomes

### Best Case (optimistic)
- NTP provides accurate drift corrections
- Reduces error from 33ms to ~20ms
- Win rate stays at 100%

### Realistic Case
- Some improvement in long-term stability
- Error: 28-30ms (modest improvement)
- Win rate: 100%

### Worst Case (if NTP adds noise)
- No improvement over 33ms baseline
- Fall back to ML-only mode automatically
- Win rate: still near 100%

## Success Criteria

1. **Primary**: Beat 33ms baseline (even by 5ms is success)
2. **Secondary**: Maintain 100% win rate vs system clock
3. **Tertiary**: Lower variance in error over time

## Implementation Phases

### Phase 1: Fix NTP Tracking Bug
- Ensure pipeline NTP collector runs throughout test
- Verify last_ntp_time updates continuously
- Add logging for NTP state tracking

### Phase 2: Implement Drift-Aware Correction
- Add uncertainty-based weight computation
- Apply corrections to BOTH offset and drift
- Add confidence-aware gating

### Phase 3: Configuration
- Add drift correction parameters to config
- Make all thresholds tunable
- Add method selection (offset_only vs drift_aware)

### Phase 4: Validation
- Run 25-minute test with v2 correction
- Compare with 33ms baseline
- Analyze when correction helps vs hurts
