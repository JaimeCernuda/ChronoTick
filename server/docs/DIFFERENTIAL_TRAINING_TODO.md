# Differential Training TODO

## Overview

Implement differential training to predict clock **drift dynamics** (rate of change) instead of absolute offsets. This approach is inspired by chrony's method of tracking both clock skew and drift acceleration.

## Current State (Offset Normalization)

**What we do now:**
- Subtract recent NTP baseline from all measurements
- Train model on **residuals** (small deviations from baseline)
- Model learns: "Offset is currently +0.2ms above baseline"

**Benefits:**
- Removes absolute bias
- Model focuses on relative changes
- Zero-centered training data

**Limitations:**
- Still training on offset values (position)
- Doesn't explicitly model drift dynamics (velocity/acceleration)
- Misses underlying clock behavior patterns

## Proposed Solution: Differential Training

### Chrony's Approach

Chrony tracks three components:
1. **Clock Skew** - Base offset between system clock and true time
2. **Clock Drift** - Rate of change of skew (μs/second)
3. **Drift Acceleration** - Rate of change of drift (μs/second²)

Model:
```
offset(t) = skew + drift * (t - t₀) + 0.5 * acceleration * (t - t₀)²
```

### Our Implementation Plan

Instead of training on `offset` values, train on **derivatives**:

#### Phase 1: First-Order Differential (Drift Rate)

**Training Data:**
```python
# Current (normalization):
training_data = [offset₁, offset₂, offset₃, ...]  # Position

# Differential Phase 1:
drift_rate = (offset[t+1] - offset[t]) / Δt  # Velocity
training_data = [drift₁, drift₂, drift₃, ...]
```

**Model learns:**
- "Clock is drifting at +2μs/s"
- "Drift is accelerating to +3μs/s"

**Prediction:**
- Model outputs: drift rate predictions
- Integrate to get offsets: `offset = offset₀ + ∫ drift dt`

#### Phase 2: Second-Order Differential (Drift Acceleration)

**Training Data:**
```python
# Second derivative:
drift_accel = (drift[t+1] - drift[t]) / Δt  # Acceleration
training_data = [accel₁, accel₂, accel₃, ...]
```

**Model learns:**
- "Drift is accelerating at +0.5μs/s²"
- Captures nonlinear clock behavior

**Prediction:**
- Model outputs: acceleration predictions
- Double integrate to get offsets

### Implementation Details

#### File Modifications

**1. `real_data_pipeline.py`**

Add method to compute derivatives:
```python
def get_differential_measurements(self, order: int = 1) -> Tuple[List[Tuple[float, float]], Optional[float]]:
    """
    Get measurements as derivatives (drift, acceleration) instead of offsets.

    Args:
        order: Derivative order (1=drift, 2=acceleration)

    Returns:
        (measurements, normalization_bias)
    """
    # Get base measurements
    measurements, bias = self.get_recent_measurements(normalize=True)

    if order == 0:
        return measurements, bias

    # Compute first derivative (drift rate)
    timestamps = [m[0] for m in measurements]
    offsets = [m[1] for m in measurements]

    drift_rates = []
    for i in range(len(offsets) - 1):
        dt = timestamps[i+1] - timestamps[i]
        if dt > 0:
            drift = (offsets[i+1] - offsets[i]) / dt
            drift_rates.append((timestamps[i], drift))

    if order == 1:
        return drift_rates, None

    # Compute second derivative (acceleration)
    if order == 2:
        # ... similar for acceleration
        pass
```

**2. `tsfm_model_wrapper.py`**

Add integration logic:
```python
def _integrate_predictions(self,
                          drift_predictions: np.ndarray,
                          initial_offset: float) -> np.ndarray:
    """
    Integrate drift rate predictions to get offset predictions.

    Args:
        drift_predictions: Array of drift rates (μs/s)
        initial_offset: Starting offset

    Returns:
        Array of offset predictions
    """
    offsets = [initial_offset]
    for drift in drift_predictions:
        next_offset = offsets[-1] + drift * 1.0  # Δt = 1 second
        offsets.append(next_offset)

    return np.array(offsets[1:])  # Skip initial
```

**3. Configuration**

Add to `config.yaml`:
```yaml
model_training:
  differential_mode: false  # Enable differential training
  derivative_order: 1       # 1=drift, 2=acceleration
  integration_method: "euler"  # "euler", "trapezoidal", "runge_kutta"
```

### Testing Strategy

1. **Unit Tests:**
   - Test derivative computation
   - Test integration accuracy
   - Test with synthetic data (known drift patterns)

2. **A/B Comparison:**
   - Run both modes side-by-side
   - Compare accuracy vs NTP ground truth
   - Measure convergence speed

3. **Drift Pattern Tests:**
   - Linear drift: Constant rate (easy)
   - Quadratic drift: Accelerating (medium)
   - Temperature-dependent drift: Complex (hard)

### Expected Benefits

1. **Better Drift Tracking:**
   - Model explicitly learns drift patterns
   - Can predict acceleration/deceleration
   - Handles nonlinear clock behavior

2. **Faster Convergence:**
   - Drift rate changes more slowly than offsets
   - More stable training signal
   - Better extrapolation

3. **Physical Intuition:**
   - Matches how clocks actually behave
   - Drift is driven by physical factors (temperature, aging)
   - More interpretable predictions

### Potential Challenges

1. **Numerical Stability:**
   - Integration can accumulate errors
   - Need careful choice of integration method
   - May need adaptive step sizes

2. **Noise Amplification:**
   - Derivatives amplify measurement noise
   - May need more aggressive filtering
   - Consider Savitzky-Golay filter

3. **Boundary Conditions:**
   - Need accurate initial offset for integration
   - Drift changes can be abrupt (temperature shifts)
   - May need hybrid approach

### Timeline

- **Week 1:** Implement first-order differential (drift rate only)
- **Week 2:** Test and compare with normalization approach
- **Week 3:** Implement second-order (acceleration) if needed
- **Week 4:** Tune and optimize integration method

### References

- **Chrony Documentation:** https://chrony.tuxfamily.org/doc/4.3/chrony.conf.html
- **Clock Drift Papers:**
  - "Clock Synchronization for Wireless Sensor Networks" (Sundararaman et al.)
  - "Modeling and Analysis of Clock Drift in Network Time Synchronization" (Lv et al.)

### Related Work

- PTP (Precision Time Protocol) uses drift compensation
- Kalman filters for clock synchronization model drift explicitly
- Chrony's adaptive drift algorithm

---

## Decision Log

**2025-10-17:** Implemented offset normalization as Phase 0
- Removes absolute bias
- Trains on residuals
- Simpler than differential, test this first

**Future:** Implement differential training if normalization insufficient
- Start with first-order (drift)
- Add second-order (acceleration) if needed
- Measure improvement empirically

---

## Contact

For questions or implementation discussion, refer to:
- `real_data_pipeline.py` - Dataset management
- `tsfm_model_wrapper.py` - Model interface
- `chronotick_inference_engine.py` - Prediction logic
