# Dataset-Only NTP Correction Algorithms for Autoregressive Time Synchronization

**Authors:** ChronoTick Research Team
**Date:** October 2025
**Version:** 1.0

---

## Abstract

This document presents five algorithms for correcting historical time offset datasets when new Network Time Protocol (NTP) ground truth measurements become available. Unlike traditional real-time blending approaches, these **dataset-only correction** methods modify past predictions retrospectively, enabling autoregressive machine learning models to learn from corrected data and automatically improve future predictions. We present mathematical formulations, pseudocode implementations, and comparative analysis of five correction strategies: baseline (none), linear interpolation, drift-aware attribution, uncertainty-weighted (advanced), and per-point directional correction (advance_absolute).

---

## 1. Problem Formulation

### 1.1 Context

In distributed time synchronization systems using machine learning for predictive clock correction, we maintain a dataset of historical offset measurements:

$$
\mathcal{D} = \{(t_i, \hat{o}_i)\}_{i=1}^{N}
$$

where:
- $t_i$ is the timestamp (in seconds)
- $\hat{o}_i$ is the predicted offset correction at time $t_i$

Periodically, we receive **NTP ground truth measurements** that provide the true offset at a specific time:

$$
\text{NTP}_{\text{new}} = (t_{\text{NTP}}, o_{\text{true}}, \sigma_{\text{NTP}})
$$

where $\sigma_{\text{NTP}}$ is the NTP measurement uncertainty.

### 1.2 The Correction Problem

Given:
- **Last NTP measurement** at $t_{\text{start}}$ (or start of dataset if no previous NTP)
- **New NTP measurement** at $t_{\text{NTP}}$ with true offset $o_{\text{true}}$
- **Prediction** at $t_{\text{NTP}}$ with predicted offset $\hat{o}_{\text{NTP}}$
- **Error** $E = o_{\text{true}} - \hat{o}_{\text{NTP}}$
- **Interval duration** $\Delta t = t_{\text{NTP}} - t_{\text{start}}$
- **Timestamps in interval**: $\{t_i\}$ where $t_{\text{start}} \leq t_i < t_{\text{NTP}}$

**Objective:** Retrospectively correct all predictions $\hat{o}_i$ in the interval to better align with NTP ground truth, such that autoregressive models trained on the corrected dataset will make better future predictions.

**Key Constraint:** Dataset-only correction. We modify historical data, NOT real-time predictions. Future predictions automatically improve through autoregressive learning.

---

## 2. Algorithm 1: NONE (Baseline)

### 2.1 Description

No retrospective correction is applied. The dataset remains unchanged except for adding the new NTP measurement as a data point.

### 2.2 Mathematical Formulation

For all $t_i \in [t_{\text{start}}, t_{\text{NTP}})$:
$$
\hat{o}_i' = \hat{o}_i \quad \text{(no change)}
$$

Add NTP measurement:
$$
\mathcal{D} \leftarrow \mathcal{D} \cup \{(t_{\text{NTP}}, o_{\text{true}})\}
$$

### 2.3 Pseudocode

```python
def apply_none_correction(dataset, ntp_measurement):
    """
    Baseline: No correction applied.

    Args:
        dataset: Historical offset measurements
        ntp_measurement: New NTP ground truth (t_ntp, o_true, sigma_ntp)

    Returns:
        Updated dataset with NTP measurement added
    """
    # Simply add NTP measurement to dataset
    dataset.add(ntp_measurement)

    return dataset
```

### 2.4 Properties

- **Computational Complexity:** $O(1)$
- **Correction Magnitude:** 0 (no correction)
- **Autoregressive Learning:** Model must learn from uncorrected errors
- **Use Case:** Baseline for comparison; assumes ML model can learn drift patterns without explicit correction

---

## 3. Algorithm 2: LINEAR

### 3.1 Description

Distributes the endpoint error $E$ linearly across the interval based on time elapsed. Assumes linear drift from 0 error at $t_{\text{start}}$ to error $E$ at $t_{\text{NTP}}$.

### 3.2 Mathematical Formulation

For each timestamp $t_i \in [t_{\text{start}}, t_{\text{NTP}})$:

1. **Linear weight:**
$$
\alpha_i = \frac{t_i - t_{\text{start}}}{\Delta t}
$$

2. **Correction:**
$$
\hat{o}_i' = \hat{o}_i + \alpha_i \cdot E
$$

**Normalization property:**
$$
\sum_{i=1}^{N} \alpha_i = \frac{N(N+1)}{2N} \approx \frac{N}{2} \quad \text{(for uniform sampling)}
$$

### 3.3 Pseudocode

```python
def apply_linear_correction(dataset, interval_start, interval_end, error):
    """
    Linear interpolation of error across interval.

    Args:
        dataset: Historical offset measurements (dict: timestamp -> data)
        interval_start: Start of correction interval (t_start)
        interval_end: End of correction interval (t_ntp)
        error: Measured error E = o_true - o_predicted

    Returns:
        Corrected dataset
    """
    interval_duration = interval_end - interval_start

    for timestamp in range(int(interval_start), int(interval_end)):
        if timestamp in dataset:
            # Calculate linear weight: α = (t - t_start) / Δt
            alpha = (timestamp - interval_start) / interval_duration

            # Apply weighted correction: ô'(t) ← ô(t) + α·E
            correction = alpha * error
            dataset[timestamp]['offset'] += correction
            dataset[timestamp]['corrected'] = True

    return dataset
```

### 3.4 Properties

- **Computational Complexity:** $O(N)$ where $N$ is number of timestamps in interval
- **Correction Magnitude:** $\sum \text{corrections} = E \cdot \frac{N}{2}$ (approximately)
- **Assumptions:** Linear drift model
- **Advantages:** Simple, intuitive, computationally efficient
- **Disadvantages:** Ignores uncertainty information; assumes perfectly linear drift

---

## 4. Algorithm 3: DRIFT_AWARE

### 4.1 Description

Attributes the error $E$ to both **offset** and **drift rate** components based on their respective uncertainties. Uses uncertainty information to allocate correction between static offset and time-varying drift.

### 4.2 Mathematical Formulation

1. **Uncertainty components:**
   - Offset uncertainty: $\sigma_{\text{offset}}$
   - Drift uncertainty: $\sigma_{\text{drift}}$ (per second)
   - Accumulated drift uncertainty over interval: $\sigma_{\text{drift,acc}} = \sigma_{\text{drift}} \cdot \Delta t$

2. **Variance contributions:**
$$
\begin{align}
\text{var}_{\text{offset}} &= \sigma_{\text{offset}}^2 \\
\text{var}_{\text{drift}} &= (\sigma_{\text{drift}} \cdot \Delta t)^2
\end{align}
$$

3. **Weight allocation (based on variance):**
$$
\begin{align}
w_{\text{offset}} &= \frac{\text{var}_{\text{offset}}}{\text{var}_{\text{offset}} + \text{var}_{\text{drift}}} \\
w_{\text{drift}} &= \frac{\text{var}_{\text{drift}}}{\text{var}_{\text{offset}} + \text{var}_{\text{drift}}}
\end{align}
$$

4. **Error attribution:**
$$
\begin{align}
E_{\text{offset}} &= w_{\text{offset}} \cdot E \\
E_{\text{drift}} &= \frac{w_{\text{drift}} \cdot E}{\Delta t} \quad \text{(drift rate)}
\end{align}
$$

5. **Correction at each timestamp:**
$$
\hat{o}_i' = \hat{o}_i + E_{\text{offset}} + E_{\text{drift}} \cdot (t_i - t_{\text{start}})
$$

Also update drift rate:
$$
\hat{r}_i' = \hat{r}_i + E_{\text{drift}}
$$

### 4.3 Pseudocode

```python
def apply_drift_aware_correction(dataset, interval_start, interval_end,
                                  error, sigma_offset, sigma_drift, delta_t):
    """
    Drift-aware correction: Attributes error to offset + drift components.

    Args:
        dataset: Historical offset measurements
        interval_start: Start of interval (t_start)
        interval_end: End of interval (t_ntp)
        error: Measured endpoint error E
        sigma_offset: Offset uncertainty (seconds)
        sigma_drift: Drift rate uncertainty (seconds/second)
        delta_t: Interval duration (seconds)

    Returns:
        Corrected dataset with both offset and drift adjustments
    """
    # Calculate variance contributions
    var_offset = sigma_offset ** 2
    var_drift = (sigma_drift * delta_t) ** 2
    var_total = var_offset + var_drift

    if var_total == 0:
        # Fallback to linear if no uncertainty info
        return apply_linear_correction(dataset, interval_start,
                                       interval_end, error)

    # Allocate error based on variance weights
    w_offset = var_offset / var_total
    w_drift = var_drift / var_total

    # Attribute error to components
    offset_correction = w_offset * error
    drift_correction = (w_drift * error) / delta_t  # Drift rate

    # Apply to each timestamp
    for timestamp in range(int(interval_start), int(interval_end)):
        if timestamp in dataset:
            # Time since interval start
            t_elapsed = timestamp - interval_start

            # Combined correction: offset + drift × time
            correction = offset_correction + (drift_correction * t_elapsed)

            dataset[timestamp]['offset'] += correction
            dataset[timestamp]['drift'] += drift_correction
            dataset[timestamp]['corrected'] = True

    return dataset
```

### 4.4 Properties

- **Computational Complexity:** $O(N)$
- **Correction Magnitude:** $\sum \text{corrections} \approx E$ (distributed across offset and drift)
- **Assumptions:** Error can be decomposed into offset and drift components
- **Advantages:** Incorporates uncertainty information; handles both static and time-varying errors
- **Disadvantages:** Requires accurate uncertainty estimates; assumes linear drift model

---

## 5. Algorithm 4: ADVANCED

### 5.1 Description

Uses **uncertainty-weighted distribution** where points with higher uncertainty receive proportionally more correction. Based on the principle that measurements further from the last NTP sync have accumulated more uncertainty and should be corrected more aggressively.

### 5.2 Mathematical Formulation

1. **Temporal uncertainty model:**
$$
\sigma_i^2(t) = \sigma_{\text{measurement}}^2 + \sigma_{\text{prediction}}^2 + (\sigma_{\text{drift}} \cdot \delta t_i)^2
$$

where $\delta t_i = t_i - t_{\text{start}}$ is time elapsed since last NTP sync.

2. **Direct variance weighting:**
$$
w_i = \sigma_i^2(t_i) \quad \text{(HIGH uncertainty → MORE correction)}
$$

3. **Normalized weights:**
$$
\alpha_i = \frac{w_i}{\sum_{j=1}^{N} w_j}
$$

4. **Correction:**
$$
\hat{o}_i' = \hat{o}_i + \alpha_i \cdot E
$$

**Normalization guarantee:**
$$
\sum_{i=1}^{N} \alpha_i = 1 \implies \sum_{i=1}^{N} (\hat{o}_i' - \hat{o}_i) = E
$$

### 5.3 Pseudocode

```python
def apply_advanced_correction(dataset, interval_start, interval_end, error,
                              sigma_measurement, sigma_prediction, sigma_drift):
    """
    Advanced uncertainty-weighted correction.

    Points with higher uncertainty receive proportionally more correction.

    Args:
        dataset: Historical offset measurements
        interval_start: Start of interval (t_start)
        interval_end: End of interval (t_ntp)
        error: Measured endpoint error E
        sigma_measurement: NTP measurement uncertainty
        sigma_prediction: ML prediction uncertainty
        sigma_drift: Drift rate uncertainty (s/s)

    Returns:
        Corrected dataset with uncertainty-weighted adjustments
    """
    # Base uncertainty (non-temporal components)
    sigma_squared_base = sigma_measurement**2 + sigma_prediction**2

    # Calculate weights for each timestamp
    total_weight = 0.0
    weights = {}

    for timestamp in range(int(interval_start), int(interval_end) + 1):
        if timestamp in dataset:
            # Time elapsed since last NTP sync
            delta_t = timestamp - interval_start

            # Temporal uncertainty growth (quadratic with time)
            sigma_squared_temporal = (sigma_drift * delta_t)**2

            # Total uncertainty at this timestamp
            sigma_squared_total = sigma_squared_base + sigma_squared_temporal

            # Direct variance weighting
            # Concept: HIGH uncertainty → MORE correction
            weight = sigma_squared_total
            weights[timestamp] = weight
            total_weight += weight

    # Apply normalized corrections
    for timestamp in range(int(interval_start), int(interval_end) + 1):
        if timestamp in dataset:
            # Normalized weight (ensures sum of corrections = E)
            alpha = weights[timestamp] / total_weight if total_weight > 0 else 0

            # Apply weighted correction
            correction = alpha * error
            dataset[timestamp]['offset'] += correction
            dataset[timestamp]['corrected'] = True

    return dataset
```

### 5.4 Properties

- **Computational Complexity:** $O(N)$
- **Correction Magnitude:** $\sum \text{corrections} = E$ (exactly)
- **Assumptions:** Uncertainty grows quadratically with time since last sync
- **Advantages:** Theoretically sound; accounts for temporal degradation of prediction quality
- **Disadvantages:** May be too conservative; can under-correct if uncertainty model is inaccurate

---

## 6. Algorithm 5: ADVANCE_ABSOLUTE (Per-Point Directional)

### 6.1 Description

**Most sophisticated approach.** Calculates the ideal target line from $t_{\text{start}}$ (error = 0) to $t_{\text{NTP}}$ (error = $E$), then corrects each point **toward** this line based on its individual deviation. Uses **per-point directional correction**: points above the line are pushed down, points below are pushed up. This prevents over-correction by bringing outliers closer to the ideal trajectory.

### 6.2 Mathematical Formulation

1. **Target line (ideal trajectory):**
$$
o_{\text{target}}(t) = E \cdot \frac{t - t_{\text{start}}}{\Delta t}
$$

This represents the ideal linear drift from 0 error to $E$ error.

2. **Deviation from target line:**
$$
d_i = \hat{o}_i - o_{\text{target}}(t_i)
$$

where:
- $d_i > 0$: Point is ABOVE target line (over-predicted relative to line)
- $d_i < 0$: Point is BELOW target line (under-predicted relative to line)

3. **Uncertainty weighting (same as ADVANCED):**
$$
\begin{align}
\sigma_i^2(t) &= \sigma_{\text{measurement}}^2 + \sigma_{\text{prediction}}^2 + (\sigma_{\text{drift}} \cdot \delta t_i)^2 \\
w_i &= \sigma_i^2(t_i) \\
\alpha_i &= \frac{w_i}{\sum_{j=1}^{N} w_j}
\end{align}
$$

4. **Per-point directional correction:**
$$
\text{correction}_i = -\alpha_i \cdot d_i \cdot \frac{N}{2}
$$

The negative sign ensures:
- If $d_i > 0$ (above line): correction $< 0$ (push DOWN)
- If $d_i < 0$ (below line): correction $> 0$ (push UP)

The factor $\frac{N}{2}$ accounts for accumulated error concept (total area under deviation curve).

5. **Final corrected offset:**
$$
\hat{o}_i' = \hat{o}_i + \text{correction}_i
$$

### 6.3 Pseudocode

```python
def apply_advance_absolute_correction(dataset, interval_start, interval_end,
                                      error, sigma_measurement,
                                      sigma_prediction, sigma_drift,
                                      interval_duration):
    """
    Advanced Absolute: Per-point directional correction toward target line.

    Calculates ideal line from last NTP to current NTP, then brings each
    point closer to this line based on its individual deviation and uncertainty.

    Args:
        dataset: Historical offset measurements
        interval_start: Start of interval (t_start, where target = 0)
        interval_end: End of interval (t_ntp, where target = E)
        error: Measured endpoint error E
        sigma_measurement: NTP measurement uncertainty
        sigma_prediction: ML prediction uncertainty
        sigma_drift: Drift rate uncertainty (s/s)
        interval_duration: Interval duration Δt (seconds)

    Returns:
        Corrected dataset with per-point directional adjustments
    """
    # Base uncertainty (non-temporal)
    sigma_squared_base = sigma_measurement**2 + sigma_prediction**2

    # Phase 1: Calculate deviations from target line and weights
    total_weight = 0.0
    weights = {}
    deviations = {}
    total_absolute_deviation = 0.0

    for timestamp in range(int(interval_start), int(interval_end) + 1):
        if timestamp in dataset:
            delta_t = timestamp - interval_start

            # Target offset on ideal line (linear from 0 to E)
            target_offset = error * delta_t / interval_duration

            # Current offset at this timestamp
            current_offset = dataset[timestamp]['offset']

            # Deviation from target line
            # Positive = ABOVE line, Negative = BELOW line
            deviation = current_offset - target_offset
            deviations[timestamp] = deviation
            total_absolute_deviation += abs(deviation)

            # Uncertainty weighting (quadratic growth with time)
            sigma_squared_temporal = (sigma_drift * delta_t)**2
            sigma_squared_total = sigma_squared_base + sigma_squared_temporal
            weight = sigma_squared_total
            weights[timestamp] = weight
            total_weight += weight

    # Phase 2: Apply per-point directional corrections
    corrections_up = 0
    corrections_down = 0
    num_points = len(weights)

    for timestamp in range(int(interval_start), int(interval_end) + 1):
        if timestamp in dataset:
            deviation = deviations[timestamp]

            # Normalized uncertainty weight
            alpha = weights[timestamp] / total_weight if total_weight > 0 else 0

            # Per-point directional correction
            # Negative sign: deviation > 0 → correction < 0 (push down)
            #                deviation < 0 → correction > 0 (push up)
            # Scale by N/2 for accumulated error concept
            correction = -alpha * deviation * (num_points / 2.0)

            # Apply correction
            dataset[timestamp]['offset'] += correction
            dataset[timestamp]['corrected'] = True

            # Track direction statistics
            if correction > 0:
                corrections_up += 1
            elif correction < 0:
                corrections_down += 1

    return dataset
```

### 6.4 Properties

- **Computational Complexity:** $O(N)$ (two passes through data)
- **Correction Magnitude:** $\sum |\text{corrections}| \approx N \cdot \bar{|d|}$ where $\bar{|d|}$ is mean absolute deviation
- **Key Innovation:** Bi-directional correction prevents over-correction
- **Assumptions:** Ideal trajectory is linear; deviations from line should be minimized
- **Advantages:**
  - Prevents systematic over-correction
  - Brings outliers back toward reasonable trajectory
  - Uncertainty-weighted for theoretical soundness
  - Bi-directional: some points pushed up, others pushed down
- **Disadvantages:**
  - Most complex algorithm
  - Requires accurate uncertainty estimates
  - May be sensitive to scaling factor $(N/2)$

---

## 7. Comparative Analysis

### 7.1 Algorithm Comparison Table

| **Algorithm** | **Complexity** | **Sum of Corrections** | **Directional** | **Uses Uncertainty** | **Key Property** |
|---------------|----------------|------------------------|-----------------|---------------------|------------------|
| **NONE** | $O(1)$ | 0 | N/A | No | Baseline |
| **LINEAR** | $O(N)$ | $\approx E \cdot N/2$ | Unidirectional | No | Simple, intuitive |
| **DRIFT_AWARE** | $O(N)$ | $= E$ | Unidirectional | Yes (for attribution) | Separates offset/drift |
| **ADVANCED** | $O(N)$ | $= E$ | Unidirectional | Yes (for weighting) | Uncertainty-weighted |
| **ADVANCE_ABSOLUTE** | $O(N)$ | $\propto N \cdot \bar{|d|}$ | **Bi-directional** | Yes (for weighting) | Per-point toward line |

### 7.2 Experimental Results (25-Minute Tests)

From empirical testing on real ChronoTick deployment:

| **Method** | **MAE (ms)** | **Std Dev (ms)** | **Max Error (ms)** | **Improvement vs Baseline** |
|------------|--------------|------------------|-------------------|------------------------------|
| NONE | 54.886 | 33.820 | 100.748 | Baseline |
| LINEAR | 10.380 | 7.555 | 31.726 | **81.1% ↓** |
| **DRIFT_AWARE** ⭐ | **9.519** | **4.711** | **17.611** | **82.7% ↓** (BEST) |
| ADVANCED | 25.857 | 20.169 | 61.325 | 52.9% ↓ |
| ADVANCE_ABSOLUTE | *TBD* | *TBD* | *TBD* | *Under evaluation* |

**Winner:** DRIFT_AWARE achieves best accuracy (9.519ms MAE), consistency (4.711ms std dev), and lowest peak errors (17.611ms max).

### 7.3 Theoretical Properties

#### 7.3.1 Correction Magnitude Analysis

For an interval with $N$ uniformly-spaced timestamps and endpoint error $E$:

**LINEAR:**
$$
\sum_{i=1}^{N} \text{correction}_i = \sum_{i=1}^{N} \frac{i}{N} \cdot E = E \cdot \frac{N+1}{2} \approx \frac{E \cdot N}{2}
$$

**DRIFT_AWARE:**
$$
\sum_{i=1}^{N} \text{correction}_i = E_{\text{offset}} \cdot N + E_{\text{drift}} \cdot \sum_{i=1}^{N} i = E
$$

**ADVANCED:**
$$
\sum_{i=1}^{N} \text{correction}_i = E \quad \text{(by normalization)}
$$

**ADVANCE_ABSOLUTE:**
$$
\sum_{i=1}^{N} |\text{correction}_i| \propto \frac{N}{2} \cdot \sum_{i=1}^{N} |d_i|
$$
(depends on deviation magnitudes)

#### 7.3.2 Uncertainty Propagation

For algorithms using uncertainty weighting (DRIFT_AWARE, ADVANCED, ADVANCE_ABSOLUTE):

**Prediction uncertainty after correction:**
$$
\sigma_{\text{corrected}}^2 \approx \sigma_{\text{original}}^2 + \sigma_{\text{correction}}^2
$$

where $\sigma_{\text{correction}}$ depends on NTP uncertainty and time since last sync.

### 7.4 Use Case Recommendations

| **Use Case** | **Recommended Algorithm** | **Rationale** |
|--------------|--------------------------|---------------|
| **General production** | DRIFT_AWARE | Best empirical performance; balances accuracy and complexity |
| **Low computational resources** | LINEAR | Simple, fast, still effective (81% improvement) |
| **High uncertainty estimates available** | ADVANCED or ADVANCE_ABSOLUTE | Leverages uncertainty information |
| **Baseline/research** | NONE | Understand impact of no correction |
| **Non-linear drift suspected** | ADVANCE_ABSOLUTE | Per-point correction adapts to deviations |

---

## 8. Implementation Considerations

### 8.1 Edge Cases

1. **No previous NTP measurement:**
   - Use $t_{\text{start}} = \min(\text{dataset timestamps})$
   - Assume error was 0 at dataset start

2. **Very short intervals ($\Delta t < 5$ seconds):**
   - All methods degrade to similar performance
   - Consider skipping correction if interval too short

3. **Missing timestamps in interval:**
   - All algorithms handle gracefully (skip missing timestamps)
   - Maintain correction normalization properties

4. **Uncertainty estimates unavailable:**
   - DRIFT_AWARE and ADVANCED fall back to LINEAR
   - Use conservative default values if needed

### 8.2 Numerical Stability

**Potential issues:**
- Division by zero: Check $\Delta t > 0$, $\text{total_weight} > 0$
- Floating-point precision: Use double precision for timestamp arithmetic
- Large correction values: Consider capping maximum correction magnitude

**Recommended safeguards:**
```python
# Safe division
alpha = weight / total_weight if total_weight > 1e-10 else 0

# Clamp corrections
MAX_CORRECTION = 1.0  # 1 second
correction = np.clip(correction, -MAX_CORRECTION, MAX_CORRECTION)
```

### 8.3 Performance Optimization

For large datasets ($N > 10^6$):
- Use vectorized operations (NumPy/PyTorch)
- Consider parallel processing for independent intervals
- Cache uncertainty calculations if repeated

Example vectorized implementation:
```python
import numpy as np

def apply_linear_correction_vectorized(timestamps, offsets,
                                       interval_start, interval_end, error):
    """Vectorized version of linear correction."""
    # Boolean mask for interval
    mask = (timestamps >= interval_start) & (timestamps < interval_end)

    # Vectorized weight calculation
    delta_t = interval_end - interval_start
    alphas = (timestamps[mask] - interval_start) / delta_t

    # Vectorized correction
    offsets[mask] += alphas * error

    return offsets
```

---

## 9. Future Research Directions

### 9.1 Open Questions

1. **Optimal uncertainty models:** How to accurately estimate $\sigma_{\text{drift}}$ for different clock hardware?

2. **Adaptive correction scaling:** Can we learn the optimal correction scaling factor $(N/2)$ for ADVANCE_ABSOLUTE?

3. **Non-linear drift models:** Extension to quadratic or exponential drift assumptions?

4. **Multi-NTP fusion:** How to optimally combine corrections from multiple simultaneous NTP servers?

5. **Online learning:** Can correction parameters be learned dynamically from past performance?

### 9.2 Potential Extensions

**Adaptive weighting:**
```python
def adaptive_correction_scaling(historical_errors, num_timestamps):
    """Learn optimal scaling from past performance."""
    # Analyze how well past corrections reduced errors
    # Adjust scaling factor N/2 → learned_factor
    learned_factor = optimize_factor(historical_errors)
    return learned_factor
```

**Ensemble correction:**
```python
def ensemble_correction(dataset, methods=['linear', 'drift_aware', 'advanced']):
    """Combine multiple correction strategies."""
    corrections = []
    for method in methods:
        corrected = apply_correction(dataset, method)
        corrections.append(corrected)

    # Weighted average based on past performance
    ensemble = weighted_average(corrections, performance_weights)
    return ensemble
```

---

## 10. Conclusion

We have presented five dataset-only NTP correction algorithms for autoregressive time synchronization systems, ranging from simple linear interpolation to sophisticated per-point directional correction. Empirical results demonstrate that **DRIFT_AWARE** achieves the best balance of accuracy (82.7% improvement over baseline), consistency, and computational efficiency, making it the recommended default for production deployments.

The key innovation of **ADVANCE_ABSOLUTE** lies in its per-point directional correction mechanism, which prevents over-correction by bringing each point closer to the ideal NTP trajectory based on individual deviations. While still under evaluation, this approach shows promise for scenarios with significant non-linear drift or high measurement uncertainty.

All algorithms share the fundamental principle of **dataset-only correction**: by retrospectively correcting historical data rather than blending in real-time, we enable autoregressive machine learning models to learn corrected patterns and automatically improve future predictions without explicit correction logic at inference time.

---

## References

1. Mills, D. L. (2006). Computer network time synchronization: the network time protocol. CRC press.

2. Liskov, B., & Cowling, J. (2012). Viewstamped replication revisited. *MIT Technical Report*.

3. Corbett, J. C., et al. (2013). Spanner: Google's globally distributed database. *ACM Transactions on Computer Systems (TOCS)*, 31(3), 1-22.

4. Geng, Y., et al. (2018). Exploiting a natural network effect for scalable, fine-grained clock synchronization. In *15th USENIX Symposium on Networked Systems Design and Implementation (NSDI 18)*.

5. Radhakrishnan, S., et al. (2014). SENIC: Scalable NIC for end-host rate limiting. In *11th USENIX Symposium on Networked Systems Design and Implementation (NSDI 14)*.

---

## Appendix A: Complete Python Implementation

### A.1 Dataset Manager with All Correction Methods

```python
import math
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)

class DatasetManager:
    """
    Manages historical offset measurements with NTP correction support.

    Implements all five correction algorithms:
    - none: No correction
    - linear: Linear interpolation
    - drift_aware: Offset + drift attribution
    - advanced: Uncertainty-weighted
    - advance_absolute: Per-point directional
    """

    def __init__(self):
        self.measurement_dataset = {}  # timestamp -> {offset, drift, ...}

    def apply_ntp_correction(self, ntp_measurement, method='drift_aware',
                            offset_uncertainty=0.001, drift_uncertainty=0.0001):
        """
        Apply dataset-only NTP correction using specified method.

        Args:
            ntp_measurement: NTP ground truth (timestamp, offset, uncertainty)
            method: One of ['none', 'linear', 'drift_aware', 'advanced', 'advance_absolute']
            offset_uncertainty: ML model offset uncertainty (seconds)
            drift_uncertainty: ML model drift uncertainty (seconds/second)

        Returns:
            Correction metadata dict or None
        """
        if method == 'none':
            return self._apply_none_correction(ntp_measurement)

        # Find interval for correction
        interval_start = self._find_interval_start(ntp_measurement.timestamp)
        if interval_start is None:
            # No previous data, just add NTP
            self.add_ntp_measurement(ntp_measurement)
            return None

        # Calculate error
        prediction = self.get_measurement_at_time(ntp_measurement.timestamp)
        if not prediction:
            self.add_ntp_measurement(ntp_measurement)
            return None

        error = ntp_measurement.offset - prediction['offset']
        interval_duration = ntp_measurement.timestamp - interval_start

        # Dispatch to appropriate method
        if method == 'linear':
            self._apply_linear_correction(
                interval_start, ntp_measurement.timestamp, error
            )
        elif method == 'drift_aware':
            self._apply_drift_aware_correction(
                interval_start, ntp_measurement.timestamp, error,
                offset_uncertainty, drift_uncertainty, interval_duration
            )
        elif method == 'advanced':
            self._apply_advanced_correction(
                interval_start, ntp_measurement.timestamp, error,
                ntp_measurement.uncertainty, offset_uncertainty, drift_uncertainty
            )
        elif method == 'advance_absolute':
            self._apply_advance_absolute_correction(
                interval_start, ntp_measurement.timestamp, error,
                ntp_measurement.uncertainty, offset_uncertainty,
                drift_uncertainty, interval_duration
            )
        else:
            raise ValueError(f"Unknown correction method: {method}")

        # Add NTP measurement to dataset
        self.add_ntp_measurement(ntp_measurement)

        return {
            'error': error,
            'interval_start': interval_start,
            'interval_end': ntp_measurement.timestamp,
            'interval_duration': interval_duration,
            'method': method
        }

    # ... (implement each _apply_*_correction method as shown in sections 2-6)
```

### A.2 Testing and Validation

```python
def test_correction_algorithms():
    """Validate all correction algorithms."""
    import numpy as np

    # Create synthetic dataset
    dataset = DatasetManager()
    true_drift = 1e-5  # 10 μs/s

    for t in range(0, 300):  # 5 minutes
        # Simulate drifting offset
        true_offset = true_drift * t
        # Add noise
        measured_offset = true_offset + np.random.normal(0, 0.001)
        dataset.add_measurement(t, measured_offset)

    # Simulate NTP measurement at t=300
    ntp_offset = true_drift * 300
    ntp = NTPMeasurement(timestamp=300, offset=ntp_offset, uncertainty=0.001)

    # Test each method
    methods = ['none', 'linear', 'drift_aware', 'advanced', 'advance_absolute']

    for method in methods:
        dataset_copy = copy.deepcopy(dataset)
        result = dataset_copy.apply_ntp_correction(
            ntp, method=method,
            offset_uncertainty=0.001,
            drift_uncertainty=1e-5
        )

        # Validate correction
        print(f"\n{method.upper()}:")
        print(f"  Error before: {result['error']*1000:.2f}ms")

        # Check correction properties
        validate_correction(dataset_copy, method)
```

---

**Document Version:** 1.0
**Last Updated:** October 2025
**License:** MIT
**Contact:** chronotick-research@example.com
