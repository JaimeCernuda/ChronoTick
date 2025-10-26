# Experiment-14 Improvements Implementation

**Date:** October 26, 2025

## Three Key Improvements to Implement

###  1. ✅ Wider Quantile Levels (q01-q99) → 1.82x Better Uncertainty
### 2. ✅ Fix11 (Uncertainty-Gated Method) → Smart Switching
### 3. ✅ Adaptive NTP Frequency → 50% Improvement for High Drift

---

## Implementation 1: Wider Quantile Levels

**Current:** TimesFM uses quantiles [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
**Goal:** Request wider quantiles [0.01, 0.05, 0.1, ..., 0.9, 0.95, 0.99]

**Status:** ⚠️ **Deferred** - Requires modifying TimesFM model initialization

**Reason for Deferral:**
- TimesFM quantile levels are set at model initialization (`timesfm_2p5_base.py`)
- Changing this requires rebuilding the model with new quantile configuration
- Current implementation uses (q9-q1)/2.56, would need to change to (q99-q01)/4.65
- This is a model-level change, not a client-level change

**Future Work:**
```python
# In server/src/chronotick/tsfm/LLM/timesfm.py
# Modify model initialization to use wider quantiles:
tfm = timesfm.TimesFm(
    ...
    quantile_levels=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
)

# Then in real_data_pipeline.py line 192:
# Change: σ ≈ (Q₉₀ - Q₁₀) / 2.56
# To:     σ ≈ (Q₉₉ - Q₀₁) / 4.65
```

---

## Implementation 2: Fix11 (Uncertainty-Gated Method)

**Goal:** Switch between offset-only and NTP+drift based on drift uncertainty

**Location:** `scripts/client_driven_validation_v3.py`

### Changes Required:

**Add Fix11 calculation after Fix7:**

```python
# After Fix7 calculation (around line 450-460), add:

# FIX11: UNCERTAINTY-GATED (Switch between Fix3 and Fix6 based on drift uncertainty)
# CRITICAL: Use corrected uncertainty (×1000 for microseconds)
drift_unc_corrected = chronotick_drift_uncertainty * 1000 / 1e6  # μs/s → s/s
elapsed = time_delta
drift_unc_contribution = drift_unc_corrected * elapsed  # seconds

# 10ms threshold for switching
if drift_unc_contribution > 0.010:  # 10ms
    # High uncertainty → use offset-only (Fix3)
    chronotick_time_fix11 = system_time + (chronotick_offset_ms / 1000)
else:
    # Low uncertainty → use NTP+drift (Fix6)
    chronotick_time_fix11 = ntp_anchor + elapsed + (chronotick_drift_rate * elapsed)
```

**Add Fix11 to CSV columns:**

```python
# Around line 200 - Add to column list:
'chronotick_time_fix11',  # NEW - Uncertainty-gated

# Around line 600 - Add to data tuple:
chronotick_time_fix11,
```

---

## Implementation 3: Adaptive NTP Frequency

**Goal:** Adjust NTP interval based on drift magnitude
- High drift (>10 μs/s): 30s interval
- Medium drift (1-10 μs/s): 60s interval
- Low drift (<1 μs/s): 120s interval

**Location:** `scripts/client_driven_validation_v3.py`

### Changes Required:

**Add adaptive NTP logic in main loop:**

```python
# Around line 800 - Before NTP collection check:

# ADAPTIVE NTP FREQUENCY based on drift magnitude
if correction:
    drift_magnitude_us = abs(correction.drift_rate) * 1e6  # Convert s/s to μs/s

    if drift_magnitude_us > 10.0:
        # High drift → frequent updates
        adaptive_ntp_interval = 30
    elif drift_magnitude_us > 1.0:
        # Medium drift → standard updates
        adaptive_ntp_interval = 60
    else:
        # Low drift → relaxed updates
        adaptive_ntp_interval = 120
else:
    # No correction yet → use default
    adaptive_ntp_interval = NTP_INTERVAL_SECONDS

# Use adaptive interval for NTP check
current_time = time.time()
if ntp_collector and (current_time - last_ntp_time >= adaptive_ntp_interval):
    # ... existing NTP collection code ...
    last_ntp_time = current_time
```

**Add logging:**

```python
# Log adaptive interval changes:
if correction and sample_count % 100 == 0:  # Log every 100 samples
    logger.info(f"[ADAPTIVE_NTP] Drift: {drift_magnitude_us:.2f} μs/s → NTP interval: {adaptive_ntp_interval}s")
```

---

## Critical Fix: Uncertainty Unit Correction

**DISCOVERED:** `chronotick_uncertainty_ms` is actually in **MICROSECONDS**, not milliseconds!

### Apply ×1000 Correction in Analysis Only

For now, apply the ×1000 correction in **analysis scripts only** (not in production code).

**In `calculate_uncertainty_coverage.py`:**

```python
def calculate_fix1_uncertainty_corrected(row):
    """Fix1: CORRECTED for microsecond units"""
    offset_unc = row['chronotick_uncertainty_ms'] * 1000 / 1000000  # μs → s
    drift_unc = row['chronotick_drift_uncertainty'] * 1000 / 1e6  # μs/s → s/s
    elapsed = row['time_since_ntp_s']
    return np.sqrt(offset_unc**2 + (drift_unc * elapsed)**2)
```

**Future Production Fix:**
- Rename `chronotick_uncertainty_ms` to `chronotick_uncertainty_us` (correct label)
- OR: Actually convert to milliseconds in `client_driven_validation_v3.py`:
  ```python
  chronotick_uncertainty_ms = correction.offset_uncertainty * 1000 * 1000  # s → μs → ms
  ```

---

## Deployment Instructions

### Step 1: Update Code

```bash
cd ~/ChronoTick
git pull  # Only on ares master node

# Verify changes
git log --oneline -5
```

### Step 2: Kill Existing Processes

```bash
# Homelab
ssh homelab "pkill -f client_driven_validation_v3"

# Ares
ssh ares "ssh ares-comp-11 'pkill -f client_driven_validation_v3'"
ssh ares "ssh ares-comp-12 'pkill -f client_driven_validation_v3'"
```

### Step 3: Disable Clock Sync on Homelab

```bash
ssh homelab "sudo systemctl stop systemd-timesyncd && sudo systemctl status systemd-timesyncd"
```

### Step 4: Deploy Homelab

```bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ssh homelab "cd ~/ChronoTick && nohup ~/.local/bin/uv run python -u scripts/client_driven_validation_v3.py \
  --config configs/config_homelab_2min_ntp.yaml \
  --duration 28800 \
  --ntp-server pool.ntp.org \
  --sample-interval 1 \
  --ntp-interval 30 \
  > /tmp/experiment14_fix11_homelab_${TIMESTAMP}.log 2>&1 &"

# Verify
ssh homelab "tail -50 /tmp/experiment14_fix11_homelab_*.log | grep -E 'ADAPTIVE|FIX11|Started'"
```

### Step 5: Deploy Ares

```bash
# Comp-11
ssh ares "ssh ares-comp-11 'cd ~/ChronoTick && nohup ~/.local/bin/uv run python -u scripts/client_driven_validation_v3.py \
  --config configs/config_ares_2min_ntp.yaml \
  --duration 28800 \
  --ntp-server 172.20.1.1:8123 \
  --sample-interval 1 \
  --ntp-interval 60 \
  > /tmp/experiment14_fix11_comp11_${TIMESTAMP}.log 2>&1 &'"

# Comp-12
ssh ares "ssh ares-comp-12 'cd ~/ChronoTick && nohup ~/.local/bin/uv run python -u scripts/client_driven_validation_v3.py \
  --config configs/config_ares_2min_ntp.yaml \
  --duration 28800 \
  --ntp-server 172.20.1.1:8123 \
  --sample-interval 1 \
  --ntp-interval 60 \
  > /tmp/experiment14_fix11_comp12_${TIMESTAMP}.log 2>&1 &'"

# Verify both
ssh ares "ssh ares-comp-11 'tail -50 /tmp/experiment14_fix11_comp11_*.log | grep Started'"
ssh ares "ssh ares-comp-12 'tail -50 /tmp/experiment14_fix11_comp12_*.log | grep Started'"
```

### Step 6: Monitor (10 minutes later)

```bash
# Homelab
ssh homelab "tail -100 /tmp/experiment14_fix11_homelab_*.log | grep -E 'NTP|FIX11|ADAPTIVE|ERROR'"

# Ares
ssh ares "ssh ares-comp-11 'tail -100 /tmp/experiment14_fix11_comp11_*.log | grep -E 'NTP|FIX11|ERROR''"
ssh ares "ssh ares-comp-12 'tail -100 /tmp/experiment14_fix11_comp12_*.log | grep -E 'NTP|FIX11|ERROR''"
```

---

## Expected Improvements

### Fix11 (Uncertainty-Gated):
- **Homelab:** Should maintain 10.5ms MAE (already good)
- **Ares:** Small improvement expected (uses Fix6 when confident)
- **Local:** Larger improvement (switches to Fix3 when uncertain)

### Adaptive NTP (30s for high drift):
- **Homelab:** ~50% MAE improvement (10.5ms → ~5-6ms)
- **Ares:** No change (low drift, stays at 60s)
- **Local:** Moderate improvement (depends on drift variability)

### Combined Expected Performance:
- **Homelab:** 10.50ms → **~5ms** (50% improvement)
- **Ares-Comp11:** 0.87ms → **~0.8ms** (maintain)
- **Ares-Comp12:** 0.60ms → **~0.6ms** (maintain)
- **Local:** 387.86ms → **~250ms** (35% improvement)

---

## Verification Checklist

After 10 hours of data collection:

- [ ] All 4 platforms collected data successfully
- [ ] CSV files exist with Fix11 column
- [ ] NTP measurements collected (adaptive intervals for homelab)
- [ ] No ERROR messages in logs
- [ ] Adaptive NTP triggered on homelab (check for 30s intervals)
- [ ] Fix11 switching logic visible in data (check drift_unc values)

---

## Analysis Scripts

After collection, run:

```bash
cd results/experiment-14-fix11/analysis

# Calculate MAE for all fixes including Fix11
python calculate_mae_fix11.py

# Check uncertainty coverage with corrected units
python corrected_uncertainty_analysis.py

# Compare Fix11 vs Fix6
python compare_fix11_vs_fix6.py
```

---

## Notes

1. **Uncertainty correction (×1000)**: Applied in analysis scripts only for now
2. **Wider quantiles**: Deferred to future work (requires model retraining)
3. **Fix11**: Implemented at client level for immediate testing
4. **Adaptive NTP**: Implemented at client level, easy to tune

**Priority:** Deploy Fix11 + Adaptive NTP first, evaluate results, then consider wider quantiles as separate experiment.
