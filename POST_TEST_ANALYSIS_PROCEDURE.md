# Post-Test Analysis Procedure
## 8-Hour Backtracking Test with Enhanced Logging

**Purpose**: Complete guide for analyzing overnight test results, generating diagnostic plots, and validating the NTP-to-ML comparison fix.

---

## Table of Contents
1. [Quick Start](#quick-start)
2. [Automated Log Analysis](#automated-log-analysis)
3. [CSV Data Files](#csv-data-files)
4. [Required Plots](#required-plots)
5. [Success Criteria](#success-criteria)
6. [Failure Patterns](#failure-patterns)
7. [Interpretation Guide](#interpretation-guide)

---

## Quick Start

### Step 1: Verify Test Completion
```bash
# Check if test process is still running
ps aux | grep 2501421

# If not running, verify log file size (should be large)
ls -lh tsfm/results/ntp_correction_experiment/overnight_8hr_backtracking_ENHANCED.log

# Check CSV output files exist
ls -lh tsfm/results/ntp_correction_experiment/overnight_8hr_backtracking_ENHANCED/
```

### Step 2: Run Automated Analysis
```bash
cd /home/jcernuda/tick_project/ChronoTick/tsfm

# Run enhanced log analyzer
uv run python scripts/analyze_ntp_correction.py \
  results/ntp_correction_experiment/overnight_8hr_backtracking_ENHANCED.log \
  > results/ntp_correction_experiment/overnight_8hr_backtracking_ENHANCED_ANALYSIS.txt

# View analysis
cat results/ntp_correction_experiment/overnight_8hr_backtracking_ENHANCED_ANALYSIS.txt
```

### Step 3: Generate Plots (detailed instructions below)

---

## Automated Log Analysis

### What the Analyzer Extracts

**From Enhanced Logging**:
- NTP correction count and timestamps
- Error magnitudes for each correction
- Prediction sources (CRITICAL: should be 'prediction_cpu', NOT 'ntp_measurement')
- Number of predictions replaced per correction
- Before/after/delta statistics for each correction:
  - Mean offset before correction
  - Mean offset after correction
  - Mean delta applied
  - Min/max delta range

**Dataset Statistics**:
- Total NTP measurements collected
- Total ML predictions generated
- First ML prediction timestamp
- Dataset growth over time

**Diagnosis**:
- Automated detection of NTP-to-NTP comparison bug
- Detection of zero-replacement corrections
- Error magnitude classification (small vs large)

### Analyzer Output Structure

```
================================================================================
NTP CORRECTION LOG ANALYSIS
================================================================================
Log file: results/ntp_correction_experiment/overnight_8hr_backtracking_ENHANCED.log

📊 DATASET STATISTICS
────────────────────────────────────────────────────────────────────────────────
  NTP measurements stored:    XXX
  ML predictions stored:      XXX
  First ML prediction time:   XXXXXXXXXX

🔧 NTP CORRECTIONS (XXX total)
────────────────────────────────────────────────────────────────────────────────

  Correction #1:
    Method: BACKTRACKING
    NTP offset: XX.XXXms
    Prediction offset: XX.XXXms
    ✅ Prediction source: prediction_cpu  ← CORRECT!
    Error: XX.XXXms ✓ SMALL or ✗ LARGE
    Duration: XXXs
    ✅ Predictions replaced: XX
    📈 BEFORE correction: mean=XX.XXXms, range=[XX.XXX, XX.XXX]ms
    📉 AFTER correction:  mean=XX.XXXms, range=[XX.XXX, XX.XXX]ms
    🔄 DELTA: mean=±XX.XXXms, range=[±XX.XXX, ±XX.XXX]ms

  [... all corrections ...]

────────────────────────────────────────────────────────────────────────────────
📈 CORRECTION SUMMARY
────────────────────────────────────────────────────────────────────────────────
  Total corrections:          XXX
  Small errors (<1ms):        XX (XX.X%)
  Large errors (≥1ms):        XX (XX.X%)

  🚨 NTP-to-NTP comparisons:  X (BUG - should be 0!)
  ✅ NTP-to-ML comparisons:   XXX (CORRECT)

  ❌ Zero replacements:       XX (no predictions in interval)
  ✅ Nonzero replacements:    XXX (corrections applied)

────────────────────────────────────────────────────────────────────────────────
🩺 DIAGNOSIS
────────────────────────────────────────────────────────────────────────────────
  ✅ GOOD: All comparisons are NTP-to-ML (bug fixed!)
  ✅ GOOD: Most corrections (XXX/XXX) are replacing predictions.
  ✅ GOOD: Most errors (XXX/XXX) are large (≥1ms).
```

---

## CSV Data Files

### Location
```
tsfm/results/ntp_correction_experiment/overnight_8hr_backtracking_ENHANCED/
├── summary_backtracking_YYYYMMDD_HHMMSS.csv
├── client_predictions_backtracking_YYYYMMDD_HHMMSS.csv
└── dataset_corrections_backtracking_YYYYMMDD_HHMMSS.csv
```

### File 1: summary_backtracking_*.csv

**Purpose**: Performance summary with ground truth comparisons

**Columns**:
- `timestamp`: Unix timestamp of client prediction
- `elapsed_seconds`: Time since test start
- `client_offset_ms`: ChronoTick predicted time correction (what we report to user)
- `client_drift_us_per_s`: Predicted drift rate in microseconds per second
- `client_source`: Prediction source ('cpu' for short-term model, 'fusion' for blended)
- `ntp_ground_truth_offset_ms`: True NTP offset when available (ground truth)
- `ntp_uncertainty_ms`: NTP measurement uncertainty
- `has_ntp`: Boolean - whether NTP ground truth exists for this timestamp
- `chronotick_error_ms`: |client_offset - ntp_ground_truth| (our prediction error)
- `system_error_ms`: |system_clock - ntp_ground_truth| (baseline comparison)
- `dataset_size`: Number of measurements in training dataset at this time
- `corrections_applied`: Cumulative count of NTP corrections applied so far

**Key Metrics to Extract**:
- **ChronoTick accuracy**: `chronotick_error_ms` over time (should decrease)
- **Baseline comparison**: `chronotick_error_ms` vs `system_error_ms` (we should be better than system clock)
- **Learning effect**: Correlation between `corrections_applied` and decreasing `chronotick_error_ms`

### File 2: client_predictions_backtracking_*.csv

**Purpose**: Raw client-facing predictions (every 10 seconds)

**Columns**:
- `timestamp`: Unix timestamp of prediction
- `offset_correction_ms`: Predicted offset correction
- `drift_rate_us_per_s`: Predicted drift rate
- `offset_uncertainty_ms`: Prediction uncertainty (should decrease over time)
- `confidence`: Confidence level (always 0.95 for now)
- `source`: Prediction source ('cpu' or 'fusion')

**Key Metrics to Extract**:
- **Offset evolution**: How predictions change over time
- **Uncertainty evolution**: Should decrease as model learns from NTP corrections
- **Source distribution**: Ratio of 'cpu' vs 'fusion' predictions

### File 3: dataset_corrections_backtracking_*.csv

**Purpose**: Detailed record of every correction applied to the dataset

**Columns**:
- `ntp_event_timestamp`: When the NTP measurement that triggered correction arrived
- `ntp_offset_ms`: NTP ground truth offset
- `ntp_uncertainty_ms`: NTP measurement uncertainty
- `correction_method`: Always 'backtracking' for this test
- `error_ms`: Error between NTP truth and ML prediction
- `interval_start`: Start timestamp of correction interval
- `interval_end`: End timestamp of correction interval
- `interval_duration_s`: Duration of correction interval in seconds
- `measurement_timestamp`: Timestamp of the measurement being corrected
- `time_since_interval_start_s`: Time offset within interval
- `offset_before_correction_ms`: ML prediction BEFORE correction
- `offset_after_correction_ms`: NTP-interpolated value AFTER correction
- `correction_delta_ms`: Change applied (after - before)
- `was_corrected`: Boolean - whether this was an ML prediction (True) or NTP measurement (False)

**Key Metrics to Extract**:
- **Correction frequency**: Time between NTP correction events
- **Error magnitude evolution**: `error_ms` over time (should decrease)
- **Replacement statistics**: Distribution of `correction_delta_ms` (how much we're changing)
- **Correction interval duration**: Average time between NTP validations

---

## Required Plots

### Plot 1: Time Series - Predictions vs Ground Truth

**Purpose**: Visualize how ChronoTick predictions compare to NTP ground truth over the 8-hour test

**Data Source**: `summary_backtracking_*.csv`

**Plot Details**:
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('summary_backtracking_*.csv')

# Convert to elapsed time in hours
df['elapsed_hours'] = df['elapsed_seconds'] / 3600

# Create figure
fig, ax = plt.subplots(figsize=(14, 6))

# Plot ChronoTick predictions (all points)
ax.plot(df['elapsed_hours'], df['client_offset_ms'],
        label='ChronoTick Prediction', alpha=0.7, linewidth=1)

# Plot NTP ground truth (sparse - only where available)
ntp_mask = df['has_ntp'] == True
ax.scatter(df[ntp_mask]['elapsed_hours'], df[ntp_mask]['ntp_ground_truth_offset_ms'],
           label='NTP Ground Truth', color='red', marker='x', s=50, zorder=5)

# Formatting
ax.set_xlabel('Time (hours)', fontsize=12)
ax.set_ylabel('Clock Offset (ms)', fontsize=12)
ax.set_title('ChronoTick Predictions vs NTP Ground Truth (8-Hour Test)', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plot1_predictions_vs_ground_truth.png', dpi=300)
```

**What to Look For**:
- ✅ **GOOD**: Predictions track close to NTP ground truth points
- ❌ **BAD**: Large gaps between predictions and NTP ground truth (>20ms consistently)
- ✅ **GOOD**: Predictions converge toward NTP ground truth over time (learning effect)

---

### Plot 2: Prediction Error Over Time

**Purpose**: Show how prediction accuracy improves as ML learns from NTP corrections

**Data Source**: `summary_backtracking_*.csv`

**Plot Details**:
```python
# Filter to only rows with NTP ground truth
df_with_ntp = df[df['has_ntp'] == True].copy()

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Subplot 1: Raw error over time
ax1.scatter(df_with_ntp['elapsed_hours'], df_with_ntp['chronotick_error_ms'],
            label='ChronoTick Error', alpha=0.6, s=20)
ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax1.set_xlabel('Time (hours)', fontsize=12)
ax1.set_ylabel('Prediction Error (ms)', fontsize=12)
ax1.set_title('ChronoTick Prediction Error Over Time', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Subplot 2: Rolling mean error (smoothed)
window_size = max(10, len(df_with_ntp) // 50)  # Adaptive window
df_with_ntp['error_rolling_mean'] = df_with_ntp['chronotick_error_ms'].rolling(window=window_size, min_periods=1).mean()

ax2.plot(df_with_ntp['elapsed_hours'], df_with_ntp['error_rolling_mean'],
         label=f'Rolling Mean Error (window={window_size})', linewidth=2)
ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax2.set_xlabel('Time (hours)', fontsize=12)
ax2.set_ylabel('Rolling Mean Error (ms)', fontsize=12)
ax2.set_title('Smoothed Prediction Error Over Time', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plot2_prediction_error_evolution.png', dpi=300)
```

**What to Look For**:
- ✅ **GOOD**: Error magnitude decreases over time (learning effect visible)
- ✅ **GOOD**: Rolling mean error trends toward zero
- ❌ **BAD**: Error stays constant or increases over time (ML not learning)
- ❌ **BAD**: Error always <1ms (suggests NTP-to-NTP comparison bug)

---

### Plot 3: ChronoTick vs System Clock (Baseline Comparison)

**Purpose**: Demonstrate that ChronoTick outperforms the system clock

**Data Source**: `summary_backtracking_*.csv`

**Plot Details**:
```python
df_with_ntp = df[df['has_ntp'] == True].copy()

fig, ax = plt.subplots(figsize=(14, 6))

# Plot both errors
ax.plot(df_with_ntp['elapsed_hours'], df_with_ntp['chronotick_error_ms'],
        label='ChronoTick Error', alpha=0.7, linewidth=2)
ax.plot(df_with_ntp['elapsed_hours'], df_with_ntp['system_error_ms'],
        label='System Clock Error (Baseline)', alpha=0.7, linewidth=2)

# Add mean lines
chronotick_mean = df_with_ntp['chronotick_error_ms'].mean()
system_mean = df_with_ntp['system_error_ms'].mean()
ax.axhline(y=chronotick_mean, color='C0', linestyle='--', linewidth=1,
           label=f'ChronoTick Mean: {chronotick_mean:.2f}ms')
ax.axhline(y=system_mean, color='C1', linestyle='--', linewidth=1,
           label=f'System Mean: {system_mean:.2f}ms')

ax.set_xlabel('Time (hours)', fontsize=12)
ax.set_ylabel('Absolute Error (ms)', fontsize=12)
ax.set_title('ChronoTick vs System Clock Error (8-Hour Test)', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plot3_chronotick_vs_system_clock.png', dpi=300)
```

**What to Look For**:
- ✅ **GOOD**: ChronoTick error < System clock error (we're better than baseline)
- ✅ **GOOD**: ChronoTick mean error significantly lower than system mean
- ❌ **BAD**: ChronoTick error ≈ System error (suggests we're just using system clock)
- ❌ **BAD**: High correlation between the two lines (suggests system clock usage)

---

### Plot 4: NTP Correction Error Evolution

**Purpose**: Show that correction errors decrease as ML learns from previous corrections

**Data Source**: `dataset_corrections_backtracking_*.csv` (filtered to correction events)

**Plot Details**:
```python
df_corr = pd.read_csv('dataset_corrections_backtracking_*.csv')

# Extract unique correction events (one row per correction)
correction_events = df_corr.groupby('ntp_event_timestamp').first().reset_index()
correction_events['correction_number'] = range(1, len(correction_events) + 1)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Subplot 1: Error magnitude per correction
ax1.scatter(correction_events['correction_number'],
            correction_events['error_ms'].abs(),
            s=50, alpha=0.7)
ax1.plot(correction_events['correction_number'],
         correction_events['error_ms'].abs(),
         alpha=0.3, linewidth=1)
ax1.set_xlabel('Correction Number', fontsize=12)
ax1.set_ylabel('Absolute Error (ms)', fontsize=12)
ax1.set_title('NTP Correction Error Magnitude Over Time', fontsize=14)
ax1.grid(True, alpha=0.3)

# Subplot 2: Signed error (shows bias)
ax2.scatter(correction_events['correction_number'],
            correction_events['error_ms'],
            s=50, alpha=0.7)
ax2.plot(correction_events['correction_number'],
         correction_events['error_ms'],
         alpha=0.3, linewidth=1)
ax2.axhline(y=0, color='red', linestyle='--', linewidth=1)
ax2.set_xlabel('Correction Number', fontsize=12)
ax2.set_ylabel('Signed Error (ms)', fontsize=12)
ax2.set_title('NTP Correction Error (Signed) Over Time', fontsize=14)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plot4_correction_error_evolution.png', dpi=300)
```

**What to Look For**:
- ✅ **GOOD**: Error magnitude decreases over correction number (ML learning)
- ✅ **GOOD**: Errors start large (10-50ms) early, become smaller (<10ms) later
- ❌ **BAD**: All errors small (<1ms) throughout (NTP-to-NTP bug)
- ❌ **BAD**: No trend or increasing errors (ML not learning)

---

### Plot 5: Correction Statistics (Before/After/Delta)

**Purpose**: Visualize the magnitude of changes applied by backtracking corrections

**Data Source**: `dataset_corrections_backtracking_*.csv`

**Plot Details**:
```python
# Calculate per-correction statistics
correction_stats = []
for ntp_event in df_corr['ntp_event_timestamp'].unique():
    event_data = df_corr[df_corr['ntp_event_timestamp'] == ntp_event]
    corrected = event_data[event_data['was_corrected'] == True]

    if len(corrected) > 0:
        correction_stats.append({
            'ntp_event': ntp_event,
            'mean_before': corrected['offset_before_correction_ms'].mean(),
            'mean_after': corrected['offset_after_correction_ms'].mean(),
            'mean_delta': corrected['correction_delta_ms'].mean(),
            'num_replaced': len(corrected),
            'error_ms': event_data['error_ms'].iloc[0]
        })

df_stats = pd.DataFrame(correction_stats)
df_stats['correction_number'] = range(1, len(df_stats) + 1)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Subplot 1: Before vs After mean offsets
ax1.plot(df_stats['correction_number'], df_stats['mean_before'],
         label='Mean Before Correction', marker='o', linewidth=2)
ax1.plot(df_stats['correction_number'], df_stats['mean_after'],
         label='Mean After Correction', marker='s', linewidth=2)
ax1.set_xlabel('Correction Number', fontsize=12)
ax1.set_ylabel('Mean Offset (ms)', fontsize=12)
ax1.set_title('Mean Offset Before vs After Each Correction', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Subplot 2: Mean delta applied
ax2.scatter(df_stats['correction_number'], df_stats['mean_delta'],
            s=df_stats['num_replaced']*2,  # Size = number replaced
            alpha=0.6, label='Mean Delta (size = # replaced)')
ax2.axhline(y=0, color='red', linestyle='--', linewidth=1)
ax2.set_xlabel('Correction Number', fontsize=12)
ax2.set_ylabel('Mean Delta Applied (ms)', fontsize=12)
ax2.set_title('Mean Correction Delta Per Event', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plot5_correction_statistics.png', dpi=300)
```

**What to Look For**:
- ✅ **GOOD**: Mean delta decreases in magnitude over time (smaller corrections needed)
- ✅ **GOOD**: Before/after lines converge (predictions getting closer to NTP truth)
- ❌ **BAD**: Delta always near zero (no meaningful corrections)
- ❌ **BAD**: Before/after lines stay far apart (corrections not helping)

---

### Plot 6: Predictions Replaced per Correction

**Purpose**: Show that backtracking is consistently replacing predictions (not just skipping)

**Data Source**: `dataset_corrections_backtracking_*.csv`

**Plot Details**:
```python
# Count predictions replaced per event
replacements = df_corr[df_corr['was_corrected'] == True].groupby('ntp_event_timestamp').size()
correction_events = df_corr.groupby('ntp_event_timestamp').first().reset_index()
correction_events['num_replaced'] = replacements.values
correction_events['correction_number'] = range(1, len(correction_events) + 1)

fig, ax = plt.subplots(figsize=(14, 6))

# Bar plot
ax.bar(correction_events['correction_number'],
       correction_events['num_replaced'],
       alpha=0.7, edgecolor='black')
ax.axhline(y=correction_events['num_replaced'].mean(),
           color='red', linestyle='--', linewidth=2,
           label=f'Mean: {correction_events["num_replaced"].mean():.1f}')

ax.set_xlabel('Correction Number', fontsize=12)
ax.set_ylabel('Number of Predictions Replaced', fontsize=12)
ax.set_title('Predictions Replaced per NTP Correction Event', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('plot6_predictions_replaced.png', dpi=300)
```

**What to Look For**:
- ✅ **GOOD**: Most corrections replace >10 predictions (backtracking working)
- ✅ **GOOD**: Consistent replacement counts across corrections
- ❌ **BAD**: Many corrections with 0 replacements (ML predictions missing)
- ❌ **BAD**: Highly variable replacement counts (timing issues)

---

### Plot 7: Uncertainty Evolution

**Purpose**: Show that prediction uncertainty decreases as model learns from NTP corrections

**Data Source**: `client_predictions_backtracking_*.csv` and `summary_backtracking_*.csv`

**Plot Details**:
```python
df_pred = pd.read_csv('client_predictions_backtracking_*.csv')
df_summary = pd.read_csv('summary_backtracking_*.csv')

# Merge to get elapsed time
df_pred['timestamp_int'] = df_pred['timestamp'].astype(int)
df_summary['timestamp_int'] = df_summary['timestamp'].astype(int)
df_merged = df_pred.merge(df_summary[['timestamp_int', 'elapsed_seconds', 'corrections_applied']],
                           on='timestamp_int', how='left')
df_merged['elapsed_hours'] = df_merged['elapsed_seconds'] / 3600

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Subplot 1: Uncertainty over time
ax1.plot(df_merged['elapsed_hours'], df_merged['offset_uncertainty_ms'],
         alpha=0.7, linewidth=1)
ax1.set_xlabel('Time (hours)', fontsize=12)
ax1.set_ylabel('Offset Uncertainty (ms)', fontsize=12)
ax1.set_title('Prediction Uncertainty Over Time', fontsize=14)
ax1.grid(True, alpha=0.3)

# Subplot 2: Uncertainty vs corrections applied
ax2.scatter(df_merged['corrections_applied'], df_merged['offset_uncertainty_ms'],
            alpha=0.3, s=10)
ax2.set_xlabel('Cumulative Corrections Applied', fontsize=12)
ax2.set_ylabel('Offset Uncertainty (ms)', fontsize=12)
ax2.set_title('Uncertainty vs Corrections Applied', fontsize=14)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plot7_uncertainty_evolution.png', dpi=300)
```

**What to Look For**:
- ✅ **GOOD**: Uncertainty decreases over time (model confidence increasing)
- ✅ **GOOD**: Negative correlation with corrections applied (more corrections → lower uncertainty)
- ❌ **BAD**: Uncertainty stays constant or increases (model not gaining confidence)

---

### Plot 8: Correction Interval Duration

**Purpose**: Understand timing between NTP corrections

**Data Source**: `dataset_corrections_backtracking_*.csv`

**Plot Details**:
```python
correction_events = df_corr.groupby('ntp_event_timestamp').first().reset_index()
correction_events = correction_events.sort_values('ntp_event_timestamp')
correction_events['correction_number'] = range(1, len(correction_events) + 1)

fig, ax = plt.subplots(figsize=(14, 6))

# Plot interval duration
ax.scatter(correction_events['correction_number'],
           correction_events['interval_duration_s'],
           s=50, alpha=0.7)
ax.axhline(y=correction_events['interval_duration_s'].mean(),
           color='red', linestyle='--', linewidth=2,
           label=f'Mean: {correction_events["interval_duration_s"].mean():.1f}s')
ax.axhline(y=180, color='green', linestyle='--', linewidth=1,
           label='Expected: 180s (3 minutes)')

ax.set_xlabel('Correction Number', fontsize=12)
ax.set_ylabel('Interval Duration (seconds)', fontsize=12)
ax.set_title('Time Between NTP Corrections', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plot8_correction_intervals.png', dpi=300)
```

**What to Look For**:
- ✅ **GOOD**: Intervals near 180s (expected NTP validation frequency)
- ✅ **GOOD**: Consistent interval durations (predictable correction schedule)
- ❌ **BAD**: Highly variable intervals (NTP collection issues)

---

## Success Criteria

### Critical Success Indicators

**1. All Prediction Sources are ML (NOT NTP)**
```bash
# Check analyzer output for:
🚨 NTP-to-NTP comparisons: 0 (BUG - should be 0!)
✅ NTP-to-ML comparisons: XXX (CORRECT)

# Verify manually:
grep "Prediction source:" overnight_8hr_backtracking_ENHANCED.log | grep -c "prediction_cpu"
# Should equal total corrections

grep "Prediction source:" overnight_8hr_backtracking_ENHANCED.log | grep -c "ntp_measurement"
# Should be 0
```

**2. Large Errors Initially, Decreasing Over Time**
- First 10 corrections: Mean error ≥5ms (ideally 10-50ms)
- Last 10 corrections: Mean error <5ms (ideally <3ms)
- Clear downward trend in Plot 4

**3. Nonzero Prediction Replacements**
```bash
# Check analyzer output for:
✅ Nonzero replacements: XXX (corrections applied)

# Most corrections (>80%) should have num_replaced > 0
# Plot 6 should show consistent bars >10
```

**4. ChronoTick Outperforms System Clock**
- Mean ChronoTick error < Mean system error (from Plot 3)
- ChronoTick error line below system error line for majority of test
- Low correlation between ChronoTick and system errors (<0.3)

**5. Uncertainty Decreases Over Time**
- Initial uncertainty: ~30-40ms
- Final uncertainty: <10ms
- Plot 7 shows clear downward trend

**6. Meaningful Correction Deltas**
- Mean delta magnitude matches error magnitude (within 2x)
- Early corrections: Mean delta ≥5ms
- Deltas decrease over time (Plot 5)

---

## Failure Patterns

### Pattern 1: Small Adjustments Throughout (<1ms)

**Symptoms**:
- All errors in analyzer <1ms
- Mean correction delta <1ms consistently
- Plots 2 and 4 show flat lines near zero

**Diagnosis**: NTP-to-NTP comparison bug still present

**Evidence to Check**:
```bash
grep "Prediction source:" overnight_8hr_backtracking_ENHANCED.log | head -20
# If shows "ntp_measurement", bug is still there
```

---

### Pattern 2: High Correlation with System Clock

**Symptoms**:
- ChronoTick error ≈ System error in Plot 3
- Correlation coefficient >0.8 between the two
- No improvement over baseline

**Diagnosis**: System clock being used instead of ML predictions

**Evidence to Check**:
```python
# Calculate correlation
import numpy as np
df_with_ntp = df[df['has_ntp'] == True]
correlation = np.corrcoef(df_with_ntp['chronotick_error_ms'],
                          df_with_ntp['system_error_ms'])[0, 1]
print(f"Correlation: {correlation:.3f}")
# Should be <0.3 for good results
```

---

### Pattern 3: Zero Replacements (Corrections Not Applied)

**Symptoms**:
- Most corrections have num_replaced = 0
- Plot 6 shows many zero bars
- Analyzer shows high "Zero replacements" count

**Diagnosis**: ML predictions not in correction intervals (timing mismatch)

**Evidence to Check**:
```bash
# Check first ML prediction time vs first correction time
grep "First ML prediction time:" overnight_8hr_backtracking_ENHANCED_ANALYSIS.txt
grep "Correction #1:" overnight_8hr_backtracking_ENHANCED_ANALYSIS.txt -A 5
# ML predictions should start BEFORE first correction
```

---

### Pattern 4: No Learning (Errors Don't Decrease)

**Symptoms**:
- Plot 2 shows no downward trend in error
- Plot 4 shows constant or increasing errors
- Rolling mean error stays constant or increases

**Diagnosis**: ML not learning from NTP corrections

**Possible Causes**:
- Corrections not being added to dataset properly
- Model not retraining on corrected data
- Correction deltas too small to influence model

**Evidence to Check**:
```bash
# Check if corrections are actually modifying dataset
grep "REPLACED.*predictions with NTP-interpolated" overnight_8hr_backtracking_ENHANCED.log | wc -l
# Should equal total corrections count
```

---

### Pattern 5: Corrections Getting Worse

**Symptoms**:
- Error magnitude increases over time in Plot 4
- Before/after gap widens in Plot 5
- Later corrections have larger deltas than early ones

**Diagnosis**: ML learning incorrect patterns from corrections

**Evidence to Check**:
- Check for systematic bias in signed errors (all positive or all negative)
- Verify NTP measurement quality (uncertainty should be <50ms)
- Check for NTP server issues (changing servers mid-test)

---

## Interpretation Guide

### Expected Results (Fix Working Correctly)

**Timeline**:
- **0-60s**: Warmup, collecting NTP measurements
- **60-180s**: First ML predictions generated
- **180s**: First NTP correction (error: 10-50ms, replacements: 15-20)
- **360-1800s (6-30 min)**: Errors decrease rapidly (10-50ms → 5-15ms)
- **1800-7200s (30min-2hr)**: Errors stabilize (5-10ms)
- **7200-28800s (2hr-8hr)**: Small refinement errors (<5ms)

**Key Metrics**:
- Total corrections: ~144 (one every ~180s)
- Mean early error (first hour): 15-30ms
- Mean late error (last hour): 2-5ms
- Mean predictions replaced: 15-20 per correction
- Final uncertainty: <10ms

**Plots Summary**:
- Plot 1: Predictions track NTP closely, converging over time
- Plot 2: Clear downward trend in both raw and smoothed error
- Plot 3: ChronoTick line well below system clock line
- Plot 4: Decreasing error magnitude from 20-30ms → 2-5ms
- Plot 5: Before/after lines converge; deltas decrease
- Plot 6: Consistent 15-20 replacements per correction
- Plot 7: Uncertainty decreases from 35ms → <10ms
- Plot 8: Consistent ~180s intervals

---

### Comparing to 12-Minute Validation Test

The 12-minute test from earlier showed:
- 2 corrections total
- Errors: 18.203ms, -10.474ms (LARGE - good!)
- Replacements: 19, 18
- All sources: 'prediction_cpu' ✅

**Expected 8-Hour Improvements**:
- More corrections (~144 vs 2) = more statistical power
- Decreasing error trend visible (not possible in 12-min test)
- Lower final uncertainty (more training data)
- Stable long-term performance metrics

---

## Automated Analysis Script Template

```python
#!/usr/bin/env python3
"""
Complete automated analysis and plotting for 8-hour backtracking test.

Usage:
    uv run python analyze_and_plot_8hr_test.py \\
        --summary results/.../summary_backtracking_*.csv \\
        --predictions results/.../client_predictions_backtracking_*.csv \\
        --corrections results/.../dataset_corrections_backtracking_*.csv \\
        --output-dir plots/
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path

def load_data(summary_path, predictions_path, corrections_path):
    """Load all three CSV files."""
    df_summary = pd.read_csv(summary_path)
    df_predictions = pd.read_csv(predictions_path)
    df_corrections = pd.read_csv(corrections_path)
    return df_summary, df_predictions, df_corrections

def generate_all_plots(df_summary, df_predictions, df_corrections, output_dir):
    """Generate all 8 required plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # [Implement Plot 1-8 as described above]
    # Save each plot to output_dir

    print(f"All plots saved to {output_dir}")

def calculate_success_metrics(df_summary, df_corrections):
    """Calculate and print success criteria metrics."""
    # [Implement metric calculations]
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze 8-hour backtracking test')
    parser.add_argument('--summary', required=True, help='Path to summary CSV')
    parser.add_argument('--predictions', required=True, help='Path to predictions CSV')
    parser.add_argument('--corrections', required=True, help='Path to corrections CSV')
    parser.add_argument('--output-dir', default='plots/', help='Output directory for plots')

    args = parser.parse_args()

    df_summary, df_predictions, df_corrections = load_data(
        args.summary, args.predictions, args.corrections
    )

    generate_all_plots(df_summary, df_predictions, df_corrections, args.output_dir)
    calculate_success_metrics(df_summary, df_corrections)
```

---

## Quick Reference Commands

### After Test Completes

```bash
cd /home/jcernuda/tick_project/ChronoTick/tsfm

# 1. Run log analyzer
uv run python scripts/analyze_ntp_correction.py \\
  results/ntp_correction_experiment/overnight_8hr_backtracking_ENHANCED.log \\
  > results/ntp_correction_experiment/overnight_8hr_backtracking_ENHANCED_ANALYSIS.txt

# 2. Find CSV files
ls -lh results/ntp_correction_experiment/overnight_8hr_backtracking_ENHANCED/

# 3. Load into Python for plotting
python3 << EOF
import pandas as pd
import matplotlib.pyplot as plt

# Load summary
df = pd.read_csv('results/ntp_correction_experiment/overnight_8hr_backtracking_ENHANCED/summary_backtracking_*.csv')
print(f"Test duration: {df['elapsed_seconds'].max()/3600:.1f} hours")
print(f"Total predictions: {len(df)}")
print(f"NTP validations: {df['has_ntp'].sum()}")
print(f"Mean ChronoTick error: {df[df['has_ntp']]['chronotick_error_ms'].mean():.2f}ms")
print(f"Mean System error: {df[df['has_ntp']]['system_error_ms'].mean():.2f}ms")
EOF
```

---

**End of Document**

For any questions or issues with analysis, refer to:
- Test info: `tsfm/results/ntp_correction_experiment/overnight_8hr_backtracking_ENHANCED/TEST_INFO.md`
- Monitor guide: `MONITOR_8HR_TEST.md`
- This analysis guide: `POST_TEST_ANALYSIS_PROCEDURE.md`
