# Validation Plan: Are Corrections Actually Working?

## Hypothesis
The backtracking corrections are insufficient because:
1. Only 2-7 predictions corrected per 180s interval (~2-4%)
2. Context window is 512s but corrections only cover ~180s
3. Model trains on 99%+ uncorrected (wrong) predictions

## Validation Steps

### Step 1: Extract Correction Events
```bash
# Find all backtracking corrections
ssh homelab "grep 'BACKTRACKING.*REPLACED' /tmp/chronotick_8hr_20251016_152510.log" > corrections.log

# Example output:
# [BACKTRACKING] REPLACED 2 predictions with NTP-interpolated values
# [BACKTRACKING] REPLACED 7 predictions with NTP-interpolated values
```

### Step 2: Analyze Correction Coverage
For each NTP measurement:
- Time between NTP measurements: 180s
- Predictions made in interval: 180s / 10s = 18 predictions
- Predictions corrected: 2-7 (from logs)
- Coverage: 11-39% of interval ❌

Expected: 100% of interval should be corrected

### Step 3: Check Context Window Contamination
When model makes prediction at time T:
- Context window: [T-512s, T]
- Number of predictions in context: 512 / 10s = ~51 predictions
- Number corrected: ~4 per NTP × (512/180) = ~11 corrections
- Contamination: 40/51 = 78% uncorrected ❌

Expected: 100% of context should be corrected

### Step 4: Compare Corrected Dataset vs External NTP
```python
# Load results
df = pd.read_csv('chronotick_stability_20251016_152510.csv')
ntp_rows = df[df['ntp_offset_ms'].notna()]

# For each NTP measurement:
for idx, row in ntp_rows.iterrows():
    ntp_time = row['timestamp']
    ntp_truth = row['ntp_offset_ms']

    # Find predictions in the interval BEFORE this NTP
    interval_start = ntp_time - 180
    interval_preds = df[(df['timestamp'] >= interval_start) &
                        (df['timestamp'] < ntp_time)]

    # What does the DATASET show for this interval?
    # (These are the values the model trains on)
    dataset_mean = interval_preds['chronotick_offset_ms'].mean()

    # What was the actual NTP truth for this interval?
    # (Interpolated between prev_ntp and current_ntp)
    prev_ntp = ntp_rows[ntp_rows['timestamp'] < ntp_time].iloc[-1]
    expected_mean = (prev_ntp['ntp_offset_ms'] + ntp_truth) / 2

    # How close is the dataset to truth?
    dataset_error = abs(dataset_mean - expected_mean)

    print(f"Interval: {interval_start:.0f}-{ntp_time:.0f}")
    print(f"  Dataset mean: {dataset_mean:.3f}ms")
    print(f"  Expected (NTP): {expected_mean:.3f}ms")
    print(f"  Error: {dataset_error:.3f}ms")
```

### Expected Results

**IF corrections are working:**
- Dataset mean ≈ NTP interpolated mean
- Error < 1ms

**IF corrections are NOT working:**
- Dataset mean = original wrong predictions
- Error >> 1ms

## Implementation Checklist

### Quick Fix (Tonight):
- [ ] Read config: context_length value
- [ ] Check current backtracking interval logic
- [ ] Modify to correct full context window (512s, not 180s)
- [ ] Test on homelab after current test completes

### Validation (Tomorrow):
- [ ] Run script above on completed test data
- [ ] Measure correction quality (dataset vs NTP truth)
- [ ] Confirm context window coverage improves
- [ ] Start new 8hr test with fixed backtracking

### Long-term:
- [ ] Add logging: "Corrected X samples out of Y in context window"
- [ ] Add metric: "Context coverage %"
- [ ] Alert if coverage drops below 95%
