# Quick Monitor: 8-Hour Backtracking Test

**Test PID**: 2501421
**Start Time**: 2025-10-14 01:04:40
**Expected End**: 2025-10-14 09:04:40 (~8 hours)

---

## Quick Status Commands

```bash
# Is the test still running?
ps aux | grep 2501421 | grep -v grep

# How many lines in the log? (growth = progress)
wc -l tsfm/results/ntp_correction_experiment/overnight_8hr_backtracking_ENHANCED.log

# Latest activity (last 10 lines)
tail -10 tsfm/results/ntp_correction_experiment/overnight_8hr_backtracking_ENHANCED.log

# Dataset size (NTP measurements collected)
grep -c "DATASET_ADD_NTP" tsfm/results/ntp_correction_experiment/overnight_8hr_backtracking_ENHANCED.log

# ML predictions generated
grep -c "DATASET_ADD_PRED" tsfm/results/ntp_correction_experiment/overnight_8hr_backtracking_ENHANCED.log

# NTP corrections applied
grep -c "NTP_CORRECTION_BACKTRACKING" tsfm/results/ntp_correction_experiment/overnight_8hr_backtracking_ENHANCED.log
```

---

## Watch for NTP Corrections

### First Correction Expected: ~3 minutes after start (t=180s)

```bash
# Watch for corrections in real-time
tail -f tsfm/results/ntp_correction_experiment/overnight_8hr_backtracking_ENHANCED.log | grep --line-buffered "\[NTP_CORRECTION\|BACKTRACKING\|REPLACEMENT"

# Check if any corrections have occurred
grep "Prediction source:" tsfm/results/ntp_correction_experiment/overnight_8hr_backtracking_ENHANCED.log

# See error magnitudes
grep "Error = " tsfm/results/ntp_correction_experiment/overnight_8hr_backtracking_ENHANCED.log | grep -v "NTP_truth"

# See replacement counts
grep "REPLACED" tsfm/results/ntp_correction_experiment/overnight_8hr_backtracking_ENHANCED.log
```

---

## Enhanced Logging Check

### View Before/After Replacement Statistics

```bash
# See all replacement summaries
grep -A 20 "REPLACEMENT SUMMARY" tsfm/results/ntp_correction_experiment/overnight_8hr_backtracking_ENHANCED.log

# See individual replacement examples
grep -A 8 "First 5 replacements:" tsfm/results/ntp_correction_experiment/overnight_8hr_backtracking_ENHANCED.log
```

---

## Analysis After Completion

```bash
# Run enhanced analyzer (after test completes)
cd /home/jcernuda/tick_project/ChronoTick/tsfm
uv run python scripts/analyze_ntp_correction.py \
  results/ntp_correction_experiment/overnight_8hr_backtracking_ENHANCED.log

# Save analysis to file
uv run python scripts/analyze_ntp_correction.py \
  results/ntp_correction_experiment/overnight_8hr_backtracking_ENHANCED.log \
  > results/ntp_correction_experiment/overnight_8hr_backtracking_ENHANCED_ANALYSIS.txt
```

---

## Success Indicators (What to Look For)

✅ **All prediction sources are ML** (not NTP):
```bash
grep "Prediction source:" tsfm/results/ntp_correction_experiment/overnight_8hr_backtracking_ENHANCED.log | grep -c "prediction_cpu"
# Should equal total corrections
```

✅ **Large errors in early corrections** (≥5ms):
```bash
grep "Error = " tsfm/results/ntp_correction_experiment/overnight_8hr_backtracking_ENHANCED.log | head -5
# First few should be 10+ ms
```

✅ **Nonzero replacements**:
```bash
grep "REPLACED" tsfm/results/ntp_correction_experiment/overnight_8hr_backtracking_ENHANCED.log | grep -v "REPLACED 0"
# Should see many entries
```

---

## Full Documentation

See detailed test info: `tsfm/results/ntp_correction_experiment/overnight_8hr_backtracking_ENHANCED/TEST_INFO.md`

---

**Current Status**: Test running (PID: 2501421)
**Progress**: ~2 minutes, warmup phase, 11 NTP measurements collected
**Next Milestone**: First NTP correction at t=180s (~01:07:40)
