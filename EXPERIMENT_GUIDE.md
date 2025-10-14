# ChronoTick Experiment Guide

Complete guide for running ChronoTick experiments, including 8-hour validation tests, configuration, and post-processing.

## Table of Contents
1. [Environment Setup](#environment-setup)
2. [Running 8-Hour Tests](#running-8-hour-tests)
3. [Configuration Files](#configuration-files)
4. [Directory Structure](#directory-structure)
5. [Dataset Files Explained](#dataset-files-explained)
6. [Post-Processing & Analysis](#post-processing--analysis)
7. [Monitoring Running Tests](#monitoring-running-tests)
8. [Troubleshooting](#troubleshooting)

---

## Environment Setup

### Prerequisites
- Python 3.10+
- `uv` package manager installed
- Network access for NTP servers (UDP port 123)
- 2-4GB RAM available
- Linux/macOS (Windows has limited multiprocessing support)

### Installation
```bash
cd /path/to/ChronoTick/tsfm

# Install dependencies (choose one based on models needed)
uv sync --extra core-models  # For Chronos/TimesFM only (recommended)
uv sync --extra ttm          # For TTM models
uv sync --extra toto         # For Toto models
uv sync --extra time-moe     # For Time-MoE models

# Install development tools
uv sync --extra dev --extra test
```

---

## Running 8-Hour Tests

### Quick Start: Standard 8-Hour Test

```bash
# Navigate to tsfm directory
cd /path/to/ChronoTick

# Create timestamped output directory
TEST_DATE=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="tsfm/results/ntp_correction_experiment/overnight_8hr_${TEST_DATE}"
mkdir -p "${OUTPUT_DIR}"

# Run 8-hour backtracking test in background
uv run python scripts/test_with_visualization_data.py backtracking \
  --config configs/config_enhanced_features.yaml \
  --output-dir "${OUTPUT_DIR}" \
  --duration 28800 \
  --interval 10 \
  > "${OUTPUT_DIR}.log" 2>&1 &

# Save PID for monitoring
echo $! > "${OUTPUT_DIR}_pid.txt"
echo "Test started! PID: $(cat ${OUTPUT_DIR}_pid.txt)"
echo "Monitor with: tail -f ${OUTPUT_DIR}.log"
```

### Test Parameters Explained

| Parameter | Description | Common Values |
|-----------|-------------|---------------|
| `backtracking` | Correction method (backtracking/drift_aware/advanced) | backtracking (recommended) |
| `--config` | Configuration file path | `configs/config_enhanced_features.yaml` |
| `--output-dir` | Where to save results | `tsfm/results/ntp_correction_experiment/<name>` |
| `--duration` | Test duration in seconds | 28800 (8 hours), 600 (10 min), 300 (5 min) |
| `--interval` | Sampling interval in seconds | 10 (recommended) |

### Test Variants

**Short Validation Tests** (recommended before 8-hour runs):
```bash
# 10-minute validation
uv run python scripts/test_with_visualization_data.py backtracking \
  --config configs/config_enhanced_features.yaml \
  --output-dir tsfm/results/ntp_correction_experiment/10min_validation \
  --duration 600 --interval 10

# 5-minute quick test
uv run python scripts/test_with_visualization_data.py backtracking \
  --config configs/config_enhanced_features.yaml \
  --output-dir tsfm/results/ntp_correction_experiment/5min_quick \
  --duration 300 --interval 10
```

**Correction Method Variants**:
```bash
# Backtracking (replaces predictions with interpolated NTP)
python scripts/test_with_visualization_data.py backtracking ...

# Drift-aware (adjusts predictions based on observed drift)
python scripts/test_with_visualization_data.py drift_aware ...

# Advanced (uses quantile-based uncertainty from TimesFM)
python scripts/test_with_visualization_data.py advanced ...
```

---

## Configuration Files

### Primary Configuration: `configs/config_enhanced_features.yaml`

This is the **recommended configuration** with all NTP improvements:

**Key Settings**:
```yaml
clock_measurement:
  ntp:
    servers:
      - pool.ntp.org
      - time.nist.gov
      - time.google.com
      - time.cloudflare.com
      - time.windows.com
      - time.apple.com
    timeout_seconds: 2.0
    max_acceptable_uncertainty: 0.100  # Relaxed from 0.010 (critical!)
    measurement_mode: advanced         # Multi-sample averaging
    use_weighted_averaging: true       # Inverse-variance weighting
  timing:
    warm_up:
      duration_seconds: 60             # Initial data collection
      measurement_interval: 5          # Collect every 5s during warmup
    normal_operation:
      measurement_interval: 180        # Collect every 3min after warmup

prediction_scheduling:
  ntp_correction:
    enabled: true
    method: backtracking               # Options: backtracking, drift_aware, none
    offset_uncertainty: 0.001          # 1ms base uncertainty
    drift_uncertainty: 0.0001          # 100μs/s drift uncertainty

short_term:
  model_name: timesfm
  enabled: true
  prediction_horizon: 30               # 30-second predictions
  inference_interval: 1.0              # Update every 1 second

long_term:
  model_name: timesfm
  enabled: true
  prediction_horizon: 60               # 60-second predictions
  inference_interval: 30.0             # Update every 30 seconds
```

### Alternative Configurations

Located in `configs/`:
- `config_test_backtracking.yaml` - Testing configuration with backtracking
- `config_enhanced_features.yaml` - **Production configuration (use this)**
- `config_cpu_only_chronos.yaml` - CPU-only with Chronos model
- `config_gpu_only_timesfm.yaml` - GPU-accelerated with TimesFM

### Critical Configuration Notes

⚠️ **Important**: The system requires ≥10 NTP measurements before starting ML predictions. With warmup settings of 60s duration and 5s intervals, this gives 12 measurements (60÷5 = 12).

⚠️ **NTP Quality Thresholds**: The `max_acceptable_uncertainty: 0.100` (100ms) is crucial. Previous value of 0.010 (10ms) caused 92% rejection rate. Current setting achieves 100% success rate with 7-8ms combined uncertainty.

---

## Directory Structure

### Recommended Organization

```
ChronoTick/
├── tsfm/
│   ├── results/
│   │   └── ntp_correction_experiment/
│   │       ├── overnight_8hr_YYYYMMDD_HHMMSS/     # Test output directory
│   │       │   ├── client_predictions_backtracking_TIMESTAMP.csv
│   │       │   ├── dataset_corrections_backtracking_TIMESTAMP.csv
│   │       │   └── ntp_ground_truth_backtracking_TIMESTAMP.csv
│   │       └── overnight_8hr_YYYYMMDD_HHMMSS.log  # Test log file
│   ├── configs/                                    # Configuration files
│   ├── scripts/
│   │   ├── test_with_visualization_data.py        # Main test script
│   │   ├── plot_8hr_results.py                    # Plotting script
│   │   └── analyze_ntp_correction_methods.py      # Analysis script
│   └── chronotick_inference/                       # Core implementation
└── server/
    └── src/chronotick/inference/
        └── real_data_pipeline.py                   # Pipeline implementation
```

### Creating New Test Directories

**Best Practice**: Use timestamped directories to keep results organized:
```bash
# Create directory with timestamp
TEST_NAME="overnight_8hr_$(date +%Y%m%d_%H%M%S)"
mkdir -p "tsfm/results/ntp_correction_experiment/${TEST_NAME}"

# Run test pointing to this directory
uv run python scripts/test_with_visualization_data.py backtracking \
  --output-dir "tsfm/results/ntp_correction_experiment/${TEST_NAME}" \
  --duration 28800 --interval 10 \
  > "tsfm/results/ntp_correction_experiment/${TEST_NAME}.log" 2>&1 &
```

---

## Dataset Files Explained

Each test generates three CSV files with timestamps. All files share the same timestamp suffix for a given test run.

### 1. `client_predictions_<method>_<timestamp>.csv`

**What it contains**: The predictions that **clients actually received** from ChronoTick.

**Purpose**: Shows what end-users see - the corrected time with uncertainty bounds.

**Columns**:
- `timestamp`: Unix timestamp (integer seconds)
- `offset_correction_ms`: Clock offset correction in milliseconds
- `drift_rate_us_per_s`: Drift rate in microseconds per second
- `offset_uncertainty_ms`: Prediction uncertainty (±1σ) in milliseconds
- `confidence`: Confidence score [0-1]
- `source`: Prediction source (cpu/gpu/ntp/fusion)

**Example row**:
```csv
1760455711,91.538,0.0,7.137,0.95,cpu
```
This means: At timestamp 1760455711, client received 91.538ms offset correction with ±7.137ms uncertainty from CPU model.

**Usage**: This is the primary data for evaluating client-facing accuracy.

---

### 2. `dataset_corrections_<method>_<timestamp>.csv`

**What it contains**: How the ML training **dataset was modified** by NTP corrections (backtracking/drift-aware).

**Purpose**: Shows the retrospective adjustments made to past predictions when new NTP measurements arrive.

**Columns**:
- `timestamp`: Unix timestamp of the correction
- `correction_method`: Method used (backtracking/drift_aware/none)
- `measurements_before`: Number of dataset entries before correction
- `measurements_after`: Number of dataset entries after correction
- `offset_change_mean_ms`: Average change in offset values
- `offset_change_std_ms`: Standard deviation of changes
- `time_range_start`: Start of correction window
- `time_range_end`: End of correction window
- `ntp_measurements_used`: Number of NTP samples used

**Example row**:
```csv
1760455711,backtracking,10,15,5.2,1.8,1760455650,1760455711,2
```
This means: At timestamp 1760455711, backtracking method used 2 NTP measurements to correct the dataset, changing 10 entries to 15 entries with average offset change of 5.2ms.

**Usage**: Analyze how corrections propagate through the training data.

---

### 3. `ntp_ground_truth_<method>_<timestamp>.csv`

**What it contains**: **Raw NTP measurements** collected during the test (ground truth).

**Purpose**: Baseline truth for evaluating prediction accuracy.

**Columns**:
- `timestamp`: Unix timestamp (integer seconds)
- `offset_ms`: Measured clock offset in milliseconds
- `uncertainty_ms`: NTP measurement uncertainty in milliseconds
- `stratum`: NTP server stratum (1-16, lower is better)
- `delay_ms`: Round-trip delay to NTP server in milliseconds
- `server_count`: Number of servers used in this measurement
- `successful_servers`: Number of servers that responded successfully

**Example row**:
```csv
1760455711,91.538,7.137,2,45.3,6,6
```
This means: At timestamp 1760455711, measured 91.538ms offset with ±7.137ms uncertainty, stratum 2, 45.3ms delay, using 6/6 servers successfully.

**Usage**: This is the **ground truth** for evaluating ChronoTick accuracy.

---

### Dataset Relationships

```
┌─────────────────────────────────────────────────────────────┐
│  Test Timeline (8 hours)                                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. NTP Measurements (every 3min after warmup)              │
│     └─> ntp_ground_truth_*.csv                              │
│                                                              │
│  2. Dataset Corrections (when new NTP arrives)              │
│     └─> dataset_corrections_*.csv                           │
│     └─> Adjusts past predictions based on NTP truth         │
│                                                              │
│  3. Client Predictions (every 10 seconds)                   │
│     └─> client_predictions_*.csv                            │
│     └─> What users actually receive                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Key Insight**: Compare `client_predictions` (what we predicted) vs `ntp_ground_truth` (actual truth) to measure ChronoTick accuracy.

---

## Post-Processing & Analysis

### 1. Generate Plots

**Primary visualization script**: `tsfm/scripts/plot_8hr_results.py`

```bash
cd /path/to/ChronoTick/tsfm

# Edit the script to point to your test results
# Modify lines 14-16:
base_path = Path("results/ntp_correction_experiment/overnight_8hr_YYYYMMDD_HHMMSS")
summary_csv = base_path / "summary_backtracking_TIMESTAMP.csv"
client_csv = base_path / "client_predictions_backtracking_TIMESTAMP.csv"

# Run plotting script
uv run python scripts/plot_8hr_results.py
```

**Output**: Generates `overnight_8hr_YYYYMMDD_HHMMSS_plots.png` with two plots:
1. **System Clock vs ChronoTick Error**: Shows improvement over uncorrected clock
2. **ChronoTick Error with Uncertainty Bands**: Shows predictions with ±1σ bounds from TimesFM

**Plot Features** (as of 2025-10-14):
- ✅ System time error (red line) shows baseline uncorrected clock
- ✅ ChronoTick error (blue line) shows our predictions
- ✅ Uncertainty bands (steelblue shading) show ±1σ prediction confidence
- ✅ No synchronization lines (removed for clarity)
- ✅ Statistics boxes show mean/std for both system and ChronoTick

---

### 2. Statistical Analysis

**Quick analysis from command line**:
```bash
# Count NTP measurements
wc -l ntp_ground_truth_*.csv

# Check NTP success rate
grep "successful_servers" ntp_ground_truth_*.csv | awk -F',' '{sum+=$7; count++} END {print "Average success rate: " sum/count " / 6 servers"}'

# Calculate mean ChronoTick error
tail -n +2 client_predictions_*.csv | awk -F',' '{sum+=sqrt(($2)^2); count++} END {print "Mean absolute error: " sum/count " ms"}'

# Check prediction uncertainty distribution
tail -n +2 client_predictions_*.csv | awk -F',' '{print $4}' | sort -n | awk '{
    arr[NR]=$1
} END {
    print "Min uncertainty: " arr[1] " ms"
    print "Median uncertainty: " arr[int(NR/2)] " ms"
    print "Max uncertainty: " arr[NR] " ms"
}'
```

---

### 3. Comprehensive Analysis Script

Use `tsfm/scripts/analyze_ntp_correction_methods.py`:

```bash
uv run python scripts/analyze_ntp_correction_methods.py \
  --base-dir results/ntp_correction_experiment \
  --test-name overnight_8hr_YYYYMMDD_HHMMSS
```

**Output**:
- Mean/median/std of prediction errors
- NTP collection statistics (success rate, uncertainty)
- Correction effectiveness (before/after comparison)
- Temporal analysis (error drift over time)

---

## Monitoring Running Tests

### Check Test Status

```bash
# Check if process is still running (using PID from earlier)
ps aux | grep $(cat overnight_8hr_YYYYMMDD_HHMMSS_pid.txt) | grep -v grep

# Monitor log file in real-time
tail -f tsfm/results/ntp_correction_experiment/overnight_8hr_YYYYMMDD_HHMMSS.log

# Check last 50 lines for recent activity
tail -50 tsfm/results/ntp_correction_experiment/overnight_8hr_YYYYMMDD_HHMMSS.log

# Check NTP success rate
grep "NTP collection:" tsfm/results/ntp_correction_experiment/overnight_8hr_YYYYMMDD_HHMMSS.log | tail -10

# Check for errors or crashes
grep -E "(ERROR|CRITICAL|Traceback)" tsfm/results/ntp_correction_experiment/overnight_8hr_YYYYMMDD_HHMMSS.log
```

### Key Health Indicators

**Healthy Test** (what to look for):
```
✅ "NTP collection: 6 successful, 0 failed"
✅ "Weighted NTP average from 6 measurements"
✅ "Final: offset=XXXμs, uncertainty=7-8ms"  # Should be <10ms
✅ "[SCHEDULER_CACHE] Caching prediction: timestamp=..."
✅ "Predictive scheduler started - ML predictions now active"
✅ NO "CRITICAL" or "ERROR" messages
```

**Warning Signs**:
```
⚠️ "NTP collection: X successful, Y failed" with Y > 0
⚠️ "uncertainty >50ms" (indicates network issues)
⚠️ "Dataset has only X measurements (need >= 10)"
⚠️ "CRITICAL: ML prediction cache miss"
```

### Monitor Schedule

Recommended monitoring intervals during 8-hour test:
- **0-15 min**: Check every 5 minutes (warmup critical period)
- **15-60 min**: Check every 15 minutes
- **1-4 hours**: Check every hour
- **4-8 hours**: Check every 2 hours
- **After completion**: Verify log shows successful completion

---

## Troubleshooting

### Issue 1: Test Crashes with "ML prediction cache miss"

**Symptom**:
```
CRITICAL: ML prediction cache miss - NO FALLBACKS IN RESEARCH MODE!
Dataset size: X measurements (need >= 10)
```

**Cause**: Scheduler started before collecting enough NTP data.

**Solution**: This was **fixed on 2025-10-14**. Ensure you're using the latest `real_data_pipeline.py` with:
```python
def _mark_warm_up_complete(self):
    # ... checks dataset size before starting scheduler
    MIN_DATASET_SIZE = 10
    if dataset_size < MIN_DATASET_SIZE:
        logger.warning(f"STAYING IN WARMUP MODE...")
        threading.Timer(5.0, self._retry_scheduler_start).start()
        return  # DON'T start scheduler yet!
```

**Verification**: Check log for:
```
"Warm-up timer fired - checking if ready to switch to predictive mode"
"Dataset populated with X NTP measurements"
"Sufficient data collected! Starting scheduler..."
"Warm-up phase complete - switching to predictive mode"
```

---

### Issue 2: High NTP Rejection Rate

**Symptom**:
```
NTP collection: 1 successful, 5 failed
```

**Cause**: `max_acceptable_uncertainty` threshold too strict (e.g., 0.010 = 10ms).

**Solution**: Ensure config has:
```yaml
clock_measurement:
  ntp:
    max_acceptable_uncertainty: 0.100  # 100ms (relaxed)
```

**Expected Result**: 100% success rate (6/6 servers) with 7-8ms combined uncertainty.

---

### Issue 3: Process Dies Silently

**Symptom**: Process exits with no error in log.

**Causes & Solutions**:

1. **Out of Memory**:
   ```bash
   # Check memory usage
   free -h

   # Reduce model memory in config
   performance:
     max_memory_mb: 1024  # Reduce from 2048
   ```

2. **Network Timeout**:
   ```bash
   # Test NTP connectivity
   ntpdate -q pool.ntp.org

   # Increase timeout in config
   clock_measurement:
     ntp:
       timeout_seconds: 5.0  # Increase from 2.0
   ```

3. **Python/UV Environment Issue**:
   ```bash
   # Recreate environment
   rm -rf .venv
   uv sync --extra core-models
   ```

---

### Issue 4: Predictions Have High Uncertainty

**Symptom**: `offset_uncertainty_ms` > 50ms in client predictions.

**Causes**:
1. **Insufficient training data**: Wait longer (>5 min) before evaluating
2. **High NTP variability**: Check `ntp_ground_truth` for uncertainty values
3. **Clock drift instability**: System clock may be unstable

**Solutions**:
```bash
# Check NTP quality
tail -100 <test>.log | grep "Final: offset=" | tail -20

# Verify dataset size
grep "Dataset size:" <test>.log | tail -10

# Check model predictions
grep "Generated.*predictions" <test>.log | tail -10
```

---

### Issue 5: Configuration File Not Found

**Symptom**:
```
FileNotFoundError: [Errno 2] No such file or directory: 'configs/config_enhanced_features.yaml'
```

**Solution**: Use absolute or correct relative path:
```bash
# From project root
cd /path/to/ChronoTick

# Use relative path from root
uv run python scripts/test_with_visualization_data.py backtracking \
  --config configs/config_enhanced_features.yaml \
  ...

# OR use absolute path
--config /path/to/ChronoTick/configs/config_enhanced_features.yaml
```

---

## Performance Benchmarks

Based on validated tests (as of 2025-10-14):

### NTP Performance
| Metric | Previous (Bad) | Current (Fixed) | Improvement |
|--------|---------------|-----------------|-------------|
| Success Rate | 8% (1/12) | 100% (6/6) | **12.5x** |
| Combined Uncertainty | 50-113ms | 7-8ms | **10-15x** |
| Servers Responding | 1-2 | 6 | **3-6x** |

### ChronoTick Accuracy
| Metric | Value | Notes |
|--------|-------|-------|
| Mean Absolute Error | ~5-10ms | Against NTP ground truth |
| Mean Uncertainty | ~7-8ms | From TimesFM quantiles |
| Prediction Latency | <1ms | Cache hit |
| Prediction Latency | ~45ms | Cache miss (with ML inference) |

### Test Durations
- **Warmup**: 60 seconds (required before predictions)
- **10-minute test**: ~11 minutes total (1min warmup + 10min test)
- **8-hour test**: ~8h 1min total (1min warmup + 8h test)

---

## Quick Reference Commands

```bash
# === SETUP ===
cd /path/to/ChronoTick/tsfm
uv sync --extra core-models

# === RUN 8-HOUR TEST ===
TEST_DIR="results/ntp_correction_experiment/overnight_8hr_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${TEST_DIR}"
uv run python scripts/test_with_visualization_data.py backtracking \
  --config configs/config_enhanced_features.yaml \
  --output-dir "${TEST_DIR}" \
  --duration 28800 --interval 10 \
  > "${TEST_DIR}.log" 2>&1 &
echo $! > "${TEST_DIR}_pid.txt"

# === MONITOR ===
tail -f "${TEST_DIR}.log"
grep "NTP collection:" "${TEST_DIR}.log" | tail -5
ps aux | grep $(cat ${TEST_DIR}_pid.txt) | grep -v grep

# === ANALYZE ===
cd tsfm
# Edit plot_8hr_results.py paths, then:
uv run python scripts/plot_8hr_results.py

# === VERIFY OUTPUT ===
ls -lh "${TEST_DIR}"/
head -5 "${TEST_DIR}"/client_predictions_*.csv
wc -l "${TEST_DIR}"/ntp_ground_truth_*.csv
```

---

## Additional Resources

- **Project Repository**: https://github.com/JaimeCernuda/ChronoTick
- **Configuration Examples**: `tsfm/configs/`
- **Test Scripts**: `tsfm/scripts/`
- **Core Implementation**: `server/src/chronotick/inference/`
- **Documentation**: `CLAUDE.md`, `design.md`, `technical.md`, `eval.md`

---

## Contact & Issues

For questions or issues with experiments:
1. Check logs for error messages
2. Verify configuration against this guide
3. Consult `CLAUDE.md` for architecture details
4. Review recent git commits for latest fixes

**Last Updated**: 2025-10-14 (Post crash fix and NTP improvements)
**Validated On**: Ubuntu 22.04 LTS (WSL2), Python 3.10, TimesFM 2.5
