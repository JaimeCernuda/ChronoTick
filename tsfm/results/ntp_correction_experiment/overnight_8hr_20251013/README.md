# ChronoTick Overnight 8-Hour Test - October 13, 2025

## Test Overview

**Test ID**: overnight_8hr_20251013
**Start Time**: 2025-10-13 00:51:12 UTC
**Duration**: 8 hours (28,800 seconds)
**Status**: Running (bash ID: 587d26)

## Purpose

This overnight test validates ChronoTick's enhanced features for long-term stability and accuracy:
1. **Enhanced NTP Protocol**: Multiple quick successive queries (100ms apart, 2-3 samples) for improved uncertainty estimation
2. **Backtracking Correction**: REPLACE ML predictions with linearly interpolated NTP ground truth
3. **Quantile Extraction**: TimesFM 2.5 prediction quantiles [0.1, 0.5, 0.9] for confidence intervals
4. **Quantile Fusion**: Inverse-variance weighted fusion of quantiles from dual models
5. **MCP Confidence Intervals**: New `get_time_with_confidence_interval` tool for AI agents

## Configuration

### Models
- **Short-term**: TimesFM 2.5 (200M params, 512 context, 30-step horizon, 1Hz updates)
- **Long-term**: TimesFM 2.5 (200M params, 512 context, 60-step horizon, 0.033Hz updates)
- **Device**: CPU (both models)
- **Fusion**: Inverse-variance weighting enabled

### Enhanced NTP Protocol
- **Mode**: Advanced (multiple samples per server)
- **Servers**: pool.ntp.org, time.nist.gov, time.google.com
- **Samples per Server**: 2-3 queries with 100ms spacing
- **Selection**: Lowest uncertainty measurement
- **Warmup**: 60 seconds @ 5-second intervals (12 measurements)
- **Operational**: 180 seconds (3 minutes) between measurements
- **Max Acceptable Uncertainty**: 100ms
- **Expected NTP Measurements**: ~160 over 8 hours

### Backtracking Correction
- **Method**: REPLACE (not adjust) predictions with interpolated NTP values
- **Trigger**: Every operational NTP measurement (every 180 seconds after warmup)
- **Scope**: All predictions between consecutive NTP measurements
- **Interpolation**: Linear interpolation between NTP_old and NTP_new
- **Warmup Behavior**: Corrections SKIPPED during 60-second warmup phase
- **Error Calculation**: Computes mean prediction error vs NTP ground truth

### Prediction Parameters
- **Client Request Interval**: 10 seconds
- **CPU Model Prediction Interval**: 5 seconds (30-step horizon)
- **GPU Model Prediction Interval**: 30 seconds (240-step horizon)
- **Expected Client Requests**: 2,880 over 8 hours
- **Expected CPU Predictions**: ~1,750 over 8 hours
- **Expected GPU Predictions**: ~960 over 8 hours

### Quantile Extraction
- **Source**: TimesFM 2.5 model quantiles [0.1, 0.5, 0.9]
- **Propagation**: PredictionResult → PredictionWithUncertainty → CorrectionWithBounds
- **Fusion**: Quantiles from both models combined using inverse-variance weights
- **Usage**: Confidence interval calculation for MCP tool

## Expected Behavior

### Warmup Phase (00:51:16 - 00:52:16)
- Collect 12 NTP measurements at 5-second intervals
- No predictions made during warmup
- No backtracking corrections applied (warmup guard enabled)
- Build initial dataset for model inference

### Operational Phase (00:52:16 - 08:51:16)
- Short-term model generates predictions every 5 seconds
- Long-term model generates predictions every 30 seconds
- Client requests time every 10 seconds
- NTP measurements every 180 seconds (3 minutes)
- Backtracking corrections trigger at each NTP measurement
- Expected ~160 backtracking correction events
- Expected ~18 predictions replaced per correction event
- Quantiles collected and fused for confidence intervals

### Expected NTP Quality
- **Uncertainty**: 15-40ms (improved from 40-50ms baseline)
- **Delay**: 40-80ms (round-trip time)
- **Server Selection**: Lowest uncertainty (typically time.google.com)
- **Samples**: 2-3 successful samples per server

### Expected Backtracking Performance
- **Predictions Replaced**: 15-20 per correction event
- **Time Span**: ~180 seconds between corrections
- **Error Magnitude**: 5-20ms (varies with clock drift)
- **Dataset Impact**: Improved alignment with NTP ground truth

## Output Files

### Main Log
- `logs/overnight_test.log`: Complete timestamped log of all events

### Visualization Data
- `visualization_data/summary_backtracking_YYYYMMDD_HHMMSS.csv`: Summary statistics per correction
- `visualization_data/client_predictions_backtracking_YYYYMMDD_HHMMSS.csv`: All client predictions with metadata
- `visualization_data/dataset_corrections_backtracking_YYYYMMDD_HHMMSS.csv`: Dataset correction events

### Analysis (Post-Test)
- TBD: Visualization scripts and analysis results will be added here

## Key Metrics to Analyze

### NTP Quality Improvements
- Compare uncertainty distribution (advanced vs simple mode)
- Measure delay variability across servers
- Server selection patterns (which server wins most often?)

### Backtracking Effectiveness
- Mean prediction error before/after correction
- Error distribution over time
- Correlation between NTP interval and error magnitude

### Model Performance
- Short-term vs long-term prediction accuracy
- Fusion weight distribution over time
- Prediction horizon vs accuracy relationship

### Quantile Quality
- Confidence interval coverage (do 90% CIs contain true value 90% of time?)
- Quantile spread evolution over time
- Fusion impact on quantile quality

### Long-Term Stability
- Drift accumulation patterns
- Warm-up vs operational accuracy
- Memory usage and performance over 8 hours

## Validation Tests (Completed)

### 10-Minute Validation Test (00:26:05 - 00:36:05)
- **Result**: SUCCESS (exit code 0)
- **Enhanced NTP**: Working (17-28ms uncertainty)
- **Backtracking**: 2 correction events, 18 predictions replaced each
- **Warmup Fix**: Confirmed (no corrections during warmup)
- **Error Tracking**: -12.97ms, -14.04ms errors captured
- **Log**: `10min_FINAL_all_features.log`

## Command to Monitor Progress

```bash
# Check test status
tail -f results/ntp_correction_experiment/overnight_8hr_20251013/logs/overnight_test.log

# Monitor bash process
# Bash ID: 587d26

# Check for completion (should complete around 08:51:16)
echo "Expected completion: 2025-10-13 08:51:16 UTC"
```

## Analysis Checklist (Post-Test)

- [ ] Verify test completed successfully (exit code 0)
- [ ] Check total NTP measurements (~160 expected)
- [ ] Verify total backtracking corrections (~160 expected)
- [ ] Analyze NTP uncertainty distribution (15-40ms range)
- [ ] Calculate mean prediction error across all corrections
- [ ] Visualize error evolution over 8 hours
- [ ] Measure quantile coverage accuracy (90% CI test)
- [ ] Compare short-term vs long-term model performance
- [ ] Generate performance summary report
- [ ] Identify any anomalies or unexpected behavior

## Feature Implementations

### Enhanced NTP (`ntp_client.py`)
- Lines 178-264: Advanced mode with multiple samples
- Lines 138-162: Sample collection and averaging
- Lines 104-134: Uncertainty calculation from samples

### Backtracking Correction (`real_data_pipeline.py`)
- Lines 1137-1158: Warmup guard and correction trigger
- Lines 847-972: DatasetManager.apply_ntp_correction() REPLACE implementation
- Lines 974-1039: Linear interpolation and error calculation

### Quantile Propagation (`tsfm_model_wrapper.py`)
- Lines 242-258: Per-timestep quantile extraction
- Lines 16-35: PredictionWithUncertainty dataclass with quantiles field

### Quantile Fusion (`real_data_pipeline.py`)
- Lines 244-265: PredictionFusionEngine quantile fusion
- Lines 174-195: Inverse-variance weighted quantile combination

### Confidence Intervals (`predictive_scheduler.py`)
- Lines 61-103: CorrectionWithBounds.get_confidence_interval() method
- Supports arbitrary confidence levels (default 0.9)

### MCP Tool (`mcp_server.py`)
- Lines 508-582: get_time_with_confidence_interval handler
- Lines 354-374: Tool registration and schema
- Returns corrected time with confidence bounds

## Notes

- This test uses the validated configuration from `config_enhanced_features.yaml`
- All features validated in 10-minute test before overnight run
- System will run unattended overnight
- Expected completion: ~08:51:16 on 2025-10-13
- Log file will continue to grow (expected ~50-100MB for 8 hours)

## Previous Test Results

See `10min_FINAL_all_features.log` for validation test that confirmed:
- Enhanced NTP working correctly
- Backtracking corrections triggering properly
- Warmup phase skipping corrections as expected
- Error tracking and logging functional
