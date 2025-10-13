# Phase 2 Testing Plan - Model Configuration Comparison

## What We're Testing

**Question:** Does the long-term model actually add value?

**Phase 1 Results** (DUAL-MODEL, no covariates):
- ✅ DRIFT_AWARE: 9.519ms MAE (winner)
- LINEAR: 10.380ms MAE
- ADVANCED: 25.857ms MAE
- NONE: 54.886ms MAE

## Phase 2: SHORT-TERM ONLY Tests

### Configuration Changes:
**config_short_only.yaml:**
```yaml
long_term:
  enabled: false  # ← DISABLED

covariates:
  enabled: false  # ← DISABLED (same as Phase 1)
```

**config_complete.yaml** (used in Phase 1):
```yaml
long_term:
  enabled: true   # ← ENABLED

covariates:
  enabled: false  # ← DISABLED (fixed from bug where it was true but not working)
```

## Test Execution

### Run Phase 2:
```bash
./scripts/run_phase2_tests.sh
```

This will run 4 tests (NONE, LINEAR, DRIFT_AWARE, ADVANCED) with:
- **SHORT-TERM model ONLY**
- **NO covariates**
- **25 minutes per test**
- **Total: ~100 minutes**

### Output Structure:
```
results/ntp_correction_experiment/
├── phase1_dual_model/              # Already completed
│   ├── experiment_1_25min_none/
│   ├── experiment_2_25min_linear/
│   ├── experiment_3_25min_drift_aware/  ← 9.519ms
│   └── experiment_4_25min_advanced/
└── phase2_short_only/              # NEW
    ├── none/
    ├── linear/
    ├── drift_aware/
    └── advanced/
```

## Verification Features

The test script now **VERIFIES and PRINTS**:
```
Config: chronotick_inference/config_short_only.yaml
Output: results/ntp_correction_experiment/phase2_short_only/drift_aware
Duration: 1500s (25 min)
Correction method: drift_aware
Long-term model: DISABLED    ← Confirms config
Covariates: DISABLED          ← Confirms config
```

## After Phase 2

### Analysis:
Compare each method:
- **NONE**: SHORT-ONLY vs DUAL-MODEL
- **LINEAR**: SHORT-ONLY vs DUAL-MODEL
- **DRIFT_AWARE**: SHORT-ONLY vs DUAL-MODEL
- **ADVANCED**: SHORT-ONLY vs DUAL-MODEL

### Decision:
- **IF** SHORT-ONLY performs equally → Use SHORT-ONLY (simpler, faster)
- **IF** DUAL-MODEL performs better → Use DUAL-MODEL (more accurate)

## Phase 3: Covariates Testing

**After determining model config winner**, test WITH covariates:

### Important Note on Covariates:
You mentioned there was a **bug where covariates were enabled in config but not actually being passed through**.

We need to:
1. **Verify** covariates are actually being used when enabled
2. **Test** the winner config WITH and WITHOUT covariates
3. **Prove** objectively if covariates add value

## Why This Approach is Better

1. ✅ **Uses existing script** with `--config` and `--output-dir` arguments
2. ✅ **Properly organizes** results into phase-specific folders
3. ✅ **Verifies configuration** by printing what's enabled/disabled
4. ✅ **No duplicate scripts** that could have bugs
5. ✅ **Clean comparison** between SHORT-ONLY and DUAL-MODEL

## Ready to Start?

Run Phase 2:
```bash
cd /home/jcernuda/tick_project/ChronoTick/tsfm
./scripts/run_phase2_tests.sh
```

This will take ~100 minutes (1h 40min) to complete all 4 tests.
