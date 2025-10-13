# ChronoTick NTP Correction - Complete Experimental Plan

## Status: Phase 1 Complete, Moving to Phase 2

---

## PHASE 1: Correction Methods (DUAL-MODEL, NO COVARIATES) ✅ COMPLETE

**Configuration:**
- Models: Short-term + Long-term (TimesFM dual-model)
- Covariates: DISABLED
- Test Duration: 25 minutes
- Methods Tested: NONE, LINEAR, DRIFT_AWARE, ADVANCED, ADVANCE_ABSOLUTE

**Results:**
1. **DRIFT_AWARE** ⭐ - 9.519ms MAE (Winner)
2. **LINEAR** - 10.380ms MAE
3. **ADVANCED** - 25.857ms MAE
4. **NONE** - 54.886ms MAE (baseline)
5. **ADVANCE_ABSOLUTE** - 70.476ms MAE (Failed)

**Conclusion:** DRIFT_AWARE is the best correction method.

---

## PHASE 2: Model Configuration Comparison (NO COVARIATES)

### Test Matrix:
Test all 4 viable correction methods with SHORT-TERM ONLY:

| Test ID | Correction Method | Short-Term | Long-Term | Covariates | Duration |
|---------|-------------------|------------|-----------|------------|----------|
| 2A      | NONE              | ✅         | ❌        | ❌         | 25 min   |
| 2B      | LINEAR            | ✅         | ❌        | ❌         | 25 min   |
| 2C      | DRIFT_AWARE       | ✅         | ❌        | ❌         | 25 min   |
| 2D      | ADVANCED          | ✅         | ❌        | ❌         | 25 min   |

**Total Runtime:** ~100 minutes (1h 40min)

### Comparison Metrics:
For each method, compare:
- SHORT-ONLY vs DUAL-MODEL (from Phase 1)
- Metrics: MAE, StdDev, Max Error
- Determine if long-term model adds value

### Decision Point:
- **IF** SHORT-ONLY performs equally well → use SHORT-ONLY (simpler, faster)
- **IF** DUAL-MODEL performs better → use DUAL-MODEL

---

## PHASE 3: Covariates Testing

### Test the WINNER configuration from Phase 2 with covariates:

**Test Matrix:**

| Test ID | Config                  | Covariates | Duration |
|---------|-------------------------|------------|----------|
| 3A      | Winner + NO covariates  | ❌         | 25 min   |
| 3B      | Winner + WITH covariates| ✅         | 25 min   |

**Covariates to test:**
- cpu_usage
- temperature
- memory_usage

**Total Runtime:** ~50 minutes

### Decision Point:
- Compare WITH vs WITHOUT covariates
- Determine final production configuration

---

## PHASE 4: Overnight Validation (8 Hours)

### Final Test:
- **Configuration:** Best from Phase 3
- **Correction Method:** Best from Phase 1 (likely DRIFT_AWARE)
- **Duration:** 8 hours (28,800 seconds)
- **Sampling:** Every 10 seconds
- **Expected NTP samples:** ~160 ground truth measurements

### Success Criteria:
- MAE remains stable over 8 hours
- No drift or degradation
- Consistent performance across different times
- Validate production readiness

---

## Summary of Experimental Variables

### Variable 1: Correction Method
- NONE, LINEAR, DRIFT_AWARE, ADVANCED ~~ADVANCE_ABSOLUTE~~
- **Winner:** DRIFT_AWARE (9.519ms MAE)

### Variable 2: Model Configuration
- SHORT-ONLY vs DUAL-MODEL (short+long)
- **Status:** Testing in Phase 2

### Variable 3: Covariates
- DISABLED vs ENABLED (cpu_usage, temperature, memory_usage)
- **Status:** Testing in Phase 3

---

## Timeline

- **Phase 1:** ✅ COMPLETE (October 11, 2025)
- **Phase 2:** 🔄 IN PROGRESS (~1h 40min)
- **Phase 3:** ⏳ PENDING (~50min)
- **Phase 4:** ⏳ PENDING (8 hours overnight)

**Total Remaining Time:** ~2.5 hours + 8-hour overnight

---

## File Organization

### Results Directory Structure:
```
results/ntp_correction_experiment/
├── phase1_dual_model_no_cov/
│   ├── experiment_1_25min_none/
│   ├── experiment_2_25min_linear/
│   ├── experiment_3_25min_drift_aware/  ← WINNER
│   ├── experiment_4_25min_advanced/
│   └── experiment_5_25min_advance_absolute/  ← FAILED
├── phase2_short_only_no_cov/
│   ├── experiment_1_25min_none/
│   ├── experiment_2_25min_linear/
│   ├── experiment_3_25min_drift_aware/
│   └── experiment_4_25min_advanced/
├── phase3_covariates/
│   ├── without_covariates/
│   └── with_covariates/
└── phase4_overnight/
    └── 8hour_final_validation/
```

---

Generated: October 11, 2025
