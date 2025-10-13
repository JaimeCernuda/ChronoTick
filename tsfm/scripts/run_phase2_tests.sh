#!/bin/bash
# Phase 2: Test SHORT-TERM ONLY vs DUAL-MODEL
# All tests use NO COVARIATES to isolate model impact

set -e

echo "================================================================================"
echo "PHASE 2: MODEL CONFIGURATION COMPARISON"
echo "================================================================================"
echo ""
echo "Testing: Does the long-term model add value?"
echo ""
echo "Tests:"
echo "  - SHORT-TERM ONLY (5 correction methods)"
echo "  - Duration: 25 minutes per test"
echo "  - No covariates for both configs"
echo ""
echo "Total runtime: ~125 minutes (2h 5min)"
echo ""
echo "Starting Phase 2 tests..."
echo ""

# Test 1: NONE - SHORT-TERM ONLY
echo "Test 1/5: NONE (baseline) - SHORT-TERM ONLY"
uv run python scripts/test_with_visualization_data.py none \
  --config chronotick_inference/config_short_only.yaml \
  --output-dir results/ntp_correction_experiment/phase2_short_only/none \
  --duration 1500 --interval 10

echo ""
sleep 10

# Test 2: LINEAR - SHORT-TERM ONLY
echo "Test 2/5: LINEAR - SHORT-TERM ONLY"
uv run python scripts/test_with_visualization_data.py linear \
  --config chronotick_inference/config_short_only.yaml \
  --output-dir results/ntp_correction_experiment/phase2_short_only/linear \
  --duration 1500 --interval 10

echo ""
sleep 10

# Test 3: DRIFT_AWARE - SHORT-TERM ONLY
echo "Test 3/5: DRIFT_AWARE - SHORT-TERM ONLY"
uv run python scripts/test_with_visualization_data.py drift_aware \
  --config chronotick_inference/config_short_only.yaml \
  --output-dir results/ntp_correction_experiment/phase2_short_only/drift_aware \
  --duration 1500 --interval 10

echo ""
sleep 10

# Test 4: ADVANCED - SHORT-TERM ONLY
echo "Test 4/5: ADVANCED - SHORT-TERM ONLY"
uv run python scripts/test_with_visualization_data.py advanced \
  --config chronotick_inference/config_short_only.yaml \
  --output-dir results/ntp_correction_experiment/phase2_short_only/advanced \
  --duration 1500 --interval 10

echo ""
sleep 10

# Test 5: ADVANCE_ABSOLUTE - SHORT-TERM ONLY
echo "Test 5/5: ADVANCE_ABSOLUTE - SHORT-TERM ONLY"
uv run python scripts/test_with_visualization_data.py advance_absolute \
  --config chronotick_inference/config_short_only.yaml \
  --output-dir results/ntp_correction_experiment/phase2_short_only/advance_absolute \
  --duration 1500 --interval 10

echo ""
echo "================================================================================"
echo "PHASE 2 COMPLETE!"
echo "================================================================================"
echo ""
echo "Results:"
echo "  SHORT-ONLY: results/ntp_correction_experiment/phase2_short_only/"
echo "  DUAL-MODEL (Phase 1): results/ntp_correction_experiment/experiment_*/"
echo ""
echo "Next: Compare results to determine winner"
echo ""
