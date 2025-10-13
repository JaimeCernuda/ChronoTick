#!/usr/bin/env bash
#
# Run NTP Correction Comparison Tests
# Tests three correction methods: none, linear, drift_aware
#

set -e

cd "$(dirname "$0")/.."

echo "========================================================================"
echo "NTP CORRECTION COMPARISON - Testing 3 Methods"
echo "========================================================================"
echo "This will run three 25-minute tests (~75 minutes total):"
echo "  1. method=none (baseline, no correction)"
echo "  2. method=linear (linear distribution)"
echo "  3. method=drift_aware (uncertainty-based attribution)"
echo ""
echo "Results will be saved to:"
echo "  results/ntp_correction_experiment/ntp_correction_none.csv"
echo "  results/ntp_correction_experiment/ntp_correction_linear.csv"
echo "  results/ntp_correction_experiment/ntp_correction_drift_aware.csv"
echo "========================================================================"
echo ""

# Function to update config method
update_config_method() {
    local method=$1
    echo "Updating config to method=${method}..."
    # Only update the ntp_correction method line (look for it in the ntp_correction section)
    sed -i "/ntp_correction:/,/offset_uncertainty:/ s/method: [a-z_]*/method: ${method}/" chronotick_inference/config_complete.yaml
}

# Function to run test and save results
run_test() {
    local method=$1
    local output_csv="results/ntp_correction_experiment/ntp_correction_${method}.csv"

    echo ""
    echo "========================================================================"
    echo "TEST ${method^^} - Starting..."
    echo "========================================================================"

    # Update config
    update_config_method "${method}"

    # Backup the default output location
    local temp_csv="results/ntp_correction_experiment/ntp_correction_v2_test.csv"

    # Run test
    uv run python scripts/test_25min_validation_v2.py

    # Move results to method-specific file
    if [ -f "${temp_csv}" ]; then
        mv "${temp_csv}" "${output_csv}"
        echo "✓ Results saved to: ${output_csv}"
    else
        echo "⚠️  Warning: Test output not found at ${temp_csv}"
    fi

    echo ""
    echo "========================================================================"
    echo "TEST ${method^^} - Complete"
    echo "========================================================================"
    echo ""
}

# Test 1: No correction (baseline)
run_test "none"

# Test 2: Linear correction
run_test "linear"

# Test 3: Drift-aware correction
run_test "drift_aware"

echo ""
echo "========================================================================"
echo "ALL TESTS COMPLETE"
echo "========================================================================"
echo ""
echo "Results saved to:"
ls -lh results/ntp_correction_experiment/ntp_correction_*.csv
echo ""
echo "Next steps:"
echo "  1. Analyze results with: uv run python scripts/analyze_ntp_corrections.py"
echo "  2. Generate comparison graphics"
echo ""
