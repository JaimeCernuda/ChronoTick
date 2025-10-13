#!/usr/bin/env bash
#
# Run Full 25-Minute NTP Correction Comparison
# Tests all 4 methods: none, linear, drift_aware, advanced
#

set -e
cd "$(dirname "$0")/.."

echo "========================================================================"
echo "FULL NTP CORRECTION COMPARISON - 4 Methods × 25 Minutes"
echo "========================================================================"
echo "This will run four 25-minute tests (~100 minutes total):"
echo "  1. none (baseline, no correction)"
echo "  2. linear (linear distribution)"
echo "  3. drift_aware (uncertainty-based attribution)"
echo "  4. advanced (confidence degradation model)"
echo ""
echo "Each test captures:"
echo "  - Client predictions (what users received)"
echo "  - Dataset corrections (how ML training data was adjusted)"
echo "  - NTP ground truth comparisons"
echo "========================================================================"
echo ""

# Run tests for all 4 methods
for method in none linear drift_aware advanced; do
    echo ""
    echo "========================================================================"
    echo "TEST: ${method^^} - Starting 25-minute test..."
    echo "========================================================================"
    
    uv run python scripts/test_with_visualization_data.py "${method}" --duration 1500 --interval 10
    
    echo ""
    echo "========================================================================"
    echo "TEST: ${method^^} - Complete"
    echo "========================================================================"
    echo ""
    
    # Sleep 2 minutes between tests to ensure clean separation
    if [ "${method}" != "advanced" ]; then
        echo "Sleeping 2 minutes before next test..."
        sleep 120
    fi
done

echo ""
echo "========================================================================"
echo "ALL TESTS COMPLETE"
echo "========================================================================"
echo ""
echo "Results saved to: results/ntp_correction_experiment/visualization_data/"
ls -lh results/ntp_correction_experiment/visualization_data/*$(date +%Y%m%d)* | tail -20
echo ""
echo "Next: Run plot_correction_effects.py for each method"
echo ""
