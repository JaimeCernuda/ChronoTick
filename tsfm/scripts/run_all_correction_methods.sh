#!/bin/bash
# Run 25-minute tests for all NTP correction methods sequentially
# Total runtime: ~100 minutes (4 methods × 25 min each)

set -e

echo "=========================================="
echo "Running All NTP Correction Method Tests"
echo "=========================================="
echo "Total estimated time: ~100 minutes"
echo ""

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

# Test each method
for method in none linear drift_aware advanced; do
    echo ""
    echo "=========================================="
    echo "Starting test for method: $method"
    echo "=========================================="

    uv run python scripts/test_25min_all_methods.py "$method"

    echo ""
    echo "✓ Completed test for method: $method"
    echo ""

    # Wait 30 seconds between tests to let system stabilize
    if [ "$method" != "advanced" ]; then
        echo "Waiting 30 seconds before next test..."
        sleep 30
    fi
done

echo ""
echo "=========================================="
echo "ALL TESTS COMPLETE"
echo "=========================================="
echo ""
echo "Results saved in: results/ntp_correction_experiment/"
echo ""
echo "Files generated:"
ls -lh results/ntp_correction_experiment/ntp_correction_*_test_*.csv
