#!/bin/bash
# Run 25-minute test for EACH method (100 minutes total)
# This is different from the 5-min test which runs 5 min per method

set -e

cd "$(dirname "$0")/.."

echo "=========================================="
echo "25-MINUTE TEST PER METHOD"
echo "=========================================="
echo "Testing all 4 methods with 25 minutes EACH"
echo "Total runtime: ~100 minutes"
echo ""

for method in none linear drift_aware advanced; do
    echo ""
    echo "=========================================="
    echo "Testing method: $method (25 minutes)"
    echo "=========================================="

    uv run python scripts/test_25min_per_method.py "$method" --duration 1500 --interval 10

    echo ""
    echo "✓ Completed 25-min test for: $method"

    # Wait 10 seconds between tests
    if [ "$method" != "advanced" ]; then
        echo "Waiting 10 seconds before next test..."
        sleep 10
    fi
done

echo ""
echo "=========================================="
echo "25-MINUTE TEST COMPLETE (ALL METHODS)"
echo "=========================================="
echo ""
echo "Check results in: results/ntp_correction_experiment/"
ls -lh results/ntp_correction_experiment/ntp_correction_*_25min_*.csv | tail -4
