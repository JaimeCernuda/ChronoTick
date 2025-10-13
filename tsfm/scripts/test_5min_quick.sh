#!/bin/bash
# Quick 5-minute test to verify corrections are working
# Run this first before the full 25-minute tests

set -e

cd "$(dirname "$0")/.."

echo "=========================================="
echo "QUICK 5-MINUTE VERIFICATION TEST"
echo "=========================================="
echo "Testing all 4 methods with 5-minute duration"
echo ""

for method in none linear drift_aware advanced; do
    echo ""
    echo "=========================================="
    echo "Testing method: $method (5 minutes)"
    echo "=========================================="

    uv run python scripts/test_25min_all_methods.py "$method" --duration 300 --interval 10

    echo ""
    echo "✓ Completed 5-min test for: $method"

    # Wait 10 seconds between tests
    if [ "$method" != "advanced" ]; then
        echo "Waiting 10 seconds before next test..."
        sleep 10
    fi
done

echo ""
echo "=========================================="
echo "5-MINUTE VERIFICATION COMPLETE"
echo "=========================================="
echo ""
echo "Check results in: results/ntp_correction_experiment/"
ls -lh results/ntp_correction_experiment/ntp_correction_*_test_*.csv | tail -4
