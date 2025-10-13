#!/bin/bash
# Minimal Client Test Runner
#
# 1. Starts ChronoTick daemon
# 2. Runs minimal client test for 15 minutes
# 3. Sleeps for 20 minutes
# 4. Checks logs and verifies everything works

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "==================================================================="
echo "ChronoTick Minimal Client Test"
echo "==================================================================="
echo "Project root: $PROJECT_ROOT"
echo

# Stay in PROJECT_ROOT

#  Step 1: Start daemon
echo "Step 1: Starting ChronoTick daemon..."
cd "$PROJECT_ROOT"
uv run python -m chronotick.inference.real_data_pipeline backtracking 2>&1 | tee "$PROJECT_ROOT/results/validation_post_cleanup/daemon_$(date +%Y%m%d_%H%M%S).log" &
DAEMON_PID=$!
echo "Daemon started with PID: $DAEMON_PID"
echo

# Wait for daemon to initialize
echo "Waiting 10 seconds for daemon to initialize..."
sleep 10
echo

# Verify daemon is running
if kill -0 $DAEMON_PID 2>/dev/null; then
    echo "✓ Daemon is running"
else
    echo "✗ Daemon failed to start!"
    exit 1
fi
echo

# Step 2: Run minimal client test for 15 minutes
cd "$PROJECT_ROOT"
echo "Step 2: Running minimal client test (15 minutes)..."
echo "Start time: $(date)"
uv run python tests/integration/test_minimal_client.py --duration 900 &
CLIENT_PID=$!
echo "Client test started with PID: $CLIENT_PID"
echo

# Step 3: Sleep for 20 minutes total (15 min test + 5 min buffer)
echo "Step 3: Waiting for test to complete..."
echo "Will sleep for 20 minutes (1200 seconds)..."
sleep 1200
echo

# Step 4: Check logs and results
echo "==================================================================="
echo "Test Complete! Checking results..."
echo "==================================================================="
echo "End time: $(date)"
echo

# Check if processes are still running
if kill -0 $DAEMON_PID 2>/dev/null; then
    echo "✓ Daemon still running (PID: $DAEMON_PID)"
else
    echo "✗ Daemon stopped unexpectedly"
fi

if kill -0 $CLIENT_PID 2>/dev/null; then
    echo "⚠  Client test still running (should have finished)"
    echo "Waiting another 60 seconds..."
    sleep 60
else
    echo "✓ Client test finished"
fi
echo

# Find the latest results
LATEST_CSV=$(ls -t "$PROJECT_ROOT/results/validation_post_cleanup/minimal_client_*.csv" 2>/dev/null | head -1)
LATEST_LOG=$(ls -t "$PROJECT_ROOT/results/validation_post_cleanup/minimal_client_*.log" 2>/dev/null | head -1)

if [ -f "$LATEST_CSV" ]; then
    echo "Results CSV: $LATEST_CSV"
    echo "Results LOG: $LATEST_LOG"
    echo

    # Show CSV summary
    echo "CSV Summary:"
    wc -l "$LATEST_CSV"
    echo

    # Show last 20 lines of log
    echo "Last 20 lines of log:"
    tail -20 "$LATEST_LOG"
    echo

    # Quick analysis with Python
    echo "Quick Analysis:"
    python3 <<EOF
import pandas as pd
try:
    df = pd.read_csv("$LATEST_CSV")
    print(f"Total samples: {len(df)}")
    print(f"Success rate: {(df['status'] == 'ok').sum() / len(df) * 100:.1f}%")

    df_ok = df[df['status'] == 'ok']
    if len(df_ok) > 0:
        print(f"\nError statistics (ms):")
        print(f"  Mean: {df_ok['error_ms'].mean():.3f}")
        print(f"  Std:  {df_ok['error_ms'].std():.3f}")
        print(f"  Min:  {df_ok['error_ms'].min():.3f}")
        print(f"  Max:  {df_ok['error_ms'].max():.3f}")
        print(f"\nUncertainty statistics (ms):")
        print(f"  Mean: {df_ok['uncertainty_ms'].mean():.3f}")
        print(f"  Std:  {df_ok['uncertainty_ms'].std():.3f}")
except Exception as e:
    print(f"Error analyzing CSV: {e}")
EOF

else
    echo "✗ No results CSV found!"
    echo "Expected location: $PROJECT_ROOT/results/validation_post_cleanup/minimal_client_*.csv"
fi
echo

echo "==================================================================="
echo "Test sequence complete!"
echo "==================================================================="
echo
echo "To stop the daemon:"
echo "  kill $DAEMON_PID"
echo
echo "Results location:"
echo "  CSV: $LATEST_CSV"
echo "  LOG: $LATEST_LOG"
