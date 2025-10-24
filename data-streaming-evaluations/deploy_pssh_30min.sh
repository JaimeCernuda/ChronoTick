#!/usr/bin/env bash
# System Clock Test using parallel-ssh (pssh)
# Simple and WORKING approach

set -e

BASE_DIR="/mnt/common/jcernudagarcia/ChronoTick/data-streaming-evaluations"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
TEST_NAME="pssh_30min_${TIMESTAMP}"

RESULTS_DIR="$BASE_DIR/results/$TEST_NAME"
LOGS_DIR="$BASE_DIR/logs/$TEST_NAME"
mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

echo "=========================================="
echo "SYSTEM CLOCK TEST - 30 MINUTES"
echo "Using parallel-ssh for deployment"
echo "Test: $TEST_NAME"
echo "=========================================="
echo

# Cleanup
echo "[1/5] Cleanup old processes..."
parallel-ssh -H "ares-comp-18 ares-comp-11 ares-comp-12" "pkill -9 -f 'worker\|coordinator' || true" 2>/dev/null
sleep 3
echo "✓ Cleanup complete"
echo

# Start workers using parallel-ssh
echo "[2/5] Starting workers on comp-11 and comp-12..."
echo "Worker B (comp-11)..."
ssh ares-comp-11 "cd $BASE_DIR && nohup .venv/bin/worker \
    --node-id comp11 \
    --listen-port 9000 \
    --ntp-server 172.20.1.1:8123 \
    --output $RESULTS_DIR/worker_comp11.csv \
    --log-level INFO \
    </dev/null >$LOGS_DIR/worker_comp11.log 2>&1 &" &

echo "Worker C (comp-12)..."
ssh ares-comp-12 "cd $BASE_DIR && nohup .venv/bin/worker \
    --node-id comp12 \
    --listen-port 9000 \
    --ntp-server 172.20.1.1:8123 \
    --output $RESULTS_DIR/worker_comp12.csv \
    --log-level INFO \
    </dev/null >$LOGS_DIR/worker_comp12.log 2>&1 &" &

# Wait for background SSH to complete
wait
sleep 5

# Verify workers started
if [ ! -f "$RESULTS_DIR/worker_comp11.csv" ] || [ ! -f "$RESULTS_DIR/worker_comp12.csv" ]; then
    echo "ERROR: Workers failed to start!"
    ls -l "$RESULTS_DIR/"
    exit 1
fi

echo "✓ Both workers started"
echo

# Warmup
echo "[3/5] Warmup (90 seconds)..."
for i in {15..90..15}; do
    sleep 15
    echo "  ${i}/90s..."
done
echo "✓ Warmup complete"
echo

# Start coordinator
echo "[4/5] Starting coordinator (3000 events, 30 minutes)..."
ssh ares-comp-18 "cd $BASE_DIR && .venv/bin/coordinator \
    --workers ares-comp-11:9000,ares-comp-12:9000 \
    --num-events 3000 \
    --target-duration 1800 \
    --ntp-server 172.20.1.1:8123 \
    --output $RESULTS_DIR/coordinator.csv \
    2>&1 | tee $LOGS_DIR/coordinator.log"

COORD_EXIT=$?
echo

# Cleanup workers
echo "[5/5] Stopping workers..."
parallel-ssh -H "ares-comp-11 ares-comp-12" "pkill -TERM -f worker || true" 2>/dev/null
sleep 2

# Results
echo "=========================================="
echo "TEST COMPLETE"
echo "=========================================="

COORD_EVENTS=$(tail -n +2 "$RESULTS_DIR/coordinator.csv" 2>/dev/null | wc -l || echo "0")
WORKER_B_EVENTS=$(tail -n +2 "$RESULTS_DIR/worker_comp11.csv" 2>/dev/null | wc -l || echo "0")
WORKER_C_EVENTS=$(tail -n +2 "$RESULTS_DIR/worker_comp12.csv" 2>/dev/null | wc -l || echo "0")

echo "Events collected:"
echo "  Coordinator: $COORD_EVENTS"
echo "  Worker B: $WORKER_B_EVENTS"
echo "  Worker C: $WORKER_C_EVENTS"
echo
echo "Results: $RESULTS_DIR"
echo "Logs: $LOGS_DIR"
echo

if [ $COORD_EXIT -eq 0 ] && [ "$WORKER_B_EVENTS" -gt 0 ] && [ "$WORKER_C_EVENTS" -gt 0 ]; then
    echo "✓ TEST SUCCESSFUL!"
    exit 0
else
    echo "✗ TEST FAILED"
    exit 1
fi
