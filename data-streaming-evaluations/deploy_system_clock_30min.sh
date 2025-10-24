#!/usr/bin/env bash
# System Clock Test - 30 Minutes
# Pure system clock comparison (NO NTP)
# Goal: Measure causality violations using ONLY raw system clocks
# Then compare separately with ChronoTick

set -e

BASE_DIR="/mnt/common/jcernudagarcia/ChronoTick/data-streaming-evaluations"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
TEST_NAME="system_clock_30min_${TIMESTAMP}"

# Node configuration
COORDINATOR_NODE="ares-comp-18"
WORKER_B_NODE="ares-comp-11"
WORKER_C_NODE="ares-comp-12"

# Network configuration
WORKER_PORT=9000

# Test parameters
NUM_EVENTS=500
TEST_DURATION=1800  # 30 minutes in seconds
EVENT_INTERVAL=$(echo "scale=2; $TEST_DURATION / $NUM_EVENTS" | bc)

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $1"
}

echo -e "${BLUE}"
cat << "EOF"
╔════════════════════════════════════════════════════════════╗
║   SYSTEM CLOCK TEST - 30 MINUTES                          ║
║   Pure system clock comparison (NO NTP, NO ChronoTick)    ║
║   500 events spread over 1800 seconds (~3.6s/event)       ║
╚════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

RESULTS_DIR="$BASE_DIR/results/$TEST_NAME"
LOGS_DIR="$BASE_DIR/logs/$TEST_NAME"
mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

log_info "Test: $TEST_NAME"
log_info "Configuration:"
log_info "  Events: $NUM_EVENTS"
log_info "  Duration: $TEST_DURATION seconds (30 minutes)"
log_info "  Event interval: ~${EVENT_INTERVAL}s"
log_info ""
log_info "What this tests:"
log_info "  - Pure system clock timestamps (Clock_A, Clock_B, Clock_C)"
log_info "  - NO NTP synchronization"
log_info "  - Causality: Does Clock_B > Clock_A? (receive > send)"
echo

# Cleanup
log_info "Cleaning up existing processes..."
ssh $COORDINATOR_NODE 'pkill -f coordinator || true' 2>/dev/null || true
ssh $WORKER_B_NODE 'pkill -f worker || true' 2>/dev/null || true
ssh $WORKER_C_NODE 'pkill -f worker || true' 2>/dev/null || true
sleep 2

# Start workers (NO NTP, NO ChronoTick - just system clock)
log_info "Starting Worker B (system clock only)..."
ssh -n $WORKER_B_NODE "cd $BASE_DIR && nohup .venv/bin/worker \
    --node-id comp11 \
    --listen-port $WORKER_PORT \
    --output $RESULTS_DIR/worker_comp11.csv \
    --log-level INFO \
    < /dev/null > $LOGS_DIR/worker_comp11.log 2>&1 &"

log_info "Starting Worker C (system clock only)..."
ssh -n $WORKER_C_NODE "cd $BASE_DIR && nohup .venv/bin/worker \
    --node-id comp12 \
    --listen-port $WORKER_PORT \
    --output $RESULTS_DIR/worker_comp12.csv \
    --log-level INFO \
    < /dev/null > $LOGS_DIR/worker_comp12.log 2>&1 &"

# Warmup
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_info "WARMUP PHASE (10 seconds)"
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
sleep 10
log_info "✓ Warmup complete!"
echo

# Start coordinator (NO NTP - just system clock)
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_info "STARTING COORDINATOR (SYSTEM CLOCK ONLY)"
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

log_info "Broadcasting $NUM_EVENTS events over $TEST_DURATION seconds (~${EVENT_INTERVAL}s per event)..."
echo

ssh $COORDINATOR_NODE "cd $BASE_DIR && .venv/bin/coordinator \
    --workers $WORKER_B_NODE:$WORKER_PORT,$WORKER_C_NODE:$WORKER_PORT \
    --num-events $NUM_EVENTS \
    --target-duration $TEST_DURATION \
    --output $RESULTS_DIR/coordinator.csv \
    2>&1 | tee $LOGS_DIR/coordinator.log"

COORD_EXIT=$?

# Finalize
log_info "Stopping workers..."
ssh $WORKER_B_NODE 'pkill -f worker || true' 2>/dev/null || true
ssh $WORKER_C_NODE 'pkill -f worker || true' 2>/dev/null || true

# Results
echo
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_info "TEST COMPLETE"
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

COORD_EVENTS=$(tail -n +2 "$RESULTS_DIR/coordinator.csv" 2>/dev/null | wc -l || echo "0")
WORKER_B_EVENTS=$(tail -n +2 "$RESULTS_DIR/worker_comp11.csv" 2>/dev/null | wc -l || echo "0")
WORKER_C_EVENTS=$(tail -n +2 "$RESULTS_DIR/worker_comp12.csv" 2>/dev/null | wc -l || echo "0")

log_info "Events:"
log_info "  Coordinator sent: $COORD_EVENTS"
log_info "  Worker B received: $WORKER_B_EVENTS"
log_info "  Worker C received: $WORKER_C_EVENTS"
echo

log_info "Results: $RESULTS_DIR"
log_info "Logs: $LOGS_DIR"
echo

log_info "Data collected:"
log_info "  - Clock_A (coordinator send time)"
log_info "  - Clock_B (worker B receive time)"
log_info "  - Clock_C (worker C receive time)"
echo

log_info "Next: Run ChronoTick test for comparison"

if [ $COORD_EXIT -eq 0 ]; then
    log_info "✓ Test completed successfully!"
else
    echo -e "${YELLOW}✗ Test failed with exit code $COORD_EXIT${NC}"
fi

exit $COORD_EXIT
