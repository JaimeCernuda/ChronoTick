#!/usr/bin/env bash
# PROPER 30-Minute System Clock vs NTP Test
# Collects: NTP_A, Clock_A, NTP_B, Clock_B, NTP_C, Clock_C
# NTP is the REFERENCE, System Clock is what we're TESTING

set -e

BASE_DIR="/mnt/common/jcernudagarcia/ChronoTick/data-streaming-evaluations"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
TEST_NAME="proper_30min_${TIMESTAMP}"

# Node configuration
COORDINATOR_NODE="ares-comp-18"
WORKER_B_NODE="ares-comp-11"
WORKER_C_NODE="ares-comp-12"

# Network configuration
WORKER_PORT=9000
NTP_SERVER="172.20.1.1:8123"  # Multi-server NTP proxy

# Test parameters
NUM_EVENTS=3000  # More events for better statistics
TEST_DURATION=1800  # 30 minutes
EVENT_INTERVAL=$(echo "scale=3; $TEST_DURATION / $NUM_EVENTS" | bc)  # ~0.6s per event

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
╔═══════════════════════════════════════════════════════════════╗
║   PROPER 30-MINUTE TEST: System Clock vs NTP Reference       ║
║   Dataset: NTP_A, Clock_A, NTP_B, Clock_B, NTP_C, Clock_C    ║
║   3000 events @ ~0.6s/event                                   ║
╚═══════════════════════════════════════════════════════════════╝
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
log_info "  NTP server: $NTP_SERVER (multi-server with outlier rejection)"
echo
log_info "Data being collected:"
log_info "  Coordinator: NTP_A (reference), Clock_A (system clock)"
log_info "  Worker B: NTP_B (reference), Clock_B (system clock)"
log_info "  Worker C: NTP_C (reference), Clock_C (system clock)"
echo
log_info "Goal: Compare system clock causality vs NTP ground truth"
log_info "  NTP establishes: NTP_A < NTP_B and NTP_A < NTP_C"
log_info "  Test: Does Clock_A < Clock_B and Clock_A < Clock_C agree?"
echo

# Cleanup
log_info "Cleaning up existing processes..."
ssh $COORDINATOR_NODE 'pkill -f coordinator || true' 2>/dev/null || true
ssh $WORKER_B_NODE 'pkill -f worker || true' 2>/dev/null || true
ssh $WORKER_C_NODE 'pkill -f worker || true' 2>/dev/null || true
sleep 2

# Start workers WITH NTP for reference timestamps
log_info "Starting Worker B (with NTP reference)..."
ssh -n $WORKER_B_NODE "cd $BASE_DIR && nohup .venv/bin/worker \
    --node-id comp11 \
    --listen-port $WORKER_PORT \
    --ntp-server $NTP_SERVER \
    --output $RESULTS_DIR/worker_comp11.csv \
    --log-level INFO \
    < /dev/null > $LOGS_DIR/worker_comp11.log 2>&1 &"

log_info "Starting Worker C (with NTP reference)..."
ssh -n $WORKER_C_NODE "cd $BASE_DIR && nohup .venv/bin/worker \
    --node-id comp12 \
    --listen-port $WORKER_PORT \
    --ntp-server $NTP_SERVER \
    --output $RESULTS_DIR/worker_comp12.csv \
    --log-level INFO \
    < /dev/null > $LOGS_DIR/worker_comp12.log 2>&1 &"

# Warmup for NTP stabilization
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_info "WARMUP PHASE (60 seconds for NTP stabilization)"
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
for i in {10..60..10}; do
    sleep 10
    log_info "Warmup: ${i}s / 60s"
done
log_info "✓ Warmup complete - NTP measurements stabilized"
echo

# Start coordinator WITH NTP for reference timestamps
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_info "STARTING COORDINATOR (WITH NTP REFERENCE)"
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

log_info "Broadcasting $NUM_EVENTS events over $TEST_DURATION seconds (~${EVENT_INTERVAL}s per event)..."
echo

ssh $COORDINATOR_NODE "cd $BASE_DIR && .venv/bin/coordinator \
    --workers $WORKER_B_NODE:$WORKER_PORT,$WORKER_C_NODE:$WORKER_PORT \
    --num-events $NUM_EVENTS \
    --target-duration $TEST_DURATION \
    --ntp-server $NTP_SERVER \
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

log_info "Dataset collected:"
log_info "  Coordinator: NTP_A, Clock_A"
log_info "  Worker B: NTP_B, Clock_B"
log_info "  Worker C: NTP_C, Clock_C"
echo

log_info "Next: Analyze system clock violations vs NTP ground truth"

if [ $COORD_EXIT -eq 0 ]; then
    log_info "✓ Test completed successfully!"
else
    echo -e "${YELLOW}✗ Test failed with exit code $COORD_EXIT${NC}"
fi

exit $COORD_EXIT
