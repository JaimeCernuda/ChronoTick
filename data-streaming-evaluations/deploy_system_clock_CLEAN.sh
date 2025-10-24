#!/usr/bin/env bash
# CLEAN System Clock Test - 30 Minutes
# Ensures no process conflicts and proper worker/coordinator synchronization

set -e

BASE_DIR="/mnt/common/jcernudagarcia/ChronoTick/data-streaming-evaluations"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
TEST_NAME="clean_system_clock_${TIMESTAMP}"

# Node configuration
COORDINATOR_NODE="ares-comp-18"
WORKER_B_NODE="ares-comp-11"
WORKER_C_NODE="ares-comp-12"

# Network configuration
WORKER_PORT=9000
NTP_SERVER="172.20.1.1:8123"

# Test parameters
NUM_EVENTS=3000
TEST_DURATION=1800  # 30 minutes
EVENT_INTERVAL=$(echo "scale=3; $TEST_DURATION / $NUM_EVENTS" | bc)

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date +%H:%M:%S)]${NC} $1"
}

echo -e "${BLUE}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════════╗
║   CLEAN SYSTEM CLOCK TEST - 30 MINUTES                       ║
║   NTP Reference vs System Clock Timestamps                    ║
║   3000 events over 1800 seconds                               ║
╚═══════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

RESULTS_DIR="$BASE_DIR/results/$TEST_NAME"
LOGS_DIR="$BASE_DIR/logs/$TEST_NAME"
mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

log_info "Test: $TEST_NAME"
log_info "Results: $RESULTS_DIR"
echo

# STEP 1: AGGRESSIVE CLEANUP
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_info "STEP 1: AGGRESSIVE CLEANUP"
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

log_info "Killing ALL coordinator and worker processes..."
ssh $COORDINATOR_NODE "pkill -9 -f 'coordinator.*ares-comp' || true" 2>/dev/null || true
ssh $WORKER_B_NODE "pkill -9 -f 'worker.*comp11' || true" 2>/dev/null || true
ssh $WORKER_C_NODE "pkill -9 -f 'worker.*comp12' || true" 2>/dev/null || true
sleep 3

log_info "Verifying no processes remain..."
COORD_PROCS=$(ssh $COORDINATOR_NODE "ps aux | grep -E 'coordinator.*ares-comp' | grep -v grep | wc -l")
WORKER_B_PROCS=$(ssh $WORKER_B_NODE "ps aux | grep -E 'worker.*comp11' | grep -v grep | wc -l")
WORKER_C_PROCS=$(ssh $WORKER_C_NODE "ps aux | grep -E 'worker.*comp12' | grep -v grep | wc -l")

if [ "$COORD_PROCS" -ne 0 ] || [ "$WORKER_B_PROCS" -ne 0 ] || [ "$WORKER_C_PROCS" -ne 0 ]; then
    log_error "Failed to kill all processes! Manual intervention required."
    exit 1
fi

log_info "✓ All processes terminated"
echo

# STEP 2: START WORKERS
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_info "STEP 2: STARTING WORKERS"
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

log_info "Starting Worker B on $WORKER_B_NODE..."
ssh -n $WORKER_B_NODE "cd $BASE_DIR && nohup .venv/bin/worker \
    --node-id comp11 \
    --listen-port $WORKER_PORT \
    --ntp-server $NTP_SERVER \
    --output $RESULTS_DIR/worker_comp11.csv \
    --log-level INFO \
    < /dev/null > $LOGS_DIR/worker_comp11.log 2>&1 &"

sleep 2

log_info "Starting Worker C on $WORKER_C_NODE..."
ssh -n $WORKER_C_NODE "cd $BASE_DIR && nohup .venv/bin/worker \
    --node-id comp12 \
    --listen-port $WORKER_PORT \
    --ntp-server $NTP_SERVER \
    --output $RESULTS_DIR/worker_comp12.csv \
    --log-level INFO \
    < /dev/null > $LOGS_DIR/worker_comp12.log 2>&1 &"

sleep 2

log_info "Verifying workers started..."
sleep 3

if [ ! -f "$RESULTS_DIR/worker_comp11.csv" ]; then
    log_error "Worker B failed to create CSV file!"
    cat "$LOGS_DIR/worker_comp11.log"
    exit 1
fi

if [ ! -f "$RESULTS_DIR/worker_comp12.csv" ]; then
    log_error "Worker C failed to create CSV file!"
    cat "$LOGS_DIR/worker_comp12.log"
    exit 1
fi

log_info "✓ Both workers started successfully"
echo

# STEP 3: WARMUP
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_info "STEP 3: WARMUP PHASE (90 seconds for NTP stabilization)"
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for i in {15..90..15}; do
    sleep 15
    log_info "Warmup: ${i}s / 90s"

    # Verify workers still alive
    WORKER_B_LINES=$(wc -l < "$RESULTS_DIR/worker_comp11.csv")
    WORKER_C_LINES=$(wc -l < "$RESULTS_DIR/worker_comp12.csv")

    if [ "$WORKER_B_LINES" -eq 0 ]; then
        log_error "Worker B died during warmup!"
        cat "$LOGS_DIR/worker_comp11.log"
        exit 1
    fi

    if [ "$WORKER_C_LINES" -eq 0 ]; then
        log_error "Worker C died during warmup!"
        cat "$LOGS_DIR/worker_comp12.log"
        exit 1
    fi
done

log_info "✓ Warmup complete - workers ready"
echo

# STEP 4: START COORDINATOR
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_info "STEP 4: STARTING COORDINATOR"
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_info "Broadcasting $NUM_EVENTS events over $TEST_DURATION seconds..."
echo

# Run coordinator in foreground (blocks until complete)
ssh $COORDINATOR_NODE "cd $BASE_DIR && .venv/bin/coordinator \
    --workers $WORKER_B_NODE:$WORKER_PORT,$WORKER_C_NODE:$WORKER_PORT \
    --num-events $NUM_EVENTS \
    --target-duration $TEST_DURATION \
    --ntp-server $NTP_SERVER \
    --output $RESULTS_DIR/coordinator.csv \
    2>&1 | tee $LOGS_DIR/coordinator.log"

COORD_EXIT=$?

# STEP 5: CLEANUP AND RESULTS
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_info "STEP 5: CLEANUP AND RESULTS"
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

log_info "Stopping workers gracefully..."
ssh $WORKER_B_NODE "pkill -TERM -f 'worker.*comp11' || true" 2>/dev/null || true
ssh $WORKER_C_NODE "pkill -TERM -f 'worker.*comp12' || true" 2>/dev/null || true
sleep 2

# Count events
COORD_EVENTS=$(tail -n +2 "$RESULTS_DIR/coordinator.csv" 2>/dev/null | wc -l || echo "0")
WORKER_B_EVENTS=$(tail -n +2 "$RESULTS_DIR/worker_comp11.csv" 2>/dev/null | wc -l || echo "0")
WORKER_C_EVENTS=$(tail -n +2 "$RESULTS_DIR/worker_comp12.csv" 2>/dev/null | wc -l || echo "0")

echo
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_info "TEST COMPLETE"
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_info "Events collected:"
log_info "  Coordinator sent: $COORD_EVENTS"
log_info "  Worker B received: $WORKER_B_EVENTS"
log_info "  Worker C received: $WORKER_C_EVENTS"
echo
log_info "Results: $RESULTS_DIR"
log_info "Logs: $LOGS_DIR"
echo

# Validate
if [ $COORD_EXIT -eq 0 ] && [ "$WORKER_B_EVENTS" -gt 0 ] && [ "$WORKER_C_EVENTS" -gt 0 ]; then
    log_info "✓ Test completed successfully!"
    exit 0
else
    log_error "✗ Test had issues:"
    log_error "  Coordinator exit code: $COORD_EXIT"
    log_error "  Worker B events: $WORKER_B_EVENTS"
    log_error "  Worker C events: $WORKER_C_EVENTS"
    exit 1
fi
