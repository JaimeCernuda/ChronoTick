#!/usr/bin/env bash
# ChronoTick Test - 30 Minutes
# Tests ChronoTick AI-based timing against NTP reference

set -e

BASE_DIR="/mnt/common/jcernudagarcia/ChronoTick/data-streaming-evaluations"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
TEST_NAME="chronotick_30min_${TIMESTAMP}"

# Node configuration
COORDINATOR_NODE="ares-comp-18"
WORKER_B_NODE="ares-comp-11"
WORKER_C_NODE="ares-comp-12"

# Network configuration
WORKER_PORT=9000
NTP_SERVER="172.20.1.1:8123"

# ChronoTick configuration
CHRONOTICK_CONFIG="$BASE_DIR/configs/chronotick_config.yaml"

# Test parameters
NUM_EVENTS=3000
TEST_DURATION=1800  # 30 minutes

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
║   CHRONOTICK TEST - 30 MINUTES                               ║
║   ChronoTick AI vs NTP Reference                             ║
║   3000 events over 1800 seconds                               ║
╚═══════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

RESULTS_DIR="$BASE_DIR/results/$TEST_NAME"
LOGS_DIR="$BASE_DIR/logs/$TEST_NAME"
mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

log_info "Test: $TEST_NAME"
log_info "Results: $RESULTS_DIR"
log_info "ChronoTick config: $CHRONOTICK_CONFIG"
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

# STEP 2: START CHRONOTICK WORKERS
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_info "STEP 2: STARTING CHRONOTICK WORKERS"
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

log_info "Starting ChronoTick Worker B on $WORKER_B_NODE..."
ssh $WORKER_B_NODE "cd $BASE_DIR && HF_HOME=/mnt/common/jcernudagarcia/.cache/huggingface PYTHONPATH=/mnt/common/jcernudagarcia/ChronoTick/server/src:$BASE_DIR /mnt/common/jcernudagarcia/ChronoTick/.venv/bin/python -m src.worker_chronotick \
    --node-id comp11 \
    --listen-port $WORKER_PORT \
    --ntp-server $NTP_SERVER \
    --chronotick-config $CHRONOTICK_CONFIG \
    --output $RESULTS_DIR/worker_comp11.csv \
    --log-level INFO \
    > $LOGS_DIR/worker_comp11.log 2>&1" &

W1=$!
sleep 2

log_info "Starting ChronoTick Worker C on $WORKER_C_NODE..."
ssh $WORKER_C_NODE "cd $BASE_DIR && HF_HOME=/mnt/common/jcernudagarcia/.cache/huggingface PYTHONPATH=/mnt/common/jcernudagarcia/ChronoTick/server/src:$BASE_DIR /mnt/common/jcernudagarcia/ChronoTick/.venv/bin/python -m src.worker_chronotick \
    --node-id comp12 \
    --listen-port $WORKER_PORT \
    --ntp-server $NTP_SERVER \
    --chronotick-config $CHRONOTICK_CONFIG \
    --output $RESULTS_DIR/worker_comp12.csv \
    --log-level INFO \
    > $LOGS_DIR/worker_comp12.log 2>&1" &

W2=$!
sleep 2

log_info "Workers starting (PIDs: $W1 $W2)..."
echo

# STEP 3: WARMUP (EXTENDED FOR CHRONOTICK)
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_info "STEP 3: WARMUP PHASE (90 seconds for ChronoTick + NTP)"
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Wait full warmup period without premature validation
# (ChronoTick models take 30-60s to load before CSV files are created)
for i in {15..90..15}; do
    sleep 15
    log_info "Warmup: ${i}s / 90s (allowing model loading time)"
done

# Validate ONCE after full warmup completes
if [ ! -f "$RESULTS_DIR/worker_comp11.csv" ]; then
    log_error "Worker B failed to create CSV after 90s warmup!"
    log_error "Worker B log:"
    cat "$LOGS_DIR/worker_comp11.log" 2>/dev/null || echo "No log file"
    exit 1
fi

if [ ! -f "$RESULTS_DIR/worker_comp12.csv" ]; then
    log_error "Worker C failed to create CSV after 90s warmup!"
    log_error "Worker C log:"
    cat "$LOGS_DIR/worker_comp12.log" 2>/dev/null || echo "No log file"
    exit 1
fi

log_info "✓ Warmup complete - ChronoTick workers ready"
echo

# STEP 4: START COORDINATOR
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_info "STEP 4: STARTING COORDINATOR"
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_info "Broadcasting $NUM_EVENTS events over $TEST_DURATION seconds..."
echo

# Run coordinator in foreground (blocks until complete)
ssh $COORDINATOR_NODE "cd $BASE_DIR && PYTHONPATH=$BASE_DIR python3 -m src.coordinator \
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

log_info "Stopping ChronoTick workers gracefully..."
kill $W1 $W2 2>/dev/null || true
wait $W1 $W2 2>/dev/null || true

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
    log_info "✓ ChronoTick test completed successfully!"
    echo
    log_info "Dataset collected:"
    log_info "  Coordinator: NTP_A, Clock_A"
    log_info "  Worker B: NTP_B, Clock_B, ChronoTick_B"
    log_info "  Worker C: NTP_C, Clock_C, ChronoTick_C"
    echo
    log_info "Next: Analyze ChronoTick vs NTP reference, compare against system clock baseline"
    exit 0
else
    log_error "✗ Test had issues:"
    log_error "  Coordinator exit code: $COORD_EXIT"
    log_error "  Worker B events: $WORKER_B_EVENTS"
    log_error "  Worker C events: $WORKER_C_EVENTS"
    exit 1
fi
