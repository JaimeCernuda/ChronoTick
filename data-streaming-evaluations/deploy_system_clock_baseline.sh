#!/usr/bin/env bash
# System Clock Baseline Test
# Tests: Raw system clock vs NTP ground truth
# Duration: 30 minutes, 500 events
# Does NOT test ChronoTick (next step)

set -e

BASE_DIR="/mnt/common/jcernudagarcia/ChronoTick/data-streaming-evaluations"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
TEST_NAME="system_clock_baseline_${TIMESTAMP}"

# Node configuration
COORDINATOR_NODE="ares-comp-18"
WORKER_B_NODE="ares-comp-11"
WORKER_C_NODE="ares-comp-12"

# Network configuration
WORKER_PORT=9000
NTP_SERVER="172.20.1.1:8123"

# Test parameters
NUM_EVENTS=500
TEST_DURATION=30  # 30 minutes

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $1"
}

echo -e "${BLUE}"
cat << "EOF"
╔══════════════════════════════════════════════════════════╗
║   System Clock Baseline Test (30 minutes)               ║
║   Tests: Raw system clock vs NTP ground truth           ║
║   Does NOT test ChronoTick (that's next step)           ║
╚══════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

RESULTS_DIR="$BASE_DIR/results/$TEST_NAME"
LOGS_DIR="$BASE_DIR/logs/$TEST_NAME"
mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

log_info "Test: $TEST_NAME"
log_info "What this tests:"
log_info "  - NTP establishes ground truth: coord_ntp < worker_ntp"
log_info "  - System clock: Does coord_system < worker_system agree?"
log_info "  - Measures baseline violation rate WITHOUT ChronoTick"
echo

# Cleanup
log_info "Cleaning up existing processes..."
ssh $COORDINATOR_NODE 'pkill -f coordinator.py || true' 2>/dev/null || true
ssh $WORKER_B_NODE 'pkill -f worker.py || true' 2>/dev/null || true
ssh $WORKER_C_NODE 'pkill -f worker.py || true' 2>/dev/null || true
sleep 2

# Start workers WITH NTP (for ground truth)
log_info "Starting Worker B with NTP ground truth..."
ssh -n $WORKER_B_NODE "cd $BASE_DIR && nohup .venv/bin/worker \
    --node-id comp11 \
    --listen-port $WORKER_PORT \
    --ntp-server $NTP_SERVER \
    --chronotick-server http://192.0.2.1:8124 \
    --output $RESULTS_DIR/worker_comp11.csv \
    --log-level INFO \
    < /dev/null > $LOGS_DIR/worker_comp11.log 2>&1 &"

log_info "Starting Worker C with NTP ground truth..."
ssh -n $WORKER_C_NODE "cd $BASE_DIR && nohup .venv/bin/worker \
    --node-id comp12 \
    --listen-port $WORKER_PORT \
    --ntp-server $NTP_SERVER \
    --chronotick-server http://192.0.2.1:8124 \
    --output $RESULTS_DIR/worker_comp12.csv \
    --log-level INFO \
    < /dev/null > $LOGS_DIR/worker_comp12.log 2>&1 &"

# Warmup
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_info "WARMUP PHASE (180 seconds = 3 minutes)"
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for i in {30..180..30}; do
    sleep 30
    log_info "Warmup progress: ${i}s / 180s"
done

log_info "✓ Warmup complete!"
echo

# Start coordinator WITH NTP (for ground truth)
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_info "STARTING COORDINATOR WITH NTP GROUND TRUTH"
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

log_info "Broadcasting $NUM_EVENTS events over $TEST_DURATION minutes..."
echo

ssh $COORDINATOR_NODE "cd $BASE_DIR && .venv/bin/coordinator \
    --workers $WORKER_B_NODE:$WORKER_PORT,$WORKER_C_NODE:$WORKER_PORT \
    --num-events $NUM_EVENTS \
    --pattern slow,slow,fast,fast,medium,slow,fast,fast,fast,medium \
    --ntp-server $NTP_SERVER \
    --output $RESULTS_DIR/coordinator.csv \
    2>&1 | tee $LOGS_DIR/coordinator.log"

COORD_EXIT=$?

# Finalize
log_info "Waiting for commit-wait measurements (100s)..."
sleep 100

log_info "Stopping workers..."
ssh $WORKER_B_NODE 'pkill -f worker.py || true' 2>/dev/null || true
ssh $WORKER_C_NODE 'pkill -f worker.py || true' 2>/dev/null || true

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

log_info "Next: Analyze with NTP ground truth comparison"
log_info "  cd $BASE_DIR"
log_info "  python3 analyze_baseline.py --experiment $TEST_NAME"

if [ $COORD_EXIT -eq 0 ]; then
    log_info "✓ Test completed successfully!"
else
    echo -e "${RED}✗ Test failed with exit code $COORD_EXIT${NC}"
fi

exit $COORD_EXIT
