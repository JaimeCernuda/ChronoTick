#!/usr/bin/env bash
# SLURM-Based 30-Minute System Clock vs NTP Test
# Uses srun for proper HPC job management
# Collects: NTP_A, Clock_A, NTP_B, Clock_B, NTP_C, Clock_C

set -e

BASE_DIR="/mnt/common/jcernudagarcia/ChronoTick/data-streaming-evaluations"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
TEST_NAME="slurm_30min_${TIMESTAMP}"

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
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $1"
}

echo -e "${BLUE}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════════╗
║   SLURM-BASED 30-MINUTE TEST                                  ║
║   System Clock vs NTP Reference                               ║
║   3000 events @ ~0.6s/event                                   ║
╚═══════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

RESULTS_DIR="$BASE_DIR/results/$TEST_NAME"
LOGS_DIR="$BASE_DIR/logs/$TEST_NAME"
mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

log_info "Test: $TEST_NAME"
log_info "Using SLURM for deployment"
echo

# Cleanup any existing processes
log_info "Cleaning up existing processes..."
srun --nodes=1 --nodelist=$WORKER_B_NODE pkill -f worker || true
srun --nodes=1 --nodelist=$WORKER_C_NODE pkill -f worker || true
srun --nodes=1 --nodelist=$COORDINATOR_NODE pkill -f coordinator || true
sleep 2

# Start workers using srun (background jobs)
log_info "Starting Worker B via SLURM..."
srun --nodes=1 --nodelist=$WORKER_B_NODE --job-name=worker_b \
    --output=$LOGS_DIR/worker_comp11.log \
    bash -c "cd $BASE_DIR && .venv/bin/worker \
        --node-id comp11 \
        --listen-port $WORKER_PORT \
        --ntp-server $NTP_SERVER \
        --output $RESULTS_DIR/worker_comp11.csv \
        --log-level INFO" &

WORKER_B_PID=$!
log_info "Worker B job started (PID: $WORKER_B_PID)"

log_info "Starting Worker C via SLURM..."
srun --nodes=1 --nodelist=$WORKER_C_NODE --job-name=worker_c \
    --output=$LOGS_DIR/worker_comp12.log \
    bash -c "cd $BASE_DIR && .venv/bin/worker \
        --node-id comp12 \
        --listen-port $WORKER_PORT \
        --ntp-server $NTP_SERVER \
        --output $RESULTS_DIR/worker_comp12.csv \
        --log-level INFO" &

WORKER_C_PID=$!
log_info "Worker C job started (PID: $WORKER_C_PID)"

# Warmup for NTP stabilization
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_info "WARMUP PHASE (90 seconds for NTP stabilization)"
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
for i in {15..90..15}; do
    sleep 15
    log_info "Warmup: ${i}s / 90s"

    # Check if workers are still running
    if ! kill -0 $WORKER_B_PID 2>/dev/null; then
        echo -e "${YELLOW}WARNING: Worker B process died!${NC}"
        cat "$LOGS_DIR/worker_comp11.log"
        exit 1
    fi

    if ! kill -0 $WORKER_C_PID 2>/dev/null; then
        echo -e "${YELLOW}WARNING: Worker C process died!${NC}"
        cat "$LOGS_DIR/worker_comp12.log"
        exit 1
    fi
done
log_info "✓ Warmup complete - NTP measurements stabilized"
echo

# Verify workers created CSV files
if [ ! -f "$RESULTS_DIR/worker_comp11.csv" ]; then
    echo -e "${YELLOW}ERROR: Worker B CSV not created!${NC}"
    cat "$LOGS_DIR/worker_comp11.log"
    exit 1
fi

if [ ! -f "$RESULTS_DIR/worker_comp12.csv" ]; then
    echo -e "${YELLOW}ERROR: Worker C CSV not created!${NC}"
    cat "$LOGS_DIR/worker_comp12.log"
    exit 1
fi

log_info "✓ Both workers initialized successfully"
echo

# Start coordinator using srun (foreground)
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_info "STARTING COORDINATOR (WITH NTP REFERENCE)"
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_info "Broadcasting $NUM_EVENTS events over $TEST_DURATION seconds (~${EVENT_INTERVAL}s per event)..."
echo

srun --nodes=1 --nodelist=$COORDINATOR_NODE --job-name=coordinator \
    bash -c "cd $BASE_DIR && .venv/bin/coordinator \
        --workers $WORKER_B_NODE:$WORKER_PORT,$WORKER_C_NODE:$WORKER_PORT \
        --num-events $NUM_EVENTS \
        --target-duration $TEST_DURATION \
        --ntp-server $NTP_SERVER \
        --output $RESULTS_DIR/coordinator.csv \
        2>&1 | tee $LOGS_DIR/coordinator.log"

COORD_EXIT=$?

# Stop workers
log_info "Stopping workers..."
kill $WORKER_B_PID $WORKER_C_PID 2>/dev/null || true
wait $WORKER_B_PID $WORKER_C_PID 2>/dev/null || true

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

if [ $COORD_EXIT -eq 0 ] && [ $WORKER_B_EVENTS -gt 0 ] && [ $WORKER_C_EVENTS -gt 0 ]; then
    log_info "✓ Test completed successfully!"
else
    echo -e "${YELLOW}✗ Test had issues - check logs${NC}"
    exit 1
fi

exit 0
