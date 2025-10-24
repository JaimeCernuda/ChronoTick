#!/bin/bash
# Smart deployment script with warmup detection and log monitoring
# Handles ChronoTick warmup period before starting coordinator

set -e  # Exit on error

#================================================================
# Configuration
#================================================================

EXPERIMENT_NAME="${1:-experiment-$(date +%Y%m%d-%H%M%S)}"
RESULTS_DIR="results/${EXPERIMENT_NAME}"
LOGS_DIR="logs/${EXPERIMENT_NAME}"

# Node configuration
COORDINATOR_NODE="ares-comp-18"
WORKER_B_NODE="ares-comp-11"
WORKER_C_NODE="ares-comp-12"

# Network configuration
WORKER_PORT=9000
NTP_SERVER="172.20.1.1:8123"
CHRONOTICK_SERVER="http://172.20.1.1:8124"

# Experiment configuration
NUM_EVENTS=100
TEST_DURATION_MINUTES=30  # 30-minute test (not 8 hours!)
WARMUP_DURATION=180  # 3 minutes for ChronoTick warmup

# Calculate event delay to spread over duration
# 30 minutes = 1800 seconds / 100 events = 18 seconds per event
EVENT_DELAY=$(echo "scale=3; (${TEST_DURATION_MINUTES} * 60) / ${NUM_EVENTS}" | bc)
BROADCAST_PATTERN="slow"  # Will use calculated delay

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

#================================================================
# Helper Functions
#================================================================

log_info() {
    echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[$(date +%H:%M:%S)]${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date +%H:%M:%S)]${NC} $1"
}

log_debug() {
    echo -e "${BLUE}[$(date +%H:%M:%S)]${NC} $1"
}

check_node() {
    local node=$1
    if ssh $node 'echo ok' &>/dev/null; then
        return 0
    else
        return 1
    fi
}

kill_processes() {
    log_debug "Cleaning up existing processes..."
    ssh $COORDINATOR_NODE 'pkill -f "uv run coordinator" || pkill -f coordinator.py || true' &>/dev/null
    ssh $WORKER_B_NODE 'pkill -f "uv run worker" || pkill -f worker.py || true' &>/dev/null
    ssh $WORKER_C_NODE 'pkill -f "uv run worker" || pkill -f worker.py || true' &>/dev/null
    sleep 2
}

wait_for_warmup() {
    local node=$1
    local log_file=$2
    local max_wait=200  # 200 seconds (slightly more than 180s warmup)

    log_info "Waiting for $node warmup to complete..."

    for i in $(seq 1 $max_wait); do
        # Check if warmup complete by looking for "WARMUP COMPLETE" in log
        if grep -q 'WARMUP COMPLETE' $log_file 2>/dev/null; then
            log_info "✓ $node warmup complete (${i}s)"
            return 0
        fi

        # Show progress every 10 seconds
        if [ $((i % 10)) -eq 0 ]; then
            log_debug "  Waiting for $node... (${i}s / ${max_wait}s)"

            # Show last line of log for visibility
            local last_line=$(tail -1 $log_file 2>/dev/null || echo "")
            if [ -n "$last_line" ]; then
                log_debug "  Last: $last_line"
            fi
        fi

        sleep 1
    done

    log_error "✗ $node warmup timeout after ${max_wait}s"
    return 1
}

monitor_logs() {
    local log_file=$1
    local label=$2

    # Show last 5 lines every 30 seconds in background
    while true; do
        sleep 30
        if [ -f "$log_file" ]; then
            log_debug "=== $label (last 5 lines) ==="
            tail -5 "$log_file" | while read line; do
                log_debug "  $line"
            done
        fi
    done
}

#================================================================
# Banner
#================================================================

clear
echo -e "${GREEN}"
cat << "EOF"
╔══════════════════════════════════════════════════════════╗
║   ChronoTick Data Streaming Evaluation                  ║
║   Smart Deployment with Warmup Detection                ║
╚══════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

log_info "Experiment: $EXPERIMENT_NAME"
log_info "Duration: ${TEST_DURATION_MINUTES} minutes (${NUM_EVENTS} events)"
log_info "Event delay: ${EVENT_DELAY}s"
echo

#================================================================
# Pre-flight Checks
#================================================================

log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_info "PRE-FLIGHT CHECKS"
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check connectivity
log_info "Checking node connectivity..."
for node in $COORDINATOR_NODE $WORKER_B_NODE $WORKER_C_NODE; do
    if check_node $node; then
        log_info "  ✓ $node accessible"
    else
        log_error "  ✗ $node unreachable"
        exit 1
    fi
done

# Check NFS
BASE_DIR=$(pwd)
if ssh $WORKER_B_NODE "test -d $BASE_DIR" 2>/dev/null; then
    log_info "  ✓ NFS mount verified"
else
    log_error "  ✗ NFS mount not accessible"
    exit 1
fi

# Create directories
mkdir -p "$RESULTS_DIR" "$LOGS_DIR"
log_info "  ✓ Created results directory: $RESULTS_DIR"
log_info "  ✓ Created logs directory: $LOGS_DIR"

# Clean old processes
kill_processes
log_info "  ✓ Cleaned up old processes"

echo

#================================================================
# Deploy Workers with Warmup
#================================================================

log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_info "DEPLOYING WORKERS"
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Worker B
log_info "Starting Worker B on $WORKER_B_NODE..."
WORKER_B_LOG="$BASE_DIR/$LOGS_DIR/worker_comp11.log"
ssh $WORKER_B_NODE "cd $BASE_DIR && nohup ~/.local/bin/uv run worker \
    --node-id comp11 \
    --listen-port $WORKER_PORT \
    --ntp-server $NTP_SERVER \
    --chronotick-server $CHRONOTICK_SERVER \
    --output $RESULTS_DIR/worker_comp11.csv \
    --log-level INFO \
    > $WORKER_B_LOG 2>&1 &"

sleep 2

# Worker C
log_info "Starting Worker C on $WORKER_C_NODE..."
WORKER_C_LOG="$BASE_DIR/$LOGS_DIR/worker_comp12.log"
ssh $WORKER_C_NODE "cd $BASE_DIR && nohup ~/.local/bin/uv run worker \
    --node-id comp12 \
    --listen-port $WORKER_PORT \
    --ntp-server $NTP_SERVER \
    --chronotick-server $CHRONOTICK_SERVER \
    --output $RESULTS_DIR/worker_comp12.csv \
    --log-level INFO \
    > $WORKER_C_LOG 2>&1 &"

sleep 2

# Verify workers started
log_info "Verifying worker processes..."
if ssh $WORKER_B_NODE 'pgrep -f worker.py' &>/dev/null; then
    log_info "  ✓ Worker B process running"
else
    log_error "  ✗ Worker B failed to start"
    log_error "Check log: $WORKER_B_LOG"
    exit 1
fi

if ssh $WORKER_C_NODE 'pgrep -f worker.py' &>/dev/null; then
    log_info "  ✓ Worker C process running"
else
    log_error "  ✗ Worker C failed to start"
    log_error "Check log: $WORKER_C_LOG"
    exit 1
fi

echo

#================================================================
# Wait for Warmup
#================================================================

log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_info "WARMUP PHASE (${WARMUP_DURATION}s)"
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

log_info "Workers are initializing ChronoTick and NTP..."
log_info "This ensures high-quality measurements from the start."
echo

# Wait for Worker B warmup
if ! wait_for_warmup "$WORKER_B_NODE" "$WORKER_B_LOG"; then
    log_error "Worker B warmup failed - check log: $WORKER_B_LOG"
    exit 1
fi

# Wait for Worker C warmup
if ! wait_for_warmup "$WORKER_C_NODE" "$WORKER_C_LOG"; then
    log_error "Worker C warmup failed - check log: $WORKER_C_LOG"
    exit 1
fi

log_info ""
log_info "✓ All workers ready!"
log_info ""

#================================================================
# Start Coordinator
#================================================================

log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_info "STARTING COORDINATOR"
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

log_info "Broadcasting ${NUM_EVENTS} events over ${TEST_DURATION_MINUTES} minutes..."
log_info "Event delay: ${EVENT_DELAY}s"
echo

# Create custom pattern with calculated delay
CUSTOM_PATTERN=""
for i in $(seq 1 10); do
    CUSTOM_PATTERN="${CUSTOM_PATTERN}slow,"
done
CUSTOM_PATTERN=${CUSTOM_PATTERN%,}  # Remove trailing comma

# Start coordinator (foreground, so we see output)
COORDINATOR_LOG="$BASE_DIR/$LOGS_DIR/coordinator.log"
log_info "Running coordinator on $COORDINATOR_NODE..."
log_info "(This will take approximately ${TEST_DURATION_MINUTES} minutes)"
echo

ssh $COORDINATOR_NODE "cd $BASE_DIR && ~/.local/bin/uv run coordinator \
    --workers $WORKER_B_NODE:$WORKER_PORT,$WORKER_C_NODE:$WORKER_PORT \
    --num-events $NUM_EVENTS \
    --pattern $CUSTOM_PATTERN \
    --output $RESULTS_DIR/coordinator.csv \
    2>&1 | tee $COORDINATOR_LOG"

COORDINATOR_EXIT=$?

if [ $COORDINATOR_EXIT -eq 0 ]; then
    log_info ""
    log_info "✓ Coordinator finished successfully!"
else
    log_error "✗ Coordinator failed (exit code: $COORDINATOR_EXIT)"
    log_error "Check log: $COORDINATOR_LOG"
fi

echo

#================================================================
# Wait for Workers to Finish
#================================================================

log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_info "FINALIZING"
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

log_info "Waiting for workers to finish processing and commit-wait measurements..."
log_info "(This takes ~90 seconds for commit-wait T+60s measurements)"
sleep 100

# Stop workers gracefully
log_info "Stopping workers..."
kill_processes

echo

#================================================================
# Collect Results
#================================================================

log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_info "RESULTS SUMMARY"
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Count events
COORD_EVENTS=$(tail -n +2 "$RESULTS_DIR/coordinator.csv" 2>/dev/null | wc -l || echo "0")
WORKER_B_EVENTS=$(tail -n +2 "$RESULTS_DIR/worker_comp11.csv" 2>/dev/null | wc -l || echo "0")
WORKER_C_EVENTS=$(tail -n +2 "$RESULTS_DIR/worker_comp12.csv" 2>/dev/null | wc -l || echo "0")

log_info "Events collected:"
log_info "  Coordinator sent: $COORD_EVENTS"
log_info "  Worker B received: $WORKER_B_EVENTS"
log_info "  Worker C received: $WORKER_C_EVENTS"

# Calculate success rate
if [ "$COORD_EVENTS" -gt 0 ]; then
    SUCCESS_B=$(echo "scale=1; ($WORKER_B_EVENTS / $COORD_EVENTS) * 100" | bc)
    SUCCESS_C=$(echo "scale=1; ($WORKER_C_EVENTS / $COORD_EVENTS) * 100" | bc)
    log_info "  Worker B success: ${SUCCESS_B}%"
    log_info "  Worker C success: ${SUCCESS_C}%"
fi

echo

# Save metadata
cat > "$RESULTS_DIR/metadata.yaml" <<EOF
experiment_name: $EXPERIMENT_NAME
timestamp: $(date -Iseconds)
duration_minutes: $TEST_DURATION_MINUTES
num_events: $NUM_EVENTS
event_delay_seconds: $EVENT_DELAY

nodes:
  coordinator: $COORDINATOR_NODE
  worker_b: $WORKER_B_NODE
  worker_c: $WORKER_C_NODE

configuration:
  worker_port: $WORKER_PORT
  ntp_server: "$NTP_SERVER"
  chronotick_server: "$CHRONOTICK_SERVER"
  warmup_duration: ${WARMUP_DURATION}s

results:
  coordinator_events: $COORD_EVENTS
  worker_b_events: $WORKER_B_EVENTS
  worker_c_events: $WORKER_C_EVENTS
  worker_b_success_rate: ${SUCCESS_B}%
  worker_c_success_rate: ${SUCCESS_C}%

logs:
  coordinator: $COORDINATOR_LOG
  worker_b: $WORKER_B_LOG
  worker_c: $WORKER_C_LOG
EOF

log_info "Metadata saved: $RESULTS_DIR/metadata.yaml"

#================================================================
# Analysis (if available)
#================================================================

echo
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_info "ANALYSIS"
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -f "analysis/generate_all_figures.py" ]; then
    log_info "Running analysis..."
    if uv run python analysis/generate_all_figures.py --experiment "$EXPERIMENT_NAME" 2>&1 | tee "$LOGS_DIR/analysis.log"; then
        log_info "✓ Analysis complete!"
        log_info "  Figures: $RESULTS_DIR/figures/"
        log_info "  Statistics: $RESULTS_DIR/statistics/"
    else
        log_warn "⚠ Analysis failed - check $LOGS_DIR/analysis.log"
    fi
else
    log_warn "⚠ Analysis script not found - skipping"
    log_info "  Results are ready for manual analysis"
fi

#================================================================
# Summary
#================================================================

echo
echo -e "${GREEN}"
cat << "EOF"
╔══════════════════════════════════════════════════════════╗
║              DEPLOYMENT COMPLETE                         ║
╚══════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

log_info "Experiment: $EXPERIMENT_NAME"
log_info "Results: $RESULTS_DIR"
log_info "Logs: $LOGS_DIR"
echo
log_info "Next steps:"
log_info "  1. Review logs for any warnings"
log_info "  2. Check results/figures/ for visualizations"
log_info "  3. See NARRATIVE.md for paper storytelling"
echo
log_info "To view logs:"
log_info "  tail -100 $LOGS_DIR/coordinator.log"
log_info "  tail -100 $LOGS_DIR/worker_comp11.log"
log_info "  tail -100 $LOGS_DIR/worker_comp12.log"
echo

exit 0
