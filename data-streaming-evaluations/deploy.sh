#!/bin/bash
# One-command deployment script for ARES cluster
# Deploys coordinator + workers, monitors execution, collects results

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
CHRONOTICK_SERVER="172.20.1.1:8124"

# Experiment configuration
NUM_EVENTS=100
BROADCAST_PATTERN="slow,fast,fast,fast,fast,medium,fast,fast,fast,fast"

# Colors for output
RED='\033[0.31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

#================================================================
# Helper Functions
#================================================================

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_node() {
    local node=$1
    log_info "Checking connectivity to $node..."
    if ssh ares "ssh $node 'echo ok'" &>/dev/null; then
        log_info "✓ $node is accessible"
        return 0
    else
        log_error "✗ Cannot reach $node"
        return 1
    fi
}

kill_processes() {
    log_info "Cleaning up any existing processes..."
    ssh ares "ssh $COORDINATOR_NODE 'pkill -f coordinator.py || true'" &>/dev/null
    ssh ares "ssh $WORKER_B_NODE 'pkill -f worker.py || true'" &>/dev/null
    ssh ares "ssh $WORKER_C_NODE 'pkill -f worker.py || true'" &>/dev/null
    sleep 2
}

#================================================================
# Pre-flight Checks
#================================================================

log_info "========================================="
log_info "Data Streaming Evaluation Deployment"
log_info "Experiment: $EXPERIMENT_NAME"
log_info "========================================="
echo

log_info "Running pre-flight checks..."

# Check node connectivity
check_node $COORDINATOR_NODE || exit 1
check_node $WORKER_B_NODE || exit 1
check_node $WORKER_C_NODE || exit 1

# Check NFS mount
log_info "Checking NFS mount..."
BASE_DIR=$(pwd)
if ssh ares "ssh $WORKER_B_NODE 'ls $BASE_DIR'" &>/dev/null; then
    log_info "✓ NFS mount verified"
else
    log_error "✗ NFS mount not accessible from worker nodes"
    exit 1
fi

# Create directories
mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

# Clean up old processes
kill_processes

log_info "Pre-flight checks complete!"
echo

#================================================================
# Deploy Workers
#================================================================

log_info "Starting workers..."

# Worker B (ares-comp-11)
log_info "Deploying worker on $WORKER_B_NODE..."
ssh ares "ssh $WORKER_B_NODE 'cd $BASE_DIR && nohup uv run worker \
    --node-id comp11 \
    --listen-port $WORKER_PORT \
    --ntp-server $NTP_SERVER \
    --chronotick-server http://$CHRONOTICK_SERVER \
    --output $RESULTS_DIR/worker_comp11.csv \
    > $LOGS_DIR/worker_comp11.log 2>&1 &'"

sleep 2

# Worker C (ares-comp-12)
log_info "Deploying worker on $WORKER_C_NODE..."
ssh ares "ssh $WORKER_C_NODE 'cd $BASE_DIR && nohup uv run worker \
    --node-id comp12 \
    --listen-port $WORKER_PORT \
    --ntp-server $NTP_SERVER \
    --chronotick-server http://$CHRONOTICK_SERVER \
    --output $RESULTS_DIR/worker_comp12.csv \
    > $LOGS_DIR/worker_comp12.log 2>&1 &'"

sleep 2

# Check if workers started
log_info "Checking worker status..."
if ssh ares "ssh $WORKER_B_NODE 'pgrep -f worker.py'" &>/dev/null; then
    log_info "✓ Worker B running on $WORKER_B_NODE"
else
    log_error "✗ Worker B failed to start"
    exit 1
fi

if ssh ares "ssh $WORKER_C_NODE 'pgrep -f worker.py'" &>/dev/null; then
    log_info "✓ Worker C running on $WORKER_C_NODE"
else
    log_error "✗ Worker C failed to start"
    exit 1
fi

log_info "Workers ready, waiting 10s for initialization..."
sleep 10
echo

#================================================================
# Deploy Coordinator
#================================================================

log_info "Starting coordinator on $COORDINATOR_NODE..."

ssh ares "ssh $COORDINATOR_NODE 'cd $BASE_DIR && uv run coordinator \
    --workers $WORKER_B_NODE:$WORKER_PORT,$WORKER_C_NODE:$WORKER_PORT \
    --num-events $NUM_EVENTS \
    --pattern $BROADCAST_PATTERN \
    --output $RESULTS_DIR/coordinator.csv \
    2>&1 | tee $LOGS_DIR/coordinator.log'"

log_info "Coordinator finished!"
echo

#================================================================
# Monitor & Collect Results
#================================================================

log_info "Waiting for workers to finish processing..."
sleep 10

# Stop workers
log_info "Stopping workers..."
kill_processes

# Collect results
log_info "Collecting results..."
log_info "Coordinator events: $(wc -l < $RESULTS_DIR/coordinator.csv) lines"
log_info "Worker B events: $(wc -l < $RESULTS_DIR/worker_comp11.csv) lines"
log_info "Worker C events: $(wc -l < $RESULTS_DIR/worker_comp12.csv) lines"

# Save metadata
cat > "$RESULTS_DIR/metadata.yaml" <<EOF
experiment_name: $EXPERIMENT_NAME
timestamp: $(date -Iseconds)
nodes:
  coordinator: $COORDINATOR_NODE
  worker_b: $WORKER_B_NODE
  worker_c: $WORKER_C_NODE
configuration:
  num_events: $NUM_EVENTS
  broadcast_pattern: "$BROADCAST_PATTERN"
  worker_port: $WORKER_PORT
  ntp_server: "$NTP_SERVER"
  chronotick_server: "$CHRONOTICK_SERVER"
EOF

log_info "Metadata saved to $RESULTS_DIR/metadata.yaml"
echo

#================================================================
# Run Analysis
#================================================================

log_info "Running analysis..."
if uv run analyze --experiment $EXPERIMENT_NAME; then
    log_info "✓ Analysis complete!"
    log_info "Figures: $RESULTS_DIR/figures/"
    log_info "Statistics: $RESULTS_DIR/statistics/"
else
    log_warn "Analysis failed or not implemented yet"
fi

echo
log_info "========================================="
log_info "Deployment Complete!"
log_info "Results: $RESULTS_DIR"
log_info "Logs: $LOGS_DIR"
log_info "========================================="
