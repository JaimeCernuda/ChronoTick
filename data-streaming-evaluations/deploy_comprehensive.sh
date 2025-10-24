#!/usr/bin/env bash
# Comprehensive data streaming evaluation with multiple test scenarios
# Tests various timing configurations and event patterns for 30-60 minutes

set -e

#================================================================
# Configuration
#================================================================

BASE_DIR="/mnt/common/jcernudagarcia/ChronoTick/data-streaming-evaluations"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# Node configuration
COORDINATOR_NODE="ares-comp-18"
WORKER_B_NODE="ares-comp-11"
WORKER_C_NODE="ares-comp-12"

# Network configuration
WORKER_PORT=9000
NTP_SERVER="172.20.1.1:8123"
CHRONOTICK_SERVER="http://172.20.1.1:8124"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $1"
}

log_section() {
    echo
    echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
    echo
}

#================================================================
# Test Scenarios
#================================================================

# Test 1: System Clock vs System Clock (30 min, 500 events)
# Baseline - no corrections applied
test_system_vs_system() {
    local name="test1_system_vs_system_${TIMESTAMP}"
    log_section "TEST 1: System Clock vs System Clock (30 min)"

    local results_dir="$BASE_DIR/results/$name"
    local logs_dir="$BASE_DIR/logs/$name"
    mkdir -p "$results_dir" "$logs_dir"

    log_info "Starting workers WITHOUT NTP/ChronoTick..."
    # Note: Workers will use fallback (system clock) when servers don't respond

    ssh -n $WORKER_B_NODE "cd $BASE_DIR && nohup .venv/bin/worker \
        --node-id comp11 \
        --listen-port $WORKER_PORT \
        --ntp-server 192.0.2.1:123 \
        --chronotick-server http://192.0.2.1:8124 \
        --output $results_dir/worker_comp11.csv \
        --log-level INFO \
        < /dev/null > $logs_dir/worker_comp11.log 2>&1 &"

    ssh -n $WORKER_C_NODE "cd $BASE_DIR && nohup .venv/bin/worker \
        --node-id comp12 \
        --listen-port $WORKER_PORT \
        --ntp-server 192.0.2.1:123 \
        --chronotick-server http://192.0.2.1:8124 \
        --output $results_dir/worker_comp12.csv \
        --log-level INFO \
        < /dev/null > $logs_dir/worker_comp12.log 2>&1 &"

    sleep 10  # Brief warmup since no real NTP

    log_info "Starting coordinator (500 events over 30 min, mixed pattern)..."
    ssh $COORDINATOR_NODE "cd $BASE_DIR && .venv/bin/coordinator \
        --workers $WORKER_B_NODE:$WORKER_PORT,$WORKER_C_NODE:$WORKER_PORT \
        --num-events 500 \
        --pattern slow,slow,fast,fast,medium,slow,fast,fast,fast,medium \
        --output $results_dir/coordinator.csv \
        2>&1 | tee $logs_dir/coordinator.log"

    log_info "Stopping workers..."
    ssh $WORKER_B_NODE 'pkill -f worker.py || true'
    ssh $WORKER_C_NODE 'pkill -f worker.py || true'

    echo "$name" >> $BASE_DIR/completed_tests.txt
    log_info "✓ Test 1 complete: $results_dir"
}

# Test 2: NTP vs NTP (30 min, 500 events)
# Both coordinator and workers use NTP
test_ntp_vs_ntp() {
    local name="test2_ntp_vs_ntp_${TIMESTAMP}"
    log_section "TEST 2: NTP vs NTP (30 min) - Fair Comparison"

    local results_dir="$BASE_DIR/results/$name"
    local logs_dir="$BASE_DIR/logs/$name"
    mkdir -p "$results_dir" "$logs_dir"

    log_info "Starting workers WITH NTP..."

    ssh -n $WORKER_B_NODE "cd $BASE_DIR && nohup .venv/bin/worker \
        --node-id comp11 \
        --listen-port $WORKER_PORT \
        --ntp-server $NTP_SERVER \
        --chronotick-server http://192.0.2.1:8124 \
        --output $results_dir/worker_comp11.csv \
        --log-level INFO \
        < /dev/null > $logs_dir/worker_comp11.log 2>&1 &"

    ssh -n $WORKER_C_NODE "cd $BASE_DIR && nohup .venv/bin/worker \
        --node-id comp12 \
        --listen-port $WORKER_PORT \
        --ntp-server $NTP_SERVER \
        --chronotick-server http://192.0.2.1:8124 \
        --output $results_dir/worker_comp12.csv \
        --log-level INFO \
        < /dev/null > $logs_dir/worker_comp12.log 2>&1 &"

    log_info "Waiting for NTP warmup (180s)..."
    sleep 180

    log_info "Starting coordinator WITH NTP (500 events over 30 min)..."
    ssh $COORDINATOR_NODE "cd $BASE_DIR && .venv/bin/coordinator \
        --workers $WORKER_B_NODE:$WORKER_PORT,$WORKER_C_NODE:$WORKER_PORT \
        --num-events 500 \
        --pattern slow,slow,fast,fast,medium,slow,fast,fast,fast,medium \
        --ntp-server $NTP_SERVER \
        --output $results_dir/coordinator.csv \
        2>&1 | tee $logs_dir/coordinator.log"

    sleep 100  # Wait for commit-wait measurements

    log_info "Stopping workers..."
    ssh $WORKER_B_NODE 'pkill -f worker.py || true'
    ssh $WORKER_C_NODE 'pkill -f worker.py || true'

    echo "$name" >> $BASE_DIR/completed_tests.txt
    log_info "✓ Test 2 complete: $results_dir"
}

# Test 3: High-frequency stress test (10 min, 1000 events)
# Fast events to test rapid clock queries
test_high_frequency() {
    local name="test3_high_frequency_${TIMESTAMP}"
    log_section "TEST 3: High-Frequency Stress (10 min, 1000 events)"

    local results_dir="$BASE_DIR/results/$name"
    local logs_dir="$BASE_DIR/logs/$name"
    mkdir -p "$results_dir" "$logs_dir"

    log_info "Starting workers WITH NTP..."

    ssh -n $WORKER_B_NODE "cd $BASE_DIR && nohup .venv/bin/worker \
        --node-id comp11 \
        --listen-port $WORKER_PORT \
        --ntp-server $NTP_SERVER \
        --chronotick-server http://192.0.2.1:8124 \
        --output $results_dir/worker_comp11.csv \
        --log-level INFO \
        < /dev/null > $logs_dir/worker_comp11.log 2>&1 &"

    ssh -n $WORKER_C_NODE "cd $BASE_DIR && nohup .venv/bin/worker \
        --node-id comp12 \
        --listen-port $WORKER_PORT \
        --ntp-server $NTP_SERVER \
        --chronotick-server http://192.0.2.1:8124 \
        --output $results_dir/worker_comp12.csv \
        --log-level INFO \
        < /dev/null > $logs_dir/worker_comp12.log 2>&1 &"

    log_info "Waiting for NTP warmup (180s)..."
    sleep 180

    log_info "Starting coordinator (1000 fast events over 10 min)..."
    ssh $COORDINATOR_NODE "cd $BASE_DIR && .venv/bin/coordinator \
        --workers $WORKER_B_NODE:$WORKER_PORT,$WORKER_C_NODE:$WORKER_PORT \
        --num-events 1000 \
        --pattern fast,fast,fast,fast,fast,fast,fast,fast,fast,medium \
        --ntp-server $NTP_SERVER \
        --output $results_dir/coordinator.csv \
        2>&1 | tee $logs_dir/coordinator.log"

    sleep 100  # Wait for commit-wait measurements

    log_info "Stopping workers..."
    ssh $WORKER_B_NODE 'pkill -f worker.py || true'
    ssh $WORKER_C_NODE 'pkill -f worker.py || true'

    echo "$name" >> $BASE_DIR/completed_tests.txt
    log_info "✓ Test 3 complete: $results_dir"
}

# Test 4: Slow-and-steady (60 min, 600 events)
# Long-duration test to observe drift
test_long_duration() {
    local name="test4_long_duration_${TIMESTAMP}"
    log_section "TEST 4: Long Duration (60 min, 600 events)"

    local results_dir="$BASE_DIR/results/$name"
    local logs_dir="$BASE_DIR/logs/$name"
    mkdir -p "$results_dir" "$logs_dir"

    log_info "Starting workers WITH NTP..."

    ssh -n $WORKER_B_NODE "cd $BASE_DIR && nohup .venv/bin/worker \
        --node-id comp11 \
        --listen-port $WORKER_PORT \
        --ntp-server $NTP_SERVER \
        --chronotick-server http://192.0.2.1:8124 \
        --output $results_dir/worker_comp11.csv \
        --log-level INFO \
        < /dev/null > $logs_dir/worker_comp11.log 2>&1 &"

    ssh -n $WORKER_C_NODE "cd $BASE_DIR && nohup .venv/bin/worker \
        --node-id comp12 \
        --listen-port $WORKER_PORT \
        --ntp-server $NTP_SERVER \
        --chronotick-server http://192.0.2.1:8124 \
        --output $results_dir/worker_comp12.csv \
        --log-level INFO \
        < /dev/null > $logs_dir/worker_comp12.log 2>&1 &"

    log_info "Waiting for NTP warmup (180s)..."
    sleep 180

    log_info "Starting coordinator (600 events over 60 min)..."
    ssh $COORDINATOR_NODE "cd $BASE_DIR && .venv/bin/coordinator \
        --workers $WORKER_B_NODE:$WORKER_PORT,$WORKER_C_NODE:$WORKER_PORT \
        --num-events 600 \
        --pattern slow,slow,slow,slow,slow,slow,slow,slow,slow,slow \
        --ntp-server $NTP_SERVER \
        --output $results_dir/coordinator.csv \
        2>&1 | tee $logs_dir/coordinator.log"

    sleep 100  # Wait for commit-wait measurements

    log_info "Stopping workers..."
    ssh $WORKER_B_NODE 'pkill -f worker.py || true'
    ssh $WORKER_C_NODE 'pkill -f worker.py || true'

    echo "$name" >> $BASE_DIR/completed_tests.txt
    log_info "✓ Test 4 complete: $results_dir"
}

#================================================================
# Main Execution
#================================================================

log_section "ChronoTick Comprehensive Evaluation Suite"
log_info "Running 4 test scenarios (total: ~2.5 hours)"
log_info "Base directory: $BASE_DIR"

# Clean up any existing processes
log_info "Cleaning up existing processes..."
ssh $COORDINATOR_NODE 'pkill -f coordinator.py || true' 2>/dev/null || true
ssh $WORKER_B_NODE 'pkill -f worker.py || true' 2>/dev/null || true
ssh $WORKER_C_NODE 'pkill -f worker.py || true' 2>/dev/null || true
sleep 2

# Run all tests
test_system_vs_system
sleep 30

test_ntp_vs_ntp
sleep 30

test_high_frequency
sleep 30

test_long_duration

#================================================================
# Summary
#================================================================

log_section "COMPREHENSIVE EVALUATION COMPLETE"
log_info "All test results saved in:"
log_info "  $BASE_DIR/results/"
echo
log_info "Completed tests:"
cat $BASE_DIR/completed_tests.txt 2>/dev/null || echo "(none recorded)"
echo
log_info "Next steps:"
log_info "  1. Run analysis on each test: .venv/bin/python analysis/generate_all_figures.py --experiment <test_name>"
log_info "  2. Compare NTP causality violations across scenarios"
log_info "  3. Look for timing patterns in logs"

exit 0
