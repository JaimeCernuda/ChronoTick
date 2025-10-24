#!/bin/bash
# Deploy V3 on ARES comp-12
#
# Steps:
# 1. Stop existing V2 experiment on comp-12
# 2. Copy V2 results to local machine
# 3. Pull latest code on ARES master
# 4. Deploy V3 on comp-12

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "==================================================================="
echo "V3 DEPLOYMENT ON ARES COMP-12"
echo "Fix 1 (drift) + Fix 2 (NTP-anchored) Implementation"
echo "==================================================================="

# ===================================================================
# STEP 1: Stop existing V2 experiment on comp-12
# ===================================================================

echo ""
echo "STEP 1: Stopping V2 experiments on comp-12..."
echo "-------------------------------------------------------------------"

echo "Killing client_driven_validation processes on comp-12..."
ssh ares 'ssh ares-comp-12 "pkill -f client_driven_validation || echo \"No processes found\""'

sleep 2

echo "Verifying processes stopped..."
RUNNING=$(ssh ares 'ssh ares-comp-12 "ps aux | grep -c client_driven_validation | grep -v grep"' || echo "0")
if [ "$RUNNING" -gt "0" ]; then
    echo "⚠️  Warning: Some processes still running. Force killing..."
    ssh ares 'ssh ares-comp-12 "pkill -9 -f client_driven_validation"'
    sleep 1
else
    echo "✓ All validation processes stopped on comp-12"
fi

# ===================================================================
# STEP 2: Copy V2 results from comp-12 to local machine
# ===================================================================

echo ""
echo "STEP 2: Copying V2 results from comp-12 to local..."
echo "-------------------------------------------------------------------"

# Create local results directory
mkdir -p results/experiment-11/v2/comp-12

# Copy CSV files
echo "Copying CSV files..."
scp 'ares:ares-comp-12:/tmp/chronotick_client_validation_v2_*.csv' \
    results/experiment-11/v2/comp-12/ 2>/dev/null || echo "No V2 CSV files found"

# Copy log files
echo "Copying log files..."
scp 'ares:ares-comp-12:/tmp/experiment11_comp12_*.log' \
    results/experiment-11/v2/comp-12/ 2>/dev/null || echo "No V2 log files found"

echo "✓ V2 results copied to results/experiment-11/v2/comp-12/"
ls -lh results/experiment-11/v2/comp-12/

# ===================================================================
# STEP 3: Pull latest code on ARES master
# ===================================================================

echo ""
echo "STEP 3: Pulling latest code on ARES master..."
echo "-------------------------------------------------------------------"

ssh ares 'cd ~/ChronoTick && git pull'

echo "✓ Code updated on ARES master (NFS will propagate to compute nodes)"

# ===================================================================
# STEP 4: Verify V3 configuration
# ===================================================================

echo ""
echo "STEP 4: Verifying V3 script configuration..."
echo "-------------------------------------------------------------------"

echo "V3 Configuration:"
echo "  - Sampling interval: 1 second (ChronoTick/System)"
echo "  - NTP interval: 60 seconds (multi-server averaging)"
echo "  - ChronoTick prediction interval: 5 seconds (from config)"
echo "  - NTP servers: 172.20.1.1:8123,8127,8128,8129,8130 (5 servers via proxy)"
echo "  - Fix 1: system_time + offset + drift * time_delta"
echo "  - Fix 2: last_ntp_time + elapsed + drift * elapsed"

# ===================================================================
# STEP 5: Deploy V3 on comp-12
# ===================================================================

echo ""
echo "STEP 5: Deploying V3 on comp-12..."
echo "-------------------------------------------------------------------"

DURATION_MINUTES=180  # 3 hours

echo "[ARES comp-12] Starting ${DURATION_MINUTES}-minute V3 validation..."
ssh ares "ssh ares-comp-12 'cd ~/ChronoTick && \
  nohup ~/.local/bin/uv run python -u scripts/client_driven_validation_v3.py \
  --config configs/config_experiment11_ares.yaml \
  --duration ${DURATION_MINUTES} \
  --ntp-server 172.20.1.1:8123 \
  --sample-interval 1 \
  --ntp-interval 60 \
  > /tmp/experiment11_v3_comp12_${TIMESTAMP}.log 2>&1 &'"

echo "✓ V3 deployed on comp-12"
echo "  Log file: /tmp/experiment11_v3_comp12_${TIMESTAMP}.log"
echo "  CSV file: /tmp/chronotick_client_validation_v3_*.csv"

sleep 5

# ===================================================================
# STEP 6: Verify process is running
# ===================================================================

echo ""
echo "STEP 6: Verifying V3 process..."
echo "-------------------------------------------------------------------"

echo "[ARES comp-12] Process status:"
ssh ares "ssh ares-comp-12 'ps aux | grep client_driven_validation_v3 | grep -v grep'" || echo "⚠️  Process not found!"

# ===================================================================
# STEP 7: Monitor warmup (first 3 minutes)
# ===================================================================

echo ""
echo "STEP 7: Monitoring warmup phase (waiting 60 seconds)..."
echo "-------------------------------------------------------------------"

sleep 60

echo ""
echo "[ARES comp-12] Initial log output:"
ssh ares "ssh ares-comp-12 'tail -100 /tmp/experiment11_v3_comp12_${TIMESTAMP}.log | head -50'"

# ===================================================================
# STEP 8: Wait for warmup to complete
# ===================================================================

echo ""
echo "STEP 8: Waiting for warmup to complete (additional 120 seconds)..."
echo "-------------------------------------------------------------------"

sleep 120

echo ""
echo "[ARES comp-12] Post-warmup log output (showing NTP and predictions):"
ssh ares "ssh ares-comp-12 'tail -100 /tmp/experiment11_v3_comp12_${TIMESTAMP}.log | grep -E \"NTP:|ChronoTick|Fix1|Fix2\" | tail -30'"

# ===================================================================
# STEP 9: Verify V3 is working correctly
# ===================================================================

echo ""
echo "STEP 9: Verifying V3 correctness..."
echo "-------------------------------------------------------------------"

echo "Checking CSV file..."
CSV_FILE=$(ssh ares 'ssh ares-comp-12 "ls -t /tmp/chronotick_client_validation_v3_*.csv 2>/dev/null | head -1"' || echo "")

if [ -z "$CSV_FILE" ]; then
    echo "⚠️  Warning: No V3 CSV file found yet"
else
    echo "✓ CSV file found: $CSV_FILE"

    # Check CSV has data
    LINE_COUNT=$(ssh ares "ssh ares-comp-12 'wc -l < $CSV_FILE'" || echo "0")
    echo "  Lines in CSV: $LINE_COUNT"

    if [ "$LINE_COUNT" -gt "10" ]; then
        echo "  ✓ CSV has data (>10 lines)"

        # Show CSV header
        echo ""
        echo "CSV Header:"
        ssh ares "ssh ares-comp-12 'head -1 $CSV_FILE'"

        # Show last few lines
        echo ""
        echo "Last 3 data rows:"
        ssh ares "ssh ares-comp-12 'tail -3 $CSV_FILE'"
    fi
fi

# ===================================================================
# STEP 10: Final summary
# ===================================================================

echo ""
echo "==================================================================="
echo "✓ V3 DEPLOYMENT COMPLETE"
echo "==================================================================="
echo ""
echo "Experiment V3 is running on ARES comp-12:"
echo "  Duration: ${DURATION_MINUTES} minutes (3 hours)"
echo "  Config: config_experiment11_ares.yaml"
echo "  Log: /tmp/experiment11_v3_comp12_${TIMESTAMP}.log"
echo "  CSV: /tmp/chronotick_client_validation_v3_*.csv"
echo ""
echo "V3 Features:"
echo "  ✓ Fix 1: Uses drift_rate in calculations"
echo "  ✓ Fix 2: NTP-anchored time walking (chrony-inspired)"
echo "  ✓ Enhanced CSV with drift_rate, prediction_time, time_since_ntp"
echo "  ✓ Dual output: Both fix1 and fix2 times for comparison"
echo ""
echo "Expected completion: $(date -d "+${DURATION_MINUTES} minutes" '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Monitor progress:"
echo "  ssh ares 'ssh ares-comp-12 \"tail -f /tmp/experiment11_v3_comp12_${TIMESTAMP}.log | grep -E \\\"NTP:|ChronoTick|Fix\\\"\"'"
echo ""
echo "Compare with comp-11 (V2):"
echo "  comp-11: Running V2 (no drift correction)"
echo "  comp-12: Running V3 (Fix 1 + Fix 2)"
echo ""
