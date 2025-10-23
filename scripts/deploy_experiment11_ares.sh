#!/bin/bash
# Deploy Experiment-11: 5-Server NTP Averaging with 5-Second Prediction Interval
#
# Date: October 23, 2025
# Duration: 3 hours test (extendable to overnight)
# Platform: ARES comp-11 and comp-12
#
# Key Changes from Experiment-10:
# - 5 NTP servers (down from 10) for averaging via NTP proxy
# - MAD outlier rejection (>3σ from median)
# - Prediction interval: 5.0 seconds (was 1.0s)
# - Expected: ChronoTick error drops from 5.16ms → ~1.8ms

set -e

echo "==========================================="
echo "EXPERIMENT-11 ARES DEPLOYMENT"
echo "5-Server NTP Averaging + 5s Predictions"
echo "==========================================="

# ============================================
# STEP 1: Code already updated on master node
# ============================================

echo ""
echo "✓ Code already updated on ARES master node"

# ============================================
# STEP 2: Deploy validation test on comp-11
# ============================================

echo ""
echo "STEP 2: Deploying on ARES comp-11..."
echo "-------------------------------------------"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DURATION_MINUTES=180  # 3 hours = 180 minutes

echo "[ARES comp-11] Starting ${DURATION_MINUTES}-minute validation with new NTP averaging..."
ssh ares "ssh ares-comp-11 'cd ~/ChronoTick && \
  nohup ~/.local/bin/uv run python -u scripts/client_driven_validation.py \
  --config configs/config_experiment11_ares.yaml \
  --duration ${DURATION_MINUTES} \
  --ntp-server 172.20.1.1:8123 \
  > /tmp/experiment11_comp11_${TIMESTAMP}.log 2>&1 &'"

sleep 5

# ============================================
# STEP 3: Deploy validation test on comp-12
# ============================================

echo ""
echo "STEP 3: Deploying on ARES comp-12..."
echo "-------------------------------------------"

echo "[ARES comp-12] Starting ${DURATION_MINUTES}-minute validation with new NTP averaging..."
ssh ares "ssh ares-comp-12 'cd ~/ChronoTick && \
  nohup ~/.local/bin/uv run python -u scripts/client_driven_validation.py \
  --config configs/config_experiment11_ares.yaml \
  --duration ${DURATION_MINUTES} \
  --ntp-server 172.20.1.1:8123 \
  > /tmp/experiment11_comp12_${TIMESTAMP}.log 2>&1 &'"

sleep 5

# ============================================
# STEP 4: Verify processes are running
# ============================================

echo ""
echo "STEP 4: Verifying processes..."
echo "-------------------------------------------"

echo "[ARES comp-11] Process status:"
ssh ares "ssh ares-comp-11 'ps aux | grep client_driven_validation | grep -v grep'"

echo ""
echo "[ARES comp-12] Process status:"
ssh ares "ssh ares-comp-12 'ps aux | grep client_driven_validation | grep -v grep'"

# ============================================
# STEP 5: Monitor startup
# ============================================

echo ""
echo "STEP 5: Monitoring initial startup (30 seconds)..."
echo "-------------------------------------------"

sleep 30

echo ""
echo "[ARES comp-11] Recent log output:"
ssh ares "ssh ares-comp-11 'tail -50 /tmp/experiment11_comp11_${TIMESTAMP}.log | grep -E \"NTP_AVERAGED|NTP_AVERAGING|REJECTED|INFO|WARNING\" | tail -20'"

echo ""
echo "[ARES comp-12] Recent log output:"
ssh ares "ssh ares-comp-12 'tail -50 /tmp/experiment11_comp12_${TIMESTAMP}.log | grep -E \"NTP_AVERAGED|NTP_AVERAGING|REJECTED|INFO|WARNING\" | tail -20'"

# ============================================
# STEP 6: Final summary
# ============================================

echo ""
echo "==========================================="
echo "✓ DEPLOYMENT COMPLETE"
echo "==========================================="
echo ""
echo "Experiment-11 is running on ARES:"
echo "  Nodes: comp-11, comp-12"
echo "  Duration: ${DURATION_MINUTES} minutes (3 hours)"
echo "  Config: config_experiment11_ares.yaml"
echo "  Changes:"
echo "    - 5 NTP servers via proxy (was 10)"
echo "    - Multi-server averaging with MAD outlier rejection"
echo "    - 5-second prediction interval (was 1 second)"
echo ""
echo "Expected completion: $(date -d "+${DURATION_MINUTES} minutes" '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Monitor progress:"
echo "  ssh ares 'ssh ares-comp-11 \"tail -f /tmp/experiment11_comp11_${TIMESTAMP}.log | grep -E \\\"NTP_AVERAGED|FUSION|ERROR\\\"\"'"
echo "  ssh ares 'ssh ares-comp-12 \"tail -f /tmp/experiment11_comp12_${TIMESTAMP}.log | grep -E \\\"NTP_AVERAGED|FUSION|ERROR\\\"\"'"
echo ""
echo "Check results:"
echo "  ssh ares 'ssh ares-comp-11 \"ls -lh /tmp/chronotick_client_validation_*.csv\"'"
echo "  ssh ares 'ssh ares-comp-12 \"ls -lh /tmp/chronotick_client_validation_*.csv\"'"
echo ""
echo "Expected improvement:"
echo "  Experiment-10: ChronoTick 5.16ms error (LOSING to system clock 2.95ms)"
echo "  Experiment-11: ChronoTick ~1.8ms error (WINNING like experiment-9)"
echo ""
