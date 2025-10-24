#!/bin/bash
# Deploy Experiment-11: 5-Server NTP Averaging with 5-Second Prediction Interval
#
# Date: October 23, 2025
# Duration: 3 hours test (extendable to overnight)
# Platform: Homelab
#
# Key Changes from Experiment-10:
# - 5 NTP servers (down from 10) for averaging
# - MAD outlier rejection (>3σ from median)
# - Prediction interval: 5.0 seconds (was 1.0s)
# - Expected: ChronoTick error drops from 5.16ms → ~1.8ms

set -e

echo "==========================================="
echo "EXPERIMENT-11 DEPLOYMENT"
echo "5-Server NTP Averaging + 5s Predictions"
echo "==========================================="

# ============================================
# STEP 1: Pull latest code on homelab
# ============================================

echo ""
echo "STEP 1: Pulling latest code on homelab..."
echo "-------------------------------------------"

ssh homelab "cd ~/ChronoTick && git pull origin main"

echo "✓ Code updated on homelab"

# ============================================
# STEP 2: Deploy validation test
# ============================================

echo ""
echo "STEP 2: Deploying validation test..."
echo "-------------------------------------------"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DURATION_HOURS=3  # Start with 3-hour test

echo "[Homelab] Starting ${DURATION_HOURS}-hour validation with new NTP averaging..."
ssh homelab "cd ~/ChronoTick && \
  nohup ~/.local/bin/uv run python -u scripts/client_driven_validation.py \
  --config configs/config_experiment11_homelab.yaml \
  --duration $((DURATION_HOURS * 3600)) \
  > /tmp/experiment11_${TIMESTAMP}.log 2>&1 &"

sleep 5

# ============================================
# STEP 3: Verify process is running
# ============================================

echo ""
echo "STEP 3: Verifying process..."
echo "-------------------------------------------"

echo "[Homelab] Process status:"
ssh homelab "ps aux | grep client_driven_validation | grep -v grep"

# ============================================
# STEP 4: Monitor startup
# ============================================

echo ""
echo "STEP 4: Monitoring initial startup (30 seconds)..."
echo "-------------------------------------------"

sleep 30

echo ""
echo "[Homelab] Recent log output:"
ssh homelab "tail -50 /tmp/experiment11_${TIMESTAMP}.log | grep -E 'NTP_AVERAGED|NTP_AVERAGING|REJECTED|INFO|WARNING' | tail -20"

# ============================================
# STEP 5: Final summary
# ============================================

echo ""
echo "==========================================="
echo "✓ DEPLOYMENT COMPLETE"
echo "==========================================="
echo ""
echo "Experiment-11 is running on homelab:"
echo "  Duration: ${DURATION_HOURS} hours"
echo "  Config: config_experiment11_homelab.yaml"
echo "  Changes:"
echo "    - 5 NTP servers (was 10)"
echo "    - Multi-server averaging with MAD outlier rejection"
echo "    - 5-second prediction interval (was 1 second)"
echo ""
echo "Expected completion: $(date -d "+${DURATION_HOURS} hours" '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Monitor progress:"
echo "  ssh homelab 'tail -f /tmp/experiment11_${TIMESTAMP}.log | grep -E \"NTP_AVERAGED|FUSION|ERROR\"'"
echo ""
echo "Check results:"
echo "  ssh homelab 'ls -lh /tmp/chronotick_client_validation_*.csv'"
echo ""
echo "Expected improvement:"
echo "  Experiment-10: ChronoTick 5.16ms error (LOSING to system clock 2.95ms)"
echo "  Experiment-11: ChronoTick ~1.8ms error (WINNING like experiment-9)"
echo ""
