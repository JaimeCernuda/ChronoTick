#!/bin/bash
# Deploy Experiment-10: Parallel NTP Testing with Updated Proxy
#
# Date: October 22, 2025
# Duration: 8 hours per platform
# Platforms: Homelab + ARES comp-11 + ARES comp-12
#
# New Features:
# - Parallel NTP queries (5-58x speedup)
# - Fallback with relaxed thresholds
# - Retry with exponential backoff
# - 10 diverse NTP servers

set -e

echo "=========================================="
echo "EXPERIMENT-10 DEPLOYMENT"
echo "Parallel NTP Testing with Updated Proxy"
echo "=========================================="

# ============================================
# STEP 1: Pull latest code on all platforms
# ============================================

echo ""
echo "STEP 1: Pulling latest code..."
echo "----------------------------------------"

echo "[Homelab] Pulling..."
ssh homelab "cd ~/ChronoTick && git pull origin main"

echo "[ARES comp-11] Pulling..."
ssh ares-comp-11.ccr.buffalo.edu "cd ~/ChronoTick && git pull origin main"

echo "[ARES comp-12] Pulling..."
ssh ares-comp-12.ccr.buffalo.edu "cd ~/ChronoTick && git pull origin main"

echo "✓ Code pulled on all platforms"

# ============================================
# STEP 2: Restart NTP proxy on ARES master
# ============================================

echo ""
echo "STEP 2: Restarting NTP proxy on ARES master..."
echo "----------------------------------------"

# Kill existing proxy
ssh ares-master-2.ccr.buffalo.edu "pkill -f ntp_proxy.py || true"
sleep 2

# Start new proxy with 10-server config
ssh ares-master-2.ccr.buffalo.edu "cd ~/ChronoTick && \
  nohup python3 ntp_proxy.py --config ntp_proxy_config.yaml \
  > /tmp/ntp_proxy.log 2>&1 &"

sleep 5

# Verify proxy is running
echo ""
echo "Verifying proxy status:"
ssh ares-master-2.ccr.buffalo.edu "ps aux | grep ntp_proxy.py | grep -v grep"

echo ""
echo "Proxy log (last 20 lines):"
ssh ares-master-2.ccr.buffalo.edu "tail -20 /tmp/ntp_proxy.log"

echo ""
read -p "Does the proxy show 10 workers running? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "✗ Proxy not running correctly. Aborting."
    exit 1
fi

echo "✓ NTP proxy restarted with 10 servers"

# ============================================
# STEP 3: Deploy validation tests
# ============================================

echo ""
echo "STEP 3: Deploying validation tests..."
echo "----------------------------------------"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Homelab (direct NTP access)
echo ""
echo "[Homelab] Starting 8-hour validation..."
ssh homelab "cd ~/ChronoTick && \
  nohup ~/.local/bin/uv run python -u scripts/client_driven_validation.py \
  --config configs/config_homelab_2min_ntp.yaml \
  --duration-minutes 480 \
  --log-file /tmp/homelab_parallel_${TIMESTAMP}.log \
  > /tmp/homelab_stdout_${TIMESTAMP}.log 2>&1 &"

sleep 3

# ARES comp-11 (via proxy)
echo "[ARES comp-11] Starting 8-hour validation..."
ssh ares-comp-11.ccr.buffalo.edu "cd ~/ChronoTick && \
  nohup uv run python -u scripts/client_driven_validation.py \
  --config configs/config_ares_2min_ntp.yaml \
  --duration-minutes 480 \
  --log-file /tmp/ares11_parallel_${TIMESTAMP}.log \
  --ntp-server 172.20.1.1:8123 \
  > /tmp/ares11_stdout_${TIMESTAMP}.log 2>&1 &"

sleep 3

# ARES comp-12 (via proxy)
echo "[ARES comp-12] Starting 8-hour validation..."
ssh ares-comp-12.ccr.buffalo.edu "cd ~/ChronoTick && \
  nohup uv run python -u scripts/client_driven_validation.py \
  --config configs/config_ares_2min_ntp.yaml \
  --duration-minutes 480 \
  --log-file /tmp/ares12_parallel_${TIMESTAMP}.log \
  --ntp-server 172.20.1.1:8123 \
  > /tmp/ares12_stdout_${TIMESTAMP}.log 2>&1 &"

sleep 5

# ============================================
# STEP 4: Verify all processes are running
# ============================================

echo ""
echo "STEP 4: Verifying processes..."
echo "----------------------------------------"

echo ""
echo "[Homelab] Process status:"
ssh homelab "ps aux | grep client_driven_validation | grep -v grep"

echo ""
echo "[ARES comp-11] Process status:"
ssh ares-comp-11.ccr.buffalo.edu "ps aux | grep client_driven_validation | grep -v grep"

echo ""
echo "[ARES comp-12] Process status:"
ssh ares-comp-12.ccr.buffalo.edu "ps aux | grep client_driven_validation | grep -v grep"

# ============================================
# STEP 5: Final summary
# ============================================

echo ""
echo "=========================================="
echo "✓ DEPLOYMENT COMPLETE"
echo "=========================================="
echo ""
echo "All three tests are now running:"
echo "  - Homelab:     8 hours, 10 direct NTP servers"
echo "  - ARES comp-11: 8 hours, 10 servers via proxy"
echo "  - ARES comp-12: 8 hours, 10 servers via proxy"
echo ""
echo "Expected completion: $(date -d '+8 hours' '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Monitor progress:"
echo "  ssh homelab 'tail -f /tmp/homelab_stdout_${TIMESTAMP}.log'"
echo "  ssh ares-comp-11.ccr.buffalo.edu 'tail -f /tmp/ares11_stdout_${TIMESTAMP}.log'"
echo "  ssh ares-comp-12.ccr.buffalo.edu 'tail -f /tmp/ares12_stdout_${TIMESTAMP}.log'"
echo ""
echo "Retrieve results after 8 hours:"
echo "  mkdir -p results/experiment-10/{homelab,ares-11,ares-12}"
echo "  scp homelab:/tmp/chronotick_client_validation_*.csv results/experiment-10/homelab/"
echo "  scp ares-comp-11.ccr.buffalo.edu:/tmp/chronotick_client_validation_*.csv results/experiment-10/ares-11/"
echo "  scp ares-comp-12.ccr.buffalo.edu:/tmp/chronotick_client_validation_*.csv results/experiment-10/ares-12/"
echo ""
