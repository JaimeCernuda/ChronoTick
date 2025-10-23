#!/bin/bash
# Experiment 8: Homelab 10-minute NTP test deployment script

set -e

echo "========================================="
echo "Experiment 8: Homelab 10-Minute NTP Test"
echo "========================================="
echo ""

# Step 1: Check if system NTP is currently disabled
echo "Step 1: Checking current NTP status..."
ssh homelab "timedatectl status | grep 'NTP service'"
echo ""

# Step 2: Re-enable and sync NTP
echo "Step 2: Re-enabling system NTP for clock sync..."
ssh homelab "sudo systemctl enable systemd-timesyncd && sudo systemctl start systemd-timesyncd"
echo "Waiting 30 seconds for NTP sync..."
sleep 30
ssh homelab "timedatectl status"
echo ""

# Step 3: Disable NTP again for test
echo "Step 3: Disabling system NTP for experiment..."
read -p "Press ENTER to disable system NTP and start test..."
ssh homelab "sudo systemctl stop systemd-timesyncd && sudo systemctl disable systemd-timesyncd"
echo "NTP disabled. Verifying..."
ssh homelab "timedatectl status | grep 'NTP service'"
echo ""

# Step 4: Launch test
echo "Step 4: Launching 8-hour validation test..."
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ssh homelab "cd /home/jcernuda/tick_project/ChronoTick && \
  python -u scripts/client_driven_validation.py \
  --config configs/config_homelab_10min_ntp.yaml \
  --duration-minutes 480 \
  --log-file /tmp/homelab_10min_ntp_${TIMESTAMP}.log \
  > /tmp/homelab_stdout_${TIMESTAMP}.log 2>&1 &"

sleep 3
echo ""

# Step 5: Verify test is running
echo "Step 5: Verifying test started..."
ssh homelab "ps aux | grep client_driven_validation | grep -v grep"
echo ""

echo "========================================="
echo "✅ Test deployed successfully!"
echo "========================================="
echo ""
echo "Monitor with:"
echo "  ssh homelab 'tail -f /tmp/homelab_stdout_${TIMESTAMP}.log'"
echo ""
echo "Check progress:"
echo "  ssh homelab 'wc -l /tmp/chronotick_client_validation_*.csv'"
echo ""
echo "Test will run for 8 hours."
echo "Data will be in /tmp/chronotick_client_validation_*.csv"
echo ""
