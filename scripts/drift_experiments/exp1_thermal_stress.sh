#!/bin/bash
# Drift Perturbation Experiment 1: CPU Thermal Stress
#
# Goal: Induce temperature changes to alter crystal oscillator frequency
# Duration: 30 minutes
#
# Timeline:
#   0-5 min:   Baseline (normal operation)
#   5-20 min:  CPU thermal stress
#   20-30 min: Cool down period
#
# Expected: Drift rate changes as temperature affects crystal frequency

set -e

DURATION=1800  # 30 minutes total
STRESS_START=300  # Start stress at 5 min
STRESS_END=1200   # End stress at 20 min
STRESS_DURATION=$((STRESS_END - STRESS_START))

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="/tmp/drift_exp1_thermal_${TIMESTAMP}.log"
CSV_OUTPUT="/tmp/drift_exp1_thermal_${TIMESTAMP}.csv"

echo "========================================" | tee -a "$LOGFILE"
echo "Drift Experiment 1: CPU Thermal Stress" | tee -a "$LOGFILE"
echo "Start time: $(date)" | tee -a "$LOGFILE"
echo "========================================" | tee -a "$LOGFILE"

# Ensure stress-ng is installed
if ! command -v stress-ng &> /dev/null; then
    echo "Installing stress-ng..." | tee -a "$LOGFILE"
    sudo apt-get update && sudo apt-get install -y stress-ng
fi

# Ensure systemd-timesyncd is stopped (unsynchronized mode)
echo "Disabling NTP sync..." | tee -a "$LOGFILE"
sudo systemctl stop systemd-timesyncd
sudo systemctl disable systemd-timesyncd

# Monitor system stats in background
(
    echo "timestamp,cpu_temp_c,cpu_freq_mhz,load_avg" > /tmp/system_stats_${TIMESTAMP}.csv
    while true; do
        TEMP=$(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | head -1)
        TEMP_C=$((TEMP / 1000))
        FREQ=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null || echo "0")
        FREQ_MHZ=$((FREQ / 1000))
        LOAD=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | tr -d ',')
        echo "$(date +%s),$TEMP_C,$FREQ_MHZ,$LOAD"
        sleep 5
    done >> /tmp/system_stats_${TIMESTAMP}.csv
) &
STATS_PID=$!

# Start ChronoTick validation in background
echo "Starting ChronoTick validation..." | tee -a "$LOGFILE"
cd ~/ChronoTick
nohup ~/.local/bin/uv run python -u scripts/client_driven_validation_v3.py \
    --config configs/config_experiment11_homelab.yaml \
    --duration $DURATION \
    --ntp-server pool.ntp.org \
    --sample-interval 1 \
    --ntp-interval 60 \
    --output "$CSV_OUTPUT" \
    >> "$LOGFILE" 2>&1 &
CHRONOTICK_PID=$!

echo "ChronoTick PID: $CHRONOTICK_PID" | tee -a "$LOGFILE"
echo "Waiting ${STRESS_START}s for baseline..." | tee -a "$LOGFILE"
sleep $STRESS_START

# Start CPU stress
echo "========================================" | tee -a "$LOGFILE"
echo "STARTING CPU THERMAL STRESS (${STRESS_DURATION}s)" | tee -a "$LOGFILE"
echo "Time: $(date)" | tee -a "$LOGFILE"
echo "========================================" | tee -a "$LOGFILE"

# Stress all CPUs to generate heat
stress-ng --cpu $(nproc) --timeout ${STRESS_DURATION}s --metrics >> "$LOGFILE" 2>&1 &
STRESS_PID=$!

# Wait for stress to complete
wait $STRESS_PID

echo "========================================" | tee -a "$LOGFILE"
echo "STRESS COMPLETE - COOL DOWN PERIOD" | tee -a "$LOGFILE"
echo "Time: $(date)" | tee -a "$LOGFILE"
echo "========================================" | tee -a "$LOGFILE"

# Wait for ChronoTick to complete
wait $CHRONOTICK_PID

# Stop system stats monitoring
kill $STATS_PID 2>/dev/null || true

echo "========================================" | tee -a "$LOGFILE"
echo "EXPERIMENT COMPLETE" | tee -a "$LOGFILE"
echo "End time: $(date)" | tee -a "$LOGFILE"
echo "ChronoTick data: $CSV_OUTPUT" | tee -a "$LOGFILE"
echo "System stats: /tmp/system_stats_${TIMESTAMP}.csv" | tee -a "$LOGFILE"
echo "Log file: $LOGFILE" | tee -a "$LOGFILE"
echo "========================================" | tee -a "$LOGFILE"

# Re-enable NTP for normal operation
sudo systemctl enable systemd-timesyncd
sudo systemctl start systemd-timesyncd
