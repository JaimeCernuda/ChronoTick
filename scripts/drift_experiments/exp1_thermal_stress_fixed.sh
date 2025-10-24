#!/bin/bash
# Drift Perturbation Experiment 1: CPU Thermal Stress (FIXED VERSION)
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
# Fixed: Remove --output flag, use thermal_zone3 for CPU temp

set -e

DURATION=30  # 30 minutes total (in minutes for validation script)
STRESS_START=300  # Start stress at 5 min (in seconds)
STRESS_END=1200   # End stress at 20 min (in seconds)
STRESS_DURATION=$((STRESS_END - STRESS_START))

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="/tmp/drift_exp1_thermal_${TIMESTAMP}.log"

echo "========================================" | tee -a "$LOGFILE"
echo "Drift Experiment 1: CPU Thermal Stress" | tee -a "$LOGFILE"
echo "Start time: $(date)" | tee -a "$LOGFILE"
echo "========================================" | tee -a "$LOGFILE"

# Find CPU package temperature sensor
CPU_TEMP_SENSOR=$(grep -l 'x86_pkg_temp' /sys/class/thermal/thermal_zone*/type 2>/dev/null | sed 's/type$/temp/' | head -1)
if [ -z "$CPU_TEMP_SENSOR" ]; then
    echo "WARNING: Could not find x86_pkg_temp sensor, using first thermal zone" | tee -a "$LOGFILE"
    CPU_TEMP_SENSOR="/sys/class/thermal/thermal_zone0/temp"
fi
echo "Using temperature sensor: $CPU_TEMP_SENSOR" | tee -a "$LOGFILE"

# Monitor system stats in background
(
    echo "timestamp,cpu_temp_c,cpu_freq_mhz,load_avg" > /tmp/system_stats_${TIMESTAMP}.csv
    while true; do
        TEMP=$(cat "$CPU_TEMP_SENSOR" 2>/dev/null || echo "0")
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
    >> "$LOGFILE" 2>&1 &
CHRONOTICK_PID=$!

echo "ChronoTick PID: $CHRONOTICK_PID" | tee -a "$LOGFILE"
echo "Waiting ${STRESS_START}s for baseline..." | tee -a "$LOGFILE"
sleep $STRESS_START

# Check current CPU temperature before stress
TEMP_BEFORE=$(cat "$CPU_TEMP_SENSOR" 2>/dev/null || echo "0")
TEMP_BEFORE_C=$((TEMP_BEFORE / 1000))
echo "CPU temperature before stress: ${TEMP_BEFORE_C}°C" | tee -a "$LOGFILE"

# Start CPU stress
echo "========================================" | tee -a "$LOGFILE"
echo "STARTING CPU THERMAL STRESS (${STRESS_DURATION}s)" | tee -a "$LOGFILE"
echo "Time: $(date)" | tee -a "$LOGFILE"
echo "========================================" | tee -a "$LOGFILE"

# Create Python CPU stress script
cat > /tmp/cpu_stress_${TIMESTAMP}.py <<'EOF'
import multiprocessing as mp
import time
import sys

def cpu_burn():
    """Burn CPU cycles to generate heat"""
    end_time = time.time() + float(sys.argv[1])
    while time.time() < end_time:
        _ = sum(i*i for i in range(10000))

if __name__ == '__main__':
    duration = float(sys.argv[1])
    num_cpus = mp.cpu_count()
    print(f"Stressing {num_cpus} CPUs for {duration}s...")

    processes = []
    for _ in range(num_cpus):
        p = mp.Process(target=cpu_burn)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("Stress complete")
EOF

# Run CPU stress
python3 /tmp/cpu_stress_${TIMESTAMP}.py $STRESS_DURATION >> "$LOGFILE" 2>&1 &
STRESS_PID=$!

# Wait for stress to complete
wait $STRESS_PID

# Check CPU temperature after stress
TEMP_AFTER=$(cat "$CPU_TEMP_SENSOR" 2>/dev/null || echo "0")
TEMP_AFTER_C=$((TEMP_AFTER / 1000))
echo "CPU temperature after stress: ${TEMP_AFTER_C}°C" | tee -a "$LOGFILE"
echo "Temperature change: $((TEMP_AFTER_C - TEMP_BEFORE_C))°C" | tee -a "$LOGFILE"

echo "========================================" | tee -a "$LOGFILE"
echo "STRESS COMPLETE - COOL DOWN PERIOD" | tee -a "$LOGFILE"
echo "Time: $(date)" | tee -a "$LOGFILE"
echo "========================================" | tee -a "$LOGFILE"

# Wait for ChronoTick to complete
wait $CHRONOTICK_PID

# Stop system stats monitoring
kill $STATS_PID 2>/dev/null || true

# Find the ChronoTick CSV file (it creates its own timestamp)
CHRONOTICK_CSV=$(ls -t /tmp/chronotick_client_validation_v2_*.csv 2>/dev/null | head -1)

echo "========================================" | tee -a "$LOGFILE"
echo "EXPERIMENT COMPLETE" | tee -a "$LOGFILE"
echo "End time: $(date)" | tee -a "$LOGFILE"
if [ -n "$CHRONOTICK_CSV" ]; then
    echo "ChronoTick data: $CHRONOTICK_CSV" | tee -a "$LOGFILE"
    echo "  Size: $(ls -lh $CHRONOTICK_CSV | awk '{print $5}')" | tee -a "$LOGFILE"
    echo "  Rows: $(wc -l < $CHRONOTICK_CSV)" | tee -a "$LOGFILE"
else
    echo "WARNING: ChronoTick CSV not found!" | tee -a "$LOGFILE"
fi
echo "System stats: /tmp/system_stats_${TIMESTAMP}.csv" | tee -a "$LOGFILE"
echo "  Size: $(ls -lh /tmp/system_stats_${TIMESTAMP}.csv 2>/dev/null | awk '{print $5}')" | tee -a "$LOGFILE"
echo "Log file: $LOGFILE" | tee -a "$LOGFILE"
echo "========================================" | tee -a "$LOGFILE"
