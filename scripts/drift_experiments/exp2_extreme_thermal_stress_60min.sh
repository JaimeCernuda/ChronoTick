#!/bin/bash
# Drift Perturbation Experiment 2: EXTREME Multi-Dimensional Thermal Stress (60 minutes)
#
# Goal: Induce SIGNIFICANT temperature changes to alter crystal oscillator frequency
# Duration: 60 minutes
#
# Timeline:
#   0-5 min:   Baseline (normal operation)
#   5-50 min:  EXTREME multi-dimensional thermal stress (CPU + Memory + Disk I/O)
#   50-60 min: Cool down period
#
# Enhancements over Exp1:
#   - 3x longer stress duration (45 min vs 15 min)
#   - Multi-dimensional stress (not just CPU)
#   - More aggressive CPU workload
#   - Memory pressure to heat RAM/controllers
#   - Disk I/O to heat storage controllers/motherboard
#
# Expected: Larger drift rate changes as sustained extreme heat affects oscillator

set -e

DURATION=60  # 60 minutes total (in minutes for validation script)
STRESS_START=300  # Start stress at 5 min (in seconds)
STRESS_END=3000   # End stress at 50 min (in seconds)
STRESS_DURATION=$((STRESS_END - STRESS_START))

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="/tmp/drift_exp2_extreme_${TIMESTAMP}.log"

echo "========================================" | tee -a "$LOGFILE"
echo "Drift Experiment 2: EXTREME Thermal Stress (60min)" | tee -a "$LOGFILE"
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
    echo "timestamp,cpu_temp_c,cpu_freq_mhz,load_avg,mem_used_pct" > /tmp/system_stats_${TIMESTAMP}.csv
    while true; do
        TEMP=$(cat "$CPU_TEMP_SENSOR" 2>/dev/null || echo "0")
        TEMP_C=$((TEMP / 1000))
        FREQ=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null || echo "0")
        FREQ_MHZ=$((FREQ / 1000))
        LOAD=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | tr -d ',')
        MEM_USED=$(free | awk '/Mem:/ {printf "%.1f", $3/$2 * 100}')
        echo "$(date +%s),$TEMP_C,$FREQ_MHZ,$LOAD,$MEM_USED"
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

# Start EXTREME multi-dimensional stress
echo "========================================" | tee -a "$LOGFILE"
echo "STARTING EXTREME MULTI-DIMENSIONAL THERMAL STRESS (${STRESS_DURATION}s / 45 min)" | tee -a "$LOGFILE"
echo "Time: $(date)" | tee -a "$LOGFILE"
echo "Components: CPU (intensive math) + Memory (allocation churn) + Disk I/O (write stress)" | tee -a "$LOGFILE"
echo "========================================" | tee -a "$LOGFILE"

# Create Python EXTREME stress script with CPU + Memory + Disk I/O
cat > /tmp/extreme_stress_${TIMESTAMP}.py <<'EOF'
import multiprocessing as mp
import time
import sys
import os
import random
import string

def cpu_burn_intensive(duration):
    """INTENSIVE CPU burn with heavy floating point operations"""
    end_time = time.time() + duration
    while time.time() < end_time:
        # Heavy floating point math + trig functions to maximize heat
        _ = sum(i**2.7 * 3.14159 for i in range(1000))
        # Add some branching to prevent optimization
        if random.random() > 0.5:
            _ = [x**3 for x in range(100)]

def memory_churn(duration):
    """Memory allocation churn to heat RAM and memory controllers"""
    end_time = time.time() + duration
    buffers = []
    while time.time() < end_time:
        # Allocate 100MB chunks and fill with random data
        try:
            buf = bytearray(100 * 1024 * 1024)  # 100MB
            for i in range(0, len(buf), 4096):
                buf[i] = random.randint(0, 255)
            buffers.append(buf)
            # Keep last 10 buffers (1GB total)
            if len(buffers) > 10:
                buffers.pop(0)
        except MemoryError:
            # If out of memory, clear and restart
            buffers.clear()
        time.sleep(0.1)

def disk_io_stress(duration, temp_dir):
    """Disk I/O stress to heat storage controllers and motherboard"""
    end_time = time.time() + duration
    file_counter = 0
    while time.time() < end_time:
        # Write 100MB files repeatedly
        filepath = os.path.join(temp_dir, f"stress_{os.getpid()}_{file_counter}.tmp")
        try:
            with open(filepath, 'wb') as f:
                # Write 100MB of random data
                for _ in range(100):
                    f.write(os.urandom(1024 * 1024))
            file_counter += 1
            # Delete after writing to create continuous I/O
            os.remove(filepath)
        except (IOError, OSError):
            pass
        time.sleep(0.5)

if __name__ == '__main__':
    duration = float(sys.argv[1])
    num_cpus = mp.cpu_count()

    # Create temp directory for I/O stress
    temp_dir = f"/tmp/stress_exp2_{int(time.time())}"
    os.makedirs(temp_dir, exist_ok=True)

    print(f"EXTREME STRESS: {num_cpus} CPUs + {num_cpus//2} memory churners + {num_cpus//4} disk I/O for {duration}s...")

    processes = []

    # CPU stress on ALL cores
    for _ in range(num_cpus):
        p = mp.Process(target=cpu_burn_intensive, args=(duration,))
        p.start()
        processes.append(p)

    # Memory churn on half the cores
    for _ in range(max(1, num_cpus // 2)):
        p = mp.Process(target=memory_churn, args=(duration,))
        p.start()
        processes.append(p)

    # Disk I/O on quarter of cores
    for _ in range(max(1, num_cpus // 4)):
        p = mp.Process(target=disk_io_stress, args=(duration, temp_dir))
        p.start()
        processes.append(p)

    print(f"Total processes: {len(processes)}")

    # Wait for all to complete
    for p in processes:
        p.join()

    # Cleanup temp files
    try:
        for f in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, f))
        os.rmdir(temp_dir)
    except:
        pass

    print("EXTREME stress complete")
EOF

# Run EXTREME stress
python3 /tmp/extreme_stress_${TIMESTAMP}.py $STRESS_DURATION >> "$LOGFILE" 2>&1 &
STRESS_PID=$!

# Monitor temperature every 60 seconds during stress
(
    sleep 60
    while kill -0 $STRESS_PID 2>/dev/null; do
        TEMP_NOW=$(cat "$CPU_TEMP_SENSOR" 2>/dev/null || echo "0")
        TEMP_NOW_C=$((TEMP_NOW / 1000))
        echo "[$(date +%H:%M:%S)] Stress in progress... CPU: ${TEMP_NOW_C}°C" | tee -a "$LOGFILE"
        sleep 60
    done
) &
TEMP_MONITOR_PID=$!

# Wait for stress to complete
wait $STRESS_PID

# Stop temperature monitoring
kill $TEMP_MONITOR_PID 2>/dev/null || true

# Check CPU temperature after stress
TEMP_AFTER=$(cat "$CPU_TEMP_SENSOR" 2>/dev/null || echo "0")
TEMP_AFTER_C=$((TEMP_AFTER / 1000))
echo "========================================" | tee -a "$LOGFILE"
echo "CPU temperature before stress: ${TEMP_BEFORE_C}°C" | tee -a "$LOGFILE"
echo "CPU temperature after stress:  ${TEMP_AFTER_C}°C" | tee -a "$LOGFILE"
echo "Temperature change: $((TEMP_AFTER_C - TEMP_BEFORE_C))°C" | tee -a "$LOGFILE"
echo "========================================" | tee -a "$LOGFILE"

echo "========================================" | tee -a "$LOGFILE"
echo "STRESS COMPLETE - COOL DOWN PERIOD (10 MINUTES)" | tee -a "$LOGFILE"
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
