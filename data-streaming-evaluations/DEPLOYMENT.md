# Data Streaming Evaluation: ARES Deployment Guide

**Experiment**: Distributed message passing with timestamp consensus analysis
**Infrastructure**: ARES cluster (ares-comp-11, ares-comp-12, ares-comp-18)
**Duration**: 10 minutes (100 broadcast events)

---

## 🎯 Experimental Setup

### Node Roles

| Node | Role | Task | Description |
|------|------|------|-------------|
| **ares-comp-18** | Coordinator | Task A | Broadcasts events to workers |
| **ares-comp-11** | Worker B | Task B | Receives events, timestamps with NTP + ChronoTick |
| **ares-comp-12** | Worker C | Task C | Receives events, timestamps with NTP + ChronoTick |

### Network Configuration

**NTP Server** (via master node proxy):
```bash
--ntp-server 172.20.1.1:8123
```
This maps to ARES master node NTP proxy which redirects to actual NTP servers.

**ChronoTick Server** (similar mapping):
```bash
--chronotick-server 172.20.1.1:8123
```
Uses the same proxy mechanism for ChronoTick inference server.

**Broadcast Network**:
- Coordinator sends to: `ares-comp-11:9000` and `ares-comp-12:9000`
- Protocol: UDP (low latency, realistic for distributed systems)
- Payload: Event ID + coordinator timestamp

---

## 📋 Prerequisites

### On Your Local Machine

1. **SSH Access**:
```bash
ssh ares  # Connects to master node (has internet)
```

2. **Code Deployment** (from local machine):
```bash
# Push latest code to repo
cd /home/jcernuda/tick_project/ChronoTick
git add data-streaming-evaluations/
git commit -m "Add data streaming evaluation framework"
git push
```

### On ARES Master Node

1. **Pull Latest Code**:
```bash
ssh ares
cd /path/to/ChronoTick  # Adjust to your NFS mount point
git pull
```

2. **Install Dependencies** (once):
```bash
cd data-streaming-evaluations
uv sync
```

3. **Verify NFS**:
```bash
# Ensure all compute nodes see the same filesystem
ls -la /path/to/ChronoTick/data-streaming-evaluations
```

---

## 🚀 One-Command Deployment

### Quick Start

From ARES master node:

```bash
cd /path/to/ChronoTick/data-streaming-evaluations
./deploy.sh
```

This will:
1. ✅ Verify all nodes are accessible
2. ✅ Start workers on ares-comp-11 and ares-comp-12
3. ✅ Start coordinator on ares-comp-18
4. ✅ Monitor progress (100 events, ~10 minutes)
5. ✅ Collect results to `results/experiment-<timestamp>/`
6. ✅ Run analysis and generate figures

### What Gets Deployed

**On ares-comp-18 (Coordinator)**:
```bash
nohup uv run coordinator \
  --config configs/coordinator_config.yaml \
  --workers ares-comp-11:9000,ares-comp-12:9000 \
  --output results/experiment-001/coordinator.csv \
  > logs/coordinator.log 2>&1 &
```

**On ares-comp-11 (Worker B)**:
```bash
nohup uv run worker \
  --config configs/worker_config.yaml \
  --node-id comp11 \
  --listen-port 9000 \
  --ntp-server 172.20.1.1:8123 \
  --chronotick-server 172.20.1.1:8124 \
  --output results/experiment-001/worker_comp11.csv \
  > logs/worker_comp11.log 2>&1 &
```

**On ares-comp-12 (Worker C)**:
```bash
nohup uv run worker \
  --config configs/worker_config.yaml \
  --node-id comp12 \
  --listen-port 9000 \
  --ntp-server 172.20.1.1:8123 \
  --chronotick-server 172.20.1.1:8124 \
  --output results/experiment-001/worker_comp12.csv \
  > logs/worker_comp12.log 2>&1 &
```

---

## 🧠 ChronoTick Worker Deployment (Embedded AI Inference)

ChronoTick workers run embedded AI inference locally on each worker node (no external HTTP server required). This deployment is for comparing AI-based timing predictions against NTP reference measurements.

### Key Differences from System Clock Workers

| Feature | System Clock Worker | ChronoTick Worker |
|---------|---------------------|-------------------|
| **Module** | `src.worker` | `src.worker_chronotick` |
| **Python Path** | Standard venv | Requires ChronoTick source in PYTHONPATH |
| **Model Loading** | None | TimesFM 2.5 models (30-60s load time) |
| **Warmup Time** | 10s | 90s (model loading + NTP collection) |
| **CSV Schema** | 8 fields (NTP only) | 17 fields (NTP + ChronoTick) |
| **Environment** | Minimal | Requires HuggingFace cache + offline mode |
| **Dependencies** | Basic | ChronoTick inference engine |

### ChronoTick Worker Command

**Base Directory**: `/mnt/common/jcernudagarcia/ChronoTick/data-streaming-evaluations`

**Worker B (ares-comp-11)**:
```bash
ssh ares-comp-11 "cd $BASE_DIR && \
  HF_HOME=/mnt/common/jcernudagarcia/.cache/huggingface \
  TRANSFORMERS_CACHE=/mnt/common/jcernudagarcia/.cache/huggingface \
  TRANSFORMERS_OFFLINE=1 \
  HF_HUB_OFFLINE=1 \
  PYTHONPATH=/mnt/common/jcernudagarcia/ChronoTick/server/src:$BASE_DIR \
  /mnt/common/jcernudagarcia/ChronoTick/.venv/bin/python -m src.worker_chronotick \
    --node-id comp11 \
    --listen-port 9000 \
    --ntp-server 172.20.1.1:8123 \
    --chronotick-config configs/chronotick_config.yaml \
    --output results/test/worker_comp11.csv \
    --log-level INFO \
    > logs/test/worker_comp11.log 2>&1" &
```

**Worker C (ares-comp-12)**:
```bash
ssh ares-comp-12 "cd $BASE_DIR && \
  HF_HOME=/mnt/common/jcernudagarcia/.cache/huggingface \
  TRANSFORMERS_CACHE=/mnt/common/jcernudagarcia/.cache/huggingface \
  TRANSFORMERS_OFFLINE=1 \
  HF_HUB_OFFLINE=1 \
  PYTHONPATH=/mnt/common/jcernudagarcia/ChronoTick/server/src:$BASE_DIR \
  /mnt/common/jcernudagarcia/ChronoTick/.venv/bin/python -m src.worker_chronotick \
    --node-id comp12 \
    --listen-port 9000 \
    --ntp-server 172.20.1.1:8123 \
    --chronotick-config configs/chronotick_config.yaml \
    --output results/test/worker_comp12.csv \
    --log-level INFO \
    > logs/test/worker_comp12.log 2>&1" &
```

### Environment Variables Explained

**CRITICAL**: Compute nodes (ares-comp-11, ares-comp-12) have NO internet access. Models must be pre-cached and transformers library must run in offline mode.

| Variable | Purpose | Value |
|----------|---------|-------|
| `HF_HOME` | HuggingFace cache root | `/mnt/common/jcernudagarcia/.cache/huggingface` |
| `TRANSFORMERS_CACHE` | Transformers model cache | Same as HF_HOME |
| `TRANSFORMERS_OFFLINE=1` | Disable transformers internet access | Required |
| `HF_HUB_OFFLINE=1` | Disable HuggingFace Hub queries | Required |
| `PYTHONPATH` | ChronoTick source + worker source | `/mnt/common/jcernudagarcia/ChronoTick/server/src:$BASE_DIR` |

**Without offline mode**: Workers will attempt to download from huggingface.co and fail (DNS resolution fails on compute nodes).

**With offline mode**: Only 2-3 HuggingFace warnings (harmless cache validation), test succeeds.

### ChronoTick Configuration File

**File**: `configs/chronotick_config.yaml` (167 lines)

**Key Settings**:
```yaml
clock_measurement:
  ntp:
    servers:
      - 172.20.1.1:8123  # Multi-server NTP proxy
      - 172.20.1.1:8124
      - 172.20.1.1:8125
      # ... (10 total servers via proxy)
    parallel_queries: true
    max_workers: 10
    enable_fallback: true
  timing:
    warm_up:
      duration_seconds: 60
      measurement_interval: 1  # Aggressive warmup

short_term:
  model_name: timesfm
  device: cpu
  prediction_horizon: 30
  model_params:
    model_repo: google/timesfm-2.5-200m-pytorch
    context_length: 512
    prediction_length: 30

long_term:
  model_name: timesfm
  device: cpu
  prediction_horizon: 60
  model_params:
    model_repo: google/timesfm-2.5-200m-pytorch
    context_length: 512
    prediction_length: 60

fusion:
  enabled: true
  method: none
  fallback_weights:
    short_term: 0.8
    long_term: 0.2
```

### ChronoTick CSV Schema (17 Fields)

**System Clock Worker** (8 fields):
```csv
event_id,node_id,sequence_number,receive_time_ns,coordinator_send_time_ns,
ntp_offset_ms,ntp_uncertainty_ms,ntp_timestamp_ns
```

**ChronoTick Worker** (17 fields):
```csv
event_id,node_id,sequence_number,receive_time_ns,coordinator_send_time_ns,
# NTP reference
ntp_offset_ms,ntp_uncertainty_ms,ntp_timestamp_ns,
# ChronoTick AI prediction
ct_offset_ms,ct_drift_rate,ct_uncertainty_ms,ct_confidence,ct_source,
ct_prediction_time,ct_timestamp_ns,ct_lower_bound_ns,ct_upper_bound_ns
```

**Sample ChronoTick Data**:
```csv
91,comp11,90,1761370722914336032,1761370722913687470,-23.59771728515625,5.0,1761370722890738315,2.1131368517096907,1.0461766489089259e-06,0.007650473972363639,0.99998176,fusion,1761370722.918138,1761370722916449168,1761370722916426217,1761370722916472119
```

**Field Explanations**:
- `ct_offset_ms`: ChronoTick predicted clock offset (milliseconds)
- `ct_drift_rate`: Clock drift rate (microseconds/second)
- `ct_uncertainty_ms`: Prediction uncertainty (milliseconds)
- `ct_confidence`: Confidence interval metric `(Q3-Q1)/2.56` (NOT a probability!)
- `ct_source`: Prediction source (`cpu`, `gpu`, or `fusion`)
- `ct_prediction_time`: Unix timestamp when prediction was made
- `ct_timestamp_ns`: Corrected timestamp (receive_time_ns + ct_offset_ms)
- `ct_lower_bound_ns`: Lower uncertainty bound
- `ct_upper_bound_ns`: Upper uncertainty bound

### Deployment Scripts

**2-Minute Quick Test** (`deploy_chronotick_2min.sh`):
- 100 events over 120 seconds
- 90s warmup (model loading + NTP)
- Purpose: Validation before full test
- Expected duration: ~3.5 minutes total

**30-Minute Production Test** (`deploy_chronotick_30min.sh`):
- 3000 events over 1800 seconds
- 90s warmup
- Purpose: Full dataset for ChronoTick vs NTP comparison
- Expected duration: ~32 minutes total

**Usage**:
```bash
ssh ares
cd /mnt/common/jcernudagarcia/ChronoTick/data-streaming-evaluations

# Quick validation test
./deploy_chronotick_2min.sh

# Full production test (after validation succeeds)
./deploy_chronotick_30min.sh
```

### Warmup Phase (90 Seconds)

**Why 90 seconds?**
1. **Model Loading** (30-60s): TimesFM 2.5 models load from disk
2. **NTP Collection** (60s): Collect initial measurements for ChronoTick baseline
3. **CSV File Creation**: File isn't created until AFTER model loading completes

**Deployment Script Validation**:
```bash
# CORRECT: Wait full 90s, then validate ONCE
for i in {15..90..15}; do
    sleep 15
    log_info "Warmup: ${i}s / 90s (allowing model loading time)"
done

# Validate ONCE after full warmup completes
if [ ! -f "$RESULTS_DIR/worker_comp11.csv" ]; then
    log_error "Worker B failed to create CSV after 90s warmup!"
    exit 1
fi
```

**INCORRECT** (old version - causes premature exit):
```bash
# DON'T DO THIS: Checking at 15s intervals causes premature failure
for i in {15..90..15}; do
    sleep 15
    if [ ! -f "$RESULTS_DIR/worker_comp11.csv" ]; then
        log_error "Worker B failed to start!"  # FAILS AT 15s!
        exit 1
    fi
done
```

### Pre-cached Models

**Location**: `/mnt/common/jcernudagarcia/.cache/huggingface/hub/models--google--timesfm-2.5-200m-pytorch`

**Verification**:
```bash
ssh ares "ls -lh /mnt/common/jcernudagarcia/.cache/huggingface/hub/models--google--timesfm-2.5-200m-pytorch/snapshots/"
```

**Expected**: Directory exists with model checkpoint files (~800MB total)

**If missing**: Models need to be downloaded on master node (has internet):
```bash
ssh ares
cd /mnt/common/jcernudagarcia/ChronoTick
source .venv/bin/activate
python -c "from transformers import AutoModelForSeq2SeqLM; AutoModelForSeq2SeqLM.from_pretrained('google/timesfm-2.5-200m-pytorch')"
```

### Troubleshooting ChronoTick Workers

#### Worker Fails to Import ChronoTick Modules

**Error**:
```
ModuleNotFoundError: No module named 'chronotick'
```

**Solution**: Check PYTHONPATH includes ChronoTick source:
```bash
PYTHONPATH=/mnt/common/jcernudagarcia/ChronoTick/server/src:$BASE_DIR
```

#### HuggingFace Download Attempts on Compute Nodes

**Error** (in worker logs):
```
Could not reach huggingface.co
Trying to load from cache...
Traceback: Connection failed
```

**Root Cause**: Missing offline mode environment variables

**Solution**: Add ALL offline mode variables:
```bash
HF_HOME=/mnt/common/jcernudagarcia/.cache/huggingface
TRANSFORMERS_CACHE=/mnt/common/jcernudagarcia/.cache/huggingface
TRANSFORMERS_OFFLINE=1
HF_HUB_OFFLINE=1
```

**Expected Result**: Only 2-3 warnings (harmless), no download attempts

#### CSV File Not Created After 90s Warmup

**Error**:
```
Worker B failed to create CSV after 90s warmup!
```

**Possible Causes**:
1. Model loading taking longer than 90s (rare)
2. Worker process crashed during initialization
3. Permissions issue writing CSV file

**Debug**:
```bash
# Check worker log for actual error
cat logs/chronotick_2min_TIMESTAMP/worker_comp11.log

# Check if worker process is running
ssh ares-comp-11 "ps aux | grep worker_chronotick"

# Check NFS permissions
ssh ares-comp-11 "touch /mnt/common/jcernudagarcia/ChronoTick/data-streaming-evaluations/results/test.txt && rm /mnt/common/jcernudagarcia/ChronoTick/data-streaming-evaluations/results/test.txt"
```

#### Fake Confidence Bug (confidence = 1.0 exactly)

**Symptoms**: All confidence values are exactly `1.0`

**Root Cause**: Bug in older ChronoTick versions where confidence calculation was broken

**Check**:
```bash
# View actual confidence values from test
tail -5 results/chronotick_2min_TIMESTAMP/worker_comp11.csv | cut -d, -f12

# Should see realistic values like: 0.9999746, 0.99997264, 0.99996895
# NOT exactly: 1.0, 1.0, 1.0
```

**Fix**: Pull latest ChronoTick code (fix committed in ChronoTick repo)
```bash
ssh ares
cd /mnt/common/jcernudagarcia/ChronoTick
git pull
```

#### Orphaned Worker Processes

**Symptoms**: Workers continue running after deployment script exits

**Cause**: Deployment script exited prematurely during warmup, workers backgrounded and never killed

**Solution**: Kill all ChronoTick workers before new test:
```bash
ssh ares-comp-11 "pkill -9 -f 'worker_chronotick' || true"
ssh ares-comp-12 "pkill -9 -f 'worker_chronotick' || true"
```

**Prevention**: Fixed in deployment scripts (commit 53e74f9) - wait full 90s before validation

---

## 🔧 Manual Deployment (Step-by-Step)

If you prefer manual control or need to debug:

### Step 1: Start Workers (Nodes 11 & 12)

**Terminal 1** (ares-comp-11):
```bash
ssh ares "ssh ares-comp-11 'cd /path/to/ChronoTick/data-streaming-evaluations && \
  uv run worker \
    --config configs/worker_config.yaml \
    --node-id comp11 \
    --listen-port 9000 \
    --ntp-server 172.20.1.1:8123 \
    --chronotick-server 172.20.1.1:8124 \
    --output results/manual-test/worker_comp11.csv'"
```

**Terminal 2** (ares-comp-12):
```bash
ssh ares "ssh ares-comp-12 'cd /path/to/ChronoTick/data-streaming-evaluations && \
  uv run worker \
    --config configs/worker_config.yaml \
    --node-id comp12 \
    --listen-port 9000 \
    --ntp-server 172.20.1.1:8123 \
    --chronotick-server 172.20.1.1:8124 \
    --output results/manual-test/worker_comp12.csv'"
```

Wait for workers to print: `"Worker ready, listening on port 9000"`

### Step 2: Start Coordinator (Node 18)

**Terminal 3** (ares-comp-18):
```bash
ssh ares "ssh ares-comp-18 'cd /path/to/ChronoTick/data-streaming-evaluations && \
  uv run coordinator \
    --config configs/coordinator_config.yaml \
    --workers ares-comp-11:9000,ares-comp-12:9000 \
    --num-events 100 \
    --output results/manual-test/coordinator.csv'"
```

### Step 3: Monitor Progress

Watch coordinator output:
```bash
Broadcasting event 1/100...
Broadcasting event 10/100...
...
Broadcasting event 100/100...
Experiment complete! Duration: 9m 42s
```

Watch worker logs:
```bash
# On comp-11
tail -f results/manual-test/worker_comp11.csv

# On comp-12
tail -f results/manual-test/worker_comp12.csv
```

### Step 4: Collect Results

From master node:
```bash
cd data-streaming-evaluations
./scripts/collect_results.sh manual-test
```

This copies all CSVs to local `results/manual-test/` directory.

### Step 5: Run Analysis

```bash
uv run analyze --experiment manual-test
```

Generates figures in `results/manual-test/figures/`:
- `causality_violations.png`
- `ordering_consensus.png`
- `window_assignment.png`
- `coordination_cost.png`
- `commit_wait.png`
- `summary_report.pdf`

---

## 📊 Configuration Files

### Coordinator Config (`configs/coordinator_config.yaml`)

```yaml
experiment:
  num_events: 100
  duration_minutes: 10  # ~6s between events

broadcast_pattern:
  # Mix of delays to test different scenarios
  fast_stream: 0.010    # 10ms - rapid events
  medium_stream: 0.050  # 50ms - potential duplicates
  slow_stream: 0.100    # 100ms - window boundaries

  # Pattern: 10 slow, 5 medium, then repeat fast
  pattern: [slow, fast, fast, fast, fast, medium, fast, fast, fast, fast]

network:
  protocol: udp
  timeout: 5.0  # seconds

logging:
  level: INFO
  console: true
  file: logs/coordinator.log
```

### Worker Config (`configs/worker_config.yaml`)

```yaml
network:
  listen_port: 9000
  buffer_size: 4096

timing:
  ntp_query_interval: 10  # Query NTP every 10s (for commit-wait analysis)
  chronotick_query_interval: 1  # ChronoTick every 1s (fresh uncertainties)

  # For commit-wait: record uncertainty at T+30s, T+60s
  commit_wait_delays: [30, 60]

chronotick:
  use_dual_model: true  # Use both short-term and long-term models
  uncertainty_multiplier: 3  # ±3σ bounds

logging:
  level: INFO
  console: true
  file: logs/worker_{node_id}.log
```

---

## 🔍 Verification

### Check Workers Are Listening

From master node:
```bash
# Test UDP connectivity
echo "TEST" | nc -u ares-comp-11 9000
echo "TEST" | nc -u ares-comp-12 9000
```

Workers should log: `"Received test packet"`

### Check NTP Proxy

```bash
ssh ares-comp-11 "ntpdate -q 172.20.1.1:8123"
# Should return offset and delay
```

### Check ChronoTick Connectivity

```bash
ssh ares-comp-11 "curl http://172.20.1.1:8124/health"
# Should return: {"status": "healthy"}
```

---

## 🐛 Troubleshooting

### Workers Not Receiving Events

**Check firewall** (on compute nodes):
```bash
ssh ares-comp-11 "sudo ufw status"
# If active, allow port 9000:
ssh ares-comp-11 "sudo ufw allow 9000/udp"
```

**Check network**:
```bash
# Ping test
ssh ares-comp-18 "ping -c 3 ares-comp-11"
```

**Check if port is bound**:
```bash
ssh ares-comp-11 "netstat -ulnp | grep 9000"
# Should show: udp 0.0.0.0:9000
```

### NTP Server Unreachable

**Check proxy status** (on master):
```bash
systemctl status ntp-proxy
```

**Check NFS mount**:
```bash
df -h | grep nfs
# Should show NFS mount point
```

**Fallback to direct NTP**:
Edit worker command, replace:
```bash
--ntp-server 172.20.1.1:8123
```
with:
```bash
--ntp-server pool.ntp.org,time.google.com
```

### ChronoTick Server Not Responding

**Check if server is running** (on master):
```bash
ps aux | grep chronotick
```

**Start server manually**:
```bash
cd /path/to/ChronoTick/tsfm
uv run python chronotick_mcp.py --port 8124 &
```

### Results Not Collected

**Check NFS permissions**:
```bash
ls -la results/
# Should be writable by your user
```

**Manual collection**:
```bash
# From master node
scp ares-comp-11:/path/to/results/experiment-001/worker_comp11.csv results/experiment-001/
scp ares-comp-12:/path/to/results/experiment-001/worker_comp12.csv results/experiment-001/
scp ares-comp-18:/path/to/results/experiment-001/coordinator.csv results/experiment-001/
```

---

## 📈 Expected Results

### Timeline

```
T=0s:     Workers start, begin NTP/ChronoTick monitoring
T=10s:    Workers report "ready"
T=15s:    Coordinator starts broadcasting
T=15s:    Event 1 broadcast
T=21s:    Event 10 broadcast (slow pattern)
...
T=9m42s:  Event 100 broadcast
T=10m:    Experiment complete
T=11m:    Workers finish commit-wait monitoring (T+60s for last event)
```

### Data Files

**coordinator.csv**:
```csv
event_id,send_time_ns,worker_b_sent,worker_c_sent
1,1234567890123456789,true,true
2,1234567890223456789,true,true
...
```

**worker_comp11.csv**:
```csv
event_id,receive_time_ns,ntp_offset_ms,ntp_uncertainty_ms,ct_offset_ms,ct_uncertainty_ms,ct_uncertainty_30s,ct_uncertainty_60s
1,1234567890125456789,2.5,5.0,2.3,8.5,6.2,4.8
...
```

**worker_comp12.csv**: Same format as comp11

### Analysis Outputs

After running `uv run analyze --experiment experiment-001`:
```
results/experiment-001/
├── figures/
│   ├── causality_violations.png
│   ├── ordering_consensus.png
│   ├── window_assignment.png
│   ├── coordination_cost.png
│   ├── commit_wait.png
│   └── summary_dashboard.png
├── statistics/
│   ├── causality_stats.json
│   ├── ordering_stats.json
│   └── overall_summary.json
└── report/
    └── experiment_report.pdf
```

---

## 🎓 Best Practices

### For Reproducible Results

1. **Record everything**:
   - Node hostnames
   - NTP server used
   - ChronoTick model version
   - Exact timestamps (start/end)

2. **Use version control**:
   ```bash
   git log -1 --oneline > results/experiment-001/git_commit.txt
   ```

3. **Save configurations**:
   ```bash
   cp configs/*.yaml results/experiment-001/configs/
   ```

4. **Monitor resource usage**:
   ```bash
   # During experiment
   ssh ares-comp-11 "top -b -n 1" > results/experiment-001/comp11_resources.txt
   ```

### For Debugging

1. **Verbose logging**:
   Edit configs, set `logging.level: DEBUG`

2. **Capture packets**:
   ```bash
   ssh ares-comp-11 "tcpdump -i any port 9000 -w capture.pcap"
   ```

3. **Test broadcast manually**:
   ```bash
   python3 -c "
   import socket
   import json
   sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
   event = {'id': 999, 'timestamp': 1234567890.123}
   sock.sendto(json.dumps(event).encode(), ('ares-comp-11', 9000))
   "
   ```

---

## 📞 Quick Reference

### SSH Command Templates

**Nested SSH** (from local machine):
```bash
ssh ares "ssh ares-comp-11 'COMMAND'"
```

**Check process** (from local machine):
```bash
ssh ares "ssh ares-comp-11 'ps aux | grep worker'"
```

**Kill process** (from local machine):
```bash
ssh ares "ssh ares-comp-11 'pkill -f worker'"
```

### File Paths

- **Code**: `/path/to/ChronoTick/data-streaming-evaluations/` (on NFS)
- **Results**: `results/experiment-<name>/`
- **Logs**: `logs/`
- **Configs**: `configs/`

### Ports

- **9000**: Worker UDP listener
- **8123**: NTP proxy (on master 172.20.1.1)
- **8124**: ChronoTick server proxy (on master 172.20.1.1)

---

## 🚀 Next Steps

After successful deployment:

1. **Review logs**: Check for any errors or warnings
2. **Verify data**: Ensure all 100 events were received by both workers
3. **Run analysis**: Generate figures and statistics
4. **Read ANALYSIS.md**: Understand the results
5. **Read NARRATIVE.md**: Craft the paper story

For questions or issues, see troubleshooting section or contact the ChronoTick team.
