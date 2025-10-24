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
