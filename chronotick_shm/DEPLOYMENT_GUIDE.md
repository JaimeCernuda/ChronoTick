# ChronoTick Deployment Guide

**Architecture Overview and Deployment Steps**

---

## Architecture Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        ChronoTick System                        │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────┐          ┌──────────────────┐
│   Application    │          │   Application    │
│   (Your Code)    │          │   (AI Agent)     │
└────────┬─────────┘          └────────┬─────────┘
         │                             │
         ├─────────────────────────────┤
         │                             │
         v                             v
┌──────────────────┐          ┌──────────────────┐
│  Client Library  │          │    MCP Tools     │
│ ChronoTickClient │          │ (SDK or Stdio)   │
└────────┬─────────┘          └────────┬─────────┘
         │                             │
         └──────────────┬──────────────┘
                        │
                        v
            ┌────────────────────┐
            │  Shared Memory     │ <--- 128 bytes, ~300ns reads
            │  (/dev/shm)        │
            └────────┬───────────┘
                     ^
                     │ writes (100 Hz)
                     │
            ┌────────┴───────────┐
            │  ChronoTick Daemon │
            │  (Background)      │
            └────────┬───────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
         v                       v
    ┌─────────┐           ┌──────────┐
    │   NTP   │           │ ML Models│
    │ Clients │           │(TimesFM) │
    └─────────┘           └──────────┘
```

---

## Deployment Steps

### Step 1: Install ChronoTick Packages

**Option A: Install from source (current)**
```bash
# Install main tsfm package (contains implementation)
cd /path/to/ChronoTick/tsfm
uv sync --extra core-models

# Install client library
cd /path/to/ChronoTick/chronotick_shm
uv sync
```

**Option B: Install from pip (future)**
```bash
# When published to PyPI:
pip install chronotick        # Client library
pip install chronotick-server # Server/daemon (optional, for running daemon)
```

---

### Step 2: Configure the Daemon

Create a configuration file or use the default:

**Example config:** `config.yaml`
```yaml
# Clock settings
clock:
  frequency_code: 9        # Seconds precision
  frequency_type: second

# NTP configuration
clock_measurement:
  ntp:
    servers:
      - pool.ntp.org
      - time.google.com
      - time.nist.gov
    timeout_seconds: 2.0
    max_acceptable_uncertainty: 0.1
  timing:
    warm_up:
      duration_seconds: 60
      measurement_interval: 5
    normal_operation:
      measurement_interval: 180

# Model configuration (DUAL-MODEL recommended)
short_term:
  enabled: true
  model_name: timesfm
  prediction_horizon: 30
  inference_interval: 1.0
  device: cpu

long_term:
  enabled: true
  model_name: timesfm
  prediction_horizon: 60
  inference_interval: 30.0
  device: cpu

# Fusion configuration
fusion:
  enabled: true
  method: inverse_variance_weighting

# Performance settings
performance:
  batch_size: 1
  cache_size: 5
  max_memory_mb: 2048
```

---

### Step 3: Start the ChronoTick Daemon

The daemon is the core server that:
1. Collects NTP measurements
2. Runs ML models for prediction
3. Writes corrections to shared memory

**Start daemon:**
```bash
# Option 1: Using entry point (recommended)
chronotick-daemon

# Option 2: With custom config
chronotick-daemon --config /path/to/config.yaml

# Option 3: With custom update frequency
chronotick-daemon --freq 100  # 100 Hz updates

# Option 4: Direct python execution
cd /path/to/ChronoTick/chronotick_shm
uv run python src/chronotick_shm/chronotick_daemon_server.py --config ../tsfm/chronotick_inference/config.yaml
```

**Verify daemon is running:**
```bash
# Check if shared memory exists
ls -lh /dev/shm/chronotick_shm

# Should show: -rw-r--r-- 1 user user 128 Oct 12 15:00 /dev/shm/chronotick_shm
```

**Daemon startup sequence:**
1. Loads configuration
2. Creates shared memory (128 bytes)
3. Initializes NTP client
4. Loads ML models
5. Enters warmup phase (60-180 seconds)
6. Starts main loop (100 Hz updates)

---

### Step 4: Deploy Client Applications

Now your applications can use ChronoTick!

#### Deployment Option A: Python Application (Direct Library)

**Install:**
```bash
pip install chronotick  # When available on PyPI
# OR for development:
pip install -e /path/to/ChronoTick/chronotick_shm
```

**Use in your application:**
```python
# your_app.py
from chronotick_shm import ChronoTickClient

def main():
    client = ChronoTickClient()

    # Check daemon is ready
    if not client.is_daemon_ready():
        print("ERROR: ChronoTick daemon not running")
        return

    # Get corrected time
    time_info = client.get_time()
    print(f"Corrected time: {time_info.corrected_timestamp}")
    print(f"Uncertainty: ±{time_info.uncertainty_seconds * 1000:.2f}ms")

    # Use for high-precision coordination
    target_time = time_info.corrected_timestamp + 10.0
    client.wait_until(target_time, tolerance_ms=1.0)
    execute_synchronized_action()

if __name__ == "__main__":
    main()
```

**Deploy:**
```bash
python your_app.py
```

---

#### Deployment Option B: Python AI Agent (SDK MCP)

**For agents using claude-agent-sdk:**

```python
# agent_app.py
from claude_agent_sdk import ClaudeSDKClient, create_sdk_mcp_server
from chronotick_shm.chronotick_sdk_mcp import (
    get_time,
    get_daemon_status,
    get_time_with_future_uncertainty
)

# Create SDK MCP server with ChronoTick tools
sdk_server = create_sdk_mcp_server(
    name="chronotick_server",
    version="1.0.0",
    tools=[get_time, get_daemon_status, get_time_with_future_uncertainty]
)

# Create agent
from claude_agent_sdk import ClaudeAgentOptions
agent_options = ClaudeAgentOptions(
    mcp_servers={"chronotick": sdk_server},
    allowed_tools=[
        "mcp__chronotick__get_time",
        "mcp__chronotick__get_daemon_status"
    ]
)

agent = ClaudeSDKClient(agent_options)

# Agent can now use ChronoTick
response = await agent.query("What is the current corrected time?")
```

---

#### Deployment Option C: Claude Code (Stdio MCP)

**For Claude Code integration:**

1. **Start the Stdio MCP server:**
```bash
chronotick-stdio-server

# Or:
uv run chronotick-stdio-server
```

2. **Configure Claude Code:**

Add to `~/.claude/config.json`:
```json
{
  "mcpServers": {
    "chronotick": {
      "command": "uv",
      "args": ["run", "chronotick-stdio-server"],
      "cwd": "/path/to/ChronoTick/chronotick_shm"
    }
  }
}
```

3. **Use in Claude Code:**
Claude Code can now use the tools:
- `mcp__chronotick__get_time`
- `mcp__chronotick__get_daemon_status`
- `mcp__chronotick__get_time_with_future_uncertainty`

---

### Step 5: Production Deployment

#### Systemd Service (Linux)

Create `/etc/systemd/system/chronotick-daemon.service`:

```ini
[Unit]
Description=ChronoTick High-Precision Time Daemon
After=network.target

[Service]
Type=simple
User=chronotick
Group=chronotick
ExecStart=/usr/local/bin/chronotick-daemon --config /etc/chronotick/config.yaml
Restart=always
RestartSec=10

# Performance optimizations
Nice=-10
CPUAffinity=0 1

# Resource limits
MemoryLimit=2G
CPUQuota=200%

[Install]
WantedBy=multi-user.target
```

**Enable and start:**
```bash
sudo systemctl enable chronotick-daemon
sudo systemctl start chronotick-daemon
sudo systemctl status chronotick-daemon
```

---

#### Docker Deployment

**Dockerfile:**
```dockerfile
FROM python:3.11-slim

# Install dependencies
RUN pip install chronotick chronotick-server

# Copy configuration
COPY config.yaml /app/config.yaml

# Expose shared memory
VOLUME /dev/shm

# Start daemon
CMD ["chronotick-daemon", "--config", "/app/config.yaml"]
```

**Build and run:**
```bash
docker build -t chronotick-daemon .
docker run -d \
  --name chronotick \
  -v /dev/shm:/dev/shm \
  -v /etc/chronotick:/etc/chronotick \
  --restart unless-stopped \
  chronotick-daemon
```

**Note:** Shared memory must be shared between daemon container and client applications!

---

#### Kubernetes Deployment

**StatefulSet for daemon:**
```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: chronotick-daemon
spec:
  serviceName: chronotick
  replicas: 1  # Only one daemon per node!
  selector:
    matchLabels:
      app: chronotick-daemon
  template:
    metadata:
      labels:
        app: chronotick-daemon
    spec:
      containers:
      - name: daemon
        image: chronotick-daemon:latest
        volumeMounts:
        - name: shm
          mountPath: /dev/shm
        - name: config
          mountPath: /etc/chronotick
      volumes:
      - name: shm
        hostPath:
          path: /dev/shm
          type: Directory
      - name: config
        configMap:
          name: chronotick-config
```

---

## Deployment Architectures

### Single-Node Deployment

```
┌───────────────────────────────────┐
│         Single Server             │
│                                   │
│  ┌─────────────────────────┐    │
│  │  ChronoTick Daemon      │    │
│  │  (systemd service)      │    │
│  └────────┬────────────────┘    │
│           │                      │
│           v                      │
│  ┌─────────────────────────┐    │
│  │  Shared Memory          │    │
│  │  /dev/shm/chronotick    │    │
│  └────────┬────────────────┘    │
│           │                      │
│           v                      │
│  ┌─────────────────────────┐    │
│  │  Your Applications      │    │
│  │  (multiple processes)   │    │
│  └─────────────────────────┘    │
│                                   │
└───────────────────────────────────┘
```

**Use case:** Single server with multiple applications

---

### Multi-Node Deployment (Distributed)

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│   Node 1         │    │   Node 2         │    │   Node 3         │
│                  │    │                  │    │                  │
│ ┌──────────────┐ │    │ ┌──────────────┐ │    │ ┌──────────────┐ │
│ │ CT Daemon    │ │    │ │ CT Daemon    │ │    │ │ CT Daemon    │ │
│ └──────┬───────┘ │    │ └──────┬───────┘ │    │ └──────┬───────┘ │
│        v         │    │        v         │    │        v         │
│ ┌──────────────┐ │    │ ┌──────────────┐ │    │ ┌──────────────┐ │
│ │ Shared Mem   │ │    │ │ Shared Mem   │ │    │ │ Shared Mem   │ │
│ └──────┬───────┘ │    │ └──────┬───────┘ │    │ └──────┬───────┘ │
│        v         │    │        v         │    │        v         │
│ ┌──────────────┐ │    │ ┌──────────────┐ │    │ ┌──────────────┐ │
│ │ App / Agent  │ │    │ │ App / Agent  │ │    │ │ App / Agent  │ │
│ └──────────────┘ │    │ └──────────────┘ │    │ └──────────────┘ │
└──────────────────┘    └──────────────────┘    └──────────────────┘
```

**Use case:** Distributed systems where each node needs its own time daemon

**Note:** Each node runs its own daemon. Applications on each node coordinate using corrected timestamps.

---

## Monitoring and Maintenance

### Health Checks

**Check daemon status:**
```python
from chronotick_shm import ChronoTickClient

client = ChronoTickClient()
info = client.get_daemon_info()

print(f"Status: {info['status']}")
print(f"Uptime: {info['daemon_uptime']:.1f}s")
print(f"NTP ready: {info['ntp_ready']}")
print(f"Models ready: {info['models_ready']}")
```

**Check from command line:**
```bash
chronotick-client status
```

### Logs

**View daemon logs:**
```bash
# Systemd
sudo journalctl -u chronotick-daemon -f

# Docker
docker logs -f chronotick

# Direct process
# Check wherever you redirected stderr
```

### Troubleshooting

**Daemon not running:**
```bash
# Check process
ps aux | grep chronotick

# Check shared memory
ls -lh /dev/shm/chronotick_shm

# Start daemon
chronotick-daemon --config config.yaml
```

**High latency:**
```python
info = client.get_daemon_info()
print(f"Avg latency: {info['average_latency_ms']}ms")
```

**NTP issues:**
```python
info = client.get_daemon_info()
print(f"NTP ready: {info['ntp_ready']}")
print(f"Measurements: {info['measurement_count']}")
print(f"Last sync: {info['seconds_since_ntp']}s ago")
```

---

## Security Considerations

1. **Shared Memory Permissions:**
   - Default: 0644 (owner read/write, others read)
   - Consider: 0640 for restricted access

2. **Daemon User:**
   - Run as dedicated user (not root)
   - Create `chronotick` user for production

3. **Network Access:**
   - Daemon needs UDP port 123 for NTP
   - Firewall rules may need adjustment

4. **Resource Limits:**
   - Set memory limits (2GB recommended)
   - Set CPU limits if needed
   - Monitor CPU usage (should be <10%)

---

## Performance Tuning

### Update Frequency

```bash
# Higher frequency = more CPU, lower latency variance
chronotick-daemon --freq 1000  # 1000 Hz (high precision)
chronotick-daemon --freq 100   # 100 Hz (balanced, default)
chronotick-daemon --freq 10    # 10 Hz (low overhead)
```

### CPU Affinity

```bash
# Pin to specific cores for consistent performance
chronotick-daemon --cpu-affinity 0 1
```

### Model Selection

**For lowest latency:**
- Use CPU-only models
- Short-term only (for <1 hour sessions)

**For long-term stability:**
- Use DUAL-MODEL (short + long term)
- Recommended for 24/7 operation

---

## Summary

**Deployment Flow:**

1. **Install** → Install tsfm + chronotick_shm packages
2. **Configure** → Create config.yaml with NTP + model settings
3. **Start Daemon** → `chronotick-daemon --config config.yaml`
4. **Verify** → Check `/dev/shm/chronotick_shm` exists
5. **Deploy Clients** → Applications use ChronoTickClient or MCP tools
6. **Monitor** → Check daemon status and health metrics

**Production Checklist:**

- [ ] Daemon runs as systemd service
- [ ] Configuration file in /etc/chronotick/
- [ ] Logs configured and monitored
- [ ] Health checks in place
- [ ] Resource limits set
- [ ] Automatic restart on failure
- [ ] Monitoring alerts configured

**Architecture:** Application → Client Library → Shared Memory → Daemon → (NTP + ML Models)

**Result:** Sub-millisecond time corrections with uncertainty quantification! 🎯
