# ChronoTick Shared Memory - Usage Guide

**Quick start guide for using ChronoTick with shared memory IPC**

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Running the Daemon](#running-the-daemon)
4. [Using SDK MCP Tools](#using-sdk-mcp-tools)
5. [Using the Evaluation Client](#using-the-evaluation-client)
6. [Integration Examples](#integration-examples)
7. [Troubleshooting](#troubleshooting)
8. [Performance Tuning](#performance-tuning)

---

## Quick Start

**3-Minute Setup:**

```bash
# 1. Navigate to chronotick_shm directory
cd chronotick_shm/

# 2. Test shared memory configuration
python shm_config.py
# Should show: "✅ All self-tests passed!"

# 3. Start the daemon (Terminal 1)
python chronotick_daemon.py --config ../tsfm/chronotick_inference/config.yaml

# Wait for: "✅ ChronoTick daemon ready - warmup complete!" (~3 minutes)

# 4. Test the client (Terminal 2)
python chronotick_client.py read
# Should show corrected time with uncertainty

# 5. Run benchmark
python chronotick_client.py benchmark --iterations 10000
# Should show <1μs average latency
```

**Quick Test with SDK Tools:**

```python
# Terminal 3 - Python REPL
from tools.chronotick_sdk_tools import get_shared_memory, read_chronotick_data
import time

# Connect and read
shm = get_shared_memory()
data = read_chronotick_data()

print(f"Corrected time: {data.get_corrected_time_at(time.time()):.6f}")
print(f"Confidence: {data.confidence:.1%}")
print(f"Source: {data.source.name}")
```

---

## Installation

### Prerequisites

**System Requirements:**
- Python 3.8+
- Linux or macOS (Windows not currently supported)
- ~200 MB RAM for daemon
- Network access for NTP queries (UDP port 123)

**Python Dependencies:**

```bash
# Core dependencies (already in ChronoTick)
cd tsfm/
uv sync

# For SDK MCP tools (agent integration)
pip install claude-agent-sdk

# Optional: For performance monitoring
pip install psutil
```

### Verify Installation

```bash
# Check Python version
python --version
# Should be 3.8 or higher

# Check ChronoTick installation
cd tsfm/
uv run python -c "from chronotick_inference import ChronoTickInferenceEngine; print('✓ ChronoTick installed')"

# Check claude-agent-sdk (if using SDK tools)
python -c "from claude_agent_sdk import tool; print('✓ claude-agent-sdk installed')"
```

---

## Running the Daemon

### Basic Usage

**Start with defaults:**
```bash
python chronotick_daemon.py
```

This uses:
- Default config: `../tsfm/chronotick_inference/config.yaml`
- Update frequency: 100 Hz
- No CPU pinning

### With Custom Configuration

**Specify config file:**
```bash
python chronotick_daemon.py --config /path/to/config.yaml
```

**Adjust update frequency:**
```bash
# Low overhead (10 Hz)
python chronotick_daemon.py --freq 10

# Balanced (100 Hz) - default
python chronotick_daemon.py --freq 100

# Maximum freshness (1000 Hz)
python chronotick_daemon.py --freq 1000
```

**CPU pinning for performance:**
```bash
# Pin to cores 0 and 1
python chronotick_daemon.py --cpu-affinity 0 1

# Pin to core 0 only (single threaded)
python chronotick_daemon.py --cpu-affinity 0
```

**Debug mode:**
```bash
python chronotick_daemon.py --log-level DEBUG
```

### Warmup Phase

The daemon requires a warmup period to collect quality NTP measurements:

```
🕒 ChronoTick warmup: 10.0% complete, 162s remaining
🕒 ChronoTick warmup: 50.0% complete, 90s remaining
🕒 ChronoTick warmup: 90.0% complete, 18s remaining
✅ ChronoTick daemon ready - warmup complete!
```

**Warmup Duration:** ~3 minutes (180 seconds)
- Phase 1: Rapid sampling (1 Hz for 180s)
- Phase 2: Normal operation (0.1 Hz ongoing)

**Why So Long?**
- Need minimum 30 quality NTP measurements
- Quality threshold: <10ms uncertainty
- Some measurements rejected due to network conditions
- Ensures stable, accurate time corrections

### Running as a System Service

**systemd (Linux):**

Create `/etc/systemd/system/chronotick.service`:
```ini
[Unit]
Description=ChronoTick Shared Memory Daemon
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/path/to/ChronoTick/chronotick_shm
ExecStart=/usr/bin/python3 chronotick_daemon.py --config /path/to/config.yaml
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable chronotick
sudo systemctl start chronotick
sudo systemctl status chronotick
```

### Stopping the Daemon

**Gracefully (Ctrl+C):**
```bash
# In daemon terminal, press Ctrl+C
# Daemon will clean up shared memory
```

**Kill command:**
```bash
# Find PID
ps aux | grep chronotick_daemon

# Send SIGTERM (graceful)
kill <PID>

# If hung, force kill (may leave shared memory)
kill -9 <PID>
```

**Manual Cleanup (if daemon crashed):**
```bash
# Linux
rm /dev/shm/chronotick_shm

# Or via Python
python -c "from multiprocessing.shared_memory import SharedMemory; SharedMemory('chronotick_shm', create=False).unlink()"
```

---

## Using SDK MCP Tools

### Basic Agent Setup

**Create agent with ChronoTick tools:**

```python
from tools.create_chronotick_agent import create_chronotick_agent
from claude_agent_sdk import ClaudeAgent

# Create agent
agent_options = create_chronotick_agent()
agent = ClaudeAgent(agent_options)

# Use agent
response = await agent.run("What is the current corrected time?")
print(response.text)
```

### Available Tools

**1. get_time**

Get corrected time with uncertainty bounds:

```python
response = await agent.run("What time is it with uncertainty?")
```

Returns:
- Corrected timestamp
- System timestamp
- Offset correction
- Drift rate
- Uncertainty estimates
- Confidence level
- Data source
- Validity period

**2. get_daemon_status**

Monitor daemon health:

```python
response = await agent.run("Show me ChronoTick daemon status")
```

Returns:
- Daemon status (ready/warmup/error)
- Uptime
- Measurement count
- Performance metrics
- Memory usage
- CPU affinity

**3. get_time_with_future_uncertainty**

Project uncertainty into future:

```python
response = await agent.run("What will the time uncertainty be in 5 minutes?")
```

Returns:
- Current time and uncertainty
- Future time and uncertainty
- Uncertainty increase

### Custom Tool Selection

**Time-only agent:**
```python
from tools.create_chronotick_agent import create_minimal_agent

agent_options = create_minimal_agent()
agent = ClaudeAgent(agent_options)
```

**Monitoring-only agent:**
```python
from tools.create_chronotick_agent import create_monitoring_agent

agent_options = create_monitoring_agent()
agent = ClaudeAgent(agent_options)
```

**Specific tools:**
```python
from tools.create_chronotick_agent import create_chronotick_agent

agent_options = create_chronotick_agent(
    allowed_tools=[
        "mcp__chronotick__get_time",
        "mcp__chronotick__get_daemon_status"
    ]
)
```

### Multi-Service Integration

**Combine ChronoTick with other MCP servers:**

```python
from claude_agent_sdk import ClaudeAgentOptions, create_sdk_mcp_server
from tools.chronotick_sdk_tools import get_time, get_daemon_status

# ChronoTick server
chronotick_server = create_sdk_mcp_server(
    name="chronotick",
    version="1.0.0",
    tools=[get_time, get_daemon_status]
)

# Add other servers (filesystem, git, etc.)
# other_server = ...

# Create multi-service agent
agent_options = ClaudeAgentOptions(
    mcp_servers={
        "chronotick": chronotick_server,
        # "filesystem": filesystem_server,
        # "git": git_server,
    },
    allowed_tools=[
        "mcp__chronotick__get_time",
        "mcp__chronotick__get_daemon_status",
        # ... other tools
    ]
)

agent = ClaudeAgent(agent_options)
```

---

## Using the Evaluation Client

### Read Time Once

```bash
python chronotick_client.py read
```

Output:
```
ChronoTick Time Reading
============================================================
Corrected Time:  1704556800.123456
System Time:     1704556800.122222
Offset:          +1234.567μs
Drift Rate:      +1.500μs/s
Uncertainty:     ±12.345μs
Confidence:      95.0%
Source:          FUSION
Valid:           ✓
NTP Ready:       ✓
Models Ready:    ✓
```

### Monitor Continuously

```bash
python chronotick_client.py monitor --interval 0.1
```

Output:
```
ChronoTick Continuous Monitor
============================================================
Press Ctrl+C to stop

Time: 1704556800.123456  Offset: +1234.5μs  Uncertainty: ±  12.3μs  Confidence:  95%  Source: FUSION
```

### Check Daemon Status

```bash
python chronotick_client.py status
```

Output:
```
ChronoTick Daemon Status
============================================================
Status:          READY
Uptime:          456.7s (7.6 min)
Warmup Complete: ✓

Data Collection:
  Measurements:  152
  Corrections:   15,230
  Last NTP:      45.2s ago
  NTP Ready:     ✓
  Models Ready:  ✓

Performance:
  Avg Latency:   0.450ms
  Call Latency:  0.0003ms
  Memory:        185.3MB
  CPU Affinity:  [0, 1]

Current Correction:
  Source:        FUSION
  Confidence:    95.0%
```

### Run Benchmark

```bash
python chronotick_client.py benchmark --iterations 100000
```

Output:
```
ChronoTick Performance Benchmark
============================================================
Running benchmark: 100,000 iterations...

Results (100,000 iterations):
  Total Time:        0.312s
  Average Latency:   312ns (0.31μs)
  Throughput:        320,512 reads/s
                     0.32 million reads/s

Performance Category:
  ⭐⭐⭐ EXCELLENT - Sub-500ns latency!
```

### Export as JSON

```bash
python chronotick_client.py json --pretty
```

Output:
```json
{
  "corrected_time": 1704556800.123456,
  "system_time": 1704556800.122222,
  "offset_correction": 0.001234,
  "drift_rate": 1.5e-06,
  "offset_uncertainty": 1e-05,
  "drift_uncertainty": 1e-09,
  "time_uncertainty": 1.2e-05,
  "confidence": 0.95,
  "source": "FUSION",
  "prediction_time": 1704556795.0,
  "valid_until": 1704556855.0,
  "is_valid": true,
  "is_ntp_ready": true,
  "is_models_ready": true,
  "is_warmup_complete": true,
  "daemon_uptime": 456.78,
  "measurement_count": 152,
  "total_corrections": 15230
}
```

---

## Integration Examples

### Example 1: Simple Time Query

```python
import asyncio
from tools.create_chronotick_agent import create_minimal_agent
from claude_agent_sdk import ClaudeAgent

async def get_current_time():
    agent_options = create_minimal_agent()
    agent = ClaudeAgent(agent_options)

    response = await agent.run("What time is it?")
    print(response.text)

asyncio.run(get_current_time())
```

### Example 2: Health Monitoring

```python
import asyncio
import time
from tools.create_chronotick_agent import create_monitoring_agent
from claude_agent_sdk import ClaudeAgent

async def monitor_health():
    agent_options = create_monitoring_agent()
    agent = ClaudeAgent(agent_options)

    while True:
        response = await agent.run("Check ChronoTick status")
        print(response.text)
        await asyncio.sleep(60)  # Check every minute

asyncio.run(monitor_health())
```

### Example 3: Direct Shared Memory Access

```python
from tools.chronotick_sdk_tools import get_shared_memory, read_chronotick_data
import time

# Connect once
shm = get_shared_memory()

# Read many times (very fast)
for i in range(1000):
    data = read_chronotick_data()
    corrected_time = data.get_corrected_time_at(time.time())

    if i % 100 == 0:
        print(f"Iteration {i}: {corrected_time:.6f}")

    time.sleep(0.01)  # 10ms between reads
```

### Example 4: Future Uncertainty Planning

```python
import asyncio
from tools.create_chronotick_agent import create_chronotick_agent
from claude_agent_sdk import ClaudeAgent

async def plan_coordinated_action():
    agent_options = create_chronotick_agent()
    agent = ClaudeAgent(agent_options)

    # Check uncertainty for action 5 minutes in the future
    response = await agent.run(
        "I need to coordinate with another agent in 300 seconds. "
        "What will the time uncertainty be?"
    )
    print(response.text)

    # Agent decides if uncertainty is acceptable
    # If yes, proceed with coordination
    # If no, wait for better conditions

asyncio.run(plan_coordinated_action())
```

---

## Troubleshooting

### Daemon Won't Start

**Error: "Configuration file not found"**
```bash
# Solution: Specify full path to config
python chronotick_daemon.py --config /full/path/to/config.yaml
```

**Error: "Shared memory already exists"**
```bash
# Solution: Remove stale shared memory
rm /dev/shm/chronotick_shm

# Or restart daemon (will reuse existing)
python chronotick_daemon.py
```

**Error: "Permission denied"**
```bash
# Solution: Check /dev/shm permissions
ls -l /dev/shm/

# If needed, run as sudo (not recommended for production)
sudo python chronotick_daemon.py
```

### Tools Can't Connect

**Error: "ChronoTick daemon not running"**
```bash
# Check if daemon is running
ps aux | grep chronotick_daemon

# Check if shared memory exists
ls -l /dev/shm/chronotick_shm

# Restart daemon
python chronotick_daemon.py
```

### Poor Performance

**Latency >1μs:**
```bash
# Check system load
top

# Run with CPU pinning
python chronotick_daemon.py --cpu-affinity 0

# Reduce update frequency to free CPU
python chronotick_daemon.py --freq 10
```

**High Memory Usage:**
```bash
# Check actual usage
python chronotick_client.py status

# If >500MB, restart daemon
# May indicate memory leak or model issues
```

### NTP Issues

**Warning: "Poor NTP measurement"**
```
# Check network connectivity
ping pool.ntp.org

# Check NTP port (UDP 123) not blocked
sudo tcpdump -i any udp port 123

# Try different NTP servers in config
```

**Warmup Takes Too Long:**
```
# Normal: ~3 minutes
# If >5 minutes, check:
# - Network latency to NTP servers
# - Firewall blocking UDP 123
# - System clock very wrong (>1 second off)
```

---

## Performance Tuning

### Update Frequency

**Choose based on use case:**

```bash
# Low overhead - system monitoring (10 Hz)
python chronotick_daemon.py --freq 10

# Balanced - general purpose (100 Hz) [DEFAULT]
python chronotick_daemon.py --freq 100

# High frequency - real-time (1000 Hz)
python chronotick_daemon.py --freq 1000
```

**CPU Usage:**
- 10 Hz: ~0.1% CPU
- 100 Hz: ~1% CPU
- 1000 Hz: ~5-10% CPU

### CPU Pinning

**Single core (daemon only):**
```bash
python chronotick_daemon.py --cpu-affinity 0
```

**Multiple cores (daemon + NTP + models):**
```bash
python chronotick_daemon.py --cpu-affinity 0 1 2
```

**Check affinity:**
```python
import psutil
p = psutil.Process()
print(p.cpu_affinity())
```

### NUMA Optimization

**For multi-socket systems:**
```bash
# Check NUMA topology
numactl --hardware

# Pin to specific NUMA node
numactl --cpunodebind=0 --membind=0 python chronotick_daemon.py
```

### Benchmarking

**Quick check:**
```bash
python chronotick_client.py benchmark --iterations 10000
```

**Comprehensive:**
```bash
python chronotick_client.py benchmark --iterations 1000000
```

**Expected Performance:**
- Sub-500ns: ⭐⭐⭐ Excellent
- 500ns-1μs: ⭐⭐ Good
- 1μs-10μs: ⭐ Acceptable
- >10μs: ⚠️ Check system load

---

## Next Steps

- **Technical Design:** See [TECHNICAL_DESIGN.md](TECHNICAL_DESIGN.md) for architecture details
- **Performance Analysis:** See [PERFORMANCE.md](PERFORMANCE.md) for benchmarks
- **Integration Guide:** See [../IPC_mechanism_python.md](../../IPC_mechanism_python.md) for IPC patterns
- **SDK Guide:** See [../GUIDE_SDK_MCP_SHARED_MEMORY.md](../../GUIDE_SDK_MCP_SHARED_MEMORY.md) for SDK integration

---

## Support

**Issues:**
- Check [Troubleshooting](#troubleshooting) section
- Review daemon logs with `--log-level DEBUG`
- File issue at ChronoTick repository

**Performance:**
- Run `chronotick_client.py benchmark`
- Check system load and CPU affinity
- Review [Performance Tuning](#performance-tuning)

**Questions:**
- See [TECHNICAL_DESIGN.md](TECHNICAL_DESIGN.md) for architecture
- Review example code in `examples/` directory
- Consult [SDK documentation](https://github.com/anthropics/claude-agent-sdk-python)
