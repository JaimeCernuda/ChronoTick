# ChronoTick Quickstart Guide

Get started with ChronoTick in under 5 minutes. This guide walks you through installation, starting the daemon, and writing your first program using high-precision time synchronization.

---

## What is ChronoTick?

ChronoTick provides **microsecond-precision time corrections** for your applications through:

- **Real NTP measurements** combined with **ML prediction** (TimesFM 2.5)
- **Sub-millisecond read latency** (~300ns via shared memory)
- **Uncertainty quantification** at every timestamp
- **Easy Python API** that hides all complexity

Perfect for distributed systems, AI agent coordination, and any application requiring high-precision time synchronization.

---

## Installation

### Option A: Install from source (current)

```bash
# Clone the repository
git clone https://github.com/yourusername/ChronoTick.git
cd ChronoTick

# Install the daemon (server)
cd tsfm
uv sync --extra core-models

# Install the client library
cd ../chronotick_shm
uv sync
```

### Option B: Install from pip (future)

```bash
# When published to PyPI:
pip install chronotick           # Client library
pip install chronotick-server    # Server/daemon (optional)
```

---

## Starting the Daemon

The ChronoTick daemon runs in the background, collecting NTP measurements and providing corrected time via shared memory.

### Quick Start

```bash
# Start with default settings
chronotick-daemon
```

You'll see output like:

```
ChronoTick Daemon Starting...
✓ Shared memory created: /dev/shm/chronotick_shm (128 bytes)
✓ NTP client initialized (4 servers)
✓ Models loaded: short_term, long_term
⏳ Warming up... (60 seconds)
✓ Daemon ready! Update frequency: 100 Hz
```

### With Custom Configuration

```bash
# Create a config file (config.yaml)
cat > config.yaml << 'EOF'
clock:
  frequency_code: 9
  frequency_type: second

clock_measurement:
  ntp:
    servers:
      - pool.ntp.org
      - time.google.com
    timeout_seconds: 2.0
    max_acceptable_uncertainty: 0.1

short_term:
  enabled: true
  model_name: timesfm
  prediction_horizon: 30
  inference_interval: 1.0

long_term:
  enabled: true
  model_name: timesfm
  prediction_horizon: 60
  inference_interval: 30.0

fusion:
  enabled: true
  method: inverse_variance_weighting
EOF

# Start with config
chronotick-daemon --config config.yaml
```

### Verify Daemon is Running

```bash
# Check shared memory exists
ls -lh /dev/shm/chronotick_shm

# Should show:
# -rw-r--r-- 1 user user 128 Oct 12 15:00 /dev/shm/chronotick_shm
```

---

## Your First Program

Create a file `my_first_chronotick.py`:

```python
#!/usr/bin/env python3
"""My first ChronoTick program"""

from chronotick_shm import ChronoTickClient

def main():
    # Create a client
    client = ChronoTickClient()

    # Check if daemon is ready
    if not client.is_daemon_ready():
        print("ERROR: ChronoTick daemon not running")
        print("Start it with: chronotick-daemon")
        return

    # Get corrected time
    time_info = client.get_time()

    # Display results
    print(f"Corrected Time: {time_info.corrected_timestamp:.6f}")
    print(f"System Time:    {time_info.system_timestamp:.6f}")
    print(f"Uncertainty:    ±{time_info.uncertainty_seconds * 1000:.3f}ms")
    print(f"Confidence:     {time_info.confidence:.1%}")
    print(f"Data Source:    {time_info.source}")

if __name__ == "__main__":
    main()
```

Run it:

```bash
python my_first_chronotick.py
```

Output:

```
Corrected Time: 1697125234.567890
System Time:    1697125234.567123
Uncertainty:    ±2.341ms
Confidence:     95.2%
Data Source:    fusion
```

**Congratulations!** You just got your first high-precision time correction with uncertainty quantification!

---

## Common Use Cases

### Use Case 1: High-Precision Synchronization

Coordinate actions across distributed systems with sub-millisecond precision:

```python
from chronotick_shm import ChronoTickClient

client = ChronoTickClient()

# Schedule action 10 seconds from now
time_info = client.get_time()
target_time = time_info.corrected_timestamp + 10.0

print(f"Scheduling action for: {target_time}")

# Wait until target time (sub-millisecond precision)
client.wait_until(target_time, tolerance_ms=0.5)

# Execute synchronized action
execute_distributed_write()
print("Action executed at precise time!")
```

### Use Case 2: Future Uncertainty Projection

Plan ahead by knowing how uncertainty will grow:

```python
from chronotick_shm import ChronoTickClient

client = ChronoTickClient()

# Current time
now = client.get_time()
print(f"Current uncertainty: ±{now.uncertainty_seconds * 1000:.3f}ms")

# What will uncertainty be in 5 minutes?
future = client.get_future_time(300)  # 300 seconds = 5 minutes
print(f"Future uncertainty:  ±{future.uncertainty_seconds * 1000:.3f}ms")

# Decide if we need to resync
if future.uncertainty_seconds > 0.010:  # 10ms threshold
    print("Need to schedule NTP measurement before action")
```

### Use Case 3: Monitoring Daemon Health

Track daemon performance in production:

```python
from chronotick_shm import ChronoTickClient

client = ChronoTickClient()

# Get daemon information
info = client.get_daemon_info()

print(f"Status: {info['status']}")
print(f"Uptime: {info['daemon_uptime']:.1f}s")
print(f"NTP Measurements: {info['measurement_count']}")
print(f"Last Sync: {info['seconds_since_ntp']:.1f}s ago")
print(f"NTP Ready: {'✓' if info['ntp_ready'] else '✗'}")
print(f"Models Ready: {'✓' if info['models_ready'] else '✗'}")
print(f"Avg Latency: {info['average_latency_ms']:.3f}ms")
```

### Use Case 4: Quick Time Access

For one-off requests, use the convenience function:

```python
from chronotick_shm import get_current_time

# Get time without creating a client
time_info = get_current_time()
print(f"Quick time: {time_info.corrected_timestamp:.6f}")
```

### Use Case 5: Context Manager Pattern

Automatic cleanup with context managers:

```python
from chronotick_shm import ChronoTickClient

# Client automatically cleaned up when done
with ChronoTickClient() as client:
    time_info = client.get_time()
    print(f"Time: {time_info.corrected_timestamp}")

    # Do work...

# Cleanup happens here automatically
```

---

## Understanding the Output

When you call `client.get_time()`, you get a `CorrectedTime` object with these fields:

| Field | Description | Example |
|-------|-------------|---------|
| `corrected_timestamp` | Unix timestamp with corrections applied | `1697125234.567890` |
| `system_timestamp` | Raw system clock timestamp | `1697125234.567123` |
| `uncertainty_seconds` | Total time uncertainty (seconds) | `0.002341` (2.341ms) |
| `confidence` | Model confidence level (0.0-1.0) | `0.952` (95.2%) |
| `source` | Data source name | `fusion`, `ntp`, `cpu_model` |
| `offset_correction` | Clock offset correction (seconds) | `+0.000767` (+767μs) |
| `drift_rate` | Clock drift rate (seconds/second) | `+0.000012` (+12μs/s) |

### Interpreting Uncertainty

The `uncertainty_seconds` field tells you the **confidence interval** for the corrected time:

- **±1ms**: Excellent - sufficient for most distributed coordination
- **±5ms**: Good - typical NTP-level accuracy
- **±10ms**: Acceptable - usable for coordination
- **>50ms**: Poor - may need to wait for new NTP measurement

Example:

```python
time_info = client.get_time()

if time_info.uncertainty_seconds < 0.001:  # <1ms
    print("Excellent precision!")
    proceed_with_coordination()
elif time_info.uncertainty_seconds < 0.010:  # <10ms
    print("Good precision")
    proceed_with_coordination()
else:
    print("Waiting for better precision...")
    time.sleep(5)  # Wait for daemon to get new NTP measurement
```

---

## AI Agent Integration (MCP)

ChronoTick supports the **Model Context Protocol (MCP)** for AI agent integration.

### Option A: SDK MCP (In-Process)

For agents using `claude-agent-sdk`:

```python
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

# Create agent with ChronoTick tools
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

### Option B: Stdio MCP (Standalone Server)

For Claude Code or other MCP clients:

```bash
# Start the stdio MCP server
chronotick-stdio-server
```

Configure in `~/.claude/config.json`:

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

The agent can now call:
- `mcp__chronotick__get_time`
- `mcp__chronotick__get_daemon_status`
- `mcp__chronotick__get_time_with_future_uncertainty`

---

## Troubleshooting

### Problem: "ChronoTick daemon not running"

**Solution:**

```bash
# Check if daemon process is running
ps aux | grep chronotick

# Check shared memory
ls -lh /dev/shm/chronotick_shm

# Start daemon
chronotick-daemon
```

### Problem: High uncertainty (>50ms)

**Possible causes:**

1. **Daemon is warming up** - Wait 60 seconds for initial NTP measurements
2. **NTP servers unreachable** - Check network connectivity
3. **Models not loaded** - Check daemon logs

**Solution:**

```python
info = client.get_daemon_info()
print(f"NTP ready: {info['ntp_ready']}")
print(f"Models ready: {info['models_ready']}")
print(f"Measurements: {info['measurement_count']}")
```

### Problem: Daemon crashes or freezes

**Solution:**

```bash
# Check daemon logs
journalctl -u chronotick-daemon -f  # If using systemd

# Restart daemon
sudo systemctl restart chronotick-daemon

# Or manually:
pkill -9 chronotick-daemon
chronotick-daemon --config config.yaml
```

### Problem: Permission denied on /dev/shm

**Solution:**

```bash
# Check permissions
ls -lh /dev/shm/chronotick_shm

# Fix permissions (if needed)
chmod 644 /dev/shm/chronotick_shm
```

---

## Performance Characteristics

**Latency:**
- First call: ~1.5ms (shared memory attach)
- Subsequent calls: ~300ns (lock-free read)
- Daemon updates: 100 Hz (10ms intervals)

**Memory:**
- Shared memory: 128 bytes
- Daemon RAM: 50-200MB (depending on models)
- Client RAM: Negligible

**CPU:**
- Daemon: 5-10% of one core
- Client: <0.1% (negligible)

**Accuracy:**
- Short-term (<1 hour): ±1-5ms typical
- Long-term (8+ hours): ±2-10ms typical
- With fresh NTP: ±0.5-2ms typical

---

## Next Steps

### Production Deployment

- See [DEPLOYMENT_GUIDE.md](../DEPLOYMENT_GUIDE.md) for systemd, Docker, and Kubernetes deployment
- Set up monitoring and alerts
- Configure automatic restarts

### Advanced Usage

- Read the full [API documentation](api.md) (when available)
- Explore distributed synchronization patterns
- Integrate with your monitoring stack

### Development

- See [chronotick_shm/examples/](../examples/) for more examples
- Read [STRUCTURE.md](../STRUCTURE.md) for architecture details
- Check [INTERFACE_IMPROVEMENTS_COMPLETE.md](../INTERFACE_IMPROVEMENTS_COMPLETE.md) for recent changes

---

## Summary

**Basic workflow:**

1. **Start daemon**: `chronotick-daemon`
2. **Create client**: `client = ChronoTickClient()`
3. **Get time**: `time_info = client.get_time()`
4. **Use corrections**: `time_info.corrected_timestamp`

**Three lines to get started:**

```python
from chronotick_shm import ChronoTickClient
client = ChronoTickClient()
time_info = client.get_time()
```

**That's it!** You now have microsecond-precision time with uncertainty quantification.

---

## Support

- **Documentation**: See `/docs` directory
- **Examples**: See `/examples` directory
- **Issues**: Report at [GitHub Issues](https://github.com/yourusername/ChronoTick/issues)

---

**Welcome to ChronoTick!** Happy synchronizing! ⏱️
