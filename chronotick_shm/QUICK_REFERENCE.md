# ChronoTick Shared Memory - Quick Reference

**Ultra-fast time sync for AI agents - Essential commands and patterns**

---

## 🚀 Quick Start (3 Minutes)

```bash
# Terminal 1: Start daemon
cd chronotick_shm
python3 chronotick_daemon.py --config ../tsfm/chronotick_inference/config.yaml

# Terminal 2: Test (after warmup ~3min)
cd chronotick_shm
python3 chronotick_client.py read
python3 chronotick_client.py benchmark --iterations 10000
```

---

## 📋 Command Reference

### Daemon Commands

```bash
# Basic start
python3 chronotick_daemon.py

# With custom config
python3 chronotick_daemon.py --config /path/to/config.yaml

# Adjust frequency (1-1000 Hz)
python3 chronotick_daemon.py --freq 100      # Default: balanced
python3 chronotick_daemon.py --freq 10       # Low CPU
python3 chronotick_daemon.py --freq 1000     # High frequency

# CPU pinning (performance)
python3 chronotick_daemon.py --cpu-affinity 0 1

# Debug mode
python3 chronotick_daemon.py --log-level DEBUG

# Full example
python3 chronotick_daemon.py \
  --config ../tsfm/chronotick_inference/config.yaml \
  --freq 100 \
  --cpu-affinity 0 1 \
  --log-level INFO
```

### Client Commands

```bash
# Read once
python3 chronotick_client.py read

# Monitor continuously (10 Hz)
python3 chronotick_client.py monitor --interval 0.1

# Daemon status
python3 chronotick_client.py status

# Performance benchmark
python3 chronotick_client.py benchmark --iterations 10000
python3 chronotick_client.py benchmark --iterations 1000000  # Comprehensive

# JSON export
python3 chronotick_client.py json
python3 chronotick_client.py json --pretty
```

---

## 💻 Code Examples

### SDK MCP Integration (Agent)

```python
from tools.create_chronotick_agent import create_chronotick_agent
from claude_agent_sdk import ClaudeAgent
import asyncio

async def main():
    # Create agent with ChronoTick tools
    agent_options = create_chronotick_agent()
    agent = ClaudeAgent(agent_options)

    # Query time
    response = await agent.run("What is the current corrected time?")
    print(response.text)

asyncio.run(main())
```

### Direct Shared Memory Access (Fast)

```python
from tools.chronotick_sdk_tools import get_shared_memory, read_chronotick_data
import time

# Connect once (global handle pattern)
shm = get_shared_memory()

# Read many times (very fast ~300ns)
for i in range(1000):
    data = read_chronotick_data()
    corrected_time = data.get_corrected_time_at(time.time())
    uncertainty = data.get_time_uncertainty(0)

    print(f"Time: {corrected_time:.6f} ±{uncertainty*1e6:.3f}μs")
    time.sleep(0.1)
```

### Minimal Example

```python
from tools.chronotick_sdk_tools import get_shared_memory, read_chronotick_data
import time

# One-liner for current corrected time
shm = get_shared_memory()
data = read_chronotick_data()
print(data.get_corrected_time_at(time.time()))
```

---

## 🔧 Troubleshooting

### Daemon Won't Start

```bash
# Check for stale shared memory
ls -l /dev/shm/chronotick_shm

# Remove if exists
rm /dev/shm/chronotick_shm

# Verify config exists
ls -l /path/to/config.yaml

# Start with full path
python3 chronotick_daemon.py --config /full/path/to/config.yaml
```

### Tools Can't Connect

```bash
# Check daemon running
ps aux | grep chronotick_daemon

# Check shared memory exists
ls -l /dev/shm/chronotick_shm

# Restart daemon
python3 chronotick_daemon.py
```

### Poor Performance

```bash
# Check system load
top

# Use CPU pinning
python3 chronotick_daemon.py --cpu-affinity 0

# Reduce frequency
python3 chronotick_daemon.py --freq 10

# Benchmark
python3 chronotick_client.py benchmark --iterations 10000
# Expected: <500ns for ⭐⭐⭐ EXCELLENT
```

### Manual Cleanup

```bash
# Linux: Remove shared memory
rm /dev/shm/chronotick_shm

# Python: Programmatic cleanup
python3 -c "from multiprocessing.shared_memory import SharedMemory; SharedMemory('chronotick_shm', create=False).unlink()"

# Check processes
ps aux | grep chronotick
kill <PID>
```

---

## 📊 Performance Expectations

| Metric              | Expected Value       | Rating           |
|---------------------|----------------------|------------------|
| Read latency        | 300-500ns           | ⭐⭐⭐ Excellent |
| Read latency        | 500ns-1μs           | ⭐⭐ Good        |
| Read latency        | 1μs-10μs            | ⭐ Acceptable    |
| Read latency        | >10μs               | ⚠️ Check load    |
| Throughput          | 1-3 million/s       | Normal           |
| Daemon CPU (100Hz)  | ~1%                 | Normal           |
| Daemon CPU (1000Hz) | ~5-10%              | Normal           |
| Daemon memory       | 150-200MB           | Normal           |

---

## 🎯 Common Use Cases

### Use Case 1: Real-Time Monitoring

```bash
# Terminal 1: Daemon
python3 chronotick_daemon.py --freq 1000 --cpu-affinity 0

# Terminal 2: Monitor
python3 chronotick_client.py monitor --interval 0.01  # 100 Hz monitoring
```

### Use Case 2: Agent Coordination

```python
# agent1.py
from tools.create_chronotick_agent import create_chronotick_agent
from claude_agent_sdk import ClaudeAgent

agent = ClaudeAgent(create_chronotick_agent())
response = await agent.run("Get corrected time for coordination")
```

### Use Case 3: Performance Testing

```bash
# Warm up daemon first (3 min)
python3 chronotick_daemon.py

# Then benchmark
python3 chronotick_client.py benchmark --iterations 1000000
```

### Use Case 4: System Integration

```python
# integration.py
from tools.chronotick_sdk_tools import get_shared_memory, read_chronotick_data
import time
import json

shm = get_shared_memory()

def get_time_json():
    data = read_chronotick_data()
    current_time = time.time()

    return json.dumps({
        "corrected_time": data.get_corrected_time_at(current_time),
        "uncertainty": data.get_time_uncertainty(0),
        "confidence": data.confidence,
        "source": data.source.name
    })

print(get_time_json())
```

---

## 🔐 Security & Permissions

### Shared Memory Permissions

```bash
# Check permissions (Linux)
ls -l /dev/shm/chronotick_shm
# Should be readable by your user

# If needed, run daemon and clients as same user
# Avoid sudo unless necessary
```

### Network Requirements

```bash
# NTP requires UDP port 123 outbound
sudo iptables -L | grep 123  # Check firewall

# Test NTP connectivity
ntpdate -q pool.ntp.org
```

---

## 📚 Documentation Links

- **[README.md](README.md)**: Project overview and features
- **[Technical Design](docs/TECHNICAL_DESIGN.md)**: Architecture details
- **[Usage Guide](docs/USAGE_GUIDE.md)**: Complete guide
- **[IPC Guide](../IPC_mechanism_python.md)**: Python IPC patterns
- **[SDK Guide](../GUIDE_SDK_MCP_SHARED_MEMORY.md)**: SDK integration

---

## ⚡ Performance Tips

1. **CPU Pinning**: Use `--cpu-affinity 0` for consistent performance
2. **Frequency**: Start with 100 Hz, adjust based on needs
3. **Warmup**: Wait full 3 minutes for quality measurements
4. **Benchmarking**: Run `benchmark` to verify performance
5. **Monitoring**: Use `status` to check daemon health

---

## 🎓 Key Concepts

### Lock-Free Pattern

```python
# Daemon writes with sequence number
seq = (seq + 1) % 2**32
write_data(buffer, data)  # Includes seq

# Readers detect torn reads
seq_before = read_seq(buffer)
data = read_data(buffer)
seq_after = read_seq(buffer)
if seq_before != seq_after:
    data = read_data(buffer)  # Retry (rare)
```

### Global Handle Pattern

```python
# Attach once (expensive ~1.5ms)
_shm = None
def get_shared_memory():
    global _shm
    if _shm is None:
        _shm = SharedMemory(name=SHARED_MEMORY_NAME, create=False)
    return _shm

# Use many times (cheap ~300ns)
shm = get_shared_memory()  # Returns cached handle
```

### Memory Layout

```
128 bytes total (2 cache lines)
├─ [0-64]:   First cache line (timestamps, corrections)
└─ [64-128]: Second cache line (metadata, stats)
```

---

## 🔄 Comparison: Queue vs Shared Memory

| Aspect            | Queue-Based | Shared Memory | Winner |
|-------------------|-------------|---------------|---------|
| Latency           | ~1ms        | ~0.3μs        | 🏆 SHM  |
| Throughput        | ~1K/s       | ~3M/s         | 🏆 SHM  |
| Memory overhead   | ~10MB       | ~4KB          | 🏆 SHM  |
| Setup complexity  | Low         | Medium        | Queue   |
| Variable data     | Yes         | No (128B)     | Queue   |
| Multiple writers  | Yes         | No            | Queue   |
| Messaging         | Yes         | No            | Queue   |

**Use Shared Memory When:**
- High frequency reads (>10/s)
- Latency critical (<1ms)
- Fixed-size data
- Read-only access pattern

**Use Queues When:**
- Variable/complex data
- Multiple writers needed
- Request-response pattern
- Infrequent access

---

## 🏁 Success Checklist

- [ ] Daemon starts without errors
- [ ] Warmup completes (~3 minutes)
- [ ] Client can read time
- [ ] Benchmark shows <1μs latency
- [ ] Status shows "READY"
- [ ] NTP measurements > 30
- [ ] Confidence > 90%

**If all ✓ → System ready! 🎉**

---

**ChronoTick Shared Memory** - Fast time for fast agents
⚡ ~300ns latency | 🔒 Lock-free | 💾 128 bytes | 🚀 3M reads/s
