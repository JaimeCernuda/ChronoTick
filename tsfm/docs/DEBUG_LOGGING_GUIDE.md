# ChronoTick MCP Server Debug Logging Guide

The ChronoTick MCP server includes comprehensive debug logging capabilities to track function calls, model inputs/outputs, IPC communication, and system diagnostics.

## Quick Start

### Enable Debug Tracing
```bash
# Enable comprehensive debug tracing
/home/jcernuda/ChronoTick/tsfm/.venv/bin/python chronotick_mcp.py --debug-trace

# Save debug logs to file
/home/jcernuda/ChronoTick/tsfm/.venv/bin/python chronotick_mcp.py --debug-trace --debug-log-file /tmp/chronotick-debug.log

# Set log level to DEBUG (shows all logs)
/home/jcernuda/ChronoTick/tsfm/.venv/bin/python chronotick_mcp.py --log-level DEBUG --debug-trace
```

### Claude Code Connection with Debug Logging
```bash
claude mcp add chronotick-debug --scope user \
  --env PYTHONPATH=/home/jcernuda/ChronoTick/tsfm \
  -- bash -c "cd /home/jcernuda/ChronoTick/tsfm && /home/jcernuda/ChronoTick/tsfm/.venv/bin/python chronotick_mcp.py --debug-trace --debug-log-file /tmp/chronotick-mcp-debug.log"
```

## Debug Logging Features

### 1. **Function Call Tracing**
Tracks entry/exit of all critical functions with:
- Function name and module path
- Input arguments (safely serialized)
- Return values (truncated if large)
- Execution timing in milliseconds
- Unique call IDs to correlate entry/exit

Example output:
```json
{
  "function": "chronotick_inference.mcp_server._handle_get_time",
  "call_id": 140234567890,
  "args": [{"include_stats": false}],
  "kwargs": {},
  "timestamp": 1692547890.123
}
```

### 2. **Model Input/Output Logging**
Specialized logging for machine learning operations:
- NumPy array shapes, dtypes, and statistics (mean, std, min, max)
- Model prediction inputs and outputs
- Feature engineering transformations
- Uncertainty calculations

Example output:
```json
{
  "function": "chronotick_inference.real_data_pipeline.get_real_clock_correction",
  "result": {
    "type": "CorrectionWithBounds",
    "fields": {
      "offset_correction": 0.000025,
      "drift_rate": 0.000001,
      "offset_uncertainty": 0.000005,
      "drift_uncertainty": 0.0000001,
      "confidence": 0.85,
      "source": "fusion"
    }
  },
  "execution_time_ms": 12.5
}
```

### 3. **IPC Communication Tracking**
Monitors inter-process communication:
- Queue operations (put/get)
- Message serialization/deserialization
- Timeout handling
- Process health checks

### 4. **System Diagnostics**
Comprehensive system monitoring:
- Memory usage tracking
- CPU affinity settings
- Process lifecycle events
- Error conditions and recovery

## Debug Logging Levels

### Standard Logging Levels
- **ERROR**: Critical failures only
- **WARNING**: Warnings and errors
- **INFO**: Normal operational messages (default)
- **DEBUG**: All messages including function tracing

### Enhanced Debug Tracing
When `--debug-trace` is enabled, additional loggers are activated:
- `chronotick_inference.mcp_server.debug`
- `chronotick_inference.real_data_pipeline.debug`
- `chronotick_inference.daemon.debug`
- `chronotick_inference.ntp_client.debug`
- `chronotick_inference.predictive_scheduler.debug`
- `chronotick_inference.engine.debug`

## Command Line Options

### Basic Usage
```bash
# Standard logging
chronotick_mcp.py --log-level INFO

# Debug level logging  
chronotick_mcp.py --log-level DEBUG

# Enhanced tracing
chronotick_mcp.py --debug-trace
```

### Advanced Options
```bash
# Comprehensive debugging with file output
chronotick_mcp.py \
  --log-level DEBUG \
  --debug-trace \
  --debug-log-file /var/log/chronotick-debug.log

# Production with error logging only
chronotick_mcp.py --log-level ERROR

# Custom config with debugging
chronotick_mcp.py \
  --config /path/to/custom.yaml \
  --debug-trace \
  --debug-log-file /tmp/debug.log
```

## Log Output Formats

### Standard Format
```
2025-08-20 01:30:45 - chronotick_inference.mcp_server - INFO - Starting ChronoTick daemon...
```

### Debug Format (with --debug-trace)
```
2025-08-20 01:30:45.123 - chronotick_inference.mcp_server.debug - DEBUG - _handle_get_time:391 - ENTRY: {...}
```

## Monitoring and Analysis

### Real-time Log Monitoring
```bash
# Follow debug logs in real-time
tail -f /tmp/chronotick-debug.log

# Filter for specific events
tail -f /tmp/chronotick-debug.log | grep "PIPELINE_"

# Monitor function timing
tail -f /tmp/chronotick-debug.log | grep "execution_time_ms"
```

### Log Analysis Examples

**Find slow operations:**
```bash
grep "execution_time_ms" /tmp/chronotick-debug.log | awk -F'"execution_time_ms": ' '{print $2}' | awk -F',' '{print $1}' | sort -n | tail -10
```

**Track model predictions:**
```bash
grep "PIPELINE_EXIT" /tmp/chronotick-debug.log | grep "CorrectionWithBounds"
```

**Monitor IPC performance:**
```bash
grep "get_correction_from_daemon" /tmp/chronotick-debug.log
```

## Performance Impact

### Debug Logging Overhead
- **No Debug Tracing**: Minimal overhead (~1-2%)
- **Debug Tracing Enabled**: Moderate overhead (~5-10%)
- **File Logging**: Additional I/O overhead (~2-5%)

### Production Recommendations
- Use `--log-level ERROR` or `--log-level WARNING` in production
- Enable `--debug-trace` only for troubleshooting
- Use `--debug-log-file` to avoid console spam

## Troubleshooting Use Cases

### 1. **MCP Connection Issues**
```bash
# Enable full tracing to diagnose connection problems
chronotick_mcp.py --debug-trace --debug-log-file /tmp/mcp-debug.log

# Look for MCP protocol messages
grep "mcp.server" /tmp/mcp-debug.log
```

### 2. **Model Performance Issues**
```bash
# Track model I/O and timing
grep "PIPELINE_" /tmp/chronotick-debug.log | head -20

# Find slow predictions
grep "execution_time_ms.*[5-9][0-9]\|[0-9][0-9][0-9]" /tmp/chronotick-debug.log
```

### 3. **IPC Communication Problems**
```bash
# Monitor daemon communication
grep "_get_correction_from_daemon\|_get_daemon_status" /tmp/chronotick-debug.log

# Check for timeouts
grep "timeout\|TimeoutError" /tmp/chronotick-debug.log
```

### 4. **Memory/Performance Analysis**
```bash
# Track memory usage
grep "memory_usage_mb" /tmp/chronotick-debug.log

# Monitor request latencies
grep "call_latency_ms" /tmp/chronotick-debug.log
```

## Security Considerations

### Sensitive Data Handling
- Function arguments are safely serialized to avoid logging sensitive data
- Large data structures are truncated to prevent log bloat
- File paths and configuration values may be logged

### Log File Security
```bash
# Set appropriate permissions for debug logs
chmod 600 /tmp/chronotick-debug.log

# Use secure directory for production logs
mkdir -p /var/log/chronotick
chown chronotick:chronotick /var/log/chronotick
chmod 750 /var/log/chronotick
```

## Integration with Monitoring Systems

### Structured Logging Output
Debug logs use JSON format for easy parsing by monitoring tools:
- Elasticsearch/Logstash/Kibana (ELK)
- Prometheus/Grafana
- Custom monitoring solutions

### Example Log Parsing
```python
import json
import re

def parse_debug_logs(log_file):
    events = []
    with open(log_file) as f:
        for line in f:
            if 'ENTRY:' in line or 'EXIT:' in line or 'ERROR:' in line:
                # Extract JSON from log line
                json_match = re.search(r'(ENTRY|EXIT|ERROR): ({.*})', line)
                if json_match:
                    event_type = json_match.group(1)
                    event_data = json.loads(json_match.group(2))
                    events.append((event_type, event_data))
    return events
```

This comprehensive debug logging system ensures you can diagnose any issues with the ChronoTick MCP server operation, from high-level MCP protocol interactions down to individual model predictions and system metrics.