# ChronoTick MCP Server Deployment Guide

The ChronoTick MCP (Model Context Protocol) server provides high-precision time services to AI agents like Claude Code through fast IPC communication.

## Features

- Real-time clock corrections with microsecond precision
- Error bounds and uncertainty quantification  
- Fast IPC communication for minimal latency
- Daemon lifecycle management (warmup, ready, error states)
- Performance monitoring and statistics

## Prerequisites

- Python 3.10 or later
- uv package manager
- Network access for NTP time synchronization
- Linux/macOS (Windows support may require modifications)

## Installation

1. **Install dependencies using uv:**
   ```bash
   cd /path/to/ChronoTick/tsfm
   uv sync
   ```

2. **Verify installation:**
   ```bash
   uv run python chronotick_mcp.py --help
   ```

## Configuration

1. **Configuration file location:**
   The MCP server automatically detects configuration files in this order:
   - `chronotick_inference/configs/hybrid_timesfm_chronos.yaml` (preferred)
   - `chronotick_inference/config.yaml` (fallback)
   - Or specify custom path with `--config` option

2. **Key configuration sections:**
   ```yaml
   clock_measurement:
     ntp:
       servers: ['pool.ntp.org', 'time.google.com']
       timeout_seconds: 2.0
       max_acceptable_uncertainty: 0.050
     timing:
       warm_up:
         duration_seconds: 60  # Warmup period for NTP measurements
         measurement_interval: 1.0
   ```

## Starting the MCP Server

### Method 1: Direct Execution
```bash
# Start with default configuration
uv run python chronotick_mcp.py

# Start with custom configuration
uv run python chronotick_mcp.py --config /path/to/custom/config.yaml

# Start with debug logging
uv run python chronotick_mcp.py --log-level DEBUG

# Start with comprehensive debug tracing (function calls, model I/O, IPC)
uv run python chronotick_mcp.py --debug-trace

# Start with debug tracing and log to file
uv run python chronotick_mcp.py --debug-trace --debug-log-file /tmp/chronotick-debug.log
```

### Method 2: Using the Entry Point
```bash
# After installation, you can also use:
uv run chronotick-mcp

# With options:
uv run chronotick-mcp --config /path/to/config.yaml --log-level INFO
```

## Server Startup Sequence

1. **Initialization:** Server loads configuration and sets up MCP handlers
2. **Daemon Start:** ChronoTick daemon process starts with real data pipeline
3. **Warmup Phase:** NTP collection begins, progress reported every 10 seconds
   ```
   🕒 ChronoTick warmup: 25.0% complete, 45s remaining
   🕒 ChronoTick warmup: 50.0% complete, 30s remaining
   🕒 ChronoTick warmup: 75.0% complete, 15s remaining
   ```
4. **Ready:** Warmup complete, server ready for agent connections
   ```
   ✅ ChronoTick daemon ready - warmup complete!
   🚀 ChronoTick MCP Server ready - accepting agent connections
   ```

## System Service Deployment

### Using systemd (Linux)

1. **Create service file:** `/etc/systemd/system/chronotick-mcp.service`
   ```ini
   [Unit]
   Description=ChronoTick MCP Server
   After=network.target
   Wants=network.target

   [Service]
   Type=simple
   User=chronotick
   Group=chronotick
   WorkingDirectory=/path/to/ChronoTick/tsfm
   Environment=PATH=/path/to/ChronoTick/tsfm/.venv/bin
   ExecStart=/path/to/ChronoTick/tsfm/.venv/bin/python chronotick_mcp.py
   Restart=always
   RestartSec=5
   StandardOutput=journal
   StandardError=journal

   [Install]
   WantedBy=multi-user.target
   ```

2. **Enable and start service:**
   ```bash
   sudo systemctl enable chronotick-mcp.service
   sudo systemctl start chronotick-mcp.service
   sudo systemctl status chronotick-mcp.service
   ```

### Using launchd (macOS)

1. **Create plist file:** `~/Library/LaunchAgents/com.chronotick.mcp.plist`
   ```xml
   <?xml version="1.0" encoding="UTF-8"?>
   <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
   <plist version="1.0">
   <dict>
       <key>Label</key>
       <string>com.chronotick.mcp</string>
       <key>ProgramArguments</key>
       <array>
           <string>/path/to/ChronoTick/tsfm/.venv/bin/python</string>
           <string>/path/to/ChronoTick/tsfm/chronotick_mcp.py</string>
       </array>
       <key>WorkingDirectory</key>
       <string>/path/to/ChronoTick/tsfm</string>
       <key>RunAtLoad</key>
       <true/>
       <key>KeepAlive</key>
       <true/>
       <key>StandardOutPath</key>
       <string>/tmp/chronotick-mcp.log</string>
       <key>StandardErrorPath</key>
       <string>/tmp/chronotick-mcp.error.log</string>
   </dict>
   </plist>
   ```

2. **Load service:**
   ```bash
   launchctl load ~/Library/LaunchAgents/com.chronotick.mcp.plist
   launchctl start com.chronotick.mcp
   ```

## Connecting to Claude Code

### Method 1: MCP Configuration File

1. **Create MCP configuration:** `~/.config/claude-code/mcp.json`
   ```json
   {
     "servers": {
       "chronotick": {
         "command": "python",
         "args": ["/path/to/ChronoTick/tsfm/chronotick_mcp.py"],
         "cwd": "/path/to/ChronoTick/tsfm"
       }
     }
   }
   ```

2. **Restart Claude Code** to load the new MCP server configuration.

### Method 2: Direct Integration

If Claude Code supports direct MCP server connections:

1. **Add server in Claude Code settings:**
   - Server name: `chronotick`
   - Command: `python /path/to/ChronoTick/tsfm/chronotick_mcp.py`
   - Working directory: `/path/to/ChronoTick/tsfm`

### Method 3: Socket-based Connection (Advanced)

For production deployments with multiple clients:

1. **Start server with socket listener:**
   ```bash
   # Start MCP server listening on Unix socket
   uv run python chronotick_mcp.py --socket /tmp/chronotick-mcp.sock
   ```

2. **Configure Claude Code to connect to socket:**
   ```json
   {
     "servers": {
       "chronotick": {
         "transport": "socket",
         "socket": "/tmp/chronotick-mcp.sock"
       }
     }
   }
   ```

## Available Tools

Once connected, Claude Code will have access to these ChronoTick tools:

### 1. `get_time`
Get high-precision corrected time with uncertainty bounds.

**Parameters:**
- `include_stats` (boolean, optional): Include detailed statistics

**Example usage in Claude Code:**
```
Get me the current corrected time with uncertainty bounds.
```

### 2. `get_daemon_status` 
Get ChronoTick daemon status and performance metrics.

**Example usage in Claude Code:**
```
Show me the ChronoTick daemon status and performance metrics.
```

### 3. `get_time_with_future_uncertainty`
Get time with uncertainty projection for a future timestamp.

**Parameters:**
- `future_seconds` (number, required): Seconds in the future (0-3600)

**Example usage in Claude Code:**
```
What will the time uncertainty be in 300 seconds?
```

## Monitoring and Troubleshooting

### Log Monitoring

**View server logs:**
```bash
# If running as systemd service
sudo journalctl -u chronotick-mcp.service -f

# If running as launchd service
tail -f /tmp/chronotick-mcp.log

# If running directly
# Logs will appear in the terminal
```

### Common Issues

1. **Server fails to start:**
   - Check Python path and dependencies: `uv run python --version`
   - Verify configuration file exists and is valid
   - Check network connectivity for NTP access

2. **Warmup takes too long:**
   - Check NTP server connectivity: `ntpdate -q pool.ntp.org`
   - Reduce warmup duration in configuration (not recommended for production)
   - Check firewall settings for UDP port 123

3. **Claude Code can't connect:**
   - Verify MCP configuration file path and syntax
   - Check server is running: `ps aux | grep chronotick_mcp`
   - Verify file permissions on MCP script
   - Restart Claude Code after configuration changes

4. **High latency or timeouts:**
   - Check system load and CPU affinity settings
   - Monitor daemon memory usage
   - Verify fast IPC communication is working

### Performance Monitoring

**Check daemon status programmatically:**
```bash
# Use the daemon status tool (if connected to Claude Code)
# Or check system metrics:
top -p $(pgrep -f chronotick_mcp)
```

**Memory usage:**
```bash
ps -o pid,vsz,rss,comm -p $(pgrep -f chronotick_mcp)
```

## Security Considerations

1. **Network access:** Server requires UDP access to NTP servers (port 123)
2. **File permissions:** Ensure configuration files are not world-readable if they contain sensitive settings
3. **Process isolation:** Consider running as dedicated user account
4. **Resource limits:** Configure appropriate memory and CPU limits for production deployment

## Performance Tuning

1. **CPU Affinity:** Set specific CPU cores for daemon process in configuration
2. **Memory:** Monitor daemon memory usage, typical usage is 50-200MB
3. **NTP Servers:** Use geographically close, reliable NTP servers
4. **Warmup Duration:** Balance accuracy vs startup time (60s recommended)

## Debug Logging

### Enable Comprehensive Debug Tracing

For troubleshooting, enable detailed debug logging:

```bash
# Enable debug tracing for all function calls, model I/O, and IPC
/home/jcernuda/ChronoTick/tsfm/.venv/bin/python chronotick_mcp.py --debug-trace

# Save debug logs to file for analysis
/home/jcernuda/ChronoTick/tsfm/.venv/bin/python chronotick_mcp.py --debug-trace --debug-log-file /tmp/chronotick-debug.log

# Connect to Claude Code with debug logging
claude mcp add chronotick-debug --scope user \
  --env PYTHONPATH=/home/jcernuda/ChronoTick/tsfm \
  -- bash -c "cd /home/jcernuda/ChronoTick/tsfm && /home/jcernuda/ChronoTick/tsfm/.venv/bin/python chronotick_mcp.py --debug-trace --debug-log-file /tmp/chronotick-mcp-debug.log"
```

### Debug Features

- **Function Call Tracing**: Entry/exit logging with timing and arguments
- **Model I/O Logging**: NumPy arrays, predictions, uncertainty calculations
- **IPC Monitoring**: Queue operations, timeouts, process communication
- **Performance Metrics**: Execution times, memory usage, latency tracking

### Log Analysis

```bash
# Monitor debug logs in real-time
tail -f /tmp/chronotick-debug.log

# Find slow operations (>50ms)
grep "execution_time_ms.*[5-9][0-9]\|[0-9][0-9][0-9]" /tmp/chronotick-debug.log

# Track model predictions
grep "PIPELINE_EXIT.*CorrectionWithBounds" /tmp/chronotick-debug.log

# Monitor IPC performance
grep "get_correction_from_daemon" /tmp/chronotick-debug.log
```

See `DEBUG_LOGGING_GUIDE.md` for comprehensive debugging documentation.

## Support

For issues and questions:
- Check logs first for error messages
- Enable debug tracing: `--debug-trace --debug-log-file /tmp/debug.log`
- Verify all prerequisites are installed
- Test daemon functionality: `uv run python chronotick_inference/daemon.py test`
- Check MCP server tests: `uv run python -m pytest tests/chronotick/test_mcp_server.py -v`
- Check debug logging tests: `uv run python -m pytest tests/chronotick/test_debug_logging.py -v`