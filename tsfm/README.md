# ChronoTick MCP Server

**High-precision time services for AI agents via Model Context Protocol**

[![Tests](https://github.com/JaimeCernuda/chronotick-mcp/workflows/tests/badge.svg)](https://github.com/JaimeCernuda/chronotick-mcp/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

ChronoTick MCP Server provides microsecond-precision time corrections to AI agents through the Model Context Protocol (MCP). It combines advanced time synchronization, machine learning inference, and fast IPC communication to deliver real-time clock corrections with uncertainty quantification.

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or later
- uv package manager
- Network access for NTP synchronization
- Linux/macOS (Windows support may require modifications)

### Installation
```bash
git clone git@github.com:JaimeCernuda/chronotick-mcp.git
cd chronotick-mcp
uv sync
```

### Start the MCP Server
```bash
# Basic start
uv run python chronotick_mcp.py

# With debug logging
uv run python chronotick_mcp.py --debug-trace --debug-log-file /tmp/chronotick-debug.log
```

### Connect to Claude Code
```bash
claude mcp add chronotick --scope user \
  --env PYTHONPATH=$(pwd) \
  -- bash -c "cd $(pwd) && $(pwd)/.venv/bin/python chronotick_mcp.py"
```

## 🎯 Features

### Core Capabilities
- **Microsecond Precision**: Real-time clock corrections with uncertainty bounds
- **Fast IPC Communication**: Optimized multiprocessing queues for minimal latency
- **Machine Learning Integration**: TimesFM + Chronos models for predictive time corrections
- **Comprehensive Debug Logging**: Function tracing, model I/O, and performance metrics

### MCP Tools Available to AI Agents
1. **`get_time`** - Get corrected time with uncertainty bounds
2. **`get_daemon_status`** - Monitor daemon health and performance
3. **`get_time_with_future_uncertainty`** - Project time uncertainty into the future

### Advanced Features
- **Quality Control**: Rejects poor NTP measurements (configurable thresholds)
- **Error Propagation**: Proper uncertainty calculation and bounds
- **Service Deployment**: systemd/launchd configurations included
- **Comprehensive Testing**: 25+ tests covering all functionality

## 📊 Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   AI Agents     │◄──►│  ChronoTick MCP  │◄──►│ ChronoTick      │
│  (Claude Code)  │    │     Server       │    │    Daemon       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │                          │
                              ▼                          ▼
                       ┌──────────────┐         ┌──────────────┐
                       │ Fast IPC     │         │ ML Pipeline  │
                       │ (Queues)     │         │ (TimesFM +   │
                       └──────────────┘         │  Chronos)    │
                                                └──────────────┘
                                                         │
                                                         ▼
                                                ┌──────────────┐
                                                │ NTP Client   │
                                                │ (4 servers)  │
                                                └──────────────┘
```

## 🔧 Configuration

The system uses YAML configuration files with comprehensive settings:

```yaml
clock_measurement:
  ntp:
    servers: ['pool.ntp.org', 'time.google.com', 'time.cloudflare.com', 'time.nist.gov']
    timeout_seconds: 2.0
    max_acceptable_uncertainty: 0.010  # 10ms threshold
  timing:
    warm_up:
      duration_seconds: 180  # 3-minute warmup
      measurement_interval: 1.0
```

## 📁 Project Structure

```
chronotick-mcp/
├── chronotick_inference/          # Core inference engine
│   ├── mcp_server.py              # MCP server implementation
│   ├── daemon.py                  # Background daemon process
│   ├── real_data_pipeline.py      # ML pipeline integration
│   └── configs/                   # Configuration files
├── chronotick_mcp.py              # Main entry point
├── tests/                         # Comprehensive test suite
│   └── chronotick/
│       ├── test_mcp_server.py     # MCP functionality tests
│       └── test_debug_logging.py  # Debug logging tests
├── docs/                          # Documentation
│   ├── MCP_DEPLOYMENT_GUIDE.md    # Deployment instructions
│   ├── DEBUG_LOGGING_GUIDE.md     # Debug logging guide
│   └── ...                       # Additional guides
└── chronotick-5min-debug.log      # Example debug output
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# All tests
uv run python -m pytest tests/ -v

# MCP server tests only
uv run python -m pytest tests/chronotick/test_mcp_server.py -v

# Debug logging tests
uv run python -m pytest tests/chronotick/test_debug_logging.py -v
```

**Test Coverage**: 25/25 tests passing
- 17 MCP server functionality tests
- 8 debug logging system tests

## 🔍 Debug Logging

Enable comprehensive debug tracing for troubleshooting:

```bash
# Full debug tracing
uv run python chronotick_mcp.py --debug-trace

# Save debug logs to file
uv run python chronotick_mcp.py --debug-trace --debug-log-file /tmp/debug.log

# Monitor debug output
tail -f /tmp/debug.log
```

Debug features include:
- Function entry/exit with timing
- Model I/O logging (NumPy arrays, predictions)
- IPC communication monitoring
- Performance metrics and latency tracking

## 🚀 Deployment

### Development
```bash
uv run python chronotick_mcp.py
```

### Production (systemd)
```bash
sudo cp scripts/chronotick-mcp.service /etc/systemd/system/
sudo systemctl enable chronotick-mcp.service
sudo systemctl start chronotick-mcp.service
```

### Production (launchd - macOS)
```bash
cp scripts/com.chronotick.mcp.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.chronotick.mcp.plist
```

## 📈 Performance

- **Typical Response Time**: <1ms for time requests
- **Warmup Period**: 3 minutes for quality NTP measurements  
- **Memory Usage**: 50-200MB depending on model loading
- **CPU Usage**: Low during normal operation, higher during ML inference

## 🛡️ Quality Assurance

The system implements strict quality controls:

- **NTP Quality Filtering**: Rejects measurements >10ms uncertainty by default
- **Error Propagation**: Proper uncertainty bounds through all calculations  
- **Conservative Fallbacks**: No predictions with insufficient data quality
- **Comprehensive Testing**: All edge cases and error conditions covered

## 📋 Requirements

### Runtime Dependencies
- `mcp>=1.0.0` - Model Context Protocol implementation
- `pydantic>=2.0.0` - Data validation and serialization
- `pydantic-settings` - Configuration management
- `numpy` - Numerical computations
- `PyYAML` - Configuration file parsing

### Development Dependencies
- `pytest>=7.0.0` - Testing framework
- `pytest-asyncio>=0.23.0` - Async testing support
- `pytest-mock` - Mock testing utilities

## 🤝 Usage with AI Agents

Once connected to Claude Code, you can request:

```
"Get me the current corrected time with uncertainty bounds"
"Show me the ChronoTick daemon status and performance metrics"  
"What will the time uncertainty be in 5 minutes?"
```

The MCP server will provide high-precision responses with detailed uncertainty quantification.

## 🔗 Related Documentation

- **[MCP Deployment Guide](docs/MCP_DEPLOYMENT_GUIDE.md)** - Complete setup instructions
- **[Debug Logging Guide](docs/DEBUG_LOGGING_GUIDE.md)** - Troubleshooting and monitoring
- **[Comprehensive Usage Guide](docs/COMPREHENSIVE_USAGE_GUIDE.md)** - Advanced usage patterns

## 🐛 Troubleshooting

### Common Issues

**Server fails to start:**
- Check Python version: `python --version` (requires 3.10+)
- Verify dependencies: `uv sync`
- Check network connectivity for NTP access

**High latency/poor performance:**
- Enable debug logging: `--debug-trace`
- Check system resources: `top -p $(pgrep chronotick)`
- Verify NTP server connectivity: `ntpdate -q pool.ntp.org`

**Claude Code connection issues:**
- Verify MCP configuration syntax
- Check file permissions on scripts
- Restart Claude Code after configuration changes

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on the Model Context Protocol (MCP) specification
- Integrates TimesFM and Chronos time series models
- Uses high-precision NTP synchronization techniques

---

**Need help?** Check the [documentation](docs/) or open an issue on GitHub.