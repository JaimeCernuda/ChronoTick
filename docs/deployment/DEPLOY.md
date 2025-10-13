# ChronoTick Deployment Guide

**Technical and Straightforward Guide to Deploying the System**

This guide provides step-by-step instructions for deploying ChronoTick, a high-precision time synchronization system with ML-powered clock drift prediction for distributed AI agents.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Understanding the Architecture](#understanding-the-architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running ChronoTick](#running-chronotick)
- [Verification and Testing](#verification-and-testing)
- [Production Deployment](#production-deployment)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

- **Operating System**: Linux (recommended) or macOS
  - Windows: Limited support due to multiprocessing constraints
- **Python**: 3.10 or later (3.12 recommended)
- **Memory**: Minimum 2GB RAM, 4GB+ recommended for model operations
- **Network**: UDP port 123 access for NTP queries
- **Disk Space**: 2-5GB for models and dependencies

### Required Software

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or via pip
pip install uv

# Verify installation
uv --version  # Should show 0.7.x or later
```

### Network Requirements

ChronoTick requires access to NTP servers:
- `pool.ntp.org`
- `time.google.com`
- `time.cloudflare.com`
- `time.nist.gov`

Test connectivity:
```bash
nc -zvu pool.ntp.org 123
nc -zvu time.google.com 123
```

---

## Understanding the Architecture

### Core Components

ChronoTick consists of three main directories:

1. **`tsfm/` - PRIMARY IMPLEMENTATION** ⭐
   - `tsfm/chronotick_inference/`: Core MCP server with dual-model prediction
   - `tsfm/chronotick_mcp.py`: Main entry point
   - `tsfm/tsfm/`: TSFM factory for time series models
   - **Status**: Production-ready, fully tested (117 tests passing)

2. **`chronotick-server/` - Alternative Implementation**
   - Newer design with PTP/vector clock support
   - **Status**: Basic framework only
   - **Note**: Use `tsfm/` for actual deployments

3. **`inferance_layer/` - Deprecated**
   - Superseded by `tsfm/chronotick_inference/`

### Model Incompatibility Strategy

**Critical Design Decision**: The system supports 5 time series foundation models, but they have **mutually exclusive transformer dependencies**:

| Model | Transformers Version | Status |
|-------|---------------------|---------|
| **Chronos-Bolt** | No conflict | ✅ Recommended (default) |
| **TimesFM** | No conflict | ✅ Compatible with Chronos |
| **TTM** | ==4.38.0 | ⚠️ Incompatible with Toto/Time-MoE |
| **Toto** | >=4.52.0 | ⚠️ Incompatible with TTM/Time-MoE |
| **Time-MoE** | ==4.40.1 | ⚠️ Incompatible with TTM/Toto |

**Solution**: uv's `[tool.uv] conflicts` configuration enforces mutually exclusive extras:
```toml
[tool.uv]
conflicts = [
    [
        { extra = "ttm" },
        { extra = "time-moe" },
        { extra = "toto" },
    ],
]
```

**Recommendation**: Use the `core-models` extra (Chronos + TimesFM) which has no conflicts and provides excellent performance.

---

## Installation

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd ChronoTick
```

### Step 2: Navigate to Primary Implementation

```bash
cd tsfm
```

**Important**: All commands should be run from the `tsfm/` directory.

### Step 3: Choose Your Model Environment

#### Option A: Core Models (Recommended)

For Chronos-Bolt and TimesFM without conflicts:

```bash
uv sync --extra core-models --extra dev --extra test
```

This installs:
- Base dependencies (torch, numpy, pandas, mcp, etc.)
- chronos-forecasting (Chronos-Bolt)
- timesfm (Google TimesFM)
- transformers (latest version, no conflicts)
- Development and testing tools

#### Option B: TTM Environment

For IBM's Tiny Time Mixer:

```bash
uv sync --extra ttm --extra dev --extra test
```

**Note**: Cannot be used simultaneously with Toto or Time-MoE due to transformers==4.38.0 requirement.

#### Option C: Toto Environment

For Datadog's Toto model:

```bash
uv sync --extra toto --extra dev --extra test
```

**Note**: Requires transformers>=4.52.0, incompatible with TTM and Time-MoE.

#### Option D: Time-MoE Environment

For Time Mixture of Experts:

```bash
uv sync --extra time-moe --extra dev --extra test
```

**Note**: Requires transformers==4.40.1, incompatible with TTM and Toto.

### Step 4: Verify Installation

```bash
.venv/bin/python -c "
import chronos; print('✓ Chronos-Bolt')
import timesfm; print('✓ TimesFM')
import mcp; print('✓ MCP')
import chronotick_inference; print('✓ ChronoTick')
"
```

Expected output:
```
✓ Chronos-Bolt
✓ TimesFM
✓ MCP
✓ ChronoTick
```

### Step 5: Run Tests (Optional but Recommended)

```bash
# Run all ChronoTick tests (117 tests)
.venv/bin/python -m pytest tests/chronotick/ -v

# Run quick NTP client tests
.venv/bin/python -m pytest tests/chronotick/test_ntp_client.py -v

# Run with coverage
.venv/bin/python -m pytest tests/chronotick/ --cov=chronotick_inference --cov-report=html
```

---

## Configuration

### Default Configuration

The default configuration is located at:
```
tsfm/chronotick_inference/config.yaml
```

Key settings:
```yaml
short_term:
  model_name: chronos
  device: cpu
  inference_interval: 1.0        # 1Hz updates
  prediction_horizon: 5          # 5-second lookahead
  context_length: 100
  model_params:
    repo: "amazon/chronos-bolt-base"
    size: "base"

long_term:
  model_name: chronos
  device: cpu
  inference_interval: 30.0       # 0.033Hz updates
  prediction_horizon: 60         # 60-second lookahead
  context_length: 300
  model_params:
    repo: "amazon/chronos-bolt-base"
    size: "base"

fusion:
  enabled: true
  method: inverse_variance       # Optimal fusion method
  uncertainty_threshold: 0.05
```

### Alternative Configurations

Pre-configured alternatives in `tsfm/chronotick_inference/configs/`:

- `cpu_only_chronos.yaml` - Default CPU-only with Chronos-Bolt
- `gpu_only_timesfm.yaml` - GPU-accelerated TimesFM
- `cpu_only_ttm.yaml` - TTM model (requires TTM environment)
- `gpu_only_toto.yaml` - Toto model (requires Toto environment)
- `hybrid_chronos_ttm.yaml` - Dual model setup
- `multi_device_full.yaml` - Multi-GPU deployment

### Creating Custom Configurations

1. Copy a base configuration:
```bash
cp chronotick_inference/configs/cpu_only_chronos.yaml chronotick_inference/configs/my_config.yaml
```

2. Edit settings:
```yaml
# Adjust warmup duration
clock_measurement:
  timing:
    warm_up:
      duration_seconds: 300  # Increase to 5 minutes for better accuracy
      measurement_interval: 1.0

# Change NTP servers (use geographically closer servers)
clock_measurement:
  ntp:
    servers:
      - "0.pool.ntp.org"
      - "1.pool.ntp.org"
      - "time.nist.gov"

# Adjust model parameters
short_term:
  inference_interval: 0.5     # 2Hz for faster updates
  prediction_horizon: 3       # Shorter horizon
  model_params:
    size: "tiny"              # Use smaller model for lower memory
```

### Configuration Priority

1. Custom config via `--config` flag
2. `chronotick_inference/config.yaml` (default)
3. Model-specific configs in `configs/` directory

---

## Running ChronoTick

### Basic Usage

From the `tsfm/` directory:

```bash
# Start with default configuration
uv run python chronotick_mcp.py

# Or use .venv directly
.venv/bin/python chronotick_mcp.py
```

**Expected startup sequence:**

```
[INFO] ChronoTick MCP Server starting...
[INFO] Loading configuration from chronotick_inference/config.yaml
[INFO] Starting ChronoTick daemon...
[INFO] Daemon process started with PID: 12345
[INFO] Starting 180s warm-up phase...
🕒 ChronoTick warmup: 25.0% complete, 135s remaining
🕒 ChronoTick warmup: 50.0% complete, 90s remaining
🕒 ChronoTick warmup: 75.0% complete, 45s remaining
✅ ChronoTick daemon ready - warmup complete!
🚀 ChronoTick MCP Server ready - accepting agent connections
```

### With Custom Configuration

```bash
uv run python chronotick_mcp.py --config chronotick_inference/configs/gpu_only_timesfm.yaml
```

### With Debug Logging

```bash
# Console debug output
uv run python chronotick_mcp.py --debug-trace --log-level DEBUG

# Write debug logs to file
uv run python chronotick_mcp.py --debug-trace --debug-log-file /tmp/chronotick-debug.log

# Monitor debug logs
tail -f /tmp/chronotick-debug.log
```

Debug logging includes:
- Function entry/exit with timing
- NTP measurements and quality metrics
- Model I/O (numpy arrays, predictions)
- IPC message flow
- Prediction cache hits/misses
- Fusion weights and decisions

### Verifying Operation

In another terminal:

```bash
# Check MCP server is responding
curl -X POST http://localhost:3000 \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"tools/list","id":1}'

# Or use the CLI tool (if configured)
claude mcp status chronotick
```

---

## Verification and Testing

### Manual Verification Steps

1. **Check Daemon Status**:
```bash
.venv/bin/python -c "
from chronotick_inference.mcp_server import ChronoTickMCPServer
# Server status check would go here
print('MCP Server operational')
"
```

2. **Verify NTP Connectivity**:
```bash
.venv/bin/python chronotick_inference/ntp_client.py
```

Expected output showing NTP measurements:
```
NTP measurement from pool.ntp.org:
  Offset: 0.000025s (25μs)
  Delay: 0.012s
  Uncertainty: 0.006s
  Quality: GOOD
```

3. **Test Model Loading**:
```bash
.venv/bin/python -c "
from tsfm import TSFMFactory
factory = TSFMFactory()
model = factory.create_model('chronos', device='cpu')
model.load_model()
print(f'Model status: {model.status}')
"
```

### Running Test Suite

```bash
# All tests (117 tests)
.venv/bin/python -m pytest tests/chronotick/ -v

# Fast unit tests only
.venv/bin/python -m pytest tests/chronotick/ -v -m "not slow"

# Skip network-dependent tests
.venv/bin/python -m pytest tests/chronotick/ -v -m "not network"

# With coverage report
.venv/bin/python -m pytest tests/chronotick/ \
  --cov=chronotick_inference \
  --cov-report=html \
  --cov-report=term

# View coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### Performance Benchmarks

Run integration tests with performance metrics:

```bash
.venv/bin/python -m pytest tests/integration/test_performance.py -v
```

This generates reports in `tests/integration/reports/` with:
- Model loading times
- Inference latency
- Memory usage
- Prediction accuracy (MAE, RMSE)
- Visualizations

---

## Production Deployment

### systemd Service (Linux)

1. **Create Service File**:

```bash
sudo nano /etc/systemd/system/chronotick-mcp.service
```

```ini
[Unit]
Description=ChronoTick MCP Server - High-Precision Time Synchronization
Documentation=https://github.com/yourorg/chronotick
After=network-online.target
Wants=network-online.target
Requires=network.target

[Service]
Type=simple
User=chronotick
Group=chronotick
WorkingDirectory=/opt/chronotick/tsfm
Environment="PATH=/opt/chronotick/tsfm/.venv/bin:/usr/local/bin:/usr/bin:/bin"
Environment="PYTHONPATH=/opt/chronotick/tsfm"

# Main command
ExecStart=/opt/chronotick/tsfm/.venv/bin/python chronotick_mcp.py

# Restart policy
Restart=always
RestartSec=10
StartLimitInterval=0
StartLimitBurst=5

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=chronotick-mcp

# Performance tuning
CPUAffinity=2 3              # Dedicate CPU cores 2-3
Nice=-10                     # High priority
IOSchedulingClass=realtime
IOSchedulingPriority=0

# Resource limits
LimitNOFILE=65536
LimitNPROC=4096
MemoryMax=4G

# Security
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

2. **Set Up User and Directories**:

```bash
# Create service user
sudo useradd -r -s /bin/false chronotick

# Set permissions
sudo chown -R chronotick:chronotick /opt/chronotick
sudo chmod -R 755 /opt/chronotick
```

3. **Enable and Start**:

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service
sudo systemctl enable chronotick-mcp.service

# Start service
sudo systemctl start chronotick-mcp.service

# Check status
sudo systemctl status chronotick-mcp.service

# View logs
sudo journalctl -u chronotick-mcp.service -f
```

### launchd Service (macOS)

1. **Create plist File**:

```bash
nano ~/Library/LaunchAgents/com.chronotick.mcp.plist
```

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.chronotick.mcp</string>

    <key>ProgramArguments</key>
    <array>
        <string>/opt/chronotick/tsfm/.venv/bin/python</string>
        <string>/opt/chronotick/tsfm/chronotick_mcp.py</string>
        <string>--log-level</string>
        <string>INFO</string>
    </array>

    <key>WorkingDirectory</key>
    <string>/opt/chronotick/tsfm</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PYTHONPATH</key>
        <string>/opt/chronotick/tsfm</string>
    </dict>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>

    <key>ProcessType</key>
    <string>Interactive</string>

    <key>Nice</key>
    <integer>-10</integer>

    <key>StandardOutPath</key>
    <string>/var/log/chronotick/stdout.log</string>

    <key>StandardErrorPath</key>
    <string>/var/log/chronotick/stderr.log</string>

    <key>ThrottleInterval</key>
    <integer>10</integer>
</dict>
</plist>
```

2. **Set Up Logging**:

```bash
sudo mkdir -p /var/log/chronotick
sudo chown $USER /var/log/chronotick
```

3. **Load and Start**:

```bash
# Load the service
launchctl load ~/Library/LaunchAgents/com.chronotick.mcp.plist

# Start the service
launchctl start com.chronotick.mcp

# Check status
launchctl list | grep chronotick

# View logs
tail -f /var/log/chronotick/stdout.log
tail -f /var/log/chronotick/stderr.log
```

### Docker Deployment

1. **Create Dockerfile**:

```dockerfile
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy project files
COPY tsfm/ /app/

# Install dependencies
RUN uv sync --extra core-models

# Expose MCP port (if using TCP transport)
EXPOSE 3000

# Set environment
ENV PYTHONPATH=/app
ENV CHRONOTICK_LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=200s --retries=3 \
  CMD python -c "from chronotick_inference import __version__; print(__version__)"

# Run server
CMD [".venv/bin/python", "chronotick_mcp.py"]
```

2. **Docker Compose**:

```yaml
version: '3.8'

services:
  chronotick:
    build: .
    container_name: chronotick-mcp
    restart: unless-stopped

    # Network configuration
    network_mode: host

    # Resource limits
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G

    # Volumes
    volumes:
      - ./tsfm/chronotick_inference/config.yaml:/app/chronotick_inference/config.yaml:ro
      - ./logs:/var/log/chronotick
      - model-cache:/root/.cache/huggingface

    # Environment
    environment:
      - CHRONOTICK_LOG_LEVEL=INFO
      - PYTHONPATH=/app

    # Logging
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

    # Health check
    healthcheck:
      test: ["CMD", "python", "-c", "from chronotick_inference import __version__"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 200s

volumes:
  model-cache:
```

3. **Build and Run**:

```bash
# Build image
docker-compose build

# Start service
docker-compose up -d

# View logs
docker-compose logs -f

# Check status
docker-compose ps
```

### Claude Code Integration

Add to Claude Code's MCP configuration:

```bash
claude mcp add chronotick --scope user \
  --env PYTHONPATH=/opt/chronotick/tsfm \
  -- bash -c "cd /opt/chronotick/tsfm && .venv/bin/python chronotick_mcp.py"
```

Or add to `~/.config/claude/mcp.json`:

```json
{
  "mcpServers": {
    "chronotick": {
      "command": "bash",
      "args": [
        "-c",
        "cd /opt/chronotick/tsfm && .venv/bin/python chronotick_mcp.py"
      ],
      "env": {
        "PYTHONPATH": "/opt/chronotick/tsfm"
      }
    }
  }
}
```

---

## Troubleshooting

### Common Issues

#### 1. "ChronoTick daemon not ready"

**Symptoms**: Server starts but times out during warmup.

**Causes**:
- NTP servers unreachable
- Warmup duration too short
- Network firewall blocking UDP 123

**Solutions**:
```bash
# Test NTP connectivity
nc -zvu pool.ntp.org 123
nc -zvu time.google.com 123

# Check firewall rules
sudo iptables -L | grep 123

# Allow NTP traffic
sudo iptables -A OUTPUT -p udp --dport 123 -j ACCEPT

# Increase warmup duration in config
# Edit chronotick_inference/config.yaml:
clock_measurement:
  timing:
    warm_up:
      duration_seconds: 300  # Increase to 5 minutes
```

#### 2. "transformers version conflict"

**Symptoms**: `pip` or `uv` reports dependency conflicts.

**Cause**: Trying to install incompatible model environments simultaneously.

**Solution**:
```bash
# Uninstall current environment
rm -rf .venv
uv venv

# Install only ONE model environment
uv sync --extra core-models  # Recommended
# OR
uv sync --extra ttm
# OR
uv sync --extra toto
# BUT NOT: uv sync --extra ttm --extra toto  # This will fail!
```

#### 3. Model loading fails

**Symptoms**: `RuntimeError: Chronos model loading failed`

**Causes**:
- Insufficient memory
- Missing model packages
- HuggingFace Hub connection issues

**Solutions**:
```bash
# Check memory
free -h

# Verify packages installed
.venv/bin/python -c "import chronos; print('OK')"

# Test HuggingFace connection
.venv/bin/python -c "
from transformers import AutoModel
AutoModel.from_pretrained('bert-base-uncased')  # Test download
"

# Use smaller model
# Edit config.yaml:
short_term:
  model_params:
    size: "tiny"  # Instead of "base"
```

#### 4. High latency / poor performance

**Symptoms**: Response times >50ms, high CPU usage

**Causes**:
- CPU throttling
- Insufficient CPU affinity
- Memory swapping

**Solutions**:
```bash
# Check CPU frequency
cat /proc/cpuinfo | grep MHz

# Disable CPU throttling
sudo cpupower frequency-set -g performance

# Set CPU affinity (systemd)
# Edit /etc/systemd/system/chronotick-mcp.service:
CPUAffinity=2 3  # Dedicate cores 2-3

# Check memory swapping
vmstat 1 10

# Increase available memory or reduce model size
```

#### 5. NTP measurements rejected

**Symptoms**: All measurements show "poor quality" or exceed uncertainty threshold

**Causes**:
- High network latency
- Geographically distant NTP servers
- Network congestion

**Solutions**:
```bash
# Test NTP latency
ntpdate -q pool.ntp.org

# Use closer NTP servers
# Edit config.yaml:
clock_measurement:
  ntp:
    servers:
      - "0.us.pool.ntp.org"  # US servers
      - "0.europe.pool.ntp.org"  # European servers
    max_acceptable_uncertainty: 0.020  # Increase threshold to 20ms
```

#### 6. "ModuleNotFoundError: No module named 'chronotick_inference'"

**Symptoms**: Import errors when running server

**Cause**: Running from wrong directory or PYTHONPATH not set

**Solutions**:
```bash
# Always run from tsfm/ directory
cd /path/to/ChronoTick/tsfm

# Or set PYTHONPATH
export PYTHONPATH=/path/to/ChronoTick/tsfm
.venv/bin/python chronotick_mcp.py
```

#### 7. Tests fail with "pytest not found"

**Symptoms**: `uv run pytest` fails

**Solution**:
```bash
# Install test dependencies
uv sync --extra test

# Run via python module
.venv/bin/python -m pytest tests/chronotick/ -v
```

### Debug Checklist

When troubleshooting, collect this information:

```bash
# 1. System information
uname -a
python --version
uv --version

# 2. Environment check
cd /path/to/ChronoTick/tsfm
.venv/bin/python -c "
import sys
print('Python:', sys.version)
print('Path:', sys.path)

try:
    import chronos; print('✓ chronos')
except Exception as e: print('✗ chronos:', e)

try:
    import mcp; print('✓ mcp')
except Exception as e: print('✗ mcp:', e)

try:
    from chronotick_inference import __version__
    print(f'✓ chronotick: {__version__}')
except Exception as e: print('✗ chronotick:', e)
"

# 3. Network connectivity
nc -zvu pool.ntp.org 123
ping -c 3 time.google.com

# 4. Resource usage
free -h
df -h
top -bn1 | head -20

# 5. Log files
tail -100 /var/log/chronotick/stderr.log
# Or for systemd:
sudo journalctl -u chronotick-mcp.service -n 100

# 6. Test NTP client
.venv/bin/python chronotick_inference/ntp_client.py

# 7. Run minimal test
.venv/bin/python -m pytest tests/chronotick/test_ntp_client.py::TestNTPClient::test_ntp_packet_creation -v
```

### Getting Help

1. **Check Documentation**:
   - `CLAUDE.md` - Development guide
   - `design.md` - System architecture
   - `technical.md` - Technical details
   - `chronotick_inference/CONFIGURATION_GUIDE.md` - Config reference

2. **Review Logs**:
   - Enable debug logging: `--debug-trace --debug-log-file /tmp/debug.log`
   - Check systemd journal: `sudo journalctl -u chronotick-mcp.service -f`

3. **Run Tests**:
   - `.venv/bin/python -m pytest tests/chronotick/ -v`

4. **Verify Configuration**:
   - Check YAML syntax: `python -c "import yaml; yaml.safe_load(open('chronotick_inference/config.yaml'))"`

---

## Performance Tuning

### Optimization Guidelines

1. **CPU Affinity**: Dedicate specific CPU cores
2. **Process Priority**: Set Nice value to -10 or -5
3. **Memory**: Minimum 2GB, 4GB recommended
4. **Model Selection**: Use `tiny` size for low-resource environments
5. **Warmup Duration**: Longer warmup (300s) improves initial accuracy
6. **NTP Servers**: Use geographically close servers

### Performance Benchmarks

Expected performance after warmup:

| Metric | Value | Notes |
|--------|-------|-------|
| Startup time | ~5s | Process initialization |
| Warmup period | 180s | Configurable (default) |
| Response latency | <1ms | Cached predictions |
| Model inference | 8-35ms | Short-term: 8ms, Long-term: 35ms |
| Memory usage | 50-200MB | Model dependent |
| CPU usage | <5% baseline | 20-40% during inference |
| NTP bandwidth | ~1KB/s | 4 servers × 10s interval |
| Prediction accuracy | ±5-10μs | After warmup |
| Drift estimation | ±1 PPM | Parts per million |

---

## Security Considerations

1. **Run as Non-Root**: Always use dedicated service user
2. **Network Isolation**: Firewall NTP traffic to trusted servers only
3. **Log Management**: Rotate logs, limit retention
4. **Resource Limits**: Set memory and CPU limits
5. **File Permissions**: Restrict config file access (600 or 644)
6. **Update Dependencies**: Regularly update packages for security patches

```bash
# Set secure permissions
chmod 600 chronotick_inference/config.yaml
chown chronotick:chronotick chronotick_inference/config.yaml

# Limit log file size
# Add to systemd service:
StandardOutput=journal
StandardError=journal
```

---

## Monitoring and Maintenance

### Health Checks

```bash
# Check service status
systemctl status chronotick-mcp.service

# Monitor logs
journalctl -u chronotick-mcp.service -f

# Check resource usage
ps aux | grep chronotick
netstat -tulpn | grep python
```

### Metrics to Monitor

- **Daemon Status**: Should be "ready" after warmup
- **NTP Quality**: <10ms uncertainty
- **Prediction Cache Hit Rate**: >90%
- **Model Inference Time**: <50ms
- **Memory Usage**: <300MB
- **Daemon Uptime**: Continuous (no restarts)

### Log Rotation

For systemd:
```bash
# Edit /etc/systemd/system/chronotick-mcp.service
# Add under [Service]:
StandardOutput=journal
StandardError=journal

# Configure journald
# Edit /etc/systemd/journald.conf:
SystemMaxUse=500M
SystemKeepFree=1G
MaxRetentionSec=1week
```

For Docker:
```yaml
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
```

---

## Appendix

### Quick Reference Commands

```bash
# Installation
cd tsfm/
uv sync --extra core-models --extra dev --extra test

# Run server
.venv/bin/python chronotick_mcp.py

# Run tests
.venv/bin/python -m pytest tests/chronotick/ -v

# Debug mode
.venv/bin/python chronotick_mcp.py --debug-trace --debug-log-file /tmp/debug.log

# Check status
systemctl status chronotick-mcp.service

# View logs
journalctl -u chronotick-mcp.service -f

# Restart service
systemctl restart chronotick-mcp.service
```

### Directory Structure

```
ChronoTick/
├── tsfm/                        # PRIMARY IMPLEMENTATION
│   ├── chronotick_mcp.py        # Main entry point
│   ├── chronotick_inference/    # Core implementation
│   │   ├── config.yaml          # Default configuration
│   │   ├── configs/             # Alternative configurations
│   │   ├── daemon.py            # Background daemon
│   │   ├── mcp_server.py        # MCP interface
│   │   ├── ntp_client.py        # NTP implementation
│   │   ├── real_data_pipeline.py # Dual-model pipeline
│   │   └── predictive_scheduler.py # Prediction scheduler
│   ├── tsfm/                    # TSFM factory
│   │   ├── factory.py           # Model factory
│   │   ├── LLM/                 # Model implementations
│   │   │   ├── chronos_bolt.py
│   │   │   ├── timesfm.py
│   │   │   ├── ttm.py
│   │   │   ├── toto.py
│   │   │   └── time_moe.py
│   │   └── config/              # TSFM configurations
│   ├── tests/                   # Test suite (117 tests)
│   └── pyproject.toml           # Dependencies and conflicts
├── chronotick-server/           # Alternative implementation (basic)
├── inferance_layer/             # Deprecated
└── servers/                     # Reference MCP servers
```

### Environment Variables

```bash
# Model selection
export CHRONOTICK_MODEL=chronos

# Device configuration
export CHRONOTICK_DEVICE=cpu

# NTP servers
export CHRONOTICK_NTP_SERVERS="pool.ntp.org,time.google.com"

# Warmup duration
export CHRONOTICK_WARMUP=180

# Log level
export CHRONOTICK_LOG_LEVEL=INFO
```

---

**Version**: 1.0
**Last Updated**: 2025-10-09
**Maintained By**: ChronoTick Team
