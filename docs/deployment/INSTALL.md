# ChronoTick Installation Guide

Quick guide for installing and running ChronoTick on multinode deployments.

## Installation with uv

```bash
# Clone repository
git clone https://github.com/JaimeCernuda/ChronoTick.git
cd ChronoTick/tsfm

# Install ChronoTick with uv
uv sync --extra chronotick

# Verify installation
uv run chronotick-mcp --help
```

## Available Entry Points

ChronoTick provides three command-line tools:

### 1. `chronotick-mcp` - MCP Server for AI Agents
Run the Model Context Protocol server for AI agent integration:
```bash
uv run chronotick-mcp
uv run chronotick-mcp --config path/to/config.yaml
uv run chronotick-mcp --log-level DEBUG
uv run chronotick-mcp --debug-trace --debug-log-file /tmp/debug.log
```

### 2. `chronotick-daemon` - Standalone Daemon
Run ChronoTick daemon standalone (for multinode deployments):
```bash
uv run chronotick-daemon
uv run chronotick-daemon --config chronotick_inference/config_enhanced_features.yaml
```

### 3. `chronotick-config` - Configuration Selector
Interactive configuration selector tool:
```bash
uv run chronotick-config
```

## Multinode Deployment

For distributed systems, deploy ChronoTick daemon on each node:

```bash
# On each node:
cd /path/to/ChronoTick/tsfm
uv sync --extra chronotick

# Start daemon with production config
uv run chronotick-daemon --config chronotick_inference/config_enhanced_features.yaml
```

## Configuration Files

Production-ready configurations:

- **`config_enhanced_features.yaml`** - Recommended for production
  - Enhanced NTP (2-3 samples/server, 100ms spacing)
  - Backtracking correction with REPLACE strategy
  - TimesFM quantiles for uncertainty

- **`config_complete.yaml`** - Full feature set
- **`config_short_only.yaml`** - Minimal configuration

## Dependencies

### Core Dependencies
Automatically installed with `--extra chronotick`:
- TimesFM 1.3.0+ (time series model)
- JAX/JAXlib (TimesFM backend)
- NTPlib (NTP client)
- MCP SDK (Model Context Protocol)

### Optional Model Environments
Due to transformers version conflicts, use separate environments for different models:

```bash
# TTM models
uv sync --extra ttm

# Time-MoE models
uv sync --extra time-moe

# Toto models
uv sync --extra toto
```

## Quick Start

### Test Installation
```bash
# Run 5-minute test
cd tsfm
uv run python scripts/test_with_visualization_data.py backtracking \
  --config chronotick_inference/config_enhanced_features.yaml \
  --duration 300 \
  --interval 10
```

### Production Deployment
```bash
# Start MCP server (for AI agents)
uv run chronotick-mcp --config chronotick_inference/config_enhanced_features.yaml

# Or run daemon standalone (for manual integration)
uv run chronotick-daemon --config chronotick_inference/config_enhanced_features.yaml
```

## Performance Expectations

Based on 8-hour overnight test results:

- **Error Reduction**: 75.5% vs system clock
- **Mean Absolute Error**: 14.38ms (vs 58.59ms system clock)
- **Temporal Resolution**: 10-second updates (18x better than NTP 180s)
- **Clock Drift Handling**: 3.03 PPM linear drift
- **Warmup Time**: ~3 minutes for quality NTP measurements
- **Response Latency**: <1ms for cached corrections

## Troubleshooting

### Entry Points Not Found
If `chronotick-mcp` command not found:
```bash
# Reinstall package
uv sync --extra chronotick
```

### Import Errors
Check you're using the correct model environment:
```bash
# For production use (TimesFM)
uv sync --extra chronotick
```

### NTP Connection Issues
Requires UDP port 123 access. Test NTP connectivity:
```bash
python -c "import ntplib; c = ntplib.NTPClient(); print(c.request('time.google.com'))"
```

## Documentation

- **FIGURE_DESCRIPTIONS.md** - Publication figures explained
- **eval/ntp_correction_v2_design.md** - Algorithm documentation
- **CLAUDE.md** - Development guide

## Support

Report issues: https://github.com/JaimeCernuda/ChronoTick/issues
