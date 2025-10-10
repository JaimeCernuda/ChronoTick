# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
Check the guies on deploy.md, design eval and technical for information on the system

## Project Overview

ChronoTick is a high-precision time synchronization system for geo-distributed AI agents using the Model Context Protocol (MCP). It combines real NTP measurements, dual machine learning models for predictive time corrections, and fast IPC communication to deliver microsecond-precision clock corrections with uncertainty quantification.

## Core Architecture

The project consists of three main directories with different maturity levels:

1. **TSFM Factory with ChronoTick MCP** (`tsfm/`): **PRIMARY IMPLEMENTATION** - Production-ready
   - `tsfm/chronotick_inference/`: Core ChronoTick MCP server implementation with dual-model prediction
   - `tsfm/chronotick_mcp.py`: Main entry point for the MCP server
   - `tsfm/tsfm/`: Unified factory interface for time series foundation models (Chronos, TimesFM, TTM, Toto, Time-MoE)
   - Status: Fully functional with comprehensive tests (25/25 passing)

2. **ChronoTick Server** (`chronotick-server/`): Alternative/future implementation
   - Newer design with PTP support preparation and vector clocks
   - Status: Basic framework only, not the active implementation
   - Note: Use `tsfm/chronotick_inference/` for actual development

3. **Inference Layer** (`inferance_layer/`): Deprecated
   - Status: Superseded by `tsfm/chronotick_inference/`

4. **MCP Servers** (`servers/`): Reference MCP implementations (git, fetch, time, etc.)

## Development Commands

### Environment Setup
```bash
# Main TSFM environment (primary workspace)
cd tsfm/
uv sync --extra core-models  # For Chronos/TimesFM only
uv sync --extra ttm          # For TTM models (transformers==4.38.0)
uv sync --extra toto         # For Toto models (transformers>=4.52.0)
uv sync --extra time-moe     # For Time-MoE (transformers==4.40.1)
uv sync --extra dev --extra test  # Development dependencies

# ChronoTick Server
cd chronotick-server/
uv sync

# Inference Layer
cd inferance_layer/
uv sync
```

### Running Services

```bash
# Start ChronoTick MCP Server (main service)
cd tsfm/
uv run python chronotick_mcp.py
uv run python chronotick_mcp.py --debug-trace --debug-log-file /tmp/chronotick-debug.log

# Start ChronoTick Server (time server)
cd chronotick-server/
uv run chronotick-server
uv run chronotick-server --node-id "node-001" --log-level DEBUG
```

### Testing
```bash
# TSFM tests
cd tsfm/
uv run pytest                           # All tests
uv run pytest -m unit                   # Unit tests only
uv run pytest -m integration            # Integration tests
uv run pytest -m "not slow"             # Skip slow tests
uv run pytest -m "not gpu"              # Skip GPU tests
uv run pytest --cov=tsfm --cov-report=html

# ChronoTick Server tests
cd chronotick-server/
uv run pytest --cov=src/chronotick_server --cov-report=html

# Specific test files
uv run pytest tests/chronotick/test_mcp_server.py -v
uv run pytest tests/integration/test_performance.py -v
```

### Code Quality
```bash
# Format code
uv run black tsfm/ tests/
uv run black src/ tests/

# Lint
uv run ruff tsfm/ tests/
uv run ruff src/ tests/

# Type checking
uv run mypy tsfm/
uv run pyright src/
```

### MCP Server Operations
```bash
# Connect to Claude Code
claude mcp add chronotick --scope user \
  --env PYTHONPATH=$(pwd) \
  -- bash -c "cd $(pwd) && $(pwd)/.venv/bin/python chronotick_mcp.py"
```

## Key Implementation Details

### MCP Tools Available
- **`get_time`**: Get corrected time with uncertainty bounds
- **`get_daemon_status`**: Monitor daemon health and performance
- **`get_time_with_future_uncertainty`**: Project time uncertainty into future

### Model Environment Management
Due to transformers version conflicts, models require mutually exclusive environments:
- TTM requires `transformers==4.38.0`
- Time-MoE requires `transformers==4.40.1`
- Toto requires `transformers>=4.52.0`
- Core models (Chronos, TimesFM) have no conflicts

Switch environments as needed when working with specific models.

### Time Source Hierarchy
1. PTP (Precision Time Protocol) - highest precision with hardware timestamping
2. NTP (Network Time Protocol) - network-based synchronization
3. System Clock - fallback option

The PrecisionClockManager automatically selects the best available source based on quality metrics.

### Performance Considerations
- Warmup period: 3 minutes for quality NTP measurements
- Response time: <1ms for time requests
- Memory usage: 50-200MB depending on loaded models
- Quality thresholds: 10ms default max acceptable uncertainty

### Testing Strategy
- Unit tests: Model-specific and feature tests
- Integration tests: End-to-end functionality with performance benchmarks
- Reports generated in `tests/integration/reports/` with visualizations
- Minimum coverage requirement: 80% for chronotick-server
- ChronoTick MCP tests: `tsfm/tests/chronotick/` (test_mcp_server.py, test_ntp_client.py, etc.)

### Configuration Files
- `tsfm/chronotick_inference/config.yaml`: Main ChronoTick configuration
- `tsfm/chronotick_inference/configs/`: Model-specific configurations (cpu_only_chronos.yaml, gpu_only_timesfm.yaml, etc.)
- `tsfm/tsfm/config/default.yaml`: TSFM factory default settings

### Key Architecture Details

**Dual-Model Prediction System** (`tsfm/chronotick_inference/`):
- **Short-term model**: 1Hz updates, 5-second horizon, 100-measurement context
- **Long-term model**: 0.033Hz (30s) updates, 60-second horizon, 300-measurement context
- **Fusion method**: Inverse variance weighting for optimal predictions
- **Retrospective correction**: Adjusts past predictions when new NTP measurements arrive

**IPC Communication** (daemon.py):
- Separate daemon process with dedicated CPU affinity
- Multiprocessing queues for request/response/command/status
- <1ms cache hit latency for time requests
- 45ms average cache miss latency (includes model inference)

**Real NTP Implementation** (ntp_client.py):
- Queries multiple servers: pool.ntp.org, time.google.com, time.cloudflare.com, time.nist.gov
- Quality filtering: Rejects measurements >10ms uncertainty by default
- Calculates offset using standard NTP formula
- Warmup period: 180 seconds at 1Hz, then 0.1Hz for operational phase

**MCP Server Entry Point** (`tsfm/chronotick_mcp.py`):
- Main script to run: `uv run python chronotick_mcp.py`
- Debug mode: `--debug-trace --debug-log-file /tmp/debug.log`
- Implements three MCP tools: `get_time`, `get_daemon_status`, `get_time_with_future_uncertainty`

### Important Implementation Notes

**Working Directory**:
- Always run from `tsfm/` directory: `cd tsfm/ && uv run python chronotick_mcp.py`
- PYTHONPATH should include the tsfm directory for imports to work correctly

**Model Loading**:
- External models must be installed separately: `pip install chronos-forecasting timesfm`
- These are NOT in pyproject.toml to avoid version conflicts
- Default configuration uses Chronos-Bolt which is compatible with all environments

**Platform Limitations**:
- Windows: Multiprocessing IPC has limited support, may require modifications
- macOS: Works well with launchd for service deployment
- Linux: Recommended platform with systemd support

**Network Requirements**:
- Requires UDP port 123 access for NTP queries
- At least one NTP server must be reachable for system to function
- Quality degrades gracefully if some servers are unavailable

### Evaluation and Testing

**Available Datasets** (see eval.md):
- `synced_tacc.csv`: Synchronized trace from TACC HPC cluster
- `unsynced.csv`, `unsynced_uc.csv`: Unsynchronized traces for testing
- Located at: https://github.com/JaimeCernuda/LLMs/tree/main/datasets

**Evaluation Plan** (eval/ directory):
1. Correctness & Accuracy with real-world datasets
2. Distributed clock synchronization on HPC
3. Multi-agent AI coordination scenarios

**Test Markers**:
- `pytest -m unit`: Fast unit tests only
- `pytest -m "not slow"`: Skip time-consuming tests
- `pytest -m "not gpu"`: Skip GPU-requiring tests
- `pytest -m "not network"`: Skip tests requiring internet