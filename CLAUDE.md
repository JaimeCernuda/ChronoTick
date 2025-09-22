# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ChronoTick is a high-precision time synchronization system for geo-distributed AI agents using the Model Context Protocol (MCP). It combines multiple time sources (System Clock, NTP, PTP), machine learning inference for predictive time corrections, and fast IPC communication to deliver microsecond-precision clock corrections with uncertainty quantification.

## Core Architecture

The project has three main components:

1. **ChronoTick Server** (`chronotick-server/`): MCP time server with nanosecond precision and multiple time source management
2. **TSFM Factory** (`tsfm/`): Unified interface for time series foundation models (TimesFM, TTM, Chronos, Toto, Time-MoE)
3. **Inference Layer** (`inferance_layer/`): ML-powered clock drift prediction using TSFM models
4. **MCP Servers** (`servers/`): Reference MCP server implementations

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

### Configuration Files
- `tsfm/chronotick_inference/config.yaml`: Main ChronoTick configuration
- `tsfm/chronotick_inference/configs/`: Model-specific configurations
- `tsfm/tsfm/config/default.yaml`: TSFM factory default settings