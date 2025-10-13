# ChronoTick Shared Memory - Project Structure

**Clean 4-section architecture for ultra-low latency time synchronization**

---

## Directory Structure

```
chronotick_shm/
├── src/chronotick_shm/
│   ├── shm_config.py                    # Shared memory layout (used by all)
│   │
│   ├── chronotick_daemon_server/        # Section 1: Daemon Server
│   │   ├── __init__.py
│   │   └── daemon.py                    # Writes to shared memory
│   │
│   ├── chronotick_client_eval/          # Section 2: Evaluation Client
│   │   ├── __init__.py
│   │   └── client.py                    # Reads, benchmarks, monitors
│   │
│   ├── chronotick_sdk_mcp/              # Section 3: SDK MCP
│   │   ├── __init__.py
│   │   ├── tools.py                     # @tool decorated functions
│   │   └── agent_helpers.py             # Agent configuration helpers
│   │
│   └── chronotick_stdio_mcp/            # Section 4: Stdio MCP Server
│       ├── __init__.py
│       └── server.py                    # FastMCP standalone server
│
├── examples/                            # Usage examples
├── docs/                                # Documentation
├── pyproject.toml                       # UV configuration
└── README.md                            # Main README
```

---

## The 4 Sections Explained

### 1. `chronotick_daemon_server/` - Daemon Server

**Purpose**: Background daemon that writes ChronoTick time corrections to shared memory

**Key File**: `daemon.py`
- Integrates with RealDataPipeline (NTP + ML models)
- Writes corrections at 100-1000 Hz (configurable)
- Lock-free single-writer pattern
- Handles signal shutdown gracefully

**Entry Point**: `uv run chronotick-daemon`

**Example**:
```bash
uv run chronotick-daemon --config ../tsfm/chronotick_inference/config.yaml --freq 100
```

**Dependencies**: Requires full ChronoTick environment (numpy, chronotick_inference from tsfm)

---

### 2. `chronotick_client_eval/` - Evaluation Client

**Purpose**: Read and benchmark ChronoTick shared memory (no MCP overhead)

**Key File**: `client.py`
- Direct shared memory access (~300ns reads)
- Performance benchmarking
- Continuous monitoring
- JSON export

**Entry Point**: `uv run chronotick-client`

**Commands**:
```bash
uv run chronotick-client read              # Read once
uv run chronotick-client monitor --interval 0.1  # Monitor continuously
uv run chronotick-client status            # Daemon status
uv run chronotick-client benchmark --iterations 10000  # Benchmark
uv run chronotick-client json --pretty     # JSON export
```

**Dependencies**: Only needs chronotick-shm package

---

### 3. `chronotick_sdk_mcp/` - SDK MCP (In-Process)

**Purpose**: MCP tools that run **in-process** with your Python agent using claude-agent-sdk

**Key Files**:
- `tools.py`: @tool decorated functions for claude-agent-sdk
- `agent_helpers.py`: Helper functions to create agents

**How it Works**:
- Tools run **inside** your Python process
- Uses `@tool` decorator from `claude-agent-sdk`
- Ultra-low latency (~300ns) because no IPC between agent and tools

**Example**:
```python
from chronotick_shm.chronotick_sdk_mcp import create_chronotick_agent
from claude_agent_sdk import ClaudeSDKClient

# Create agent with ChronoTick tools
agent_options = create_chronotick_agent()
agent = ClaudeSDKClient(agent_options)

# Query time
response = await agent.query("What is the corrected time?")
print(response.text)
```

**Tools Available**:
- `get_time()` - Get corrected time with uncertainty
- `get_daemon_status()` - Monitor daemon health
- `get_time_with_future_uncertainty(future_seconds)` - Project uncertainty

**Dependencies**: `claude-agent-sdk`

---

### 4. `chronotick_stdio_mcp/` - Stdio MCP Server (Standalone)

**Purpose**: Standalone MCP server that runs as **separate process** for Claude Code/other MCP clients

**Key File**: `server.py` (built with fastmcp)

**How it Works**:
- Runs as separate process
- Communicates via stdio (standard input/output)
- Claude Code connects to it as external tool
- Reads from shared memory and serves via MCP protocol

**Entry Point**: `uv run chronotick-stdio-server`

**Connect from Claude Code**:
Add to `~/.claude/config.json`:
```json
{
  "mcpServers": {
    "chronotick": {
      "command": "uv",
      "args": ["run", "chronotick-stdio-server"],
      "cwd": "/path/to/chronotick_shm"
    }
  }
}
```

**Dependencies**: `fastmcp`

---

## Key Differences: SDK MCP vs Stdio MCP

| Aspect | SDK MCP (Section 3) | Stdio MCP (Section 4) |
|--------|---------------------|----------------------|
| **Process** | In-process (same Python) | Separate process |
| **Integration** | `from chronotick_shm.chronotick_sdk_mcp` | stdio connection |
| **Use Case** | Programmatic agents | Claude Code, external clients |
| **Latency** | ~300ns (direct memory) | ~300ns + stdio overhead |
| **Library** | claude-agent-sdk | fastmcp |
| **Decorator** | `@tool` from claude-agent-sdk | `@mcp.tool()` from fastmcp |

---

## Data Flow

```
┌─────────────────────────┐
│ Section 1: Daemon       │  NTP + ML Models
│ (chronotick_daemon)     │  ↓
└──────────┬──────────────┘  Writes at 100-1000 Hz
           │
           ↓
┌──────────────────────────┐
│ Shared Memory (128 bytes)│  Lock-free, cache-aligned
│ /dev/shm/chronotick_shm  │
└──────────┬───────────────┘
           │
           ├─────────────┬──────────────┬─────────────┐
           ↓             ↓              ↓             ↓
    Section 2:     Section 3:     Section 4:    Your Apps
    Client Eval    SDK MCP        Stdio MCP
    (benchmarks)   (in-process)   (stdio)
```

---

## Entry Points Summary

```bash
# Daemon
uv run chronotick-daemon [--config PATH] [--freq HZ]

# Client
uv run chronotick-client {read,monitor,status,benchmark,json}

# Stdio MCP Server
uv run chronotick-stdio-server [--debug]
```

---

## Installation

```bash
cd chronotick_shm

# Install all dependencies
uv sync

# Verify installation
uv run python -c "from chronotick_shm import SHARED_MEMORY_NAME; print('✓ Installed')"
```

---

## Quick Start

### 1. Start Daemon (Terminal 1)
```bash
cd chronotick_shm
uv run chronotick-daemon --config ../tsfm/chronotick_inference/config.yaml
# Wait for warmup (~3 minutes)
```

### 2. Test Client (Terminal 2)
```bash
cd chronotick_shm
uv run chronotick-client read
uv run chronotick-client benchmark --iterations 10000
```

### 3. Use SDK MCP (Python)
```python
from chronotick_shm.chronotick_sdk_mcp import get_shared_memory, read_chronotick_data
import time

# Direct access (fastest)
shm = get_shared_memory()
data = read_chronotick_data()
corrected_time = data.get_corrected_time_at(time.time())
print(f"Time: {corrected_time:.6f}")
```

### 4. Use Stdio MCP (Claude Code)
```bash
# Add to ~/.claude/config.json then restart Claude Code
```

---

## Dependencies

**Core** (all sections):
- Python >=3.10
- psutil

**Section 1** (daemon only):
- numpy, chronotick_inference (from tsfm environment)

**Section 3** (SDK MCP only):
- claude-agent-sdk

**Section 4** (Stdio MCP only):
- fastmcp

---

## Testing

```bash
# Test imports
uv run python -c "from chronotick_shm.chronotick_sdk_mcp import get_time; print('✓')"
uv run python -c "from chronotick_shm.chronotick_stdio_mcp import main; print('✓')"
uv run python -c "from chronotick_shm.chronotick_client_eval import ChronoTickClient; print('✓')"

# Test entry points (daemon not needed for these to exist)
uv run chronotick-client --help
uv run chronotick-stdio-server --help
uv run chronotick-daemon --help
```

---

## Common Use Cases

### Use Case 1: Python Agent with SDK MCP
**Need**: Build agent that uses ChronoTick time
**Section**: 3 (SDK MCP)
```python
from chronotick_shm.chronotick_sdk_mcp import create_chronotick_agent
from claude_agent_sdk import ClaudeSDKClient

agent = ClaudeSDKClient(create_chronotick_agent())
response = await agent.query("What time is it?")
```

### Use Case 2: Claude Code Integration
**Need**: Use ChronoTick from Claude Code
**Section**: 4 (Stdio MCP)
- Start: `uv run chronotick-stdio-server`
- Configure in ~/.claude/config.json
- Use tools in Claude Code

### Use Case 3: Performance Testing
**Need**: Benchmark shared memory reads
**Section**: 2 (Client Eval)
```bash
uv run chronotick-client benchmark --iterations 100000
```

### Use Case 4: Direct Integration
**Need**: Maximum performance, no MCP
**Section**: 3 (SDK MCP tools, direct import)
```python
from chronotick_shm.chronotick_sdk_mcp import get_shared_memory, read_chronotick_data
# Use directly
```

---

## Migration from Old Structure

**Old** → **New**:
- `chronotick_daemon.py` → `chronotick_daemon_server/daemon.py`
- `chronotick_client.py` → `chronotick_client_eval/client.py`
- `tools/chronotick_sdk_tools.py` → `chronotick_sdk_mcp/tools.py`
- `tools/create_chronotick_agent.py` → `chronotick_sdk_mcp/agent_helpers.py`
- `chronotick_sdk_mcp_server.py` → `chronotick_stdio_mcp/server.py` (rewritten with fastmcp)

**Entry points updated**:
- `chronotick-daemon` → now points to `chronotick_daemon_server:main`
- `chronotick-client` → now points to `chronotick_client_eval:main`
- `chronotick-server` → removed (was confusing)
- NEW: `chronotick-stdio-server` → points to `chronotick_stdio_mcp:main`

---

## Summary

**4 Clean Sections**:
1. ✅ **Daemon Server** - Writes to shared memory (NTP + ML)
2. ✅ **Client Eval** - Reads, benchmarks, monitors (direct access)
3. ✅ **SDK MCP** - In-process tools for Python agents (claude-agent-sdk)
4. ✅ **Stdio MCP** - Standalone server for Claude Code (fastmcp)

Each section has a clear purpose and minimal dependencies. No confusion!
