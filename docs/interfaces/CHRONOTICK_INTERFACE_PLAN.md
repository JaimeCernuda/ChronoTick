# ChronoTick Interface Improvements Plan

**Date:** 2025-10-12
**Status:** Design Phase - Ready for Implementation
**Working Directory:** `chronotick_shm/` (keeps `tsfm/` untouched during evaluation)

---

## Executive Summary

ChronoTick has an excellent two-layer architecture:
- **Layer 1 (tsfm/):** Implementation layer - NTP + ML models (the "engine")
- **Layer 2 (chronotick_shm/):** Interface layer - Shared memory + MCP servers (the "library")

This plan focuses on **improving the interface layer** to make ChronoTick easier to use for both programs and AI agents, while keeping the implementation layer completely untouched.

---

## Current Architecture

```
ChronoTick/
│
├── tsfm/                               🔒 IMPLEMENTATION LAYER - DON'T TOUCH
│   ├── chronotick_inference/          Core NTP + ML inference
│   │   ├── real_data_pipeline.py      RealDataPipeline class
│   │   ├── ntp_client.py              Real NTP measurements
│   │   ├── daemon.py                  Multiprocessing daemon
│   │   └── config.yaml                Configuration
│   └── pyproject.toml                 Separate package (tsfm-factory)
│
└── chronotick_shm/                    ✅ INTERFACE LAYER - WORK HERE
    ├── src/chronotick_shm/            Client interface code
    │   ├── chronotick_daemon_server.py   Daemon (imports from tsfm)
    │   ├── chronotick_client_eval.py     Evaluation client
    │   ├── chronotick_sdk_mcp.py         SDK MCP tools
    │   ├── chronotick_stdio_mcp.py       Stdio MCP server
    │   └── shm_config.py                 Shared memory (128 bytes)
    ├── examples/                      Usage examples
    ├── docs/                          Documentation
    └── pyproject.toml                 Client package
```

**Key Design Pattern:**
- `chronotick_shm/chronotick_daemon_server.py` imports `RealDataPipeline` from `tsfm/`
- Daemon runs the pipeline and writes corrections to shared memory (128 bytes)
- Clients read from shared memory with ~300ns latency (lock-free)
- This separation allows research in `tsfm/` without breaking production clients

---

## Current State Assessment

### What Works Well ✅
- Ultra-low latency shared memory (~300ns reads)
- SDK MCP for Python agents (in-process tools)
- Stdio MCP for Claude Code (standalone server)
- Lock-free single-writer-multiple-reader pattern
- Comprehensive documentation and examples
- Clean separation between implementation and interface

### What Needs Improvement 🔨
1. **Package naming:** `chronotick-shm` → `chronotick` (simpler)
2. **API complexity:** Users need to understand shared memory internals
3. **Import ergonomics:** No high-level client class
4. **Documentation:** Scattered, not audience-specific
5. **Examples:** Limited to SDK MCP, missing basic usage

---

## Proposed Improvements

### 1. High-Level Client API (Priority 0)

**Problem:** Users must understand shared memory details to use ChronoTick.

**Solution:** Create simple `ChronoTickClient` class that hides complexity.

**New File:** `chronotick_shm/src/chronotick_shm/client.py`

```python
from chronotick import ChronoTickClient

# Simple usage - no shared memory knowledge needed
client = ChronoTickClient()

# Get corrected time
time_info = client.get_time()
print(f"Corrected: {time_info.corrected_timestamp}")
print(f"Uncertainty: ±{time_info.uncertainty_seconds * 1000:.2f}ms")
print(f"Confidence: {time_info.confidence:.1%}")

# High-precision synchronization
target_time = time_info.corrected_timestamp + 5.0
client.wait_until(target_time, tolerance_ms=0.5)
execute_synchronized_action()
```

**Implementation:**
```python
class ChronoTickClient:
    """High-level ChronoTick client for easy integration"""

    def __init__(self):
        """Initialize client (lazy connection to shared memory)"""
        self._shm = None

    def get_time(self) -> CorrectedTime:
        """Get corrected time with uncertainty bounds"""
        # Handles shared memory internally

    def is_daemon_ready(self) -> bool:
        """Check if daemon is running and ready"""

    def wait_until(self, target_corrected_time: float, tolerance_ms: float = 1.0):
        """High-precision wait until target corrected time"""
```

---

### 2. Package Reorganization (Priority 1)

**Problem:** Flat file structure, unclear what's what.

**Solution:** Organize into logical subsystems.

```
chronotick_shm/
├── pyproject.toml                     Updated config
├── README.md                          User-focused docs
├── src/chronotick_shm/
│   ├── __init__.py                    Export ChronoTickClient
│   ├── client.py                      NEW: High-level API
│   ├── shm/                           NEW: Shared memory subsystem
│   │   ├── __init__.py
│   │   ├── config.py                  (moved from shm_config.py)
│   │   └── reader.py                  Read logic
│   ├── daemon/                        NEW: Daemon subsystem
│   │   ├── __init__.py
│   │   └── server.py                  (moved from chronotick_daemon_server.py)
│   ├── mcp/                           NEW: MCP subsystem
│   │   ├── __init__.py
│   │   ├── sdk_server.py              (moved from chronotick_sdk_mcp.py)
│   │   └── stdio_server.py            (moved from chronotick_stdio_mcp.py)
│   └── eval/                          NEW: Evaluation subsystem
│       ├── __init__.py
│       └── benchmark.py               (moved from chronotick_client_eval.py)
├── examples/
│   ├── simple_client.py               NEW: Basic usage example
│   ├── distributed_sync.py            NEW: Multi-node coordination
│   ├── agent_integration.py           Existing SDK MCP example
│   └── monitoring_dashboard.py        NEW: Live monitoring
└── docs/
    ├── quickstart.md                  NEW: Getting started
    ├── api_reference.md               NEW: API documentation
    ├── mcp_integration.md             NEW: MCP usage guide
    └── architecture.md                NEW: System design
```

**Updated `__init__.py`:**
```python
"""ChronoTick - High-precision time synchronization"""

from .client import ChronoTickClient, CorrectedTime
from .shm.config import ChronoTickData, CorrectionSource

__all__ = [
    "ChronoTickClient",
    "CorrectedTime",
    "ChronoTickData",
    "CorrectionSource",
]

__version__ = "1.0.0"
```

---

### 3. Simplified Package Naming (Priority 1)

**Current:** `chronotick-shm` (internal implementation detail exposed)
**Proposed:** `chronotick` (clean, user-friendly)

**Updated `pyproject.toml`:**
```toml
[project]
name = "chronotick"  # Changed from "chronotick-shm"
version = "1.0.0"
description = "ChronoTick - High-precision time synchronization with ML-powered clock correction"
requires-python = ">=3.10"
dependencies = [
    "claude-agent-sdk>=0.1.0",
    "fastmcp>=0.2.0",
    "psutil>=5.9.0",
]

[project.scripts]
chronotick-daemon = "chronotick.daemon.server:main"
chronotick-client = "chronotick.eval.benchmark:main"
chronotick-stdio-server = "chronotick.mcp.stdio_server:main"
```

**User Installation:**
```bash
pip install chronotick  # Clean and simple!
```

---

### 4. Enhanced MCP Tools (Priority 2)

**Current MCP Tools:**
- `get_time` - Get corrected time
- `get_daemon_status` - Get daemon status
- `get_time_with_future_uncertainty` - Project uncertainty

**Proposed New Tools:**

```python
@mcp.tool()
def schedule_action(target_corrected_time: float, action_id: str) -> dict:
    """
    Schedule an action for a specific corrected time.

    Returns when to execute relative to system clock, accounting for
    drift and uncertainty. Useful for distributed coordination.
    """
    # Calculate system time that corresponds to target corrected time
    # accounting for offset and drift

@mcp.tool()
def synchronize_with_peers(peer_times: List[dict]) -> dict:
    """
    Coordinate distributed actions across multiple agents.

    Takes peer timestamps and uncertainties, returns optimal
    coordination strategy using vector clock logic.
    """

@mcp.tool()
def get_time_bounds(duration_seconds: float) -> dict:
    """
    Get guaranteed time bounds for a future duration.

    Useful for SLA guarantees and distributed consensus.
    Returns: [earliest_time, latest_time] accounting for uncertainty.
    """

@mcp.tool()
def wait_until_time(target_corrected_time: float, tolerance_ms: float = 1.0) -> dict:
    """
    Block until target corrected time is reached.

    High-precision waiting with sub-millisecond tolerance.
    Returns: actual_wake_time, accuracy_achieved
    """
```

---

### 5. Documentation Reorganization (Priority 1)

**Current:** Mixed documentation across README, STRUCTURE, REFERENCE, etc.

**Proposed:** Audience-specific documentation structure.

```
chronotick_shm/docs/
├── README.md                      Main landing page
├── quickstart/
│   ├── installation.md            Installing ChronoTick
│   ├── starting_daemon.md         Starting the daemon
│   └── first_client.md            Your first client program
├── guides/
│   ├── python_programs.md         Using from Python programs
│   ├── python_agents.md           SDK MCP for Python agents
│   ├── claude_code.md             Stdio MCP for Claude Code
│   ├── distributed_sync.md        Multi-node coordination
│   └── monitoring.md              Monitoring and observability
├── api/
│   ├── client_api.md              ChronoTickClient API reference
│   ├── mcp_tools.md               MCP tools reference
│   └── advanced_shm.md            Advanced: Direct shared memory
└── architecture/
    ├── overview.md                System architecture
    ├── performance.md             Performance characteristics
    ├── tsfm_integration.md        How tsfm and chronotick_shm connect
    └── shared_memory.md           Shared memory design
```

---

### 6. Example Programs (Priority 1)

**New Example: Simple Client**
```python
# examples/simple_client.py
from chronotick import ChronoTickClient

def main():
    client = ChronoTickClient()

    # Check daemon
    if not client.is_daemon_ready():
        print("ERROR: ChronoTick daemon not running")
        print("Start with: chronotick-daemon")
        return

    # Get time
    time_info = client.get_time()
    print(f"Corrected Time: {time_info.corrected_timestamp}")
    print(f"System Time:    {time_info.system_timestamp}")
    print(f"Uncertainty:    ±{time_info.uncertainty_seconds * 1000:.3f}ms")
    print(f"Confidence:     {time_info.confidence:.1%}")
    print(f"Source:         {time_info.source}")

if __name__ == "__main__":
    main()
```

**New Example: Distributed Synchronization**
```python
# examples/distributed_sync.py
from chronotick import ChronoTickClient
import socket
import json

def coordinate_distributed_write():
    """Coordinate writes across multiple nodes"""
    client = ChronoTickClient()

    # Leader proposes a sync point
    current_time = client.get_time()
    sync_point = current_time.corrected_timestamp + 10.0  # 10 seconds from now

    # Broadcast to followers
    broadcast_sync_point(sync_point)

    # Wait until sync point
    client.wait_until(sync_point, tolerance_ms=1.0)

    # Execute write - all nodes execute within 1ms window
    execute_database_write()

    print(f"Write executed at {client.get_time().corrected_timestamp}")
```

**New Example: Agent Integration**
```python
# examples/agent_integration.py
from claude_agent_sdk import ClaudeSDKClient
from chronotick.mcp import create_chronotick_sdk_server

async def main():
    # Create ChronoTick MCP server
    chronotick_server = create_chronotick_sdk_server()

    # Create agent with ChronoTick tools
    agent = ClaudeSDKClient({
        "mcp_servers": {"chronotick": chronotick_server},
        "allowed_tools": [
            "mcp__chronotick__get_time",
            "mcp__chronotick__schedule_action",
            "mcp__chronotick__synchronize_with_peers"
        ]
    })

    # Agent can now reason about time
    response = await agent.query(
        "What is the current corrected time? Schedule a backup 5 minutes "
        "from now and account for clock drift uncertainty."
    )

    print(response)
```

---

## Interface Options Matrix

| Interface | Target Users | Latency | Ease of Use | Status | Installation |
|-----------|-------------|---------|-------------|--------|--------------|
| **Python Client** | Python programs | ~300ns | High | 🔨 New API | `pip install chronotick` |
| **SDK MCP** | Python AI agents | ~300ns | High | ✅ Exists | `pip install chronotick` |
| **Stdio MCP** | Claude Code, any language | <1ms | High | ✅ Exists | `pip install chronotick` |
| **Direct SHM** | Performance-critical C/Go | ~300ns | Medium | ✅ Exists | Manual integration |
| **HTTP REST** | Non-Python programs | ~5ms | High | 🆕 Future | `pip install chronotick[http]` |

---

## Implementation Priority & Timeline

| Priority | Task | Files Modified/Added | Effort | Impact |
|----------|------|---------------------|--------|--------|
| **P0** | Create ChronoTickClient API | Add `client.py` | 2 hours | Critical - Makes ChronoTick easy to use |
| **P0** | Update `__init__.py` exports | `__init__.py` | 15 min | Critical - Clean imports |
| **P0** | Create simple example | `examples/simple_client.py` | 30 min | Critical - Shows how to use |
| **P1** | Write quickstart docs | `docs/quickstart.md` | 2 hours | High - User onboarding |
| **P1** | Update package naming | `pyproject.toml` | 30 min | High - Better UX |
| **P1** | Distributed sync example | `examples/distributed_sync.py` | 1 hour | High - Shows power |
| **P2** | Reorganize package structure | Move/rename files | 3 hours | Medium - Better organization |
| **P2** | Enhanced MCP tools | `mcp/sdk_server.py`, `mcp/stdio_server.py` | 3 hours | Medium - Richer agent interactions |
| **P3** | HTTP REST API (optional) | New subsystem | 4 hours | Low - Future enhancement |

**Total Estimated Time for P0-P1:** ~7 hours

---

## User Experience Vision

### For Python Programs:
```python
pip install chronotick

from chronotick import ChronoTickClient

client = ChronoTickClient()
time_info = client.get_time()
print(f"Time: {time_info.corrected_timestamp}")
print(f"Uncertainty: ±{time_info.uncertainty_seconds * 1000:.2f}ms")
```

### For Python AI Agents:
```python
from chronotick.mcp import create_chronotick_sdk_server
from claude_agent_sdk import ClaudeSDKClient

server = create_chronotick_sdk_server()
agent = ClaudeSDKClient({"mcp_servers": {"chronotick": server}})

response = await agent.query("Get current corrected time")
```

### For Claude Code:
```json
// ~/.claude/config.json
{
  "mcpServers": {
    "chronotick": {
      "command": "uv",
      "args": ["run", "chronotick-stdio-server"]
    }
  }
}
```

---

## Key Constraints

### What We WILL NOT Touch:
- ❌ `tsfm/` directory - Evaluation running, completely off-limits
- ❌ `tsfm/chronotick_inference/` - Implementation layer stays as-is
- ❌ Any running evaluation processes or tests
- ❌ Configuration files in `tsfm/`

### What We WILL Do:
- ✅ Work entirely in `chronotick_shm/` directory
- ✅ Add new client API and documentation
- ✅ Reorganize for better structure
- ✅ Create examples and guides
- ✅ Improve package naming and installation

---

## Success Criteria

1. **Ease of Use:** A new user can install and use ChronoTick in <5 minutes
2. **Documentation:** Clear quickstart guides for each user type
3. **Examples:** Working examples for programs, agents, and Claude Code
4. **API Quality:** High-level API that hides complexity
5. **Backwards Compatibility:** Existing code continues to work
6. **Package Naming:** Clean, professional naming (`chronotick` not `chronotick-shm`)

---

## Next Steps

1. ✅ Get approval for this plan
2. 🔨 Implement P0 tasks (ChronoTickClient API, examples)
3. 🔨 Implement P1 tasks (documentation, package naming)
4. 📝 Review and test all changes
5. 📦 Publish updated package
6. 📢 Update documentation and examples

---

## Technical Notes

### Architecture Insight
The two-layer design is excellent:
- **tsfm/** = Implementation ("PostgreSQL server" - the engine)
- **chronotick_shm/** = Interface ("psycopg2 library" - the client)

This separation allows:
- Research and experimentation in `tsfm/` without breaking users
- Stable, production-ready interface in `chronotick_shm/`
- Easy integration via shared memory (~300ns reads)
- Multiple interface options (direct, SDK MCP, stdio MCP)

### Performance
- Shared memory reads: ~300ns (after first attach)
- First shared memory attach: ~1.5ms
- MCP tool calls: <1ms total latency
- Daemon update frequency: 100 Hz (configurable 1-1000 Hz)

### Dependencies
- **Implementation (tsfm/):** numpy, timesfm, chronos, psutil, yaml
- **Interface (chronotick_shm/):** claude-agent-sdk, fastmcp, psutil (minimal!)

---

**Document Status:** Complete - Ready for Implementation
**Author:** Claude (based on analysis of existing codebase)
**Date:** 2025-10-12
