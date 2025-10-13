"""
ChronoTick Shared Memory - Ultra-low latency time synchronization

This package provides ~300ns read latency for high-precision time corrections
via shared memory IPC, enabling real-time coordination between AI agents.

Quick Start:
    >>> from chronotick_shm import ChronoTickClient
    >>> client = ChronoTickClient()
    >>> time_info = client.get_time()
    >>> print(f"Corrected time: {time_info.corrected_timestamp}")
    >>> print(f"Uncertainty: ±{time_info.uncertainty_seconds * 1000:.2f}ms")

Components:
- ChronoTickClient: High-level client API (recommended for most users)
- CorrectedTime: Result from get_time() with all time information
- get_current_time: Convenience function for quick time access
- tools.sdk_tools: SDK MCP tools (@tool decorated functions)
- tools.agents: Agent configuration helpers
- shm_config: Memory layout and serialization (advanced)
- chronotick_daemon: Background daemon (writer)
- chronotick_stdio_mcp: Stdio MCP server

MCP Integration:
    # SDK MCP (in-process)
    >>> from chronotick_shm.tools.agents import create_chronotick_agent
    >>> agent_options = create_chronotick_agent()

    # Stdio MCP (standalone server)
    >>> # Run: chronotick-stdio-server
"""

# High-level client API (recommended for most users)
from chronotick_shm.client import (
    ChronoTickClient,
    CorrectedTime,
    get_current_time,
)

# Low-level shared memory API (advanced users)
from chronotick_shm.shm_config import (
    SHARED_MEMORY_NAME,
    SHARED_MEMORY_SIZE,
    ChronoTickData,
    CorrectionSource,
    StatusFlags,
    write_data,
    read_data,
    read_data_with_retry,
)

__version__ = "0.1.0"

# Export high-level API prominently
__all__ = [
    # High-level client API (start here!)
    "ChronoTickClient",
    "CorrectedTime",
    "get_current_time",
    # Low-level shared memory API (advanced)
    "SHARED_MEMORY_NAME",
    "SHARED_MEMORY_SIZE",
    "ChronoTickData",
    "CorrectionSource",
    "StatusFlags",
    "write_data",
    "read_data",
    "read_data_with_retry",
]

# MCP tools available at:
# - chronotick_shm.tools.sdk_tools (SDK MCP @tool functions)
# - chronotick_shm.tools.agents (Agent configuration)
# - chronotick_shm.chronotick_stdio_mcp (Stdio MCP server)
