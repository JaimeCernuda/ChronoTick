"""
ChronoTick Shared Memory - Ultra-low latency time synchronization

This package provides ~300ns read latency for high-precision time corrections
via shared memory IPC, enabling real-time coordination between AI agents.

Components:
- shm_config: Memory layout and serialization
- chronotick_daemon: Background daemon (writer)
- chronotick_client: Evaluation client (reader)
- chronotick_sdk_mcp_server: Standalone MCP server
- tools: SDK MCP tools and agent helpers
"""

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

__all__ = [
    "SHARED_MEMORY_NAME",
    "SHARED_MEMORY_SIZE",
    "ChronoTickData",
    "CorrectionSource",
    "StatusFlags",
    "write_data",
    "read_data",
    "read_data_with_retry",
]
