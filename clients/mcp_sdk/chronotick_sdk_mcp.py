#!/usr/bin/env python3
"""
ChronoTick SDK MCP Tools

MCP tools for claude-agent-sdk that provide high-precision time services
via shared memory IPC. These tools integrate with the ChronoTick daemon
for ultra-low latency time corrections (~300ns read latency).

Features:
- get_time: Get corrected time with uncertainty bounds
- get_daemon_status: Monitor daemon health and performance
- get_time_with_future_uncertainty: Project uncertainty into future

Performance:
- First call: ~1.5ms (shared memory attachment + read)
- Subsequent calls: ~0.0003ms (300ns) - 5000x faster
- Zero serialization overhead
- Lock-free reads (no contention)

Usage:
    from claude_agent_sdk import create_sdk_mcp_server
    from chronotick_sdk_tools import get_time, get_daemon_status, get_time_with_future_uncertainty

    sdk_server = create_sdk_mcp_server(
        name="chronotick_server",
        version="1.0.0",
        tools=[get_time, get_daemon_status, get_time_with_future_uncertainty]
    )
"""

import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from multiprocessing.shared_memory import SharedMemory

# NOTE: claude_agent_sdk must be installed: pip install claude-agent-sdk
try:
    from claude_agent_sdk import tool
except ImportError:
    raise ImportError(
        "claude-agent-sdk not found. Install with: pip install claude-agent-sdk"
    )

from chronotick_shm.shm_config import (
    SHARED_MEMORY_NAME,
    ChronoTickData,
    read_data_with_retry,
    CorrectionSource
)

logger = logging.getLogger(__name__)

# ============================================================================
# Global Shared Memory Handle (Singleton Pattern)
# ============================================================================
# This is the KEY optimization: attach once, reuse forever
# First call: ~1.5ms, subsequent calls: ~0.3μs (5000x faster)

_shared_memory_handle: Optional[SharedMemory] = None


def get_shared_memory() -> SharedMemory:
    """
    Get or create shared memory handle (singleton pattern).

    This function implements the global handle pattern for maximum performance:
    - First call: Attaches to shared memory (~1.5ms)
    - Subsequent calls: Returns cached handle (~1ns)

    Returns:
        SharedMemory handle

    Raises:
        RuntimeError: If daemon not running (shared memory doesn't exist)
    """
    global _shared_memory_handle

    if _shared_memory_handle is None:
        try:
            _shared_memory_handle = SharedMemory(
                name=SHARED_MEMORY_NAME,
                create=False  # Attach to existing, don't create
            )
            logger.info(f"Attached to shared memory: {SHARED_MEMORY_NAME}")
        except FileNotFoundError:
            raise RuntimeError(
                f"ChronoTick daemon not running.\n"
                f"Shared memory '{SHARED_MEMORY_NAME}' not found.\n\n"
                f"Start the daemon with:\n"
                f"  python chronotick_daemon.py\n\n"
                f"Or:\n"
                f"  python chronotick_daemon.py --config /path/to/config.yaml --freq 100"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to attach to shared memory: {e}")

    return _shared_memory_handle


def read_chronotick_data() -> ChronoTickData:
    """
    Read ChronoTickData from shared memory with retry logic.

    Returns:
        ChronoTickData from shared memory

    Raises:
        RuntimeError: If daemon not running or data read fails
    """
    shm = get_shared_memory()

    try:
        # Read with automatic retry on torn reads
        data = read_data_with_retry(shm.buf, max_retries=3)
        return data
    except Exception as e:
        raise RuntimeError(f"Failed to read ChronoTick data: {e}")


# ============================================================================
# MCP Tools (@tool decorated functions)
# ============================================================================

@tool(
    name="get_time",
    description=(
        "Get high-precision corrected time with uncertainty bounds from ChronoTick. "
        "Returns corrected timestamp, uncertainty estimates, and metadata about the correction. "
        "Use this when you need accurate time with error bounds for coordination or timestamping."
    ),
    input_schema={}  # No parameters required
)
async def get_time(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get corrected time with uncertainty bounds.

    This tool provides:
    - Corrected timestamp (system time + offset + drift correction)
    - System timestamp (raw uncorrected time)
    - Offset correction and drift rate
    - Uncertainty estimates (offset and drift)
    - Total time uncertainty at current moment
    - Confidence level (0-1)
    - Data source (NTP, CPU model, GPU model, fusion)
    - Validity period

    Returns:
        Dict with corrected time and all metadata

    Example response:
        {
          "corrected_time": 1704556800.123456,
          "system_time": 1704556800.122222,
          "offset_correction": 0.001234,
          "drift_rate": 1.5e-06,
          "offset_uncertainty": 1e-05,
          "drift_uncertainty": 1e-09,
          "time_uncertainty": 1.2e-05,
          "confidence": 0.95,
          "source": "fusion",
          "prediction_time": 1704556795.0,
          "valid_until": 1704556855.0,
          "is_valid": true,
          "is_ntp_ready": true,
          "is_models_ready": true,
          "call_latency_ms": 0.0003
        }
    """
    try:
        call_start = time.time()

        # Read from shared memory
        data = read_chronotick_data()

        # Calculate current corrected time
        current_system_time = time.time()
        time_delta = current_system_time - data.prediction_time
        corrected_time = data.get_corrected_time_at(current_system_time)

        # Calculate time uncertainty at current moment
        time_uncertainty = data.get_time_uncertainty(time_delta)

        call_latency = (time.time() - call_start) * 1000  # Convert to ms

        # Build response
        response = {
            "corrected_time": corrected_time,
            "system_time": current_system_time,
            "offset_correction": data.offset_correction,
            "drift_rate": data.drift_rate,
            "offset_uncertainty": data.offset_uncertainty,
            "drift_uncertainty": data.drift_uncertainty,
            "time_uncertainty": time_uncertainty,
            "confidence": data.confidence,
            "source": data.source.name.lower(),
            "prediction_time": data.prediction_time,
            "valid_until": data.valid_until,
            "is_valid": data.is_valid,
            "is_ntp_ready": data.is_ntp_ready,
            "is_models_ready": data.is_models_ready,
            "call_latency_ms": call_latency
        }

        return {
            "content": [{
                "type": "text",
                "text": (
                    f"ChronoTick Corrected Time\n"
                    f"{'='*50}\n"
                    f"Corrected Time: {corrected_time:.6f}\n"
                    f"System Time:    {current_system_time:.6f}\n"
                    f"Offset:         {data.offset_correction*1e6:+.3f}μs\n"
                    f"Drift Rate:     {data.drift_rate*1e6:+.3f}μs/s\n"
                    f"Uncertainty:    ±{time_uncertainty*1e6:.3f}μs\n"
                    f"Confidence:     {data.confidence:.1%}\n"
                    f"Source:         {data.source.name}\n"
                    f"Valid Until:    {data.valid_until:.1f} ({data.valid_until - current_system_time:.1f}s remaining)\n"
                    f"Status:         {'✓' if data.is_valid else '✗'} Valid, "
                    f"{'✓' if data.is_ntp_ready else '✗'} NTP, "
                    f"{'✓' if data.is_models_ready else '✗'} Models\n"
                    f"Call Latency:   {call_latency:.4f}ms\n"
                    f"\nFull Data (JSON):\n{response}"
                )
            }]
        }

    except RuntimeError as e:
        # Daemon not running or data read failed
        return {
            "content": [{
                "type": "text",
                "text": (
                    f"❌ ChronoTick Error\n"
                    f"{'='*50}\n"
                    f"{str(e)}\n\n"
                    f"Please ensure the ChronoTick daemon is running."
                )
            }]
        }

    except Exception as e:
        logger.error(f"Unexpected error in get_time: {e}")
        return {
            "content": [{
                "type": "text",
                "text": f"❌ Unexpected error: {e}"
            }]
        }


@tool(
    name="get_daemon_status",
    description=(
        "Get ChronoTick daemon status, health metrics, and performance statistics. "
        "Use this to monitor daemon health, check warmup progress, or diagnose issues."
    ),
    input_schema={}  # No parameters required
)
async def get_daemon_status(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get daemon status and performance metrics.

    Returns:
        Dict with comprehensive daemon status

    Example response:
        {
          "status": "ready",
          "warmup_complete": true,
          "measurement_count": 1523,
          "total_corrections": 152300,
          "daemon_uptime": 456.78,
          "last_ntp_sync": 1704556795.123,
          "ntp_ready": true,
          "models_ready": true,
          "memory_usage_mb": 185.3,
          "average_latency_ms": 0.45,
          "data_source": "fusion",
          "confidence": 0.95
        }
    """
    try:
        call_start = time.time()

        # Read from shared memory
        data = read_chronotick_data()

        # Get process info
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            cpu_affinity = process.cpu_affinity()
        except:
            memory_mb = None
            cpu_affinity = None

        call_latency = (time.time() - call_start) * 1000

        # Build status response
        status_data = {
            "status": "ready" if data.is_valid and data.is_warmup_complete else "warming_up",
            "warmup_complete": data.is_warmup_complete,
            "measurement_count": data.measurement_count,
            "total_corrections": data.total_corrections,
            "daemon_uptime": data.daemon_uptime,
            "last_ntp_sync": data.last_ntp_sync,
            "seconds_since_ntp": time.time() - data.last_ntp_sync if data.last_ntp_sync > 0 else None,
            "ntp_ready": data.is_ntp_ready,
            "models_ready": data.is_models_ready,
            "memory_usage_mb": memory_mb,
            "cpu_affinity": cpu_affinity,
            "average_latency_ms": data.average_latency_ms,
            "data_source": data.source.name.lower(),
            "confidence": data.confidence,
            "call_latency_ms": call_latency
        }

        return {
            "content": [{
                "type": "text",
                "text": (
                    f"ChronoTick Daemon Status\n"
                    f"{'='*50}\n"
                    f"Status:          {status_data['status'].upper()}\n"
                    f"Uptime:          {data.daemon_uptime:.1f}s ({data.daemon_uptime/60:.1f} min)\n"
                    f"Warmup:          {'✓ Complete' if data.is_warmup_complete else '⏳ In Progress'}\n"
                    f"\n"
                    f"Data Collection:\n"
                    f"  Measurements:  {data.measurement_count}\n"
                    f"  Corrections:   {data.total_corrections}\n"
                    f"  Last NTP:      {status_data['seconds_since_ntp']:.1f}s ago\n"
                    f"  NTP Ready:     {'✓' if data.is_ntp_ready else '✗'}\n"
                    f"  Models Ready:  {'✓' if data.is_models_ready else '✗'}\n"
                    f"\n"
                    f"Performance:\n"
                    f"  Avg Latency:   {data.average_latency_ms:.3f}ms\n"
                    f"  Call Latency:  {call_latency:.4f}ms\n"
                    f"  Memory:        {memory_mb:.1f}MB\n"
                    f"  CPU Affinity:  {cpu_affinity or 'not set'}\n"
                    f"\n"
                    f"Current Correction:\n"
                    f"  Source:        {data.source.name}\n"
                    f"  Confidence:    {data.confidence:.1%}\n"
                    f"\n"
                    f"Full Status (JSON):\n{status_data}"
                )
            }]
        }

    except RuntimeError as e:
        return {
            "content": [{
                "type": "text",
                "text": (
                    f"❌ ChronoTick Daemon Status: NOT RUNNING\n"
                    f"{'='*50}\n"
                    f"{str(e)}"
                )
            }]
        }

    except Exception as e:
        logger.error(f"Unexpected error in get_daemon_status: {e}")
        return {
            "content": [{
                "type": "text",
                "text": f"❌ Unexpected error: {e}"
            }]
        }


@tool(
    name="get_time_with_future_uncertainty",
    description=(
        "Get corrected time with uncertainty projection for a future timestamp. "
        "Useful for planning actions that will occur in the future and need to account "
        "for increasing uncertainty over time due to clock drift."
    ),
    input_schema={
        "future_seconds": float  # Seconds into future (0-3600)
    }
)
async def get_time_with_future_uncertainty(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get time with uncertainty projection into the future.

    Args:
        args: Dict with 'future_seconds' key (seconds into future, 0-3600)

    Returns:
        Dict with current time, future time, and projected uncertainty

    Example response:
        {
          "current_corrected_time": 1704556800.123456,
          "current_uncertainty": 1.2e-05,
          "future_seconds": 60,
          "future_corrected_time": 1704556860.123456,
          "future_uncertainty": 0.00015,
          "uncertainty_increase": 0.000138
        }
    """
    try:
        call_start = time.time()

        # Validate input
        future_seconds = args.get("future_seconds", 0)
        if not isinstance(future_seconds, (int, float)):
            return {
                "content": [{
                    "type": "text",
                    "text": "❌ Error: future_seconds must be a number"
                }]
            }

        if future_seconds < 0 or future_seconds > 3600:
            return {
                "content": [{
                    "type": "text",
                    "text": "❌ Error: future_seconds must be between 0 and 3600 (1 hour)"
                }]
            }

        # Read from shared memory
        data = read_chronotick_data()

        # Calculate current time
        current_system_time = time.time()
        current_corrected_time = data.get_corrected_time_at(current_system_time)
        current_time_delta = current_system_time - data.prediction_time
        current_uncertainty = data.get_time_uncertainty(current_time_delta)

        # Project into future
        future_system_time = current_system_time + future_seconds
        future_corrected_time = data.get_corrected_time_at(future_system_time)
        future_time_delta = future_system_time - data.prediction_time
        future_uncertainty = data.get_time_uncertainty(future_time_delta)

        uncertainty_increase = future_uncertainty - current_uncertainty

        call_latency = (time.time() - call_start) * 1000

        response = {
            "current_corrected_time": current_corrected_time,
            "current_system_time": current_system_time,
            "current_uncertainty": current_uncertainty,
            "future_seconds": future_seconds,
            "future_corrected_time": future_corrected_time,
            "future_system_time": future_system_time,
            "future_uncertainty": future_uncertainty,
            "uncertainty_increase": uncertainty_increase,
            "confidence": data.confidence,
            "call_latency_ms": call_latency
        }

        return {
            "content": [{
                "type": "text",
                "text": (
                    f"ChronoTick Future Time Projection\n"
                    f"{'='*50}\n"
                    f"Current Time:\n"
                    f"  Corrected:     {current_corrected_time:.6f}\n"
                    f"  Uncertainty:   ±{current_uncertainty*1e6:.3f}μs\n"
                    f"\n"
                    f"Future Time (+{future_seconds}s):\n"
                    f"  Corrected:     {future_corrected_time:.6f}\n"
                    f"  Uncertainty:   ±{future_uncertainty*1e6:.3f}μs\n"
                    f"  Increase:      +{uncertainty_increase*1e6:.3f}μs\n"
                    f"  Confidence:    {data.confidence:.1%}\n"
                    f"\n"
                    f"Call Latency:    {call_latency:.4f}ms\n"
                    f"\n"
                    f"Full Data (JSON):\n{response}"
                )
            }]
        }

    except RuntimeError as e:
        return {
            "content": [{
                "type": "text",
                "text": (
                    f"❌ ChronoTick Error\n"
                    f"{'='*50}\n"
                    f"{str(e)}"
                )
            }]
        }

    except Exception as e:
        logger.error(f"Unexpected error in get_time_with_future_uncertainty: {e}")
        return {
            "content": [{
                "type": "text",
                "text": f"❌ Unexpected error: {e}"
            }]
        }


# ============================================================================
# Helper function for SDK server creation (optional convenience)
# ============================================================================

def create_chronotick_sdk_server():
    """
    Helper function to create ChronoTick SDK MCP server.

    Returns:
        SDK MCP server instance

    Usage:
        from chronotick_sdk_tools import create_chronotick_sdk_server

        sdk_server = create_chronotick_sdk_server()
    """
    try:
        from claude_agent_sdk import create_sdk_mcp_server
    except ImportError:
        raise ImportError(
            "claude-agent-sdk not found. Install with: pip install claude-agent-sdk"
        )

    return create_sdk_mcp_server(
        name="chronotick_server",
        version="1.0.0",
        tools=[get_time, get_daemon_status, get_time_with_future_uncertainty]
    )


if __name__ == "__main__":
    # Self-test
    print("ChronoTick SDK MCP Tools - Self Test")
    print("=" * 60)

    print("\nChecking claude-agent-sdk installation...")
    try:
        from claude_agent_sdk import create_sdk_mcp_server, tool as sdk_tool
        print("✓ claude-agent-sdk installed")
    except ImportError:
        print("✗ claude-agent-sdk NOT installed")
        print("  Install with: pip install claude-agent-sdk")
        sys.exit(1)

    print("\nChecking ChronoTick daemon...")
    try:
        shm = get_shared_memory()
        print(f"✓ Connected to shared memory: {SHARED_MEMORY_NAME}")

        data = read_chronotick_data()
        print(f"✓ Read data successfully")
        print(f"  - Valid: {data.is_valid}")
        print(f"  - NTP ready: {data.is_ntp_ready}")
        print(f"  - Uptime: {data.daemon_uptime:.1f}s")

    except RuntimeError as e:
        print(f"✗ Daemon not running:")
        print(f"  {e}")
        sys.exit(1)

    print("\n✅ All checks passed - SDK tools ready to use!")
    print("\nTools available:")
    print("  - get_time")
    print("  - get_daemon_status")
    print("  - get_time_with_future_uncertainty")
#!/usr/bin/env python3
"""
ChronoTick Agent Configuration

Example configuration for creating Claude agents with ChronoTick SDK MCP tools.
Demonstrates how to integrate ChronoTick time services into programmatic agents
using the claude-agent-sdk.

Usage:
    from create_chronotick_agent import create_chronotick_agent
    from claude_agent_sdk import ClaudeSDKClient

    # Create agent with ChronoTick tools
    agent_options = create_chronotick_agent()
    agent = ClaudeSDKClient(agent_options)

    # Use agent
    response = await agent.query("What is the current corrected time?")
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

try:
    from claude_agent_sdk import ClaudeAgentOptions, create_sdk_mcp_server
except ImportError:
    raise ImportError(
        "claude-agent-sdk not found. Install with: uv add claude-agent-sdk"
    )



def create_chronotick_agent(
    allowed_tools: Optional[List[str]] = None,
    server_key: str = "chronotick"
) -> ClaudeAgentOptions:
    """
    Create ClaudeAgentOptions with ChronoTick SDK MCP tools.

    Args:
        allowed_tools: List of tool names to allow. If None, allows all ChronoTick tools.
        server_key: Key for the MCP server in the mcp_servers dict. This becomes the
                   tool name prefix: mcp__{server_key}__{tool_name}

    Returns:
        ClaudeAgentOptions configured with ChronoTick tools

    Example:
        >>> agent_options = create_chronotick_agent()
        >>> agent = ClaudeSDKClient(agent_options)
        >>> response = await agent.query("What time is it?")

    Tool Naming:
        The tool names in allowed_tools MUST use the format: mcp__{server_key}__{tool_name}

        For example, with server_key="chronotick":
        - "mcp__chronotick__get_time"
        - "mcp__chronotick__get_daemon_status"
        - "mcp__chronotick__get_time_with_future_uncertainty"

        ⚠️  CRITICAL: Use the server_key (dict key), NOT the server name!
    """
    # Create SDK MCP server with ChronoTick tools
    sdk_server = create_sdk_mcp_server(
        name="chronotick_server",  # Server name (for logging only)
        version="1.0.0",
        tools=[
            get_time,
            get_daemon_status,
            get_time_with_future_uncertainty
        ]
    )

    # Default allowed tools (use server_key, not server name!)
    if allowed_tools is None:
        allowed_tools = [
            f"mcp__{server_key}__get_time",
            f"mcp__{server_key}__get_daemon_status",
            f"mcp__{server_key}__get_time_with_future_uncertainty"
        ]

    # Create agent options
    agent_options = ClaudeAgentOptions(
        # CRITICAL: The dict key becomes the tool name prefix, NOT the server name!
        mcp_servers={server_key: sdk_server},
        allowed_tools=allowed_tools
    )

    return agent_options


def create_minimal_agent() -> ClaudeAgentOptions:
    """
    Create minimal agent with only get_time tool.

    Useful for simple time-only applications.

    Returns:
        ClaudeAgentOptions with only get_time tool
    """
    sdk_server = create_sdk_mcp_server(
        name="chronotick_minimal",
        version="1.0.0",
        tools=[get_time]
    )

    return ClaudeAgentOptions(
        mcp_servers={"time": sdk_server},
        allowed_tools=["mcp__time__get_time"]
    )


def create_monitoring_agent() -> ClaudeAgentOptions:
    """
    Create monitoring agent with only daemon status tool.

    Useful for health monitoring and alerting applications.

    Returns:
        ClaudeAgentOptions with only daemon status tool
    """
    sdk_server = create_sdk_mcp_server(
        name="chronotick_monitoring",
        version="1.0.0",
        tools=[get_daemon_status]
    )

    return ClaudeAgentOptions(
        mcp_servers={"monitor": sdk_server},
        allowed_tools=["mcp__monitor__get_daemon_status"]
    )


def create_multi_service_agent(include_other_tools: bool = True) -> ClaudeAgentOptions:
    """
    Create agent with ChronoTick AND other MCP servers.

    Demonstrates how to combine ChronoTick with other services in a single agent.

    Args:
        include_other_tools: Whether to include example "other" tools in allowed list

    Returns:
        ClaudeAgentOptions with multiple MCP servers

    Example:
        This shows the pattern for adding ChronoTick alongside other MCP services
        like filesystem, git, web fetch, etc.
    """
    # ChronoTick server
    chronotick_server = create_sdk_mcp_server(
        name="chronotick_server",
        version="1.0.0",
        tools=[get_time, get_daemon_status, get_time_with_future_uncertainty]
    )

    # In a real application, you would add other MCP servers here:
    # - Filesystem MCP
    # - Git MCP
    # - Web fetch MCP
    # - Database MCP
    # - etc.

    mcp_servers = {
        "chronotick": chronotick_server,
        # "filesystem": filesystem_server,
        # "git": git_server,
        # etc.
    }

    allowed_tools = [
        # ChronoTick tools
        "mcp__chronotick__get_time",
        "mcp__chronotick__get_daemon_status",
        "mcp__chronotick__get_time_with_future_uncertainty",
    ]

    # Add other tools if requested
    if include_other_tools:
        # In a real app, add your other tool names here
        # allowed_tools.extend([
        #     "mcp__filesystem__read_file",
        #     "mcp__git__commit",
        #     # etc.
        # ])
        pass

    return ClaudeAgentOptions(
        mcp_servers=mcp_servers,
        allowed_tools=allowed_tools
    )


# ============================================================================
# Example Usage and Testing
# ============================================================================

async def example_usage():
    """
    Example of using ChronoTick with Claude Agent SDK.

    This demonstrates the complete workflow:
    1. Create agent options with ChronoTick tools
    2. Initialize agent
    3. Run queries that use ChronoTick
    """
    try:
        from claude_agent_sdk import ClaudeSDKClient
    except ImportError:
        print("❌ claude-agent-sdk not installed")
        print("   Install with: uv add claude-agent-sdk")
        return

    print("ChronoTick Agent Example")
    print("=" * 60)

    # Create agent
    print("\n1. Creating agent with ChronoTick tools...")
    agent_options = create_chronotick_agent()
    agent = ClaudeSDKClient(agent_options)
    print("   ✓ Agent created")

    # Example queries
    queries = [
        "What is the current corrected time with uncertainty?",
        "Show me the ChronoTick daemon status",
        "What will the time uncertainty be in 5 minutes?",
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n{i}. Query: {query}")
        print("-" * 60)

        try:
            response = await agent.query(query)
            print(response.text)
        except Exception as e:
            print(f"   ❌ Error: {e}")

    print("\n" + "=" * 60)
    print("Example complete!")


async def test_minimal_agent():
    """Test minimal agent with only get_time tool."""
    try:
        from claude_agent_sdk import ClaudeSDKClient
    except ImportError:
        print("❌ claude-agent-sdk not installed")
        return

    print("\nTesting Minimal Agent (time only)...")
    print("-" * 60)

    agent_options = create_minimal_agent()
    agent = ClaudeSDKClient(agent_options)

    response = await agent.query("What time is it?")
    print(response.text)


if __name__ == "__main__":
    import asyncio

    print("ChronoTick Agent Configuration Module")
    print("=" * 60)

    # Check dependencies
    print("\nChecking dependencies...")
    try:
        from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions
        print("✓ claude-agent-sdk installed")
    except ImportError:
        print("✗ claude-agent-sdk NOT installed")
        print("  Install with: uv add claude-agent-sdk")
        sys.exit(1)

    # Check ChronoTick daemon
    print("\nChecking ChronoTick daemon...")
    try:
        shm = get_shared_memory()
        print("✓ ChronoTick daemon running")
    except RuntimeError as e:
        print(f"✗ ChronoTick daemon NOT running")
        print(f"  {e}")
        sys.exit(1)

    print("\n✅ All checks passed!")
    print("\nAgent creation functions available:")
    print("  - create_chronotick_agent()       - Full ChronoTick tools")
    print("  - create_minimal_agent()          - Time only")
    print("  - create_monitoring_agent()       - Status monitoring only")
    print("  - create_multi_service_agent()    - ChronoTick + other services")

    print("\nRunning example usage...")
    asyncio.run(example_usage())
