#!/usr/bin/env python3
"""
ChronoTick SDK MCP Tools

MCP tools for claude-agent-sdk that provide high-precision time services
via shared memory IPC. These tools integrate with the ChronoTick daemon
for ultra-low latency time corrections (~300ns read latency).

This module contains ONLY the @tool decorated functions.
Agent configuration is in agents.py.

Features:
- get_time: Get corrected time with uncertainty bounds
- get_daemon_status: Monitor daemon health and performance
- get_time_with_future_uncertainty: Project uncertainty into future

Performance:
- First call: ~1.5ms (shared memory attachment + read)
- Subsequent calls: ~0.0003ms (300ns) - 5000x faster
- Zero serialization overhead
- Lock-free reads (no contention)

Architecture (from sdk-mcp/GUIDE_SDK_MCP_SHARED_MEMORY.md):
    Background Daemon → Shared Memory (128 bytes) → SDK MCP Tools
    (100 Hz updates)    (Lock-free read/write)      (~300ns reads)
"""

import sys
import time
import logging
from typing import Dict, Any, Optional
from multiprocessing.shared_memory import SharedMemory

# NOTE: claude_agent_sdk must be installed: pip install claude-agent-sdk
try:
    from claude_agent_sdk import tool
except ImportError:
    raise ImportError(
        "claude-agent-sdk not found. Install with: uv add claude-agent-sdk"
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
#
# From GUIDE_SDK_MCP_SHARED_MEMORY.md section C3:
# "Global handle amortizes attachment cost to zero"

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
                f"  chronotick-daemon\n\n"
                f"Or with custom config:\n"
                f"  chronotick-daemon --config /path/to/config.yaml"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to attach to shared memory: {e}")

    return _shared_memory_handle


def read_chronotick_data() -> ChronoTickData:
    """
    Read ChronoTickData from shared memory with retry logic.

    Uses lock-free read with sequence number checking (from guide section B5).

    Returns:
        ChronoTickData from shared memory

    Raises:
        RuntimeError: If daemon not running or data read fails
    """
    shm = get_shared_memory()

    try:
        # Read with automatic retry on torn reads
        # Implements sequence number pattern from guide
        data = read_data_with_retry(shm.buf, max_retries=3)
        return data
    except Exception as e:
        raise RuntimeError(f"Failed to read ChronoTick data: {e}")


# ============================================================================
# MCP Tools (@tool decorated functions)
# ============================================================================
# From guide section A2:
# - Function MUST be `async def`
# - Decorated with `@tool()` decorator
# - Takes `args: Dict[str, Any]` parameter
# - Returns `Dict[str, Any]` with `"content"` key
# - Content is list of blocks with `"type"` and type-specific fields

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
        Dict with "content" key containing formatted time data

    Example response format (from guide section A2):
        {
          "content": [{
            "type": "text",
            "text": "ChronoTick Corrected Time\\n..."
          }]
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

        # Build JSON data for programmatic access
        json_data = {
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

        # Format human-readable text
        text_output = (
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
            f"\nJSON Data:\n{json_data}"
        )

        # CRITICAL: Must return this exact structure (guide section A2)
        return {
            "content": [{
                "type": "text",
                "text": text_output
            }]
        }

    except RuntimeError as e:
        # Daemon not running - return helpful error (guide section C5)
        # From guide: "Tools should NEVER raise exceptions. Always return error as text content."
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
        Dict with "content" key containing daemon status information
    """
    try:
        call_start = time.time()

        # Read from shared memory
        data = read_chronotick_data()

        # Get process info (optional - requires psutil)
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            cpu_affinity = process.cpu_affinity()
        except:
            memory_mb = None
            cpu_affinity = None

        call_latency = (time.time() - call_start) * 1000

        # Build status data
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

        # Format text output
        text_output = (
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
            f"\nJSON Data:\n{status_data}"
        )

        return {
            "content": [{
                "type": "text",
                "text": text_output
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
        Dict with "content" key containing current and future time data
    """
    try:
        call_start = time.time()

        # Validate input (from guide section A4)
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

        # Build JSON response
        json_response = {
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

        # Format text output
        text_output = (
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
            f"\nJSON Data:\n{json_response}"
        )

        return {
            "content": [{
                "type": "text",
                "text": text_output
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
# Self-Test (only runs when executed directly)
# ============================================================================

if __name__ == "__main__":
    # Self-test
    print("ChronoTick SDK MCP Tools - Self Test")
    print("=" * 60)

    print("\nChecking claude-agent-sdk installation...")
    try:
        from claude_agent_sdk import create_sdk_mcp_server
        print("✓ claude-agent-sdk installed")
    except ImportError:
        print("✗ claude-agent-sdk NOT installed")
        print("  Install with: uv add claude-agent-sdk")
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
    print("\nTo create an agent with these tools, use:")
    print("  from chronotick_shm.tools.agents import create_chronotick_agent")
