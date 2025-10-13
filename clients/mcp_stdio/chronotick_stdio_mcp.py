#!/usr/bin/env python3
"""
ChronoTick Stdio MCP Server

Standalone MCP server that uses shared memory for ultra-low latency time services.
This server can be connected to from Claude Code or other MCP clients via stdio.

Built with fastmcp for easy server creation.

Usage:
    python -m chronotick_shm.chronotick_stdio_mcp.server

    # Or with debug logging
    python -m chronotick_shm.chronotick_stdio_mcp.server --debug

Connect from Claude Code:
    Add to ~/.claude/config.json:
    {
      "mcpServers": {
        "chronotick": {
          "command": "uv",
          "args": ["run", "chronotick-stdio-server"]
        }
      }
    }

Performance:
    - Read latency: ~300ns (after first call)
    - First call: ~1.5ms (shared memory attachment)
    - No queue overhead, no serialization
    - 5000x faster than queue-based approach
"""

import sys
import time
import logging
from pathlib import Path

try:
    from fastmcp import FastMCP
except ImportError:
    print("ERROR: fastmcp not installed", file=sys.stderr)
    print("Install with: uv add fastmcp", file=sys.stderr)
    sys.exit(1)

from chronotick_shm.shm_config import SHARED_MEMORY_NAME
from chronotick_shm.chronotick_sdk_mcp import (
    get_shared_memory,
    read_chronotick_data,
)

logger = logging.getLogger(__name__)

# Create FastMCP server
mcp = FastMCP("ChronoTick Time Server")


@mcp.tool()
def get_time() -> dict:
    """
    Get high-precision corrected time with uncertainty bounds from ChronoTick.

    Returns corrected timestamp, uncertainty estimates, and metadata about the correction.
    Use this when you need accurate time with error bounds for coordination or timestamping.

    Returns:
        Dictionary containing:
        - corrected_time: Corrected Unix timestamp
        - system_time: Raw system timestamp
        - offset_correction: Time offset in seconds
        - drift_rate: Clock drift rate (seconds/second)
        - offset_uncertainty: Uncertainty in offset (seconds)
        - drift_uncertainty: Uncertainty in drift rate
        - time_uncertainty: Total time uncertainty at current moment (seconds)
        - confidence: Confidence level (0-1)
        - source: Data source (ntp, cpu_model, gpu_model, fusion)
        - prediction_time: When prediction was made
        - valid_until: When prediction expires
        - is_valid: Whether data is currently valid
        - is_ntp_ready: Whether NTP is ready
        - is_models_ready: Whether ML models are ready
        - call_latency_ms: Latency of this call (milliseconds)
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

        return {
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
            "call_latency_ms": call_latency,
        }

    except RuntimeError as e:
        return {"error": str(e), "daemon_running": False}
    except Exception as e:
        logger.error(f"Unexpected error in get_time: {e}")
        return {"error": f"Unexpected error: {e}"}


@mcp.tool()
def get_daemon_status() -> dict:
    """
    Get ChronoTick daemon status, health metrics, and performance statistics.

    Use this to monitor daemon health, check warmup progress, or diagnose issues.

    Returns:
        Dictionary containing:
        - status: Daemon status ("ready" or "warming_up")
        - warmup_complete: Whether warmup is complete
        - measurement_count: Number of NTP measurements collected
        - total_corrections: Total number of corrections made
        - daemon_uptime: Daemon uptime in seconds
        - last_ntp_sync: Timestamp of last NTP sync
        - seconds_since_ntp: Seconds since last NTP sync
        - ntp_ready: Whether NTP is ready
        - models_ready: Whether ML models are ready
        - average_latency_ms: Average read latency (milliseconds)
        - data_source: Current correction source
        - confidence: Confidence level (0-1)
        - call_latency_ms: Latency of this call (milliseconds)
    """
    try:
        call_start = time.time()

        # Read from shared memory
        data = read_chronotick_data()

        call_latency = (time.time() - call_start) * 1000

        return {
            "status": "ready" if data.is_valid and data.is_warmup_complete else "warming_up",
            "warmup_complete": data.is_warmup_complete,
            "measurement_count": data.measurement_count,
            "total_corrections": data.total_corrections,
            "daemon_uptime": data.daemon_uptime,
            "last_ntp_sync": data.last_ntp_sync,
            "seconds_since_ntp": (
                time.time() - data.last_ntp_sync if data.last_ntp_sync > 0 else None
            ),
            "ntp_ready": data.is_ntp_ready,
            "models_ready": data.is_models_ready,
            "average_latency_ms": data.average_latency_ms,
            "data_source": data.source.name.lower(),
            "confidence": data.confidence,
            "call_latency_ms": call_latency,
        }

    except RuntimeError as e:
        return {
            "error": str(e),
            "daemon_running": False,
            "status": "not_running",
        }
    except Exception as e:
        logger.error(f"Unexpected error in get_daemon_status: {e}")
        return {"error": f"Unexpected error: {e}"}


@mcp.tool()
def get_time_with_future_uncertainty(future_seconds: float) -> dict:
    """
    Get corrected time with uncertainty projection for a future timestamp.

    Useful for planning actions that will occur in the future and need to account
    for increasing uncertainty over time due to clock drift.

    Args:
        future_seconds: Seconds into the future (0-3600)

    Returns:
        Dictionary containing:
        - current_corrected_time: Current corrected timestamp
        - current_system_time: Current system timestamp
        - current_uncertainty: Current time uncertainty (seconds)
        - future_seconds: Requested future time offset
        - future_corrected_time: Future corrected timestamp
        - future_system_time: Future system timestamp
        - future_uncertainty: Projected future uncertainty (seconds)
        - uncertainty_increase: Increase in uncertainty (seconds)
        - confidence: Confidence level (0-1)
        - call_latency_ms: Latency of this call (milliseconds)
    """
    try:
        # Validate input
        if not isinstance(future_seconds, (int, float)):
            return {"error": "future_seconds must be a number"}

        if future_seconds < 0 or future_seconds > 3600:
            return {"error": "future_seconds must be between 0 and 3600 (1 hour)"}

        call_start = time.time()

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

        return {
            "current_corrected_time": current_corrected_time,
            "current_system_time": current_system_time,
            "current_uncertainty": current_uncertainty,
            "future_seconds": future_seconds,
            "future_corrected_time": future_corrected_time,
            "future_system_time": future_system_time,
            "future_uncertainty": future_uncertainty,
            "uncertainty_increase": uncertainty_increase,
            "confidence": data.confidence,
            "call_latency_ms": call_latency,
        }

    except RuntimeError as e:
        return {"error": str(e), "daemon_running": False}
    except Exception as e:
        logger.error(f"Unexpected error in get_time_with_future_uncertainty: {e}")
        return {"error": f"Unexpected error: {e}"}


def main():
    """Main entry point for stdio server"""
    import argparse

    parser = argparse.ArgumentParser(
        description="ChronoTick Stdio MCP Server - Ultra-low latency time via shared memory"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],  # Log to stderr, not stdout
    )

    # Check daemon connection
    logger.info("ChronoTick Stdio MCP Server starting...")
    try:
        shm = get_shared_memory()
        data = read_chronotick_data()
        logger.info(f"✅ Connected to ChronoTick daemon (uptime: {data.daemon_uptime:.1f}s)")
    except RuntimeError as e:
        logger.warning(f"⚠️  ChronoTick daemon not running: {e}")
        logger.warning("   Tools will return errors until daemon is started")
        logger.warning("   Start daemon with: uv run chronotick-daemon")

    logger.info("🚀 ChronoTick Stdio MCP Server ready")

    # Run server
    mcp.run()


if __name__ == "__main__":
    main()
