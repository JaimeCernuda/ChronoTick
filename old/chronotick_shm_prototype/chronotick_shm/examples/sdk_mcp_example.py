#!/usr/bin/env python3
"""
ChronoTick SDK MCP Example

Demonstrates using claude-agent-sdk to create an agent with ChronoTick tools
that read from shared memory (as described in GUIDE_SDK_MCP_SHARED_MEMORY.md).

This is the "SDK MCP" approach - tools run in-process with the agent for
ultra-low latency (~300ns reads after first connection).

Usage:
    # Make sure daemon is running first
    cd ../
    python3 chronotick_daemon.py

    # Then run this example
    python3 examples/sdk_mcp_example.py

Requirements:
    pip install claude-agent-sdk
"""

import sys
import asyncio
from pathlib import Path

try:
    from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, create_sdk_mcp_server
except ImportError:
    print("ERROR: claude-agent-sdk not installed")
    print("Install with: uv add claude-agent-sdk")
    sys.exit(1)

from chronotick_shm.tools.chronotick_sdk_tools import (
    get_time,
    get_daemon_status,
    get_time_with_future_uncertainty
)


async def example_basic_usage():
    """Basic example: Create agent and query time"""
    print("=" * 70)
    print("Example 1: Basic Time Query")
    print("=" * 70)

    # Step 1: Create SDK MCP server with ChronoTick tools
    sdk_server = create_sdk_mcp_server(
        name="chronotick_server",  # Server name (for logging)
        version="1.0.0",
        tools=[get_time, get_daemon_status, get_time_with_future_uncertainty]
    )

    # Step 2: Create agent options
    # CRITICAL: Use dict key "chronotick" in tool names, NOT server name
    agent_options = ClaudeAgentOptions(
        mcp_servers={"chronotick": sdk_server},
        allowed_tools=[
            "mcp__chronotick__get_time",
            "mcp__chronotick__get_daemon_status",
            "mcp__chronotick__get_time_with_future_uncertainty"
        ]
    )

    # Step 3: Create agent
    agent = ClaudeSDKClient(agent_options)

    # Step 4: Query time
    response = await agent.query("What is the current corrected time with uncertainty?")
    print("\nAgent Response:")
    print(response.text)


async def example_performance_test():
    """Performance example: Measure SDK MCP latency"""
    print("\n" + "=" * 70)
    print("Example 2: Performance Test")
    print("=" * 70)

    # Create agent (same as above)
    sdk_server = create_sdk_mcp_server(
        name="chronotick_server",
        version="1.0.0",
        tools=[get_time]
    )

    agent_options = ClaudeAgentOptions(
        mcp_servers={"chronotick": sdk_server},
        allowed_tools=["mcp__chronotick__get_time"]
    )

    agent = ClaudeSDKClient(agent_options)

    # Warm up (first call attaches to shared memory, ~1.5ms)
    print("\nWarmup call (attaches to shared memory)...")
    await agent.query("What time is it?")

    # Performance test - multiple calls
    print("\nPerformance test (10 calls)...")
    import time

    times = []
    for i in range(10):
        start = time.perf_counter()
        await agent.query("What time is it?")
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Call {i+1}: {elapsed*1000:.2f}ms")

    avg_time = sum(times) / len(times)
    print(f"\nAverage time: {avg_time*1000:.2f}ms")
    print(f"(Includes agent overhead + shared memory read ~0.3μs)")


async def example_daemon_monitoring():
    """Monitoring example: Check daemon health"""
    print("\n" + "=" * 70)
    print("Example 3: Daemon Status Monitoring")
    print("=" * 70)

    sdk_server = create_sdk_mcp_server(
        name="chronotick_server",
        version="1.0.0",
        tools=[get_daemon_status]
    )

    agent_options = ClaudeAgentOptions(
        mcp_servers={"monitor": sdk_server},
        allowed_tools=["mcp__monitor__get_daemon_status"]
    )

    agent = ClaudeSDKClient(agent_options)

    response = await agent.query("Show me the ChronoTick daemon status and health metrics")
    print("\nAgent Response:")
    print(response.text)


async def example_future_uncertainty():
    """Future projection example: Planning with uncertainty"""
    print("\n" + "=" * 70)
    print("Example 4: Future Uncertainty Projection")
    print("=" * 70)

    sdk_server = create_sdk_mcp_server(
        name="chronotick_server",
        version="1.0.0",
        tools=[get_time_with_future_uncertainty]
    )

    agent_options = ClaudeAgentOptions(
        mcp_servers={"chronotick": sdk_server},
        allowed_tools=["mcp__chronotick__get_time_with_future_uncertainty"]
    )

    agent = ClaudeSDKClient(agent_options)

    response = await agent.query(
        "I need to coordinate with another agent in 5 minutes (300 seconds). "
        "What will the time uncertainty be?"
    )
    print("\nAgent Response:")
    print(response.text)


async def example_direct_shared_memory():
    """Direct access example: Bypassing agent for maximum performance"""
    print("\n" + "=" * 70)
    print("Example 5: Direct Shared Memory Access (No Agent)")
    print("=" * 70)

    from chronotick_shm.tools.chronotick_sdk_tools import get_shared_memory, read_chronotick_data
    import time

    print("\nDirect shared memory access (bypasses agent entirely)...")

    # Connect to shared memory (once)
    shm = get_shared_memory()
    print(f"✓ Connected to shared memory")

    # Read multiple times and measure
    print("\nReading 10 times...")
    times = []

    for i in range(10):
        start = time.perf_counter()
        data = read_chronotick_data()
        corrected_time = data.get_corrected_time_at(time.time())
        elapsed = time.perf_counter() - start
        times.append(elapsed)

        print(f"  Read {i+1}: {elapsed*1e6:.2f}μs (time: {corrected_time:.6f})")

    avg_time = sum(times) / len(times)
    print(f"\nAverage read time: {avg_time*1e6:.2f}μs")
    print(f"This is the raw shared memory read performance (~300ns expected)")


async def main():
    """Run all examples"""
    print("\n" + "=" * 70)
    print("ChronoTick SDK MCP Examples")
    print("=" * 70)
    print("\nThis demonstrates using claude-agent-sdk with shared memory")
    print("as described in GUIDE_SDK_MCP_SHARED_MEMORY.md")
    print()

    # Check if daemon is running
    try:
        from chronotick_shm.tools.chronotick_sdk_tools import get_shared_memory
        shm = get_shared_memory()
        print("✓ ChronoTick daemon is running")
    except RuntimeError as e:
        print(f"✗ ChronoTick daemon NOT running")
        print(f"  {e}")
        print("\nPlease start the daemon first:")
        print("  cd ..")
        print("  python3 chronotick_daemon.py --config ../tsfm/chronotick_inference/config.yaml")
        return

    # Run examples
    try:
        await example_basic_usage()
        await example_performance_test()
        await example_daemon_monitoring()
        await example_future_uncertainty()
        await example_direct_shared_memory()

        print("\n" + "=" * 70)
        print("All Examples Complete!")
        print("=" * 70)

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
