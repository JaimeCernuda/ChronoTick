#!/usr/bin/env python3
"""
ChronoTick SDK MCP Agent Example

This example demonstrates how to create and use a Claude agent with ChronoTick
SDK MCP tools. The agent can query high-precision corrected time from the
ChronoTick daemon via shared memory.

Architecture:
    Agent → SDK MCP Tools → Shared Memory (~300ns) → ChronoTick Daemon

Based on: sdk-mcp/GUIDE_SDK_MCP_SHARED_MEMORY.md

Requirements:
    - claude-agent-sdk: uv add claude-agent-sdk
    - ChronoTick daemon must be running: chronotick-daemon

Run:
    python examples/sdk_agent_example.py

Features Demonstrated:
    1. Creating agent with ChronoTick SDK MCP tools
    2. Querying corrected time with uncertainty
    3. Monitoring daemon status
    4. Projecting future uncertainty
    5. Understanding tool naming conventions
"""

import asyncio
import sys
from pathlib import Path

# ============================================================================
# Check Dependencies
# ============================================================================

try:
    from claude_agent_sdk import ClaudeSDKClient
except ImportError:
    print("❌ claude-agent-sdk not installed")
    print("   Install with: uv add claude-agent-sdk")
    sys.exit(1)

try:
    from chronotick_shm.tools.agents import (
        create_chronotick_agent,
        create_minimal_agent,
        create_monitoring_agent
    )
except ImportError as e:
    print(f"❌ ChronoTick tools not found: {e}")
    print("   Make sure you're in the chronotick_shm directory")
    sys.exit(1)


# ============================================================================
# Example 1: Full Agent with All Tools
# ============================================================================

async def example_full_agent():
    """
    Example 1: Create agent with all ChronoTick tools.

    This is the most common use case - agent can access all time services.
    """
    print("\n" + "="*70)
    print("Example 1: Full Agent with All ChronoTick Tools")
    print("="*70)

    # Create agent with all ChronoTick tools
    # From guide section A3: Create agent with SDK MCP server
    print("\n1. Creating agent...")
    agent_options = create_chronotick_agent()
    agent = ClaudeSDKClient(agent_options)
    print("   ✓ Agent created with ChronoTick tools")

    # Test queries
    queries = [
        "What is the current corrected time? Show me the uncertainty.",
        "Give me a summary of the daemon status.",
        "What will the time uncertainty be in 5 minutes from now?",
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n{i}. Query: {query}")
        print("-" * 70)

        try:
            response = await agent.query(query)
            print(response.text)
        except Exception as e:
            print(f"   ❌ Error: {e}")

    print("\n" + "="*70)
    print("Example 1 Complete!\n")


# ============================================================================
# Example 2: Minimal Agent (Time Only)
# ============================================================================

async def example_minimal_agent():
    """
    Example 2: Create minimal agent with only time retrieval.

    Useful when you only need basic time services, not monitoring.
    """
    print("\n" + "="*70)
    print("Example 2: Minimal Agent (Time Only)")
    print("="*70)

    print("\n1. Creating minimal agent...")
    agent_options = create_minimal_agent()
    agent = ClaudeSDKClient(agent_options)
    print("   ✓ Minimal agent created (get_time only)")

    # This agent can only use get_time tool
    # From guide section A3: Tool names use dict key, not server name
    print("\n2. Available tools:")
    print("   - mcp__time__get_time (uses dict key 'time')")

    query = "What time is it right now?"
    print(f"\n3. Query: {query}")
    print("-" * 70)

    try:
        response = await agent.query(query)
        print(response.text)
    except Exception as e:
        print(f"   ❌ Error: {e}")

    print("\n" + "="*70)
    print("Example 2 Complete!\n")


# ============================================================================
# Example 3: Monitoring Agent (Status Only)
# ============================================================================

async def example_monitoring_agent():
    """
    Example 3: Create monitoring agent for health checks.

    Useful for alerting and monitoring systems that only need daemon status.
    """
    print("\n" + "="*70)
    print("Example 3: Monitoring Agent (Status Only)")
    print("="*70)

    print("\n1. Creating monitoring agent...")
    agent_options = create_monitoring_agent()
    agent = ClaudeSDKClient(agent_options)
    print("   ✓ Monitoring agent created (daemon_status only)")

    # This agent can only check daemon status
    print("\n2. Available tools:")
    print("   - mcp__monitor__get_daemon_status (uses dict key 'monitor')")

    query = "Check if the ChronoTick daemon is healthy and report key metrics."
    print(f"\n3. Query: {query}")
    print("-" * 70)

    try:
        response = await agent.query(query)
        print(response.text)
    except Exception as e:
        print(f"   ❌ Error: {e}")

    print("\n" + "="*70)
    print("Example 3 Complete!\n")


# ============================================================================
# Example 4: Custom Tool Selection
# ============================================================================

async def example_custom_tools():
    """
    Example 4: Create agent with custom tool selection.

    Demonstrates how to selectively enable only certain tools.
    """
    print("\n" + "="*70)
    print("Example 4: Custom Tool Selection")
    print("="*70)

    # Create agent with only get_time and get_time_with_future_uncertainty
    # (excluding get_daemon_status)
    print("\n1. Creating agent with custom tool selection...")

    # From guide section A3: CRITICAL naming pattern
    custom_tools = [
        "mcp__chronotick__get_time",                      # Include
        "mcp__chronotick__get_time_with_future_uncertainty"  # Include
        # "mcp__chronotick__get_daemon_status",          # Exclude
    ]

    agent_options = create_chronotick_agent(allowed_tools=custom_tools)
    agent = ClaudeSDKClient(agent_options)
    print("   ✓ Agent created with custom tools")

    print("\n2. Available tools:")
    for tool in custom_tools:
        print(f"   - {tool}")
    print("   (daemon_status excluded)")

    query = "What is the time and what will the uncertainty be in 2 minutes?"
    print(f"\n3. Query: {query}")
    print("-" * 70)

    try:
        response = await agent.query(query)
        print(response.text)
    except Exception as e:
        print(f"   ❌ Error: {e}")

    print("\n" + "="*70)
    print("Example 4 Complete!\n")


# ============================================================================
# Example 5: Understanding Tool Naming
# ============================================================================

def example_tool_naming():
    """
    Example 5: Explain MCP tool naming conventions.

    CRITICAL concept from guide section A3 that often causes confusion.
    """
    print("\n" + "="*70)
    print("Example 5: Understanding Tool Naming")
    print("="*70)

    print("""
Tool Naming Pattern (from sdk-mcp/GUIDE_SDK_MCP_SHARED_MEMORY.md):

    mcp_servers = {"my_key": server}  # Dict key is "my_key"
                    ^^^^^^^^
                       ↓
    allowed_tools = ["mcp__my_key__tool_name"]  # Tool name uses dict key
                          ^^^^^^^^

CORRECT Example:
    sdk_server = create_sdk_mcp_server(name="chronotick_server", ...)
    mcp_servers = {"chronotick": sdk_server}  # Dict key: "chronotick"
    allowed_tools = ["mcp__chronotick__get_time"]  # Uses dict key ✓

WRONG Example:
    sdk_server = create_sdk_mcp_server(name="chronotick_server", ...)
    mcp_servers = {"chronotick": sdk_server}  # Dict key: "chronotick"
    allowed_tools = ["mcp__chronotick_server__get_time"]  # Uses server name ✗

Key Points:
1. Server name (in create_sdk_mcp_server) is for logging only
2. Dict key (in mcp_servers) becomes the tool name prefix
3. Tool names MUST use the dict key, not the server name
4. Format: mcp__{dict_key}__{tool_name}

ChronoTick Tool Names:
- mcp__chronotick__get_time                       (full agent)
- mcp__chronotick__get_daemon_status              (full agent)
- mcp__chronotick__get_time_with_future_uncertainty (full agent)
- mcp__time__get_time                             (minimal agent)
- mcp__monitor__get_daemon_status                 (monitoring agent)
""")

    print("="*70)
    print("Example 5 Complete!\n")


# ============================================================================
# Example 6: Error Handling
# ============================================================================

async def example_error_handling():
    """
    Example 6: Demonstrate graceful error handling.

    Shows what happens when daemon is not running or tools fail.
    """
    print("\n" + "="*70)
    print("Example 6: Error Handling")
    print("="*70)

    print("\n1. Creating agent...")
    agent_options = create_chronotick_agent()
    agent = ClaudeSDKClient(agent_options)
    print("   ✓ Agent created")

    # Try to query even if daemon might not be running
    # From guide section C5: "Tools should NEVER raise exceptions"
    query = "What is the current time?"
    print(f"\n2. Query: {query}")
    print("-" * 70)

    try:
        response = await agent.query(query)
        print(response.text)

        # Check if error is returned in text
        if "Error" in response.text or "❌" in response.text:
            print("\n⚠️  Note: Daemon appears to be not running or encountered an error")
            print("   The tool returned an error message instead of raising an exception")
            print("   This is correct behavior per the SDK MCP guide")
    except Exception as e:
        print(f"   ❌ Unexpected exception: {e}")
        print("   Note: SDK tools should return errors as text, not raise exceptions")

    print("\n" + "="*70)
    print("Example 6 Complete!\n")


# ============================================================================
# Main Function
# ============================================================================

async def main():
    """Run all examples."""
    print("="*70)
    print("ChronoTick SDK MCP Agent Examples")
    print("="*70)
    print("\nBased on: sdk-mcp/GUIDE_SDK_MCP_SHARED_MEMORY.md")
    print("\nThis demonstrates:")
    print("- Creating agents with SDK MCP tools")
    print("- Different agent configurations (full, minimal, monitoring)")
    print("- Tool naming conventions")
    print("- Error handling")

    # Check if daemon is running (warn but continue)
    try:
        from chronotick_shm.tools.sdk_tools import get_shared_memory
        shm = get_shared_memory()
        print("\n✓ ChronoTick daemon is running")
    except RuntimeError as e:
        print(f"\n⚠️  Warning: ChronoTick daemon not running")
        print(f"   {e}")
        print("\n   Examples will demonstrate error handling")

    # Run examples
    try:
        await example_full_agent()
        await example_minimal_agent()
        await example_monitoring_agent()
        await example_custom_tools()
        example_tool_naming()  # Synchronous
        await example_error_handling()

    except KeyboardInterrupt:
        print("\n\n⏹  Interrupted by user")
        return

    print("\n" + "="*70)
    print("✅ All Examples Complete!")
    print("="*70)
    print("\nNext Steps:")
    print("1. Read sdk-mcp/GUIDE_SDK_MCP_SHARED_MEMORY.md for detailed guide")
    print("2. Check chronotick_shm/tools/sdk_tools.py for tool definitions")
    print("3. Check chronotick_shm/tools/agents.py for agent configuration")
    print("4. Try creating your own agent with custom tool selection")
    print("\nFor production use:")
    print("- Start daemon: chronotick-daemon")
    print("- Import tools: from chronotick_shm.tools.agents import create_chronotick_agent")
    print("- Create agent: agent_options = create_chronotick_agent()")
    print("- Use agent: agent = ClaudeSDKClient(agent_options)")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⏹  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
