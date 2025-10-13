#!/usr/bin/env python3
"""
ChronoTick Agent Configuration

Agent configuration for creating Claude agents with ChronoTick SDK MCP tools.
Demonstrates how to integrate ChronoTick time services into programmatic agents
using the claude-agent-sdk.

This module follows the pattern from sdk-mcp/GUIDE_SDK_MCP_SHARED_MEMORY.md Section A3:
1. Import tool functions
2. Create SDK MCP server with create_sdk_mcp_server()
3. Configure ClaudeAgentOptions with correct naming

CRITICAL naming pattern (from guide section A3):
    mcp_servers = {"my_key": server}  # Dict key is "my_key"
                    ^^^^^^^^
                       ↓
    allowed_tools = ["mcp__my_key__tool_name"]  # Tool name uses dict key
                          ^^^^^^^^

Usage:
    from chronotick_shm.tools.agents import create_chronotick_agent
    from claude_agent_sdk import ClaudeSDKClient

    # Create agent with ChronoTick tools
    agent_options = create_chronotick_agent()
    agent = ClaudeSDKClient(agent_options)

    # Use agent
    response = await agent.query("What is the current corrected time?")
"""

import sys
from typing import List, Optional

try:
    from claude_agent_sdk import ClaudeAgentOptions, create_sdk_mcp_server
except ImportError:
    raise ImportError(
        "claude-agent-sdk not found. Install with: uv add claude-agent-sdk"
    )

# Import tool functions from sdk_tools module
from chronotick_shm.tools.sdk_tools import (
    get_time,
    get_daemon_status,
    get_time_with_future_uncertainty
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

    Tool Naming (CRITICAL from guide section A3):
        The tool names in allowed_tools MUST use the format: mcp__{server_key}__{tool_name}

        For example, with server_key="chronotick":
        - "mcp__chronotick__get_time"
        - "mcp__chronotick__get_daemon_status"
        - "mcp__chronotick__get_time_with_future_uncertainty"

        ⚠️  CRITICAL: Use the server_key (dict key), NOT the server name!
    """
    # Create SDK MCP server with ChronoTick tools
    # From guide section A5: "Create server with multiple tools"
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
    # From guide section A3: "CRITICAL: Dict key becomes tool name prefix, NOT the server name!"
    if allowed_tools is None:
        allowed_tools = [
            f"mcp__{server_key}__get_time",
            f"mcp__{server_key}__get_daemon_status",
            f"mcp__{server_key}__get_time_with_future_uncertainty"
        ]

    # Create agent options
    # CRITICAL (from guide): The dict key becomes the tool name prefix, NOT the server name!
    agent_options = ClaudeAgentOptions(
        mcp_servers={server_key: sdk_server},  # Key "chronotick" → "mcp__chronotick__*"
        allowed_tools=allowed_tools
    )

    return agent_options


def create_minimal_agent() -> ClaudeAgentOptions:
    """
    Create minimal agent with only get_time tool.

    Useful for simple time-only applications.

    Returns:
        ClaudeAgentOptions with only get_time tool

    Example:
        >>> agent_options = create_minimal_agent()
        >>> agent = ClaudeSDKClient(agent_options)
        >>> response = await agent.query("What is the current time?")
    """
    sdk_server = create_sdk_mcp_server(
        name="chronotick_minimal",
        version="1.0.0",
        tools=[get_time]
    )

    return ClaudeAgentOptions(
        mcp_servers={"time": sdk_server},  # Key "time" → "mcp__time__*"
        allowed_tools=["mcp__time__get_time"]  # Uses dict key "time"
    )


def create_monitoring_agent() -> ClaudeAgentOptions:
    """
    Create monitoring agent with only daemon status tool.

    Useful for health monitoring and alerting applications.

    Returns:
        ClaudeAgentOptions with only daemon status tool

    Example:
        >>> agent_options = create_monitoring_agent()
        >>> agent = ClaudeSDKClient(agent_options)
        >>> response = await agent.query("Check the daemon status")
    """
    sdk_server = create_sdk_mcp_server(
        name="chronotick_monitoring",
        version="1.0.0",
        tools=[get_daemon_status]
    )

    return ClaudeAgentOptions(
        mcp_servers={"monitor": sdk_server},  # Key "monitor" → "mcp__monitor__*"
        allowed_tools=["mcp__monitor__get_daemon_status"]  # Uses dict key "monitor"
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
        >>> agent_options = create_multi_service_agent()
        >>> agent = ClaudeSDKClient(agent_options)

    Pattern:
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
        # ChronoTick tools (using dict key "chronotick")
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

    From guide section "Complete Implementation Example".
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


# ============================================================================
# Self-Test (only runs when executed directly)
# ============================================================================

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
        from chronotick_shm.tools.sdk_tools import get_shared_memory
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
