#!/usr/bin/env python3
"""
ChronoTick MCP Server Entry Point

This script provides an easy way to start the ChronoTick MCP server
that provides high-precision time services to AI agents.

Usage:
    python chronotick_mcp.py [--config CONFIG_PATH] [--log-level LEVEL]

The server will:
1. Start the ChronoTick daemon with real data pipeline
2. Wait for warmup completion (NTP measurements)
3. Accept MCP connections from AI agents
4. Provide fast time corrections with uncertainty bounds
"""

import sys
import logging
from pathlib import Path

# Add the project to Python path
sys.path.insert(0, str(Path(__file__).parent))

from chronotick_inference.mcp_server import cli_main

if __name__ == "__main__":
    cli_main()