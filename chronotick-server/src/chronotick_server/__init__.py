"""ChronoTick: High-precision MCP time server for geo-distributed agent synchronization."""

import asyncio
import sys

from .server import serve


def main():
    """Main entry point for ChronoTick server."""
    import argparse
    from typing import Optional
    
    parser = argparse.ArgumentParser(
        description="ChronoTick: High-precision MCP time server"
    )
    parser.add_argument(
        "--node-id",
        type=str,
        help="Unique node identifier for this instance"
    )
    parser.add_argument(
        "--local-timezone",
        type=str,
        help="Local timezone override (IANA timezone name)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    import logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    try:
        asyncio.run(serve(args.node_id, args.local_timezone))
    except KeyboardInterrupt:
        print("ChronoTick server stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"ChronoTick server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
