"""Enhanced MCP time server with high-precision capabilities."""

import json
import logging
from datetime import datetime
from enum import Enum
from typing import Sequence, Dict, Any, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource
from pydantic import BaseModel

from .models import (
    PrecisionTimestamp,
    ClockSource,
    VectorClock,
    TimeConversionRequest
)
from .manager import PrecisionClockManager


class ChronoTickTools(str, Enum):
    """Available ChronoTick MCP tools."""
    GET_CURRENT_TIME = "get_current_time"
    CONVERT_TIME = "convert_time"
    GET_SYNC_STATUS = "get_sync_status"
    CREATE_VECTOR_CLOCK = "create_vector_clock"
    COMPARE_TIMESTAMPS = "compare_timestamps"
    MEASURE_CLOCK_DRIFT = "measure_clock_drift"


class TimeConversionResult(BaseModel):
    """High-precision time conversion result."""
    source: PrecisionTimestamp
    target: PrecisionTimestamp
    time_difference: str
    offset_seconds: float


class ChronoTickServer:
    """Enhanced MCP time server with high-precision capabilities."""
    
    def __init__(self, node_id: Optional[str] = None):
        self.clock_manager = PrecisionClockManager(node_id)
        self.logger = logging.getLogger(__name__)
    
    async def get_current_time(self, timezone: str, source: Optional[str] = None) -> PrecisionTimestamp:
        """Get high-precision current time in specified timezone."""
        source_enum = None
        if source:
            try:
                source_enum = ClockSource(source)
            except ValueError:
                raise ValueError(f"Invalid clock source: {source}")
        
        return await self.clock_manager.get_precision_time(timezone, source_enum)
    
    async def convert_time(self, source_timezone: str, time_str: str, target_timezone: str) -> TimeConversionResult:
        """Convert time between timezones with high precision."""
        # Parse the time string (HH:MM format)
        try:
            parsed_time = datetime.strptime(time_str, "%H:%M").time()
        except ValueError:
            raise ValueError("Invalid time format. Expected HH:MM [24-hour format]")
        
        # Get current date in source timezone to build the full timestamp
        current_timestamp = await self.clock_manager.get_precision_time(source_timezone)
        current_dt = current_timestamp.to_datetime()
        
        # Create a precision timestamp with the specified time
        from zoneinfo import ZoneInfo
        from datetime import timezone
        
        if source_timezone == "UTC":
            tz = timezone.utc
        else:
            tz = ZoneInfo(source_timezone)
        
        source_time_dt = datetime(
            current_dt.year,
            current_dt.month,
            current_dt.day,
            parsed_time.hour,
            parsed_time.minute,
            tzinfo=tz
        )
        
        # Create source PrecisionTimestamp
        source_timestamp_ns = int(source_time_dt.timestamp() * 1e9)
        source_precision = PrecisionTimestamp(
            utc_seconds=source_timestamp_ns // 1_000_000_000,
            utc_nanoseconds=source_timestamp_ns % 1_000_000_000,
            timezone=source_timezone,
            is_dst=bool(source_time_dt.dst()) if source_time_dt.dst() is not None else False,
            monotonic_ns=current_timestamp.monotonic_ns,
            source=current_timestamp.source,
            quality=current_timestamp.quality,
            offset_ns=current_timestamp.offset_ns,
            drift_rate=current_timestamp.drift_rate,
            uncertainty_ns=current_timestamp.uncertainty_ns
        )
        
        # Convert to target timezone
        request = TimeConversionRequest(
            source_timezone=source_timezone,
            target_timezone=target_timezone,
            timestamp=source_precision
        )
        
        conversion_result = await self.clock_manager.convert_time(request)
        
        # Calculate time difference string
        offset_hours = conversion_result.offset_seconds / 3600
        if offset_hours.is_integer():
            time_diff_str = f"{offset_hours:+.1f}h"
        else:
            time_diff_str = f"{offset_hours:+.2f}".rstrip("0").rstrip(".") + "h"
        
        return TimeConversionResult(
            source=conversion_result.source,
            target=conversion_result.target,
            time_difference=time_diff_str,
            offset_seconds=conversion_result.offset_seconds
        )
    
    
    async def get_sync_status(self, source: Optional[str] = None) -> Dict[str, Any]:
        """Get synchronization status."""
        if source:
            # Get status for specific source
            all_status = await self.clock_manager.get_sync_status()
            if source in all_status:
                return {"source": source, "status": all_status[source].model_dump()}
            else:
                raise ValueError(f"Unknown time source: {source}")
        else:
            # Get overall status
            overall_status = await self.clock_manager.get_overall_sync_status()
            all_status = await self.clock_manager.get_sync_status()
            
            return {
                "overall": overall_status.model_dump(),
                "sources": {name: status.model_dump() for name, status in all_status.items()}
            }
    
    def create_vector_clock(self) -> VectorClock:
        """Create a new vector clock for event ordering."""
        return self.clock_manager.create_vector_clock()
    
    def compare_timestamps(self, timestamp1: Dict[str, Any], timestamp2: Dict[str, Any]) -> Dict[str, Any]:
        """Compare two timestamps for causality and ordering."""
        try:
            ts1 = PrecisionTimestamp(**timestamp1)
            ts2 = PrecisionTimestamp(**timestamp2)
            return self.clock_manager.compare_timestamps(ts1, ts2)
        except Exception as e:
            raise ValueError(f"Invalid timestamp format: {e}")
    
    async def measure_clock_drift(self, duration_seconds: int = 60) -> Dict[str, Any]:
        """Measure clock drift over specified duration."""
        if duration_seconds < 1 or duration_seconds > 3600:
            raise ValueError("Duration must be between 1 and 3600 seconds")
        
        return await self.clock_manager.measure_clock_drift(duration_seconds)
    


async def serve(node_id: Optional[str] = None, local_timezone: Optional[str] = None) -> None:
    """Start the ChronoTick MCP server."""
    server = Server("chronotick-server")
    chronotick_server = ChronoTickServer(node_id)
    
    # Determine local timezone for defaults
    if local_timezone is None:
        from tzlocal import get_localzone_name
        local_timezone = get_localzone_name() or "UTC"
    
    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List available ChronoTick tools."""
        return [
            Tool(
                name=ChronoTickTools.GET_CURRENT_TIME.value,
                description="Get high-precision current time in a specific timezone",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "timezone": {
                            "type": "string",
                            "description": f"IANA timezone name (e.g., 'America/New_York', 'Europe/London'). Use '{local_timezone}' as local timezone if no timezone provided by the user.",
                        },
                        "source": {
                            "type": "string",
                            "description": "Preferred time source: system, ntp",
                            "enum": ["system", "ntp"]
                        }
                    },
                    "required": ["timezone"],
                },
            ),
            Tool(
                name=ChronoTickTools.CONVERT_TIME.value,
                description="Convert time between timezones with high precision",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "source_timezone": {
                            "type": "string",
                            "description": f"Source IANA timezone name (e.g., 'America/New_York', 'Europe/London'). Use '{local_timezone}' as local timezone if no source timezone provided by the user.",
                        },
                        "time": {
                            "type": "string",
                            "description": "Time to convert in 24-hour format (HH:MM)",
                        },
                        "target_timezone": {
                            "type": "string",
                            "description": f"Target IANA timezone name (e.g., 'Asia/Tokyo', 'America/San_Francisco'). Use '{local_timezone}' as local timezone if no target timezone provided by the user.",
                        },
                    },
                    "required": ["source_timezone", "time", "target_timezone"],
                },
            ),
            Tool(
                name=ChronoTickTools.GET_SYNC_STATUS.value,
                description="Get clock synchronization status and health metrics",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "source": {
                            "type": "string",
                            "description": "Specific source to check (optional, returns all if not specified)",
                            "enum": ["system", "ntp"]
                        }
                    },
                },
            ),
            Tool(
                name=ChronoTickTools.CREATE_VECTOR_CLOCK.value,
                description="Create a new vector clock for distributed event ordering",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name=ChronoTickTools.COMPARE_TIMESTAMPS.value,
                description="Compare two timestamps for causality and temporal ordering",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "timestamp1": {
                            "type": "object",
                            "description": "First timestamp (PrecisionTimestamp format)"
                        },
                        "timestamp2": {
                            "type": "object",
                            "description": "Second timestamp (PrecisionTimestamp format)"
                        }
                    },
                    "required": ["timestamp1", "timestamp2"],
                },
            ),
            Tool(
                name=ChronoTickTools.MEASURE_CLOCK_DRIFT.value,
                description="Measure clock drift over a specified duration",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "duration_seconds": {
                            "type": "integer",
                            "description": "Duration to measure drift (1-3600 seconds, default: 60)",
                            "minimum": 1,
                            "maximum": 3600,
                            "default": 60
                        }
                    },
                },
            ),
        ]
    
    @server.call_tool()
    async def call_tool(
        name: str, arguments: dict
    ) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
        """Handle tool calls for ChronoTick queries."""
        try:
            result = None
            
            match name:
                case ChronoTickTools.GET_CURRENT_TIME.value:
                    timezone = arguments.get("timezone")
                    source = arguments.get("source")
                    if not timezone:
                        raise ValueError("Missing required argument: timezone")
                    result = await chronotick_server.get_current_time(timezone, source)
                
                case ChronoTickTools.CONVERT_TIME.value:
                    if not all(k in arguments for k in ["source_timezone", "time", "target_timezone"]):
                        raise ValueError("Missing required arguments")
                    result = await chronotick_server.convert_time(
                        arguments["source_timezone"],
                        arguments["time"],
                        arguments["target_timezone"],
                    )
                
                
                case ChronoTickTools.GET_SYNC_STATUS.value:
                    source = arguments.get("source")
                    result = await chronotick_server.get_sync_status(source)
                
                case ChronoTickTools.CREATE_VECTOR_CLOCK.value:
                    result = chronotick_server.create_vector_clock()
                
                case ChronoTickTools.COMPARE_TIMESTAMPS.value:
                    if not all(k in arguments for k in ["timestamp1", "timestamp2"]):
                        raise ValueError("Missing required arguments")
                    result = chronotick_server.compare_timestamps(
                        arguments["timestamp1"],
                        arguments["timestamp2"]
                    )
                
                case ChronoTickTools.MEASURE_CLOCK_DRIFT.value:
                    duration = arguments.get("duration_seconds", 60)
                    result = await chronotick_server.measure_clock_drift(duration)
                
                
                case _:
                    raise ValueError(f"Unknown tool: {name}")
            
            # Convert result to JSON
            if hasattr(result, 'model_dump'):
                json_result = result.model_dump()
            elif isinstance(result, dict):
                json_result = result
            else:
                json_result = result.__dict__ if hasattr(result, '__dict__') else str(result)
            
            return [
                TextContent(type="text", text=json.dumps(json_result, indent=2, default=str))
            ]
        
        except Exception as e:
            chronotick_server.logger.error(f"Error processing {name}: {e}")
            raise ValueError(f"Error processing ChronoTick query: {str(e)}")
    
    # Start the server
    options = server.create_initialization_options()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, options)