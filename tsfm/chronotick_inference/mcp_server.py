#!/usr/bin/env python3
"""
ChronoTick MCP Server

Model Context Protocol server that provides high-precision time services
to AI agents through fast IPC communication with the ChronoTick daemon.

Features:
- Real-time clock corrections with microsecond precision
- Error bounds and uncertainty quantification
- Fast IPC communication for minimal latency
- Daemon lifecycle management (warmup, ready, error states)
- Performance monitoring and statistics
"""

import asyncio
import json
import logging
import time
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import functools
import inspect

import mcp.server.stdio
import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
import anyio

from .daemon import ChronoTickDaemon
from .real_data_pipeline import CorrectionWithBounds

logger = logging.getLogger(__name__)
debug_logger = logging.getLogger(f"{__name__}.debug")


def debug_trace(include_args=True, include_result=True, include_timing=True):
    """
    Decorator for comprehensive debug logging of function calls.
    
    Args:
        include_args: Log function arguments
        include_result: Log function return value
        include_timing: Log execution time
    """
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not debug_logger.isEnabledFor(logging.DEBUG):
                return await func(*args, **kwargs)
            
            func_name = f"{func.__module__}.{func.__qualname__}"
            call_id = id(args) if args else id(kwargs)
            
            # Log function entry
            debug_info = {"function": func_name, "call_id": call_id}
            
            if include_args:
                # Safely serialize args/kwargs, handling complex objects
                try:
                    debug_args = []
                    for i, arg in enumerate(args):
                        if hasattr(arg, '__dict__'):
                            debug_args.append(f"<{type(arg).__name__} object>")
                        elif isinstance(arg, (str, int, float, bool, type(None))):
                            debug_args.append(repr(arg))
                        else:
                            debug_args.append(f"<{type(arg).__name__}>")
                    
                    debug_kwargs = {}
                    for k, v in kwargs.items():
                        if hasattr(v, '__dict__'):
                            debug_kwargs[k] = f"<{type(v).__name__} object>"
                        elif isinstance(v, (str, int, float, bool, type(None), dict, list)):
                            debug_kwargs[k] = v
                        else:
                            debug_kwargs[k] = f"<{type(v).__name__}>"
                    
                    debug_info["args"] = debug_args
                    debug_info["kwargs"] = debug_kwargs
                except Exception as e:
                    debug_info["args_error"] = str(e)
            
            debug_logger.debug(f"ENTRY: {json.dumps(debug_info, indent=2)}")
            
            # Execute function with timing
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Log function exit
                exit_info = {
                    "function": func_name,
                    "call_id": call_id,
                    "status": "success"
                }
                
                if include_timing:
                    exit_info["execution_time_ms"] = execution_time * 1000
                
                if include_result:
                    try:
                        if isinstance(result, (str, int, float, bool, type(None))):
                            exit_info["result"] = result
                        elif isinstance(result, (dict, list)):
                            # Truncate large results
                            result_str = json.dumps(result)
                            if len(result_str) > 1000:
                                exit_info["result"] = result_str[:1000] + "... (truncated)"
                            else:
                                exit_info["result"] = result
                        else:
                            exit_info["result"] = f"<{type(result).__name__} object>"
                    except Exception as e:
                        exit_info["result_error"] = str(e)
                
                debug_logger.debug(f"EXIT: {json.dumps(exit_info, indent=2)}")
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                error_info = {
                    "function": func_name,
                    "call_id": call_id,
                    "status": "error",
                    "error": str(e),
                    "error_type": type(e).__name__
                }
                if include_timing:
                    error_info["execution_time_ms"] = execution_time * 1000
                
                debug_logger.debug(f"ERROR: {json.dumps(error_info, indent=2)}")
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            if not debug_logger.isEnabledFor(logging.DEBUG):
                return func(*args, **kwargs)
            
            func_name = f"{func.__module__}.{func.__qualname__}"
            call_id = id(args) if args else id(kwargs)
            
            # Log function entry (same logic as async)
            debug_info = {"function": func_name, "call_id": call_id}
            
            if include_args:
                try:
                    debug_args = []
                    for i, arg in enumerate(args):
                        if hasattr(arg, '__dict__'):
                            debug_args.append(f"<{type(arg).__name__} object>")
                        elif isinstance(arg, (str, int, float, bool, type(None))):
                            debug_args.append(repr(arg))
                        else:
                            debug_args.append(f"<{type(arg).__name__}>")
                    
                    debug_kwargs = {}
                    for k, v in kwargs.items():
                        if hasattr(v, '__dict__'):
                            debug_kwargs[k] = f"<{type(v).__name__} object>"
                        elif isinstance(v, (str, int, float, bool, type(None), dict, list)):
                            debug_kwargs[k] = v
                        else:
                            debug_kwargs[k] = f"<{type(v).__name__}>"
                    
                    debug_info["args"] = debug_args
                    debug_info["kwargs"] = debug_kwargs
                except Exception as e:
                    debug_info["args_error"] = str(e)
            
            debug_logger.debug(f"ENTRY: {json.dumps(debug_info, indent=2)}")
            
            # Execute function with timing
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Log function exit
                exit_info = {
                    "function": func_name,
                    "call_id": call_id,
                    "status": "success"
                }
                
                if include_timing:
                    exit_info["execution_time_ms"] = execution_time * 1000
                
                if include_result:
                    try:
                        if isinstance(result, (str, int, float, bool, type(None))):
                            exit_info["result"] = result
                        elif isinstance(result, (dict, list)):
                            result_str = json.dumps(result)
                            if len(result_str) > 1000:
                                exit_info["result"] = result_str[:1000] + "... (truncated)"
                            else:
                                exit_info["result"] = result
                        else:
                            exit_info["result"] = f"<{type(result).__name__} object>"
                    except Exception as e:
                        exit_info["result_error"] = str(e)
                
                debug_logger.debug(f"EXIT: {json.dumps(exit_info, indent=2)}")
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                error_info = {
                    "function": func_name,
                    "call_id": call_id,
                    "status": "error",
                    "error": str(e),
                    "error_type": type(e).__name__
                }
                if include_timing:
                    error_info["execution_time_ms"] = execution_time * 1000
                
                debug_logger.debug(f"ERROR: {json.dumps(error_info, indent=2)}")
                raise
        
        # Return appropriate wrapper based on function type
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


@dataclass
class TimeResponse:
    """Response structure for time requests"""
    corrected_time: float           # System time + corrections
    system_time: float              # Raw system time
    offset_correction: float        # Applied offset correction (seconds)
    drift_rate: float              # Current drift rate (seconds/second)
    offset_uncertainty: float      # Offset uncertainty (seconds)
    drift_uncertainty: float       # Drift uncertainty (seconds/second)
    time_uncertainty: float        # Total time uncertainty at request time
    confidence: float              # Model confidence [0,1]
    source: str                    # Correction source (cpu, gpu, fusion, ntp, no_data)
    prediction_time: float         # When the correction was predicted
    valid_until: float             # When the correction expires
    daemon_status: str             # Daemon status (warmup, ready, error)
    call_latency_ms: float         # MCP call processing time


@dataclass
class DaemonStatus:
    """Daemon status information"""
    status: str                    # warmup, ready, error, stopped
    warmup_progress: float         # 0.0 to 1.0 during warmup
    warmup_remaining_seconds: float # Seconds until warmup complete
    total_corrections: int         # Total corrections served
    success_rate: float           # Correction success rate
    average_latency_ms: float     # Average correction latency
    memory_usage_mb: float        # Daemon memory usage
    cpu_affinity: List[int]       # CPU cores assigned
    uptime_seconds: float         # Daemon uptime
    last_error: Optional[str]     # Last error message if any


class ChronoTickMCPServer:
    """
    MCP server that provides high-precision time services to AI agents.
    
    Manages a ChronoTick daemon process and provides fast IPC communication
    for time correction requests with microsecond precision and error bounds.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize MCP server with optional configuration"""
        self.config_path = config_path or self._get_default_config()
        self.daemon: Optional[ChronoTickDaemon] = None
        self.daemon_process: Optional[mp.Process] = None
        self.request_queue: Optional[mp.Queue] = None
        self.response_queue: Optional[mp.Queue] = None
        self.status_queue: Optional[mp.Queue] = None
        
        # Server state
        self.server_start_time = time.time()
        self.total_requests = 0
        self.successful_requests = 0
        self.request_latencies = []
        
        # Initialize MCP server
        self.server = Server("chronotick")
        self._setup_mcp_handlers()
    
    def _get_default_config(self) -> str:
        """Get default configuration path"""
        config_dir = Path(__file__).parent / "configs"
        default_config = config_dir / "hybrid_timesfm_chronos.yaml"
        if default_config.exists():
            return str(default_config)
        
        # Fallback to basic config
        basic_config = Path(__file__).parent / "config.yaml"
        if basic_config.exists():
            return str(basic_config)
        
        raise FileNotFoundError("No ChronoTick configuration found")
    
    def _setup_mcp_handlers(self):
        """Setup MCP server request handlers"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[types.Tool]:
            """List available ChronoTick tools"""
            return [
                types.Tool(
                    name="get_time",
                    description="Get high-precision corrected time with uncertainty bounds",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "include_stats": {
                                "type": "boolean",
                                "description": "Include detailed statistics and metadata",
                                "default": False
                            }
                        }
                    }
                ),
                types.Tool(
                    name="get_daemon_status",
                    description="Get ChronoTick daemon status and performance metrics",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                types.Tool(
                    name="get_time_with_future_uncertainty",
                    description="Get time with uncertainty projection for future timestamp",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "future_seconds": {
                                "type": "number",
                                "description": "Seconds in the future to project uncertainty",
                                "minimum": 0,
                                "maximum": 3600
                            }
                        },
                        "required": ["future_seconds"]
                    }
                ),
                types.Tool(
                    name="get_time_with_confidence_interval",
                    description="Get corrected time with confidence interval bounds based on TimesFM quantiles",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "confidence_level": {
                                "type": "number",
                                "description": "Confidence level for interval (e.g., 0.9 for 90% confidence)",
                                "minimum": 0.0,
                                "maximum": 1.0,
                                "default": 0.9
                            },
                            "include_stats": {
                                "type": "boolean",
                                "description": "Include detailed statistics",
                                "default": False
                            }
                        }
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
            """Handle tool calls"""
            request_start = time.time()
            
            try:
                if name == "get_time":
                    result = await self._handle_get_time(arguments)
                elif name == "get_daemon_status":
                    result = await self._handle_get_daemon_status()
                elif name == "get_time_with_future_uncertainty":
                    result = await self._handle_get_time_with_future_uncertainty(arguments)
                elif name == "get_time_with_confidence_interval":
                    result = await self._handle_get_time_with_confidence_interval(arguments)
                else:
                    raise ValueError(f"Unknown tool: {name}")
                
                # Track request statistics
                latency = (time.time() - request_start) * 1000
                self._track_request_latency(latency)
                
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
                
            except Exception as e:
                self.total_requests += 1
                logger.error(f"Tool call failed: {e}")
                return [types.TextContent(
                    type="text", 
                    text=json.dumps({
                        "error": str(e),
                        "tool": name,
                        "arguments": arguments,
                        "timestamp": time.time()
                    }, indent=2)
                )]
    
    @debug_trace(include_args=True, include_result=True, include_timing=True)
    async def _handle_get_time(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get_time tool calls"""
        if not self._is_daemon_ready():
            raise RuntimeError("ChronoTick daemon not ready - check daemon status")
        
        request_start = time.time()
        
        # Get correction from daemon via IPC
        correction = await self._get_correction_from_daemon()
        if not correction:
            raise RuntimeError("Failed to get time correction from daemon")
        
        # Calculate corrected time
        system_time = time.time()
        time_delta = system_time - correction.prediction_time
        corrected_time = system_time + correction.offset_correction + (correction.drift_rate * time_delta)
        
        # Calculate uncertainty at current time
        time_uncertainty = correction.get_time_uncertainty(time_delta)
        
        # Get daemon status
        daemon_status = await self._get_daemon_status_from_daemon()
        
        response = TimeResponse(
            corrected_time=corrected_time,
            system_time=system_time,
            offset_correction=correction.offset_correction,
            drift_rate=correction.drift_rate,
            offset_uncertainty=correction.offset_uncertainty,
            drift_uncertainty=correction.drift_uncertainty,
            time_uncertainty=time_uncertainty,
            confidence=correction.confidence,
            source=correction.source,
            prediction_time=correction.prediction_time,
            valid_until=correction.valid_until,
            daemon_status=daemon_status.status if daemon_status else "unknown",
            call_latency_ms=(time.time() - request_start) * 1000
        )
        
        result = asdict(response)
        
        # Add detailed stats if requested
        if arguments.get("include_stats", False):
            result["detailed_stats"] = await self._get_detailed_stats()
        
        return result
    
    @debug_trace(include_args=True, include_result=True, include_timing=True)
    async def _handle_get_daemon_status(self) -> Dict[str, Any]:
        """Handle daemon status requests"""
        daemon_status = await self._get_daemon_status_from_daemon()
        
        if daemon_status:
            result = asdict(daemon_status)
        else:
            result = {
                "status": "error",
                "error": "Cannot communicate with daemon"
            }
        
        # Add MCP server stats
        result["mcp_server"] = {
            "uptime_seconds": time.time() - self.server_start_time,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "success_rate": self.successful_requests / max(1, self.total_requests),
            "average_latency_ms": sum(self.request_latencies) / max(1, len(self.request_latencies))
        }
        
        return result
    
    @debug_trace(include_args=True, include_result=True, include_timing=True)
    async def _handle_get_time_with_future_uncertainty(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle time requests with future uncertainty projection"""
        future_seconds = arguments["future_seconds"]
        
        # Get current time correction
        current_response = await self._handle_get_time({"include_stats": False})
        
        # Project uncertainty into the future
        correction = await self._get_correction_from_daemon()
        if correction:
            future_uncertainty = correction.get_time_uncertainty(future_seconds)
            future_time = current_response["corrected_time"] + future_seconds
            
            current_response.update({
                "future_timestamp": future_time,
                "future_seconds": future_seconds,
                "future_uncertainty": future_uncertainty,
                "uncertainty_increase": future_uncertainty - current_response["time_uncertainty"]
            })
        
        return current_response

    @debug_trace(include_args=True, include_result=True, include_timing=True)
    async def _handle_get_time_with_confidence_interval(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle time requests with confidence interval bounds from quantiles"""
        if not self._is_daemon_ready():
            raise RuntimeError("ChronoTick daemon not ready - check daemon status")

        confidence_level = arguments.get("confidence_level", 0.9)
        request_start = time.time()

        # Get correction from daemon
        correction = await self._get_correction_from_daemon()
        if not correction:
            raise RuntimeError("Failed to get time correction from daemon")

        # Calculate corrected time
        system_time = time.time()
        time_delta = system_time - correction.prediction_time
        corrected_time = system_time + correction.offset_correction + (correction.drift_rate * time_delta)

        # Calculate uncertainty at current time
        time_uncertainty = correction.get_time_uncertainty(time_delta)

        # Get confidence interval from quantiles
        confidence_interval = correction.get_confidence_interval(confidence_level)

        # Get daemon status
        daemon_status = await self._get_daemon_status_from_daemon()

        # Build base response
        response = {
            "corrected_time": corrected_time,
            "system_time": system_time,
            "offset_correction": correction.offset_correction,
            "drift_rate": correction.drift_rate,
            "offset_uncertainty": correction.offset_uncertainty,
            "drift_uncertainty": correction.drift_uncertainty,
            "time_uncertainty": time_uncertainty,
            "confidence": correction.confidence,
            "source": correction.source,
            "prediction_time": correction.prediction_time,
            "valid_until": correction.valid_until,
            "daemon_status": daemon_status.status if daemon_status else "unknown",
            "call_latency_ms": (time.time() - request_start) * 1000,
            "confidence_level": confidence_level
        }

        # Add confidence interval if available
        if confidence_interval:
            lower_bound, upper_bound = confidence_interval
            # Apply same drift correction to bounds
            lower_time = system_time + lower_bound + (correction.drift_rate * time_delta)
            upper_time = system_time + upper_bound + (correction.drift_rate * time_delta)

            response.update({
                "confidence_interval": {
                    "lower_bound": lower_time,
                    "upper_bound": upper_time,
                    "range": upper_time - lower_time,
                    "offset_lower": lower_bound,
                    "offset_upper": upper_bound
                },
                "quantiles_available": True
            })
        else:
            response.update({
                "confidence_interval": None,
                "quantiles_available": False,
                "note": "Quantiles not available from model - using uncertainty bounds as fallback"
            })

        # Add detailed stats if requested
        if arguments.get("include_stats", False):
            response["detailed_stats"] = await self._get_detailed_stats()

        return response

    @debug_trace(include_args=False, include_result=True, include_timing=True)
    async def _get_correction_from_daemon(self) -> Optional[CorrectionWithBounds]:
        """Get time correction from daemon via IPC"""
        try:
            if not self.request_queue or not self.response_queue:
                return None
            
            # Send request to daemon
            self.request_queue.put({"type": "get_time", "timestamp": time.time()})
            
            # Wait for response with timeout
            try:
                response = self.response_queue.get(timeout=0.3)  # 300ms timeout to handle GIL contention
                
                if response["type"] == "correction":
                    # Reconstruct CorrectionWithBounds from response data
                    return CorrectionWithBounds(**response["data"])
                else:
                    logger.error(f"Unexpected response type: {response['type']}")
                    return None
                    
            except mp.TimeoutError:
                logger.warning("Daemon response timeout")
                return None
                
        except Exception as e:
            logger.error(f"IPC communication error: {e}")
            return None
    
    @debug_trace(include_args=False, include_result=True, include_timing=True)
    async def _get_daemon_status_from_daemon(self) -> Optional[DaemonStatus]:
        """Get daemon status via IPC"""
        try:
            if not self.request_queue or not self.response_queue:
                return None
            
            # Send status request
            self.request_queue.put({"type": "get_status", "timestamp": time.time()})
            
            try:
                response = self.response_queue.get(timeout=0.3)  # 300ms timeout to handle GIL contention

                if response["type"] == "status":
                    return DaemonStatus(**response["data"])
                else:
                    return None
                    
            except mp.TimeoutError:
                return None
                
        except Exception as e:
            logger.error(f"Status IPC error: {e}")
            return None
    
    def _track_request_latency(self, latency_ms: float):
        """Track request latency and manage history"""
        self.total_requests += 1
        self.successful_requests += 1
        self.request_latencies.append(latency_ms)
        
        # Keep only recent latencies for statistics
        if len(self.request_latencies) > 1000:
            self.request_latencies = self.request_latencies[-500:]
    
    def _is_daemon_ready(self) -> bool:
        """Check if daemon is ready to serve requests"""
        if not self.daemon_process or not self.daemon_process.is_alive():
            return False
        
        # Check if we can get status (quick check)
        try:
            if self.status_queue and not self.status_queue.empty():
                status_update = self.status_queue.get_nowait()
                return status_update.get("status") == "ready"
        except:
            pass
        
        return True
    
    async def _get_detailed_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics"""
        daemon_status = await self._get_daemon_status_from_daemon()
        
        stats = {
            "daemon": asdict(daemon_status) if daemon_status else None,
            "mcp_server": {
                "uptime_seconds": time.time() - self.server_start_time,
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "success_rate": self.successful_requests / max(1, self.total_requests),
                "request_latencies": {
                    "count": len(self.request_latencies),
                    "average_ms": sum(self.request_latencies) / max(1, len(self.request_latencies)),
                    "min_ms": min(self.request_latencies) if self.request_latencies else 0,
                    "max_ms": max(self.request_latencies) if self.request_latencies else 0
                }
            }
        }
        
        return stats
    
    async def start_daemon(self):
        """Start the ChronoTick daemon process"""
        logger.info("Starting ChronoTick daemon...")
        
        # Create IPC queues
        self.request_queue = mp.Queue()
        self.response_queue = mp.Queue()
        self.status_queue = mp.Queue()
        
        # Create and start daemon process
        self.daemon_process = mp.Process(
            target=self._run_daemon_process,
            args=(self.config_path, self.request_queue, self.response_queue, self.status_queue),
            daemon=False  # Don't make it a daemon process so it can be managed properly
        )
        self.daemon_process.start()
        
        # Wait for daemon to initialize
        logger.info("Waiting for daemon initialization...")
        await self._wait_for_daemon_ready()
    
    def _run_daemon_process(self, config_path: str, request_queue: mp.Queue, 
                           response_queue: mp.Queue, status_queue: mp.Queue):
        """Run daemon in separate process"""
        try:
            daemon = ChronoTickDaemon(config_path)
            daemon.run_with_ipc(request_queue, response_queue, status_queue)
        except Exception as e:
            logger.error(f"Daemon process failed: {e}")
            status_queue.put({"status": "error", "error": str(e)})
    
    async def _wait_for_daemon_ready(self, timeout_seconds: float = 300):
        """Wait for daemon to complete warmup and become ready"""
        start_time = time.time()
        last_status_log = 0
        
        while time.time() - start_time < timeout_seconds:
            # Check daemon process health
            if not self.daemon_process or not self.daemon_process.is_alive():
                raise RuntimeError("Daemon process died during initialization")
            
            # Check for status updates
            try:
                if self.status_queue and not self.status_queue.empty():
                    status_update = self.status_queue.get_nowait()
                    status = status_update.get("status", "unknown")
                    
                    if status == "ready":
                        logger.info("✅ ChronoTick daemon ready - warmup complete!")
                        return
                    elif status == "warmup":
                        progress = status_update.get("progress", 0)
                        remaining = status_update.get("remaining_seconds", 0)
                        
                        # Log status updates every 10 seconds
                        if time.time() - last_status_log > 10:
                            logger.info(f"🕒 ChronoTick warmup: {progress:.1%} complete, "
                                      f"{remaining:.0f}s remaining")
                            last_status_log = time.time()
                    elif status == "error":
                        error_msg = status_update.get("error", "Unknown error")
                        raise RuntimeError(f"Daemon initialization failed: {error_msg}")
            except:
                pass
            
            await asyncio.sleep(0.5)  # Check every 500ms
        
        raise TimeoutError(f"Daemon failed to become ready within {timeout_seconds}s")
    
    async def stop_daemon(self):
        """Stop the daemon process gracefully"""
        if self.daemon_process and self.daemon_process.is_alive():
            logger.info("Stopping ChronoTick daemon...")
            
            # Send shutdown signal via IPC
            if self.request_queue:
                self.request_queue.put({"type": "shutdown"})
            
            # Wait for graceful shutdown
            self.daemon_process.join(timeout=10.0)
            
            # Force terminate if needed
            if self.daemon_process.is_alive():
                logger.warning("Force terminating daemon process")
                self.daemon_process.terminate()
                self.daemon_process.join(timeout=5.0)
            
            logger.info("Daemon stopped")
    
    async def run_server(self):
        """Run the MCP server"""
        # Start daemon first
        await self.start_daemon()
        
        try:
            # Run MCP server
            async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
                logger.info("🚀 ChronoTick MCP Server ready - accepting agent connections")
                await self.server.run(
                    read_stream,
                    write_stream,
                    InitializationOptions(
                        server_name="chronotick",
                        server_version="1.0.0",
                        capabilities=self.server.get_capabilities(
                            notification_options=NotificationOptions(),
                            experimental_capabilities={}
                        )
                    )
                )
        finally:
            await self.stop_daemon()


def create_mcp_server(config_path: Optional[str] = None) -> ChronoTickMCPServer:
    """Create ChronoTick MCP server instance"""
    return ChronoTickMCPServer(config_path)


async def main():
    """Main entry point for ChronoTick MCP server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ChronoTick MCP Server")
    parser.add_argument("--config", type=str, help="Path to ChronoTick configuration file")
    parser.add_argument("--log-level", type=str, default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--debug-trace", action="store_true",
                       help="Enable comprehensive debug tracing of function calls, model I/O, and IPC")
    parser.add_argument("--debug-log-file", type=str,
                       help="Write debug logs to file (in addition to console)")
    
    args = parser.parse_args()
    
    # Setup enhanced logging
    log_level = getattr(logging, args.log_level)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    simple_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    if args.debug_trace or log_level == logging.DEBUG:
        console_handler.setFormatter(detailed_formatter)
    else:
        console_handler.setFormatter(simple_formatter)
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)
    
    # File handler for debug logs
    if args.debug_log_file:
        file_handler = logging.FileHandler(args.debug_log_file)
        file_handler.setFormatter(detailed_formatter)
        file_handler.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        logger.info(f"Debug logs will be written to: {args.debug_log_file}")
    
    # Configure debug trace logging
    if args.debug_trace:
        # Enable debug logging for all chronotick modules
        debug_modules = [
            "chronotick_inference.mcp_server.debug",
            "chronotick_inference.real_data_pipeline.debug", 
            "chronotick_inference.daemon.debug",
            "chronotick_inference.ntp_client.debug",
            "chronotick_inference.predictive_scheduler.debug",
            "chronotick_inference.engine.debug"
        ]
        
        for module in debug_modules:
            debug_logger = logging.getLogger(module)
            debug_logger.setLevel(logging.DEBUG)
        
        logger.info("🔍 Enhanced debug tracing enabled - function calls, model I/O, and IPC will be logged")
        logger.info("📊 Debug modules enabled: " + ", ".join(debug_modules))
    
    # Log startup configuration
    logger.info("🚀 Starting ChronoTick MCP Server")
    logger.info(f"📝 Log level: {args.log_level}")
    logger.info(f"⚙️  Config: {args.config or 'default'}")
    if args.debug_trace:
        logger.info("🔍 Debug tracing: ENABLED")
    if args.debug_log_file:
        logger.info(f"📄 Debug log file: {args.debug_log_file}")
    
    # Create and run server
    server = create_mcp_server(args.config)
    await server.run_server()


def cli_main():
    """Synchronous entry point for CLI usage"""
    asyncio.run(main())


if __name__ == "__main__":
    cli_main()