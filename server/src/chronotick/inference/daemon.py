#!/usr/bin/env python3
"""
ChronoTick Inference Daemon

A separate process that runs the inference engine with proper CPU affinity
and communicates via IPC for providing corrected timestamps.
"""

import os
import sys
import time
import json
import signal
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from queue import Empty
import threading
import psutil
import functools
import inspect

# Add the chronotick package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chronotick.inference import (
    ChronoTickInferenceEngine,
    SystemMetricsCollector
)
from chronotick.inference.real_data_pipeline import RealDataPipeline


@dataclass
class TimeRequest:
    """Request for corrected time."""
    request_id: str
    timestamp: float
    include_uncertainty: bool = True
    include_bounds: bool = True


@dataclass
class TimeResponse:
    """Response with corrected time and uncertainty."""
    request_id: str
    corrected_time: float
    raw_time: float
    offset_correction: float
    uncertainty: Optional[float] = None
    confidence: Optional[float] = None
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    status: str = "success"
    error: Optional[str] = None
    inference_time: Optional[float] = None


@dataclass
class DaemonCommand:
    """Commands for controlling the daemon."""
    command: str  # start, stop, status, reconfigure, health
    params: Optional[Dict[str, Any]] = None


@dataclass
class DaemonStatus:
    """Status information from daemon."""
    running: bool
    pid: Optional[int] = None
    uptime: Optional[float] = None
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_inference_time: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    cpu_affinity: Optional[list] = None
    last_prediction_time: Optional[float] = None
    engine_status: Optional[str] = None
    error_message: Optional[str] = None


class ChronoTickDaemon:
    """
    ChronoTick inference daemon that runs in a separate process.
    
    Features:
    - Runs inference engine with CPU affinity
    - Collects clock offset measurements
    - Provides corrected timestamps via IPC
    - Monitors system health and performance
    """
    
    def __init__(self, config_path: str, cpu_affinity: Optional[list] = None, 
                 shared_memory_size: int = 1024 * 1024):
        """
        Initialize the daemon.
        
        Args:
            config_path: Path to inference configuration
            cpu_affinity: List of CPU cores to bind to
            shared_memory_size: Size of shared memory buffer
        """
        self.config_path = config_path
        self.cpu_affinity = cpu_affinity or []
        self.shared_memory_size = shared_memory_size
        
        # IPC components
        self.request_queue = mp.Queue()
        self.response_queue = mp.Queue()
        self.command_queue = mp.Queue()
        self.status_queue = mp.Queue()
        
        # Process management
        self.process: Optional[mp.Process] = None
        self.running = False
        self.start_time = 0.0
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'inference_times': []
        }
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('ChronoTickDaemon')
    
    def start_daemon(self) -> bool:
        """Start the inference daemon process."""
        if self.running:
            self.logger.warning("Daemon already running")
            return False
        
        try:
            self.process = mp.Process(
                target=self._daemon_worker,
                args=(
                    self.config_path,
                    self.cpu_affinity,
                    self.request_queue,
                    self.response_queue,
                    self.command_queue,
                    self.status_queue
                )
            )
            self.process.start()
            self.running = True
            self.start_time = time.time()
            
            self.logger.info(f"Daemon started with PID {self.process.pid}")
            
            # Wait for daemon to initialize
            time.sleep(2.0)
            
            # Check if daemon started successfully
            if self.process.is_alive():
                return True
            else:
                self.running = False
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to start daemon: {e}")
            self.running = False
            return False
    
    def stop_daemon(self) -> bool:
        """Stop the inference daemon process."""
        if not self.running or not self.process:
            return True
        
        try:
            # Send stop command
            self.command_queue.put(DaemonCommand("stop"))
            
            # Wait for graceful shutdown
            self.process.join(timeout=10.0)
            
            if self.process.is_alive():
                # Force termination
                self.logger.warning("Force terminating daemon process")
                self.process.terminate()
                self.process.join(timeout=5.0)
                
                if self.process.is_alive():
                    self.process.kill()
            
            self.running = False
            self.process = None
            self.logger.info("Daemon stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping daemon: {e}")
            return False
    
    def get_corrected_time(self, include_uncertainty: bool = True, 
                          include_bounds: bool = True, timeout: float = 1.0) -> TimeResponse:
        """
        Get corrected timestamp from the daemon.
        
        Args:
            include_uncertainty: Include uncertainty estimation
            include_bounds: Include confidence bounds
            timeout: Timeout for response
            
        Returns:
            TimeResponse with corrected time and metadata
        """
        if not self.running:
            return TimeResponse(
                request_id="",
                corrected_time=time.time(),
                raw_time=time.time(),
                offset_correction=0.0,
                status="error",
                error="Daemon not running"
            )
        
        # Create request
        request_id = f"req_{int(time.time() * 1000000)}"
        raw_time = time.time()
        
        request = TimeRequest(
            request_id=request_id,
            timestamp=raw_time,
            include_uncertainty=include_uncertainty,
            include_bounds=include_bounds
        )
        
        try:
            # Send request
            self.request_queue.put(request, timeout=timeout)
            
            # Wait for response
            response = self.response_queue.get(timeout=timeout)
            
            if response.request_id == request_id:
                self.stats['total_requests'] += 1
                if response.status == "success":
                    self.stats['successful_requests'] += 1
                    if response.inference_time:
                        self.stats['inference_times'].append(response.inference_time)
                else:
                    self.stats['failed_requests'] += 1
                
                return response
            else:
                # Wrong response, put it back and return error
                self.response_queue.put(response)
                return TimeResponse(
                    request_id=request_id,
                    corrected_time=raw_time,
                    raw_time=raw_time,
                    offset_correction=0.0,
                    status="error",
                    error="Response mismatch"
                )
                
        except Empty:
            return TimeResponse(
                request_id=request_id,
                corrected_time=raw_time,
                raw_time=raw_time,
                offset_correction=0.0,
                status="error",
                error="Timeout"
            )
        except Exception as e:
            return TimeResponse(
                request_id=request_id,
                corrected_time=raw_time,
                raw_time=raw_time,
                offset_correction=0.0,
                status="error",
                error=str(e)
            )
    
    def get_daemon_status(self, timeout: float = 1.0) -> DaemonStatus:
        """Get daemon status and statistics."""
        base_status = DaemonStatus(
            running=self.running,
            pid=self.process.pid if self.process else None,
            uptime=time.time() - self.start_time if self.running else None,
            total_requests=self.stats['total_requests'],
            successful_requests=self.stats['successful_requests'],
            failed_requests=self.stats['failed_requests'],
            avg_inference_time=sum(self.stats['inference_times'][-100:]) / 
                             len(self.stats['inference_times'][-100:]) 
                             if self.stats['inference_times'] else None
        )
        
        if not self.running:
            return base_status
        
        try:
            # Request status from daemon
            self.command_queue.put(DaemonCommand("status"))
            daemon_status = self.status_queue.get(timeout=timeout)
            
            # Merge with base status
            for key, value in asdict(daemon_status).items():
                if value is not None:
                    setattr(base_status, key, value)
            
            return base_status
            
        except Empty:
            base_status.error_message = "Status timeout"
            return base_status
        except Exception as e:
            base_status.error_message = str(e)
            return base_status
    
    @staticmethod
    def _daemon_worker(config_path: str, cpu_affinity: list,
                      request_queue: mp.Queue, response_queue: mp.Queue,
                      command_queue: mp.Queue, status_queue: mp.Queue):
        """
        Main worker function for the daemon process.
        
        This runs in a separate process and handles:
        - Setting CPU affinity
        - Initializing inference engine
        - Processing time requests
        - Monitoring system health
        """
        logger = logging.getLogger('ChronoTickDaemon.Worker')
        
        try:
            # Set CPU affinity
            if cpu_affinity:
                try:
                    psutil.Process().cpu_affinity(cpu_affinity)
                    logger.info(f"Set CPU affinity to cores: {cpu_affinity}")
                except Exception as e:
                    logger.warning(f"Failed to set CPU affinity: {e}")
            
            # Initialize inference engine
            logger.info("Initializing inference engine...")
            engine = ChronoTickInferenceEngine(config_path)
            success = engine.initialize_models()
            
            if not success:
                logger.error("Failed to initialize inference models")
                return
            
            logger.info("Inference engine initialized successfully")

            # Initialize real data pipeline (replaces synthetic ClockDataGenerator)
            logger.info("Initializing real data pipeline...")
            real_data_pipeline = RealDataPipeline(config_path)

            # Create model wrappers to bridge engine to pipeline
            from chronotick.inference.tsfm_model_wrapper import create_model_wrappers
            logger.info("Creating TSFM model wrappers...")
            cpu_wrapper, gpu_wrapper = create_model_wrappers(
                inference_engine=engine,
                dataset_manager=real_data_pipeline.dataset_manager,
                system_metrics=real_data_pipeline.system_metrics
            )

            # Initialize pipeline with models
            logger.info("Connecting ML models to pipeline...")
            # NOTE: initialize() calls set_model_interfaces() internally with pipeline=self
            real_data_pipeline.initialize(cpu_model=cpu_wrapper, gpu_model=gpu_wrapper)

            # Predictive scheduler already configured by initialize()
            logger.info("Predictive scheduler configured by pipeline initialization")
            
            # State management
            offset_history = []
            last_prediction = None
            last_prediction_time = 0.0
            prediction_valid_duration = 5.0  # Predictions valid for 5 seconds
            
            # Statistics
            request_count = 0
            start_time = time.time()
            
            logger.info("Daemon worker ready, entering main loop...")
            
            while True:
                try:
                    # Check for commands (non-blocking)
                    try:
                        command = command_queue.get_nowait()
                        if command.command == "stop":
                            logger.info("Received stop command")
                            break
                        elif command.command == "status":
                            # Send status
                            process = psutil.Process()
                            status = DaemonStatus(
                                running=True,
                                pid=os.getpid(),
                                uptime=time.time() - start_time,
                                memory_usage_mb=process.memory_info().rss / (1024 * 1024),
                                cpu_affinity=process.cpu_affinity(),
                                last_prediction_time=last_prediction_time,
                                engine_status="healthy"
                            )
                            status_queue.put(status)
                    except:
                        pass  # No command available
                    
                    # Process time requests (non-blocking)
                    try:
                        request = request_queue.get_nowait()
                        request_start_time = time.time()
                        
                        # Get real clock correction from pipeline
                        current_correction = real_data_pipeline.get_real_clock_correction(request_start_time)
                        current_offset = current_correction.offset_correction
                        
                        # Store in history for fallback (but pipeline manages its own dataset)
                        offset_history.append(current_offset)
                        
                        # Keep history manageable
                        if len(offset_history) > 3600:  # Keep 1 hour
                            offset_history = offset_history[-1800:]  # Trim to 30 minutes
                        
                        # Check if we need a new prediction
                        current_time = time.time()
                        if (last_prediction is None or 
                            current_time - last_prediction_time > prediction_valid_duration or
                            len(offset_history) % 10 == 0):  # Update every 10 measurements
                            
                            if len(offset_history) >= 30:  # Need minimum history
                                # Get system metrics
                                system_metrics = metrics_collector.get_recent_metrics(
                                    window_seconds=60
                                )
                                
                                # Make new prediction
                                context_data = offset_history[-100:]  # Last 100 measurements
                                prediction_result = engine.predict_fused(
                                    context_data, system_metrics
                                )
                                
                                if prediction_result:
                                    last_prediction = prediction_result
                                    last_prediction_time = current_time
                                    logger.debug(f"New prediction: {prediction_result.prediction*1e6:.3f}μs")
                        
                        # Calculate corrected time using real data pipeline
                        raw_time = request.timestamp
                        
                        # Use real correction from pipeline (includes offset + drift)
                        time_delta = raw_time - current_correction.prediction_time
                        offset_correction = current_correction.offset_correction + current_correction.drift_rate * time_delta
                        corrected_time = raw_time + offset_correction
                        
                        # Calculate uncertainty bounds using mathematical error propagation
                        uncertainty = None
                        lower_bound = None
                        upper_bound = None
                        confidence = current_correction.confidence
                        
                        if request.include_uncertainty:
                            # Use mathematical error propagation: sqrt(offset_unc² + (drift_unc * time_delta)²)
                            uncertainty = current_correction.get_time_uncertainty(time_delta)
                            
                        if request.include_bounds and uncertainty:
                            # Calculate bounds using corrected uncertainty
                            lower_bound = corrected_time - uncertainty
                            upper_bound = corrected_time + uncertainty
                        
                        # Create response
                        inference_time = time.time() - request_start_time
                        response = TimeResponse(
                            request_id=request.request_id,
                            corrected_time=corrected_time,
                            raw_time=raw_time,
                            offset_correction=offset_correction,
                            uncertainty=uncertainty,
                            confidence=confidence,
                            lower_bound=lower_bound,
                            upper_bound=upper_bound,
                            status="success",
                            inference_time=inference_time
                        )
                        
                        response_queue.put(response)
                        request_count += 1
                        
                    except:
                        pass  # No request available
                    
                    # Small sleep to prevent busy waiting
                    time.sleep(0.001)  # 1ms
                    
                except Exception as e:
                    logger.error(f"Error in daemon main loop: {e}")
                    time.sleep(0.1)
            
            # Cleanup
            logger.info("Shutting down daemon worker...")
            real_data_pipeline.shutdown()
            engine.shutdown()
            logger.info("Daemon worker shutdown complete")
            
        except Exception as e:
            logger.error(f"Fatal error in daemon worker: {e}")
            import traceback
            traceback.print_exc()
    
    def __enter__(self):
        """Context manager entry."""
        self.start_daemon()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_daemon()
    
    def run_with_ipc(self, request_queue: mp.Queue, response_queue: mp.Queue, status_queue: mp.Queue):
        """
        Run daemon with external IPC queues for MCP server integration.
        
        This method is called by the MCP server to run the daemon in a separate process
        with fast IPC communication for time correction requests.
        """
        import signal
        import os
        
        # Set up logging for this process
        logger = logging.getLogger('ChronoTickDaemon.MCP')
        
        # Set up signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down gracefully...")
            status_queue.put({"status": "shutting_down"})
            sys.exit(0)
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        # Set CPU affinity if specified
        if self.cpu_affinity:
            try:
                os.sched_setaffinity(0, self.cpu_affinity)
                logger.info(f"Set CPU affinity to cores: {self.cpu_affinity}")
            except Exception as e:
                logger.warning(f"Failed to set CPU affinity: {e}")
        
        try:
            # STEP 1: Initialize inference engine with ML models
            logger.info("Initializing ChronoTick inference engine with ML models...")
            status_queue.put({"status": "initializing_models"})

            from chronotick.inference.engine import ChronoTickInferenceEngine
            from chronotick.inference.tsfm_model_wrapper import create_model_wrappers

            inference_engine = ChronoTickInferenceEngine(self.config_path)
            success = inference_engine.initialize_models()

            if not success:
                logger.error("Failed to initialize ML models!")
                status_queue.put({"status": "error", "error": "Model initialization failed"})
                return

            logger.info("✓ ML models initialized successfully")

            # STEP 2: Initialize real data pipeline (NTP, dataset, metrics)
            logger.info("Initializing ChronoTick real data pipeline...")
            status_queue.put({"status": "initializing_pipeline"})

            real_data_pipeline = RealDataPipeline(self.config_path)

            # STEP 3: Create model wrappers to bridge engine to pipeline
            logger.info("Creating TSFM model wrappers...")
            cpu_wrapper, gpu_wrapper = create_model_wrappers(
                inference_engine=inference_engine,
                dataset_manager=real_data_pipeline.dataset_manager,
                system_metrics=real_data_pipeline.system_metrics
            )

            # STEP 4: Initialize pipeline with models
            logger.info("Connecting ML models to pipeline...")
            # NOTE: initialize() calls set_model_interfaces() internally with pipeline=self
            real_data_pipeline.initialize(cpu_model=cpu_wrapper, gpu_model=gpu_wrapper)

            # STEP 5: Predictive scheduler already configured by initialize()
            logger.info("Predictive scheduler configured by pipeline initialization")

            logger.info("✓ Full ChronoTick integration complete!")
            logger.info("  - Real NTP measurements: ACTIVE")
            logger.info("  - ML clock drift prediction: ACTIVE")
            logger.info("  - System metrics (covariates): ACTIVE")
            logger.info("  - Dual-model architecture: ACTIVE")
            logger.info("  - Prediction fusion: ACTIVE")
            
            # Start warmup phase
            logger.info("Starting warmup phase...")
            status_queue.put({"status": "warmup", "progress": 0.0})
            
            # Start NTP collection
            real_data_pipeline.ntp_collector.start_collection()
            
            # Monitor warmup progress
            warmup_start = time.time()
            warmup_duration = real_data_pipeline.ntp_collector.warm_up_duration
            
            while time.time() - warmup_start < warmup_duration:
                elapsed = time.time() - warmup_start
                progress = elapsed / warmup_duration
                remaining = warmup_duration - elapsed
                
                status_queue.put({
                    "status": "warmup",
                    "progress": progress,
                    "remaining_seconds": remaining
                })
                
                time.sleep(1.0)  # Update every second
            
            # Warmup complete
            logger.info("Warmup complete - daemon ready!")
            status_queue.put({"status": "ready"})
            
            # Main service loop
            request_count = 0
            start_time = time.time()
            total_latency = 0.0
            
            while True:
                try:
                    # Check for shutdown request
                    try:
                        request = request_queue.get_nowait()
                        
                        if request.get("type") == "shutdown":
                            logger.info("Received shutdown request")
                            break
                        elif request.get("type") == "get_time":
                            # Process time request
                            request_start = time.time()
                            
                            # Get real clock correction
                            correction = real_data_pipeline.get_real_clock_correction(request["timestamp"])
                            
                            # Send response
                            response = {
                                "type": "correction",
                                "data": {
                                    "offset_correction": correction.offset_correction,
                                    "drift_rate": correction.drift_rate,
                                    "offset_uncertainty": correction.offset_uncertainty,
                                    "drift_uncertainty": correction.drift_uncertainty,
                                    "prediction_time": correction.prediction_time,
                                    "valid_until": correction.valid_until,
                                    "confidence": correction.confidence,
                                    "source": correction.source
                                }
                            }
                            response_queue.put(response)
                            
                            # Update statistics
                            request_count += 1
                            latency = (time.time() - request_start) * 1000
                            total_latency += latency
                            
                        elif request.get("type") == "get_status":
                            # Send status response
                            uptime = time.time() - start_time
                            avg_latency = total_latency / max(1, request_count)
                            
                            process = psutil.Process()
                            memory_mb = process.memory_info().rss / (1024 * 1024)
                            
                            status_response = {
                                "type": "status",
                                "data": {
                                    "status": "ready",
                                    "warmup_progress": 1.0,
                                    "warmup_remaining_seconds": 0.0,
                                    "total_corrections": request_count,
                                    "success_rate": 1.0,  # TODO: Track failures
                                    "average_latency_ms": avg_latency,
                                    "memory_usage_mb": memory_mb,
                                    "cpu_affinity": list(process.cpu_affinity()),
                                    "uptime_seconds": uptime,
                                    "last_error": None
                                }
                            }
                            response_queue.put(status_response)
                            
                    except mp.queues.Empty:
                        # No request available, continue
                        pass
                    
                    # Brief sleep to prevent busy waiting
                    time.sleep(0.001)  # 1ms sleep for responsive IPC
                    
                except KeyboardInterrupt:
                    logger.info("Received keyboard interrupt")
                    break
                except Exception as e:
                    logger.error(f"Error in daemon main loop: {e}")
                    status_queue.put({"status": "error", "error": str(e)})
                    
        except Exception as e:
            logger.error(f"Failed to initialize daemon: {e}")
            status_queue.put({"status": "error", "error": str(e)})
        finally:
            # Cleanup
            try:
                real_data_pipeline.ntp_collector.stop_collection()
            except:
                pass
            logger.info("Daemon shutdown complete")


def main():
    """CLI interface for the daemon."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ChronoTick Inference Daemon")
    parser.add_argument('command', choices=['start', 'stop', 'status', 'test'],
                       help='Daemon command')
    parser.add_argument('--config', type=str,
                       default='chronotick/inference/config.yaml',
                       help='Configuration file path')
    parser.add_argument('--cpu-affinity', type=int, nargs='+',
                       help='CPU cores to bind to (e.g., --cpu-affinity 0 1 2)')
    parser.add_argument('--daemon', action='store_true',
                       help='Run as background daemon')
    
    args = parser.parse_args()
    
    if args.command == 'test':
        # Test the daemon functionality
        print("Testing ChronoTick Daemon...")
        
        daemon = ChronoTickDaemon(args.config, args.cpu_affinity)
        
        if daemon.start_daemon():
            print("✓ Daemon started successfully")
            
            # Test time requests
            for i in range(5):
                response = daemon.get_corrected_time()
                
                if response.status == "success":
                    print(f"✓ Request {i+1}:")
                    print(f"  Raw time: {response.raw_time:.6f}")
                    print(f"  Corrected time: {response.corrected_time:.6f}")
                    print(f"  Offset correction: {response.offset_correction*1e6:.3f}μs")
                    if response.uncertainty:
                        print(f"  Uncertainty: ±{response.uncertainty*1e6:.3f}μs")
                    if response.lower_bound and response.upper_bound:
                        print(f"  95% bounds: [{response.lower_bound:.6f}, {response.upper_bound:.6f}]")
                    print(f"  Inference time: {response.inference_time*1000:.1f}ms")
                else:
                    print(f"✗ Request {i+1} failed: {response.error}")
                
                time.sleep(1)
            
            # Get status
            status = daemon.get_daemon_status()
            print(f"\n--- Daemon Status ---")
            print(f"✓ Running: {status.running}")
            print(f"✓ PID: {status.pid}")
            print(f"✓ Uptime: {status.uptime:.1f}s")
            print(f"✓ Total requests: {status.total_requests}")
            print(f"✓ Success rate: {status.successful_requests}/{status.total_requests}")
            if status.avg_inference_time:
                print(f"✓ Avg inference time: {status.avg_inference_time*1000:.1f}ms")
            if status.memory_usage_mb:
                print(f"✓ Memory usage: {status.memory_usage_mb:.1f}MB")
            if status.cpu_affinity:
                print(f"✓ CPU affinity: {status.cpu_affinity}")
            
            daemon.stop_daemon()
            print("✓ Daemon stopped")
        else:
            print("✗ Failed to start daemon")
    
    elif args.command == 'start':
        daemon = ChronoTickDaemon(args.config, args.cpu_affinity)
        
        if daemon.start_daemon():
            print(f"ChronoTick daemon started with PID {daemon.process.pid}")
            print("Use 'python daemon.py stop' to stop the daemon")
            
            if not args.daemon:
                try:
                    # Keep running until interrupted
                    while daemon.running:
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("\nShutting down daemon...")
                    daemon.stop_daemon()
        else:
            print("Failed to start daemon")
            sys.exit(1)
    
    elif args.command == 'stop':
        # This would need to be implemented with PID files for production
        print("Stop command would stop running daemon (not implemented)")
    
    elif args.command == 'status':
        # This would need to connect to running daemon for production
        print("Status command would show running daemon status (not implemented)")


if __name__ == "__main__":
    main()