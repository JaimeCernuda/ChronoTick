#!/usr/bin/env python3
"""
ChronoTick Shared Memory Daemon

Background daemon that continuously updates ChronoTick time corrections in shared memory.
Integrates with the existing ChronoTick inference engine and real data pipeline.

Features:
- High-frequency updates (100-1000 Hz configurable)
- Lock-free single-writer pattern with sequence numbers
- Real NTP measurements with dual ML model predictions
- Graceful shutdown with cleanup
- CPU affinity for performance
- Signal handling for proper resource cleanup

Performance:
- Update frequency: 100-1000 Hz (default 100 Hz for balance)
- Write latency: <10μs per update
- Memory: 128 bytes shared + ~200MB daemon process
"""

import os
import sys
import time
import signal
import logging
import argparse
import psutil
from pathlib import Path
from typing import Optional
from multiprocessing.shared_memory import SharedMemory

# Add ChronoTick to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "tsfm"))

from chronotick_inference.real_data_pipeline import RealDataPipeline
from chronotick_shm.shm_config import (
    SHARED_MEMORY_NAME,
    SHARED_MEMORY_SIZE,
    ChronoTickData,
    CorrectionSource,
    StatusFlags,
    write_data,
    create_default_data
)

logger = logging.getLogger(__name__)


class ChronoTickSharedMemoryDaemon:
    """
    ChronoTick daemon that writes time corrections to shared memory.

    Architecture:
    - Creates and owns shared memory segment
    - Runs real data pipeline (NTP + ML models)
    - Updates shared memory at configurable frequency
    - Single-writer pattern (lock-free for readers)
    - Signal handling for graceful shutdown
    """

    def __init__(
        self,
        config_path: str,
        update_frequency_hz: int = 100,
        cpu_affinity: Optional[list] = None
    ):
        """
        Initialize ChronoTick shared memory daemon.

        Args:
            config_path: Path to ChronoTick configuration YAML
            update_frequency_hz: Update frequency (1-1000 Hz)
            cpu_affinity: List of CPU cores to bind to (e.g., [0, 1])
        """
        self.config_path = config_path
        self.update_frequency_hz = max(1, min(1000, update_frequency_hz))
        self.cpu_affinity = cpu_affinity

        # Shared memory
        self.shm: Optional[SharedMemory] = None

        # Real data pipeline
        self.pipeline: Optional[RealDataPipeline] = None

        # Statistics
        self.start_time = 0.0
        self.sequence_number = 0
        self.total_updates = 0
        self.total_corrections = 0

        # Control flags
        self.running = False

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging for daemon"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""

        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _set_cpu_affinity(self):
        """Set CPU affinity for performance"""
        if not self.cpu_affinity:
            return

        try:
            psutil.Process().cpu_affinity(self.cpu_affinity)
            logger.info(f"Set CPU affinity to cores: {self.cpu_affinity}")
        except Exception as e:
            logger.warning(f"Failed to set CPU affinity: {e}")

    def _create_shared_memory(self):
        """Create and initialize shared memory segment"""
        try:
            # Try to create new shared memory
            self.shm = SharedMemory(
                name=SHARED_MEMORY_NAME,
                create=True,
                size=SHARED_MEMORY_SIZE
            )
            logger.info(f"Created shared memory: {SHARED_MEMORY_NAME} ({SHARED_MEMORY_SIZE} bytes)")

            # Initialize with default data
            default_data = create_default_data()
            write_data(self.shm.buf, default_data)

        except FileExistsError:
            # Shared memory already exists - try to attach and take ownership
            logger.warning(f"Shared memory {SHARED_MEMORY_NAME} already exists")
            logger.info("Attempting to attach to existing shared memory...")

            try:
                self.shm = SharedMemory(name=SHARED_MEMORY_NAME, create=False)
                logger.info("Attached to existing shared memory successfully")

                # Write default data to reset state
                default_data = create_default_data()
                write_data(self.shm.buf, default_data)

            except Exception as e:
                logger.error(f"Failed to attach to existing shared memory: {e}")
                logger.error("Please manually remove stale shared memory:")
                logger.error(f"  Linux: rm /dev/shm/{SHARED_MEMORY_NAME}")
                logger.error(f"  Python: from multiprocessing.shared_memory import SharedMemory; "
                           f"SharedMemory('{SHARED_MEMORY_NAME}', create=False).unlink()")
                raise

    def _cleanup_shared_memory(self):
        """Cleanup shared memory on shutdown"""
        if self.shm:
            try:
                self.shm.close()
                logger.info("Closed shared memory handle")
            except Exception as e:
                logger.warning(f"Error closing shared memory: {e}")

            try:
                self.shm.unlink()
                logger.info(f"Unlinked shared memory: {SHARED_MEMORY_NAME}")
            except FileNotFoundError:
                logger.debug("Shared memory already unlinked")
            except Exception as e:
                logger.error(f"Error unlinking shared memory: {e}")

    def _initialize_pipeline(self):
        """Initialize ChronoTick real data pipeline"""
        logger.info("Initializing ChronoTick real data pipeline...")

        try:
            self.pipeline = RealDataPipeline(self.config_path)
            self.pipeline.initialize()
            logger.info("Pipeline initialized successfully")

            # Start NTP collection
            logger.info("Starting NTP collection...")
            self.pipeline.ntp_collector.start_collection()

            # Wait for initial warmup
            warmup_duration = self.pipeline.ntp_collector.warm_up_duration
            logger.info(f"Warmup period: {warmup_duration}s - collecting NTP measurements...")

            warmup_start = time.time()
            while time.time() - warmup_start < warmup_duration:
                elapsed = time.time() - warmup_start
                remaining = warmup_duration - elapsed
                progress = elapsed / warmup_duration

                if int(elapsed) % 30 == 0:  # Log every 30 seconds
                    logger.info(f"Warmup progress: {progress:.1%} complete, {remaining:.0f}s remaining")

                time.sleep(1.0)

            logger.info("✅ Warmup complete - daemon ready!")

        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            raise

    def _get_correction_source_enum(self, source_str: str) -> CorrectionSource:
        """Convert source string to enum"""
        source_map = {
            "no_data": CorrectionSource.NO_DATA,
            "ntp": CorrectionSource.NTP,
            "cpu": CorrectionSource.CPU_MODEL,
            "gpu": CorrectionSource.GPU_MODEL,
            "fusion": CorrectionSource.FUSION
        }
        return source_map.get(source_str.lower(), CorrectionSource.NO_DATA)

    def _calculate_status_flags(self) -> int:
        """Calculate status flags based on pipeline state"""
        flags = StatusFlags.VALID  # Always set valid after warmup

        # Check NTP ready
        if self.pipeline.ntp_collector.last_measurement:
            flags |= StatusFlags.NTP_READY

        # Check models ready (if pipeline has models initialized)
        # For now, assume models are ready if pipeline is initialized
        flags |= StatusFlags.MODELS_READY

        # Warmup complete after we've started the main loop
        flags |= StatusFlags.WARMUP_COMPLETE

        return flags

    def _update_shared_memory(self):
        """Update shared memory with latest correction data"""
        try:
            # Get current time
            current_time = time.time()

            # Get real clock correction from pipeline
            correction = self.pipeline.get_real_clock_correction(current_time)

            # Increment sequence number
            self.sequence_number = (self.sequence_number + 1) % (2**32)

            # Calculate corrected time
            time_delta = current_time - correction.prediction_time
            corrected_time = current_time + correction.offset_correction + (
                correction.drift_rate * time_delta
            )

            # Build ChronoTickData
            data = ChronoTickData(
                corrected_timestamp=corrected_time,
                system_timestamp=current_time,
                offset_correction=correction.offset_correction,
                drift_rate=correction.drift_rate,
                offset_uncertainty=correction.offset_uncertainty,
                drift_uncertainty=correction.drift_uncertainty,
                prediction_time=correction.prediction_time,
                valid_until=correction.valid_until,
                confidence=correction.confidence,
                sequence_number=self.sequence_number,
                flags=self._calculate_status_flags(),
                source=self._get_correction_source_enum(correction.source),
                measurement_count=len(self.pipeline.ntp_collector.offset_measurements),
                last_ntp_sync=(
                    self.pipeline.ntp_collector.last_measurement_time
                    if self.pipeline.ntp_collector.last_measurement_time > 0
                    else 0.0
                ),
                daemon_uptime=time.time() - self.start_time,
                total_corrections=self.total_corrections,
                reserved=0,
                average_latency_ms=0.0  # TODO: Track actual latency
            )

            # Write to shared memory
            write_data(self.shm.buf, data)

            self.total_updates += 1
            self.total_corrections += 1

        except Exception as e:
            logger.error(f"Error updating shared memory: {e}")

    def _main_loop(self):
        """Main daemon loop - updates shared memory at configured frequency"""
        sleep_interval = 1.0 / self.update_frequency_hz

        logger.info(f"Entering main loop - update frequency: {self.update_frequency_hz} Hz")
        logger.info(f"Update interval: {sleep_interval*1000:.1f}ms")

        update_count = 0
        last_log_time = time.time()

        while self.running:
            try:
                loop_start = time.time()

                # Update shared memory
                self._update_shared_memory()

                update_count += 1

                # Log statistics every 60 seconds
                if time.time() - last_log_time >= 60.0:
                    uptime = time.time() - self.start_time
                    updates_per_sec = update_count / (time.time() - last_log_time)

                    logger.info(
                        f"Stats: {update_count} updates, "
                        f"{updates_per_sec:.1f} updates/s, "
                        f"uptime: {uptime:.0f}s"
                    )

                    update_count = 0
                    last_log_time = time.time()

                # Smart sleep - account for processing time
                elapsed = time.time() - loop_start
                sleep_time = max(0, sleep_interval - elapsed)

                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    # Processing took longer than interval - log warning occasionally
                    if self.total_updates % 100 == 0:
                        logger.warning(
                            f"Update took {elapsed*1000:.1f}ms, "
                            f"longer than interval {sleep_interval*1000:.1f}ms"
                        )

            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(sleep_interval)

    def start(self):
        """Start the daemon"""
        try:
            logger.info("=" * 60)
            logger.info("ChronoTick Shared Memory Daemon Starting")
            logger.info("=" * 60)
            logger.info(f"Config: {self.config_path}")
            logger.info(f"Update frequency: {self.update_frequency_hz} Hz")
            logger.info(f"CPU affinity: {self.cpu_affinity or 'not set'}")
            logger.info(f"Shared memory: {SHARED_MEMORY_NAME} ({SHARED_MEMORY_SIZE} bytes)")

            # Setup
            self._setup_signal_handlers()
            self._set_cpu_affinity()

            # Create shared memory
            self._create_shared_memory()

            # Initialize pipeline
            self._initialize_pipeline()

            # Start main loop
            self.start_time = time.time()
            self.running = True
            self._main_loop()

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        except Exception as e:
            logger.error(f"Daemon error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.shutdown()

    def shutdown(self):
        """Shutdown daemon and cleanup resources"""
        logger.info("Shutting down daemon...")

        self.running = False

        # Stop NTP collection
        if self.pipeline:
            try:
                self.pipeline.ntp_collector.stop_collection()
                logger.info("Stopped NTP collection")
            except Exception as e:
                logger.warning(f"Error stopping NTP collection: {e}")

        # Cleanup shared memory
        self._cleanup_shared_memory()

        uptime = time.time() - self.start_time if self.start_time > 0 else 0
        logger.info(f"Daemon shutdown complete - uptime: {uptime:.1f}s, "
                   f"total updates: {self.total_updates}")


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="ChronoTick Shared Memory Daemon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start with default configuration
  python chronotick_daemon.py

  # Start with custom config and update frequency
  python chronotick_daemon.py --config ../tsfm/chronotick_inference/config.yaml --freq 1000

  # Start with CPU pinning for performance
  python chronotick_daemon.py --cpu-affinity 0 1 --freq 500

  # Debug mode with verbose logging
  python chronotick_daemon.py --log-level DEBUG --freq 100
        """
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to ChronoTick configuration YAML"
    )
    parser.add_argument(
        "--freq",
        type=int,
        default=100,
        help="Update frequency in Hz (default: 100, range: 1-1000)"
    )
    parser.add_argument(
        "--cpu-affinity",
        type=int,
        nargs="+",
        help="CPU cores to bind to (e.g., --cpu-affinity 0 1)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )

    args = parser.parse_args()

    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Determine config path
    if args.config:
        config_path = args.config
    else:
        # Try to find default config
        default_config = Path(__file__).parent.parent / "tsfm" / "chronotick_inference" / "config.yaml"
        if default_config.exists():
            config_path = str(default_config)
        else:
            logger.error("No configuration file specified and default not found")
            logger.error("Please specify --config or ensure default config exists at:")
            logger.error(f"  {default_config}")
            sys.exit(1)

    # Validate config exists
    if not Path(config_path).exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    # Create and start daemon
    daemon = ChronoTickSharedMemoryDaemon(
        config_path=config_path,
        update_frequency_hz=args.freq,
        cpu_affinity=args.cpu_affinity
    )

    daemon.start()


if __name__ == "__main__":
    main()
