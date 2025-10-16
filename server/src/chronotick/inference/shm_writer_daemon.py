#!/usr/bin/env python3
"""
ChronoTick Shared Memory Writer Daemon

Simple daemon that runs RealDataPipeline and writes corrections to shared memory
for Python clients to read.

This bridges the real_data_pipeline with the shared memory interface that clients expect.
"""

import time
import sys
import signal
import logging
from pathlib import Path
from multiprocessing.shared_memory import SharedMemory

from shm_config import (
    SHARED_MEMORY_NAME, SHARED_MEMORY_SIZE, ChronoTickData,
    CorrectionSource, StatusFlags, write_data
)

# Import real data pipeline
from real_data_pipeline import RealDataPipeline

# Import model wrappers
from tsfm_model_wrapper import create_model_wrappers
from engine import ChronoTickInferenceEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('SHMWriter')


class SharedMemoryWriterDaemon:
    """
    Daemon that writes ChronoTick corrections to shared memory.

    Clients can read from this shared memory for high-performance time synchronization.
    """

    def __init__(self, config_path: str):
        """Initialize the daemon."""
        self.config_path = config_path
        self.shm = None
        self.pipeline = None
        self.running = False
        self.start_time = time.time()

    def start(self):
        """Start the daemon."""
        logger.info("Starting ChronoTick Shared Memory Writer Daemon")
        logger.info(f"Config: {self.config_path}")

        try:
            # Create shared memory
            logger.info(f"Creating shared memory: {SHARED_MEMORY_NAME}")
            try:
                # Try to unlink if it exists
                existing_shm = SharedMemory(name=SHARED_MEMORY_NAME, create=False)
                existing_shm.close()
                existing_shm.unlink()
                logger.info("Cleaned up existing shared memory")
            except FileNotFoundError:
                pass

            self.shm = SharedMemory(
                name=SHARED_MEMORY_NAME,
                create=True,
                size=SHARED_MEMORY_SIZE
            )
            logger.info(f"✓ Shared memory created: {SHARED_MEMORY_SIZE} bytes")

            # Initialize with zeros
            self.shm.buf[:] = bytes(SHARED_MEMORY_SIZE)

            # Initialize inference engine
            logger.info("Initializing inference engine...")
            engine = ChronoTickInferenceEngine(self.config_path)
            success = engine.initialize_models()

            if not success:
                raise RuntimeError("Failed to initialize ML models")

            logger.info("✓ ML models initialized")

            # Initialize real data pipeline
            logger.info("Initializing real data pipeline...")
            self.pipeline = RealDataPipeline(self.config_path)

            # Create model wrappers
            logger.info("Creating TSFM model wrappers...")
            cpu_wrapper, gpu_wrapper = create_model_wrappers(
                inference_engine=engine,
                dataset_manager=self.pipeline.dataset_manager,
                system_metrics=self.pipeline.system_metrics
            )

            # Initialize pipeline with models
            logger.info("Connecting ML models to pipeline...")
            self.pipeline.initialize(cpu_model=cpu_wrapper, gpu_model=gpu_wrapper)

            logger.info("✓ ChronoTick daemon ready!")
            logger.info("  - Shared memory: /dev/shm/chronotick_shm")
            logger.info("  - Clients can now connect")

            # Register signal handlers
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

            # Main loop
            self.running = True
            self._main_loop()

        except Exception as e:
            logger.error(f"Failed to start daemon: {e}")
            import traceback
            traceback.print_exc()
            self.cleanup()
            sys.exit(1)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

    def _main_loop(self):
        """Main loop that writes corrections to shared memory."""
        update_interval = 1.0  # Update shared memory every 1 second
        next_update = time.time()

        correction_count = 0

        while self.running:
            try:
                current_time = time.time()

                if current_time >= next_update:
                    # Get correction from pipeline
                    correction = self.pipeline.get_real_clock_correction(current_time)

                    # Determine source enum
                    source_map = {
                        'ntp': CorrectionSource.NTP,
                        'ntp_warm_up': CorrectionSource.NTP,
                        'cpu': CorrectionSource.CPU_MODEL,
                        'gpu': CorrectionSource.GPU_MODEL,
                        'fusion': CorrectionSource.FUSION,
                    }
                    source = source_map.get(correction.source, CorrectionSource.NO_DATA)

                    # Build flags
                    flags = StatusFlags.VALID | StatusFlags.NTP_READY | StatusFlags.MODELS_READY
                    if self.pipeline.warm_up_complete:
                        flags |= StatusFlags.WARMUP_COMPLETE

                    # Create ChronoTickData
                    data = ChronoTickData(
                        corrected_timestamp=current_time + correction.offset_correction,
                        system_timestamp=current_time,
                        offset_correction=correction.offset_correction,
                        drift_rate=correction.drift_rate,
                        offset_uncertainty=correction.offset_uncertainty,
                        drift_uncertainty=correction.drift_uncertainty,
                        prediction_time=correction.prediction_time,
                        valid_until=current_time + 60.0,  # Valid for 60 seconds
                        confidence=correction.confidence,
                        sequence_number=correction_count,
                        flags=flags,
                        source=source,
                        measurement_count=self.pipeline.stats['ntp_measurements'],
                        last_ntp_sync=self.pipeline.last_ntp_time,
                        daemon_uptime=time.time() - self.start_time,
                        total_corrections=self.pipeline.stats['total_corrections'],
                        reserved=0,
                        average_latency_ms=1.0  # Approximate
                    )

                    # Write to shared memory
                    write_data(self.shm.buf, data)

                    correction_count += 1

                    # Log periodically
                    if correction_count % 10 == 0:
                        logger.info(f"Corrections written: {correction_count}, "
                                  f"offset={correction.offset_correction*1000:.3f}ms, "
                                  f"uncertainty=±{correction.offset_uncertainty*1000:.3f}ms, "
                                  f"source={correction.source}")

                    next_update += update_interval

                # Sleep briefly
                time.sleep(0.01)  # 10ms

            except KeyboardInterrupt:
                logger.info("Interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1.0)

        self.cleanup()

    def cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up...")

        if self.pipeline:
            try:
                self.pipeline.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down pipeline: {e}")

        if self.shm:
            try:
                self.shm.close()
                self.shm.unlink()
                logger.info("✓ Shared memory cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up shared memory: {e}")

        logger.info("Daemon shutdown complete")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="ChronoTick Shared Memory Writer Daemon"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config_enhanced_features.yaml',
        help='Path to configuration file'
    )

    args = parser.parse_args()

    # Resolve config path
    config_path = Path(args.config)
    if not config_path.is_absolute():
        # Try relative to project root
        project_root = Path(__file__).parent.parent.parent.parent.parent
        config_path = project_root / args.config

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    logger.info(f"Using config: {config_path}")

    daemon = SharedMemoryWriterDaemon(str(config_path))
    daemon.start()


if __name__ == '__main__':
    main()
