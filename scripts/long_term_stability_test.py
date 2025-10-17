#!/usr/bin/env python3
"""
Long-term stability test for ChronoTick.
Continuously measures ChronoTick, system clock, and NTP for overnight stability analysis.

This test runs indefinitely and logs to:
- CSV: {output_dir}/chronotick_stability_{timestamp}.csv
- Log: {output_dir}/test.log
"""

import time
import sys
import argparse
from pathlib import Path
from datetime import datetime
import ntplib
import csv
import logging

# Setup paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from chronotick.inference.engine import ChronoTickInferenceEngine
from chronotick.inference.real_data_pipeline import RealDataPipeline
from chronotick.inference.tsfm_model_wrapper import create_model_wrappers

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Long-term stability test for ChronoTick')
    parser.add_argument('--config', default='configs/config_test_drift_aware.yaml', help='Config file path')
    parser.add_argument('--duration', type=int, help='Test duration in seconds (optional)')
    parser.add_argument('--output-dir', default='results/long_term_stability', help='Output directory for results')
    args = parser.parse_args()

    # Create results directory
    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # CSV output with timestamp
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = results_dir / f"chronotick_stability_{timestamp_str}.csv"

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(results_dir / 'test.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)

    # NTP client
    ntp_client = ntplib.NTPClient()
    ntp_servers = ["pool.ntp.org", "time.google.com", "time.nist.gov"]

    def get_ntp_offset():
        """Get NTP offset, try multiple servers"""
        for server in ntp_servers:
            try:
                response = ntp_client.request(server, version=3, timeout=2)
                return response.offset, response.delay, server
            except Exception as e:
                continue
        return None, None, None

    logger.info("=" * 80)
    logger.info("CHRONOTICK LONG-TERM STABILITY TEST - STARTING")
    logger.info("=" * 80)
    logger.info(f"Results will be saved to: {csv_path}")
    logger.info(f"Output directory: {results_dir}")
    logger.info("Test will run indefinitely. Press Ctrl+C to stop.")
    logger.info("")

    # Initialize ChronoTick (proper way, like working test scripts)
    config_path = args.config
    logger.info(f"Loading configuration from {config_path}")

    # Initialize engine with models
    logger.info("Loading ChronoTick Inference Engine with models...")
    engine = ChronoTickInferenceEngine(config_path)
    engine.initialize_models()
    logger.info("✓ Models loaded")

    # Initialize pipeline
    pipeline = RealDataPipeline(config_path)
    cpu_wrapper, gpu_wrapper = create_model_wrappers(
        engine, pipeline.dataset_manager, pipeline.system_metrics
    )
    # NOTE: pipeline.initialize() now injects dataset_manager into scheduler automatically
    pipeline.initialize(cpu_model=cpu_wrapper, gpu_model=gpu_wrapper)
    logger.info("✓ Pipeline initialized")

    logger.info("Waiting for warm-up phase (60 seconds)...")
    time.sleep(60)
    logger.info("Warm-up complete. Starting data collection.")

    # CSV header
    fieldnames = [
        'timestamp', 'elapsed_seconds', 'datetime',
        'system_time', 'chronotick_time',
        'chronotick_offset_ms', 'chronotick_uncertainty_ms',
        'chronotick_confidence', 'chronotick_source',
        'ntp_offset_ms', 'ntp_delay_ms', 'ntp_server',
        'chronotick_error_vs_ntp_ms', 'system_error_vs_ntp_ms'
    ]

    start_time = time.time()
    sample_count = 0

    try:
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            csvfile.flush()

            logger.info("Starting indefinite data collection...")
            logger.info("Sampling every 10 seconds")

            while True:
                try:
                    # Get current time
                    current_time = time.time()
                    elapsed = current_time - start_time

                    # Get ChronoTick correction
                    correction = pipeline.get_real_clock_correction(current_time)
                    chronotick_time = current_time + correction.offset_correction

                    # Get NTP offset (every 10 samples to avoid rate limiting)
                    ntp_offset = None
                    ntp_delay = None
                    ntp_server = None
                    chronotick_error = None
                    system_error = None

                    if sample_count % 10 == 0:  # Every 100 seconds
                        ntp_offset, ntp_delay, ntp_server = get_ntp_offset()
                        if ntp_offset is not None:
                            # Calculate errors vs NTP
                            chronotick_error = (chronotick_time - (current_time + ntp_offset)) * 1000  # ms
                            system_error = -ntp_offset * 1000  # ms

                            # LAYER 5: Early Warning System
                            abs_error = abs(chronotick_error)
                            if abs_error > 500:  # 500ms warning threshold
                                logger.warning(f"⚠️ [LAYER 5] DIVERGENCE WARNING: Error = {chronotick_error:.1f}ms (threshold: 500ms)")
                                logger.warning(f"   ChronoTick offset: {correction.offset_correction*1000:.1f}ms")
                                logger.warning(f"   NTP offset: {ntp_offset*1000:.1f}ms")
                                logger.warning(f"   Source: {correction.source}")

                            if abs_error > 2000:  # 2 second critical threshold
                                logger.critical(f"🛑 [LAYER 5] CRITICAL DIVERGENCE: Error = {chronotick_error:.1f}ms")
                                logger.critical(f"   System is unstable - STOPPING TEST to prevent data corruption!")
                                logger.critical(f"   ChronoTick: {correction.offset_correction*1000:.1f}ms")
                                logger.critical(f"   NTP: {ntp_offset*1000:.1f}ms")
                                logger.critical(f"   Samples collected: {sample_count}")
                                logger.critical(f"   Runtime: {elapsed/3600:.2f} hours")
                                raise RuntimeError(f"Test stopped due to critical divergence: {abs_error:.1f}ms")

                    # Write row
                    row = {
                        'timestamp': current_time,
                        'elapsed_seconds': elapsed,
                        'datetime': datetime.fromtimestamp(current_time).isoformat(),
                        'system_time': current_time,
                        'chronotick_time': chronotick_time,
                        'chronotick_offset_ms': correction.offset_correction * 1000,
                        'chronotick_uncertainty_ms': correction.offset_uncertainty * 1000,
                        'chronotick_confidence': correction.confidence,
                        'chronotick_source': correction.source,
                        'ntp_offset_ms': ntp_offset * 1000 if ntp_offset else '',
                        'ntp_delay_ms': ntp_delay * 1000 if ntp_delay else '',
                        'ntp_server': ntp_server or '',
                        'chronotick_error_vs_ntp_ms': f"{chronotick_error:.2f}" if chronotick_error else '',
                        'system_error_vs_ntp_ms': f"{system_error:.2f}" if system_error else ''
                    }

                    writer.writerow(row)
                    csvfile.flush()

                    sample_count += 1

                    # Log progress every 60 samples (10 minutes)
                    if sample_count % 60 == 0:
                        hours = elapsed / 3600
                        logger.info(f"[{sample_count} samples] Running for {hours:.2f} hours | "
                                  f"ChronoTick offset: {correction.offset_correction*1000:.2f}ms | "
                                  f"Source: {correction.source}")

                    # Sleep for 10 seconds
                    time.sleep(10)

                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    logger.error(f"Error in collection loop: {e}", exc_info=True)
                    time.sleep(10)

    except KeyboardInterrupt:
        logger.info("\n" + "=" * 80)
        logger.info("TEST STOPPED BY USER")
        logger.info("=" * 80)

    finally:
        logger.info(f"Collected {sample_count} samples over {elapsed/3600:.2f} hours")
        logger.info(f"Results saved to: {csv_path}")

        try:
            logger.info("Cleaning up...")
            engine.shutdown()
            pipeline.system_metrics.stop_collection()
            logger.info("✓ Test complete - shutdown successful")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

if __name__ == "__main__":
    main()
