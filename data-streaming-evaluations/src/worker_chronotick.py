#!/usr/bin/env python3
"""
Worker with Embedded ChronoTick: Receives events and timestamps with NTP + ChronoTick

Runs on ares-comp-11 and ares-comp-12
Receives broadcast events from coordinator
Timestamps each event with both NTP and ChronoTick (embedded inference engine)
Records uncertainty evolution for commit-wait analysis
"""

import argparse
import time
import csv
import signal
import sys
from pathlib import Path
from typing import Optional
import logging

# Data streaming imports
from src.common import (
    Event,
    UDPListener,
    NTPClient,
    setup_logging
)

# ChronoTick imports
from chronotick.inference.engine import ChronoTickInferenceEngine
from chronotick.inference.real_data_pipeline import RealDataPipeline
from chronotick.inference.tsfm_model_wrapper import create_model_wrappers


class ChronoTickWorker:
    """Worker with embedded ChronoTick inference engine"""

    def __init__(self,
                 node_id: str,
                 listen_port: int,
                 ntp_server: str,
                 chronotick_config: Path,
                 output_file: Path):

        self.node_id = node_id
        self.output_file = output_file
        self.logger = logging.getLogger(f"{__name__}.ChronoTickWorker.{node_id}")

        # Network
        self.logger.info(f"Initializing UDP listener on port {listen_port}...")
        self.listener = UDPListener(listen_port)

        # NTP Client (for reference timestamps)
        self.logger.info(f"Initializing NTP client (servers: {ntp_server})...")
        ntp_servers = ntp_server.split(',')
        self.ntp_client = NTPClient(ntp_servers)

        # ChronoTick System
        self.logger.info(f"Initializing ChronoTick system (config: {chronotick_config})...")
        self.engine = ChronoTickInferenceEngine(str(chronotick_config))
        self.logger.info("Loading ChronoTick models...")
        self.engine.initialize_models()
        self.logger.info("✓ ChronoTick models loaded")

        self.pipeline = RealDataPipeline(str(chronotick_config))
        cpu_wrapper, gpu_wrapper = create_model_wrappers(
            self.engine,
            self.pipeline.dataset_manager,
            self.pipeline.system_metrics
        )
        self.pipeline.initialize(cpu_model=cpu_wrapper, gpu_model=gpu_wrapper)
        self.logger.info("✓ ChronoTick pipeline initialized")

        # CSV output
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.csv_file = open(self.output_file, 'w', newline='')

        # CSV schema: NTP + ChronoTick fields
        fieldnames = [
            'event_id',
            'node_id',
            'sequence_number',
            'receive_time_ns',
            'coordinator_send_time_ns',
            # NTP reference
            'ntp_offset_ms',
            'ntp_uncertainty_ms',
            'ntp_timestamp_ns',
            # ChronoTick
            'ct_offset_ms',
            'ct_drift_rate',
            'ct_uncertainty_ms',
            'ct_confidence',
            'ct_source',
            'ct_prediction_time',
            'ct_timestamp_ns',
            'ct_lower_bound_ns',
            'ct_upper_bound_ns',
        ]

        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.csv_writer.writeheader()
        self.csv_file.flush()

        # Statistics
        self.events_received = 0
        self.events_processed = 0

        # Graceful shutdown
        self.running = True
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False

    def _warmup(self):
        """Warmup ChronoTick and NTP"""
        warmup_duration = self.pipeline.ntp_collector.warm_up_duration
        self.logger.info("="*60)
        self.logger.info("WARMUP PHASE: Initializing ChronoTick and NTP")
        self.logger.info(f"Duration: {warmup_duration} seconds")
        self.logger.info("="*60)

        for i in range(warmup_duration):
            if not self.running:
                break

            try:
                current_time = time.time()
                correction = self.pipeline.get_real_clock_correction(current_time)

                if i % 10 == 0:
                    self.logger.info(
                        f"Warmup [{i:3d}s]: ChronoTick offset={correction.offset_correction*1000:+7.2f}ms, "
                        f"uncertainty={correction.offset_uncertainty*1000:5.2f}ms, "
                        f"source={correction.source}"
                    )
            except Exception as e:
                self.logger.warning(f"Warmup [{i:3d}s]: ChronoTick query failed: {e}")

            time.sleep(1)

        self.logger.info("="*60)
        self.logger.info("WARMUP COMPLETE - Worker ready to receive events")
        self.logger.info("="*60)

    def _process_event(self, event: Event, receive_time_ns: int):
        """Process a received event with NTP + ChronoTick timestamps"""
        try:
            receive_time_s = receive_time_ns / 1e9

            # Query NTP (reference timestamp)
            ntp_offset_ms, ntp_uncertainty_ms = self.ntp_client.query()
            ntp_timestamp_ns = receive_time_ns + int(ntp_offset_ms * 1_000_000)

            # Query ChronoTick
            correction = self.pipeline.get_real_clock_correction(receive_time_s)

            # Extract ChronoTick data
            ct_offset_ms = correction.offset_correction * 1000
            ct_drift_rate = correction.drift_rate
            ct_uncertainty_ms = correction.offset_uncertainty * 1000
            ct_confidence = correction.confidence
            ct_source = correction.source
            ct_prediction_time = correction.prediction_time

            # Calculate ChronoTick timestamp and bounds
            ct_timestamp_ns = receive_time_ns + int(ct_offset_ms * 1_000_000)
            ct_lower_ns = ct_timestamp_ns - int(3 * ct_uncertainty_ms * 1_000_000)
            ct_upper_ns = ct_timestamp_ns + int(3 * ct_uncertainty_ms * 1_000_000)

            # Build record
            record = {
                'event_id': event.event_id,
                'node_id': self.node_id,
                'sequence_number': event.sequence_number,
                'receive_time_ns': receive_time_ns,
                'coordinator_send_time_ns': event.coordinator_timestamp_ns,
                # NTP
                'ntp_offset_ms': ntp_offset_ms,
                'ntp_uncertainty_ms': ntp_uncertainty_ms,
                'ntp_timestamp_ns': ntp_timestamp_ns,
                # ChronoTick
                'ct_offset_ms': ct_offset_ms,
                'ct_drift_rate': ct_drift_rate,
                'ct_uncertainty_ms': ct_uncertainty_ms,
                'ct_confidence': ct_confidence,
                'ct_source': ct_source,
                'ct_prediction_time': ct_prediction_time,
                'ct_timestamp_ns': ct_timestamp_ns,
                'ct_lower_bound_ns': ct_lower_ns,
                'ct_upper_bound_ns': ct_upper_ns,
            }

            # Write to CSV
            self.csv_writer.writerow(record)
            self.csv_file.flush()

            self.events_processed += 1

            # Log progress
            if self.events_processed % 10 == 0:
                self.logger.info(
                    f"Progress: {self.events_processed} events | "
                    f"NTP: {ntp_offset_ms:+.2f}±{ntp_uncertainty_ms:.2f}ms | "
                    f"ChronoTick: {ct_offset_ms:+.2f}±{ct_uncertainty_ms:.2f}ms ({ct_source})"
                )

        except Exception as e:
            self.logger.error(f"Failed to process event {event.event_id}: {e}", exc_info=True)

    def run(self):
        """Main worker loop"""
        self.logger.info("="*60)
        self.logger.info(f"ChronoTick Worker {self.node_id} starting")
        self.logger.info(f"Output: {self.output_file}")
        self.logger.info("="*60)

        # Warmup phase
        self._warmup()

        if not self.running:
            self.logger.info("Shutdown requested during warmup")
            return

        # Main event processing loop
        self.logger.info("Listening for events...")
        start_time = time.time()

        while self.running:
            try:
                # Receive event (blocking with timeout)
                self.listener.sock.settimeout(1.0)
                event, receive_time_ns = self.listener.receive()

                self.events_received += 1
                self.logger.debug(f"Received event {event.event_id}")

                # Process event
                self._process_event(event, receive_time_ns)

            except TimeoutError:
                # No event received, check if we should continue
                continue
            except Exception as e:
                self.logger.error(f"Error receiving event: {e}", exc_info=True)
                continue

        # Shutdown
        duration = time.time() - start_time
        self.logger.info("="*60)
        self.logger.info("ChronoTick Worker shutting down")
        self.logger.info(f"Duration: {duration:.1f}s")
        self.logger.info(f"Events received: {self.events_received}")
        self.logger.info(f"Events processed: {self.events_processed}")
        self.logger.info(f"Success rate: {self.events_processed/max(1, self.events_received)*100:.1f}%")
        self.logger.info("="*60)

        # Cleanup
        self.csv_file.close()
        self.listener.close()


def main():
    parser = argparse.ArgumentParser(description='ChronoTick Worker (Embedded Inference)')
    parser.add_argument('--node-id', type=str, required=True,
                       help='Node identifier (e.g., comp11)')
    parser.add_argument('--listen-port', type=int, required=True,
                       help='UDP port to listen on')
    parser.add_argument('--ntp-server', type=str, required=True,
                       help='NTP server(s) for reference timestamps (comma-separated)')
    parser.add_argument('--chronotick-config', type=Path, required=True,
                       help='ChronoTick configuration YAML file')
    parser.add_argument('--output', type=Path, required=True,
                       help='Output CSV file')
    parser.add_argument('--log-file', type=Path, default=None,
                       help='Log file (default: stdout only)')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    # Setup logging
    setup_logging(log_file=args.log_file, level=args.log_level)

    # Create and run ChronoTick worker
    worker = ChronoTickWorker(
        node_id=args.node_id,
        listen_port=args.listen_port,
        ntp_server=args.ntp_server,
        chronotick_config=args.chronotick_config,
        output_file=args.output
    )

    worker.run()


if __name__ == '__main__':
    main()
