#!/usr/bin/env python3
"""
Worker (Task B/C): Receives events and timestamps with NTP + ChronoTick

Runs on ares-comp-11 and ares-comp-12
Receives broadcast events from coordinator
Timestamps each event with both NTP and ChronoTick
Records uncertainty evolution for commit-wait analysis
"""

import argparse
import time
import csv
import threading
import signal
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from collections import deque
import logging

from common import (
    Event,
    TimestampRecord,
    UDPListener,
    NTPClient,
    ChronoTickClient,
    setup_logging
)


class CommitWaitTracker:
    """Tracks uncertainty evolution for commit-wait analysis"""

    def __init__(self, chronotick_client: ChronoTickClient, delays: list[int]):
        self.chronotick_client = chronotick_client
        self.delays = delays  # e.g., [30, 60]
        self.pending_measurements = deque()
        self.results = {}
        self.logger = logging.getLogger(f"{__name__}.CommitWaitTracker")
        self.running = True
        self.thread = threading.Thread(target=self._background_loop, daemon=True)
        self.thread.start()

    def schedule(self, event_id: int, initial_uncertainty: float):
        """Schedule future uncertainty measurements for this event"""
        schedule_time = time.time()
        for delay_s in self.delays:
            self.pending_measurements.append({
                'event_id': event_id,
                'schedule_time': schedule_time,
                'delay_s': delay_s,
                'measure_at': schedule_time + delay_s,
                'initial_uncertainty': initial_uncertainty
            })
        self.logger.debug(f"Scheduled commit-wait for event {event_id} at T+{self.delays}s")

    def _background_loop(self):
        """Background thread to measure uncertainty at scheduled times"""
        while self.running:
            now = time.time()

            # Process pending measurements
            ready = []
            remaining = deque()

            while self.pending_measurements:
                measurement = self.pending_measurements.popleft()
                if now >= measurement['measure_at']:
                    ready.append(measurement)
                else:
                    remaining.append(measurement)

            self.pending_measurements = remaining

            # Take measurements
            for measurement in ready:
                try:
                    _, uncertainty_ms = self.chronotick_client.query()
                    event_id = measurement['event_id']
                    delay_key = f"ct_uncertainty_{measurement['delay_s']}s_ms"

                    if event_id not in self.results:
                        self.results[event_id] = {}
                    self.results[event_id][delay_key] = uncertainty_ms

                    self.logger.info(
                        f"Commit-wait: Event {event_id} @ T+{measurement['delay_s']}s: "
                        f"uncertainty {measurement['initial_uncertainty']:.2f}ms → {uncertainty_ms:.2f}ms"
                    )
                except Exception as e:
                    self.logger.error(f"Commit-wait measurement failed: {e}")

            time.sleep(1)  # Check every second

    def get_results(self, event_id: int) -> Dict[str, float]:
        """Get commit-wait results for an event"""
        return self.results.get(event_id, {})

    def stop(self):
        """Stop background thread"""
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=5)


class Worker:
    """Main worker class"""

    def __init__(self,
                 node_id: str,
                 listen_port: int,
                 ntp_server: str,
                 chronotick_server: str,
                 output_file: Path,
                 commit_wait_delays: list[int] = [30, 60]):

        self.node_id = node_id
        self.output_file = output_file
        self.logger = logging.getLogger(f"{__name__}.Worker.{node_id}")

        # Network
        self.logger.info(f"Initializing UDP listener on port {listen_port}...")
        self.listener = UDPListener(listen_port)

        # Timing clients
        self.logger.info(f"Initializing NTP client (servers: {ntp_server})...")
        ntp_servers = ntp_server.split(',')
        self.ntp_client = NTPClient(ntp_servers)

        self.logger.info(f"Initializing ChronoTick client (server: {chronotick_server})...")
        self.chronotick_client = ChronoTickClient(chronotick_server)

        # Commit-wait tracker
        self.logger.info(f"Initializing commit-wait tracker (delays: {commit_wait_delays}s)...")
        self.commit_wait = CommitWaitTracker(self.chronotick_client, commit_wait_delays)

        # CSV output
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.csv_file = open(self.output_file, 'w', newline='')
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=[
            'event_id',
            'node_id',
            'sequence_number',
            'receive_time_ns',
            'coordinator_send_time_ns',
            'ntp_offset_ms',
            'ntp_uncertainty_ms',
            'ntp_timestamp_ns',
            'ct_offset_ms',
            'ct_uncertainty_ms',
            'ct_timestamp_ns',
            'ct_lower_bound_ns',
            'ct_upper_bound_ns',
            'ct_uncertainty_30s_ms',
            'ct_uncertainty_60s_ms',
        ])
        self.csv_writer.writeheader()
        self.csv_file.flush()

        # Statistics
        self.events_received = 0
        self.events_processed = 0
        self.ntp_failures = 0
        self.ct_failures = 0

        # Warmup state
        self.warmup_complete = False
        self.warmup_start_time = time.time()

        # Graceful shutdown
        self.running = True
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False

    def _warmup(self):
        """Warmup ChronoTick and NTP for 3 minutes"""
        self.logger.info("="*60)
        self.logger.info("WARMUP PHASE: Initializing ChronoTick and NTP")
        self.logger.info("Duration: 180 seconds (3 minutes)")
        self.logger.info("="*60)

        warmup_duration = 180  # 3 minutes
        query_interval = 10  # Query every 10 seconds

        for elapsed in range(0, warmup_duration, query_interval):
            if not self.running:
                break

            try:
                # Query NTP
                ntp_offset, ntp_uncertainty = self.ntp_client.query()
                self.logger.info(
                    f"Warmup [{elapsed:3d}s]: NTP offset={ntp_offset:+7.2f}ms, "
                    f"uncertainty={ntp_uncertainty:5.2f}ms"
                )
            except Exception as e:
                self.logger.warning(f"Warmup [{elapsed:3d}s]: NTP query failed: {e}")

            try:
                # Query ChronoTick
                ct_offset, ct_uncertainty = self.chronotick_client.query()
                self.logger.info(
                    f"Warmup [{elapsed:3d}s]: ChronoTick offset={ct_offset:+7.2f}ms, "
                    f"uncertainty={ct_uncertainty:5.2f}ms"
                )
            except Exception as e:
                self.logger.warning(f"Warmup [{elapsed:3d}s]: ChronoTick query failed: {e}")

            time.sleep(query_interval)

        self.warmup_complete = True
        self.logger.info("="*60)
        self.logger.info("WARMUP COMPLETE - Worker ready to receive events")
        self.logger.info("="*60)

    def _process_event(self, event: Event, receive_time_ns: int):
        """Process a received event"""
        try:
            # Query NTP (cached for 10s)
            ntp_offset_ms, ntp_uncertainty_ms = self.ntp_client.query()

            # Query ChronoTick (fresh every time)
            ct_offset_ms, ct_uncertainty_ms = self.chronotick_client.query()

            # Calculate timestamps
            ntp_timestamp_ns = receive_time_ns + int(ntp_offset_ms * 1_000_000)
            ct_timestamp_ns = receive_time_ns + int(ct_offset_ms * 1_000_000)
            ct_lower_ns = ct_timestamp_ns - int(3 * ct_uncertainty_ms * 1_000_000)
            ct_upper_ns = ct_timestamp_ns + int(3 * ct_uncertainty_ms * 1_000_000)

            # Schedule commit-wait measurements
            self.commit_wait.schedule(event.event_id, ct_uncertainty_ms)

            # Create record (commit-wait results will be added later)
            record = {
                'event_id': event.event_id,
                'node_id': self.node_id,
                'sequence_number': event.sequence_number,
                'receive_time_ns': receive_time_ns,
                'coordinator_send_time_ns': event.coordinator_timestamp_ns,
                'ntp_offset_ms': ntp_offset_ms,
                'ntp_uncertainty_ms': ntp_uncertainty_ms,
                'ntp_timestamp_ns': ntp_timestamp_ns,
                'ct_offset_ms': ct_offset_ms,
                'ct_uncertainty_ms': ct_uncertainty_ms,
                'ct_timestamp_ns': ct_timestamp_ns,
                'ct_lower_bound_ns': ct_lower_ns,
                'ct_upper_bound_ns': ct_upper_ns,
                'ct_uncertainty_30s_ms': None,  # Will be filled by commit-wait
                'ct_uncertainty_60s_ms': None,
            }

            # Write to CSV
            self.csv_writer.writerow(record)
            self.csv_file.flush()

            self.events_processed += 1

            # Log progress
            if self.events_processed % 10 == 0:
                self.logger.info(
                    f"Progress: {self.events_processed} events processed | "
                    f"NTP: {ntp_offset_ms:+.2f}±{ntp_uncertainty_ms:.2f}ms | "
                    f"ChronoTick: {ct_offset_ms:+.2f}±{ct_uncertainty_ms:.2f}ms"
                )

        except Exception as e:
            self.logger.error(f"Failed to process event {event.event_id}: {e}", exc_info=True)

    def run(self):
        """Main worker loop"""
        self.logger.info(f"="*60)
        self.logger.info(f"Worker {self.node_id} starting")
        self.logger.info(f"Output: {self.output_file}")
        self.logger.info(f"="*60)

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
        self.logger.info("Worker shutting down")
        self.logger.info(f"Duration: {duration:.1f}s")
        self.logger.info(f"Events received: {self.events_received}")
        self.logger.info(f"Events processed: {self.events_processed}")
        self.logger.info(f"Success rate: {self.events_processed/max(1, self.events_received)*100:.1f}%")
        self.logger.info("="*60)

        # Wait for commit-wait measurements to complete
        self.logger.info("Waiting for commit-wait measurements to complete (90s)...")
        time.sleep(90)

        # Update CSV with commit-wait results
        self._update_commit_wait_results()

        # Cleanup
        self.commit_wait.stop()
        self.csv_file.close()
        self.listener.close()

    def _update_commit_wait_results(self):
        """Update CSV file with commit-wait results"""
        self.logger.info("Updating commit-wait results in CSV...")

        # Read current CSV
        import pandas as pd
        df = pd.read_csv(self.output_file)

        # Update with commit-wait results
        for event_id in df['event_id']:
            results = self.commit_wait.get_results(event_id)
            if results:
                for key, value in results.items():
                    df.loc[df['event_id'] == event_id, key] = value

        # Write back
        df.to_csv(self.output_file, index=False)
        self.logger.info(f"Updated {len(df)} rows with commit-wait data")


def main():
    parser = argparse.ArgumentParser(description='Data Streaming Worker (Task B/C)')
    parser.add_argument('--node-id', type=str, required=True,
                       help='Node identifier (e.g., comp11)')
    parser.add_argument('--listen-port', type=int, required=True,
                       help='UDP port to listen on')
    parser.add_argument('--ntp-server', type=str, required=True,
                       help='NTP server(s) (comma-separated)')
    parser.add_argument('--chronotick-server', type=str, required=True,
                       help='ChronoTick server URL')
    parser.add_argument('--output', type=Path, required=True,
                       help='Output CSV file')
    parser.add_argument('--log-file', type=Path, default=None,
                       help='Log file (default: stdout only)')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    # Setup logging
    setup_logging(log_file=args.log_file, level=args.log_level)

    # Create and run worker
    worker = Worker(
        node_id=args.node_id,
        listen_port=args.listen_port,
        ntp_server=args.ntp_server,
        chronotick_server=args.chronotick_server,
        output_file=args.output
    )

    worker.run()


if __name__ == '__main__':
    main()
