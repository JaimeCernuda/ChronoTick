#!/usr/bin/env python3
"""
Coordinator (Task A): Broadcasts events to worker nodes

Runs on ares-comp-18
Sends timestamped events to ares-comp-11 and ares-comp-12
Records send times for ground truth comparison
"""

import argparse
import time
import csv
from pathlib import Path
from typing import List, Tuple
import yaml

from src.common import (
    Event,
    UDPBroadcaster,
    setup_logging,
    high_precision_sleep
)


class BroadcastPattern:
    """Defines timing pattern for event broadcasts"""

    PATTERNS = {
        'fast_stream': 0.010,    # 10ms - rapid events
        'medium_stream': 0.050,  # 50ms - potential duplicates
        'slow_stream': 0.100,    # 100ms - window boundaries
        'very_slow': 0.500,      # 500ms - large gaps
    }

    def __init__(self, pattern_config: List[str]):
        """
        pattern_config: List like ['slow', 'fast', 'fast', 'medium', ...]
        """
        self.pattern = [self.PATTERNS[p + '_stream'] for p in pattern_config]
        self.index = 0

    def get_next_delay(self) -> float:
        """Get next delay in seconds"""
        delay = self.pattern[self.index % len(self.pattern)]
        self.index += 1
        return delay


class Coordinator:
    """Main coordinator class"""

    def __init__(self,
                 workers: List[Tuple[str, int]],
                 num_events: int,
                 pattern: BroadcastPattern,
                 output_file: Path):
        self.workers = workers
        self.num_events = num_events
        self.pattern = pattern
        self.output_file = output_file

        self.broadcaster = UDPBroadcaster(timeout=5.0)
        self.logger = setup_logging(level="INFO")

        # Prepare output CSV
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.csv_file = open(self.output_file, 'w', newline='')
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=[
            'event_id',
            'sequence_number',
            'send_time_ns',
            'send_time_iso',
            'pattern_delay_ms',
            'workers_sent',
            'workers_failed'
        ])
        self.csv_writer.writeheader()

    def run(self):
        """Main broadcast loop"""
        self.logger.info(f"Starting coordinator: {self.num_events} events to {len(self.workers)} workers")
        self.logger.info(f"Workers: {self.workers}")

        start_time = time.time()

        for seq in range(self.num_events):
            # Create event
            event = Event(
                event_id=seq + 1,
                coordinator_timestamp_ns=time.time_ns(),
                sequence_number=seq,
                payload=f"event_{seq+1}"
            )

            # Broadcast
            send_time_ns = time.time_ns()
            results = self.broadcaster.send(event, self.workers)

            # Record
            workers_sent = [addr for addr, success in results.items() if success]
            workers_failed = [addr for addr, success in results.items() if not success]

            self.csv_writer.writerow({
                'event_id': event.event_id,
                'sequence_number': seq,
                'send_time_ns': send_time_ns,
                'send_time_iso': time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(send_time_ns / 1e9)),
                'pattern_delay_ms': self.pattern.pattern[seq % len(self.pattern.pattern)] * 1000,
                'workers_sent': ','.join(workers_sent),
                'workers_failed': ','.join(workers_failed) if workers_failed else ''
            })

            # Log progress
            if (seq + 1) % 10 == 0:
                elapsed = time.time() - start_time
                rate = (seq + 1) / elapsed
                remaining = (self.num_events - seq - 1) / rate if rate > 0 else 0
                self.logger.info(
                    f"Progress: {seq+1}/{self.num_events} events "
                    f"({rate:.1f} events/s, ~{remaining:.0f}s remaining)"
                )

            # Delay before next event
            if seq < self.num_events - 1:
                delay = self.pattern.get_next_delay()
                high_precision_sleep(delay)

        # Cleanup
        duration = time.time() - start_time
        self.logger.info(f"Broadcast complete! {self.num_events} events in {duration:.1f}s")
        self.logger.info(f"Average rate: {self.num_events/duration:.2f} events/s")
        self.logger.info(f"Results saved to: {self.output_file}")

        self.csv_file.close()
        self.broadcaster.close()


def parse_workers(workers_str: str) -> List[Tuple[str, int]]:
    """Parse workers string like 'host1:port1,host2:port2'"""
    workers = []
    for worker in workers_str.split(','):
        host, port = worker.strip().split(':')
        workers.append((host, int(port)))
    return workers


def main():
    parser = argparse.ArgumentParser(description='Data Streaming Coordinator (Task A)')
    parser.add_argument('--config', type=Path, help='YAML config file')
    parser.add_argument('--workers', type=str, required=True,
                       help='Worker addresses (host1:port1,host2:port2)')
    parser.add_argument('--num-events', type=int, default=100,
                       help='Number of events to broadcast')
    parser.add_argument('--output', type=Path, required=True,
                       help='Output CSV file')
    parser.add_argument('--pattern', type=str, default='slow,fast,fast,fast,fast,medium',
                       help='Broadcast pattern (comma-separated: slow,fast,medium)')

    args = parser.parse_args()

    # Load config if provided
    config = {}
    if args.config and args.config.exists():
        with open(args.config) as f:
            config = yaml.safe_load(f)

    # Parse workers
    workers = parse_workers(args.workers)

    # Create pattern
    pattern_list = args.pattern.split(',')
    pattern = BroadcastPattern(pattern_list)

    # Create and run coordinator
    coordinator = Coordinator(
        workers=workers,
        num_events=args.num_events,
        pattern=pattern,
        output_file=args.output
    )

    coordinator.run()


if __name__ == '__main__':
    main()
