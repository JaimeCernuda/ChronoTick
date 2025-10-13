#!/usr/bin/env python3
"""
Minimal ChronoTick Client Integration Test

Demonstrates production usage of ChronoTick as an installed library.

Installation:
    cd ChronoTick/
    uv sync --extra chronotick

Usage:
    uv run python tests/integration/test_minimal_client.py --duration 900

This is what you show users:
    git clone <repo>
    cd ChronoTick/
    uv sync --extra chronotick
    uv run python tests/integration/test_minimal_client.py
"""

import time
import csv
import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add clients/python to sys.path for chronotick_client import
client_dir = Path(__file__).parent.parent.parent / "clients" / "python"
sys.path.insert(0, str(client_dir))

# Import ChronoTick client
try:
    from chronotick_client import ChronoTickClient
except ImportError as e:
    print(f"ERROR: ChronoTick client not found: {e}")
    print(f"\nSearched in: {client_dir}")
    print("\nMake sure you're running from the ChronoTick/ directory")
    sys.exit(1)


class MinimalValidator:
    """Minimal ChronoTick validation - production example."""

    def __init__(self, duration: int = 900, interval: float = 1.0):
        """
        Initialize validator.

        Args:
            duration: How long to collect data (seconds)
            interval: Collection interval (seconds)
        """
        self.duration = duration
        self.interval = interval

        # Output to results/
        output_dir = Path("results/validation_post_cleanup")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = output_dir / f"minimal_client_{timestamp}.csv"
        self.log_path = output_dir / f"minimal_client_{timestamp}.log"

        self.client = None

    def connect(self) -> bool:
        """Connect to ChronoTick daemon."""
        try:
            print("Connecting to ChronoTick daemon...")
            self.client = ChronoTickClient()

            # Test connection
            test_time = self.client.get_time()
            print(f"✓ Connected successfully!")
            print(f"  System time:     {test_time.system_timestamp:.6f}")
            print(f"  Corrected time:  {test_time.corrected_timestamp:.6f}")
            print(f"  Uncertainty:     ±{test_time.uncertainty_seconds * 1000:.3f}ms")
            print(f"  Confidence:      {test_time.confidence:.3f}")
            print(f"  Source:          {test_time.source}")
            print()
            return True

        except Exception as e:
            print(f"✗ Failed to connect: {e}")
            print("\nMake sure ChronoTick daemon is running:")
            print("  cd tsfm/")
            print("  uv run python chronotick_inference/real_data_pipeline.py backtracking &")
            return False

    def run(self):
        """Run validation."""
        print(f"ChronoTick Minimal Client Test")
        print(f"=" * 70)
        print(f"Duration:  {self.duration}s ({self.duration/60:.1f} minutes)")
        print(f"Interval:  {self.interval}s")
        print(f"CSV:       {self.csv_path}")
        print(f"Log:       {self.log_path}")
        print()

        start_time = time.time()
        sample_count = 0
        error_count = 0

        # Open CSV and log files
        with open(self.csv_path, 'w', newline='') as csvfile, \
             open(self.log_path, 'w') as logfile:

            fieldnames = [
                'timestamp', 'system_time', 'chronotick_time', 'error_ms',
                'offset_ms', 'drift_us_per_s', 'uncertainty_ms',
                'confidence', 'source', 'status'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            next_sample = start_time

            while (time.time() - start_time) < self.duration:
                # Wait for next sample time
                sleep_time = next_sample - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)

                # Collect sample
                sys_time = time.time()

                try:
                    ct = self.client.get_time()

                    row = {
                        'timestamp': sys_time,
                        'system_time': sys_time,
                        'chronotick_time': ct.corrected_timestamp,
                        'error_ms': (ct.corrected_timestamp - sys_time) * 1000,
                        'offset_ms': ct.offset_correction * 1000,
                        'drift_us_per_s': ct.drift_rate * 1e6,
                        'uncertainty_ms': ct.uncertainty_seconds * 1000,
                        'confidence': ct.confidence,
                        'source': ct.source,
                        'status': 'ok'
                    }

                except Exception as e:
                    row = {
                        'timestamp': sys_time,
                        'system_time': sys_time,
                        'chronotick_time': None,
                        'error_ms': None,
                        'offset_ms': None,
                        'drift_us_per_s': None,
                        'uncertainty_ms': None,
                        'confidence': None,
                        'source': None,
                        'status': f'error: {e}'
                    }
                    error_count += 1

                writer.writerow(row)
                csvfile.flush()

                sample_count += 1

                # Log every 10 samples
                if sample_count % 10 == 0:
                    elapsed = time.time() - start_time
                    remaining = self.duration - elapsed
                    msg = (f"[{elapsed:6.1f}s] Sample {sample_count:4d} | "
                           f"Err: {row.get('error_ms', 0):7.3f}ms | "
                           f"Unc: ±{row.get('uncertainty_ms', 0):6.3f}ms | "
                           f"Src: {row.get('source', 'N/A'):6s} | "
                           f"Remaining: {remaining:6.1f}s")
                    print(msg)
                    logfile.write(msg + '\n')
                    logfile.flush()

                next_sample += self.interval

        # Summary
        actual_duration = time.time() - start_time

        summary = [
            "",
            "=" * 70,
            "VALIDATION COMPLETE",
            "=" * 70,
            f"Duration:     {actual_duration:.1f}s ({actual_duration/60:.1f} min)",
            f"Samples:      {sample_count}",
            f"Errors:       {error_count}",
            f"Success rate: {(sample_count - error_count) / sample_count * 100:.1f}%",
            f"CSV output:   {self.csv_path}",
            f"Log output:   {self.log_path}",
            ""
        ]

        for line in summary:
            print(line)

        return sample_count, error_count


def main():
    parser = argparse.ArgumentParser(
        description="Minimal ChronoTick client validation"
    )
    parser.add_argument(
        '--duration', '-d',
        type=int,
        default=900,
        help='Duration in seconds (default: 900 = 15 min)'
    )
    parser.add_argument(
        '--interval', '-i',
        type=float,
        default=1.0,
        help='Sampling interval in seconds (default: 1.0)'
    )

    args = parser.parse_args()

    validator = MinimalValidator(
        duration=args.duration,
        interval=args.interval
    )

    if validator.connect():
        try:
            validator.run()
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            print(f"Partial results: {validator.csv_path}")
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
