#!/usr/bin/env python3
"""
ChronoTick Validation Client

Compares ChronoTick time corrections against:
- Direct NTP queries
- System clock (time.time())

Measures accuracy and generates validation report.
"""

import time
import ntplib
import statistics
from dataclasses import dataclass
from typing import List, Dict, Optional
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from chronotick.inference.real_data_pipeline import RealDataPipeline


@dataclass
class TimeComparison:
    """Single time comparison measurement."""
    timestamp: float
    system_time: float
    ntp_offset: float
    ntp_delay: float
    chronotick_offset: float
    chronotick_uncertainty: float
    chronotick_vs_ntp_diff: float  # ChronoTick prediction error vs NTP


class ChronoTickValidator:
    """Validates ChronoTick accuracy against NTP and system clock."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config_path = config_path
        self.ntp_client = ntplib.NTPClient()
        self.ntp_server = "pool.ntp.org"
        self.measurements: List[TimeComparison] = []

    def query_ntp(self) -> Optional[tuple]:
        """
        Query NTP server directly.

        Returns:
            Tuple of (offset, delay) or None if failed
        """
        try:
            response = self.ntp_client.request(self.ntp_server, timeout=2)
            return (response.offset, response.delay)
        except Exception as e:
            print(f"NTP query failed: {e}")
            return None

    def get_chronotick_correction(self, pipeline: RealDataPipeline, timestamp: float):
        """
        Get ChronoTick correction.

        Args:
            pipeline: RealDataPipeline instance
            timestamp: Current timestamp

        Returns:
            Tuple of (offset, uncertainty) or (None, None) if failed
        """
        try:
            correction = pipeline.get_correction(timestamp)
            return (correction.offset_correction, correction.uncertainty)
        except Exception as e:
            print(f"ChronoTick query failed: {e}")
            return (None, None)

    def collect_measurement(self, pipeline: RealDataPipeline) -> Optional[TimeComparison]:
        """
        Collect a single comparative measurement.

        Args:
            pipeline: RealDataPipeline instance

        Returns:
            TimeComparison or None if measurement failed
        """
        # Get current time
        system_time = time.time()

        # Query NTP
        ntp_result = self.query_ntp()
        if ntp_result is None:
            return None
        ntp_offset, ntp_delay = ntp_result

        # Query ChronoTick
        ct_offset, ct_uncertainty = self.get_chronotick_correction(pipeline, system_time)
        if ct_offset is None:
            return None

        # Calculate difference between ChronoTick and NTP
        diff = abs(ct_offset - ntp_offset)

        measurement = TimeComparison(
            timestamp=system_time,
            system_time=system_time,
            ntp_offset=ntp_offset,
            ntp_delay=ntp_delay,
            chronotick_offset=ct_offset,
            chronotick_uncertainty=ct_uncertainty,
            chronotick_vs_ntp_diff=diff
        )

        return measurement

    def run_validation(
        self,
        duration_seconds: int = 300,
        interval_seconds: float = 5.0,
        warmup_seconds: int = 60
    ):
        """
        Run validation test.

        Args:
            duration_seconds: How long to run validation (default 5 minutes)
            interval_seconds: Time between measurements (default 5 seconds)
            warmup_seconds: Warmup period before starting measurements (default 60 seconds)
        """
        print("=" * 70)
        print("ChronoTick Validation Test")
        print("=" * 70)
        print(f"\nConfiguration:")
        print(f"  Duration: {duration_seconds}s ({duration_seconds / 60:.1f} minutes)")
        print(f"  Interval: {interval_seconds}s")
        print(f"  Warmup: {warmup_seconds}s")
        print(f"  NTP Server: {self.ntp_server}")
        print(f"\nInitializing ChronoTick...")

        # Initialize RealDataPipeline
        # NOTE: For this to work with real ML models, daemon.py integration must be used
        # This is a simplified version for testing
        try:
            pipeline = RealDataPipeline(self.config_path)
            print("✓ ChronoTick initialized")
        except Exception as e:
            print(f"✗ Failed to initialize ChronoTick: {e}")
            return

        # Warmup period
        if warmup_seconds > 0:
            print(f"\nWarmup period ({warmup_seconds}s)...")
            print("  Collecting initial NTP measurements for baseline...")
            time.sleep(warmup_seconds)
            print("✓ Warmup complete")

        # Run measurements
        print(f"\nStarting validation ({duration_seconds}s)...\n")
        print(f"{'Time':>10} | {'NTP Offset':>12} | {'CT Offset':>12} | {'CT vs NTP':>12} | {'Uncertainty':>12}")
        print("-" * 70)

        start_time = time.time()
        measurement_count = 0

        while time.time() - start_time < duration_seconds:
            measurement = self.collect_measurement(pipeline)

            if measurement:
                self.measurements.append(measurement)
                measurement_count += 1

                # Print progress
                print(
                    f"{measurement_count:>10} | "
                    f"{measurement.ntp_offset:>12.9f} | "
                    f"{measurement.chronotick_offset:>12.9f} | "
                    f"{measurement.chronotick_vs_ntp_diff:>12.9f} | "
                    f"{measurement.chronotick_uncertainty:>12.9f}"
                )

            # Wait for next interval
            time.sleep(interval_seconds)

        print("\n" + "=" * 70)
        print("Validation Complete")
        print("=" * 70)

        # Generate report
        self.print_report()

    def print_report(self):
        """Print validation report with statistics."""
        if not self.measurements:
            print("\nNo measurements collected!")
            return

        # Calculate statistics
        errors = [m.chronotick_vs_ntp_diff for m in self.measurements]
        ntp_offsets = [m.ntp_offset for m in self.measurements]
        ct_offsets = [m.chronotick_offset for m in self.measurements]
        uncertainties = [m.chronotick_uncertainty for m in self.measurements]

        print(f"\n📊 Validation Results ({len(self.measurements)} measurements)")
        print("=" * 70)

        print("\n1. ChronoTick Prediction Error (vs NTP)")
        print("-" * 70)
        print(f"  Mean Error:      {statistics.mean(errors):>12.9f}s ({statistics.mean(errors)*1000:>8.3f}ms)")
        print(f"  Median Error:    {statistics.median(errors):>12.9f}s ({statistics.median(errors)*1000:>8.3f}ms)")
        print(f"  Std Dev:         {statistics.stdev(errors) if len(errors) > 1 else 0:>12.9f}s")
        print(f"  Min Error:       {min(errors):>12.9f}s ({min(errors)*1000:>8.3f}ms)")
        print(f"  Max Error:       {max(errors):>12.9f}s ({max(errors)*1000:>8.3f}ms)")

        print("\n2. NTP Offset Statistics")
        print("-" * 70)
        print(f"  Mean Offset:     {statistics.mean(ntp_offsets):>12.9f}s")
        print(f"  Std Dev:         {statistics.stdev(ntp_offsets) if len(ntp_offsets) > 1 else 0:>12.9f}s")
        print(f"  Range:           {min(ntp_offsets):>12.9f}s to {max(ntp_offsets):.9f}s")

        print("\n3. ChronoTick Offset Statistics")
        print("-" * 70)
        print(f"  Mean Offset:     {statistics.mean(ct_offsets):>12.9f}s")
        print(f"  Std Dev:         {statistics.stdev(ct_offsets) if len(ct_offsets) > 1 else 0:>12.9f}s")
        print(f"  Range:           {min(ct_offsets):>12.9f}s to {max(ct_offsets):.9f}s")

        print("\n4. ChronoTick Uncertainty Statistics")
        print("-" * 70)
        print(f"  Mean Uncertainty: {statistics.mean(uncertainties):>12.9f}s ({statistics.mean(uncertainties)*1000:>8.3f}ms)")
        print(f"  Median Uncertainty: {statistics.median(uncertainties):>12.9f}s")
        print(f"  Range:           {min(uncertainties):>12.9f}s to {max(uncertainties):.9f}s")

        print("\n5. Accuracy Assessment")
        print("-" * 70)
        # Check how many measurements fall within uncertainty bounds
        within_bounds = sum(1 for m in self.measurements if m.chronotick_vs_ntp_diff <= m.chronotick_uncertainty)
        within_bounds_pct = (within_bounds / len(self.measurements)) * 100

        print(f"  Measurements within uncertainty bounds: {within_bounds}/{len(self.measurements)} ({within_bounds_pct:.1f}%)")

        # Error thresholds
        errors_under_1ms = sum(1 for e in errors if e < 0.001)
        errors_under_10ms = sum(1 for e in errors if e < 0.010)
        errors_under_100ms = sum(1 for e in errors if e < 0.100)

        print(f"\n  Error Distribution:")
        print(f"    < 1ms:   {errors_under_1ms}/{len(errors)} ({errors_under_1ms/len(errors)*100:.1f}%)")
        print(f"    < 10ms:  {errors_under_10ms}/{len(errors)} ({errors_under_10ms/len(errors)*100:.1f}%)")
        print(f"    < 100ms: {errors_under_100ms}/{len(errors)} ({errors_under_100ms/len(errors)*100:.1f}%)")

        print("\n6. Overall Assessment")
        print("-" * 70)
        mean_error_ms = statistics.mean(errors) * 1000
        if mean_error_ms < 1:
            grade = "EXCELLENT"
        elif mean_error_ms < 10:
            grade = "GOOD"
        elif mean_error_ms < 100:
            grade = "ACCEPTABLE"
        else:
            grade = "NEEDS IMPROVEMENT"

        print(f"  Grade: {grade}")
        print(f"  Mean Prediction Error: {mean_error_ms:.3f}ms")
        print(f"  Accuracy vs NTP: {(1 - statistics.mean(errors)/statistics.mean([abs(o) for o in ntp_offsets]))*100:.1f}%")

        print("\n" + "=" * 70)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate ChronoTick accuracy against NTP and system clock"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=300,
        help="Validation duration in seconds (default: 300)"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Measurement interval in seconds (default: 5.0)"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=60,
        help="Warmup period in seconds (default: 60)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="ChronoTick config path (default: configs/config.yaml)"
    )
    parser.add_argument(
        "--ntp-server",
        type=str,
        default="pool.ntp.org",
        help="NTP server to use for validation (default: pool.ntp.org)"
    )

    args = parser.parse_args()

    # Create validator
    validator = ChronoTickValidator(config_path=args.config)
    validator.ntp_server = args.ntp_server

    try:
        # Run validation
        validator.run_validation(
            duration_seconds=args.duration,
            interval_seconds=args.interval,
            warmup_seconds=args.warmup
        )
    except KeyboardInterrupt:
        print("\n\n⚠ Validation interrupted by user")
        if validator.measurements:
            validator.print_report()
    except Exception as e:
        print(f"\n✗ Validation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
