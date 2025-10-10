#!/usr/bin/env python3
"""
ChronoTick Shared Memory Client

Simple client for reading ChronoTick time corrections from shared memory.
Useful for:
- Testing and validation
- Performance benchmarking
- Monitoring daemon health
- Integration examples

Features:
- Real-time time display with continuous updates
- Performance benchmarking (read latency, throughput)
- Health monitoring
- JSON export for integration
- No MCP overhead - direct shared memory access

Performance:
- Read latency: ~300ns after first attachment
- Throughput: 1-3 million reads/second
"""

import sys
import time
import json
import argparse
import logging
from pathlib import Path
from typing import Optional
from multiprocessing.shared_memory import SharedMemory

from chronotick_shm.shm_config import (
    SHARED_MEMORY_NAME,
    ChronoTickData,
    read_data_with_retry,
    benchmark_read_latency
)

logger = logging.getLogger(__name__)


class ChronoTickClient:
    """
    ChronoTick shared memory client for reading time corrections.

    This is a lightweight client that directly reads from shared memory
    without any MCP or agent SDK overhead. Perfect for:
    - Performance testing
    - System integration
    - Monitoring and alerting
    - Educational examples
    """

    def __init__(self):
        """Initialize client and attach to shared memory"""
        self.shm: Optional[SharedMemory] = None
        self._connect()

    def _connect(self):
        """Connect to ChronoTick shared memory"""
        try:
            self.shm = SharedMemory(name=SHARED_MEMORY_NAME, create=False)
            logger.info(f"Connected to shared memory: {SHARED_MEMORY_NAME}")
        except FileNotFoundError:
            raise RuntimeError(
                f"ChronoTick daemon not running.\n"
                f"Shared memory '{SHARED_MEMORY_NAME}' not found.\n\n"
                f"Start the daemon with:\n"
                f"  python chronotick_daemon.py"
            )

    def read_data(self) -> ChronoTickData:
        """
        Read current ChronoTickData from shared memory.

        Returns:
            ChronoTickData with all time correction information
        """
        if not self.shm:
            raise RuntimeError("Not connected to shared memory")

        return read_data_with_retry(self.shm.buf, max_retries=3)

    def get_corrected_time(self) -> float:
        """
        Get current corrected timestamp.

        Returns:
            Corrected Unix timestamp (float)
        """
        data = self.read_data()
        current_time = time.time()
        return data.get_corrected_time_at(current_time)

    def get_time_with_uncertainty(self) -> tuple[float, float]:
        """
        Get corrected time with uncertainty estimate.

        Returns:
            Tuple of (corrected_time, uncertainty_seconds)
        """
        data = self.read_data()
        current_time = time.time()
        corrected_time = data.get_corrected_time_at(current_time)
        time_delta = current_time - data.prediction_time
        uncertainty = data.get_time_uncertainty(time_delta)
        return corrected_time, uncertainty

    def get_status(self) -> dict:
        """
        Get daemon status as dictionary.

        Returns:
            Dict with status information
        """
        data = self.read_data()
        current_time = time.time()

        return {
            "daemon_running": True,
            "daemon_uptime": data.daemon_uptime,
            "warmup_complete": data.is_warmup_complete,
            "ntp_ready": data.is_ntp_ready,
            "models_ready": data.is_models_ready,
            "data_valid": data.is_valid,
            "measurement_count": data.measurement_count,
            "total_corrections": data.total_corrections,
            "last_ntp_sync": data.last_ntp_sync,
            "seconds_since_ntp": current_time - data.last_ntp_sync if data.last_ntp_sync > 0 else None,
            "source": data.source.name,
            "confidence": data.confidence,
            "offset_correction": data.offset_correction,
            "drift_rate": data.drift_rate
        }

    def benchmark_read_performance(self, iterations: int = 10000) -> dict:
        """
        Benchmark read performance.

        Args:
            iterations: Number of read iterations

        Returns:
            Dict with benchmark results
        """
        if not self.shm:
            raise RuntimeError("Not connected to shared memory")

        print(f"Running benchmark: {iterations:,} iterations...")

        # Benchmark raw read latency
        latency_ns = benchmark_read_latency(self.shm.buf, iterations=iterations)

        # Benchmark throughput
        start = time.perf_counter()
        for _ in range(iterations):
            _ = self.read_data()
        elapsed = time.perf_counter() - start

        reads_per_second = iterations / elapsed

        return {
            "iterations": iterations,
            "total_time_seconds": elapsed,
            "average_latency_ns": latency_ns,
            "average_latency_us": latency_ns / 1000,
            "average_latency_ms": latency_ns / 1_000_000,
            "reads_per_second": reads_per_second,
            "reads_per_second_millions": reads_per_second / 1_000_000
        }

    def close(self):
        """Close shared memory connection"""
        if self.shm:
            self.shm.close()
            logger.info("Closed shared memory connection")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


# ============================================================================
# CLI Commands
# ============================================================================

def cmd_read_once(args):
    """Read and display time once"""
    with ChronoTickClient() as client:
        data = client.read_data()
        current_time = time.time()
        corrected_time = data.get_corrected_time_at(current_time)
        time_delta = current_time - data.prediction_time
        uncertainty = data.get_time_uncertainty(time_delta)

        print("ChronoTick Time Reading")
        print("=" * 60)
        print(f"Corrected Time:  {corrected_time:.6f}")
        print(f"System Time:     {current_time:.6f}")
        print(f"Offset:          {data.offset_correction*1e6:+.3f}μs")
        print(f"Drift Rate:      {data.drift_rate*1e6:+.3f}μs/s")
        print(f"Uncertainty:     ±{uncertainty*1e6:.3f}μs")
        print(f"Confidence:      {data.confidence:.1%}")
        print(f"Source:          {data.source.name}")
        print(f"Valid:           {'✓' if data.is_valid else '✗'}")
        print(f"NTP Ready:       {'✓' if data.is_ntp_ready else '✗'}")
        print(f"Models Ready:    {'✓' if data.is_models_ready else '✗'}")


def cmd_monitor(args):
    """Continuously monitor time"""
    print("ChronoTick Continuous Monitor")
    print("=" * 60)
    print("Press Ctrl+C to stop\n")

    with ChronoTickClient() as client:
        try:
            while True:
                data = client.read_data()
                current_time = time.time()
                corrected_time = data.get_corrected_time_at(current_time)
                time_delta = current_time - data.prediction_time
                uncertainty = data.get_time_uncertainty(time_delta)

                # Clear line and print status
                print(f"\r"
                      f"Time: {corrected_time:.6f}  "
                      f"Offset: {data.offset_correction*1e6:+6.1f}μs  "
                      f"Uncertainty: ±{uncertainty*1e6:5.1f}μs  "
                      f"Confidence: {data.confidence:4.0%}  "
                      f"Source: {data.source.name:6s}",
                      end="", flush=True)

                time.sleep(args.interval)

        except KeyboardInterrupt:
            print("\n\nMonitoring stopped")


def cmd_status(args):
    """Display daemon status"""
    with ChronoTickClient() as client:
        status = client.get_status()

        print("ChronoTick Daemon Status")
        print("=" * 60)
        print(f"Status:          {'READY' if status['warmup_complete'] else 'WARMING UP'}")
        print(f"Uptime:          {status['daemon_uptime']:.1f}s ({status['daemon_uptime']/60:.1f} min)")
        print(f"Warmup Complete: {'✓' if status['warmup_complete'] else '✗'}")
        print(f"NTP Ready:       {'✓' if status['ntp_ready'] else '✗'}")
        print(f"Models Ready:    {'✓' if status['models_ready'] else '✗'}")
        print(f"Data Valid:      {'✓' if status['data_valid'] else '✗'}")
        print()
        print(f"Measurements:    {status['measurement_count']:,}")
        print(f"Corrections:     {status['total_corrections']:,}")
        print(f"Last NTP Sync:   {status['seconds_since_ntp']:.1f}s ago")
        print()
        print(f"Current Source:  {status['source']}")
        print(f"Confidence:      {status['confidence']:.1%}")
        print(f"Offset:          {status['offset_correction']*1e6:+.3f}μs")
        print(f"Drift Rate:      {status['drift_rate']*1e6:+.3f}μs/s")


def cmd_benchmark(args):
    """Run performance benchmark"""
    with ChronoTickClient() as client:
        print("ChronoTick Performance Benchmark")
        print("=" * 60)

        results = client.benchmark_read_performance(iterations=args.iterations)

        print(f"\nResults ({results['iterations']:,} iterations):")
        print(f"  Total Time:        {results['total_time_seconds']:.3f}s")
        print(f"  Average Latency:   {results['average_latency_ns']:.0f}ns ({results['average_latency_us']:.2f}μs)")
        print(f"  Throughput:        {results['reads_per_second']:,.0f} reads/s")
        print(f"                     {results['reads_per_second_millions']:.2f} million reads/s")
        print()
        print("Performance Category:")
        if results['average_latency_ns'] < 500:
            print("  ⭐⭐⭐ EXCELLENT - Sub-500ns latency!")
        elif results['average_latency_ns'] < 1000:
            print("  ⭐⭐ GOOD - Sub-1μs latency")
        elif results['average_latency_ns'] < 10000:
            print("  ⭐ ACCEPTABLE - Sub-10μs latency")
        else:
            print("  ⚠️  SLOW - >10μs latency (check system load)")


def cmd_json(args):
    """Export data as JSON"""
    with ChronoTickClient() as client:
        data = client.read_data()
        current_time = time.time()
        corrected_time = data.get_corrected_time_at(current_time)
        time_delta = current_time - data.prediction_time
        uncertainty = data.get_time_uncertainty(time_delta)

        output = {
            "corrected_time": corrected_time,
            "system_time": current_time,
            "offset_correction": data.offset_correction,
            "drift_rate": data.drift_rate,
            "offset_uncertainty": data.offset_uncertainty,
            "drift_uncertainty": data.drift_uncertainty,
            "time_uncertainty": uncertainty,
            "confidence": data.confidence,
            "source": data.source.name,
            "prediction_time": data.prediction_time,
            "valid_until": data.valid_until,
            "is_valid": data.is_valid,
            "is_ntp_ready": data.is_ntp_ready,
            "is_models_ready": data.is_models_ready,
            "is_warmup_complete": data.is_warmup_complete,
            "daemon_uptime": data.daemon_uptime,
            "measurement_count": data.measurement_count,
            "total_corrections": data.total_corrections
        }

        print(json.dumps(output, indent=2 if args.pretty else None))


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="ChronoTick Shared Memory Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Read time once
  python chronotick_client.py read

  # Monitor continuously
  python chronotick_client.py monitor --interval 0.1

  # Check daemon status
  python chronotick_client.py status

  # Run performance benchmark
  python chronotick_client.py benchmark --iterations 100000

  # Export as JSON
  python chronotick_client.py json --pretty
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Read command
    subparsers.add_parser("read", help="Read and display time once")

    # Monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Continuously monitor time")
    monitor_parser.add_argument("--interval", type=float, default=0.1,
                               help="Update interval in seconds (default: 0.1)")

    # Status command
    subparsers.add_parser("status", help="Display daemon status")

    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Run performance benchmark")
    benchmark_parser.add_argument("--iterations", type=int, default=10000,
                                 help="Number of iterations (default: 10000)")

    # JSON command
    json_parser = subparsers.add_parser("json", help="Export data as JSON")
    json_parser.add_argument("--pretty", action="store_true",
                            help="Pretty-print JSON output")

    # Parse arguments
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Execute command
    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == "read":
            cmd_read_once(args)
        elif args.command == "monitor":
            cmd_monitor(args)
        elif args.command == "status":
            cmd_status(args)
        elif args.command == "benchmark":
            cmd_benchmark(args)
        elif args.command == "json":
            cmd_json(args)

    except RuntimeError as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
