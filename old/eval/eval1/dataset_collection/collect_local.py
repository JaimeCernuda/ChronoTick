#!/usr/bin/env python3
"""
ChronoTick Evaluation 1 - Local Dataset Collector

Collects timing measurements on local machine (AMD GPU system).
Focuses on thermal drift patterns and high-precision ground truth.
"""

import os
import sys
import time
import signal
import argparse
import threading
from pathlib import Path
from typing import Optional

# Add to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tsfm"))

from utils import (
    TimingMeasurement, NTPClient, SystemMetricsCollector,
    DatasetWriter, load_config, setup_logging, get_system_info
)

# Try to import ChronoTick for ground truth
try:
    import chronotick
    CHRONOTICK_AVAILABLE = True
except ImportError:
    CHRONOTICK_AVAILABLE = False
    print("Warning: ChronoTick not available - using NTP ensemble for ground truth")


class LocalDataCollector:
    """Data collector optimized for local AMD GPU machine"""

    def __init__(self, config_path: Path, output_dir: Path):
        self.config = load_config(config_path)
        self.output_dir = Path(output_dir)
        self.running = False
        self.shutdown_event = threading.Event()

        # Initialize components
        self.ntp_client = NTPClient(
            servers=self.config['ntp']['servers'],
            timeout=self.config['ntp']['timeout_seconds']
        )
        self.metrics_collector = SystemMetricsCollector()
        self.writer = DatasetWriter(
            output_dir=self.output_dir,
            node_id=self.config['environments']['local']['node_id'],
            max_file_size_mb=self.config['collection']['max_file_size_mb']
        )

        # Ground truth system
        self.chronotick_started = False
        if CHRONOTICK_AVAILABLE and self._should_use_chronotick():
            self._start_chronotick()

        # Statistics
        self.measurement_count = 0
        self.error_count = 0
        self.last_backup = time.time()

    def _should_use_chronotick(self) -> bool:
        """Check if we should use ChronoTick for ground truth"""
        return self.config.get('ground_truth', {}).get('method') == 'chronotick'

    def _start_chronotick(self):
        """Start ChronoTick for ground truth measurements"""
        try:
            print("Starting ChronoTick for ground truth...")
            success = chronotick.start()
            if success:
                # Wait for warmup
                print("Waiting for ChronoTick warmup...")
                time.sleep(30)  # Allow warmup
                self.chronotick_started = True
                print("✅ ChronoTick started and warmed up")
            else:
                print("❌ Failed to start ChronoTick")
        except Exception as e:
            print(f"❌ ChronoTick startup error: {e}")

    def _get_ground_truth_offset(self) -> Optional[float]:
        """Get ground truth clock offset"""
        if self.chronotick_started:
            try:
                # Use ChronoTick as ground truth
                chronotick_time = chronotick.time()
                system_time = time.time()
                return chronotick_time - system_time
            except Exception as e:
                print(f"ChronoTick ground truth error: {e}")

        # Fallback: NTP ensemble
        return self._get_ntp_ensemble_offset()

    def _get_ntp_ensemble_offset(self) -> Optional[float]:
        """Get consensus offset from multiple NTP servers"""
        measurements = []
        for server in self.config['ground_truth']['reference_servers']:
            result = self.ntp_client.query_server(server)
            if result:
                offset, delay, stratum = result
                if delay < self.config['ground_truth']['max_reference_uncertainty']:
                    measurements.append(offset)

        if len(measurements) >= self.config['ground_truth']['consensus_threshold']:
            # Return median as consensus
            measurements.sort()
            n = len(measurements)
            if n % 2 == 0:
                return (measurements[n//2 - 1] + measurements[n//2]) / 2
            else:
                return measurements[n//2]

        return None

    def collect_single_measurement(self) -> Optional[TimingMeasurement]:
        """Collect a single timing measurement"""
        measurement_time = time.time()

        # Get NTP measurement for offset
        ntp_result = self.ntp_client.get_best_measurement()
        if not ntp_result:
            self.error_count += 1
            return None

        offset, delay, server, stratum = ntp_result

        # Get system metrics
        metrics = self.metrics_collector.collect_all_metrics()

        # Calculate drift rate (simple finite difference)
        drift_rate = 0.0  # Will be calculated from sequence later

        # Get ground truth
        ground_truth_offset = self._get_ground_truth_offset()

        # Quality control flags
        quality_flags = []
        if abs(offset) > self.config['quality_control']['max_offset_seconds']:
            quality_flags.append('large_offset')
        if delay > self.config['quality_control']['max_ntp_delay']:
            quality_flags.append('high_ntp_delay')
        if stratum < self.config['quality_control']['min_ntp_stratum']:
            quality_flags.append('low_ntp_stratum')

        # Calculate uncertainty
        uncertainty = max(delay / 2.0, 1e-6)  # At least 1μs

        measurement = TimingMeasurement(
            timestamp=measurement_time,
            node_id=self.config['environments']['local']['node_id'],
            clock_offset=offset,
            drift_rate=drift_rate,
            ntp_delay=delay,
            ntp_stratum=stratum,
            ntp_server=server,
            cpu_temp=metrics['cpu_temp'],
            gpu_temp=metrics['gpu_temp'],
            cpu_freq=metrics['cpu_freq'],
            cpu_load=metrics['cpu_load'],
            memory_usage=metrics['memory_usage'],
            network_latency=metrics['network_latency'],
            ground_truth_offset=ground_truth_offset,
            measurement_uncertainty=uncertainty,
            source_type='local_collection',
            quality_flags=quality_flags
        )

        return measurement

    def _backup_progress(self):
        """Create backup of current progress"""
        try:
            backup_file = self.output_dir / f"progress_backup_{int(time.time())}.json"
            progress_data = {
                'measurement_count': self.measurement_count,
                'error_count': self.error_count,
                'last_backup': self.last_backup,
                'uptime_hours': (time.time() - self.start_time) / 3600,
                'success_rate': (self.measurement_count / max(1, self.measurement_count + self.error_count)),
                'system_info': get_system_info()._asdict()
            }

            import json
            with open(backup_file, 'w') as f:
                json.dump(progress_data, f, indent=2, default=str)

            self.last_backup = time.time()
            print(f"Progress backup saved: {backup_file}")

        except Exception as e:
            print(f"Backup failed: {e}")

    def run_collection(self, duration_hours: Optional[float] = None):
        """Run data collection for specified duration"""
        self.running = True
        self.start_time = time.time()

        if duration_hours:
            end_time = self.start_time + (duration_hours * 3600)
        else:
            end_time = self.start_time + (self.config['collection']['duration_hours'] * 3600)

        interval = self.config['collection']['sampling_interval_seconds']
        backup_interval = self.config['collection']['backup_interval_hours'] * 3600

        print(f"Starting data collection for {duration_hours or self.config['collection']['duration_hours']} hours")
        print(f"Output directory: {self.output_dir}")
        print(f"Sampling interval: {interval}s")
        print("Press Ctrl+C to stop gracefully")

        try:
            while self.running and time.time() < end_time:
                measurement_start = time.time()

                # Collect measurement
                measurement = self.collect_single_measurement()
                if measurement:
                    self.writer.write_measurement(measurement)
                    self.measurement_count += 1

                    # Progress reporting
                    if self.measurement_count % 300 == 0:  # Every 5 minutes
                        hours_running = (time.time() - self.start_time) / 3600
                        success_rate = self.measurement_count / max(1, self.measurement_count + self.error_count)
                        print(f"Progress: {self.measurement_count} measurements, "
                              f"{hours_running:.1f}h running, "
                              f"{success_rate:.1%} success rate")

                # Backup progress periodically
                if time.time() - self.last_backup > backup_interval:
                    self._backup_progress()

                # Sleep until next measurement
                elapsed = time.time() - measurement_start
                sleep_time = max(0, interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

                # Check for shutdown
                if self.shutdown_event.is_set():
                    break

        except KeyboardInterrupt:
            print("\nReceived interrupt signal - stopping gracefully...")
        except Exception as e:
            print(f"Collection error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop()

    def stop(self):
        """Stop data collection and cleanup"""
        self.running = False
        print("Stopping data collection...")

        # Final backup
        self._backup_progress()

        # Close writer
        self.writer.close()

        # Stop ChronoTick if we started it
        if self.chronotick_started:
            try:
                chronotick.stop()
                print("ChronoTick stopped")
            except:
                pass

        hours_running = (time.time() - self.start_time) / 3600
        success_rate = self.measurement_count / max(1, self.measurement_count + self.error_count)

        print(f"\nCollection complete:")
        print(f"  Measurements: {self.measurement_count}")
        print(f"  Errors: {self.error_count}")
        print(f"  Success rate: {success_rate:.1%}")
        print(f"  Duration: {hours_running:.1f} hours")
        print(f"  Output: {self.output_dir}")


def signal_handler(signum, frame, collector):
    """Handle shutdown signals"""
    print(f"\nReceived signal {signum} - initiating graceful shutdown...")
    collector.shutdown_event.set()


def main():
    parser = argparse.ArgumentParser(description="ChronoTick Evaluation 1 - Local Data Collector")
    parser.add_argument('--config', type=Path,
                        default=Path(__file__).parent / 'collect_config.yaml',
                        help='Configuration file path')
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).parent.parent / 'datasets' / 'raw' / 'local',
                        help='Output directory')
    parser.add_argument('--duration', type=float, default=None,
                        help='Collection duration in hours (overrides config)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO', help='Logging level')
    parser.add_argument('--log-file', type=Path, default=None,
                        help='Log file path')

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level, args.log_file)

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Create collector
    collector = LocalDataCollector(args.config, args.output)

    # Setup signal handlers
    signal.signal(signal.SIGINT, lambda s, f: signal_handler(s, f, collector))
    signal.signal(signal.SIGTERM, lambda s, f: signal_handler(s, f, collector))

    # Run collection
    collector.run_collection(args.duration)


if __name__ == "__main__":
    main()