#!/usr/bin/env python3
"""
ChronoTick Evaluation 1 - ARES HPC Direct Node Collector

Collects timing measurements on ARES HPC cluster with direct node access.
Focuses on HPC-specific effects: InfiniBand, PFS, high-performance computing workloads.
"""

import os
import sys
import time
import signal
import argparse
import threading
import subprocess
from pathlib import Path
from typing import Optional, Dict, List

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    TimingMeasurement, NTPClient, SystemMetricsCollector,
    DatasetWriter, load_config, setup_logging, get_system_info
)


class AresMetricsCollector(SystemMetricsCollector):
    """Extended metrics collector for ARES HPC environment"""

    def __init__(self):
        super().__init__()
        self.infiniband_available = self._check_infiniband()
        self.slurm_available = self._check_slurm()
        self.pfs_mounts = self._find_pfs_mounts()

    def _check_infiniband(self) -> bool:
        """Check if InfiniBand is available"""
        try:
            result = subprocess.run(['ibstat'], capture_output=True, timeout=5)
            return result.returncode == 0
        except:
            return False

    def _check_slurm(self) -> bool:
        """Check if SLURM is available"""
        try:
            result = subprocess.run(['sinfo', '--version'], capture_output=True, timeout=5)
            return result.returncode == 0
        except:
            return False

    def _find_pfs_mounts(self) -> List[str]:
        """Find parallel file system mounts"""
        pfs_mounts = []
        try:
            with open('/proc/mounts') as f:
                for line in f:
                    if any(fs in line for fs in ['lustre', 'gpfs', 'nfs4', 'panfs']):
                        mount_point = line.split()[1]
                        pfs_mounts.append(mount_point)
        except:
            pass
        return pfs_mounts

    def get_infiniband_stats(self) -> Optional[Dict]:
        """Get InfiniBand performance statistics"""
        if not self.infiniband_available:
            return None

        try:
            result = subprocess.run(['ibstatus'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                # Parse basic IB status
                lines = result.stdout.split('\n')
                stats = {}
                for line in lines:
                    if 'Rate:' in line:
                        stats['rate'] = line.split('Rate:')[1].strip()
                    elif 'State:' in line:
                        stats['state'] = line.split('State:')[1].strip()
                return stats
        except:
            pass

        return None

    def get_pfs_performance(self) -> Optional[Dict]:
        """Measure parallel file system performance"""
        if not self.pfs_mounts:
            return None

        stats = {}
        for mount in self.pfs_mounts:
            try:
                # Simple I/O latency test
                test_file = Path(mount) / f"chronotick_iotest_{os.getpid()}"

                # Write test
                start = time.time()
                with open(test_file, 'w') as f:
                    f.write('test')
                    f.flush()
                    os.fsync(f.fileno())
                write_latency = (time.time() - start) * 1000  # ms

                # Read test
                start = time.time()
                with open(test_file, 'r') as f:
                    _ = f.read()
                read_latency = (time.time() - start) * 1000  # ms

                # Cleanup
                test_file.unlink()

                stats[mount] = {
                    'write_latency_ms': write_latency,
                    'read_latency_ms': read_latency
                }

            except Exception as e:
                stats[mount] = {'error': str(e)}

        return stats

    def get_slurm_node_info(self) -> Optional[Dict]:
        """Get SLURM node information"""
        if not self.slurm_available:
            return None

        try:
            hostname = os.uname().nodename
            result = subprocess.run([
                'scontrol', 'show', 'node', hostname
            ], capture_output=True, text=True, timeout=5)

            if result.returncode == 0:
                info = {}
                for line in result.stdout.split('\n'):
                    if 'CPULoad=' in line:
                        load_part = line.split('CPULoad=')[1].split()[0]
                        info['cpu_load'] = float(load_part) if load_part != 'N/A' else None
                    elif 'RealMemory=' in line:
                        mem_part = line.split('RealMemory=')[1].split()[0]
                        info['real_memory'] = int(mem_part) if mem_part.isdigit() else None

                return info

        except Exception as e:
            return {'error': str(e)}

        return None

    def collect_all_metrics(self) -> Dict:
        """Collect all metrics including HPC-specific ones"""
        metrics = super().collect_all_metrics()

        # Add HPC-specific metrics
        metrics.update({
            'infiniband_stats': self.get_infiniband_stats(),
            'pfs_performance': self.get_pfs_performance(),
            'slurm_node_info': self.get_slurm_node_info()
        })

        return metrics


class AresDataCollector:
    """Data collector optimized for ARES HPC cluster"""

    def __init__(self, config_path: Path, output_dir: Path, node_id: Optional[str] = None):
        self.config = load_config(config_path)
        self.output_dir = Path(output_dir)
        self.running = False
        self.shutdown_event = threading.Event()

        # Determine node ID
        if node_id:
            self.node_id = node_id
        else:
            hostname = os.uname().nodename
            prefix = self.config['environments']['ares']['node_id_prefix']
            self.node_id = f"{prefix}-{hostname}"

        # Initialize components
        self.ntp_client = NTPClient(
            servers=self.config['ntp']['servers'],
            timeout=self.config['ntp']['timeout_seconds']
        )
        self.metrics_collector = AresMetricsCollector()
        self.writer = DatasetWriter(
            output_dir=self.output_dir,
            node_id=self.node_id,
            max_file_size_mb=self.config['collection']['max_file_size_mb']
        )

        # Statistics
        self.measurement_count = 0
        self.error_count = 0
        self.last_backup = time.time()

        # Previous measurement for drift calculation
        self.previous_measurement = None

    def _calculate_drift_rate(self, current_offset: float, current_time: float) -> float:
        """Calculate drift rate from previous measurement"""
        if self.previous_measurement is None:
            return 0.0

        prev_offset, prev_time = self.previous_measurement
        time_delta = current_time - prev_time

        if time_delta > 0:
            drift_rate = (current_offset - prev_offset) / time_delta
            return drift_rate

        return 0.0

    def _get_ground_truth_offset(self) -> Optional[float]:
        """Get ground truth using NTP ensemble"""
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
        """Collect a single timing measurement with HPC-specific data"""
        measurement_time = time.time()

        # Get NTP measurement
        ntp_result = self.ntp_client.get_best_measurement()
        if not ntp_result:
            self.error_count += 1
            return None

        offset, delay, server, stratum = ntp_result

        # Calculate drift rate
        drift_rate = self._calculate_drift_rate(offset, measurement_time)
        self.previous_measurement = (offset, measurement_time)

        # Get system metrics (including HPC-specific)
        metrics = self.metrics_collector.collect_all_metrics()

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
        if abs(drift_rate) > self.config['quality_control']['max_drift_rate']:
            quality_flags.append('high_drift_rate')

        # Add HPC-specific quality flags
        if metrics.get('infiniband_stats') and metrics['infiniband_stats'].get('state') != 'Active':
            quality_flags.append('infiniband_issue')

        if metrics.get('pfs_performance'):
            for mount, stats in metrics['pfs_performance'].items():
                if isinstance(stats, dict) and 'write_latency_ms' in stats:
                    if stats['write_latency_ms'] > 100:  # > 100ms write latency
                        quality_flags.append('pfs_slow_write')

        # Calculate uncertainty
        uncertainty = max(delay / 2.0, 1e-6)

        # Flatten HPC metrics for CSV storage
        hpc_metrics = {}
        if metrics.get('infiniband_stats'):
            hpc_metrics.update({f"ib_{k}": v for k, v in metrics['infiniband_stats'].items()})
        if metrics.get('slurm_node_info'):
            hpc_metrics.update({f"slurm_{k}": v for k, v in metrics['slurm_node_info'].items()})

        measurement = TimingMeasurement(
            timestamp=measurement_time,
            node_id=self.node_id,
            clock_offset=offset,
            drift_rate=drift_rate,
            ntp_delay=delay,
            ntp_stratum=stratum,
            ntp_server=server,
            cpu_temp=metrics.get('cpu_temp'),
            gpu_temp=metrics.get('gpu_temp'),  # Usually None on HPC compute nodes
            cpu_freq=metrics.get('cpu_freq'),
            cpu_load=metrics.get('cpu_load'),
            memory_usage=metrics.get('memory_usage'),
            network_latency=metrics.get('network_latency'),
            ground_truth_offset=ground_truth_offset,
            measurement_uncertainty=uncertainty,
            source_type='ares_hpc_collection',
            quality_flags=quality_flags
        )

        return measurement

    def _backup_progress(self):
        """Create backup with HPC-specific information"""
        try:
            backup_file = self.output_dir / f"progress_backup_{self.node_id}_{int(time.time())}.json"
            progress_data = {
                'node_id': self.node_id,
                'measurement_count': self.measurement_count,
                'error_count': self.error_count,
                'last_backup': self.last_backup,
                'uptime_hours': (time.time() - self.start_time) / 3600,
                'success_rate': (self.measurement_count / max(1, self.measurement_count + self.error_count)),
                'system_info': get_system_info()._asdict(),
                'hpc_environment': {
                    'infiniband_available': self.metrics_collector.infiniband_available,
                    'slurm_available': self.metrics_collector.slurm_available,
                    'pfs_mounts': self.metrics_collector.pfs_mounts
                }
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

        print(f"Starting ARES HPC data collection on node {self.node_id}")
        print(f"Duration: {duration_hours or self.config['collection']['duration_hours']} hours")
        print(f"Output directory: {self.output_dir}")
        print(f"HPC Features:")
        print(f"  InfiniBand: {self.metrics_collector.infiniband_available}")
        print(f"  SLURM: {self.metrics_collector.slurm_available}")
        print(f"  PFS mounts: {len(self.metrics_collector.pfs_mounts)}")

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
                        print(f"[{self.node_id}] Progress: {self.measurement_count} measurements, "
                              f"{hours_running:.1f}h running, {success_rate:.1%} success rate")

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
            print(f"\n[{self.node_id}] Received interrupt signal - stopping gracefully...")
        except Exception as e:
            print(f"[{self.node_id}] Collection error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop()

    def stop(self):
        """Stop data collection and cleanup"""
        self.running = False
        print(f"[{self.node_id}] Stopping data collection...")

        # Final backup
        self._backup_progress()

        # Close writer
        self.writer.close()

        hours_running = (time.time() - self.start_time) / 3600
        success_rate = self.measurement_count / max(1, self.measurement_count + self.error_count)

        print(f"\n[{self.node_id}] Collection complete:")
        print(f"  Measurements: {self.measurement_count}")
        print(f"  Errors: {self.error_count}")
        print(f"  Success rate: {success_rate:.1%}")
        print(f"  Duration: {hours_running:.1f} hours")


def signal_handler(signum, frame, collector):
    """Handle shutdown signals"""
    print(f"\n[{collector.node_id}] Received signal {signum} - initiating graceful shutdown...")
    collector.shutdown_event.set()


def main():
    parser = argparse.ArgumentParser(description="ChronoTick Evaluation 1 - ARES HPC Data Collector")
    parser.add_argument('--config', type=Path,
                        default=Path(__file__).parent / 'collect_config.yaml',
                        help='Configuration file path')
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).parent.parent / 'datasets' / 'raw' / 'ares',
                        help='Output directory')
    parser.add_argument('--node-id', type=str, default=None,
                        help='Node identifier (auto-detected if not provided)')
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
    collector = AresDataCollector(args.config, args.output, args.node_id)

    # Setup signal handlers
    signal.signal(signal.SIGINT, lambda s, f: signal_handler(s, f, collector))
    signal.signal(signal.SIGTERM, lambda s, f: signal_handler(s, f, collector))

    # Run collection
    collector.run_collection(args.duration)


if __name__ == "__main__":
    main()