#!/usr/bin/env python3
"""
ChronoTick Evaluation 1 - Chameleon Cloud Direct Node Collector

Collects timing measurements on Chameleon Cloud with direct node access.
Focuses on cloud/virtualization effects and GPU-accelerated workloads.
"""

import os
import sys
import time
import signal
import argparse
import threading
import subprocess
import requests
from pathlib import Path
from typing import Optional, Dict, List

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    TimingMeasurement, NTPClient, SystemMetricsCollector,
    DatasetWriter, load_config, setup_logging, get_system_info
)


class ChameleonMetricsCollector(SystemMetricsCollector):
    """Extended metrics collector for Chameleon Cloud environment"""

    def __init__(self):
        super().__init__()
        self.openstack_metadata = self._get_openstack_metadata()
        self.instance_type = self._get_instance_type()
        self.cloud_init_complete = self._check_cloud_init()

    def _get_openstack_metadata(self) -> Optional[Dict]:
        """Get OpenStack instance metadata"""
        try:
            # OpenStack metadata service
            metadata_url = "http://169.254.169.254/openstack/latest/meta_data.json"
            response = requests.get(metadata_url, timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            pass

        # Fallback: try EC2-style metadata
        try:
            metadata_url = "http://169.254.169.254/latest/meta-data/"
            response = requests.get(metadata_url, timeout=5)
            if response.status_code == 200:
                return {'ec2_style': True, 'available': True}
        except:
            pass

        return None

    def _get_instance_type(self) -> Optional[str]:
        """Get cloud instance type/flavor"""
        if self.openstack_metadata:
            # Check OpenStack metadata first
            if 'meta' in self.openstack_metadata:
                return self.openstack_metadata['meta'].get('flavor_name')

        # Try DMI information
        try:
            with open('/sys/class/dmi/id/product_name') as f:
                product = f.read().strip()
            with open('/sys/class/dmi/id/sys_vendor') as f:
                vendor = f.read().strip()
            return f"{vendor}_{product}"
        except:
            pass

        return None

    def _check_cloud_init(self) -> bool:
        """Check if cloud-init has completed"""
        try:
            result = subprocess.run(['cloud-init', 'status'],
                                  capture_output=True, text=True, timeout=5)
            return 'done' in result.stdout.lower()
        except:
            # Assume completed if cloud-init not available
            return True

    def get_virtualization_info(self) -> Optional[Dict]:
        """Get virtualization layer information"""
        info = {}

        # Check hypervisor type
        try:
            with open('/sys/hypervisor/type') as f:
                info['hypervisor'] = f.read().strip()
        except:
            pass

        # Check if running in container
        try:
            with open('/proc/1/cgroup') as f:
                cgroup_content = f.read()
                if 'docker' in cgroup_content:
                    info['container'] = 'docker'
                elif 'lxc' in cgroup_content:
                    info['container'] = 'lxc'
        except:
            pass

        # Check CPU features (virtualization indicators)
        try:
            with open('/proc/cpuinfo') as f:
                cpuinfo = f.read()
                if 'hypervisor' in cpuinfo:
                    info['virtualized'] = True
                if 'vmx' in cpuinfo or 'svm' in cpuinfo:
                    info['nested_virt_capable'] = True
        except:
            pass

        return info if info else None

    def get_network_performance(self) -> Optional[Dict]:
        """Measure cloud network performance"""
        stats = {}

        # Measure latency to common targets
        targets = {
            'google_dns': '8.8.8.8',
            'cloudflare_dns': '1.1.1.1',
            'local_gateway': None  # Will determine dynamically
        }

        # Find local gateway
        try:
            result = subprocess.run(['ip', 'route', 'show', 'default'],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                gateway = result.stdout.split()[2]
                targets['local_gateway'] = gateway
        except:
            pass

        # Ping each target
        for name, target in targets.items():
            if target is None:
                continue

            try:
                result = subprocess.run([
                    'ping', '-c', '3', '-W', '2', target
                ], capture_output=True, text=True, timeout=10)

                if result.returncode == 0:
                    # Parse ping output for average latency
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if 'avg' in line and 'ms' in line:
                            # Extract average from "min/avg/max/mdev = ..."
                            avg_part = line.split('=')[1].split('/')[1]
                            stats[f'{name}_latency_ms'] = float(avg_part)
                            break

            except Exception as e:
                stats[f'{name}_error'] = str(e)

        # Bandwidth test (simple)
        try:
            # Download speed test using curl
            start = time.time()
            result = subprocess.run([
                'curl', '-s', '-o', '/dev/null', '-w', '%{speed_download}',
                'http://speedtest.ftp.otenet.gr/files/test100k.db'
            ], capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                speed_bytes_per_sec = float(result.stdout.strip())
                stats['download_speed_mbps'] = (speed_bytes_per_sec * 8) / (1024 * 1024)

        except Exception as e:
            stats['bandwidth_test_error'] = str(e)

        return stats

    def get_cloud_storage_performance(self) -> Optional[Dict]:
        """Test cloud storage I/O performance"""
        stats = {}

        # Test local disk performance
        test_dirs = ['/tmp', '/home', '/var/tmp']

        for test_dir in test_dirs:
            if not os.path.exists(test_dir):
                continue

            try:
                test_file = os.path.join(test_dir, f'chronotick_iotest_{os.getpid()}')

                # Write test (1MB)
                test_data = b'x' * (1024 * 1024)
                start = time.time()
                with open(test_file, 'wb') as f:
                    f.write(test_data)
                    f.flush()
                    os.fsync(f.fileno())
                write_time = time.time() - start

                # Read test
                start = time.time()
                with open(test_file, 'rb') as f:
                    _ = f.read()
                read_time = time.time() - start

                # Cleanup
                os.remove(test_file)

                stats[f'{test_dir}_write_mbps'] = len(test_data) / (write_time * 1024 * 1024)
                stats[f'{test_dir}_read_mbps'] = len(test_data) / (read_time * 1024 * 1024)

            except Exception as e:
                stats[f'{test_dir}_error'] = str(e)

        return stats

    def collect_all_metrics(self) -> Dict:
        """Collect all metrics including cloud-specific ones"""
        metrics = super().collect_all_metrics()

        # Add cloud-specific metrics
        metrics.update({
            'virtualization_info': self.get_virtualization_info(),
            'network_performance': self.get_network_performance(),
            'storage_performance': self.get_cloud_storage_performance(),
            'instance_type': self.instance_type,
            'cloud_init_complete': self.cloud_init_complete
        })

        return metrics


class ChameleonDataCollector:
    """Data collector optimized for Chameleon Cloud"""

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
            prefix = self.config['environments']['chameleon']['node_id_prefix']
            self.node_id = f"{prefix}-{hostname}"

        # Initialize components
        self.ntp_client = NTPClient(
            servers=self.config['ntp']['servers'],
            timeout=self.config['ntp']['timeout_seconds']
        )
        self.metrics_collector = ChameleonMetricsCollector()
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
        """Collect a single timing measurement with cloud-specific data"""
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

        # Get system metrics (including cloud-specific)
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

        # Add cloud-specific quality flags
        if not metrics.get('cloud_init_complete'):
            quality_flags.append('cloud_init_incomplete')

        if metrics.get('network_performance'):
            net_perf = metrics['network_performance']
            if net_perf.get('google_dns_latency_ms', 0) > 100:
                quality_flags.append('high_network_latency')

        if metrics.get('virtualization_info'):
            virt_info = metrics['virtualization_info']
            if virt_info.get('container'):
                quality_flags.append('containerized')

        # Calculate uncertainty (higher in virtualized environments)
        base_uncertainty = max(delay / 2.0, 1e-6)
        if metrics.get('virtualization_info', {}).get('virtualized'):
            uncertainty = base_uncertainty * 1.5  # Virtualization penalty
        else:
            uncertainty = base_uncertainty

        measurement = TimingMeasurement(
            timestamp=measurement_time,
            node_id=self.node_id,
            clock_offset=offset,
            drift_rate=drift_rate,
            ntp_delay=delay,
            ntp_stratum=stratum,
            ntp_server=server,
            cpu_temp=metrics.get('cpu_temp'),
            gpu_temp=metrics.get('gpu_temp'),
            cpu_freq=metrics.get('cpu_freq'),
            cpu_load=metrics.get('cpu_load'),
            memory_usage=metrics.get('memory_usage'),
            network_latency=metrics.get('network_latency'),
            ground_truth_offset=ground_truth_offset,
            measurement_uncertainty=uncertainty,
            source_type='chameleon_cloud_collection',
            quality_flags=quality_flags
        )

        return measurement

    def _backup_progress(self):
        """Create backup with cloud-specific information"""
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
                'cloud_environment': {
                    'openstack_metadata': self.metrics_collector.openstack_metadata,
                    'instance_type': self.metrics_collector.instance_type,
                    'cloud_init_complete': self.metrics_collector.cloud_init_complete,
                    'gpu_available': self.metrics_collector.gpu_available
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

        print(f"Starting Chameleon Cloud data collection on node {self.node_id}")
        print(f"Duration: {duration_hours or self.config['collection']['duration_hours']} hours")
        print(f"Output directory: {self.output_dir}")
        print(f"Cloud Features:")
        print(f"  Instance type: {self.metrics_collector.instance_type}")
        print(f"  GPU available: {self.metrics_collector.gpu_available}")
        print(f"  Cloud-init complete: {self.metrics_collector.cloud_init_complete}")

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
    parser = argparse.ArgumentParser(description="ChronoTick Evaluation 1 - Chameleon Cloud Data Collector")
    parser.add_argument('--config', type=Path,
                        default=Path(__file__).parent / 'collect_config.yaml',
                        help='Configuration file path')
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).parent.parent / 'datasets' / 'raw' / 'chameleon',
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
    collector = ChameleonDataCollector(args.config, args.output, args.node_id)

    # Setup signal handlers
    signal.signal(signal.SIGINT, lambda s, f: signal_handler(s, f, collector))
    signal.signal(signal.SIGTERM, lambda s, f: signal_handler(s, f, collector))

    # Run collection
    collector.run_collection(args.duration)


if __name__ == "__main__":
    main()