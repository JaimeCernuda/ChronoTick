#!/usr/bin/env python3
"""
ChronoTick Evaluation 1 - Collection Utilities

Shared utilities for dataset collection across all environments.
"""

import os
import sys
import time
import socket
import struct
import logging
import yaml
import csv
import gzip
import json
import psutil
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import subprocess
import platform

# Add ChronoTick to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tsfm"))

logger = logging.getLogger(__name__)


class SystemInfo(NamedTuple):
    """System information for metadata"""
    hostname: str
    platform: str
    cpu_count: int
    memory_gb: float
    gpu_info: Optional[str]
    network_interfaces: List[str]
    kernel_version: str


@dataclass
class TimingMeasurement:
    """Single timing measurement with all metadata"""
    timestamp: float
    node_id: str
    clock_offset: float
    drift_rate: float
    ntp_delay: float
    ntp_stratum: int
    ntp_server: str
    cpu_temp: Optional[float]
    gpu_temp: Optional[float]
    cpu_freq: Optional[float]
    cpu_load: float
    memory_usage: float
    network_latency: Optional[float]
    ground_truth_offset: Optional[float]
    measurement_uncertainty: float
    source_type: str
    quality_flags: List[str]


class NTPClient:
    """Lightweight NTP client for reference measurements"""

    def __init__(self, servers: List[str], timeout: float = 2.0):
        self.servers = servers
        self.timeout = timeout
        self.NTP_PACKET_FORMAT = "!12I"
        self.NTP_EPOCH_OFFSET = 2208988800

    def query_server(self, server: str) -> Optional[Tuple[float, float, int]]:
        """Query single NTP server, return (offset, delay, stratum)"""
        try:
            t1 = time.time()

            # Create NTP request packet
            packet = [0] * 12
            packet[0] = 0x1B000000  # NTP v3 client request
            ntp_packet = struct.pack(self.NTP_PACKET_FORMAT, *packet)

            # Send request
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(self.timeout)

            try:
                sock.sendto(ntp_packet, (server, 123))
                response, _ = sock.recvfrom(1024)
                t4 = time.time()
            finally:
                sock.close()

            # Parse response
            unpacked = struct.unpack(self.NTP_PACKET_FORMAT, response)

            # Extract stratum
            stratum = (unpacked[0] >> 16) & 0xFF

            # Extract timestamps (convert from NTP to Unix time)
            t2 = unpacked[8] - self.NTP_EPOCH_OFFSET + (unpacked[9] / 2**32)
            t3 = unpacked[10] - self.NTP_EPOCH_OFFSET + (unpacked[11] / 2**32)

            # Calculate offset and delay
            offset = ((t2 - t1) + (t3 - t4)) / 2.0
            delay = (t4 - t1) - (t3 - t2)

            return offset, delay, stratum

        except Exception as e:
            logger.warning(f"NTP query failed for {server}: {e}")
            return None

    def get_best_measurement(self) -> Optional[Tuple[float, float, str, int]]:
        """Get best measurement from available servers"""
        measurements = []

        for server in self.servers:
            result = self.query_server(server)
            if result:
                offset, delay, stratum = result
                measurements.append((offset, delay, server, stratum))

        if not measurements:
            return None

        # Select best measurement (lowest delay, highest stratum)
        best = min(measurements, key=lambda x: (x[1], -x[3]))
        return best


class SystemMetricsCollector:
    """Collect system metrics for covariates"""

    def __init__(self):
        self.gpu_available = self._check_gpu_availability()
        self.thermal_zones = self._find_thermal_zones()

    def _check_gpu_availability(self) -> bool:
        """Check if NVIDIA GPU is available"""
        try:
            result = subprocess.run(['nvidia-smi', '-L'],
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except:
            return False

    def _find_thermal_zones(self) -> List[str]:
        """Find available thermal zones"""
        zones = []
        thermal_path = Path("/sys/class/thermal")
        if thermal_path.exists():
            for zone_dir in thermal_path.glob("thermal_zone*"):
                try:
                    with open(zone_dir / "type") as f:
                        zone_type = f.read().strip()
                    zones.append(str(zone_dir))
                except:
                    continue
        return zones

    def get_cpu_temperature(self) -> Optional[float]:
        """Get CPU temperature in Celsius"""
        try:
            # Try multiple methods

            # Method 1: psutil sensors
            if hasattr(psutil, 'sensors_temperatures'):
                temps = psutil.sensors_temperatures()
                if 'coretemp' in temps and temps['coretemp']:
                    return temps['coretemp'][0].current
                elif 'cpu_thermal' in temps and temps['cpu_thermal']:
                    return temps['cpu_thermal'][0].current

            # Method 2: thermal zones
            for zone_path in self.thermal_zones:
                try:
                    with open(f"{zone_path}/temp") as f:
                        temp = int(f.read().strip()) / 1000.0
                        if 20 < temp < 100:  # Reasonable CPU temp range
                            return temp
                except:
                    continue

            return None

        except Exception as e:
            logger.debug(f"CPU temperature unavailable: {e}")
            return None

    def get_gpu_temperature(self) -> Optional[float]:
        """Get GPU temperature in Celsius"""
        if not self.gpu_available:
            return None

        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=temperature.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)

            if result.returncode == 0:
                return float(result.stdout.strip())

        except Exception as e:
            logger.debug(f"GPU temperature unavailable: {e}")

        return None

    def get_cpu_frequency(self) -> Optional[float]:
        """Get current CPU frequency in MHz"""
        try:
            freq = psutil.cpu_freq()
            if freq:
                return freq.current
        except Exception as e:
            logger.debug(f"CPU frequency unavailable: {e}")

        return None

    def get_network_latency(self, target: str = "8.8.8.8") -> Optional[float]:
        """Measure network latency via ping"""
        try:
            start_time = time.time()
            result = subprocess.run([
                'ping', '-c', '1', '-W', '1', target
            ], capture_output=True, timeout=3)

            if result.returncode == 0:
                # Simple latency estimation
                return (time.time() - start_time) * 1000  # ms

        except Exception as e:
            logger.debug(f"Network latency measurement failed: {e}")

        return None

    def collect_all_metrics(self) -> Dict:
        """Collect all available system metrics"""
        return {
            'cpu_temp': self.get_cpu_temperature(),
            'gpu_temp': self.get_gpu_temperature(),
            'cpu_freq': self.get_cpu_frequency(),
            'cpu_load': psutil.cpu_percent(interval=None),
            'memory_usage': psutil.virtual_memory().percent,
            'network_latency': self.get_network_latency()
        }


class DatasetWriter:
    """Write measurements to CSV files with compression and rotation"""

    def __init__(self, output_dir: Path, node_id: str, max_file_size_mb: int = 100):
        self.output_dir = Path(output_dir)
        self.node_id = node_id
        self.max_file_size = max_file_size_mb * 1024 * 1024
        self.current_file = None
        self.writer = None
        self.file_counter = 0
        self.lock = threading.Lock()

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # CSV headers
        self.headers = [
            'timestamp', 'node_id', 'clock_offset', 'drift_rate', 'ntp_delay',
            'ntp_stratum', 'ntp_server', 'cpu_temp', 'gpu_temp', 'cpu_freq',
            'cpu_load', 'memory_usage', 'network_latency', 'ground_truth_offset',
            'measurement_uncertainty', 'source_type', 'quality_flags'
        ]

    def _get_next_filename(self) -> Path:
        """Get next filename with rotation"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chronotick_eval1_{self.node_id}_{timestamp}_{self.file_counter:03d}.csv.gz"
        self.file_counter += 1
        return self.output_dir / filename

    def _rotate_file_if_needed(self):
        """Rotate file if size limit exceeded"""
        if (self.current_file and
            self.current_file.exists() and
            self.current_file.stat().st_size > self.max_file_size):

            self._close_current_file()

    def _open_new_file(self):
        """Open new output file"""
        self._close_current_file()

        self.current_file = self._get_next_filename()
        self.file_handle = gzip.open(self.current_file, 'wt', newline='')
        self.writer = csv.writer(self.file_handle)
        self.writer.writerow(self.headers)

        logger.info(f"Started new data file: {self.current_file}")

    def _close_current_file(self):
        """Close current file"""
        if hasattr(self, 'file_handle') and self.file_handle:
            self.file_handle.close()
            self.writer = None

    def write_measurement(self, measurement: TimingMeasurement):
        """Write single measurement to file"""
        with self.lock:
            if not self.writer:
                self._open_new_file()

            self._rotate_file_if_needed()

            row = [
                measurement.timestamp,
                measurement.node_id,
                measurement.clock_offset,
                measurement.drift_rate,
                measurement.ntp_delay,
                measurement.ntp_stratum,
                measurement.ntp_server,
                measurement.cpu_temp,
                measurement.gpu_temp,
                measurement.cpu_freq,
                measurement.cpu_load,
                measurement.memory_usage,
                measurement.network_latency,
                measurement.ground_truth_offset,
                measurement.measurement_uncertainty,
                measurement.source_type,
                ','.join(measurement.quality_flags) if measurement.quality_flags else ''
            ]

            self.writer.writerow(row)
            self.file_handle.flush()

    def close(self):
        """Close writer and files"""
        with self.lock:
            self._close_current_file()


def get_system_info() -> SystemInfo:
    """Collect system information for metadata"""

    # Get GPU info
    gpu_info = None
    try:
        result = subprocess.run(['nvidia-smi', '-L'],
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            gpu_info = result.stdout.strip()
    except:
        pass

    # Get network interfaces
    interfaces = []
    try:
        net_if = psutil.net_if_addrs()
        interfaces = list(net_if.keys())
    except:
        pass

    return SystemInfo(
        hostname=socket.gethostname(),
        platform=platform.platform(),
        cpu_count=psutil.cpu_count(),
        memory_gb=psutil.virtual_memory().total / (1024**3),
        gpu_info=gpu_info,
        network_interfaces=interfaces,
        kernel_version=platform.release()
    )


def load_config(config_path: Path) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_logging(level: str = "INFO", log_file: Optional[Path] = None):
    """Setup logging configuration"""
    logging_config = {
        'level': getattr(logging, level.upper()),
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'datefmt': '%Y-%m-%d %H:%M:%S'
    }

    if log_file:
        logging_config['filename'] = str(log_file)

    logging.basicConfig(**logging_config)