#!/usr/bin/env python3
"""
ChronoTick NTP Client

High-precision NTP client for reference time measurements.
Replaces synthetic clock data with real NTP offset measurements.
"""

import time
import socket
import struct
import threading
import logging
from typing import List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
import statistics
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


class NTPMeasurement(NamedTuple):
    """Single NTP measurement result"""
    offset: float          # Local clock - NTP time (seconds)
    delay: float          # Round-trip delay (seconds)  
    stratum: int          # NTP stratum level
    precision: float      # Server precision (seconds)
    server: str           # NTP server used
    timestamp: float      # Local time when measurement taken
    uncertainty: float    # Estimated measurement uncertainty


@dataclass
class NTPConfig:
    """NTP client configuration"""
    servers: List[str]
    timeout_seconds: float = 2.0
    max_acceptable_uncertainty: float = 0.010  # 10ms
    min_stratum: int = 3
    max_delay: float = 0.100  # 100ms max acceptable delay


class NTPClient:
    """
    High-precision NTP client for clock offset measurements.
    
    Queries multiple NTP servers and selects the best measurement
    based on delay, stratum, and precision.
    """
    
    def __init__(self, config: NTPConfig):
        """Initialize NTP client with configuration"""
        self.config = config
        self.measurement_history = []
        self.lock = threading.Lock()
        
        # NTP packet format constants
        self.NTP_PACKET_FORMAT = "!12I"
        self.NTP_EPOCH_OFFSET = 2208988800  # Seconds between 1900 and 1970
        
    def measure_offset(self, server: str) -> Optional[NTPMeasurement]:
        """
        Measure clock offset against single NTP server.
        
        Returns: NTPMeasurement or None if measurement failed
        """
        try:
            # Record precise local time before NTP request
            t1_local = time.time()
            
            # Create NTP request packet
            ntp_packet = self._create_ntp_request()
            
            # Send request to NTP server
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(self.config.timeout_seconds)
            
            try:
                sock.sendto(ntp_packet, (server, 123))
                response, _ = sock.recvfrom(1024)
                t4_local = time.time()  # Record local time after response
                
            finally:
                sock.close()
            
            # Parse NTP response
            ntp_data = self._parse_ntp_response(response)
            if not ntp_data:
                return None
                
            # Extract timestamps from NTP packet
            # t1 = client send time (we use local time)
            # t2 = server receive time 
            # t3 = server transmit time
            # t4 = client receive time (we use local time)
            t1 = t1_local
            t2 = ntp_data['receive_timestamp']
            t3 = ntp_data['transmit_timestamp'] 
            t4 = t4_local
            
            # Calculate offset and delay using standard NTP formulas
            # Offset = ((t2 - t1) + (t3 - t4)) / 2
            # Delay = (t4 - t1) - (t3 - t2)
            offset = ((t2 - t1) + (t3 - t4)) / 2.0
            delay = (t4 - t1) - (t3 - t2)
            
            # Estimate uncertainty based on network delay
            uncertainty = max(delay / 2.0, ntp_data['precision'])
            
            # Validate measurement quality
            if (delay > self.config.max_delay or 
                ntp_data['stratum'] < self.config.min_stratum or
                uncertainty > self.config.max_acceptable_uncertainty):
                logger.warning(f"Poor NTP measurement from {server}: "
                             f"delay={delay*1000:.1f}ms, stratum={ntp_data['stratum']}, "
                             f"uncertainty={uncertainty*1000:.1f}ms")
                return None
            
            measurement = NTPMeasurement(
                offset=offset,
                delay=delay,
                stratum=ntp_data['stratum'],
                precision=ntp_data['precision'],
                server=server,
                timestamp=t1_local,
                uncertainty=uncertainty
            )
            
            logger.debug(f"NTP measurement from {server}: "
                        f"offset={offset*1e6:.1f}μs, delay={delay*1000:.1f}ms")
            
            return measurement
            
        except socket.timeout:
            logger.warning(f"NTP timeout for server {server}")
            return None
        except Exception as e:
            logger.error(f"NTP measurement failed for {server}: {e}")
            return None
    
    def _create_ntp_request(self) -> bytes:
        """Create NTP request packet"""
        # NTP packet: 48 bytes
        # First word: LI=0, VN=3, Mode=3 (client request) in MSB
        # Rest: zeros for request
        packet = [0] * 12
        packet[0] = 0x1B000000  # 00 011 011 in most significant byte
        
        return struct.pack(self.NTP_PACKET_FORMAT, *packet)
    
    def _parse_ntp_response(self, packet: bytes) -> Optional[dict]:
        """Parse NTP response packet"""
        try:
            if len(packet) < 48:
                return None
                
            # Unpack NTP packet (12 32-bit words)
            data = struct.unpack(self.NTP_PACKET_FORMAT, packet)
            
            # Extract key fields
            li_vn_mode = data[0] >> 24
            stratum = (data[0] >> 16) & 0xFF
            precision = struct.unpack('>b', struct.pack('>B', data[0] & 0xFF))[0]  # Precision is in bits 0-7
            
            # Convert precision from log2 seconds to seconds
            precision_seconds = 2.0 ** precision
            
            # Extract timestamps (seconds since 1900)
            receive_timestamp_int = data[8]
            receive_timestamp_frac = data[9]
            transmit_timestamp_int = data[10] 
            transmit_timestamp_frac = data[11]
            
            # Convert to Unix timestamps (seconds since 1970)
            receive_timestamp = (receive_timestamp_int - self.NTP_EPOCH_OFFSET + 
                               receive_timestamp_frac / (2**32))
            transmit_timestamp = (transmit_timestamp_int - self.NTP_EPOCH_OFFSET +
                                transmit_timestamp_frac / (2**32))
            
            return {
                'stratum': stratum,
                'precision': precision_seconds,
                'receive_timestamp': receive_timestamp,
                'transmit_timestamp': transmit_timestamp
            }
            
        except Exception as e:
            logger.error(f"Failed to parse NTP response: {e}")
            return None
    
    def get_best_measurement(self) -> Optional[NTPMeasurement]:
        """
        Query multiple NTP servers and return the best measurement.
        
        Selection criteria:
        1. Lowest delay (best network path)
        2. Highest stratum (more accurate)
        3. Best precision (server quality)
        """
        measurements = []
        
        # Query all configured servers
        for server in self.config.servers:
            measurement = self.measure_offset(server)
            if measurement:
                measurements.append(measurement)
        
        if not measurements:
            logger.error("No successful NTP measurements from any server")
            return None
        
        # Select best measurement
        # Primary: lowest delay, Secondary: highest stratum
        best_measurement = min(measurements, 
                             key=lambda m: (m.delay, -m.stratum, m.uncertainty))
        
        logger.info(f"Selected NTP measurement from {best_measurement.server}: "
                   f"offset={best_measurement.offset*1e6:.1f}μs, "
                   f"delay={best_measurement.delay*1000:.1f}ms, "
                   f"stratum={best_measurement.stratum}")
        
        # Store in history
        with self.lock:
            self.measurement_history.append(best_measurement)
            # Keep recent history only
            if len(self.measurement_history) > 100:
                self.measurement_history = self.measurement_history[-50:]
        
        return best_measurement
    
    def get_measurement_statistics(self) -> dict:
        """Get statistics on recent NTP measurements"""
        with self.lock:
            if not self.measurement_history:
                return {"status": "no_measurements"}
            
            recent = self.measurement_history[-10:]  # Last 10 measurements
            
            offsets = [m.offset for m in recent]
            delays = [m.delay for m in recent]
            
            return {
                "status": "active",
                "total_measurements": len(self.measurement_history),
                "recent_count": len(recent),
                "offset_stats": {
                    "mean": statistics.mean(offsets),
                    "stdev": statistics.stdev(offsets) if len(offsets) > 1 else 0,
                    "range": max(offsets) - min(offsets)
                },
                "delay_stats": {
                    "mean": statistics.mean(delays),
                    "min": min(delays),
                    "max": max(delays)
                },
                "servers_used": list(set(m.server for m in recent))
            }


class ClockMeasurementCollector:
    """
    Collects real clock offset measurements with configurable timing.
    Replaces synthetic ClockDataGenerator with real NTP measurements.
    """
    
    def __init__(self, config_path: str):
        """Initialize collector with configuration"""
        self.config = self._load_config(config_path)
        
        # NTP configuration
        ntp_config = NTPConfig(
            servers=self.config['clock_measurement']['ntp']['servers'],
            timeout_seconds=self.config['clock_measurement']['ntp']['timeout_seconds'],
            max_acceptable_uncertainty=self.config['clock_measurement']['ntp']['max_acceptable_uncertainty'],
            min_stratum=self.config['clock_measurement']['ntp']['min_stratum']
        )
        self.ntp_client = NTPClient(ntp_config)
        
        # Timing configuration
        self.warm_up_duration = self.config['clock_measurement']['timing']['warm_up']['duration_seconds']
        self.warm_up_interval = self.config['clock_measurement']['timing']['warm_up']['measurement_interval']
        self.normal_interval = self.config['clock_measurement']['timing']['normal_operation']['measurement_interval']
        
        # Collection state
        self.collection_thread = None
        self.collection_running = False
        self.start_time = 0
        self.last_measurement = None
        self.last_measurement_time = 0
        self.lock = threading.Lock()
        
        # Real measurement storage
        self.offset_measurements = []  # (timestamp, offset, uncertainty)
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config {config_path}: {e}")
            raise
    
    def start_collection(self):
        """Start collecting NTP measurements with warm-up then normal intervals"""
        if self.collection_running:
            logger.warning("Clock measurement collection already running")
            return
        
        self.collection_running = True
        self.start_time = time.time()
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        
        logger.info(f"Started clock measurement collection - "
                   f"warm-up: {self.warm_up_duration}s @ {self.warm_up_interval}s intervals, "
                   f"then {self.normal_interval}s intervals")
    
    def stop_collection(self):
        """Stop collecting measurements"""
        self.collection_running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5.0)
        logger.info("Stopped clock measurement collection")
    
    def _collection_loop(self):
        """Main collection loop with warm-up then normal intervals"""
        while self.collection_running:
            try:
                current_time = time.time()
                elapsed = current_time - self.start_time
                
                # Determine measurement interval based on warm-up phase
                if elapsed < self.warm_up_duration:
                    interval = self.warm_up_interval
                    phase = "warm-up"
                else:
                    interval = self.normal_interval
                    phase = "normal"
                
                # Take NTP measurement
                measurement = self.ntp_client.get_best_measurement()
                
                if measurement:
                    with self.lock:
                        self.last_measurement = measurement
                        self.last_measurement_time = current_time
                        self.offset_measurements.append((
                            measurement.timestamp,
                            measurement.offset,
                            measurement.uncertainty
                        ))
                        
                        # Manage storage size
                        if len(self.offset_measurements) > 1000:
                            self.offset_measurements = self.offset_measurements[-500:]
                    
                    logger.debug(f"Collected {phase} measurement: "
                               f"offset={measurement.offset*1e6:.1f}μs")
                else:
                    logger.warning(f"Failed to get NTP measurement in {phase} phase")
                
                # Wait for next interval
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Collection loop error: {e}")
                time.sleep(min(self.warm_up_interval, self.normal_interval))
    
    def get_latest_offset(self) -> Optional[float]:
        """Get the most recent clock offset measurement"""
        with self.lock:
            if self.last_measurement:
                return self.last_measurement.offset
            return None
    
    def has_new_measurement(self) -> bool:
        """Check if there's a new NTP measurement since last check"""
        # This would be used to trigger retrospective correction
        # Implementation depends on how we track "new" measurements
        return False  # Placeholder
    
    def get_recent_measurements(self, window_seconds: int = 300) -> List[Tuple[float, float, float]]:
        """Get recent offset measurements within time window"""
        with self.lock:
            current_time = time.time()
            cutoff_time = current_time - window_seconds
            
            return [(ts, offset, uncertainty) for ts, offset, uncertainty in self.offset_measurements 
                   if ts >= cutoff_time]


def create_test_collector():
    """Create a test collector for development"""
    config_path = Path(__file__).parent / "configs" / "hybrid_timesfm_chronos.yaml"
    return ClockMeasurementCollector(str(config_path))


if __name__ == "__main__":
    # Test NTP client
    print("Testing NTP Client...")
    
    config = NTPConfig(
        servers=["pool.ntp.org", "time.google.com"],
        timeout_seconds=2.0
    )
    
    client = NTPClient(config)
    measurement = client.get_best_measurement()
    
    if measurement:
        print(f"✓ NTP measurement successful:")
        print(f"  Server: {measurement.server}")
        print(f"  Offset: {measurement.offset*1e6:.1f} μs")
        print(f"  Delay: {measurement.delay*1000:.1f} ms")
        print(f"  Stratum: {measurement.stratum}")
        print(f"  Uncertainty: {measurement.uncertainty*1e6:.1f} μs")
    else:
        print("✗ NTP measurement failed")
    
    print("\nNTP client test completed!")