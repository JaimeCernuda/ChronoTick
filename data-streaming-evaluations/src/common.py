"""Common utilities for data streaming evaluation"""

import time
import socket
import json
import logging
import struct
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class Event:
    """Event broadcast from coordinator to workers"""
    event_id: int
    coordinator_timestamp_ns: int  # Nanosecond precision
    sequence_number: int
    payload: Optional[str] = None

    def to_json(self) -> str:
        """Serialize to JSON for network transmission"""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str: str) -> 'Event':
        """Deserialize from JSON"""
        data = json.loads(json_str)
        return cls(**data)


@dataclass
class TimestampRecord:
    """Worker's timestamp record for a received event"""
    event_id: int
    node_id: str

    # Receive time
    receive_time_ns: int  # System time when packet arrived

    # NTP measurements
    ntp_offset_ms: float
    ntp_uncertainty_ms: float
    ntp_timestamp_ns: int  # Derived: receive_time + ntp_offset

    # ChronoTick measurements
    ct_offset_ms: float
    ct_uncertainty_ms: float
    ct_timestamp_ns: int  # Derived: receive_time + ct_offset
    ct_lower_bound_ns: int  # timestamp - 3*uncertainty
    ct_upper_bound_ns: int  # timestamp + 3*uncertainty

    # Commit-wait: uncertainty at future time points
    ct_uncertainty_30s_ms: Optional[float] = None
    ct_uncertainty_60s_ms: Optional[float] = None

    # Ground truth (from coordinator)
    coordinator_send_time_ns: Optional[int] = None

    def to_csv_row(self) -> Dict[str, Any]:
        """Convert to CSV-friendly dictionary"""
        return asdict(self)


class UDPBroadcaster:
    """Helper for UDP broadcast to multiple workers"""

    def __init__(self, timeout: float = 5.0):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(timeout)
        self.logger = logging.getLogger(__name__)

    def send(self, event: Event, workers: list[Tuple[str, int]]) -> Dict[str, bool]:
        """
        Broadcast event to all workers.

        Returns dict mapping worker address to success/failure.
        """
        results = {}
        message = event.to_json().encode('utf-8')

        for host, port in workers:
            try:
                self.sock.sendto(message, (host, port))
                results[f"{host}:{port}"] = True
                self.logger.debug(f"Sent event {event.event_id} to {host}:{port}")
            except Exception as e:
                results[f"{host}:{port}"] = False
                self.logger.error(f"Failed to send to {host}:{port}: {e}")

        return results

    def close(self):
        """Close socket"""
        self.sock.close()


class UDPListener:
    """Helper for UDP listening on worker nodes"""

    def __init__(self, port: int, buffer_size: int = 4096):
        self.port = port
        self.buffer_size = buffer_size
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('0.0.0.0', port))
        self.logger = logging.getLogger(__name__)

    def receive(self) -> Tuple[Event, int]:
        """
        Receive event from network.

        Returns: (event, receive_time_ns)
        """
        data, addr = self.sock.recvfrom(self.buffer_size)
        receive_time_ns = time.time_ns()  # Capture ASAP!

        try:
            event = Event.from_json(data.decode('utf-8'))
            self.logger.debug(f"Received event {event.event_id} from {addr}")
            return event, receive_time_ns
        except Exception as e:
            self.logger.error(f"Failed to parse event from {addr}: {e}")
            raise

    def close(self):
        """Close socket"""
        self.sock.close()


class NTPClient:
    """
    Robust NTP client using manual NTP protocol implementation.

    Queries multiple servers, calculates median offset and uncertainty.
    For ARES deployment, server can be "172.20.1.1:8123" (NTP proxy).
    """

    def __init__(self, ntp_servers: list[str]):
        self.ntp_servers = ntp_servers
        self.logger = logging.getLogger(__name__)
        self._last_offset = None
        self._last_uncertainty = None
        self._last_query_time = 0

        # NTP constants
        self.NTP_PACKET_FORMAT = "!12I"
        self.NTP_EPOCH = 2208988800  # 1970-01-01 00:00:00

    def _create_ntp_packet(self) -> bytes:
        """Create NTP request packet"""
        # NTP v3, client mode
        packet = bytearray(48)
        packet[0] = 0x1B  # LI=0, VN=3, Mode=3 (client)
        return bytes(packet)

    def _parse_ntp_packet(self, data: bytes) -> float:
        """
        Parse NTP response and return offset in seconds.

        NTP offset calculation:
        offset = ((T2 - T1) + (T3 - T4)) / 2

        Where:
        T1 = client send time
        T2 = server receive time
        T3 = server transmit time
        T4 = client receive time
        """
        if len(data) < 48:
            raise ValueError(f"Invalid NTP packet size: {len(data)}")

        # Extract timestamps (convert from NTP epoch to Unix epoch)
        unpacked = struct.unpack(self.NTP_PACKET_FORMAT, data)

        # Transmit timestamp (server send time)
        tx_seconds = unpacked[10] - self.NTP_EPOCH
        tx_fraction = unpacked[11] / 2**32
        t3 = tx_seconds + tx_fraction

        # Receive timestamp at client
        t4 = time.time()

        # For simplicity, we estimate offset as (T3 - T4)
        # This assumes symmetric delay, which is reasonable for LAN
        offset = t3 - t4

        return offset

    def _query_single_server(self, server_spec: str, timeout: float = 2.0) -> Optional[float]:
        """
        Query single NTP server and return offset in seconds.

        Args:
            server_spec: Either "host" or "host:port"
            timeout: Query timeout in seconds

        Returns:
            Offset in seconds, or None if query failed
        """
        try:
            # Parse server specification
            if ':' in server_spec:
                host, port_str = server_spec.rsplit(':', 1)
                port = int(port_str)
            else:
                host = server_spec
                port = 123  # Standard NTP port

            # Create socket and send request
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(timeout)

            packet = self._create_ntp_packet()
            sock.sendto(packet, (host, port))

            # Receive response
            data, addr = sock.recvfrom(1024)
            sock.close()

            # Parse offset
            offset = self._parse_ntp_packet(data)

            self.logger.debug(f"NTP query {host}:{port} → offset={offset*1000:.2f}ms")
            return offset

        except socket.timeout:
            self.logger.warning(f"NTP timeout: {server_spec}")
            return None
        except Exception as e:
            self.logger.warning(f"NTP query failed ({server_spec}): {e}")
            return None

    def query(self) -> Tuple[float, float]:
        """
        Query NTP servers and return (offset_ms, uncertainty_ms).

        Cached for 10 seconds to avoid excessive queries.
        Queries all servers, takes median offset.
        Estimates uncertainty from spread + base uncertainty.
        """
        now = time.time()

        # Return cached value if recent enough
        if now - self._last_query_time < 10 and self._last_offset is not None:
            return self._last_offset, self._last_uncertainty

        # Query all servers
        offsets = []
        for server in self.ntp_servers:
            offset_s = self._query_single_server(server)
            if offset_s is not None:
                offsets.append(offset_s)

        if not offsets:
            # All queries failed - return cached value or fallback
            if self._last_offset is not None:
                self.logger.warning("All NTP queries failed, using cached value")
                return self._last_offset, self._last_uncertainty
            else:
                self.logger.error("All NTP queries failed, no cached value available")
                # Return zero offset with high uncertainty
                return 0.0, 50.0

        # Calculate median offset
        import numpy as np
        offset_s = float(np.median(offsets))
        offset_ms = offset_s * 1000

        # Estimate uncertainty
        # Base uncertainty: ~5ms for NTP over LAN
        # Add standard deviation of measurements
        if len(offsets) > 1:
            std_ms = float(np.std(offsets)) * 1000
            uncertainty_ms = 5.0 + std_ms
        else:
            uncertainty_ms = 5.0

        # Cache results
        self._last_offset = offset_ms
        self._last_uncertainty = uncertainty_ms
        self._last_query_time = now

        self.logger.debug(
            f"NTP query result: offset={offset_ms:+.2f}ms, "
            f"uncertainty={uncertainty_ms:.2f}ms ({len(offsets)} servers)"
        )

        return offset_ms, uncertainty_ms


class ChronoTickClient:
    """
    ChronoTick client (connects to ChronoTick MCP server via HTTP).

    For ARES deployment, connects to proxy at http://172.20.1.1:8124
    """

    def __init__(self, server_url: str):
        self.server_url = server_url.rstrip('/')  # Remove trailing slash
        self.logger = logging.getLogger(__name__)
        self._uncertainty_history = []  # For commit-wait analysis

    def query(self) -> Tuple[float, float]:
        """
        Query ChronoTick server for current offset and uncertainty.

        Returns: (offset_ms, uncertainty_ms)

        Makes HTTP POST to MCP server's tool call endpoint.
        """
        try:
            import requests

            # MCP tool call format
            # Try direct tool endpoint first (if supported)
            endpoint = f"{self.server_url}/get_time"

            self.logger.debug(f"Querying ChronoTick: {endpoint}")

            response = requests.post(
                endpoint,
                json={},
                timeout=5
            )

            response.raise_for_status()
            data = response.json()

            # Extract offset and uncertainty
            # Expected format: {"offset_ms": float, "uncertainty_ms": float, ...}
            offset_ms = float(data.get('offset_ms', 0.0))
            uncertainty_ms = float(data.get('uncertainty_ms', 10.0))

            # Record for commit-wait analysis
            self._uncertainty_history.append({
                'time': time.time(),
                'uncertainty': uncertainty_ms
            })

            self.logger.debug(
                f"ChronoTick result: offset={offset_ms:+.2f}ms, "
                f"uncertainty={uncertainty_ms:.2f}ms"
            )

            return offset_ms, uncertainty_ms

        except ImportError:
            self.logger.error("requests library not available - install with: pip install requests")
            # Return fallback values
            return 0.0, 15.0

        except requests.exceptions.Timeout:
            self.logger.warning(f"ChronoTick query timeout: {self.server_url}")
            # Return fallback values
            return 0.0, 20.0

        except requests.exceptions.RequestException as e:
            self.logger.warning(f"ChronoTick query failed: {e}")
            # Return fallback values
            return 0.0, 15.0

        except (KeyError, ValueError) as e:
            self.logger.error(f"Failed to parse ChronoTick response: {e}")
            # Return fallback values
            return 0.0, 15.0

    def get_historical_uncertainty(self, seconds_ago: int) -> Optional[float]:
        """
        Get uncertainty value from N seconds ago (for commit-wait analysis).

        Returns None if not available.
        """
        target_time = time.time() - seconds_ago

        # Find closest historical record
        closest = None
        min_diff = float('inf')

        for record in self._uncertainty_history:
            diff = abs(record['time'] - target_time)
            if diff < min_diff:
                min_diff = diff
                closest = record

        if closest and min_diff < 5:  # Within 5 seconds
            return closest['uncertainty']
        return None


def setup_logging(log_file: Optional[Path] = None, level: str = "INFO") -> logging.Logger:
    """Setup logging configuration"""
    log_level = getattr(logging, level.upper())

    handlers = [logging.StreamHandler()]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

    return logging.getLogger()


def high_precision_sleep(duration_s: float):
    """High-precision sleep using busy-wait for last millisecond"""
    if duration_s <= 0:
        return

    # Sleep most of the time normally
    if duration_s > 0.001:
        time.sleep(duration_s - 0.001)
        remaining = 0.001
    else:
        remaining = duration_s

    # Busy-wait for the last millisecond for precision
    target = time.time() + remaining
    while time.time() < target:
        pass
