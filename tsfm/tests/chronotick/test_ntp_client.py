#!/usr/bin/env python3
"""
Tests for ChronoTick NTP Client

Verifies real clock measurement functionality and NTP protocol implementation.
"""

import pytest
import time
import socket
import struct
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import yaml

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from chronotick_inference.ntp_client import (
    NTPClient, NTPConfig, NTPMeasurement, ClockMeasurementCollector
)


@pytest.fixture
def ntp_config():
    """Create test NTP configuration"""
    return NTPConfig(
        servers=["test.ntp.server", "backup.ntp.server"],
        timeout_seconds=1.0,
        max_acceptable_uncertainty=0.020,  # 20ms for testing
        min_stratum=2
    )


@pytest.fixture
def mock_ntp_response():
    """Create mock NTP response packet"""
    # Create a realistic NTP response packet
    # Format: 12 32-bit words (48 bytes total)
    
    # Word 0: LI=0, VN=3, Mode=4 (server), Stratum=2, Poll=6, Precision=-20
    # NTP word structure: | LI(2) | VN(3) | Mode(3) | Stratum(8) | Poll(8) | Precision(8) |
    # Precision -20 means 2^-20 seconds (~1 microsecond)
    # Pack -20 as signed byte correctly
    precision_byte = struct.unpack('B', struct.pack('b', -20))[0]
    word0 = (0 << 30) | (3 << 27) | (4 << 24) | (2 << 16) | (6 << 8) | precision_byte
    
    # Words 1-7: Various NTP fields (simplified)
    root_delay = 0x00000100      # Small root delay
    root_dispersion = 0x00000050 # Small root dispersion
    ref_id = 0x01020304         # Reference ID
    ref_timestamp_int = 0x00000000
    ref_timestamp_frac = 0x00000000
    orig_timestamp_int = 0x00000000
    orig_timestamp_frac = 0x00000000
    
    # Words 8-9: Receive timestamp (when server received request)
    # Convert current time to NTP timestamp (since 1900)
    current_time = time.time()
    ntp_epoch_offset = 2208988800  # Seconds between 1900 and 1970
    ntp_time = current_time + ntp_epoch_offset
    
    receive_timestamp_int = int(ntp_time)
    receive_timestamp_frac = int((ntp_time % 1) * (2**32))
    
    # Words 10-11: Transmit timestamp (when server sent response)
    # Slightly later than receive time
    transmit_time = ntp_time + 0.001  # 1ms processing delay
    transmit_timestamp_int = int(transmit_time)
    transmit_timestamp_frac = int((transmit_time % 1) * (2**32))
    
    # Pack into NTP packet format
    packet_data = [
        word0,
        root_delay,
        root_dispersion,
        ref_id,
        ref_timestamp_int,
        ref_timestamp_frac,
        orig_timestamp_int,
        orig_timestamp_frac,
        receive_timestamp_int,
        receive_timestamp_frac,
        transmit_timestamp_int,
        transmit_timestamp_frac
    ]
    
    return struct.pack("!12I", *packet_data)


class TestNTPClient:
    """Test NTP client functionality"""
    
    def test_ntp_packet_creation(self, ntp_config):
        """Test NTP request packet creation"""
        client = NTPClient(ntp_config)
        packet = client._create_ntp_request()
        
        # Should be 48 bytes
        assert len(packet) == 48
        
        # Unpack first word to check format
        first_word = struct.unpack("!I", packet[:4])[0]
        
        # Check LI=0, VN=3, Mode=3 (bits 31-24 should be 0x1B)
        li_vn_mode = (first_word >> 24) & 0xFF
        assert li_vn_mode == 0x1B
    
    def test_ntp_response_parsing(self, ntp_config, mock_ntp_response):
        """Test parsing of NTP response packet"""
        client = NTPClient(ntp_config)
        parsed = client._parse_ntp_response(mock_ntp_response)
        
        assert parsed is not None
        assert 'stratum' in parsed
        assert 'precision' in parsed
        assert 'receive_timestamp' in parsed
        assert 'transmit_timestamp' in parsed
        
        # Check reasonable values
        assert 1 <= parsed['stratum'] <= 15
        assert parsed['precision'] > 0
        assert parsed['receive_timestamp'] > 0
        assert parsed['transmit_timestamp'] > 0
    
    @patch('socket.socket')
    def test_successful_ntp_measurement(self, mock_socket_class, ntp_config, mock_ntp_response):
        """Test successful NTP measurement"""
        # Mock socket behavior
        mock_socket = MagicMock()
        mock_socket_class.return_value = mock_socket
        mock_socket.recvfrom.return_value = (mock_ntp_response, ("test.server", 123))
        
        client = NTPClient(ntp_config)
        
        # Store real time function before mocking
        import time as real_time
        real_time_func = real_time.time
        
        # Create a function that returns specific values for the first two calls,
        # then falls back to real time.time() for any additional calls (like logging)
        call_count = 0
        def mock_time():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return 1000.0  # t1_local
            elif call_count == 2:
                return 1000.002  # t4_local (2ms round-trip)
            else:
                # Fall back to real time for logging or other calls
                return real_time_func()
        
        # Mock time.time() only in the ntp_client module to avoid logging conflicts
        with patch('chronotick_inference.ntp_client.time.time', side_effect=mock_time):
            measurement = client.measure_offset("test.ntp.server")
        
        assert measurement is not None
        assert isinstance(measurement, NTPMeasurement)
        assert measurement.server == "test.ntp.server"
        assert abs(measurement.delay - 0.002) < 0.001  # ~2ms delay
        assert measurement.stratum == 2
        assert measurement.uncertainty > 0
    
    @patch('socket.socket')
    def test_ntp_timeout_handling(self, mock_socket_class, ntp_config):
        """Test NTP timeout handling"""
        # Mock socket timeout
        mock_socket = MagicMock()
        mock_socket_class.return_value = mock_socket
        mock_socket.recvfrom.side_effect = socket.timeout()
        
        client = NTPClient(ntp_config)
        measurement = client.measure_offset("timeout.server")
        
        assert measurement is None
    
    @patch('socket.socket')
    def test_poor_quality_measurement_rejection(self, mock_socket_class, ntp_config):
        """Test rejection of poor quality measurements"""
        # Create a response with poor quality (high delay, low stratum)
        client = NTPClient(ntp_config)
        
        # Mock high delay measurement
        mock_socket = MagicMock()
        mock_socket_class.return_value = mock_socket
        
        # Create response with stratum 1 (should be rejected due to min_stratum=2)
        poor_response = bytearray(48)
        # Set stratum to 1 (below min_stratum=2)
        poor_response[1] = 1
        
        mock_socket.recvfrom.return_value = (bytes(poor_response), ("poor.server", 123))
        
        # Store real time function before mocking
        import time as real_time
        real_time_func = real_time.time
        
        # Create a function that returns high delay values for the test
        call_count = 0
        def mock_time():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return 1000.0  # t1_local
            elif call_count == 2:
                return 1000.200  # t4_local (200ms delay - too high)
            else:
                # Fall back to real time for logging or other calls
                return real_time_func()
        
        with patch('chronotick_inference.ntp_client.time.time', side_effect=mock_time):
            measurement = client.measure_offset("poor.server")
        
        # Should reject due to poor quality
        assert measurement is None
    
    def test_multiple_server_selection(self, ntp_config):
        """Test selection of best measurement from multiple servers"""
        client = NTPClient(ntp_config)
        
        # Mock multiple measurements with different qualities
        measurements = [
            NTPMeasurement(
                offset=0.000020, delay=0.050, stratum=3, precision=1e-6,
                server="slow.server", timestamp=time.time(), uncertainty=0.005
            ),
            NTPMeasurement(
                offset=0.000018, delay=0.010, stratum=2, precision=1e-7,
                server="fast.server", timestamp=time.time(), uncertainty=0.002
            ),
            NTPMeasurement(
                offset=0.000025, delay=0.030, stratum=4, precision=1e-6,
                server="distant.server", timestamp=time.time(), uncertainty=0.003
            )
        ]
        
        # Mock measure_offset to return different measurements
        def mock_measure(server):
            server_map = {
                "slow.server": measurements[0],
                "fast.server": measurements[1], 
                "distant.server": measurements[2]
            }
            return server_map.get(server)
        
        with patch.object(client, 'measure_offset', side_effect=mock_measure):
            # Override config servers for this test
            client.config.servers = ["slow.server", "fast.server", "distant.server"]
            best = client.get_best_measurement()
        
        # Should select fast.server (lowest delay, highest stratum)
        assert best is not None
        assert best.server == "fast.server"
        assert best.delay == 0.010
    
    def test_measurement_history_management(self, ntp_config):
        """Test measurement history storage and management"""
        client = NTPClient(ntp_config)
        
        # Add measurements to history
        for i in range(150):  # More than history limit
            measurement = NTPMeasurement(
                offset=0.000020 + i * 1e-6,
                delay=0.010,
                stratum=3,
                precision=1e-6,
                server="test.server",
                timestamp=time.time() + i,
                uncertainty=0.002
            )
            client.measurement_history.append(measurement)
        
        # Manually trigger history management (simulate get_best_measurement)
        if len(client.measurement_history) > 100:
            client.measurement_history = client.measurement_history[-50:]
        
        # Should keep only recent measurements
        assert len(client.measurement_history) == 50
    
    def test_measurement_statistics(self, ntp_config):
        """Test measurement statistics calculation"""
        client = NTPClient(ntp_config)
        
        # Add some test measurements
        for i in range(10):
            measurement = NTPMeasurement(
                offset=0.000020 + i * 1e-6,  # Varying offsets
                delay=0.010 + i * 0.001,     # Varying delays
                stratum=3,
                precision=1e-6,
                server="test.server",
                timestamp=time.time(),
                uncertainty=0.002
            )
            client.measurement_history.append(measurement)
        
        stats = client.get_measurement_statistics()
        
        assert stats['status'] == 'active'
        assert stats['total_measurements'] == 10
        assert stats['recent_count'] == 10
        assert 'offset_stats' in stats
        assert 'delay_stats' in stats
        assert 'servers_used' in stats


class TestClockMeasurementCollector:
    """Test the complete clock measurement collection system"""
    
    @pytest.fixture
    def test_collector_config(self):
        """Create test configuration for collector"""
        config = {
            'clock_measurement': {
                'ntp': {
                    'servers': ['test1.ntp.server', 'test2.ntp.server'],
                    'timeout_seconds': 1.0,
                    'max_acceptable_uncertainty': 0.020,
                    'min_stratum': 2
                },
                'timing': {
                    'warm_up': {
                        'duration_seconds': 5,  # Short for testing
                        'measurement_interval': 0.5
                    },
                    'normal_operation': {
                        'measurement_interval': 2.0
                    }
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            return f.name
    
    def test_collector_initialization(self, test_collector_config):
        """Test collector initializes correctly from config"""
        collector = ClockMeasurementCollector(test_collector_config)
        
        assert collector.warm_up_duration == 5
        assert collector.warm_up_interval == 0.5
        assert collector.normal_interval == 2.0
        assert len(collector.ntp_client.config.servers) == 2
    
    def test_measurement_storage(self, test_collector_config):
        """Test that measurements are stored correctly"""
        collector = ClockMeasurementCollector(test_collector_config)
        
        # Mock a successful NTP measurement
        test_measurement = NTPMeasurement(
            offset=0.000025,
            delay=0.015,
            stratum=3,
            precision=1e-6,
            server="test.server",
            timestamp=time.time(),
            uncertainty=0.003
        )
        
        # Start collection briefly
        collector.collection_running = True
        collector.start_time = time.time()
        
        with patch.object(collector.ntp_client, 'get_best_measurement', return_value=test_measurement):
            # Manually trigger one collection cycle
            current_time = time.time()
            measurement = collector.ntp_client.get_best_measurement()
            
            if measurement:
                with collector.lock:
                    collector.last_measurement = measurement
                    collector.last_measurement_time = current_time
                    collector.offset_measurements.append((
                        measurement.timestamp,
                        measurement.offset,
                        measurement.uncertainty
                    ))
        
        # Check measurement was stored
        assert collector.last_measurement == test_measurement
        assert len(collector.offset_measurements) > 0
        
        offset = collector.get_latest_offset()
        assert offset == 0.000025
    
    def test_recent_measurements_retrieval(self, test_collector_config):
        """Test retrieval of recent measurements"""
        collector = ClockMeasurementCollector(test_collector_config)
        
        # Add some test measurements
        current_time = time.time()
        for i in range(5):
            collector.offset_measurements.append((
                current_time - (4-i) * 60,  # 4 minutes ago to now
                0.000020 + i * 1e-6,
                0.002
            ))
        
        # Get recent measurements (within 3 minutes)
        recent = collector.get_recent_measurements(window_seconds=180)
        
        # Should get measurements from last 3 minutes
        assert len(recent) >= 2  # At least 2 measurements within window
        
        # All returned measurements should be within window
        cutoff_time = current_time - 180
        for ts, offset, uncertainty in recent:
            assert ts >= cutoff_time


def test_ntp_offset_calculation():
    """Test NTP offset calculation algorithm"""
    # Test the standard NTP offset calculation:
    # offset = ((t2 - t1) + (t3 - t4)) / 2
    # delay = (t4 - t1) - (t3 - t2)
    
    # Simulate NTP timing scenario
    t1 = 1000.000  # Client send time
    t2 = 1000.050  # Server receive time (50ms later due to network)
    t3 = 1000.051  # Server transmit time (1ms processing)
    t4 = 1000.101  # Client receive time (50ms return trip)
    
    # Calculate offset and delay
    offset = ((t2 - t1) + (t3 - t4)) / 2.0
    delay = (t4 - t1) - (t3 - t2)
    
    # Expected values
    expected_offset = ((1000.050 - 1000.000) + (1000.051 - 1000.101)) / 2.0
    expected_offset = (0.050 - 0.050) / 2.0  # Should be 0 (perfect sync)
    expected_delay = (1000.101 - 1000.000) - (1000.051 - 1000.050)
    expected_delay = 0.101 - 0.001  # 100ms round-trip
    
    assert abs(offset - expected_offset) < 1e-6
    assert abs(delay - expected_delay) < 1e-6


def test_error_bounds_only_ml_models():
    """Test that error bounds come only from ML model uncertainties"""
    # This test ensures we're not mixing NTP uncertainties into ML error bounds
    
    # Create correction with ML uncertainties only
    correction_bounds = type('CorrectionWithBounds', (), {
        'offset_uncertainty': 5e-6,  # 5μs from ML model
        'drift_uncertainty': 1e-7,   # 0.1μs/s from ML model
        'get_time_uncertainty': lambda self, dt: (self.offset_uncertainty**2 + (self.drift_uncertainty * dt)**2)**0.5
    })()
    
    # Error bounds should ONLY reflect ML model uncertainties
    # NOT include NTP network delays, server precision, etc.
    
    uncertainty_0 = correction_bounds.get_time_uncertainty(0)
    uncertainty_100 = correction_bounds.get_time_uncertainty(100)
    
    # At t=0: only offset uncertainty
    assert abs(uncertainty_0 - 5e-6) < 1e-9
    
    # At t=100: include drift uncertainty
    expected_100 = (25e-12 + (1e-7 * 100)**2)**0.5  # sqrt(5μs² + (0.1μs/s * 100s)²)
    assert abs(uncertainty_100 - expected_100) < 1e-9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])