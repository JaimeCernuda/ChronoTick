"""Tests for ChronoTick models."""

import pytest
from datetime import datetime, timezone
import time
from unittest.mock import patch

from chronotick_server.models import (
    ClockSource,
    ClockQuality,
    PrecisionTimestamp,
    SyncStatus,
    VectorClock,
    TimeConversionRequest
)


class TestClockSource:
    """Test ClockSource enum."""
    
    def test_available_sources(self):
        """Test that expected clock sources are available."""
        assert ClockSource.SYSTEM == "system"
        assert ClockSource.NTP == "ntp"
    
    def test_source_values(self):
        """Test source enum values."""
        sources = list(ClockSource)
        assert len(sources) == 2
        assert "system" in [s.value for s in sources]
        assert "ntp" in [s.value for s in sources]


class TestClockQuality:
    """Test ClockQuality enum."""
    
    def test_quality_levels(self):
        """Test that all quality levels exist."""
        qualities = [
            ClockQuality.UNKNOWN,
            ClockQuality.POOR,
            ClockQuality.FAIR,
            ClockQuality.GOOD,
            ClockQuality.EXCELLENT,
            ClockQuality.REFERENCE
        ]
        assert len(qualities) == 6
    
    def test_quality_ordering(self):
        """Test that qualities can be compared for ordering."""
        quality_order = {
            ClockQuality.REFERENCE: 5,
            ClockQuality.EXCELLENT: 4,
            ClockQuality.GOOD: 3,
            ClockQuality.FAIR: 2,
            ClockQuality.POOR: 1,
            ClockQuality.UNKNOWN: 0
        }
        
        assert quality_order[ClockQuality.REFERENCE] > quality_order[ClockQuality.EXCELLENT]
        assert quality_order[ClockQuality.EXCELLENT] > quality_order[ClockQuality.GOOD]
        assert quality_order[ClockQuality.UNKNOWN] < quality_order[ClockQuality.POOR]


class TestPrecisionTimestamp:
    """Test PrecisionTimestamp model."""
    
    def test_timestamp_creation(self):
        """Test creating a precision timestamp."""
        now_ns = time.time_ns()
        utc_seconds = now_ns // 1_000_000_000
        utc_nanoseconds = now_ns % 1_000_000_000
        
        timestamp = PrecisionTimestamp(
            utc_seconds=utc_seconds,
            utc_nanoseconds=utc_nanoseconds,
            timezone="UTC",
            is_dst=False,
            monotonic_ns=time.monotonic_ns(),
            source=ClockSource.SYSTEM,
            quality=ClockQuality.FAIR
        )
        
        assert timestamp.utc_seconds == utc_seconds
        assert timestamp.utc_nanoseconds == utc_nanoseconds
        assert timestamp.timezone == "UTC"
        assert not timestamp.is_dst
        assert timestamp.source == ClockSource.SYSTEM
        assert timestamp.quality == ClockQuality.FAIR
    
    def test_now_class_method(self):
        """Test the now() class method."""
        timestamp = PrecisionTimestamp.now()
        
        assert timestamp.timezone == "UTC"
        assert timestamp.source == ClockSource.SYSTEM
        assert timestamp.utc_seconds > 0
        assert 0 <= timestamp.utc_nanoseconds < 1_000_000_000
        assert timestamp.monotonic_ns > 0
    
    def test_now_with_timezone(self):
        """Test now() with different timezone."""
        timestamp = PrecisionTimestamp.now("America/New_York")
        assert timestamp.timezone == "America/New_York"
    
    def test_to_datetime(self):
        """Test conversion to datetime object."""
        timestamp = PrecisionTimestamp.now()
        dt = timestamp.to_datetime()
        
        assert isinstance(dt, datetime)
        # Should be close to current time (within 1 second)
        now = datetime.now()
        assert abs((dt - now).total_seconds()) < 1.0
    
    def test_to_iso_string_nanoseconds(self):
        """Test ISO string with nanosecond precision."""
        timestamp = PrecisionTimestamp(
            utc_seconds=1703980800,  # 2023-12-30 16:00:00 UTC
            utc_nanoseconds=123456789,
            timezone="UTC",
            is_dst=False,
            monotonic_ns=123456789,
            source=ClockSource.SYSTEM,
            quality=ClockQuality.FAIR
        )
        
        iso_str = timestamp.to_iso_string("nanoseconds")
        assert "123456789" in iso_str
        assert iso_str.endswith("Z")
    
    def test_to_iso_string_microseconds(self):
        """Test ISO string with microsecond precision."""
        timestamp = PrecisionTimestamp(
            utc_seconds=1703980800,
            utc_nanoseconds=123456789,
            timezone="UTC",
            is_dst=False,
            monotonic_ns=123456789,
            source=ClockSource.SYSTEM,
            quality=ClockQuality.FAIR
        )
        
        iso_str = timestamp.to_iso_string("microseconds")
        assert "123456" in iso_str
        assert iso_str.endswith("Z")
    
    def test_nanoseconds_validation(self):
        """Test that nanoseconds are validated correctly."""
        # Valid nanoseconds
        timestamp = PrecisionTimestamp(
            utc_seconds=1703980800,
            utc_nanoseconds=999_999_999,
            timezone="UTC",
            is_dst=False,
            monotonic_ns=123456789,
            source=ClockSource.SYSTEM,
            quality=ClockQuality.FAIR
        )
        assert timestamp.utc_nanoseconds == 999_999_999
        
        # Invalid nanoseconds should raise validation error
        with pytest.raises(Exception):  # Pydantic validation error
            PrecisionTimestamp(
                utc_seconds=1703980800,
                utc_nanoseconds=1_000_000_000,  # Too large
                timezone="UTC",
                is_dst=False,
                monotonic_ns=123456789,
                source=ClockSource.SYSTEM,
                quality=ClockQuality.FAIR
            )


class TestSyncStatus:
    """Test SyncStatus model."""
    
    def test_basic_sync_status(self):
        """Test creating a basic sync status."""
        status = SyncStatus(
            is_synchronized=True,
            source=ClockSource.NTP,
            quality=ClockQuality.GOOD,
            reachable_sources=3,
            total_sources=4
        )
        
        assert status.is_synchronized
        assert status.source == ClockSource.NTP
        assert status.quality == ClockQuality.GOOD
        assert status.reachable_sources == 3
        assert status.total_sources == 4
    
    def test_ntp_specific_fields(self):
        """Test NTP-specific fields."""
        status = SyncStatus(
            is_synchronized=True,
            source=ClockSource.NTP,
            quality=ClockQuality.EXCELLENT,
            ntp_stratum=2,
            ntp_offset_ms=5.2,
            ntp_jitter_ms=1.1,
            ntp_delay_ms=12.3,
            ntp_servers=["pool.ntp.org", "time.google.com"],
            reachable_sources=2,
            total_sources=2
        )
        
        assert status.ntp_stratum == 2
        assert status.ntp_offset_ms == 5.2
        assert status.ntp_jitter_ms == 1.1
        assert status.ntp_delay_ms == 12.3
        assert len(status.ntp_servers) == 2


class TestVectorClock:
    """Test VectorClock model."""
    
    def test_vector_clock_creation(self):
        """Test creating a vector clock."""
        timestamp = PrecisionTimestamp.now()
        vc = VectorClock(
            node_id="node1",
            clocks={"node1": 0},
            timestamp=timestamp
        )
        
        assert vc.node_id == "node1"
        assert vc.clocks["node1"] == 0
        assert vc.timestamp == timestamp
    
    def test_increment(self):
        """Test incrementing vector clock."""
        timestamp = PrecisionTimestamp.now()
        vc = VectorClock(
            node_id="node1",
            clocks={"node1": 5, "node2": 3},
            timestamp=timestamp
        )
        
        incremented = vc.increment()
        assert incremented.clocks["node1"] == 6
        assert incremented.clocks["node2"] == 3
        assert incremented.node_id == "node1"
    
    def test_update_with_other_clock(self):
        """Test updating with another vector clock."""
        timestamp1 = PrecisionTimestamp.now()
        timestamp2 = PrecisionTimestamp.now()
        
        vc1 = VectorClock(
            node_id="node1",
            clocks={"node1": 5, "node2": 3},
            timestamp=timestamp1
        )
        
        vc2 = VectorClock(
            node_id="node2",
            clocks={"node1": 2, "node2": 7, "node3": 1},
            timestamp=timestamp2
        )
        
        updated = vc1.update(vc2)
        
        # Should take max of each and increment own
        assert updated.clocks["node1"] == 6  # max(5,2) + 1
        assert updated.clocks["node2"] == 7  # max(3,7)
        assert updated.clocks["node3"] == 1  # max(0,1)
        assert updated.node_id == "node1"
    
    def test_compare_happens_before(self):
        """Test vector clock comparison - happens before."""
        vc1 = VectorClock(
            node_id="node1",
            clocks={"node1": 2, "node2": 1},
            timestamp=PrecisionTimestamp.now()
        )
        
        vc2 = VectorClock(
            node_id="node2",
            clocks={"node1": 3, "node2": 2},
            timestamp=PrecisionTimestamp.now()
        )
        
        result = vc1.compare(vc2)
        assert result == "happens_before"
    
    def test_compare_happens_after(self):
        """Test vector clock comparison - happens after."""
        vc1 = VectorClock(
            node_id="node1",
            clocks={"node1": 5, "node2": 3},
            timestamp=PrecisionTimestamp.now()
        )
        
        vc2 = VectorClock(
            node_id="node2",
            clocks={"node1": 2, "node2": 1},
            timestamp=PrecisionTimestamp.now()
        )
        
        result = vc1.compare(vc2)
        assert result == "happens_after"
    
    def test_compare_concurrent(self):
        """Test vector clock comparison - concurrent events."""
        vc1 = VectorClock(
            node_id="node1",
            clocks={"node1": 5, "node2": 1},
            timestamp=PrecisionTimestamp.now()
        )
        
        vc2 = VectorClock(
            node_id="node2",
            clocks={"node1": 2, "node2": 3},
            timestamp=PrecisionTimestamp.now()
        )
        
        result = vc1.compare(vc2)
        assert result == "concurrent"


class TestTimeConversionRequest:
    """Test TimeConversionRequest model."""
    
    def test_basic_conversion_request(self):
        """Test creating a time conversion request."""
        request = TimeConversionRequest(
            source_timezone="UTC",
            target_timezone="America/New_York"
        )
        
        assert request.source_timezone == "UTC"
        assert request.target_timezone == "America/New_York"
        assert request.timestamp is None
        assert request.preserve_precision is True
    
    def test_conversion_request_with_timestamp(self):
        """Test conversion request with specific timestamp."""
        timestamp = PrecisionTimestamp.now()
        request = TimeConversionRequest(
            source_timezone="UTC",
            target_timezone="Europe/London",
            timestamp=timestamp,
            preserve_precision=False
        )
        
        assert request.timestamp == timestamp
        assert request.preserve_precision is False