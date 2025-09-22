"""Tests for ChronoTick precision clock manager."""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timedelta

from chronotick_server.manager import PrecisionClockManager, DriftMonitor
from chronotick_server.models import (
    ClockSource,
    ClockQuality,
    PrecisionTimestamp,
    SyncStatus,
    VectorClock,
    TimeConversionRequest
)


class TestPrecisionClockManager:
    """Test PrecisionClockManager."""
    
    @pytest.fixture
    def manager(self):
        """Create a precision clock manager."""
        return PrecisionClockManager(node_id="test-node")
    
    def test_initialization(self, manager):
        """Test manager initialization."""
        assert manager.node_id == "test-node"
        assert manager.registry is not None
        assert manager._vector_clock is not None
        assert manager._vector_clock.node_id == "test-node"
        assert manager._drift_monitor is not None
    
    def test_default_sources_registered(self, manager):
        """Test that default sources are registered."""
        sources = manager.registry.list_sources()
        assert "system" in sources
        assert "ntp" in sources
        assert "chrony" in sources
    
    @pytest.mark.asyncio
    async def test_get_precision_time_default(self, manager):
        """Test getting precision time with default source."""
        # Mock the registry to return a mock source
        mock_source = MagicMock()
        mock_timestamp = PrecisionTimestamp.now("UTC")
        mock_source.get_time = AsyncMock(return_value=mock_timestamp)
        mock_source.is_available = AsyncMock(return_value=True)
        
        with patch.object(manager.registry, 'get_best_source', return_value=mock_source):
            timestamp = await manager.get_precision_time("UTC")
            
            assert timestamp == mock_timestamp
            mock_source.get_time.assert_called_once_with("UTC")
    
    @pytest.mark.asyncio
    async def test_get_precision_time_preferred_source(self, manager):
        """Test getting precision time with preferred source."""
        # Mock NTP source
        mock_ntp_source = MagicMock()
        mock_ntp_source.source_type = ClockSource.NTP
        mock_ntp_source.is_available = AsyncMock(return_value=True)
        mock_timestamp = PrecisionTimestamp.now("UTC")
        mock_ntp_source.get_time = AsyncMock(return_value=mock_timestamp)
        
        manager.registry.sources = {"ntp": mock_ntp_source}
        
        timestamp = await manager.get_precision_time("UTC", ClockSource.NTP)
        
        assert timestamp == mock_timestamp
        mock_ntp_source.get_time.assert_called_once_with("UTC")
    
    @pytest.mark.asyncio
    async def test_get_precision_time_fallback(self, manager):
        """Test fallback to system clock when preferred source fails."""
        # Mock NTP source that fails
        mock_ntp_source = MagicMock()
        mock_ntp_source.source_type = ClockSource.NTP
        mock_ntp_source.is_available = AsyncMock(return_value=True)
        mock_ntp_source.get_time = AsyncMock(side_effect=Exception("NTP failed"))
        
        # Mock system source that works
        mock_system_source = MagicMock()
        mock_system_source.name = "system"
        mock_timestamp = PrecisionTimestamp.now("UTC")
        mock_system_source.get_time = AsyncMock(return_value=mock_timestamp)
        
        manager.registry.sources = {"ntp": mock_ntp_source, "system": mock_system_source}
        
        with patch.object(manager.registry, 'get_source', return_value=mock_system_source):
            timestamp = await manager.get_precision_time("UTC", ClockSource.NTP)
            
            assert timestamp == mock_timestamp
    
    @pytest.mark.asyncio
    async def test_get_precision_time_no_sources(self, manager):
        """Test error when no sources are available."""
        with patch.object(manager.registry, 'get_best_source', return_value=None):
            with patch.object(manager.registry, 'get_source', return_value=None):
                with pytest.raises(RuntimeError, match="No time sources available"):
                    await manager.get_precision_time("UTC")
    
    @pytest.mark.asyncio
    async def test_get_sync_status(self, manager):
        """Test getting sync status for all sources."""
        # Mock sources
        mock_system_status = SyncStatus(
            is_synchronized=True,
            source=ClockSource.SYSTEM,
            quality=ClockQuality.FAIR
        )
        
        mock_ntp_status = SyncStatus(
            is_synchronized=True,
            source=ClockSource.NTP,
            quality=ClockQuality.GOOD
        )
        
        mock_system_source = MagicMock()
        mock_system_source.get_sync_status = AsyncMock(return_value=mock_system_status)
        
        mock_ntp_source = MagicMock()
        mock_ntp_source.get_sync_status = AsyncMock(return_value=mock_ntp_status)
        
        manager.registry.sources = {
            "system": mock_system_source,
            "ntp": mock_ntp_source
        }
        
        status = await manager.get_sync_status()
        
        assert "system" in status
        assert "ntp" in status
        assert status["system"] == mock_system_status
        assert status["ntp"] == mock_ntp_status
    
    @pytest.mark.asyncio
    async def test_get_overall_sync_status(self, manager):
        """Test getting overall sync status."""
        # Mock sync statuses with different qualities
        mock_statuses = {
            "system": SyncStatus(
                is_synchronized=True,
                source=ClockSource.SYSTEM,
                quality=ClockQuality.FAIR
            ),
            "ntp": SyncStatus(
                is_synchronized=True,
                source=ClockSource.NTP,
                quality=ClockQuality.EXCELLENT
            )
        }
        
        with patch.object(manager, 'get_sync_status', return_value=mock_statuses):
            overall = await manager.get_overall_sync_status()
            
            assert overall.is_synchronized
            assert overall.source == ClockSource.NTP  # Best quality
            assert overall.quality == ClockQuality.EXCELLENT
            assert overall.reachable_sources == 2
            assert overall.total_sources == 2
    
    @pytest.mark.asyncio
    async def test_convert_time(self, manager):
        """Test time conversion."""
        # Mock current time
        mock_timestamp = PrecisionTimestamp(
            utc_seconds=1703980800,  # 2023-12-30 16:00:00 UTC
            utc_nanoseconds=123456789,
            timezone="UTC",
            is_dst=False,
            monotonic_ns=123456789,
            source=ClockSource.SYSTEM,
            quality=ClockQuality.FAIR
        )
        
        with patch.object(manager, 'get_precision_time', return_value=mock_timestamp):
            request = TimeConversionRequest(
                source_timezone="UTC",
                target_timezone="America/New_York"
            )
            
            result = await manager.convert_time(request)
            
            assert result.source.timezone == "UTC"
            assert result.target.timezone == "America/New_York"
            assert result.offset_seconds != 0  # Should have timezone offset
    
    def test_create_vector_clock(self, manager):
        """Test creating vector clock."""
        vc = manager.create_vector_clock()
        
        assert vc.node_id == "test-node"
        assert vc.clocks["test-node"] == 1  # Should increment from initial 0
    
    def test_update_vector_clock(self, manager):
        """Test updating vector clock."""
        # Create initial vector clock
        manager.create_vector_clock()
        
        # Create another vector clock to merge
        other_vc = VectorClock(
            node_id="other-node",
            clocks={"test-node": 0, "other-node": 5},
            timestamp=PrecisionTimestamp.now()
        )
        
        updated = manager.update_vector_clock(other_vc)
        
        assert updated.clocks["test-node"] == 2  # max(1, 0) + 1
        assert updated.clocks["other-node"] == 5  # max(0, 5)
    
    def test_compare_timestamps(self, manager):
        """Test timestamp comparison."""
        ts1 = PrecisionTimestamp(
            utc_seconds=1703980800,
            utc_nanoseconds=0,
            timezone="UTC",
            is_dst=False,
            monotonic_ns=1000,
            source=ClockSource.SYSTEM,
            quality=ClockQuality.FAIR
        )
        
        ts2 = PrecisionTimestamp(
            utc_seconds=1703980801,  # 1 second later
            utc_nanoseconds=0,
            timezone="UTC",
            is_dst=False,
            monotonic_ns=2000,
            source=ClockSource.SYSTEM,
            quality=ClockQuality.FAIR
        )
        
        result = manager.compare_timestamps(ts1, ts2)
        
        assert result["time_difference_ns"] == 1_000_000_000  # 1 second
        assert result["monotonic_difference_ns"] == 1000
        assert result["ts1_before_ts2"]
        assert result["physical_ordering"] == "ts1_before_ts2"
    
    @pytest.mark.asyncio
    async def test_measure_clock_drift(self, manager):
        """Test measuring clock drift."""
        mock_drift_result = {
            "system": {
                "drift_rate_ppm": 0.5,
                "time_elapsed_seconds": 60.0,
                "absolute_drift_ns": 500000,
                "quality": "good"
            }
        }
        
        with patch.object(manager._drift_monitor, 'measure_drift', return_value=mock_drift_result):
            result = await manager.measure_clock_drift(60)
            
            assert result == mock_drift_result


class TestDriftMonitor:
    """Test DriftMonitor."""
    
    @pytest.fixture
    def drift_monitor(self):
        """Create a drift monitor."""
        return DriftMonitor(max_samples=10)
    
    @pytest.mark.asyncio
    async def test_record_sample(self, drift_monitor):
        """Test recording a time sample."""
        timestamp = PrecisionTimestamp.now()
        
        await drift_monitor.record_sample("test_source", timestamp)
        
        assert "test_source" in drift_monitor.samples
        assert len(drift_monitor.samples["test_source"]) == 1
        
        sample = drift_monitor.samples["test_source"][0]
        assert sample["timestamp"] == timestamp
        assert "system_time_ns" in sample
        assert "monotonic_ns" in sample
        assert "recorded_at" in sample
    
    @pytest.mark.asyncio
    async def test_max_samples_limit(self, drift_monitor):
        """Test that sample count is limited."""
        timestamp = PrecisionTimestamp.now()
        
        # Record more samples than max_samples
        for i in range(15):
            await drift_monitor.record_sample("test_source", timestamp)
        
        # Should only keep max_samples (10)
        assert len(drift_monitor.samples["test_source"]) == 10
    
    @pytest.mark.asyncio
    async def test_measure_drift(self, drift_monitor):
        """Test measuring drift over time."""
        # Record initial sample
        timestamp1 = PrecisionTimestamp.now()
        await drift_monitor.record_sample("test_source", timestamp1)
        
        # Mock the drift measurement by modifying the internal state
        initial_sample = drift_monitor.samples["test_source"][0]
        
        # Mock asyncio.sleep to avoid actual delay
        with patch('asyncio.sleep') as mock_sleep:
            # Create a function that will record the second sample when sleep is called
            async def record_second_sample(*args):
                # Simulate final sample
                timestamp2 = PrecisionTimestamp(
                    utc_seconds=timestamp1.utc_seconds + 60,
                    utc_nanoseconds=timestamp1.utc_nanoseconds,
                    timezone=timestamp1.timezone,
                    is_dst=timestamp1.is_dst,
                    monotonic_ns=timestamp1.monotonic_ns + 60_000_000_000,  # 60 seconds
                    source=timestamp1.source,
                    quality=timestamp1.quality
                )
                
                # Record final sample
                await drift_monitor.record_sample("test_source", timestamp2)
                
                # Update the recorded_at time to simulate time passage
                final_sample = drift_monitor.samples["test_source"][-1]
                final_sample["recorded_at"] = initial_sample["recorded_at"] + timedelta(seconds=60)
            
            mock_sleep.side_effect = record_second_sample
            
            result = await drift_monitor.measure_drift(1)  # 1 second duration
            
            # Should have drift result for test_source
            assert "test_source" in result
            drift_info = result["test_source"]
            assert "drift_rate_ppm" in drift_info
            assert "time_elapsed_seconds" in drift_info
            assert "absolute_drift_ns" in drift_info
            assert "quality" in drift_info
    
    @pytest.mark.asyncio
    async def test_measure_drift_no_samples(self, drift_monitor):
        """Test measuring drift with no samples."""
        with patch('asyncio.sleep'):
            result = await drift_monitor.measure_drift(1)
            
            # Should return empty result
            assert result == {}