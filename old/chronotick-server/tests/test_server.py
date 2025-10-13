"""Tests for ChronoTick MCP server."""

import pytest
import json
from unittest.mock import MagicMock, AsyncMock, patch

from chronotick_server.server import ChronoTickServer, ChronoTickTools
from chronotick_server.models import (
    ClockSource,
    ClockQuality,
    PrecisionTimestamp,
    SyncStatus,
    VectorClock
)


class TestChronoTickServer:
    """Test ChronoTickServer."""
    
    @pytest.fixture
    def server(self):
        """Create a ChronoTick server."""
        return ChronoTickServer(node_id="test-server")
    
    def test_initialization(self, server):
        """Test server initialization."""
        assert server.clock_manager is not None
        assert server.clock_manager.node_id == "test-server"
    
    @pytest.mark.asyncio
    async def test_get_current_time(self, server):
        """Test get_current_time method."""
        mock_timestamp = PrecisionTimestamp.now("UTC")
        
        with patch.object(server.clock_manager, 'get_precision_time', return_value=mock_timestamp):
            result = await server.get_current_time("UTC")
            
            assert result == mock_timestamp
            server.clock_manager.get_precision_time.assert_called_once_with("UTC", None)
    
    @pytest.mark.asyncio
    async def test_get_current_time_with_source(self, server):
        """Test get_current_time with specific source."""
        mock_timestamp = PrecisionTimestamp.now("UTC")
        
        with patch.object(server.clock_manager, 'get_precision_time', return_value=mock_timestamp):
            result = await server.get_current_time("UTC", "ntp")
            
            assert result == mock_timestamp
            server.clock_manager.get_precision_time.assert_called_once_with("UTC", ClockSource.NTP)
    
    @pytest.mark.asyncio
    async def test_get_current_time_invalid_source(self, server):
        """Test get_current_time with invalid source."""
        with pytest.raises(ValueError, match="Invalid clock source"):
            await server.get_current_time("UTC", "invalid_source")
    
    @pytest.mark.asyncio
    async def test_convert_time(self, server):
        """Test convert_time method."""
        # Mock current timestamp
        mock_current = PrecisionTimestamp(
            utc_seconds=1703980800,  # 2023-12-30 16:00:00 UTC
            utc_nanoseconds=0,
            timezone="UTC",
            is_dst=False,
            monotonic_ns=123456789,
            source=ClockSource.SYSTEM,
            quality=ClockQuality.FAIR
        )
        
        # Mock conversion result
        mock_conversion_result = MagicMock()
        mock_conversion_result.source = mock_current
        mock_conversion_result.target = mock_current  # Simplified
        mock_conversion_result.offset_seconds = -18000  # -5 hours
        
        with patch.object(server.clock_manager, 'get_precision_time', return_value=mock_current):
            with patch.object(server.clock_manager, 'convert_time', return_value=mock_conversion_result):
                result = await server.convert_time("UTC", "14:30", "America/New_York")
                
                assert result.source == mock_current
                assert result.target == mock_current
                assert result.time_difference == "-5.0h"
                assert result.offset_seconds == -18000
    
    @pytest.mark.asyncio
    async def test_convert_time_invalid_format(self, server):
        """Test convert_time with invalid time format."""
        with pytest.raises(ValueError, match="Invalid time format"):
            await server.convert_time("UTC", "invalid_time", "America/New_York")
    
    @pytest.mark.asyncio
    async def test_get_sync_status_all(self, server):
        """Test get_sync_status for all sources."""
        mock_overall_status = SyncStatus(
            is_synchronized=True,
            source=ClockSource.NTP,
            quality=ClockQuality.GOOD
        )
        
        mock_all_status = {
            "system": SyncStatus(
                is_synchronized=True,
                source=ClockSource.SYSTEM,
                quality=ClockQuality.FAIR
            ),
            "ntp": SyncStatus(
                is_synchronized=True,
                source=ClockSource.NTP,
                quality=ClockQuality.GOOD
            )
        }
        
        with patch.object(server.clock_manager, 'get_overall_sync_status', return_value=mock_overall_status):
            with patch.object(server.clock_manager, 'get_sync_status', return_value=mock_all_status):
                result = await server.get_sync_status()
                
                assert "overall" in result
                assert "sources" in result
                assert result["overall"] == mock_overall_status.model_dump()
                assert len(result["sources"]) == 2
    
    @pytest.mark.asyncio
    async def test_get_sync_status_specific_source(self, server):
        """Test get_sync_status for specific source."""
        mock_status = SyncStatus(
            is_synchronized=True,
            source=ClockSource.NTP,
            quality=ClockQuality.GOOD
        )
        
        mock_all_status = {"ntp": mock_status}
        
        with patch.object(server.clock_manager, 'get_sync_status', return_value=mock_all_status):
            result = await server.get_sync_status("ntp")
            
            assert result["source"] == "ntp"
            assert result["status"] == mock_status.model_dump()
    
    @pytest.mark.asyncio
    async def test_get_sync_status_unknown_source(self, server):
        """Test get_sync_status for unknown source."""
        with patch.object(server.clock_manager, 'get_sync_status', return_value={}):
            with pytest.raises(ValueError, match="Unknown time source"):
                await server.get_sync_status("unknown")
    
    def test_create_vector_clock(self, server):
        """Test create_vector_clock method."""
        mock_vector_clock = VectorClock(
            node_id="test-server",
            clocks={"test-server": 1},
            timestamp=PrecisionTimestamp.now()
        )
        
        with patch.object(server.clock_manager, 'create_vector_clock', return_value=mock_vector_clock):
            result = server.create_vector_clock()
            
            assert result == mock_vector_clock
    
    def test_compare_timestamps(self, server):
        """Test compare_timestamps method."""
        ts1_dict = {
            "utc_seconds": 1703980800,
            "utc_nanoseconds": 0,
            "timezone": "UTC",
            "is_dst": False,
            "monotonic_ns": 1000,
            "source": "system",
            "quality": "fair"
        }
        
        ts2_dict = {
            "utc_seconds": 1703980801,
            "utc_nanoseconds": 0,
            "timezone": "UTC",
            "is_dst": False,
            "monotonic_ns": 2000,
            "source": "system",
            "quality": "fair"
        }
        
        mock_comparison = {
            "time_difference_ns": 1_000_000_000,
            "ts1_before_ts2": True
        }
        
        with patch.object(server.clock_manager, 'compare_timestamps', return_value=mock_comparison):
            result = server.compare_timestamps(ts1_dict, ts2_dict)
            
            assert result == mock_comparison
    
    def test_compare_timestamps_invalid_format(self, server):
        """Test compare_timestamps with invalid format."""
        invalid_ts = {"invalid": "format"}
        valid_ts = {
            "utc_seconds": 1703980800,
            "utc_nanoseconds": 0,
            "timezone": "UTC",
            "is_dst": False,
            "monotonic_ns": 1000,
            "source": "system",
            "quality": "fair"
        }
        
        with pytest.raises(ValueError, match="Invalid timestamp format"):
            server.compare_timestamps(invalid_ts, valid_ts)
    
    @pytest.mark.asyncio
    async def test_measure_clock_drift(self, server):
        """Test measure_clock_drift method."""
        mock_drift_result = {
            "system": {
                "drift_rate_ppm": 0.5,
                "quality": "good"
            }
        }
        
        with patch.object(server.clock_manager, 'measure_clock_drift', return_value=mock_drift_result):
            result = await server.measure_clock_drift(60)
            
            assert result == mock_drift_result
            server.clock_manager.measure_clock_drift.assert_called_once_with(60)
    
    @pytest.mark.asyncio
    async def test_measure_clock_drift_invalid_duration(self, server):
        """Test measure_clock_drift with invalid duration."""
        with pytest.raises(ValueError, match="Duration must be between 1 and 3600 seconds"):
            await server.measure_clock_drift(0)
        
        with pytest.raises(ValueError, match="Duration must be between 1 and 3600 seconds"):
            await server.measure_clock_drift(3601)


class TestChronoTickTools:
    """Test ChronoTickTools enum."""
    
    def test_available_tools(self):
        """Test that expected tools are available."""
        expected_tools = [
            "get_current_time",
            "convert_time",
            "get_sync_status",
            "create_vector_clock",
            "compare_timestamps",
            "measure_clock_drift"
        ]
        
        for tool in expected_tools:
            assert hasattr(ChronoTickTools, tool.upper())
            assert getattr(ChronoTickTools, tool.upper()).value == tool
    
    def test_tool_count(self):
        """Test that we have the expected number of tools."""
        tools = list(ChronoTickTools)
        assert len(tools) == 6  # Should have exactly 6 tools