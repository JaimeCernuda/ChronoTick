"""Tests for ChronoTick time sources."""

import pytest
import asyncio
import time
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timezone

from chronotick_server.sources import (
    SystemTimeSource,
    NTPTimeSource,
    ChronyTimeSource,
    TimeSourceRegistry
)
from chronotick_server.models import ClockSource, ClockQuality


class TestSystemTimeSource:
    """Test SystemTimeSource."""
    
    @pytest.fixture
    def system_source(self):
        """Create a system time source."""
        return SystemTimeSource()
    
    def test_source_type(self, system_source):
        """Test source type is SYSTEM."""
        assert system_source.source_type == ClockSource.SYSTEM
    
    @pytest.mark.asyncio
    async def test_get_time_utc(self, system_source):
        """Test getting UTC time."""
        timestamp = await system_source.get_time("UTC")
        
        assert timestamp.timezone == "UTC"
        assert timestamp.source == ClockSource.SYSTEM
        assert timestamp.quality == ClockQuality.FAIR
        assert not timestamp.is_dst
        assert timestamp.utc_seconds > 0
        assert 0 <= timestamp.utc_nanoseconds < 1_000_000_000
        assert timestamp.uncertainty_ns == 1_000_000  # 1ms
    
    @pytest.mark.asyncio
    async def test_get_time_timezone(self, system_source):
        """Test getting time in different timezone."""
        timestamp = await system_source.get_time("America/New_York")
        
        assert timestamp.timezone == "America/New_York"
        assert timestamp.source == ClockSource.SYSTEM
    
    @pytest.mark.asyncio
    async def test_get_sync_status(self, system_source):
        """Test getting sync status."""
        status = await system_source.get_sync_status()
        
        assert status.is_synchronized
        assert status.source == ClockSource.SYSTEM
        assert status.quality == ClockQuality.FAIR
        assert status.reachable_sources == 1
        assert status.total_sources == 1
    
    @pytest.mark.asyncio
    async def test_is_available(self, system_source):
        """Test that system clock is always available."""
        assert await system_source.is_available()
    
    @pytest.mark.asyncio
    async def test_sync_count_increments(self, system_source):
        """Test that sync count increments."""
        initial_count = system_source.sync_count
        await system_source.get_time()
        assert system_source.sync_count == initial_count + 1


class TestNTPTimeSource:
    """Test NTPTimeSource."""
    
    @pytest.fixture
    def ntp_source(self):
        """Create an NTP time source."""
        return NTPTimeSource(servers=["test.ntp.server"])
    
    def test_source_type(self, ntp_source):
        """Test source type is NTP."""
        assert ntp_source.source_type == ClockSource.NTP
    
    @pytest.mark.asyncio
    async def test_get_time_success(self, ntp_source):
        """Test successful NTP time retrieval."""
        # Mock NTP response
        mock_response = MagicMock()
        mock_response.tx_time = time.time()
        mock_response.delay = 0.05  # 50ms
        mock_response.offset = 0.001  # 1ms offset
        mock_response.stratum = 2
        
        with patch.object(ntp_source.client, 'request', return_value=mock_response):
            timestamp = await ntp_source.get_time("UTC")
            
            assert timestamp.timezone == "UTC"
            assert timestamp.source == ClockSource.NTP
            assert timestamp.quality == ClockQuality.GOOD  # < 100ms delay
            assert timestamp.offset_ns == 1_000_000  # 1ms in nanoseconds
            assert timestamp.uncertainty_ns == 25_000_000  # Half of 50ms delay
    
    @pytest.mark.asyncio
    async def test_get_time_all_servers_fail(self, ntp_source):
        """Test when all NTP servers fail."""
        with patch.object(ntp_source.client, 'request', side_effect=Exception("Network error")):
            with pytest.raises(RuntimeError, match="No NTP servers reachable"):
                await ntp_source.get_time("UTC")
    
    @pytest.mark.asyncio
    async def test_get_time_selects_best_server(self):
        """Test that NTP source selects server with lowest delay."""
        ntp_source = NTPTimeSource(servers=["server1", "server2"])
        
        # Mock responses with different delays
        def mock_request(server, **kwargs):
            response = MagicMock()
            response.tx_time = time.time()
            response.stratum = 2
            if server == "server1":
                response.delay = 0.1  # 100ms
                response.offset = 0.005
            else:
                response.delay = 0.05  # 50ms (better)
                response.offset = 0.002
            return response
        
        with patch.object(ntp_source.client, 'request', side_effect=mock_request):
            timestamp = await ntp_source.get_time("UTC")
            
            # Should use server2 (lower delay)
            assert timestamp.uncertainty_ns == 25_000_000  # Half of 50ms
    
    def test_determine_ntp_quality(self, ntp_source):
        """Test NTP quality determination."""
        # Test reference quality (stratum 1)
        response = MagicMock()
        response.stratum = 1
        response.delay = 0.5
        assert ntp_source._determine_ntp_quality(response) == ClockQuality.REFERENCE
        
        # Test excellent quality (< 10ms)
        response.stratum = 2
        response.delay = 0.005
        assert ntp_source._determine_ntp_quality(response) == ClockQuality.EXCELLENT
        
        # Test good quality (< 100ms)
        response.delay = 0.05
        assert ntp_source._determine_ntp_quality(response) == ClockQuality.GOOD
        
        # Test fair quality (< 1s)
        response.delay = 0.5
        assert ntp_source._determine_ntp_quality(response) == ClockQuality.FAIR
        
        # Test poor quality (>= 1s)
        response.delay = 1.5
        assert ntp_source._determine_ntp_quality(response) == ClockQuality.POOR
    
    @pytest.mark.asyncio
    async def test_get_sync_status(self, ntp_source):
        """Test getting NTP sync status."""
        # Mock last response
        mock_response = MagicMock()
        mock_response.offset = 0.005  # 5ms
        mock_response.delay = 0.03   # 30ms
        mock_response.stratum = 3
        ntp_source._last_response = mock_response
        ntp_source._server_status = {"test.ntp.server": True}
        
        status = await ntp_source.get_sync_status()
        
        assert status.is_synchronized
        assert status.source == ClockSource.NTP
        assert status.ntp_stratum == 3
        assert status.ntp_offset_ms == 5.0
        assert status.ntp_delay_ms == 30.0
        assert status.reachable_sources == 1
        assert status.total_sources == 1
    
    @pytest.mark.asyncio
    async def test_is_available_success(self, ntp_source):
        """Test NTP availability check success."""
        mock_response = MagicMock()
        with patch.object(ntp_source.client, 'request', return_value=mock_response):
            assert await ntp_source.is_available()
    
    @pytest.mark.asyncio
    async def test_is_available_failure(self, ntp_source):
        """Test NTP availability check failure."""
        with patch.object(ntp_source.client, 'request', side_effect=Exception("Network error")):
            assert not await ntp_source.is_available()


class TestChronyTimeSource:
    """Test ChronyTimeSource."""
    
    @pytest.fixture
    def chrony_source(self):
        """Create a chrony time source."""
        return ChronyTimeSource()
    
    def test_source_type(self, chrony_source):
        """Test source type is NTP."""
        assert chrony_source.source_type == ClockSource.NTP
    
    @pytest.mark.asyncio
    async def test_get_time_success(self, chrony_source):
        """Test successful chrony time retrieval."""
        # Mock chrony tracking info
        mock_tracking = {
            'system_time_offset': 0.002,  # 2ms offset
            'root_dispersion': 0.001,     # 1ms dispersion
            'leap_status': 'Normal'
        }
        
        with patch.object(chrony_source, '_get_chrony_tracking', return_value=mock_tracking):
            with patch.object(chrony_source, 'is_available', return_value=True):
                timestamp = await chrony_source.get_time("UTC")
                
                assert timestamp.timezone == "UTC"
                assert timestamp.source == ClockSource.NTP
                assert timestamp.offset_ns == 2_000_000  # 2ms in nanoseconds
                assert timestamp.uncertainty_ns == 1_000_000  # 1ms in nanoseconds
    
    @pytest.mark.asyncio
    async def test_get_time_not_available(self, chrony_source):
        """Test chrony time when not available."""
        with patch.object(chrony_source, 'is_available', return_value=False):
            with pytest.raises(RuntimeError, match="Chrony not available"):
                await chrony_source.get_time("UTC")
    
    @pytest.mark.asyncio
    async def test_get_chrony_tracking_success(self, chrony_source):
        """Test parsing chrony tracking output."""
        mock_output = '''Reference ID    : 12345678 (server.example.com)
Stratum         : 3
Ref time (UTC)  : Mon Dec 30 16:00:00 2023
System time     : 0.002345678 seconds fast of NTP time
Last offset     : +0.001234567 seconds
RMS offset      : 0.000456789 seconds
Frequency       : 15.123 ppm slow
Residual freq   : +0.123 ppm
Skew            : 0.456 ppm
Root delay      : 0.030000000 seconds
Root dispersion : 0.001000000 seconds
Update interval : 64.0 seconds
Leap status     : Normal'''
        
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = mock_output
        
        with patch('subprocess.run', return_value=mock_result):
            tracking = await chrony_source._get_chrony_tracking()
            
            assert tracking is not None
            assert 'stratum' in tracking
            assert tracking['stratum'] == 3
            assert 'system_time' in tracking
    
    @pytest.mark.asyncio
    async def test_get_chrony_tracking_failure(self, chrony_source):
        """Test chrony tracking when command fails."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        
        with patch('subprocess.run', return_value=mock_result):
            tracking = await chrony_source._get_chrony_tracking()
            assert tracking is None
    
    def test_determine_chrony_quality(self, chrony_source):
        """Test chrony quality determination."""
        # Test not synchronized
        tracking = {'leap_status': 'Not synchronised'}
        assert chrony_source._determine_chrony_quality(tracking) == ClockQuality.POOR
        
        # Test reference quality (stratum 1)
        tracking = {'leap_status': 'Normal', 'stratum': 1, 'root_delay': 0.001, 'root_dispersion': 0.001}
        assert chrony_source._determine_chrony_quality(tracking) == ClockQuality.REFERENCE
        
        # Test excellent quality
        tracking = {'leap_status': 'Normal', 'stratum': 2, 'root_delay': 0.005, 'root_dispersion': 0.003}
        assert chrony_source._determine_chrony_quality(tracking) == ClockQuality.EXCELLENT
        
        # Test good quality
        tracking = {'leap_status': 'Normal', 'stratum': 3, 'root_delay': 0.05, 'root_dispersion': 0.03}
        assert chrony_source._determine_chrony_quality(tracking) == ClockQuality.GOOD
    
    @pytest.mark.asyncio
    async def test_get_sync_status(self, chrony_source):
        """Test getting chrony sync status."""
        mock_tracking = {
            'leap_status': 'Normal (synchronised)',
            'stratum': 2,
            'system_time_offset': 0.003,
        }
        
        mock_sources = [
            {'name': 'server1.example.com', 'reach': '377'},
            {'name': 'server2.example.com', 'reach': '000'},
        ]
        
        with patch.object(chrony_source, '_get_chrony_tracking', return_value=mock_tracking):
            with patch.object(chrony_source, '_get_chrony_sources', return_value=mock_sources):
                status = await chrony_source.get_sync_status()
                
                assert status.is_synchronized
                assert status.source == ClockSource.NTP
                assert status.ntp_stratum == 2
                assert status.ntp_offset_ms == 3.0
                assert len(status.ntp_servers) == 2
                assert status.reachable_sources == 1  # Only one with non-zero reach
                assert status.total_sources == 2
    
    @pytest.mark.asyncio
    async def test_is_available_success(self, chrony_source):
        """Test chrony availability check success."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        
        with patch('subprocess.run', return_value=mock_result):
            assert await chrony_source.is_available()
    
    @pytest.mark.asyncio
    async def test_is_available_failure(self, chrony_source):
        """Test chrony availability check failure."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        
        with patch('subprocess.run', return_value=mock_result):
            assert not await chrony_source.is_available()


class TestTimeSourceRegistry:
    """Test TimeSourceRegistry."""
    
    @pytest.fixture
    def registry(self):
        """Create a time source registry."""
        return TimeSourceRegistry()
    
    @pytest.fixture
    def mock_sources(self):
        """Create mock time sources."""
        system_source = SystemTimeSource()
        ntp_source = NTPTimeSource()
        return system_source, ntp_source
    
    def test_register_source(self, registry, mock_sources):
        """Test registering a time source."""
        system_source, _ = mock_sources
        registry.register(system_source)
        
        assert "system" in registry.sources
        assert registry.get_source("system") == system_source
        assert registry.primary == system_source  # First registered becomes primary
    
    def test_set_primary(self, registry, mock_sources):
        """Test setting primary source."""
        system_source, ntp_source = mock_sources
        registry.register(system_source)
        registry.register(ntp_source)
        
        assert registry.primary == system_source  # Initially first registered
        
        registry.set_primary("ntp")
        assert registry.primary == ntp_source
    
    def test_set_primary_unknown(self, registry):
        """Test setting unknown primary source."""
        with pytest.raises(ValueError, match="Unknown time source"):
            registry.set_primary("unknown")
    
    def test_list_sources(self, registry, mock_sources):
        """Test listing source names."""
        system_source, ntp_source = mock_sources
        registry.register(system_source)
        registry.register(ntp_source)
        
        sources = registry.list_sources()
        assert "system" in sources
        assert "ntp" in sources
        assert len(sources) == 2
    
    @pytest.mark.asyncio
    async def test_get_best_source(self, registry):
        """Test getting best available source."""
        # Create mock sources with different qualities
        system_source = MagicMock()
        system_source.is_available = AsyncMock(return_value=True)
        system_source.get_sync_status = AsyncMock()
        system_source.get_sync_status.return_value.quality = ClockQuality.FAIR
        
        ntp_source = MagicMock()
        ntp_source.is_available = AsyncMock(return_value=True)
        ntp_source.get_sync_status = AsyncMock()
        ntp_source.get_sync_status.return_value.quality = ClockQuality.EXCELLENT
        
        registry.sources = {"system": system_source, "ntp": ntp_source}
        
        best = await registry.get_best_source()
        assert best == ntp_source  # Should pick EXCELLENT over FAIR
    
    @pytest.mark.asyncio
    async def test_get_best_source_none_available(self, registry):
        """Test getting best source when none are available."""
        mock_source = MagicMock()
        mock_source.is_available = AsyncMock(return_value=False)
        registry.sources = {"test": mock_source}
        
        best = await registry.get_best_source()
        assert best is None