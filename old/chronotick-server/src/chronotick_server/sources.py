"""Time source abstraction layer for ChronoTick."""

import time
import asyncio
import ntplib
import psutil
import subprocess
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from zoneinfo import ZoneInfo

from .models import (
    PrecisionTimestamp, 
    ClockSource, 
    ClockQuality, 
    SyncStatus
)


class TimeSource(ABC):
    """Abstract base class for time sources."""
    
    def __init__(self, name: str):
        self.name = name
        self._last_sync: Optional[datetime] = None
        self._sync_count = 0
        self._error_count = 0
    
    @abstractmethod
    async def get_time(self, timezone_name: str = "UTC") -> PrecisionTimestamp:
        """Get current time from this source."""
        pass
    
    @abstractmethod
    async def get_sync_status(self) -> SyncStatus:
        """Get synchronization status for this source."""
        pass
    
    @abstractmethod
    async def is_available(self) -> bool:
        """Check if this time source is available."""
        pass
    
    @property
    def source_type(self) -> ClockSource:
        """Get the source type enum."""
        return ClockSource.SYSTEM
    
    @property
    def sync_count(self) -> int:
        """Number of successful synchronizations."""
        return self._sync_count
    
    @property
    def error_count(self) -> int:
        """Number of synchronization errors."""
        return self._error_count


class SystemTimeSource(TimeSource):
    """System clock time source."""
    
    def __init__(self):
        super().__init__("system")
    
    @property
    def source_type(self) -> ClockSource:
        return ClockSource.SYSTEM
    
    async def get_time(self, timezone_name: str = "UTC") -> PrecisionTimestamp:
        """Get current time from system clock."""
        try:
            # Get high-resolution time
            now_ns = time.time_ns()
            utc_seconds = now_ns // 1_000_000_000
            utc_nanoseconds = now_ns % 1_000_000_000
            monotonic_ns = time.monotonic_ns()
            
            # Handle timezone
            if timezone_name == "UTC":
                dt = datetime.fromtimestamp(utc_seconds, tz=timezone.utc)
                is_dst = False
            else:
                tz = ZoneInfo(timezone_name)
                dt = datetime.fromtimestamp(utc_seconds, tz=tz)
                is_dst = bool(dt.dst()) if dt.dst() is not None else False
            
            self._sync_count += 1
            self._last_sync = datetime.now()
            
            return PrecisionTimestamp(
                utc_seconds=utc_seconds,
                utc_nanoseconds=utc_nanoseconds,
                timezone=timezone_name,
                is_dst=is_dst,
                monotonic_ns=monotonic_ns,
                source=ClockSource.SYSTEM,
                quality=ClockQuality.FAIR,  # System clock has moderate quality
                uncertainty_ns=1_000_000    # ~1ms uncertainty for system clock
            )
        except Exception as e:
            self._error_count += 1
            raise RuntimeError(f"System time source error: {e}")
    
    async def get_sync_status(self) -> SyncStatus:
        """Get system clock sync status."""
        uptime = int(time.time() - psutil.boot_time()) if psutil else None
        
        return SyncStatus(
            is_synchronized=True,  # System clock is always "synchronized"
            source=ClockSource.SYSTEM,
            quality=ClockQuality.FAIR,
            reachable_sources=1,
            total_sources=1,
            uptime_seconds=uptime
        )
    
    async def is_available(self) -> bool:
        """System clock is always available."""
        return True


class NTPTimeSource(TimeSource):
    """NTP-based time source."""
    
    def __init__(self, servers: List[str] = None):
        super().__init__("ntp")
        self.servers = servers or [
            "pool.ntp.org",
            "time.nist.gov",
            "time.google.com",
            "time.cloudflare.com"
        ]
        self.client = ntplib.NTPClient()
        self._last_response: Optional[ntplib.NTPStats] = None
        self._server_status: Dict[str, bool] = {}
    
    @property
    def source_type(self) -> ClockSource:
        return ClockSource.NTP
    
    async def get_time(self, timezone_name: str = "UTC") -> PrecisionTimestamp:
        """Get current time from NTP servers."""
        best_response = None
        best_server = None
        
        for server in self.servers:
            try:
                # Run NTP query in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None, 
                    lambda: self.client.request(server, version=3, timeout=5)
                )
                
                self._server_status[server] = True
                
                # Select response with lowest delay
                if best_response is None or response.delay < best_response.delay:
                    best_response = response
                    best_server = server
                    
            except Exception:
                self._server_status[server] = False
                continue
        
        if best_response is None:
            self._error_count += 1
            raise RuntimeError("No NTP servers reachable")
        
        self._last_response = best_response
        self._sync_count += 1
        self._last_sync = datetime.now()
        
        # Convert NTP timestamp to our format
        ntp_time = best_response.tx_time
        utc_seconds = int(ntp_time)
        utc_nanoseconds = int((ntp_time - utc_seconds) * 1e9)
        monotonic_ns = time.monotonic_ns()
        
        # Handle timezone
        if timezone_name == "UTC":
            dt = datetime.fromtimestamp(ntp_time, tz=timezone.utc)
            is_dst = False
        else:
            tz = ZoneInfo(timezone_name)
            dt = datetime.fromtimestamp(ntp_time, tz=tz)
            is_dst = bool(dt.dst()) if dt.dst() is not None else False
        
        # Determine quality based on NTP metrics
        quality = self._determine_ntp_quality(best_response)
        
        return PrecisionTimestamp(
            utc_seconds=utc_seconds,
            utc_nanoseconds=utc_nanoseconds,
            timezone=timezone_name,
            is_dst=is_dst,
            monotonic_ns=monotonic_ns,
            source=ClockSource.NTP,
            quality=quality,
            offset_ns=int(best_response.offset * 1e9),
            uncertainty_ns=int(best_response.delay * 1e9 / 2)  # Half of round-trip delay
        )
    
    def _determine_ntp_quality(self, response: ntplib.NTPStats) -> ClockQuality:
        """Determine clock quality based on NTP metrics."""
        if response.stratum == 1:
            return ClockQuality.REFERENCE
        elif response.delay < 0.01:  # < 10ms
            return ClockQuality.EXCELLENT
        elif response.delay < 0.1:   # < 100ms
            return ClockQuality.GOOD
        elif response.delay < 1.0:   # < 1s
            return ClockQuality.FAIR
        else:
            return ClockQuality.POOR
    
    async def get_sync_status(self) -> SyncStatus:
        """Get NTP synchronization status."""
        reachable = sum(1 for status in self._server_status.values() if status)
        total = len(self.servers)
        
        ntp_offset_ms = None
        ntp_delay_ms = None
        ntp_stratum = None
        quality = ClockQuality.UNKNOWN
        
        if self._last_response:
            ntp_offset_ms = self._last_response.offset * 1000
            ntp_delay_ms = self._last_response.delay * 1000
            ntp_stratum = self._last_response.stratum
            quality = self._determine_ntp_quality(self._last_response)
        
        return SyncStatus(
            is_synchronized=reachable > 0,
            source=ClockSource.NTP,
            quality=quality,
            ntp_stratum=ntp_stratum,
            ntp_offset_ms=ntp_offset_ms,
            ntp_delay_ms=ntp_delay_ms,
            ntp_servers=self.servers,
            reachable_sources=reachable,
            total_sources=total,
            sync_interval_seconds=64  # Typical NTP poll interval
        )
    
    async def is_available(self) -> bool:
        """Check if any NTP servers are reachable."""
        for server in self.servers:
            try:
                loop = asyncio.get_event_loop()
                await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: self.client.request(server, version=3, timeout=2)
                    ),
                    timeout=3
                )
                return True
            except:
                continue
        return False


class ChronyTimeSource(TimeSource):
    """Chrony-based NTP time source using system chrony daemon."""
    
    def __init__(self):
        super().__init__("chrony")
        self._last_tracking = None
        self._last_sources = None
    
    @property
    def source_type(self) -> ClockSource:
        return ClockSource.NTP
    
    async def get_time(self, timezone_name: str = "UTC") -> PrecisionTimestamp:
        """Get current time using chrony synchronization info."""
        if not await self.is_available():
            self._error_count += 1
            raise RuntimeError("Chrony not available")
        
        # Get high-resolution time
        now_ns = time.time_ns()
        utc_seconds = now_ns // 1_000_000_000
        utc_nanoseconds = now_ns % 1_000_000_000
        monotonic_ns = time.monotonic_ns()
        
        # Get chrony tracking information
        tracking_info = await self._get_chrony_tracking()
        
        # Handle timezone
        if timezone_name == "UTC":
            dt = datetime.fromtimestamp(utc_seconds, tz=timezone.utc)
            is_dst = False
        else:
            tz = ZoneInfo(timezone_name)
            dt = datetime.fromtimestamp(utc_seconds, tz=tz)
            is_dst = bool(dt.dst()) if dt.dst() is not None else False
        
        # Determine quality based on chrony metrics
        quality = self._determine_chrony_quality(tracking_info)
        
        # Extract offset and uncertainty from chrony
        offset_ns = None
        uncertainty_ns = None
        if tracking_info:
            # Chrony offset is typically in seconds, convert to nanoseconds
            if 'system_time_offset' in tracking_info:
                offset_ns = int(tracking_info['system_time_offset'] * 1e9)
            if 'root_dispersion' in tracking_info:
                uncertainty_ns = int(tracking_info['root_dispersion'] * 1e9)
        
        self._sync_count += 1
        self._last_sync = datetime.now()
        
        return PrecisionTimestamp(
            utc_seconds=utc_seconds,
            utc_nanoseconds=utc_nanoseconds,
            timezone=timezone_name,
            is_dst=is_dst,
            monotonic_ns=monotonic_ns,
            source=ClockSource.NTP,
            quality=quality,
            offset_ns=offset_ns,
            uncertainty_ns=uncertainty_ns or 1_000_000  # Default 1ms uncertainty
        )
    
    async def _get_chrony_tracking(self) -> Optional[Dict[str, Any]]:
        """Get chrony tracking information."""
        try:
            # Run chronyc tracking command
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    ['chronyc', 'tracking'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
            )
            
            if result.returncode != 0:
                return None
            
            # Parse tracking output
            tracking_info = {}
            for line in result.stdout.strip().split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower().replace(' ', '_')
                    value = value.strip()
                    
                    # Try to convert numeric values
                    try:
                        if 'seconds' in value or 'ns' in value or 'ms' in value or 'us' in value:
                            # Extract numeric part and convert to seconds
                            import re
                            match = re.search(r'([+-]?\d*\.?\d+)', value)
                            if match:
                                num_val = float(match.group(1))
                                if 'ms' in value:
                                    num_val /= 1000
                                elif 'us' in value:
                                    num_val /= 1_000_000
                                elif 'ns' in value:
                                    num_val /= 1_000_000_000
                                tracking_info[key] = num_val
                            else:
                                tracking_info[key] = value
                        else:
                            # Try to convert to float if possible
                            try:
                                tracking_info[key] = float(value)
                            except ValueError:
                                tracking_info[key] = value
                    except:
                        tracking_info[key] = value
            
            self._last_tracking = tracking_info
            return tracking_info
        
        except Exception:
            return None
    
    async def _get_chrony_sources(self) -> Optional[List[Dict[str, Any]]]:
        """Get chrony sources information."""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    ['chronyc', 'sources', '-v'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
            )
            
            if result.returncode != 0:
                return None
            
            # Parse sources output
            sources = []
            lines = result.stdout.strip().split('\n')
            
            # Skip header lines
            data_started = False
            for line in lines:
                if line.startswith('MS Name/IP address'):
                    data_started = True
                    continue
                if not data_started or not line.strip():
                    continue
                
                # Parse source line
                parts = line.split()
                if len(parts) >= 9:
                    source = {
                        'mode': parts[0][:2],
                        'state': parts[0][2:] if len(parts[0]) > 2 else '',
                        'name': parts[1],
                        'stratum': int(parts[2]) if parts[2].isdigit() else 0,
                        'poll': int(parts[3]) if parts[3].isdigit() else 0,
                        'reach': parts[4],
                        'last_rx': parts[5],
                        'last_sample': parts[6:]
                    }
                    sources.append(source)
            
            self._last_sources = sources
            return sources
        
        except Exception:
            return None
    
    def _determine_chrony_quality(self, tracking_info: Optional[Dict[str, Any]]) -> ClockQuality:
        """Determine clock quality based on chrony metrics."""
        if not tracking_info:
            return ClockQuality.UNKNOWN
        
        # Check if synchronized
        leap_status = tracking_info.get('leap_status', '')
        if 'not synchronised' in leap_status.lower():
            return ClockQuality.POOR
        
        # Check stratum
        stratum = tracking_info.get('stratum', 16)
        if isinstance(stratum, str):
            try:
                stratum = int(stratum)
            except:
                stratum = 16
        
        if stratum == 1:
            return ClockQuality.REFERENCE
        
        # Check root delay and dispersion
        root_delay = tracking_info.get('root_delay', 1.0)
        root_dispersion = tracking_info.get('root_dispersion', 1.0)
        
        if isinstance(root_delay, str):
            try:
                root_delay = float(root_delay.split()[0])
            except:
                root_delay = 1.0
        
        if isinstance(root_dispersion, str):
            try:
                root_dispersion = float(root_dispersion.split()[0])
            except:
                root_dispersion = 1.0
        
        total_error = root_delay + root_dispersion
        
        if total_error < 0.01:  # < 10ms
            return ClockQuality.EXCELLENT
        elif total_error < 0.1:  # < 100ms
            return ClockQuality.GOOD
        elif total_error < 1.0:  # < 1s
            return ClockQuality.FAIR
        else:
            return ClockQuality.POOR
    
    async def get_sync_status(self) -> SyncStatus:
        """Get chrony synchronization status."""
        tracking_info = await self._get_chrony_tracking()
        sources_info = await self._get_chrony_sources()
        
        is_synchronized = False
        quality = ClockQuality.UNKNOWN
        ntp_stratum = None
        ntp_offset_ms = None
        ntp_servers = []
        reachable_sources = 0
        total_sources = 0
        
        if tracking_info:
            leap_status = tracking_info.get('leap_status', '')
            is_synchronized = 'synchronised' in leap_status.lower()
            quality = self._determine_chrony_quality(tracking_info)
            
            if 'stratum' in tracking_info:
                ntp_stratum = int(tracking_info['stratum']) if str(tracking_info['stratum']).isdigit() else None
            
            if 'system_time_offset' in tracking_info:
                ntp_offset_ms = tracking_info['system_time_offset'] * 1000
        
        if sources_info:
            total_sources = len(sources_info)
            for source in sources_info:
                ntp_servers.append(source['name'])
                # Check if source is reachable (reach should have some bits set)
                reach = source.get('reach', '0')
                if reach != '0' and reach != '000':
                    reachable_sources += 1
        
        return SyncStatus(
            is_synchronized=is_synchronized,
            source=ClockSource.NTP,
            quality=quality,
            ntp_stratum=ntp_stratum,
            ntp_offset_ms=ntp_offset_ms,
            ntp_servers=ntp_servers,
            reachable_sources=reachable_sources,
            total_sources=total_sources
        )
    
    async def is_available(self) -> bool:
        """Check if chrony is available and running."""
        try:
            # Check if chronyc command exists and chrony daemon is running
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    ['chronyc', 'tracking'],
                    capture_output=True,
                    text=True,
                    timeout=3
                )
            )
            return result.returncode == 0
        except Exception:
            return False



class TimeSourceRegistry:
    """Registry for managing multiple time sources."""
    
    def __init__(self):
        self.sources: Dict[str, TimeSource] = {}
        self._primary_source: Optional[str] = None
    
    def register(self, source: TimeSource) -> None:
        """Register a time source."""
        self.sources[source.name] = source
        
        # Set first registered source as primary
        if self._primary_source is None:
            self._primary_source = source.name
    
    def get_source(self, name: str) -> Optional[TimeSource]:
        """Get a time source by name."""
        return self.sources.get(name)
    
    def set_primary(self, name: str) -> None:
        """Set the primary time source."""
        if name in self.sources:
            self._primary_source = name
        else:
            raise ValueError(f"Unknown time source: {name}")
    
    @property
    def primary(self) -> Optional[TimeSource]:
        """Get the primary time source."""
        if self._primary_source:
            return self.sources.get(self._primary_source)
        return None
    
    async def get_best_source(self) -> Optional[TimeSource]:
        """Get the best available time source based on quality."""
        available_sources = []
        
        for source in self.sources.values():
            if await source.is_available():
                status = await source.get_sync_status()
                available_sources.append((source, status.quality))
        
        if not available_sources:
            return None
        
        # Sort by quality (REFERENCE > EXCELLENT > GOOD > FAIR > POOR > UNKNOWN)
        quality_order = {
            ClockQuality.REFERENCE: 5,
            ClockQuality.EXCELLENT: 4,
            ClockQuality.GOOD: 3,
            ClockQuality.FAIR: 2,
            ClockQuality.POOR: 1,
            ClockQuality.UNKNOWN: 0
        }
        
        available_sources.sort(key=lambda x: quality_order.get(x[1], 0), reverse=True)
        return available_sources[0][0]
    
    def list_sources(self) -> List[str]:
        """List all registered source names."""
        return list(self.sources.keys())