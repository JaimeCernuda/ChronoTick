"""Precision clock manager for ChronoTick."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from uuid import uuid4

from .models import (
    PrecisionTimestamp,
    ClockSource,
    ClockQuality,
    SyncStatus,
    VectorClock,
    TimeConversionRequest,
    TimeConversionResult
)
from .sources import (
    TimeSourceRegistry,
    SystemTimeSource,
    NTPTimeSource,
    ChronyTimeSource
)


class PrecisionClockManager:
    """Central manager for high-precision time services."""
    
    def __init__(self, node_id: Optional[str] = None):
        self.node_id = node_id or str(uuid4())
        self.registry = TimeSourceRegistry()
        self._vector_clock: Optional[VectorClock] = None
        self._drift_monitor = DriftMonitor()
        self.logger = logging.getLogger(__name__)
        
        # Initialize with default sources
        self._initialize_default_sources()
    
    def _initialize_default_sources(self) -> None:
        """Initialize with default time sources."""
        # Always register system clock
        self.registry.register(SystemTimeSource())
        
        # Register NTP source
        ntp_source = NTPTimeSource()
        self.registry.register(ntp_source)
        
        # Register Chrony source if available
        chrony_source = ChronyTimeSource()
        self.registry.register(chrony_source)
        
        
        # Initialize vector clock
        self._vector_clock = VectorClock(
            node_id=self.node_id,
            clocks={self.node_id: 0},
            timestamp=PrecisionTimestamp.now()
        )
    
    async def get_precision_time(self, 
                               timezone_name: str = "UTC",
                               source_preference: Optional[ClockSource] = None) -> PrecisionTimestamp:
        """Get high-precision current time."""
        source = None
        
        if source_preference:
            # Try to use preferred source
            for src in self.registry.sources.values():
                if src.source_type == source_preference and await src.is_available():
                    source = src
                    break
        
        if source is None:
            # Use best available source
            source = await self.registry.get_best_source()
        
        if source is None:
            # Fallback to system clock
            source = self.registry.get_source("system")
        
        if source is None:
            raise RuntimeError("No time sources available")
        
        try:
            timestamp = await source.get_time(timezone_name)
            
            # Update drift monitoring
            await self._drift_monitor.record_sample(source.name, timestamp)
            
            return timestamp
        except Exception as e:
            self.logger.error(f"Error getting time from {source.name}: {e}")
            
            # Fallback to system clock if primary fails
            if source.name != "system":
                system_source = self.registry.get_source("system")
                if system_source:
                    return await system_source.get_time(timezone_name)
            
            raise RuntimeError(f"Failed to get time: {e}")
    
    async def get_sync_status(self) -> Dict[str, SyncStatus]:
        """Get synchronization status for all sources."""
        status = {}
        
        for name, source in self.registry.sources.items():
            try:
                status[name] = await source.get_sync_status()
            except Exception as e:
                self.logger.error(f"Error getting sync status for {name}: {e}")
                status[name] = SyncStatus(
                    is_synchronized=False,
                    source=source.source_type,
                    quality=ClockQuality.UNKNOWN
                )
        
        return status
    
    async def get_overall_sync_status(self) -> SyncStatus:
        """Get overall synchronization status."""
        all_status = await self.get_sync_status()
        
        # Find best synchronized source
        best_quality = ClockQuality.UNKNOWN
        best_source = ClockSource.SYSTEM
        synchronized_count = 0
        total_count = len(all_status)
        
        for status in all_status.values():
            if status.is_synchronized:
                synchronized_count += 1
                
                quality_order = {
                    ClockQuality.REFERENCE: 5,
                    ClockQuality.EXCELLENT: 4,
                    ClockQuality.GOOD: 3,
                    ClockQuality.FAIR: 2,
                    ClockQuality.POOR: 1,
                    ClockQuality.UNKNOWN: 0
                }
                
                if quality_order.get(status.quality, 0) > quality_order.get(best_quality, 0):
                    best_quality = status.quality
                    best_source = status.source
        
        return SyncStatus(
            is_synchronized=synchronized_count > 0,
            source=best_source,
            quality=best_quality,
            reachable_sources=synchronized_count,
            total_sources=total_count
        )
    
    async def convert_time(self, request: TimeConversionRequest) -> TimeConversionResult:
        """Convert time between timezones with high precision."""
        if request.timestamp is None:
            source_timestamp = await self.get_precision_time(request.source_timezone)
        else:
            source_timestamp = request.timestamp
        
        # Target timezone will be calculated during conversion
        
        # Calculate the actual conversion by applying timezone offset
        # This is a simplified implementation - a full implementation would
        # handle DST transitions, leap seconds, etc.
        
        from zoneinfo import ZoneInfo
        from datetime import datetime, timezone as dt_timezone
        
        # Convert source timestamp to UTC datetime
        source_dt = datetime.fromtimestamp(
            source_timestamp.utc_seconds + source_timestamp.utc_nanoseconds / 1e9,
            tz=dt_timezone.utc
        )
        
        # Convert to target timezone
        if request.target_timezone == "UTC":
            target_tz = dt_timezone.utc
        else:
            target_tz = ZoneInfo(request.target_timezone)
        
        target_dt = source_dt.astimezone(target_tz)
        
        # Calculate offset
        source_offset = source_dt.utcoffset() or timedelta()
        target_offset = target_dt.utcoffset() or timedelta()
        offset_seconds = (target_offset - source_offset).total_seconds()
        
        # Create target timestamp
        target_utc_seconds = int(target_dt.timestamp())
        target_utc_nanoseconds = source_timestamp.utc_nanoseconds  # Preserve nanoseconds
        
        converted_target = PrecisionTimestamp(
            utc_seconds=target_utc_seconds,
            utc_nanoseconds=target_utc_nanoseconds,
            timezone=request.target_timezone,
            is_dst=bool(target_dt.dst()) if target_dt.dst() is not None else False,
            monotonic_ns=source_timestamp.monotonic_ns,
            source=source_timestamp.source,
            quality=source_timestamp.quality,
            offset_ns=source_timestamp.offset_ns,
            drift_rate=source_timestamp.drift_rate,
            uncertainty_ns=source_timestamp.uncertainty_ns
        )
        
        return TimeConversionResult(
            source=source_timestamp,
            target=converted_target,
            offset_seconds=offset_seconds,
            conversion_metadata={
                "preserve_precision": request.preserve_precision,
                "source_dst": source_timestamp.is_dst,
                "target_dst": converted_target.is_dst
            }
        )
    
    def create_vector_clock(self) -> VectorClock:
        """Create a new vector clock for event ordering."""
        if self._vector_clock is None:
            self._vector_clock = VectorClock(
                node_id=self.node_id,
                clocks={self.node_id: 0},
                timestamp=PrecisionTimestamp.now()
            )
        
        self._vector_clock = self._vector_clock.increment()
        return self._vector_clock
    
    def update_vector_clock(self, received_clock: VectorClock) -> VectorClock:
        """Update vector clock based on received clock."""
        if self._vector_clock is None:
            self._vector_clock = VectorClock(
                node_id=self.node_id,
                clocks={self.node_id: 0},
                timestamp=PrecisionTimestamp.now()
            )
        
        self._vector_clock = self._vector_clock.update(received_clock)
        return self._vector_clock
    
    def compare_timestamps(self, ts1: PrecisionTimestamp, ts2: PrecisionTimestamp) -> Dict[str, Any]:
        """Compare two timestamps for causality and ordering."""
        # Physical time comparison
        ts1_total_ns = ts1.utc_seconds * 1_000_000_000 + ts1.utc_nanoseconds
        ts2_total_ns = ts2.utc_seconds * 1_000_000_000 + ts2.utc_nanoseconds
        
        time_diff_ns = ts2_total_ns - ts1_total_ns
        
        # Monotonic clock comparison (for local ordering)
        monotonic_diff_ns = ts2.monotonic_ns - ts1.monotonic_ns
        
        result = {
            "time_difference_ns": time_diff_ns,
            "monotonic_difference_ns": monotonic_diff_ns,
            "ts1_before_ts2": time_diff_ns > 0,
            "physical_ordering": "ts1_before_ts2" if time_diff_ns > 0 else "ts2_before_ts1" if time_diff_ns < 0 else "simultaneous",
            "uncertainty_overlap": False
        }
        
        # Check if uncertainties overlap (indicating potential simultaneity)
        if ts1.uncertainty_ns and ts2.uncertainty_ns:
            total_uncertainty = ts1.uncertainty_ns + ts2.uncertainty_ns
            result["uncertainty_overlap"] = abs(time_diff_ns) <= total_uncertainty
        
        return result
    
    async def measure_clock_drift(self, duration_seconds: int = 60) -> Dict[str, Any]:
        """Measure clock drift over a specified duration."""
        return await self._drift_monitor.measure_drift(duration_seconds)
    


class DriftMonitor:
    """Monitor clock drift over time."""
    
    def __init__(self, max_samples: int = 1000):
        self.max_samples = max_samples
        self.samples: Dict[str, List[Dict[str, Any]]] = {}
    
    async def record_sample(self, source_name: str, timestamp: PrecisionTimestamp) -> None:
        """Record a time sample for drift analysis."""
        if source_name not in self.samples:
            self.samples[source_name] = []
        
        sample = {
            "timestamp": timestamp,
            "system_time_ns": timestamp.utc_seconds * 1_000_000_000 + timestamp.utc_nanoseconds,
            "monotonic_ns": timestamp.monotonic_ns,
            "recorded_at": datetime.now()
        }
        
        self.samples[source_name].append(sample)
        
        # Keep only recent samples
        if len(self.samples[source_name]) > self.max_samples:
            self.samples[source_name] = self.samples[source_name][-self.max_samples:]
    
    async def measure_drift(self, duration_seconds: int) -> Dict[str, Any]:
        """Measure drift over a specific duration."""
        initial_samples = {}
        
        # Record initial samples
        for source_name in self.samples:
            if self.samples[source_name]:
                initial_samples[source_name] = self.samples[source_name][-1]
        
        # Wait for duration
        await asyncio.sleep(duration_seconds)
        
        # Record final samples and calculate drift
        drift_results = {}
        
        for source_name, initial_sample in initial_samples.items():
            if source_name in self.samples and self.samples[source_name]:
                final_sample = self.samples[source_name][-1]
                
                # Calculate drift rate
                time_elapsed = (final_sample["recorded_at"] - initial_sample["recorded_at"]).total_seconds()
                system_time_diff = final_sample["system_time_ns"] - initial_sample["system_time_ns"]
                monotonic_diff = final_sample["monotonic_ns"] - initial_sample["monotonic_ns"]
                
                if time_elapsed > 0:
                    # Drift rate in parts per million (ppm)
                    drift_ppm = ((system_time_diff - monotonic_diff) / monotonic_diff) * 1_000_000
                    
                    drift_results[source_name] = {
                        "drift_rate_ppm": drift_ppm,
                        "time_elapsed_seconds": time_elapsed,
                        "absolute_drift_ns": abs(system_time_diff - monotonic_diff),
                        "quality": "good" if abs(drift_ppm) < 1 else "fair" if abs(drift_ppm) < 10 else "poor"
                    }
        
        return drift_results