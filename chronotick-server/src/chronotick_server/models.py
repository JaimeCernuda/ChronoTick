"""Enhanced time models with nanosecond precision for ChronoTick."""

from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any
import time
from pydantic import BaseModel, Field


class ClockSource(str, Enum):
    """Available time sources for synchronization."""
    SYSTEM = "system"
    NTP = "ntp"


class ClockQuality(str, Enum):
    """Clock synchronization quality levels."""
    UNKNOWN = "unknown"
    POOR = "poor"          # >1s drift
    FAIR = "fair"          # 100ms-1s drift  
    GOOD = "good"          # 10ms-100ms drift
    EXCELLENT = "excellent" # <10ms drift
    REFERENCE = "reference" # Reference clock (stratum 0/1)


class PrecisionTimestamp(BaseModel):
    """High-precision timestamp with nanosecond accuracy."""
    
    utc_seconds: int = Field(description="UTC seconds since epoch")
    utc_nanoseconds: int = Field(description="Nanoseconds within the second", ge=0, le=999_999_999)
    timezone: str = Field(description="IANA timezone identifier")
    is_dst: bool = Field(description="Whether daylight saving time is active")
    monotonic_ns: int = Field(description="Monotonic clock nanoseconds for ordering")
    
    # Clock quality metadata
    source: ClockSource = Field(description="Time source used")
    quality: ClockQuality = Field(description="Synchronization quality")
    offset_ns: Optional[int] = Field(None, description="Offset from reference in nanoseconds")
    drift_rate: Optional[float] = Field(None, description="Clock drift rate in ppm")
    uncertainty_ns: Optional[int] = Field(None, description="Time uncertainty in nanoseconds")
    
    @classmethod
    def now(cls, timezone: str = "UTC", source: ClockSource = ClockSource.SYSTEM) -> "PrecisionTimestamp":
        """Create a precision timestamp for the current time."""
        now = datetime.now()
        monotonic_ns = time.time_ns()
        
        return cls(
            utc_seconds=int(time.time()),
            utc_nanoseconds=monotonic_ns % 1_000_000_000,
            timezone=timezone,
            is_dst=bool(now.dst()) if now.dst() is not None else False,
            monotonic_ns=monotonic_ns,
            source=source,
            quality=ClockQuality.UNKNOWN
        )
    
    def to_datetime(self) -> datetime:
        """Convert to standard datetime object."""
        return datetime.fromtimestamp(self.utc_seconds + self.utc_nanoseconds / 1e9)
    
    def to_iso_string(self, precision: str = "nanoseconds") -> str:
        """Convert to ISO string with specified precision."""
        dt = self.to_datetime()
        if precision == "nanoseconds":
            # Format with nanosecond precision
            return f"{dt.isoformat()}.{self.utc_nanoseconds:09d}Z"
        elif precision == "microseconds":
            return f"{dt.isoformat()}.{self.utc_nanoseconds // 1000:06d}Z"
        elif precision == "milliseconds":
            return f"{dt.isoformat()}.{self.utc_nanoseconds // 1000000:03d}Z"
        else:
            return dt.isoformat() + "Z"


class SyncStatus(BaseModel):
    """Clock synchronization status and health metrics."""
    
    is_synchronized: bool = Field(description="Whether clock is synchronized")
    source: ClockSource = Field(description="Primary time source")
    quality: ClockQuality = Field(description="Overall synchronization quality")
    
    # NTP-specific metrics
    ntp_stratum: Optional[int] = Field(None, description="NTP stratum level")
    ntp_offset_ms: Optional[float] = Field(None, description="NTP offset in milliseconds")
    ntp_jitter_ms: Optional[float] = Field(None, description="NTP jitter in milliseconds")
    ntp_delay_ms: Optional[float] = Field(None, description="NTP round-trip delay in milliseconds")
    ntp_servers: Optional[list[str]] = Field(None, description="List of NTP servers")
    
    # PTP-specific metrics
    ptp_master_id: Optional[str] = Field(None, description="PTP master clock ID")
    ptp_offset_ns: Optional[int] = Field(None, description="PTP offset in nanoseconds")
    ptp_path_delay_ns: Optional[int] = Field(None, description="PTP path delay in nanoseconds")
    ptp_domain: Optional[int] = Field(None, description="PTP domain number")
    
    # General metrics
    drift_rate_ppm: Optional[float] = Field(None, description="Clock drift rate in parts per million")
    last_sync_time: Optional[PrecisionTimestamp] = Field(None, description="Last successful synchronization")
    sync_interval_seconds: Optional[int] = Field(None, description="Synchronization interval")
    
    # Health indicators
    reachable_sources: int = Field(0, description="Number of reachable time sources")
    total_sources: int = Field(0, description="Total configured time sources")
    uptime_seconds: Optional[int] = Field(None, description="Time service uptime")


class VectorClock(BaseModel):
    """Vector clock for distributed event ordering."""
    
    node_id: str = Field(description="Unique identifier for this node")
    clocks: Dict[str, int] = Field(description="Clock values for each known node")
    timestamp: PrecisionTimestamp = Field(description="Physical timestamp when created")
    
    def increment(self) -> "VectorClock":
        """Increment this node's clock value."""
        new_clocks = self.clocks.copy()
        new_clocks[self.node_id] = new_clocks.get(self.node_id, 0) + 1
        return VectorClock(
            node_id=self.node_id,
            clocks=new_clocks,
            timestamp=PrecisionTimestamp.now()
        )
    
    def update(self, other: "VectorClock") -> "VectorClock":
        """Update vector clock based on received vector clock."""
        new_clocks = {}
        all_nodes = set(self.clocks.keys()) | set(other.clocks.keys())
        
        for node in all_nodes:
            self_val = self.clocks.get(node, 0)
            other_val = other.clocks.get(node, 0)
            new_clocks[node] = max(self_val, other_val)
        
        # Increment our own clock
        new_clocks[self.node_id] = new_clocks.get(self.node_id, 0) + 1
        
        return VectorClock(
            node_id=self.node_id,
            clocks=new_clocks,
            timestamp=PrecisionTimestamp.now()
        )
    
    def compare(self, other: "VectorClock") -> str:
        """Compare vector clocks to determine causal relationship."""
        self_dominates = False
        other_dominates = False
        
        all_nodes = set(self.clocks.keys()) | set(other.clocks.keys())
        
        for node in all_nodes:
            self_val = self.clocks.get(node, 0)
            other_val = other.clocks.get(node, 0)
            
            if self_val > other_val:
                self_dominates = True
            elif other_val > self_val:
                other_dominates = True
        
        if self_dominates and not other_dominates:
            return "happens_after"
        elif other_dominates and not self_dominates:
            return "happens_before"
        elif not self_dominates and not other_dominates:
            return "concurrent"
        else:
            return "concurrent"  # Both dominate in different dimensions


class TimeConversionRequest(BaseModel):
    """Request for high-precision time conversion."""
    
    source_timezone: str = Field(description="Source IANA timezone")
    target_timezone: str = Field(description="Target IANA timezone")
    timestamp: Optional[PrecisionTimestamp] = Field(None, description="Timestamp to convert (current time if None)")
    preserve_precision: bool = Field(True, description="Whether to preserve nanosecond precision")


class TimeConversionResult(BaseModel):
    """Result of high-precision time conversion."""
    
    source: PrecisionTimestamp = Field(description="Source timestamp")
    target: PrecisionTimestamp = Field(description="Converted timestamp")
    offset_seconds: float = Field(description="Timezone offset in seconds")
    conversion_metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional conversion metadata")


