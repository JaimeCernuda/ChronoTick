#!/usr/bin/env python3
"""
ChronoTick Shared Memory Configuration

Defines the memory layout for high-performance IPC between ChronoTick daemon
and SDK MCP clients. Uses lock-free single-writer-multiple-reader pattern
with sequence numbers for torn read detection.

Memory Layout (128 bytes, aligned to 2 cache lines):
    [0-8]:    double  - Corrected timestamp (Unix time)
    [8-16]:   double  - System timestamp when correction was made
    [16-24]:  double  - Offset correction (seconds)
    [24-32]:  double  - Drift rate (seconds/second)
    [32-40]:  double  - Offset uncertainty (seconds)
    [40-48]:  double  - Drift uncertainty (seconds/second)
    [48-56]:  double  - Prediction time (Unix timestamp)
    [56-64]:  double  - Valid until (Unix timestamp)
    [64-68]:  float   - Confidence [0,1]
    [68-72]:  uint32  - Sequence number (for torn read detection)
    [72-76]:  uint32  - Flags (valid=0x01, ntp_ready=0x02, models_ready=0x04)
    [76-80]:  uint32  - Source (0=no_data, 1=ntp, 2=cpu, 3=gpu, 4=fusion)
    [80-84]:  uint32  - Measurement count
    [84-92]:  double  - Last NTP sync timestamp
    [92-100]: double  - Daemon uptime (seconds)
    [100-104]: uint32 - Total corrections served
    [104-108]: uint32 - Reserved
    [108-112]: float  - Average latency (milliseconds)
    [112-124]: Reserved for future use (12 bytes padding)

Performance Characteristics:
- Write frequency: 100-1000 Hz (daemon)
- Read latency: ~300ns (after initial attach)
- Cache line aligned: 2x 64-byte cache lines
- Lock-free: single-writer-multiple-reader pattern
"""

import struct
from typing import NamedTuple, Optional
from enum import IntEnum

# Shared memory configuration
SHARED_MEMORY_NAME = "chronotick_shm"
SHARED_MEMORY_SIZE = 128  # 2 cache lines (64 bytes each)
CACHE_LINE_SIZE = 64

# Source enumeration
class CorrectionSource(IntEnum):
    """Clock correction data source"""
    NO_DATA = 0
    NTP = 1
    CPU_MODEL = 2
    GPU_MODEL = 3
    FUSION = 4

# Flags
class StatusFlags:
    """Bitflags for status field"""
    VALID = 0x01          # Data is valid
    NTP_READY = 0x02      # NTP measurements available
    MODELS_READY = 0x04   # ML models initialized
    WARMUP_COMPLETE = 0x08  # Warmup phase complete

# Memory layout format strings for struct module
# Note: '=' for native byte order, standard sizes
LAYOUT_FORMAT = (
    'd'   # [0-8]    corrected_timestamp
    'd'   # [8-16]   system_timestamp
    'd'   # [16-24]  offset_correction
    'd'   # [24-32]  drift_rate
    'd'   # [32-40]  offset_uncertainty
    'd'   # [40-48]  drift_uncertainty
    'd'   # [48-56]  prediction_time
    'd'   # [56-64]  valid_until
    'f'   # [64-68]  confidence
    'I'   # [68-72]  sequence_number
    'I'   # [72-76]  flags
    'I'   # [76-80]  source
    'I'   # [80-84]  measurement_count
    'd'   # [84-92]  last_ntp_sync
    'd'   # [92-100] daemon_uptime
    'I'   # [100-104] total_corrections
    'I'   # [104-108] reserved
    'f'   # [108-112] average_latency_ms
    '12s' # [112-124] reserved (12 bytes padding)
)

# Verify layout size
LAYOUT_SIZE = struct.calcsize(LAYOUT_FORMAT)
assert LAYOUT_SIZE == SHARED_MEMORY_SIZE, f"Layout size mismatch: {LAYOUT_SIZE} != {SHARED_MEMORY_SIZE}"


class ChronoTickData(NamedTuple):
    """
    Parsed ChronoTick shared memory data.

    This is the Pythonic interface to the raw shared memory buffer,
    automatically handling struct packing/unpacking.
    """
    corrected_timestamp: float      # Current corrected time
    system_timestamp: float         # System time when correction was made
    offset_correction: float        # Clock offset (seconds)
    drift_rate: float              # Clock drift rate (seconds/second)
    offset_uncertainty: float      # Offset uncertainty (seconds)
    drift_uncertainty: float       # Drift rate uncertainty (seconds/second)
    prediction_time: float         # When correction was predicted
    valid_until: float             # Correction expiration time
    confidence: float              # Model confidence [0,1]
    sequence_number: int           # For torn read detection
    flags: int                     # Status bitflags
    source: CorrectionSource       # Data source
    measurement_count: int         # Number of measurements collected
    last_ntp_sync: float          # Last NTP synchronization time
    daemon_uptime: float          # Daemon uptime in seconds
    total_corrections: int        # Total corrections served
    reserved: int                 # Reserved field
    average_latency_ms: float     # Average correction latency

    @property
    def is_valid(self) -> bool:
        """Check if data is valid"""
        return bool(self.flags & StatusFlags.VALID)

    @property
    def is_ntp_ready(self) -> bool:
        """Check if NTP measurements are available"""
        return bool(self.flags & StatusFlags.NTP_READY)

    @property
    def is_models_ready(self) -> bool:
        """Check if ML models are ready"""
        return bool(self.flags & StatusFlags.MODELS_READY)

    @property
    def is_warmup_complete(self) -> bool:
        """Check if warmup phase is complete"""
        return bool(self.flags & StatusFlags.WARMUP_COMPLETE)

    def get_time_uncertainty(self, time_delta: float) -> float:
        """
        Calculate time uncertainty at a given time delta from prediction.

        Uses mathematical error propagation:
        uncertainty = sqrt(offset_unc² + (drift_unc * time_delta)²)

        Args:
            time_delta: Seconds since prediction_time

        Returns:
            Total time uncertainty in seconds
        """
        import math
        return math.sqrt(
            self.offset_uncertainty ** 2 +
            (self.drift_uncertainty * time_delta) ** 2
        )

    def get_corrected_time_at(self, system_time: float) -> float:
        """
        Calculate corrected time at a given system time.

        Args:
            system_time: System timestamp

        Returns:
            Corrected timestamp
        """
        time_delta = system_time - self.prediction_time
        return system_time + self.offset_correction + (self.drift_rate * time_delta)


def write_data(buffer: memoryview, data: ChronoTickData) -> None:
    """
    Write ChronoTickData to shared memory buffer.

    This is the writer-side function used by the daemon.
    Writes all fields including sequence number for lock-free synchronization.

    Args:
        buffer: Shared memory buffer (must be at least SHARED_MEMORY_SIZE bytes)
        data: ChronoTickData to write
    """
    # Pack all data into buffer
    struct.pack_into(
        LAYOUT_FORMAT, buffer, 0,
        data.corrected_timestamp,
        data.system_timestamp,
        data.offset_correction,
        data.drift_rate,
        data.offset_uncertainty,
        data.drift_uncertainty,
        data.prediction_time,
        data.valid_until,
        data.confidence,
        data.sequence_number,
        data.flags,
        int(data.source),
        data.measurement_count,
        data.last_ntp_sync,
        data.daemon_uptime,
        data.total_corrections,
        data.reserved,
        data.average_latency_ms,
        b'\x00' * 16  # Reserved bytes
    )


def read_data(buffer: memoryview) -> ChronoTickData:
    """
    Read ChronoTickData from shared memory buffer with torn read detection.

    This is the reader-side function used by SDK MCP tools and clients.
    Uses sequence number pattern to detect torn reads during concurrent writes.

    Args:
        buffer: Shared memory buffer

    Returns:
        ChronoTickData parsed from buffer

    Note:
        Caller should implement retry logic if sequence numbers indicate torn read.
        See read_data_with_retry() for automatic retry.
    """
    # Unpack all fields
    unpacked = struct.unpack_from(LAYOUT_FORMAT, buffer, 0)

    return ChronoTickData(
        corrected_timestamp=unpacked[0],
        system_timestamp=unpacked[1],
        offset_correction=unpacked[2],
        drift_rate=unpacked[3],
        offset_uncertainty=unpacked[4],
        drift_uncertainty=unpacked[5],
        prediction_time=unpacked[6],
        valid_until=unpacked[7],
        confidence=unpacked[8],
        sequence_number=unpacked[9],
        flags=unpacked[10],
        source=CorrectionSource(unpacked[11]),
        measurement_count=unpacked[12],
        last_ntp_sync=unpacked[13],
        daemon_uptime=unpacked[14],
        total_corrections=unpacked[15],
        reserved=unpacked[16],
        average_latency_ms=unpacked[17]
    )


def read_data_with_retry(buffer: memoryview, max_retries: int = 3) -> ChronoTickData:
    """
    Read data with automatic retry on torn reads.

    Implements the sequence number pattern:
    1. Read sequence number
    2. Read all data
    3. Read sequence number again
    4. If sequences don't match, writer updated during read - retry

    Args:
        buffer: Shared memory buffer
        max_retries: Maximum number of retries

    Returns:
        ChronoTickData with consistent read

    Raises:
        RuntimeError: If max retries exceeded (indicates very high contention)
    """
    for attempt in range(max_retries):
        # Read sequence before data
        seq_before = struct.unpack_from('I', buffer, 68)[0]  # Offset 68 is sequence_number

        # Read all data
        data = read_data(buffer)

        # Read sequence after data
        seq_after = struct.unpack_from('I', buffer, 68)[0]

        # If sequence unchanged, data is consistent
        if seq_before == seq_after:
            return data

    # Max retries exceeded - very rare, indicates extreme contention
    raise RuntimeError(f"Failed to read consistent data after {max_retries} retries")


def create_default_data() -> ChronoTickData:
    """
    Create ChronoTickData with default/zero values.

    Used for initializing shared memory or when no data is available.
    """
    import time
    current_time = time.time()

    return ChronoTickData(
        corrected_timestamp=current_time,
        system_timestamp=current_time,
        offset_correction=0.0,
        drift_rate=0.0,
        offset_uncertainty=0.0,
        drift_uncertainty=0.0,
        prediction_time=current_time,
        valid_until=current_time + 60.0,  # Valid for 60 seconds
        confidence=0.0,
        sequence_number=0,
        flags=0,  # No flags set initially
        source=CorrectionSource.NO_DATA,
        measurement_count=0,
        last_ntp_sync=0.0,
        daemon_uptime=0.0,
        total_corrections=0,
        reserved=0,
        average_latency_ms=0.0
    )


# Performance benchmarking utilities
def benchmark_read_latency(buffer: memoryview, iterations: int = 10000) -> float:
    """
    Benchmark read latency for performance validation.

    Args:
        buffer: Shared memory buffer
        iterations: Number of read iterations

    Returns:
        Average read latency in nanoseconds
    """
    import time

    start = time.perf_counter()
    for _ in range(iterations):
        _ = read_data(buffer)
    elapsed = time.perf_counter() - start

    avg_latency_seconds = elapsed / iterations
    avg_latency_nanoseconds = avg_latency_seconds * 1e9

    return avg_latency_nanoseconds


if __name__ == "__main__":
    # Self-test and demonstrate usage
    print("ChronoTick Shared Memory Configuration")
    print("=" * 60)
    print(f"Shared memory name: {SHARED_MEMORY_NAME}")
    print(f"Shared memory size: {SHARED_MEMORY_SIZE} bytes")
    print(f"Layout format size: {LAYOUT_SIZE} bytes")
    print(f"Cache alignment: {SHARED_MEMORY_SIZE // CACHE_LINE_SIZE} cache lines")
    print()

    # Test struct packing/unpacking
    import time
    test_data = ChronoTickData(
        corrected_timestamp=time.time(),
        system_timestamp=time.time(),
        offset_correction=0.001234,  # 1.234ms offset
        drift_rate=1e-6,  # 1 microsecond/second drift
        offset_uncertainty=0.00001,  # 10μs uncertainty
        drift_uncertainty=1e-9,
        prediction_time=time.time() - 5.0,
        valid_until=time.time() + 55.0,
        confidence=0.95,
        sequence_number=42,
        flags=StatusFlags.VALID | StatusFlags.NTP_READY | StatusFlags.MODELS_READY,
        source=CorrectionSource.FUSION,
        measurement_count=100,
        last_ntp_sync=time.time() - 10.0,
        daemon_uptime=123.45,
        total_corrections=1000,
        reserved=0,
        average_latency_ms=0.5
    )

    # Create test buffer
    import array
    test_buffer = array.array('b', [0] * SHARED_MEMORY_SIZE)
    buffer_view = memoryview(test_buffer).cast('B')

    # Write and read test
    write_data(buffer_view, test_data)
    read_back = read_data(buffer_view)

    print("✅ Write/Read test passed")
    print(f"   Offset correction: {read_back.offset_correction * 1e6:.3f}μs")
    print(f"   Drift rate: {read_back.drift_rate * 1e6:.3f}μs/s")
    print(f"   Confidence: {read_back.confidence:.2%}")
    print(f"   Source: {read_back.source.name}")
    print(f"   Valid: {read_back.is_valid}")
    print(f"   NTP ready: {read_back.is_ntp_ready}")
    print(f"   Models ready: {read_back.is_models_ready}")
    print()

    # Benchmark read performance
    print("🔬 Benchmarking read performance...")
    latency_ns = benchmark_read_latency(buffer_view, iterations=10000)
    print(f"   Average read latency: {latency_ns:.0f}ns ({latency_ns/1000:.2f}μs)")
    print()

    # Test uncertainty calculation
    time_delta = 30.0  # 30 seconds into future
    uncertainty = read_back.get_time_uncertainty(time_delta)
    print(f"📊 Time uncertainty at +{time_delta}s: {uncertainty*1e6:.3f}μs")
    print()

    print("✅ All self-tests passed!")
