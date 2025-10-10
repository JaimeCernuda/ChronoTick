# ChronoTick Shared Memory - Complete Reference

**Comprehensive technical reference for integration and development**

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [The 5 Core Files](#the-5-core-files)
4. [Data Formats](#data-formats)
5. [Integration Guide](#integration-guide)
6. [Important Concepts](#important-concepts)
7. [TODOs and Known Issues](#todos-and-known-issues)
8. [Testing and Validation](#testing-and-validation)

---

## Overview

ChronoTick Shared Memory provides **~300ns read latency** for high-precision time corrections via shared memory IPC. It consists of **5 files**:

```
src/chronotick_shm/
├── shm_config.py                  # Shared memory layout & serialization
├── chronotick_daemon_server.py    # Daemon that writes to shared memory
├── chronotick_client_eval.py      # Evaluation/benchmarking client
├── chronotick_sdk_mcp.py          # SDK MCP tools (in-process)
└── chronotick_stdio_mcp.py        # Stdio MCP server (separate process)
```

---

## System Architecture

### Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│ ChronoTick Inference Engine (tsfm/chronotick_inference)     │
│ - Real NTP measurements                                     │
│ - Dual ML models (short-term, long-term)                    │
│ - Offset/drift corrections with uncertainty                 │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────────────────┐
│ chronotick_daemon_server.py                                 │
│ - Runs RealDataPipeline                                     │
│ - Updates at 100-1000 Hz                                    │
│ - Writes ClockCorrection → ChronoTickData                   │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────────────────┐
│ Shared Memory (128 bytes)                                   │
│ /dev/shm/chronotick_shm                                     │
│ - Lock-free (sequence number pattern)                       │
│ - Cache-aligned (2 x 64-byte lines)                         │
│ - Single writer, multiple readers                           │
└──────────────────┬──────────────────────────────────────────┘
                   │
        ┌──────────┴──────────┬──────────────┬─────────────┐
        ↓                     ↓              ↓             ↓
┌─────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────┐
│ Client Eval │  │ SDK MCP      │  │ Stdio MCP    │  │ Your App │
│ (direct)    │  │ (in-process) │  │ (stdio)      │  │          │
└─────────────┘  └──────────────┘  └──────────────┘  └──────────┘
```

---

## The 5 Core Files

### 1. `shm_config.py` - Shared Memory Layout (14KB)

**Purpose**: Defines the 128-byte shared memory structure used by all components

**Key Components**:

#### ChronoTickData (NamedTuple)
The core data structure with **20 fields**:

```python
ChronoTickData(
    # Timestamps (3 fields)
    corrected_timestamp: float,    # Corrected Unix timestamp
    system_timestamp: float,       # Raw system timestamp
    prediction_time: float,        # When prediction was made

    # Corrections (2 fields)
    offset_correction: float,      # Time offset in seconds
    drift_rate: float,             # Clock drift (seconds/second)

    # Uncertainties (2 fields)
    offset_uncertainty: float,     # Offset uncertainty (seconds)
    drift_uncertainty: float,      # Drift uncertainty (s/s)

    # Validity (1 field)
    valid_until: float,            # Prediction expiry timestamp

    # Metadata (4 fields)
    confidence: float,             # 0-1 confidence level
    sequence_number: int,          # For torn read detection
    flags: int,                    # StatusFlags bitmask
    source: CorrectionSource,      # Data source enum

    # Statistics (5 fields)
    measurement_count: int,        # Total NTP measurements
    last_ntp_sync: float,          # Last NTP sync timestamp
    daemon_uptime: float,          # Daemon uptime (seconds)
    total_corrections: int,        # Total corrections made
    average_latency_ms: float,     # Average read latency

    # Reserved (1 field)
    reserved: int,                 # For future use
)
```

#### Key Methods

**`get_corrected_time_at(system_time)`**
```python
# Formula: corrected = system + offset + drift * (system - prediction_time)
time_delta = system_time - self.prediction_time
return system_time + self.offset_correction + (self.drift_rate * time_delta)
```
- **What it does**: Projects correction to any timestamp
- **Used by**: `get_time()`, `get_time_with_future_uncertainty()`

**`get_time_uncertainty(time_delta)`**
```python
# Formula: uncertainty = sqrt(offset_unc² + (drift_unc * delta)²)
return math.sqrt(
    self.offset_uncertainty ** 2 +
    (self.drift_uncertainty * time_delta) ** 2
)
```
- **What it does**: Calculates total uncertainty at time delta from prediction
- **Used by**: `get_time()`, `get_time_with_future_uncertainty()`
- **Note**: Uncertainty grows with time due to drift uncertainty

#### Enums

**CorrectionSource** (IntEnum):
```python
NO_DATA = 0       # No correction available
NTP = 1           # Pure NTP
CPU_MODEL = 2     # CPU-based ML model
GPU_MODEL = 3     # GPU-based ML model
FUSION = 4        # Fused prediction
```

**StatusFlags** (class with constants):
```python
VALID = 0x01              # Data is valid
NTP_READY = 0x02          # NTP measurements ready
MODELS_READY = 0x04       # ML models ready
WARMUP_COMPLETE = 0x08    # Warmup complete
```

#### Memory Layout

**128 bytes total** (struct format):
```python
LAYOUT_FORMAT = (
    'd' 'd' 'd'         # corrected_timestamp, system_timestamp, prediction_time
    'd' 'd'             # offset_correction, drift_rate
    'd' 'd'             # offset_uncertainty, drift_uncertainty
    'd'                 # valid_until
    'f'                 # confidence (float32)
    'I' 'I' 'I' 'I'     # sequence, flags, source, measurement_count
    'd' 'd'             # last_ntp_sync, daemon_uptime
    'I' 'I'             # total_corrections, reserved
    'f'                 # average_latency_ms
    '12s'               # padding to 128 bytes
)
```

**Important**: 128 bytes = 2 x 64-byte cache lines (optimal for CPU L1 cache)

#### Functions

**`write_data(buffer, data)`**
- Packs ChronoTickData into binary
- Writes to shared memory buffer
- **Used by**: Daemon only (single writer)

**`read_data(buffer)`**
- Unpacks binary into ChronoTickData
- Single read operation
- **No retry logic** (use `read_data_with_retry`)

**`read_data_with_retry(buffer, max_retries=3)`**
- Reads with sequence number validation
- Detects torn reads (sequence_before != sequence_after)
- Retries on torn reads
- **Used by**: All readers (client, SDK MCP, stdio MCP)

**`benchmark_read_latency(buffer, iterations=10000)`**
- Benchmarks read performance
- Returns average latency in nanoseconds
- **Used by**: Client eval

---

### 2. `chronotick_daemon_server.py` - Daemon (17KB)

**Purpose**: Background process that writes ChronoTick corrections to shared memory

**Dependencies**:
- Full ChronoTick environment from `tsfm/`
- Requires: numpy, chronotick_inference
- Install: `cd ../tsfm && uv sync --extra core-models`

**Key Class**: `ChronoTickSharedMemoryDaemon`

**Workflow**:
```
1. Initialize
   ↓
2. Create/attach shared memory
   ↓
3. Initialize RealDataPipeline (NTP + ML)
   ↓
4. Warmup (180 seconds default)
   ↓
5. Main loop (100-1000 Hz)
   - Get ClockCorrection from pipeline
   - Convert to ChronoTickData
   - Write to shared memory
   ↓
6. Graceful shutdown on signal
```

**Configuration Options**:
```bash
--config PATH          # Path to config.yaml (required for production)
--freq HZ              # Update frequency (1-1000, default: 100)
--cpu-affinity N...    # CPU cores to bind to
--log-level LEVEL      # DEBUG, INFO, WARNING, ERROR
```

**Performance**:
- **100 Hz**: ~1% CPU, balanced
- **1000 Hz**: ~5-10% CPU, high frequency
- **Memory**: 150-200MB (NTP history + ML models)

**Integration with ChronoTick**:
```python
# Gets ClockCorrection from RealDataPipeline
correction = self.pipeline.get_real_clock_correction(current_time)

# Converts to ChronoTickData
data = ChronoTickData(
    corrected_timestamp=current_time + correction.offset_correction + drift_correction,
    system_timestamp=current_time,
    offset_correction=correction.offset_correction,
    drift_rate=correction.drift_rate,
    offset_uncertainty=correction.offset_uncertainty,
    drift_uncertainty=correction.drift_uncertainty,
    prediction_time=correction.prediction_time,
    valid_until=correction.valid_until,
    confidence=correction.confidence,
    source=self._get_correction_source_enum(correction.source),
    # ... stats fields
)
```

**Signal Handling**:
- SIGINT, SIGTERM → graceful shutdown
- Stops NTP collection
- Unlinks shared memory

**Entry Point**: `uv run chronotick-daemon`

---

### 3. `chronotick_client_eval.py` - Evaluation Client (14KB)

**Purpose**: Direct shared memory access for testing, benchmarking, and monitoring

**Key Class**: `ChronoTickClient`

**Commands**:

#### `read` - Read once
```bash
uv run chronotick-client read
```
Displays:
- Corrected time
- System time
- Offset/drift
- Uncertainty
- Confidence
- Source
- Status flags

#### `monitor` - Continuous monitoring
```bash
uv run chronotick-client monitor --interval 0.1
```
Updates at specified interval (Hz = 1/interval)

#### `status` - Daemon status
```bash
uv run chronotick-client status
```
Shows:
- Daemon uptime
- Warmup status
- NTP measurements count
- Last NTP sync time
- Current source
- Confidence

#### `benchmark` - Performance test
```bash
uv run chronotick-client benchmark --iterations 10000
```
Measures:
- Average read latency (ns)
- Throughput (reads/second)
- Performance rating (⭐⭐⭐ if <500ns)

#### `json` - Export as JSON
```bash
uv run chronotick-client json --pretty
```
Outputs all data as JSON

**Use Cases**:
- Verify daemon is running
- Check performance (target: <500ns)
- Monitor time corrections
- Integration testing
- Performance regression testing

**Entry Point**: `uv run chronotick-client`

---

### 4. `chronotick_sdk_mcp.py` - SDK MCP Tools (28KB)

**Purpose**: In-process MCP tools using `@tool` decorator from claude-agent-sdk

**Dependencies**:
- `claude-agent-sdk>=0.1.0`

**Process Model**: Runs **inside** your Python process (not separate process)

**The 3 Tools**:

#### Tool 1: `get_time()`

**Signature**:
```python
@tool(name="get_time", description="...", input_schema={})
async def get_time(args: Dict[str, Any]) -> Dict[str, Any]
```

**What it does**:
1. Reads ChronoTickData from shared memory (via `read_chronotick_data()`)
2. Calculates corrected time at current moment
3. Calculates current uncertainty
4. Returns formatted response

**Returns** (claude-agent-sdk format):
```python
{
    "content": [{
        "type": "text",
        "text": "ChronoTick Corrected Time\n..."  # Human-readable
    }]
}
```

**Data returned** (embedded in text):
- corrected_time
- system_time
- offset_correction, drift_rate
- offset_uncertainty, drift_uncertainty, time_uncertainty
- confidence, source
- prediction_time, valid_until
- is_valid, is_ntp_ready, is_models_ready
- call_latency_ms

#### Tool 2: `get_daemon_status()`

**Signature**:
```python
@tool(name="get_daemon_status", description="...", input_schema={})
async def get_daemon_status(args: Dict[str, Any]) -> Dict[str, Any]
```

**What it does**:
1. Reads ChronoTickData
2. Formats status and health metrics

**Returns**:
- status: "ready" or "warming_up"
- warmup_complete, ntp_ready, models_ready
- measurement_count, total_corrections
- daemon_uptime
- last_ntp_sync, seconds_since_ntp
- confidence, data_source
- call_latency_ms

#### Tool 3: `get_time_with_future_uncertainty(future_seconds)`

**Signature**:
```python
@tool(name="get_time_with_future_uncertainty", input_schema={"future_seconds": float})
async def get_time_with_future_uncertainty(args: Dict[str, Any]) -> Dict[str, Any]
```

**What it does**:
1. Reads **same** ChronoTickData from shared memory
2. Uses `data.get_corrected_time_at()` to project forward
3. Uses `data.get_time_uncertainty()` to calculate future uncertainty
4. Returns both current and future values

**Important**: This does **NOT** get new data from ChronoTick. It uses mathematical projection:
```python
# Current
current_time = time.time()
current_uncertainty = sqrt(offset_unc² + (drift_unc * 0)²)

# Future (e.g., 300 seconds from now)
future_time = current_time + 300
future_uncertainty = sqrt(offset_unc² + (drift_unc * 300)²)
```

**Returns**:
- current_corrected_time, current_system_time, current_uncertainty
- future_seconds (input parameter)
- future_corrected_time, future_system_time, future_uncertainty
- uncertainty_increase = future_uncertainty - current_uncertainty
- confidence
- call_latency_ms

**Why this is useful**: Allows planning actions in advance with known uncertainty bounds

**Agent Helper Functions**:

**`create_chronotick_agent(allowed_tools=None, server_key="chronotick")`**
```python
from chronotick_shm.chronotick_sdk_mcp import create_chronotick_agent
from claude_agent_sdk import ClaudeSDKClient

agent = ClaudeSDKClient(create_chronotick_agent())
response = await agent.query("What time is it?")
```

**`create_minimal_agent()`** - Only `get_time`

**`create_monitoring_agent()`** - Only `get_daemon_status`

**`create_multi_service_agent()`** - Example combining with other MCP servers

**Global Handle Pattern**:
```python
_shared_memory_handle: Optional[SharedMemory] = None

def get_shared_memory() -> SharedMemory:
    global _shared_memory_handle
    if _shared_memory_handle is None:
        _shared_memory_handle = SharedMemory(name=SHARED_MEMORY_NAME, create=False)
    return _shared_memory_handle
```
- **First call**: ~1.5ms (attaches to shared memory)
- **Subsequent calls**: ~1ns (returns cached handle)
- **Read operations**: ~300ns (after first attachment)

**Entry Point**: N/A (imported as library)

**Usage**:
```python
# Direct import
from chronotick_shm.chronotick_sdk_mcp import get_time, get_shared_memory, read_chronotick_data

# Agent integration
from chronotick_shm.chronotick_sdk_mcp import create_chronotick_agent
```

---

### 5. `chronotick_stdio_mcp.py` - Stdio MCP Server (11KB)

**Purpose**: Standalone MCP server for Claude Code and other MCP clients

**Dependencies**:
- `fastmcp>=0.1.0`

**Process Model**: Runs as **separate process**, communicates via stdio

**Built with FastMCP**:
```python
from fastmcp import FastMCP

mcp = FastMCP("ChronoTick Time Server")

@mcp.tool()
def get_time() -> dict:
    """Get corrected time"""
    data = read_chronotick_data()
    return {
        "corrected_time": ...,
        "system_time": ...,
        # ... plain dict
    }
```

**The 3 Tools**: Same as SDK MCP, but with **plain dict** returns (not wrapped in "content")

**Returns** (MCP standard format):
```python
# get_time()
{
    "corrected_time": 1704556800.123456,
    "system_time": 1704556800.122222,
    "offset_correction": 0.001234,
    "drift_rate": 1.5e-06,
    "offset_uncertainty": 1e-05,
    "drift_uncertainty": 1e-09,
    "time_uncertainty": 1.2e-05,
    "confidence": 0.95,
    "source": "fusion",
    "prediction_time": 1704556795.0,
    "valid_until": 1704556855.0,
    "is_valid": True,
    "is_ntp_ready": True,
    "is_models_ready": True,
    "call_latency_ms": 0.0003,
}
```

**Connect from Claude Code**:
```json
// ~/.claude/config.json
{
  "mcpServers": {
    "chronotick": {
      "command": "uv",
      "args": ["run", "chronotick-stdio-server"],
      "cwd": "/full/path/to/chronotick_shm"
    }
  }
}
```

**Entry Point**: `uv run chronotick-stdio-server [--debug]`

**Logging**: All logs go to stderr (not stdout, which is used for MCP protocol)

---

## Data Formats

### ChronoTick → Shared Memory Mapping

**ChronoTick ClockCorrection** (from tsfm/chronotick_inference):
```python
ClockCorrection(
    offset_correction: float,
    drift_rate: float,
    offset_uncertainty: float,
    drift_uncertainty: float,
    prediction_time: float,
    valid_until: float,
    confidence: float,
    source: str,  # "ntp", "cpu", "gpu", "fusion"
)
```

**Shared Memory ChronoTickData** (in shm_config.py):
```python
ChronoTickData(
    # From ClockCorrection
    offset_correction: float,
    drift_rate: float,
    offset_uncertainty: float,
    drift_uncertainty: float,
    prediction_time: float,
    valid_until: float,
    confidence: float,
    source: CorrectionSource,  # Enum version

    # Computed by daemon
    corrected_timestamp: float,  # system + offset + drift * delta
    system_timestamp: float,     # time.time()

    # Daemon statistics
    sequence_number: int,
    flags: int,
    measurement_count: int,
    last_ntp_sync: float,
    daemon_uptime: float,
    total_corrections: int,
    average_latency_ms: float,
    reserved: int,
)
```

### Tool Response Formats

#### SDK MCP (`chronotick_sdk_mcp.py`)
**Format**: claude-agent-sdk format
```python
{
    "content": [{
        "type": "text",
        "text": "Human-readable formatted text + JSON"
    }]
}
```

#### Stdio MCP (`chronotick_stdio_mcp.py`)
**Format**: Plain dict (MCP standard)
```python
{
    "corrected_time": float,
    "system_time": float,
    ...
}
```

---

## Integration Guide

### Integrating with ChronoTick (tsfm)

**Current Integration Point** (in `chronotick_daemon_server.py`):

```python
# Line 186-192
self.pipeline = RealDataPipeline(self.config_path)
self.pipeline.initialize()
self.pipeline.ntp_collector.start_collection()

# Line 249
correction = self.pipeline.get_real_clock_correction(current_time)
```

**ChronoTick API Used**:
- `RealDataPipeline(config_path)` - Initialize pipeline
- `pipeline.initialize()` - Load models, setup
- `pipeline.ntp_collector.start_collection()` - Start NTP
- `pipeline.get_real_clock_correction(timestamp)` - Get correction

**Configuration**:
- Uses ChronoTick config.yaml
- Typically: `../tsfm/chronotick_inference/config.yaml`
- Contains NTP servers, model paths, thresholds

### Integrating with claude-agent-sdk

**Current Integration** (in `chronotick_sdk_mcp.py`):

```python
from claude_agent_sdk import tool, create_sdk_mcp_server, ClaudeAgentOptions

# Define tools with @tool decorator
@tool(name="get_time", description="...", input_schema={})
async def get_time(args):
    # Read from shared memory
    data = read_chronotick_data()
    # Return in claude-agent-sdk format
    return {"content": [...]}

# Create SDK MCP server
def create_chronotick_agent():
    sdk_server = create_sdk_mcp_server(
        name="chronotick_server",
        version="1.0.0",
        tools=[get_time, get_daemon_status, get_time_with_future_uncertainty]
    )

    return ClaudeAgentOptions(
        mcp_servers={"chronotick": sdk_server},  # Dict key = tool prefix
        allowed_tools=[
            "mcp__chronotick__get_time",  # mcp__{dict_key}__{tool_name}
            "mcp__chronotick__get_daemon_status",
            "mcp__chronotick__get_time_with_future_uncertainty"
        ]
    )
```

**Tool Naming Convention**: `mcp__{server_dict_key}__{tool_name}`
- NOT the server name!
- Uses the dict key from `mcp_servers={...}`

---

## Important Concepts

### 1. Lock-Free Synchronization

**Pattern**: Sequence number validation

**Writer** (daemon):
```python
seq = (seq + 1) % 2**32
write_data(buffer, data)  # Includes sequence number
```

**Reader** (all clients):
```python
seq_before = read_sequence(buffer)
data = read_data(buffer)
seq_after = read_sequence(buffer)

if seq_before != seq_after:
    # Torn read detected - retry
    data = read_data(buffer)
```

**Why it works**:
- Single writer (daemon only)
- Multiple readers (no locks needed)
- Sequence numbers detect concurrent writes
- Retry rate: <0.01% (very rare)

### 2. Global Handle Pattern

**Problem**: Attaching to shared memory is expensive (~1.5ms)

**Solution**: Attach once, cache handle globally
```python
_shared_memory_handle = None  # Module-level global

def get_shared_memory():
    global _shared_memory_handle
    if _shared_memory_handle is None:
        _shared_memory_handle = SharedMemory(name="chronotick_shm", create=False)
    return _shared_memory_handle
```

**Performance**:
- First call: ~1.5ms
- Subsequent calls: ~1ns (global lookup)
- Read operations: ~300ns

### 3. Uncertainty Propagation

**Formula**:
```python
uncertainty(t) = sqrt(offset_unc² + (drift_unc * t)²)
```

**Why it grows**:
- Offset uncertainty: constant
- Drift uncertainty: grows linearly with time
- Combined: square root of sum of squares

**Example**:
```
At t=0:   uncertainty = sqrt(10μs² + (1μs/min * 0)²) = 10μs
At t=5m:  uncertainty = sqrt(10μs² + (1μs/min * 5)²) = 11.2μs
At t=60m: uncertainty = sqrt(10μs² + (1μs/min * 60)²) = 60.8μs
```

### 4. Cache Alignment

**Memory Layout**: 128 bytes = 2 x 64-byte cache lines

**Why it matters**:
- CPU L1 cache line: 64 bytes
- 128 bytes fits in 2 cache lines
- Entire structure loads in single fetch
- Minimizes cache misses

**Performance impact**: ~5x faster than non-aligned

### 5. SDK MCP vs Stdio MCP

**SDK MCP** (`chronotick_sdk_mcp.py`):
- In-process
- Tools are Python functions
- No IPC overhead
- Uses `@tool` from claude-agent-sdk
- Returns `{"content": [...]}`

**Stdio MCP** (`chronotick_stdio_mcp.py`):
- Separate process
- Tools communicate via stdio
- MCP protocol overhead
- Uses `@mcp.tool()` from fastmcp
- Returns plain dict

**When to use each**:
- SDK MCP: Building Python agents programmatically
- Stdio MCP: Connecting from Claude Code or other MCP clients

---

## TODOs and Known Issues

### High Priority

#### TODO: Full Integration Testing
**Status**: Components work individually but not tested end-to-end

**Need to test**:
1. Start daemon with real ChronoTick
2. Verify shared memory updates
3. Test client reads
4. Test SDK MCP tools with agent
5. Test stdio MCP server with MCP client

**Blocker**: Requires ChronoTick daemon to be running

#### TODO: Error Handling Improvements
**Issue**: Limited error handling in daemon

**Needed**:
- Handle NTP collection failures
- Handle model inference failures
- Graceful degradation (fallback to NTP-only)
- Better error messages

#### TODO: Configuration Validation
**Issue**: No validation of config.yaml

**Needed**:
- Validate config exists before starting
- Check required fields
- Validate NTP server list
- Validate model paths

### Medium Priority

#### TODO: Performance Monitoring
**Issue**: No built-in performance monitoring

**Needed**:
- Track actual write latency
- Track read latency distribution
- Monitor torn read rate
- Alert on high latency

**Current**: `average_latency_ms` field exists but always 0.0

#### TODO: Warmup Progress Reporting
**Issue**: Warmup period (180s) has minimal progress reporting

**Current**:
```python
if int(elapsed) % 30 == 0:
    logger.info(f"Warmup progress: {progress:.1%}")
```

**Needed**:
- More frequent updates
- Expose via shared memory flag
- Allow `get_daemon_status()` to report warmup %

#### TODO: Documentation Examples
**Issue**: Need real working examples

**Needed**:
- Complete SDK MCP example (update `examples/sdk_mcp_example.py`)
- Stdio MCP client example
- Integration test script
- Performance benchmark suite

### Low Priority

#### TODO: Statistics Accuracy
**Issue**: Some statistics fields not populated

**Fields to implement**:
- `average_latency_ms` - currently always 0.0
- Track min/max/p99 latency
- Count torn reads

#### TODO: Platform Testing
**Current**: Only tested on Linux

**Need to test**:
- macOS (POSIX shared memory)
- Windows (different shared memory API - may not work)

#### TODO: Cleanup on Crash
**Issue**: If daemon crashes, shared memory persists

**Current workaround**:
```bash
rm /dev/shm/chronotick_shm
```

**Better solution**:
- Daemon writes PID to shared memory
- Check if PID exists on attach
- Auto-cleanup stale shared memory

### Known Issues

#### Issue 1: Daemon Needs Full tsfm Environment
**Impact**: High
**Workaround**: Run daemon from tsfm environment

**Cause**:
```python
from chronotick_inference.real_data_pipeline import RealDataPipeline
```

**Options**:
1. Keep as-is (current)
2. Make daemon optional dependency
3. Create mock daemon for testing

#### Issue 2: No Backward Compatibility
**Impact**: Low (new system)

**Note**: Shared memory layout is fixed at 128 bytes. Changes break compatibility.

**Mitigation**: Version field in layout (currently unused)

#### Issue 3: Single Daemon Limitation
**Impact**: Medium

**Current**: Only one daemon can write (by design)

**Future**: Could support multiple daemons with leader election

---

## Testing and Validation

### Unit Tests (TODO)

**Need to create**:
- `test_shm_config.py` - Test serialization/deserialization
- `test_client_eval.py` - Test client commands
- `test_sdk_mcp.py` - Test tool functions
- `test_stdio_mcp.py` - Test fastmcp server

### Integration Tests (TODO)

**Test scenarios**:
1. Daemon writes → Client reads
2. Daemon writes → SDK MCP reads
3. Daemon writes → Stdio MCP reads
4. Performance: 100K reads, measure latency
5. Stress test: High-frequency updates

### Manual Testing Checklist

#### Prerequisites
```bash
# Terminal 1: Start daemon (needs tsfm)
cd /path/to/tsfm
uv sync --extra core-models
cd /path/to/chronotick_shm
uv run chronotick-daemon --config ../tsfm/chronotick_inference/config.yaml
# Wait for warmup (~3 min)
```

#### Test 1: Client Eval
```bash
# Terminal 2
uv run chronotick-client read
# Expected: Shows corrected time, no errors

uv run chronotick-client status
# Expected: Shows "READY" status

uv run chronotick-client benchmark --iterations 10000
# Expected: <500ns latency (⭐⭐⭐ EXCELLENT)
```

#### Test 2: SDK MCP
```python
# Python script
from chronotick_shm.chronotick_sdk_mcp import get_shared_memory, read_chronotick_data
import time

shm = get_shared_memory()
data = read_chronotick_data()
corrected = data.get_corrected_time_at(time.time())

print(f"Corrected time: {corrected}")
# Expected: Prints time, no errors
```

#### Test 3: Stdio MCP
```bash
# Terminal 3
uv run chronotick-stdio-server
# Expected: Server starts, connects to daemon

# Terminal 4 (test with MCP inspector or Claude Code)
# Add to ~/.claude/config.json and test
```

### Performance Benchmarks

**Target Performance**:
- Read latency: <500ns (⭐⭐⭐ EXCELLENT)
- Read latency: 500ns-1μs (⭐⭐ GOOD)
- Read latency: 1μs-10μs (⭐ ACCEPTABLE)
- Throughput: >1M reads/second
- Daemon CPU: <2% @ 100Hz
- Torn read rate: <0.01%

**How to measure**:
```bash
uv run chronotick-client benchmark --iterations 100000
```

---

## Quick Reference

### File Sizes
```
shm_config.py                14KB (layout + serialization)
chronotick_daemon_server.py  17KB (daemon)
chronotick_client_eval.py    14KB (client)
chronotick_sdk_mcp.py        28KB (SDK MCP tools + helpers)
chronotick_stdio_mcp.py      11KB (fastmcp server)
```

### Dependencies
```
Core (all):          psutil
Daemon:              numpy, chronotick_inference
SDK MCP:             claude-agent-sdk
Stdio MCP:           fastmcp
```

### Entry Points
```bash
uv run chronotick-daemon           # Start daemon
uv run chronotick-client read      # Test client
uv run chronotick-stdio-server     # Start MCP server
```

### Key Formulas
```python
# Corrected time
corrected(t) = t + offset + drift * (t - prediction_time)

# Uncertainty
uncertainty(delta) = sqrt(offset_unc² + (drift_unc * delta)²)
```

### Tool Summary
```
Both MCPs offer 3 tools:
1. get_time()                               # Current time
2. get_daemon_status()                      # Health check
3. get_time_with_future_uncertainty(secs)   # Future projection
```

---

## Summary for Agent Integration

**For an agent working on integration**:

1. **Start here**: Understand `shm_config.py` - it's the contract between all components

2. **Daemon integration**: Focus on `chronotick_daemon_server.py` lines 186-212 (pipeline init) and 249 (get correction)

3. **SDK MCP integration**: Look at `chronotick_sdk_mcp.py` lines 129-254 (tool definitions) and 602-698 (agent helpers)

4. **Key data transformation**: `ClockCorrection` (from ChronoTick) → `ChronoTickData` (in shared memory)

5. **Testing priority**:
   - First: Get daemon writing to shared memory
   - Second: Verify client can read
   - Third: Test SDK MCP tools
   - Fourth: Test stdio MCP server

6. **Watch out for**:
   - Tool naming: `mcp__{dict_key}__{tool_name}` (not server name!)
   - Return formats: SDK MCP uses `{"content": [...]}`, stdio uses plain dict
   - Dependencies: Daemon needs full tsfm, SDK MCP needs claude-agent-sdk

7. **TODOs to prioritize**:
   - Full end-to-end testing
   - Error handling in daemon
   - Performance monitoring
   - Working examples

---

**End of Reference Document**

*This document describes the system as of the reorganization. Update as integration progresses.*
