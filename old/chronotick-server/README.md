# ChronoTick Server

High-precision MCP time server for geo-distributed agent synchronization with nanosecond accuracy.

## Features

### Phase 1 Implementation (Current)

- **Enhanced Time Models**: Nanosecond precision timestamps with clock quality metadata
- **Time Source Abstraction**: Support for System Clock, NTP, and PTP (with hardware timestamping preparation)
- **Precision Clock Manager**: Central service managing multiple time sources with automatic best-source selection
- **Backward Compatibility**: Full compatibility with existing MCP time server tools

### Core Capabilities

- **Nanosecond Precision**: Timestamps with nanosecond accuracy and uncertainty tracking
- **Multiple Time Sources**: System clock, NTP servers, PTP (preparation for hardware timestamping)
- **Clock Quality Assessment**: Automatic quality grading from UNKNOWN to REFERENCE level
- **Drift Monitoring**: Continuous monitoring of clock drift rates and compensation
- **Vector Clocks**: Logical timestamps for distributed event ordering and causality tracking
- **Cluster Synchronization**: Distributed time consensus algorithms for geo-distributed systems

## Installation

```bash
# Using uv (recommended)
uv sync
uv run chronotick-server

# Or install with pip
pip install -e .
chronotick-server
```

## Usage

### Basic MCP Server

```bash
# Start with default settings
uv run chronotick-server

# Start with custom node ID and timezone
uv run chronotick-server --node-id "node-001" --local-timezone "America/New_York"

# Enable debug logging
uv run chronotick-server --log-level DEBUG
```

### MCP Tools Available

- `get_current_time`: Get high-precision current time in specified timezone with nanosecond accuracy
- `convert_time`: Convert time between timezones with high precision
- `get_sync_status`: Get clock synchronization health and quality metrics  
- `create_vector_clock`: Generate vector clock for distributed event ordering
- `compare_timestamps`: Compare timestamps for causality and temporal relationships
- `measure_clock_drift`: Monitor clock drift over specified duration

## Architecture

### Time Sources
- **System Clock**: Local system time with ~1ms uncertainty
- **NTP**: Network Time Protocol with multiple server support and sub-second precision
- **Chrony**: System chrony daemon integration for enhanced NTP synchronization with real-time status

### Clock Quality Levels
- **REFERENCE**: Stratum 0/1 clocks (atomic/GPS reference)
- **EXCELLENT**: <10ms offset, high-quality synchronization
- **GOOD**: 10-100ms offset, stable synchronization
- **FAIR**: 100ms-1s offset, basic synchronization
- **POOR**: >1s offset, unreliable synchronization
- **UNKNOWN**: Quality not determined

### Vector Clocks
Support for distributed event ordering with:
- Node-specific logical timestamps
- Causal relationship detection (happens-before, happens-after, concurrent)
- Automatic clock advancement and merging

## Development

```bash
# Install development dependencies
uv sync

# Run tests
uv run pytest

# Lint code
uv run ruff check
uv run pyright

# Format code
uv run ruff format
```

## Configuration

The server automatically detects and configures available time sources:
- System clock (always available)
- NTP servers (pool.ntp.org, time.nist.gov, time.google.com, time.cloudflare.com)
- Chrony daemon interface (when chronyc is available)

## Future Phases

### Phase 2: Synchronization Services
- Full NTP client integration with chrony/ntpd
- Hardware PTP timestamping support
- GPS/GNSS time source integration
- Advanced drift compensation algorithms

### Phase 3: Distributed Coordination
- Cluster-wide consensus algorithms
- Network delay compensation
- Byzantine fault tolerance for time consensus
- Multi-master clock coordination

### Phase 4: Enhanced Features
- Sub-nanosecond precision with specialized hardware
- Leap second handling and TAI support
- Custom time protocols for isolated networks
- Performance optimization for high-frequency applications

## License

MIT License - see LICENSE file for details.