# ChronoTick

**High-precision time synchronization for distributed systems using machine learning**

ChronoTick delivers microsecond-accurate time corrections by combining real NTP measurements with predictive ML models. In an 8-hour validation test, ChronoTick reduced time synchronization errors by **75.5%** compared to standard system clocks (58.59ms → 14.38ms mean absolute error).

## What Makes ChronoTick Different?

Most systems rely on periodic NTP updates every 3 minutes. Between updates, your clock drifts. ChronoTick predicts and corrects this drift in real-time using a dual-model ML architecture:

```
System Clock (NTP only):  ----•---------•---------•----  (updates every 180s)
                              ↓ drift   ↓ drift   ↓ drift

ChronoTick:              ----•••••••••••••••••••••----  (corrections every 10s)
                              ↑ ML predicts drift
```

**Result**: 18x more frequent corrections with 75% lower error.

## Quick Start

Install and run ChronoTick in three commands:

```bash
# 1. Install with UV
uv sync --extra chronotick

# 2. Start the MCP server (for AI agents)
uv run chronotick-mcp

# 3. Or run standalone daemon (for distributed systems)
uv run chronotick-daemon --config configs/config_enhanced_features.yaml
```

That's it. ChronoTick will:
- Collect NTP measurements from multiple servers
- Train ML models on your system's clock behavior
- Provide corrected timestamps with uncertainty bounds
- Update predictions every 10 seconds

## How It Works

ChronoTick uses a **three-layer architecture**:

```
┌─────────────────────────────────────────────────────┐
│  Application Layer                                  │
│  ┌─────────────┐      ┌──────────────┐            │
│  │  MCP Server │      │  REST API    │            │
│  │  (AI Agents)│      │  (Optional)  │            │
│  └─────────────┘      └──────────────┘            │
└─────────────────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────┐
│  ML Prediction Layer                                │
│  ┌──────────────┐      ┌──────────────┐           │
│  │ Short-term   │      │  Long-term   │           │
│  │ Model (1Hz)  │◄────►│  Model (30s) │           │
│  │ 5s horizon   │      │  60s horizon │           │
│  └──────────────┘      └──────────────┘           │
│           ▼                      ▼                  │
│        ┌──────────────────────────┐                │
│        │   Prediction Fusion      │                │
│        │  (Inverse Variance)      │                │
│        └──────────────────────────┘                │
└─────────────────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────┐
│  Data Collection Layer                              │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────┐ │
│  │ NTP Client  │  │System Metrics│  │  Dataset  │ │
│  │(pool, google│  │  (CPU, mem)  │  │  Manager  │ │
│  │ cloudflare) │  │              │  │           │ │
│  └─────────────┘  └──────────────┘  └───────────┘ │
└─────────────────────────────────────────────────────┘
```

### Key Features

**Enhanced NTP Protocol**
Queries 2-3 samples per server with 100ms spacing for quality filtering. Rejects measurements with >10ms uncertainty.

**Backtracking Correction**
When new NTP measurements arrive, ChronoTick retroactively corrects past predictions using linear interpolation (REPLACE strategy).

**Dual-Model Architecture**
- Short-term: 1Hz updates, 5-second prediction horizon, 100-sample context
- Long-term: 0.033Hz updates, 60-second horizon, 300-sample context
- Fusion: Inverse-variance weighted combination for optimal accuracy

**Uncertainty Quantification**
Every prediction includes confidence intervals using TimesFM 2.5 quantiles [0.1, 0.5, 0.9].

## Performance Results

From 8-hour overnight validation (October 13, 2025):

| Metric | System Clock (NTP) | ChronoTick | Improvement |
|--------|-------------------|------------|-------------|
| Mean Absolute Error | 58.59 ms | **14.38 ms** | **75.5%** |
| Error Std Deviation | 12.9 ms | 8.3 ms | 35.7% |
| Update Frequency | 180s | **10s** | **18x faster** |
| Cumulative Error (8hr) | 16,492 ms | 4,045 ms | 75.5% |

*See full results with publication figures in `results/ntp_correction_experiment/overnight_8hr_20251013/`*

## Configuration

ChronoTick includes three preset configurations:

**1. Enhanced Features** (recommended for production)
```bash
uv run chronotick-mcp --config configs/config_enhanced_features.yaml
```
- TimesFM 2.5 short-term + Chronos-Bolt long-term
- Backtracking correction enabled
- Quantile-based confidence intervals

**2. Complete** (maximum accuracy)
```bash
uv run chronotick-mcp --config configs/config_complete.yaml
```
- Dual GPU models with system covariates
- Highest accuracy, requires more resources

**3. Short Only** (minimal resources)
```bash
uv run chronotick-mcp --config configs/config_short_only.yaml
```
- Single model, CPU-only
- Good for development/testing

## Use Cases

**AI Agent Coordination**
ChronoTick provides microsecond-precision timestamps to AI agents via the Model Context Protocol (MCP). Perfect for:
- Multi-agent systems requiring precise event ordering
- Distributed AI training with synchronized checkpoints
- Real-time decision systems across geo-distributed nodes

**Distributed Systems**
Run ChronoTick as a standalone daemon for:
- HPC clusters with synchronized experiments
- Microservices with strict timing requirements
- IoT networks with intermittent connectivity

**Research & Benchmarking**
ChronoTick logs detailed corrections for post-hoc analysis:
- Study clock drift patterns in your infrastructure
- Validate timing-sensitive algorithms
- Generate publication-quality performance figures

## Project Structure

```
ChronoTick/
├── server/              # ChronoTick server implementation
│   └── src/chronotick/  # Main package
│       ├── inference/   # ML prediction & NTP client
│       └── tsfm/        # Time series model factory
├── clients/             # Example client implementations
├── configs/             # Configuration presets
├── scripts/             # Evaluation & testing scripts
├── results/             # Validation test results
└── docs/                # Detailed documentation
```

## Documentation

- **[Installation Guide](docs/deployment/INSTALL.md)** - Detailed setup for production
- **[Architecture Overview](docs/architecture/design.md)** - System design philosophy
- **[NTP Correction Algorithms](docs/algorithms/NTP_CORRECTION_ALGORITHMS_SCIENTIFIC.md)** - Scientific analysis
- **[Evaluation Methodology](docs/evaluation/eval.md)** - How we test ChronoTick
- **[Test Results](docs/evaluation/TEST_STATUS.md)** - Current benchmarks

## Development

ChronoTick is built with Python 3.11+ and managed with UV:

```bash
# Clone and install
git clone https://github.com/JaimeCernuda/ChronoTick.git
cd ChronoTick
uv sync --extra chronotick --extra dev

# Run tests
uv run pytest server/tests/

# Format code
uv run black server/
uv run ruff server/
```

See [CLAUDE.md](CLAUDE.md) for detailed development guidelines.

## Citation

If you use ChronoTick in your research:

```bibtex
@misc{chronotick2025,
  title={ChronoTick: ML-Enhanced High-Precision Time Synchronization},
  author={ChronoTick Team},
  year={2025},
  note={8-hour validation demonstrates 75.5\% error reduction vs system clock}
}
```

## License

MIT License - See LICENSE file for details

---

**Status**: Production-ready (v1.0.0)
**Last Updated**: October 13, 2025
**Validation**: 8-hour overnight test completed successfully
