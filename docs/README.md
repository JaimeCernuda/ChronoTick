# ChronoTick Documentation

Comprehensive documentation for ChronoTick - High-Precision Time Synchronization System

## Quick Start

- **[Installation Guide](deployment/INSTALL.md)** - Get ChronoTick running in 5 minutes
- **[Deployment Guide](deployment/DEPLOY.md)** - Production deployment on distributed systems

## Architecture

- **[Design Overview](architecture/design.md)** - System architecture and design philosophy
- **[Technical Details](architecture/technical.md)** - Deep dive into implementation details

## Algorithms

- **[NTP Correction Algorithms](algorithms/NTP_CORRECTION_ALGORITHMS_SCIENTIFIC.md)** - Scientific analysis of correction methods
  - Enhanced NTP protocol (2-3 samples/server)
  - Backtracking correction with REPLACE strategy
  - Inverse-variance fusion for dual-model predictions

## Evaluation & Testing

- **[Evaluation Methodology](evaluation/eval.md)** - How we test ChronoTick
- **[Test Status](evaluation/TEST_STATUS.md)** - Current test results and benchmarks
- **Overnight 8-Hour Test Results**: `../tsfm/results/ntp_correction_experiment/overnight_8hr_20251013/`
  - **[Analysis Summary](../tsfm/results/ntp_correction_experiment/overnight_8hr_20251013/analysis/ANALYSIS_SUMMARY.md)**
  - **[Figure Descriptions](../tsfm/results/ntp_correction_experiment/overnight_8hr_20251013/analysis/FIGURE_DESCRIPTIONS.md)** - Publication-ready figures

## Interfaces & Integration

- **[Interface Plan](interfaces/CHRONOTICK_INTERFACE_PLAN.md)** - ChronoTick interface design
- **[ChronoTick SHM Docs](interfaces/chronotick_shm_docs/)** - Shared memory interface documentation (if available)

## Key Performance Results

From 8-hour overnight test (2025-10-13):

| Metric | System Clock | ChronoTick | Improvement |
|--------|--------------|------------|-------------|
| **Mean Absolute Error** | 58.59 ms | 14.38 ms | **75.5%** |
| **Error Std Deviation** | 12.9 ms | 8.3 ms | 35.7% |
| **Cumulative Error (8hr)** | 16,492 ms | 4,045 ms | **75.5%** |
| **Update Frequency** | 180s (NTP) | 10s | **18x** |
| **Clock Drift** | 3.03 PPM | Corrected | Linear |

## Repository Structure

```
ChronoTick/
├── tsfm/                          # Main implementation (ACTIVE)
│   ├── chronotick_inference/      # ChronoTick MCP server
│   ├── chronotick_mcp.py          # Entry point
│   ├── tsfm/                      # TSFM factory
│   ├── scripts/                   # Evaluation scripts
│   └── results/                   # Test results
├── chronotick_shm/                # Shared memory interface
├── servers/                       # MCP reference servers
├── sdk-mcp/                      # MCP SDK
├── docs/                         # Documentation (YOU ARE HERE)
└── old/                          # Deprecated code
    ├── chronotick-server/         # Alternative implementation
    ├── inferance_layer/           # Deprecated
    └── eval/                      # Old evaluation code
```

## For Developers

### Entry Points (after `uv sync --extra chronotick`):
- `chronotick-mcp` - MCP server for AI agents
- `chronotick-daemon` - Standalone daemon for multinode
- `chronotick-config` - Configuration selector

### Development Workflow:
1. Read **[CLAUDE.md](../tsfm/CLAUDE.md)** for development guidelines
2. Check **[Installation Guide](deployment/INSTALL.md)** for setup
3. Review **[Technical Details](architecture/technical.md)** for internals
4. Run tests: `cd tsfm && uv run pytest`

### Configuration Files:
- **`tsfm/chronotick_inference/config_enhanced_features.yaml`** - Production config (recommended)
- **`tsfm/chronotick_inference/config_complete.yaml`** - Full feature set
- **`tsfm/chronotick_inference/config_short_only.yaml`** - Minimal config

## Research & Publications

### Citation
If you use ChronoTick in your research, please cite:

```bibtex
@misc{chronotick2025,
  title={ChronoTick: ML-Enhanced High-Precision Time Synchronization},
  author={ChronoTick Team},
  year={2025},
  note={8-hour validation test demonstrates 75.5\% error reduction vs system clock}
}
```

### Key Innovations
1. **Dual-Model ML Prediction**: Short-term (1Hz, 5s horizon) + Long-term (0.033Hz, 60s horizon)
2. **Enhanced NTP Protocol**: 2-3 samples per server, 100ms spacing for quality filtering
3. **Backtracking Correction**: REPLACE strategy with linear interpolation between NTP measurements
4. **Uncertainty Quantification**: TimesFM 2.5 quantiles [0.1, 0.5, 0.9] for prediction intervals
5. **IPC Optimization**: <1ms cache hit latency, 45ms cache miss (with model inference)

## Troubleshooting

### Common Issues
1. **Import errors**: Check model environment (TTM/Toto/Time-MoE conflicts)
2. **NTP timeout**: Requires UDP port 123 access
3. **Entry points not found**: Run `uv sync --extra chronotick`

### Support
- Report issues: https://github.com/JaimeCernuda/ChronoTick/issues
- Development guide: [CLAUDE.md](../tsfm/CLAUDE.md)

## License

MIT License - See repository root for details

---

**Last Updated**: 2025-10-13
**Version**: 1.0.0 (Production-ready with 8-hour validation)
