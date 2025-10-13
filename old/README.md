# Deprecated/Old Code

This directory contains deprecated or alternative implementations that are not actively used.

## Contents

### `inferance_layer/` - **DEPRECATED**
Original inference layer implementation, superseded by `tsfm/chronotick_inference/`.

**Status**: Deprecated
**Replaced by**: `tsfm/chronotick_inference/`
**Reason**: Unified implementation with TSFM factory integration

### `chronotick-server/` - Alternative Implementation
Alternative ChronoTick server design with PTP support preparation.

**Status**: Not Active
**Active implementation**: `tsfm/chronotick_inference/`
**Note**: This was an experimental design. The production implementation is in `tsfm/`.

### `eval/` - Old Evaluation Code
Original evaluation scripts and test code.

**Status**: Superseded
**Replaced by**: `tsfm/scripts/` and `tsfm/results/`
**Note**: Modern evaluation framework with visualization data is in `tsfm/scripts/test_with_visualization_data.py`

## Why Keep This Code?

These implementations are preserved for:
1. **Historical reference** - Understanding design evolution
2. **Algorithm comparison** - Comparing old vs new approaches
3. **Recovery** - In case specific features need to be ported
4. **Research** - Academic comparison of different architectures

## Active Codebase

For current development, see:
- **Main implementation**: `tsfm/chronotick_inference/`
- **Entry points**: `chronotick-mcp`, `chronotick-daemon`, `chronotick-config`
- **Documentation**: `docs/`
- **Test results**: `tsfm/results/ntp_correction_experiment/`

---

**Do not use code from this directory in production.**
**Refer to `tsfm/` for the active, maintained implementation.**
