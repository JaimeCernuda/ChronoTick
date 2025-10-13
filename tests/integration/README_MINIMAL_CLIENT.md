# Minimal Client Test

This directory contains a minimal client test that demonstrates ChronoTick usage after the post-cleanup organization.

## Files Created

1. **`test_minimal_client.py`** - Minimal ChronoTick client validation script
   - Connects to ChronoTick daemon
   - Collects timing data for N minutes
   - Saves results to CSV with errors and uncertainties

2. **`../../scripts/run_minimal_client_test.sh`** - Complete test runner
   - Starts ChronoTick daemon
   - Runs client test for 15 minutes
   - Sleeps for 20 minutes
   - Analyzes results

## Quick Start

```bash
cd ChronoTick/

# Run the complete test sequence (15 min test + 20 min sleep)
nohup ./scripts/run_minimal_client_test.sh > results/validation_post_cleanup/test_runner.log 2>&1 &

# Or run just the client test manually:
uv run python tests/integration/test_minimal_client.py --duration 900
```

## What This Tests

- Client can connect to daemon via shared memory
- Client receives corrected time, offset, drift, uncertainty
- System works continuously for 15+ minutes
- Error statistics are reasonable
- No crashes or failures

## Expected Results

After the test completes (35 minutes total):

- CSV file: `results/validation_post_cleanup/minimal_client_*.csv`
- Log file: `results/validation_post_cleanup/minimal_client_*.log`
- Success rate: >99%
- Mean error: <5ms
- Uncertainty: 1-10ms typical

## Verification Steps

After running, check:

```bash
# Find latest results
ls -lht results/validation_post_cleanup/minimal_client_*.csv | head -1

# Quick stats
python3 <<EOF
import pandas as df
df = pd.read_csv("results/validation_post_cleanup/minimal_client_*.csv")
print(f"Samples: {len(df)}")
print(f"Success rate: {(df['status'] == 'ok').sum() / len(df) * 100:.1f}%")
print(f"Mean error: {df[df['status']=='ok']['error_ms'].mean():.3f}ms")
EOF
```

## Post-Cleanup Organization Validation

This test confirms that after the repository reorganization:

- ✅ Client imports work (`clients/python/`)
- ✅ Daemon is accessible
- ✅ Shared memory communication works
- ✅ Uncertainty quantification is returned
- ✅ No path/import issues from reorganization
