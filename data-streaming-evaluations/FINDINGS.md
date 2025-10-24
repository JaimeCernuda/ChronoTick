# Data Streaming Evaluation - Findings

## Summary of Investigation (October 24, 2025)

### The Problem You Identified

The initial results showed **100% causality violations for NTP** vs **0% for ChronoTick**, which was highly suspicious and didn't make sense.

### Root Cause

**Unfair comparison**: We were comparing apples to oranges.

#### What Actually Happened:

**Coordinator** (production-run-1):
- Only recorded `send_time_ns` from **uncorrected system clock**
- Did NOT query NTP (missing `--ntp-server` argument)
- Did NOT query ChronoTick

**Workers**:
- Recorded `receive_time_ns` from system clock
- Recorded `ntp_timestamp_ns` from **NTP-corrected** time
- Recorded `ct_timestamp_ns` (but ChronoTick wasn't running, used fallback offset=0)

#### The Unfair Comparison:

```
Analysis compared:
  Coordinator system clock (uncorrected, ~4.5ms ahead)
vs
  Worker NTP timestamp (NTP-corrected, pulled back -4.57ms)

Result: Worker NTP timestamp < Coordinator send time
→ 100% causality violations (ARTIFICIAL!)
```

#### Why "ChronoTick" Had 0% Violations:

- ChronoTick wasn't actually running (offset=0, fallback mode)
- It was comparing raw system clocks on both sides
- System latency ~0.4-1ms positive → no violations
- **This wasn't testing ChronoTick at all!**

### The Correct Experimental Design

**NTP = Ground Truth** (not what we're testing)
- Both coordinator and workers query NTP to establish causality ordering
- `coord_ntp_send < worker_ntp_recv` → physics truth

**What We Should Compare:**

1. **System Clock Violations:**
   ```
   Ground truth: coord_ntp < worker_ntp (NTP says A→B)
   Test: Does coord_system < worker_system agree?
   ```

2. **ChronoTick Violations:**
   ```
   Ground truth: coord_ntp < worker_ntp (NTP says A→B)
   Test: Do ChronoTick uncertainty bounds respect this?

   Specifically:
   - coord_ct_upper < worker_ct_lower → Provably ordered
   - Bounds overlap → Ambiguous (fuzzy semantics)
   - worker_ct_upper < coord_ct_lower → VIOLATION
   ```

### Data We Currently Have (production-run-1)

**Available:**
- ✓ 100 events successfully broadcast
- ✓ Both workers received all events
- ✓ System clock timestamps on coordinator
- ✓ System clock + NTP timestamps on workers
- ✓ ChronoTick timestamps (but using fallback, not real)

**Missing:**
- ✗ NTP timestamps on coordinator (can't establish ground truth)
- ✗ Real ChronoTick timestamps (server wasn't running)

### Quick Diagnosis Results

From `quick_analysis.py`:

```
System clock latency (correct comparison):
  recv_system - send_system: 0.4-1.0 ms (positive, no violations)

Unfair comparison (what analysis did):
  recv_ntp - send_system: -3.6 to -4.2 ms (negative, 100% violations)
```

**Visualization:** `results/production-run-1/diagnosis.png`
- Top plot: System clock comparison (correct)
- Bottom plot: Unfair comparison (NTP vs system)

### Next Steps

1. **Start Real ChronoTick MCP Server**
   - Run `tsfm/chronotick_mcp.py` on ARES master
   - Expose on port 8124 (or configure different port)
   - Workers and coordinator should query real ChronoTick

2. **Run Proper Comparison Test**
   - Coordinator must use `--ntp-server` and `--chronotick-server`
   - Workers use same servers
   - Collect ALL THREE timestamps:
     * System clock (baseline)
     * NTP (ground truth)
     * ChronoTick (our contribution)

3. **Fixed Analysis Logic**
   - Use NTP as ground truth reference
   - Compare system clock violations vs NTP ground truth
   - Compare ChronoTick bounds vs NTP ground truth
   - Evaluate:
     * Causality violations
     * Ordering consensus (provable vs ambiguous)
     * Window assignment with uncertainty quantification
     * Commit-wait measurements (uncertainty evolution)

### Test Parameters for Proper Evaluation

**Short validation (30 min):**
- 500 events
- Mixed pattern (slow, fast, medium)
- All three timestamps on both coordinator and workers

**Long-duration test (60 min):**
- 600 events
- Slower pattern to observe clock drift
- Check if ChronoTick uncertainty bounds stay stable

**High-frequency stress (10 min):**
- 1000 events
- Fast pattern to test rapid queries
- Verify NTP caching doesn't cause issues

### Expected Results (Once Fixed)

**System Clock:**
- Low violation rate (~0-5%) on well-synchronized cluster
- Some drift over longer tests
- No uncertainty quantification

**ChronoTick:**
- 0% violations when respecting uncertainty bounds
- Higher "ambiguous" classification (correctly identifies uncertainty)
- Demonstrates value of fuzzy clock semantics
- Shows commit-wait can reduce uncertainty over time

### Files Modified

- `src/coordinator.py`: Added NTP/ChronoTick support
- `deploy_ntp_fair_test.sh`: Test script (had port conflict, needs retry)
- `deploy_comprehensive.sh`: Full test suite
- `quick_analysis.py`: Diagnostic tool
- `FINDINGS.md`: This document

### Commits

1. `f9bee69`: Add NTP/ChronoTick support to coordinator
2. `5434aa9`: Add comprehensive test suite

---

**Status:** Ready to deploy proper test once ChronoTick MCP server is running.
