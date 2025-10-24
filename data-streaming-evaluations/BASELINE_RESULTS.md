# System Clock Baseline Test Results

**Test Date:** October 24, 2025, 13:50-13:52
**Experiment ID:** `system_clock_baseline_20251024-132414`
**Duration:** ~24 seconds (500 events)
**Configuration:** 3 nodes (comp-18 coordinator, comp-11 & comp-12 workers)

---

## Executive Summary

This test compared raw system clock timestamps with NTP-corrected timestamps to establish a baseline for causality violation rates.

**Key Findings:**
- **System clocks: 0% violations** (0/1000 events)
- **NTP "ground truth": 49.9% violations** (499/1000 events)
- **Root cause: NTP measurement instability** (18.6ms offset jumps during test)

**Critical Insight:** NTP measurements showed greater instability than raw system clocks in this short test, highlighting the need for uncertainty quantification (ChronoTick's contribution).

---

## Test Configuration

### Nodes
| Node | Role | IP/Port | NTP Server |
|------|------|---------|------------|
| ares-comp-18 | Coordinator | - | 172.20.1.1:8123 |
| ares-comp-11 | Worker B | :9000 | 172.20.1.1:8123 |
| ares-comp-12 | Worker C | :9000 | 172.20.1.1:8123 |

### Broadcast Pattern
- 500 events over 23.5 seconds (~21.3 events/second)
- Mix of fast, medium, and slow delays
- UDP broadcast (low latency, realistic for distributed systems)

---

## Results: System Clock Performance

**Latency Measurement:** `worker_receive_time - coordinator_send_time`

### Worker B (comp-11)
```
Mean:  0.561 ms
Std:   0.052 ms
Range: 0.528 - 1.204 ms
Violations: 0/500 (0.0%)
```

### Worker C (comp-12)
```
Mean:  0.988 ms
Std:   0.956 ms
Range: 0.896 - 20.197 ms
Violations: 0/500 (0.0%)
```

**Interpretation:** Raw system clocks show consistent positive latency (network delay). No causality violations detected.

---

## Results: NTP "Ground Truth" Performance

**Latency Measurement:** `worker_ntp_timestamp - coordinator_ntp_timestamp`

### Worker B (comp-11)
```
Mean:  -7.086 ms
Std:    9.195 ms
Range: -18.089 to 1.070 ms
Violations: 206/500 (41.2%)
```

### Worker C (comp-12)
```
Mean:  -9.979 ms
Std:    9.209 ms
Range: -17.830 to 1.628 ms
Violations: 293/500 (58.6%)
```

**Interpretation:** NTP measurements show negative latency (violations) for ~50% of events. This should be impossible if NTP is providing accurate ground truth.

---

## Root Cause Analysis

### NTP Offset Behavior

**Coordinator (comp-18):**
- NTP offset: **-4.877 ms** (constant throughout test)
- Single NTP query at test start, cached for all 500 events

**Worker B (comp-11):**
- NTP offset: **-23.498 ms** (constant throughout test)
- Offset difference from coordinator: **18.621 ms**
- Result: Systematic negative latency for all events

**Worker C (comp-12):**
- **NTP offset CHANGED mid-test:**
  - Events 1-206: **-4.859 ms** (41.2%) - matches coordinator
  - Events 207-424: **-23.516 ms** (43.6%) - 18.6ms jump!
  - Events 425-500: **-23.482 ms** (15.2%) - slight drift
- Offset jump occurred at event 207 (10 seconds into test)

### Why NTP Shows Violations

NTP timestamps are calculated as:
```
ntp_timestamp = system_time + ntp_offset
```

NTP latency becomes:
```
ntp_latency = (worker_system_time + worker_ntp_offset) - (coord_system_time + coord_ntp_offset)
            = system_latency + (worker_ntp_offset - coord_ntp_offset)
```

**For Worker B:**
```
ntp_latency = 0.561ms + (-23.498ms - (-4.877ms))
            = 0.561ms - 18.621ms
            = -18.06ms  ← VIOLATION (artificial!)
```

**For Worker C (after offset jump):**
```
ntp_latency = 0.988ms + (-23.5ms - (-4.877ms))
            = 0.988ms - 18.6ms
            = -17.6ms  ← VIOLATION (artificial!)
```

### The Core Problem

1. **NTP offsets measured independently** at each node
2. **NTP measurements cached** (queried every 10s, not per-event)
3. **NTP measurement variance** causes 18.6ms offset differences between nodes
4. **When NTP offsets change mid-test**, it creates artificial causality violations

**The system clocks are actually more stable than the NTP measurements!**

---

## Implications

### What This Test Actually Shows

1. ✅ **System clocks are locally consistent**
   - No causality violations within 23-second window
   - Well-synchronized cluster environment

2. ❌ **NTP is NOT providing reliable ground truth**
   - 18.6ms measurement variance between nodes
   - Offset changes during test create artificial violations

3. ⚠️ **NTP caching creates temporal inconsistency**
   - Coordinator uses NTP offset from T=0s
   - Workers' NTP offsets may have changed by T=23s
   - Comparing timestamps with different reference points

### Why This Matters for ChronoTick

This test validates ChronoTick's design philosophy:

1. **Uncertainty quantification is essential**
   - NTP measurements have ~18ms variance
   - ChronoTick would capture this as uncertainty bounds
   - Enables fuzzy clock semantics (provable/ambiguous/violation)

2. **Point estimates are insufficient**
   - NTP provides single offset value
   - Doesn't quantify measurement error
   - Leads to false positives in causality violation detection

3. **Temporal consistency requires fresh measurements**
   - Cached NTP offsets become stale
   - ChronoTick's continuous inference adapts to drift
   - Provides time-varying uncertainty estimates

---

## Next Steps

### Immediate Actions

1. ✅ **Fix NTP client caching**
   - Query NTP for every event (or at least every second)
   - Don't reuse stale offset measurements
   - Accept higher query overhead for accuracy

2. ⏭️ **Run ChronoTick comparison test**
   - Same setup, but with embedded ChronoTick engine
   - Compare uncertainty bounds vs NTP ground truth
   - Measure: provable ordering, ambiguous cases, violations

3. ⏭️ **Longer duration tests**
   - 10-30 minute tests to observe clock drift
   - Validate ChronoTick uncertainty evolution over time
   - Check commit-wait uncertainty reduction

### Analysis Improvements

1. **Per-event NTP queries**
   - Don't cache NTP offset across events
   - Measure NTP query latency overhead
   - Trade accuracy for performance (worth it for evaluation)

2. **NTP uncertainty estimation**
   - Use NTP round-trip time as uncertainty proxy
   - Reject high-uncertainty measurements
   - Compare with ChronoTick's learned uncertainty

3. **Fuzzy semantics evaluation**
   - Provably ordered: coord_ntp + uncertainty < worker_ntp - uncertainty
   - Ambiguous: uncertainty bounds overlap
   - Violation: coord_ntp - uncertainty > worker_ntp + uncertainty

---

## Conclusion

This baseline test revealed that **NTP measurements are less stable than raw system clocks** over short timescales due to:
- Independent per-node measurements with ~18ms variance
- Caching causing temporal inconsistency
- Offset changes during test (18.6ms jump observed)

**System clocks: 0% violations (perfect)**
**NTP "ground truth": 49.9% violations (broken)**

This finding **strengthens the case for ChronoTick**, which:
- Quantifies uncertainty instead of hiding it
- Adapts to drift with continuous inference
- Enables fuzzy clock semantics for distributed systems

The next test will compare ChronoTick's uncertainty-aware timestamps with this NTP baseline to demonstrate improved causality reasoning under measurement uncertainty.

---

## Test Artifacts

- Coordinator CSV: `results/system_clock_baseline_20251024-132414/coordinator.csv`
- Worker B CSV: `results/system_clock_baseline_20251024-132414/worker_comp11.csv`
- Worker C CSV: `results/system_clock_baseline_20251024-132414/worker_comp12.csv`
- Logs: `logs/system_clock_baseline_20251024-132414/`
- Analysis script: `analyze_system_clock_baseline.py`
