# Data Streaming Evaluation: Analysis Guide

**After data collection**: How to analyze results and generate figures

---

## 📊 Quick Start

### One-Command Analysis

```bash
cd data-streaming-evaluations
uv run analyze --experiment experiment-001
```

This automatically:
1. ✅ Loads data from `results/experiment-001/`
2. ✅ Runs all 5 analysis modules
3. ✅ Generates figures in `results/experiment-001/figures/`
4. ✅ Saves statistics to `results/experiment-001/statistics/`
5. ✅ Creates summary report in `results/experiment-001/report/`

---

## 🔍 Data Format

### Input Files

**coordinator.csv** (from ares-comp-18):
```csv
event_id,send_time_ns,worker_b_addr,worker_c_addr
1,1234567890123456789,ares-comp-11:9000,ares-comp-12:9000
2,1234567890223456789,ares-comp-11:9000,ares-comp-12:9000
```

**worker_comp11.csv** (from ares-comp-11):
```csv
event_id,receive_time_ns,ntp_offset_ms,ntp_uncertainty_ms,ct_offset_ms,ct_uncertainty_ms,ct_uncertainty_30s,ct_uncertainty_60s
1,1234567890125456789,2.5,5.0,2.3,8.5,6.2,4.8
2,1234567890225456789,2.6,5.1,2.4,8.3,6.0,4.6
```

**worker_comp12.csv** (from ares-comp-12): Same format

---

## 📈 Analysis Modules

### Module 1: Causality Analysis

**Question**: Do timestamps violate causality (effect before cause)?

**Method**:
```python
for each event:
    coordinator_send_time = coordinator_csv[event_id]['send_time']
    worker_receive_time = worker_csv[event_id]['receive_time']

    # NTP causality check
    ntp_timestamp = receive_time + (ntp_offset_ms / 1000)
    if ntp_timestamp < coordinator_send_time:
        → CAUSALITY VIOLATION (effect before cause!)

    # ChronoTick causality check
    ct_timestamp = receive_time + (ct_offset_ms / 1000)
    ct_upper_bound = ct_timestamp + (3 * ct_uncertainty_ms / 1000)
    if ct_upper_bound < coordinator_send_time:
        → CAUSALITY VIOLATION (even with uncertainty!)
```

**Output**:
- `causality_violations.png`: Timeline showing violations
- `causality_stats.json`:
  ```json
  {
    "ntp_violations": 18,
    "ntp_violation_rate": 0.18,
    "ct_violations": 0,
    "ct_violation_rate": 0.0
  }
  ```

**Expected Results**:
- NTP: 15-20% violations
- ChronoTick: 0% violations

---

### Module 2: Ordering Consensus

**Question**: Can nodes agree on event ordering without communication?

**Method**:
```python
for each event:
    # Get timestamps from both workers
    ntp_b = worker_b['ntp_timestamp']
    ntp_c = worker_c['ntp_timestamp']

    ct_b_range = [worker_b['ct_lower'], worker_b['ct_upper']]
    ct_c_range = [worker_c['ct_lower'], worker_c['ct_upper']]

    # NTP ordering (always claims to know)
    ntp_order = 'b_first' if ntp_b < ntp_c else 'c_first'

    # ChronoTick ordering (provable or ambiguous)
    if ct_b_range[1] < ct_c_range[0]:
        ct_order = 'b_first'  # Provable!
    elif ct_c_range[1] < ct_b_range[0]:
        ct_order = 'c_first'  # Provable!
    else:
        ct_order = 'concurrent'  # Ambiguous (true concurrency)
```

**Output**:
- `ordering_consensus.png`: Scatter plot showing provable vs ambiguous
- `ordering_stats.json`:
  ```json
  {
    "ntp_agreement_rate": 0.88,
    "ntp_disagreements": 12,
    "ct_provable": 80,
    "ct_ambiguous": 20,
    "ct_agreement_rate": 1.0
  }
  ```

**Expected Results**:
- NTP: 85-90% agreement (10-15% contradictions)
- ChronoTick: 100% agreement (80% provable, 20% correctly ambiguous)

---

### Module 3: Window Assignment

**Question**: For stream processing windows, do nodes agree on assignment?

**Method**:
```python
for window_size in [50, 100, 500, 1000]:  # milliseconds
    for each event:
        # NTP window assignment
        ntp_window_b = int(ntp_timestamp_b / (window_size/1000))
        ntp_window_c = int(ntp_timestamp_c / (window_size/1000))
        ntp_agrees = (ntp_window_b == ntp_window_c)

        # ChronoTick window assignment
        ct_window_lower_b = int(ct_lower_b / (window_size/1000))
        ct_window_upper_b = int(ct_upper_b / (window_size/1000))

        if ct_window_lower_b == ct_window_upper_b:
            # Confident assignment (entirely within one window)
            ct_confident_b = True
            ct_window_b = ct_window_lower_b
        else:
            # Ambiguous (spans window boundary)
            ct_confident_b = False

        # Similar for worker C
        # Check agreement
        if ct_confident_b and ct_confident_c:
            ct_agrees = (ct_window_b == ct_window_c)
        else:
            ct_agrees = None  # Ambiguous (need coordination)
```

**Output**:
- `window_assignment.png`: Bar chart across window sizes
- `window_assignment_stats.json`:
  ```json
  {
    "100ms_window": {
      "ntp_agreement": 0.68,
      "ct_confident": 78,
      "ct_ambiguous": 22,
      "ct_agreement": 1.0
    },
    "500ms_window": { ... }
  }
  ```

**Expected Results**:
- NTP: 58-85% agreement (worse for smaller windows)
- ChronoTick: 100% agreement (knows when to coordinate)

---

### Module 4: Coordination Cost

**Question**: How many coordination operations are needed?

**Method**:
```python
# NTP approach: Process all immediately, coordinate to fix disagreements
ntp_immediate = 100  # All events
ntp_disagreements = count_disagreements(ntp_decisions)
ntp_coordination_cost = ntp_disagreements

# ChronoTick approach: Coordinate only on ambiguous events
ct_immediate = count_confident(ct_decisions)  # ~80 events
ct_ambiguous = count_ambiguous(ct_decisions)  # ~20 events
ct_coordination_cost = ct_ambiguous

savings = ntp_coordination_cost - ct_coordination_cost
savings_pct = savings / ntp_coordination_cost * 100
```

**Output**:
- `coordination_cost.png`: Flow diagram comparing approaches
- `coordination_cost_stats.json`:
  ```json
  {
    "ntp_coordination": 28,
    "ct_coordination": 20,
    "savings": 8,
    "savings_percent": 28.5,
    "ct_immediate_correct": 80,
    "ct_immediate_correctness": 1.0
  }
  ```

**Expected Results**:
- NTP: 25-30 coordination operations (to fix disagreements)
- ChronoTick: 18-22 coordination operations (only ambiguous)
- Savings: 25-35%

---

### Module 5: Commit-Wait Analysis

**Question**: Can waiting reduce ambiguity?

**Method**:
```python
# For initially ambiguous events, track uncertainty over time
initially_ambiguous = get_ambiguous_events()

for event in initially_ambiguous:
    uncertainty_t0 = event['ct_uncertainty_initial']
    uncertainty_t30 = event['ct_uncertainty_30s']
    uncertainty_t60 = event['ct_uncertainty_60s']

    # Check if now confident at each time point
    for t, uncertainty in [(0, uncertainty_t0), (30, uncertainty_t30), (60, uncertainty_t60)]:
        ct_lower = event['ct_timestamp'] - 3 * uncertainty / 1000
        ct_upper = event['ct_timestamp'] + 3 * uncertainty / 1000

        window_lower = int(ct_lower / (100/1000))
        window_upper = int(ct_upper / (100/1000))

        if window_lower == window_upper:
            event[f'confident_at_{t}s'] = True
        else:
            event[f'confident_at_{t}s'] = False
```

**Output**:
- `commit_wait.png`: Line plot showing uncertainty decay
- `commit_wait_stats.json`:
  ```json
  {
    "initially_ambiguous": 20,
    "confident_after_30s": 6,
    "confident_after_60s": 12,
    "still_ambiguous_60s": 8,
    "coordination_reduction_60s": 0.60
  }
  ```

**Expected Results**:
- 30-40% of ambiguous events become confident after 30s
- 50-65% become confident after 60s
- Coordination reduction: 50-70% with commit-wait

---

## 📊 Generated Figures

### Figure 1: Causality Violations Timeline
**File**: `causality_violations.png`
**Type**: Timeline plot
**Content**:
- X-axis: Event ID (1-100)
- Y-axis: Time offset from coordinator send (ms)
- Horizontal line at Y=0: Coordinator send time
- Red dots below line: NTP causality violations
- Green ranges: ChronoTick bounds (always above or spanning line)

**Interpretation**:
- Any red dot shows NTP claiming impossible timing
- Green ranges respect physics (always possible)

---

### Figure 2: Ordering Consensus Scatter
**File**: `ordering_consensus.png`
**Type**: Scatter plot
**Content**:
- X-axis: Actual arrival difference (ms) [ground truth]
- Y-axis: ChronoTick overlap size (ms)
- Three regions:
  - Y=0, positive X: "B provably first" (green)
  - Y=0, negative X: "C provably first" (blue)
  - Y>0: "Concurrent/Ambiguous" (gold)
- Overlay: NTP decisions (✓ correct, ✗ wrong)

**Interpretation**:
- Green/blue points: ChronoTick proves ordering (80%)
- Gold points: True concurrency detected (20%)
- Red ✗: NTP wrong despite appearing certain

---

### Figure 3: Window Assignment Agreement
**File**: `window_assignment.png`
**Type**: Grouped bar chart
**Content**:
- X-axis: Window sizes (50ms, 100ms, 500ms, 1s)
- Y-axis: Agreement rate (0-100%)
- Three bars per window size:
  - Red: NTP agreement rate (65-85%)
  - Green: ChronoTick confident agreement (100%)
  - Gold: ChronoTick ambiguous rate (15-30%)

**Interpretation**:
- NTP: Appears deterministic but wrong
- ChronoTick: Perfect agreement, knows when ambiguous
- Smaller windows → more NTP failures

---

### Figure 4: Coordination Cost Flow
**File**: `coordination_cost.png`
**Type**: Side-by-side flow diagrams
**Content**:
- Left panel: NTP approach (100 → process → 28 conflicts → coordinate)
- Right panel: ChronoTick approach (80 confident → process, 20 ambiguous → coordinate)
- Bottom: Net savings calculation (28 → 20, -28.5%)

**Interpretation**:
- NTP: All appear immediate, many wrong → fix later
- ChronoTick: Correctly split immediate vs coordinate → fewer operations

---

### Figure 5: Commit-Wait Uncertainty Decay
**File**: `commit_wait.png`
**Type**: Multi-line plot
**Content**:
- X-axis: Wait time (0s, 30s, 60s, 90s)
- Y-axis: Uncertainty ±3σ (ms)
- Multiple colored traces (one per ambiguous event)
- Horizontal line: Confidence threshold (±5ms for 100ms windows)
- Shaded regions: Red (ambiguous) vs Green (confident)

**Interpretation**:
- Lines crossing threshold: Ambiguous → Confident
- Lines staying above: Truly concurrent (within precision)
- More wait → more confident → less coordination

---

## 📁 Output Directory Structure

After running analysis:

```
results/experiment-001/
├── figures/
│   ├── causality_violations.png
│   ├── ordering_consensus.png
│   ├── window_assignment.png
│   ├── coordination_cost.png
│   ├── commit_wait.png
│   └── summary_dashboard.png (all 5 in one figure)
├── statistics/
│   ├── causality_stats.json
│   ├── ordering_stats.json
│   ├── window_assignment_stats.json
│   ├── coordination_cost_stats.json
│   ├── commit_wait_stats.json
│   └── overall_summary.json
├── report/
│   └── experiment_report.md (auto-generated summary)
└── data/
    ├── coordinator.csv (raw)
    ├── worker_comp11.csv (raw)
    ├── worker_comp12.csv (raw)
    └── merged_analysis.csv (processed)
```

---

## 🔧 Custom Analysis

### Run Individual Modules

```bash
# Just causality analysis
uv run python analysis/causality_analysis.py --experiment experiment-001

# Just window assignment
uv run python analysis/window_assignment.py --experiment experiment-001 --window-sizes 50,100,500,1000
```

### Python API

```python
from analysis import causality_analysis, ordering_consensus

# Load data
experiment_dir = "results/experiment-001"

# Run causality analysis
causality_stats = causality_analysis.analyze(experiment_dir)
print(f"NTP violations: {causality_stats['ntp_violations']}")
print(f"CT violations: {causality_stats['ct_violations']}")

# Generate figure
causality_analysis.generate_figure(experiment_dir)

# Run ordering analysis
ordering_stats = ordering_consensus.analyze(experiment_dir)
print(f"Provable events: {ordering_stats['ct_provable']}")
print(f"Ambiguous events: {ordering_stats['ct_ambiguous']}")
```

---

## 📊 Comparing Multiple Experiments

### Compare Across Configurations

```bash
uv run analyze --compare experiment-001,experiment-002,experiment-003 \
  --output results/comparison/
```

Generates:
- Comparison tables (statistics across experiments)
- Side-by-side figures
- Trend analysis (if varying parameters)

### Example Comparisons

**Different NTP servers**:
- Experiment 1: pool.ntp.org
- Experiment 2: time.google.com
- Experiment 3: time.cloudflare.com

**Different broadcast patterns**:
- Experiment 1: Fast stream (10ms gaps)
- Experiment 2: Medium stream (50ms gaps)
- Experiment 3: Slow stream (100ms gaps)

**Different window sizes**:
- Experiment 1: 50ms windows
- Experiment 2: 100ms windows
- Experiment 3: 500ms windows

---

## ✅ Validation Checks

### Data Quality Checks

```bash
uv run python analysis/validate_data.py --experiment experiment-001
```

Checks:
- ✓ All 100 events present in coordinator log
- ✓ All 100 events received by worker B
- ✓ All 100 events received by worker C
- ✓ No missing timestamps
- ✓ No duplicate event IDs
- ✓ Timestamps in chronological order
- ✓ Uncertainty values reasonable (0-50ms)

### Sanity Checks

Expected ranges:
- NTP offset: -100ms to +100ms
- NTP uncertainty: 1ms to 20ms
- ChronoTick offset: -100ms to +100ms
- ChronoTick uncertainty: 2ms to 50ms
- Receive delay: 0.1ms to 10ms (LAN)

If values outside these ranges → investigate!

---

## 🎯 Key Metrics Summary

### For Paper Abstract

```json
{
  "causality_violations": {
    "ntp": "18%",
    "chronotick": "0%"
  },
  "distributed_consensus": {
    "ntp_agreement": "72%",
    "chronotick_agreement": "100%"
  },
  "coordination_reduction": {
    "operations": "28 → 20",
    "percent": "-28.5%"
  },
  "provable_without_coordination": {
    "percent": "80%",
    "correctness": "100%"
  }
}
```

### For Figures

Use the generated figures directly in papers/talks:
- Figure 1 (Causality): Shows 18% → 0% violations
- Figure 2 (Ordering): Shows provable (80%) vs ambiguous (20%)
- Figure 3 (Windows): Shows 68% → 100% agreement
- Figure 4 (Coordination): Shows -28.5% cost reduction
- Figure 5 (Commit-Wait): Shows 60% further reduction

---

## 🐛 Troubleshooting

### Missing Data

**Problem**: "Event 47 missing from worker_comp11.csv"

**Diagnosis**:
```bash
grep "event_id,47" results/experiment-001/worker_comp11.csv
```

**Causes**:
- Network packet loss (UDP)
- Worker crashed during event 47
- Worker buffer overflow

**Solution**:
- Re-run experiment
- Use TCP for reliability (trade-off: higher latency)
- Increase worker buffer size

### Unrealistic Timestamps

**Problem**: "ChronoTick uncertainty = 250ms (way too high!)"

**Diagnosis**:
```bash
# Check ChronoTick server logs
ssh ares "ssh ares-comp-11 'tail -100 /tmp/chronotick-debug.log'"
```

**Causes**:
- ChronoTick server not warmed up (needs 3 min)
- NTP quality degraded
- Network issues to ChronoTick server

**Solution**:
- Wait for warmup before starting experiment
- Check NTP server connectivity
- Verify ChronoTick server health

### Figures Don't Match Expectations

**Problem**: "Only 40% provable, expected 80%"

**Diagnosis**:
- Check broadcast pattern (too fast → more concurrency)
- Check NTP quality during experiment
- Check clock drift between nodes

**Solution**:
- Adjust broadcast delays in config
- Verify NTP synchronization quality
- Run calibration experiment first

---

## 📚 Next Steps

After analysis:

1. **Review figures**: Do they tell the story? (See NARRATIVE.md)
2. **Check statistics**: Match expected ranges?
3. **Read generated report**: Auto-summary in `report/experiment_report.md`
4. **Compare to baselines**: How do results compare to literature?
5. **Iterate**: If results unexpected, adjust config and re-run

For paper writing, see NARRATIVE.md for storytelling guidance.

For deployment issues, see DEPLOYMENT.md for troubleshooting.
