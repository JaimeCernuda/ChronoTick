# Stream Processing Evaluation: Experimental Design

**Goal**: Demonstrate that bounded clocks (ChronoTick ±3σ) enable stream processing consensus where single-point clocks (NTP) fail

**Duration**: 30-60 minutes (focused, dense data)

**Status**: Design document for future deployment

---

## 🎯 The Core Narratives to Test

### Narrative 1: Single-Point Clocks Disagree
**Problem**: Even with NTP synchronization, two nodes' clocks disagree on:
- What time an event occurred
- Which window an event belongs to
- Whether two events are concurrent

**Test**: Cross-node comparison of NTP measurements at same wall-clock moments

### Narrative 2: Bounded Clocks Enable Consensus
**Solution**: ChronoTick's ±3σ ranges overlap, creating "consensus zones"

**Test**: Cross-node comparison of ChronoTick ranges - do they overlap?

### Narrative 3: Stream Processing Applications
**Practical value**: Consensus zones enable:
- Window assignment with confidence levels
- Distributed joins without coordinators
- Duplicate detection

**Test**: Simulate Apache Flink window assignment on real event stream

---

## 🔬 Experimental Setup

### Phase 1: Data Collection (30-60 minutes)

#### Node Configuration
```yaml
deployment:
  nodes: 2 (ares-comp-11, ares-comp-12)
  duration: 30-60 minutes

chronotick_sampling:
  frequency: 1 Hz (every 1 second)
  measurements: 1800-3600 samples per node
  fields: [system_time, chronotick_time, chronotick_offset_ms, chronotick_uncertainty_ms]

ntp_sampling:
  strategy: "synchronized_trigger"  # NEW!
  frequency: 0.1 Hz (every 10 seconds)
  measurements: 180-360 samples per node
  fields: [ntp_time, ntp_offset_ms, ntp_uncertainty_ms]

synchronization:
  method: "coordinator_broadcast"  # One node signals when to query NTP
  tolerance: ±1 second (both nodes query within 1s of each other)
```

#### Key Innovation: Synchronized NTP Queries

**Problem with current data**:
```
Node 1 NTP: [120s, 240s, 360s, ...]
Node 2 NTP: [122s, 242s, 362s, ...]  # 122s deployment offset
→ Rarely aligned! Only 7 simultaneous measurements over 8 hours
```

**Solution**:
```python
# Coordinator script (runs on Node 1)
import time
import socket

while True:
    # Broadcast "query NTP now!" to both nodes
    timestamp = time.time()
    broadcast_to_nodes("QUERY_NTP", timestamp)

    # Both nodes query NTP within 1 second
    # Now we have synchronized NTP measurements!

    time.sleep(10)  # Every 10 seconds
```

**Expected result**: 180-360 SYNCHRONIZED NTP pairs (vs current 7!)

---

## 📊 Evaluation Metrics

### Metric 1: Cross-Node NTP Agreement (Single-Point Baseline)

**For each synchronized NTP pair**:
```python
# Both nodes queried NTP at wall-clock moment T
ntp1_offset = node1_ntp_offset_ms
ntp2_offset = node2_ntp_offset_ms

# Do they agree?
agreement = abs(ntp1_offset - ntp2_offset) < 10ms

# Expected: 10-20% agreement (based on current data)
```

**Visualization**:
- X-axis: Time (minutes)
- Y-axis: NTP offset (ms)
- Two lines (Node 1, Node 2) showing disagreement
- Highlight disagreement zones

### Metric 2: Cross-Node ChronoTick Consensus (Bounded Clock)

**For each wall-clock moment T** (1 Hz sampling):
```python
# Node 1 ChronoTick range
range1 = [chronotick1 - 3σ, chronotick1 + 3σ]

# Node 2 ChronoTick range
range2 = [chronotick2 - 3σ, chronotick2 + 3σ]

# Do ranges overlap?
consensus = (range1[1] >= range2[0]) and (range2[1] >= range1[0])

# Expected: 100% overlap (based on current data)
```

**Visualization**:
- X-axis: Time (minutes)
- Y-axis: Clock offset (ms)
- Two ranges (Node 1 ±3σ, Node 2 ±3σ)
- Gold shading for consensus zones
- Expected: Gold everywhere!

### Metric 3: Window Assignment (Stream Processing)

**For multiple window sizes** (100ms, 500ms, 1s, 5s):
```python
# Simulate event stream (1 event per second at both nodes)
for t in range(1800):  # 30 minutes
    # Which window does this event belong to?

    # Node 1: Using NTP
    window1_ntp = (system_time + ntp1_offset) // window_size

    # Node 2: Using NTP
    window2_ntp = (system_time + ntp2_offset) // window_size

    # Node 1: Using ChronoTick
    window1_ct = (system_time + chronotick1_offset) // window_size

    # Node 2: Using ChronoTick
    window2_ct = (system_time + chronotick2_offset) // window_size

    # Agreement?
    ntp_agrees = (window1_ntp == window2_ntp)
    chronotick_agrees = (window1_ct == window2_ct)

# Expected:
# - NTP agreement: 85-95% (for 1s windows)
# - ChronoTick agreement: 96-99% (for 1s windows)
```

**Visualization**:
- Bar chart per window size
- NTP agreement vs ChronoTick agreement
- Show improvement

### Metric 4: Ambiguity Detection (Bounded Clock Value)

**Identify events near window boundaries**:
```python
# For each event, check if ChronoTick can determine window confidently
chronotick_time = system_time + chronotick_offset

# Event's possible time range
lower = chronotick_time - 3σ
upper = chronotick_time + 3σ

# Which windows could this event belong to?
window_lower = lower // window_size
window_upper = upper // window_size

if window_lower == window_upper:
    # Confident! Entire range in one window
    confident_assignment = True
else:
    # Ambiguous! Range spans window boundary
    ambiguous = True

# Report:
# - Confident assignments: X%
# - Ambiguous (need buffering): Y%
```

**Visualization**:
- Pie chart: Confident vs Ambiguous
- Show that ChronoTick KNOWS when it's uncertain!

---

## 🚀 Implementation Plan

### Step 1: Enhanced Client Script

Modify `chronotick_client_validation.py`:

```python
# Add synchronized NTP trigger
class SynchronizedNTPClient:
    def __init__(self, coordinator_host=None):
        self.is_coordinator = (coordinator_host is None)
        self.coordinator_host = coordinator_host
        self.last_ntp_query = 0

    def should_query_ntp(self):
        if self.is_coordinator:
            # Query every 10 seconds and broadcast
            if time.time() - self.last_ntp_query > 10:
                self.broadcast_ntp_trigger()
                return True
        else:
            # Listen for coordinator's trigger
            if self.received_ntp_trigger():
                return True
        return False

    def broadcast_ntp_trigger(self):
        # Send UDP broadcast: "QUERY_NTP"
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.sendto(b"QUERY_NTP", ('<broadcast>', 9999))

    def received_ntp_trigger(self):
        # Check for UDP broadcast
        # Return True if received within last second
        ...
```

### Step 2: Run Deployment

**On Node 1 (ares-comp-11) - Coordinator**:
```bash
python3 chronotick_client_validation.py \
  --duration 30 \
  --chronotick-freq 1.0 \
  --ntp-coordinator \
  --ntp-freq 0.1 \
  --output experiment-stream-30min-node1.csv
```

**On Node 2 (ares-comp-12) - Follower**:
```bash
python3 chronotick_client_validation.py \
  --duration 30 \
  --chronotick-freq 1.0 \
  --ntp-follow ares-comp-11 \
  --output experiment-stream-30min-node2.csv
```

**Expected data**:
- 1800 ChronoTick samples per node (every 1s for 30 min)
- 180 NTP samples per node (every 10s for 30 min)
- **180 synchronized NTP pairs!** (vs current 7)

### Step 3: Analysis Script

```python
# scripts/crazy_ideas/streaming_eval_30min.py

def analyze_synchronized_ntp(df1, df2):
    """Analyze synchronized NTP measurements."""
    # Match NTP samples within 1 second tolerance
    pairs = []
    for idx1, row1 in df1[df1['has_ntp']].iterrows():
        t1 = row1['elapsed_seconds']
        # Find Node 2 NTP within ±1 second
        match = df2[df2['has_ntp'] &
                   (abs(df2['elapsed_seconds'] - t1) < 1)]
        if len(match) > 0:
            row2 = match.iloc[0]
            pairs.append({
                'time': t1,
                'ntp1': row1['ntp_offset_ms'],
                'ntp2': row2['ntp_offset_ms'],
                'diff': abs(row1['ntp_offset_ms'] - row2['ntp_offset_ms']),
                'agrees': abs(row1['ntp_offset_ms'] - row2['ntp_offset_ms']) < 10
            })

    df_pairs = pd.DataFrame(pairs)

    print(f"Synchronized NTP pairs: {len(df_pairs)}")
    print(f"Agreement (<10ms): {df_pairs['agrees'].sum()}/{len(df_pairs)} = {df_pairs['agrees'].mean()*100:.1f}%")

    return df_pairs

def analyze_consensus_zones(df1, df2):
    """Analyze ChronoTick consensus zones."""
    # Match ChronoTick samples at every second
    consensus = []
    for t in range(int(df1['elapsed_seconds'].min()),
                   int(df1['elapsed_seconds'].max())):
        # Node 1 ChronoTick at time t
        s1 = df1.iloc[(df1['elapsed_seconds'] - t).abs().idxmin()]

        # Node 2 ChronoTick at time t
        s2 = df2.iloc[(df2['elapsed_seconds'] - t).abs().idxmin()]

        # Ranges
        range1 = [s1['chronotick_offset_ms'] - 3*s1['chronotick_uncertainty_ms'],
                  s1['chronotick_offset_ms'] + 3*s1['chronotick_uncertainty_ms']]
        range2 = [s2['chronotick_offset_ms'] - 3*s2['chronotick_uncertainty_ms'],
                  s2['chronotick_offset_ms'] + 3*s2['chronotick_uncertainty_ms']]

        overlaps = (range1[1] >= range2[0]) and (range2[1] >= range1[0])

        consensus.append({
            'time': t,
            'overlaps': overlaps,
            'range1': range1,
            'range2': range2
        })

    df_consensus = pd.DataFrame(consensus)

    print(f"Consensus zones: {df_consensus['overlaps'].sum()}/{len(df_consensus)} = {df_consensus['overlaps'].mean()*100:.0f}%")

    return df_consensus

def analyze_window_assignment(df1, df2, window_sizes=[100, 500, 1000, 5000]):
    """Simulate window assignment for stream processing."""
    results = {}

    for window_ms in window_sizes:
        # Match samples at every second
        agreements_ntp = []
        agreements_ct = []

        for t in range(...):
            s1 = df1.iloc[...]
            s2 = df2.iloc[...]

            # Window assignment using NTP
            if s1['has_ntp'] and s2['has_ntp']:
                w1_ntp = (s1['system_time'] + s1['ntp_offset_ms']/1000) // (window_ms/1000)
                w2_ntp = (s2['system_time'] + s2['ntp_offset_ms']/1000) // (window_ms/1000)
                agreements_ntp.append(w1_ntp == w2_ntp)

            # Window assignment using ChronoTick
            w1_ct = (s1['system_time'] + s1['chronotick_offset_ms']/1000) // (window_ms/1000)
            w2_ct = (s2['system_time'] + s2['chronotick_offset_ms']/1000) // (window_ms/1000)
            agreements_ct.append(w1_ct == w2_ct)

        results[window_ms] = {
            'ntp_agreement': np.mean(agreements_ntp) * 100,
            'chronotick_agreement': np.mean(agreements_ct) * 100,
        }

    return results
```

### Step 4: Visualizations

**Figure 1: Synchronized NTP Disagreement** (10-minute focused view)
- X-axis: Time (0-10 minutes)
- Y-axis: NTP offset (ms)
- Two lines: Node 1 (green), Node 2 (blue)
- Show ALL 60 NTP pairs in this window
- Annotate disagreement regions

**Figure 2: Consensus Zones** (10-minute focused view)
- X-axis: Time (0-10 minutes)
- Y-axis: Clock offset (ms)
- Two ranges: Node 1 ±3σ (green), Node 2 ±3σ (blue)
- Gold shading: Consensus zones
- Show ALL 600 ChronoTick samples in this window

**Figure 3: Window Assignment Agreement**
- Bar chart for each window size (100ms, 500ms, 1s, 5s)
- NTP vs ChronoTick agreement
- Show improvement

**Figure 4: Full 30-Minute Timeline**
- Show full deployment
- Panel (a): NTP disagreement over time
- Panel (b): Consensus zones over time
- Panel (c): Agreement rate evolution

---

## 📈 Expected Results

Based on current 8-hour data extrapolated to dense 30-minute test:

### Single-Point (NTP)
- **Synchronized pairs**: 180 (vs current 7!)
- **Agreement (<10ms)**: 10-20% (based on current 0.8%, but with better sync might improve)
- **Mean disagreement**: 2-8ms

### Bounded (ChronoTick)
- **Consensus zones**: 1800 moments tested
- **Overlap rate**: 100% (based on current data)
- **Median consensus size**: 6ms

### Window Assignment
- **100ms windows**: ChronoTick +20% vs NTP baseline
- **1000ms windows**: ChronoTick +4-8% vs NTP baseline

---

## 🔧 Alternative Approaches

### Option A: Simpler - Just Increase Sampling Frequency

**No code changes needed!**

Just run existing client with:
```bash
# Increase NTP frequency (but may hit rate limits)
--ntp-interval 10  # Query every 10 seconds instead of 120s

# Increase ChronoTick frequency
--sample-interval 1  # Sample every 1 second instead of 10s
```

**Pros**: No code changes, can run immediately
**Cons**: Still won't have synchronized NTP (nodes start at different times)

### Option B: Post-Processing Alignment

**Use existing data differently!**

Instead of requiring synchronized NTP, use:
- **All NTP measurements** (237 per node)
- **Linear interpolation** to estimate what each node's NTP would be at ANY moment
- Compare interpolated NTP vs ChronoTick at every ChronoTick sample

```python
# For each ChronoTick sample at time T:
# Interpolate what Node's NTP offset would be at time T
ntp_interpolated = np.interp(T, ntp_times, ntp_offsets)

# Compare with ChronoTick prediction
chronotick_at_T = chronotick_offset_ms

# This gives us 2873 comparisons instead of 7!
```

**Pros**: Can use existing data, no new deployment
**Cons**: Interpolation assumes linear drift (may not be accurate)

### Option C: Multiple Window Views (Use Current Data Better!)

**Show same data at different time scales!**

Instead of sampling, show:
- **Figure 1**: First 10 minutes (ALL points in this window)
- **Figure 2**: Minutes 60-70 (ALL points)
- **Figure 3**: Minutes 240-250 (ALL points)
- **Figure 4**: Minutes 420-430 (ALL points)

This shows consistency across deployment without subsampling!

---

## ✅ Recommendation

**Immediate (tonight)**: Use Option C - create multiple window views showing ALL data

**Short-term (tomorrow)**: Run Option A - 30-minute test with higher frequency sampling

**Long-term (future paper)**: Implement Option B - synchronized NTP triggers for perfect comparison

---

## 📝 Concrete Next Steps for Tonight

1. **Fix current visualization**:
   - Show ALL 237 NTP points (not just 13)
   - Create multiple focused windows (10-min, 20-min, etc.)
   - Stop arbitrary subsampling

2. **Create comprehensive figure set**:
   - Overview: Full 8-hour timeline
   - Window 1: Minutes 0-10 (startup phase)
   - Window 2: Minutes 120-130 (mid-deployment)
   - Window 3: Minutes 420-430 (late-deployment)

3. **Document experimental design** (this file!) for future 30-min test

Want me to implement the "multiple window views" approach with current data right now?
