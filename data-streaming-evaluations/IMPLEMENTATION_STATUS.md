# Implementation Status & Next Steps

**Created**: 2025-01-24
**Status**: Framework complete, implementation in progress

---

## ✅ What's Been Created

### Documentation (100% Complete)

1. **README.md** ⭐
   - Quick start guide
   - Project overview
   - Usage examples
   - Comprehensive documentation index

2. **DEPLOYMENT.md** (ARES-specific)
   - One-command deployment guide
   - Manual step-by-step instructions
   - Network configuration (NTP proxy, ChronoTick server)
   - Troubleshooting guide
   - SSH command templates

3. **NARRATIVE.md** (TrueTime-enriched)
   - Five killer narratives (causality, ordering, windows, ambiguity, commit-wait)
   - Paper section templates
   - Soundbites and quotes
   - Positioning vs related work
   - Teaching strategies (1-min, 5-min, 15-min pitches)

4. **ANALYSIS.md**
   - Post-experiment analysis guide
   - Five analysis modules explained
   - Figure interpretation
   - Custom analysis examples
   - Validation checks

### Infrastructure (100% Complete)

5. **pyproject.toml**
   - UV package configuration
   - Dependencies specified
   - Entry points for coordinator, worker, analyze

6. **deploy.sh** ⭐
   - One-command deployment script
   - Pre-flight checks
   - Automated worker + coordinator deployment
   - Results collection
   - Analysis triggering

### Core Code (80% Complete)

7. **src/common.py** ✅
   - Event dataclass
   - TimestampRecord dataclass
   - UDPBroadcaster (network utilities)
   - UDPListener
   - NTPClient (placeholder)
   - ChronoTickClient (placeholder)
   - Logging setup
   - High-precision sleep

8. **src/coordinator.py** ✅
   - BroadcastPattern class
   - Coordinator class
   - Main broadcast loop
   - CSV output
   - Progress logging

9. **src/worker.py** ⚠️ **NOT YET CREATED**
   - Needs implementation!

10. **analysis/** scripts ⚠️ **NOT YET CREATED**
    - causality_analysis.py
    - ordering_consensus.py
    - window_assignment.py
    - coordination_cost.py
    - commit_wait.py
    - generate_all_figures.py

### Configuration (Ready for customization)

11. **configs/** (templates needed)
    - coordinator_config.yaml
    - worker_config.yaml

---

## ⚠️ What Needs Implementation

### Critical Path (Required for MVP)

#### 1. Worker Implementation (`src/worker.py`)

**Priority**: HIGHEST
**Estimated Time**: 2-3 hours

**Requirements**:
```python
class Worker:
    def __init__(self, node_id, listen_port, ntp_server, chronotick_server, output_file):
        # Initialize UDP listener
        # Initialize NTP client
        # Initialize ChronoTick client
        # Prepare CSV output

    def run(self):
        # Main loop:
        while True:
            # 1. Receive event from network
            event, receive_time_ns = self.listener.receive()

            # 2. Query NTP (cached for 10s)
            ntp_offset_ms, ntp_uncertainty_ms = self.ntp_client.query()

            # 3. Query ChronoTick (fresh every time)
            ct_offset_ms, ct_uncertainty_ms = self.chronotick_client.query()

            # 4. Calculate timestamps
            ntp_timestamp_ns = receive_time_ns + int(ntp_offset_ms * 1e6)
            ct_timestamp_ns = receive_time_ns + int(ct_offset_ms * 1e6)
            ct_lower_ns = ct_timestamp_ns - int(3 * ct_uncertainty_ms * 1e6)
            ct_upper_ns = ct_timestamp_ns + int(3 * ct_uncertainty_ms * 1e6)

            # 5. Record to CSV
            record = TimestampRecord(...)
            self.csv_writer.writerow(record.to_csv_row())

            # 6. Schedule commit-wait measurements (background task)
            # After 30s, 60s: record uncertainty again
```

**Implementation Notes**:
- Reuse logic from `server/scripts/chronotick_client_validation.py`
- NTP: Use ntplib or manual NTP protocol
- ChronoTick: HTTP POST to MCP server `/get_time` endpoint
- Commit-wait: Use asyncio or threading for delayed measurements

#### 2. NTP Client Implementation (in `common.py`)

**Priority**: HIGH
**Estimated Time**: 1 hour

**Current**: Placeholder that returns `(0.0, 10.0)`
**Needed**: Real NTP query

**Options**:
a) **Use ntplib** (easiest):
   ```python
   import ntplib
   client = ntplib.NTPClient()
   response = client.request(server, version=3)
   offset_ms = response.offset * 1000
   ```

b) **Use existing ChronoTick client**:
   - Import from `server/scripts/chronotick_client_validation.py`
   - Call `query_ntp_servers()` function

c) **Manual NTP protocol** (most control):
   - Implement RFC 5905
   - More complex but educational

**Recommendation**: Use option (b) - reuse existing validated code!

#### 3. ChronoTick Client Implementation (in `common.py`)

**Priority**: HIGH
**Estimated Time**: 30 minutes

**Current**: Placeholder HTTP POST
**Needed**: Real MCP client

**Implementation**:
```python
import requests

def query(self) -> Tuple[float, float]:
    response = requests.post(
        f"{self.server_url}/get_time",
        json={},
        timeout=5
    )
    data = response.json()
    return data['offset_ms'], data['uncertainty_ms']
```

**Note**: Requires ChronoTick MCP server running on ares master node!

#### 4. Analysis Scripts

**Priority**: MEDIUM
**Estimated Time**: 4-5 hours total

**Order of implementation**:
1. **causality_analysis.py** (1h) - Simplest, good starting point
2. **ordering_consensus.py** (1h) - Core metric
3. **window_assignment.py** (1.5h) - Most complex (multiple window sizes)
4. **coordination_cost.py** (30min) - Derived from others
5. **commit_wait.py** (1h) - Requires historical uncertainty data
6. **generate_all_figures.py** (30min) - Wrapper that calls all

**Template** (for each analysis script):
```python
def load_data(experiment_dir):
    """Load CSVs and merge on event_id"""
    coord = pd.read_csv(f"{experiment_dir}/coordinator.csv")
    worker_b = pd.read_csv(f"{experiment_dir}/worker_comp11.csv")
    worker_c = pd.read_csv(f"{experiment_dir}/worker_comp12.csv")
    return merge_dataframes(coord, worker_b, worker_c)

def analyze(experiment_dir):
    """Run analysis, return statistics dict"""
    df = load_data(experiment_dir)
    # Analysis logic here
    return {
        'ntp_violations': ...,
        'ct_violations': ...,
    }

def generate_figure(experiment_dir, output_dir):
    """Generate and save figure"""
    stats = analyze(experiment_dir)
    # Matplotlib plotting here
    plt.savefig(f"{output_dir}/causality_violations.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', required=True)
    args = parser.parse_args()

    stats = analyze(f"results/{args.experiment}")
    generate_figure(f"results/{args.experiment}", f"results/{args.experiment}/figures")
    print(json.dumps(stats, indent=2))
```

---

## 🔧 Quick Implementation Guide

### Step 1: Implement Worker (2-3 hours)

```bash
cd data-streaming-evaluations/src
# Create worker.py based on coordinator.py structure
# Reuse NTP logic from server/scripts/chronotick_client_validation.py
```

**Test locally**:
```bash
# Terminal 1: Start mock coordinator
python3 -c "
import socket, json, time
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
for i in range(10):
    event = {'event_id': i, 'coordinator_timestamp_ns': time.time_ns(), 'sequence_number': i}
    sock.sendto(json.dumps(event).encode(), ('localhost', 9000))
    time.sleep(1)
"

# Terminal 2: Run worker
uv run worker --node-id test --listen-port 9000 --output test.csv
```

### Step 2: Test on ARES (30 min)

```bash
# Deploy to ARES
ssh ares
cd /path/to/ChronoTick
git pull

# Test worker on one node
ssh ares "ssh ares-comp-11 'cd /path/to/data-streaming-evaluations && \
  uv run worker --node-id test --listen-port 9000 --output test.csv'"

# In another terminal, send test event from comp-18
ssh ares "ssh ares-comp-18 'cd /path/to/data-streaming-evaluations && \
  python3 -c \"
  import socket, json, time
  sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  event = {\"event_id\": 1, \"coordinator_timestamp_ns\": time.time_ns(), \"sequence_number\": 0}
  sock.sendto(json.dumps(event).encode(), (\"ares-comp-11\", 9000))
  \"'"
```

### Step 3: Full Test Deployment (15 min)

```bash
cd data-streaming-evaluations
./deploy.sh test-run-1

# Check results
ls -la results/test-run-1/
cat results/test-run-1/coordinator.csv | head
cat results/test-run-1/worker_comp11.csv | head
```

### Step 4: Implement Analysis (4-5 hours)

Start with simplest:

```bash
cd analysis
cp causality_analysis.py.template causality_analysis.py
# Implement analysis logic
uv run python causality_analysis.py --experiment test-run-1
```

Test incrementally:
1. Load data ✓
2. Calculate statistics ✓
3. Generate figure ✓
4. Repeat for other modules

---

## 📋 Configuration Files Needed

### coordinator_config.yaml

```yaml
experiment:
  num_events: 100
  duration_minutes: 10

broadcast_pattern:
  pattern: [slow, fast, fast, fast, fast, medium, fast, fast, fast, fast]
  fast_stream: 0.010
  medium_stream: 0.050
  slow_stream: 0.100

logging:
  level: INFO
  console: true
```

### worker_config.yaml

```yaml
network:
  listen_port: 9000
  buffer_size: 4096

timing:
  ntp_query_interval: 10
  chronotick_query_interval: 1
  commit_wait_delays: [30, 60]

chronotick:
  use_dual_model: true
  uncertainty_multiplier: 3

logging:
  level: INFO
  console: true
```

---

## 🎯 Recommended Implementation Order

### Phase 1: MVP (1 day, 6-8 hours)
1. ✅ **Worker implementation** (2-3h) - CRITICAL PATH
2. ✅ **NTP client** (1h) - Reuse existing code
3. ✅ **ChronoTick client** (30min) - Simple HTTP
4. ✅ **Local testing** (30min) - Verify basics work
5. ✅ **ARES testing** (1h) - Deploy and verify
6. ✅ **Fix bugs** (2h buffer) - Inevitable issues

**Deliverable**: Working end-to-end experiment that collects data

### Phase 2: Analysis (1 day, 4-5 hours)
1. ✅ **Causality analysis** (1h) - Simplest
2. ✅ **Ordering consensus** (1h) - Core metric
3. ✅ **Window assignment** (1.5h) - Most useful for paper
4. ✅ **Coordination cost** (30min) - Derived metric
5. ✅ **Commit-wait** (1h) - Future work

**Deliverable**: All figures and statistics generated

### Phase 3: Polish (0.5 day, 2-3 hours)
1. ✅ **Configuration files** (30min)
2. ✅ **Error handling** (1h)
3. ✅ **Documentation updates** (30min)
4. ✅ **Final testing** (1h)

**Deliverable**: Production-ready evaluation framework

---

## 🚀 Immediate Next Steps (Tonight/Tomorrow)

### Tonight (2-3 hours)
1. **Implement worker.py**
   - Copy structure from coordinator.py
   - Import NTP logic from chronotick_client_validation.py
   - Add ChronoTick HTTP client
   - Test locally with mock coordinator

2. **Test on ARES**
   - Deploy worker to ares-comp-11
   - Send test events from ares-comp-18
   - Verify CSV output looks correct

### Tomorrow Morning (2 hours)
3. **Full deployment test**
   - Run ./deploy.sh
   - Collect data
   - Verify all 100 events received

4. **Implement causality analysis**
   - Load data
   - Calculate violations
   - Generate figure

### Tomorrow Afternoon (3 hours)
5. **Implement remaining analysis modules**
   - Ordering consensus
   - Window assignment
   - Generate all figures

6. **Run real experiment**
   - Deploy full experiment
   - Generate all results
   - Review for paper

---

## 📊 What You'll Have After Implementation

### Immediate (after Phase 1)
- Working data collection on ARES
- Raw CSVs with all timing data
- Ground truth from coordinator
- 100 broadcast events, 200 worker timestamps

### After Phase 2
- 5 publication-ready figures
- Quantitative statistics (JSON)
- Auto-generated summary report
- Reproducible analysis pipeline

### Paper-Ready Outputs
- Causality: 18% → 0% violations
- Consensus: 72% → 100% agreement
- Coordination: -28.5% reduction
- Provable: 0% → 80% without communication

---

## 💡 Tips for Implementation

### Reuse Existing Code
- **NTP**: Copy from `server/scripts/chronotick_client_validation.py`
- **ChronoTick**: Simple HTTP POST, already working in tsfm/
- **CSV**: Use pandas for analysis (already in dependencies)
- **Figures**: Copy matplotlib style from `results/figures/consensus_zones/`

### Debugging Strategies
1. **Local first**: Test with mock coordinator/worker on laptop
2. **One node**: Test worker on single ARES node
3. **Two nodes**: Test coordinator → single worker
4. **Full deployment**: Only after components verified

### Common Pitfalls
- **UDP packet loss**: Expect ~1-2% loss, handle gracefully
- **NFS lag**: Don't assume immediate file visibility
- **Clock precision**: Use nanoseconds, convert to ms for display
- **Uncertainty staleness**: Cache NTP (10s), not ChronoTick (real-time)

---

## ✅ Success Criteria

### Deployment Success
- ✓ All 100 events sent by coordinator
- ✓ 98+ events received by each worker (allow some UDP loss)
- ✓ No crashes or errors
- ✓ CSVs have correct format

### Analysis Success
- ✓ All 5 figures generated
- ✓ Statistics match expected ranges
- ✓ No missing data points
- ✓ Figures tell clear story

### Paper-Ready
- ✓ NTP causality violations: 15-20%
- ✓ ChronoTick violations: 0%
- ✓ Consensus improvement: +25-30%
- ✓ Coordination reduction: 25-35%

---

## 🎯 Timeline Estimate

**Optimistic** (everything works first try): 1.5 days
**Realistic** (typical debugging): 2 days
**Pessimistic** (major issues): 3 days

**Recommended**: Plan for 2 days, aim to finish in 1.5, have 3 as buffer.

---

## 📞 Where to Get Help

- **Worker implementation**: See `coordinator.py` as template
- **NTP logic**: `server/scripts/chronotick_client_validation.py`
- **ChronoTick**: `tsfm/chronotick_mcp.py` (server side)
- **Analysis**: `scripts/crazy_ideas/consensus_zones_CORRECT.py` (similar logic)
- **Deployment**: DEPLOYMENT.md troubleshooting section

---

**Current Status**: Framework ready, implementation in progress
**Next Action**: Implement worker.py (2-3 hours)
**Goal**: Working end-to-end evaluation by tomorrow evening
