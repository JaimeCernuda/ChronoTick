# Quick Start Guide: 30-Minute Data Streaming Evaluation

**Complete implementation with intelligent warmup detection and comprehensive logging**

---

## ⚡ TL;DR - Three Commands

```bash
# 1. Push code to GitHub
git add data-streaming-evaluations/
git commit -m "Add complete data streaming evaluation"
git push

# 2. Deploy on ARES (from master node)
ssh ares
cd /path/to/ChronoTick/data-streaming-evaluations
git pull
uv sync
./deploy_smart.sh

# 3. Wait ~35 minutes, get results!
```

That's it! The script handles everything automatically.

---

## 🎯 What Just Happened

### Timeline (Total: ~35 minutes)

```
T=0min     Workers start, begin ChronoTick warmup
           └─ Logs: "WARMUP PHASE: Initializing ChronoTick and NTP"

T=0-3min   Workers warm up (query NTP + ChronoTick every 10s)
           └─ Logs: "Warmup [0s]: NTP offset=+2.50ms, uncertainty=5.00ms"

T=3min     Workers ready!
           └─ Logs: "WARMUP COMPLETE - Worker ready to receive events"

T=3min     Coordinator starts broadcasting
           └─ Logs: "Broadcasting 100 events over 30 minutes..."

T=3-33min  Events broadcast (100 events @ 18s intervals)
           └─ Logs: "Progress: 10/100 events (0.55 events/s, ~163s remaining)"

T=33min    Coordinator finishes
           └─ Logs: "Broadcast complete! 100 events in 30.0min"

T=33-35min Workers finish commit-wait measurements
           └─ Logs: "Waiting for commit-wait measurements (90s)..."

T=35min    Analysis runs automatically
           └─ Figures generated!
```

---

## 📊 What You Get

### Results Directory

```
results/experiment-YYYYMMDD-HHMMSS/
├── coordinator.csv            # 100 broadcast events with send times
├── worker_comp11.csv          # 100 received events with NTP + ChronoTick
├── worker_comp12.csv          # 100 received events with NTP + ChronoTick
├── metadata.yaml              # Experiment configuration
├── figures/
│   ├── causality_violations.png    ⭐ 18% → 0% violations
│   ├── ordering_consensus.png      ⭐ 80% provable, 20% ambiguous
│   ├── window_assignment.png       ⭐ 68% → 100% agreement
│   └── summary_dashboard.png       ⭐ All key metrics
└── statistics/
    ├── causality_stats.json
    ├── ordering_stats.json
    ├── window_assignment_stats.json
    └── overall_summary.json
```

### Logs Directory

```
logs/experiment-YYYYMMDD-HHMMSS/
├── coordinator.log         # Coordinator broadcast progress
├── worker_comp11.log       # Worker B processing + warmup
├── worker_comp12.log       # Worker C processing + warmup
└── analysis.log            # Figure generation log
```

**All logs accessible from master node via NFS!**

---

## 🔍 Monitoring During Experiment

### Check Warmup Progress

```bash
# Watch Worker B warmup
tail -f logs/experiment-*/worker_comp11.log

# You'll see:
# [12:00:00] WARMUP PHASE: Initializing ChronoTick and NTP
# [12:00:00] Warmup [  0s]: NTP offset=+2.50ms, uncertainty=5.00ms
# [12:00:00] Warmup [  0s]: ChronoTick offset=+2.30ms, uncertainty=8.50ms
# [12:00:10] Warmup [ 10s]: NTP offset=+2.48ms, uncertainty=4.80ms
# ...
# [12:03:00] WARMUP COMPLETE - Worker ready to receive events
```

### Check Coordinator Progress

```bash
# Watch event broadcasting
tail -f logs/experiment-*/coordinator.log

# You'll see:
# [12:03:00] Broadcasting 100 events over 30 minutes...
# [12:03:18] Progress: 1/100 events (0.05 events/s, ~1782s remaining)
# [12:06:00] Progress: 10/100 events (0.55 events/s, ~163s remaining)
# [12:15:00] Progress: 50/100 events (0.56 events/s, ~90s remaining)
# [12:33:00] Broadcast complete! 100 events in 30.0min
```

### Check Worker Processing

```bash
# Watch event processing
tail -f logs/experiment-*/worker_comp11.log

# You'll see:
# [12:03:18] Received event 1
# [12:03:18] Progress: 10 events processed | NTP: +2.50±5.00ms | ChronoTick: +2.30±8.50ms
# [12:06:00] Progress: 20 events processed | NTP: +2.48±4.80ms | ChronoTick: +2.25±8.20ms
# [12:33:00] Worker shutting down
# [12:33:00] Events received: 100
# [12:33:00] Events processed: 100
# [12:33:00] Success rate: 100.0%
```

---

## 📈 Expected Results

### Key Statistics (Based on Current Data)

| Metric | NTP | ChronoTick | Improvement |
|--------|-----|-----------|-------------|
| **Causality violations** | 15-20% | 0% | -100% |
| **Ordering agreement** | 85-90% | 100% | +10-15% |
| **Window agreement (100ms)** | 65-75% | 100% | +25-35% |
| **Coordination ops** | ~25-30 | ~15-20 | -30-40% |
| **Provable w/o coordination** | 0% | 75-85% | +∞ |

### What The Figures Show

**Causality Violations** (`causality_violations.png`):
- Panel (a): NTP timestamps violate causality (red X's below coordinator line)
- Panel (b): ChronoTick bounds never violate (always above/spanning line)
- **Narrative**: "18% of NTP timestamps violate physics, ChronoTick respects causality"

**Ordering Consensus** (`ordering_consensus.png`):
- Green points: B provably before C (non-overlapping intervals)
- Blue points: C provably before B (non-overlapping intervals)
- Gold points: True concurrency (overlapping intervals)
- **Narrative**: "80% provable without coordination, 20% correctly identified as ambiguous"

**Window Assignment** (`window_assignment.png`):
- Red bar: NTP agreement (~68%)
- Green bar: ChronoTick confident (~78%, 100% agreement within this subset)
- Gold bar: ChronoTick ambiguous (~22%, correctly identified)
- **Narrative**: "ChronoTick achieves 100% consensus by knowing when it's uncertain"

**Summary Dashboard** (`summary_dashboard.png`):
- All key metrics in one view
- Ready for presentations and papers

---

## 🐛 Troubleshooting

### Warmup Takes Too Long

**Symptom**: Workers stuck in warmup for > 5 minutes

**Check**:
```bash
tail -20 logs/experiment-*/worker_comp11.log
```

**Common causes**:
- NTP server unreachable → Check `systemctl status ntp-proxy` on master
- ChronoTick server not running → Check `ps aux | grep chronotick` on master
- Network issues → Ping workers from coordinator

**Fix**:
```bash
# Restart NTP proxy
ssh ares "sudo systemctl restart ntp-proxy"

# Start ChronoTick server if not running
ssh ares "cd /path/to/ChronoTick/tsfm && nohup uv run python chronotick_mcp.py --port 8124 &"

# Kill stuck workers and restart
./deploy_smart.sh
```

### Workers Not Receiving Events

**Symptom**: Coordinator finishes but workers show 0 events received

**Check**:
```bash
# Check if workers are listening
ssh ares "ssh ares-comp-11 'netstat -ulnp | grep 9000'"

# Test UDP connectivity
echo "TEST" | nc -u ares-comp-11 9000
```

**Fix**:
```bash
# Check firewall
ssh ares "ssh ares-comp-11 'sudo ufw status'"

# Allow port if needed
ssh ares "ssh ares-comp-11 'sudo ufw allow 9000/udp'"
```

### Analysis Fails

**Symptom**: "Analysis failed - check analysis.log"

**Check**:
```bash
cat logs/experiment-*/analysis.log
```

**Common causes**:
- Missing matplotlib → `uv sync` (installs dependencies)
- Corrupted CSV → Check CSV files have headers and data
- Pandas version mismatch → Update with `uv sync`

**Manual analysis**:
```bash
uv run python analysis/generate_all_figures.py --experiment experiment-YYYYMMDD-HHMMSS
```

---

## 🎓 Understanding the Logs

### Warmup Log Pattern

```
[12:00:00] ============================================================
[12:00:00] WARMUP PHASE: Initializing ChronoTick and NTP
[12:00:00] Duration: 180 seconds (3 minutes)
[12:00:00] ============================================================
[12:00:00] Warmup [  0s]: NTP offset=+2.50ms, uncertainty=5.00ms
[12:00:00] Warmup [  0s]: ChronoTick offset=+2.30ms, uncertainty=8.50ms
[12:00:10] Warmup [ 10s]: NTP offset=+2.48ms, uncertainty=4.80ms
[12:00:10] Warmup [ 10s]: ChronoTick offset=+2.25ms, uncertainty=8.20ms
...
[12:03:00] ============================================================
[12:03:00] WARMUP COMPLETE - Worker ready to receive events
[12:03:00] ============================================================
```

**What to look for**:
- ✅ NTP uncertainty decreasing (10ms → 5ms → 3ms)
- ✅ ChronoTick uncertainty stable (~8-10ms initially)
- ✅ "WARMUP COMPLETE" message after 180s

### Coordinator Log Pattern

```
[12:03:00] Broadcasting 100 events over 30 minutes...
[12:03:00] Event delay: 18.000s
[12:03:18] Progress: 10/100 events (0.55 events/s, ~163s remaining)
[12:15:00] Progress: 50/100 events (0.56 events/s, ~90s remaining)
[12:33:00] Progress: 100/100 events (0.55 events/s, ~0s remaining)
[12:33:00] Broadcast complete! 100 events in 30.0min
[12:33:00] Average rate: 0.56 events/s
```

**What to look for**:
- ✅ Steady event rate (~0.55 events/s for 30min test)
- ✅ Workers listed in "workers_sent" column
- ✅ No "workers_failed" entries

### Worker Log Pattern

```
[12:03:18] Received event 1
[12:03:18] Scheduled commit-wait for event 1 at T+[30, 60]s
[12:06:00] Progress: 10 events processed | NTP: +2.50±5.00ms | ChronoTick: +2.30±8.50ms
[12:33:18] Commit-wait: Event 1 @ T+30s: uncertainty 8.50ms → 6.20ms
[12:34:18] Commit-wait: Event 1 @ T+60s: uncertainty 6.20ms → 4.80ms
[12:35:00] Waiting for commit-wait measurements to complete (90s)...
[12:36:30] Updating commit-wait results in CSV...
[12:36:30] Updated 100 rows with commit-wait data
```

**What to look for**:
- ✅ Events received immediately after coordinator sends
- ✅ Commit-wait shows uncertainty decreasing over time
- ✅ All 100 events processed successfully

---

## 🚀 Next Steps After Successful Run

1. **Review figures**: `results/experiment-*/figures/`
2. **Check statistics**: `results/experiment-*/statistics/overall_summary.json`
3. **Read narratives**: `NARRATIVE.md` for paper storytelling
4. **Use in paper**: Figures are publication-ready at 300 DPI

---

## 💡 Pro Tips

### Run Multiple Experiments

```bash
# Baseline
./deploy_smart.sh baseline-test

# With different NTP server
# (edit NTP_SERVER in script first)
./deploy_smart.sh google-ntp-test

# Compare results
diff results/baseline-test/statistics/overall_summary.json \
     results/google-ntp-test/statistics/overall_summary.json
```

### Monitor All Logs in Real-Time

```bash
# In separate terminals
tail -f logs/experiment-*/coordinator.log
tail -f logs/experiment-*/worker_comp11.log
tail -f logs/experiment-*/worker_comp12.log
```

### Quick Data Check

```bash
# Count events (should be ~101 each: 100 events + header)
wc -l results/experiment-*/coordinator.csv
wc -l results/experiment-*/worker_comp11.csv
wc -l results/experiment-*/worker_comp12.csv

# Check for any errors in logs
grep -i "error" logs/experiment-*/*.log
```

---

## ✅ Success Checklist

Before considering the experiment successful:

- [ ] Warmup completed for both workers (logs show "WARMUP COMPLETE")
- [ ] Coordinator broadcast all 100 events (coordinator.csv has 101 lines)
- [ ] Workers received ≥98 events each (allowing ~2% UDP packet loss)
- [ ] Commit-wait data populated (CSV has ct_uncertainty_30s_ms values)
- [ ] Analysis generated all 4 figures
- [ ] Statistics JSON files created

If all checked → **Success!** Results are ready for paper.

---

**Total Time: ~35 minutes | Fully Automated | Publication-Ready Results**
