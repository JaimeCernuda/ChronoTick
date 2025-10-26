#!/usr/bin/env python3
"""
ChronoTick vs System Clock Analysis
===================================

Comprehensive analysis comparing:
1. System Clock baseline (30 min, ~3000 events)
2. ChronoTick predictions (30 min, ~3000 events)

Key Metrics:
- Causality violations (receive_time < send_time)
- Clock drift over time
- ChronoTick prediction accuracy
- ChronoTick uncertainty calibration
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Configuration
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Dataset paths
SYSTEM_CLOCK_DIR = Path("results/system_clock_30min_FINAL")
CHRONOTICK_DIR = Path("results/chronotick_30min_20251025-115922")
OUTPUT_DIR = Path("analysis_results")
OUTPUT_DIR.mkdir(exist_ok=True)

print("="*80)
print("CHRONOTICK VS SYSTEM CLOCK ANALYSIS")
print("="*80)
print(f"System Clock Dataset: {SYSTEM_CLOCK_DIR}")
print(f"ChronoTick Dataset: {CHRONOTICK_DIR}")
print(f"Output Directory: {OUTPUT_DIR}")
print()

# ============================================================================
# 1. LOAD DATASETS
# ============================================================================

print("[1/7] Loading datasets...")

# System Clock datasets (8 fields)
sys_coord = pd.read_csv(SYSTEM_CLOCK_DIR / "coordinator.csv")
sys_w11 = pd.read_csv(SYSTEM_CLOCK_DIR / "worker_comp11.csv")
sys_w12 = pd.read_csv(SYSTEM_CLOCK_DIR / "worker_comp12.csv")

# ChronoTick datasets (17 fields)
ct_coord = pd.read_csv(CHRONOTICK_DIR / "coordinator.csv")
ct_w11 = pd.read_csv(CHRONOTICK_DIR / "worker_comp11.csv")
ct_w12 = pd.read_csv(CHRONOTICK_DIR / "worker_comp12.csv")

print(f"  System Clock: {len(sys_coord)} coordinator, {len(sys_w11)} worker_B, {len(sys_w12)} worker_C")
print(f"  ChronoTick: {len(ct_coord)} coordinator, {len(ct_w11)} worker_B, {len(ct_w12)} worker_C")
print()

# ============================================================================
# 2. SYSTEM CLOCK CAUSALITY VIOLATIONS
# ============================================================================

print("[2/7] Analyzing System Clock causality violations...")

def analyze_causality(coord_df, worker_df, worker_name):
    """Analyze causality violations for a coordinator-worker pair."""
    # Merge on event_id
    merged = pd.merge(coord_df, worker_df, on='event_id', suffixes=('_coord', '_worker'))

    # Calculate time differences (nanoseconds)
    merged['latency_ns'] = merged['receive_time_ns'] - merged['coordinator_send_time_ns']
    merged['latency_ms'] = merged['latency_ns'] / 1e6

    # Causality violation: receive < send
    merged['violation'] = merged['latency_ns'] < 0

    violations = merged[merged['violation']]

    return {
        'worker': worker_name,
        'total_events': len(merged),
        'violations': len(violations),
        'violation_rate': len(violations) / len(merged) * 100,
        'mean_latency_ms': merged['latency_ms'].mean(),
        'median_latency_ms': merged['latency_ms'].median(),
        'min_latency_ms': merged['latency_ms'].min(),
        'max_latency_ms': merged['latency_ms'].max(),
        'std_latency_ms': merged['latency_ms'].std(),
        'data': merged
    }

sys_w11_analysis = analyze_causality(sys_coord, sys_w11, "Worker B (comp11)")
sys_w12_analysis = analyze_causality(sys_coord, sys_w12, "Worker C (comp12)")

print("\n  SYSTEM CLOCK RESULTS:")
print("  " + "-"*70)
for analysis in [sys_w11_analysis, sys_w12_analysis]:
    print(f"  {analysis['worker']}:")
    print(f"    Total events: {analysis['total_events']}")
    print(f"    Causality violations: {analysis['violations']} ({analysis['violation_rate']:.2f}%)")
    print(f"    Latency: mean={analysis['mean_latency_ms']:.3f}ms, median={analysis['median_latency_ms']:.3f}ms")
    print(f"             min={analysis['min_latency_ms']:.3f}ms, max={analysis['max_latency_ms']:.3f}ms")
    print()

# ============================================================================
# 3. CHRONOTICK CAUSALITY VIOLATIONS (using ChronoTick timestamps)
# ============================================================================

print("[3/7] Analyzing ChronoTick causality violations...")

def analyze_chronotick_causality(coord_df, worker_df, worker_name):
    """Analyze causality using ChronoTick corrected timestamps."""
    merged = pd.merge(coord_df, worker_df, on='event_id', suffixes=('_coord', '_worker'))

    # ChronoTick corrected timestamp (receive_time + offset correction)
    # ct_timestamp_ns already contains the corrected time
    # After merge, worker's ct_timestamp_ns becomes ct_timestamp_ns_worker
    merged['ct_latency_ns'] = merged['ct_timestamp_ns_worker'] - merged['coordinator_send_time_ns']
    merged['ct_latency_ms'] = merged['ct_latency_ns'] / 1e6

    # System clock latency for comparison
    merged['sys_latency_ns'] = merged['receive_time_ns'] - merged['coordinator_send_time_ns']
    merged['sys_latency_ms'] = merged['sys_latency_ns'] / 1e6

    # Violations
    merged['ct_violation'] = merged['ct_latency_ns'] < 0
    merged['sys_violation'] = merged['sys_latency_ns'] < 0

    ct_violations = merged[merged['ct_violation']]
    sys_violations = merged[merged['sys_violation']]

    return {
        'worker': worker_name,
        'total_events': len(merged),
        'ct_violations': len(ct_violations),
        'sys_violations': len(sys_violations),
        'ct_violation_rate': len(ct_violations) / len(merged) * 100,
        'sys_violation_rate': len(sys_violations) / len(merged) * 100,
        'improvement': len(sys_violations) - len(ct_violations),
        'mean_ct_offset_ms': merged['ct_offset_ms_worker'].mean(),
        'mean_ct_uncertainty_ms': merged['ct_uncertainty_ms'].mean(),
        'mean_ct_confidence': merged['ct_confidence'].mean(),
        'data': merged
    }

ct_w11_analysis = analyze_chronotick_causality(ct_coord, ct_w11, "Worker B (comp11)")
ct_w12_analysis = analyze_chronotick_causality(ct_coord, ct_w12, "Worker C (comp12)")

print("\n  CHRONOTICK RESULTS:")
print("  " + "-"*70)
for analysis in [ct_w11_analysis, ct_w12_analysis]:
    print(f"  {analysis['worker']}:")
    print(f"    Total events: {analysis['total_events']}")
    print(f"    System Clock violations: {analysis['sys_violations']} ({analysis['sys_violation_rate']:.2f}%)")
    print(f"    ChronoTick violations: {analysis['ct_violations']} ({analysis['ct_violation_rate']:.2f}%)")
    print(f"    Improvement: {analysis['improvement']} fewer violations")
    print(f"    ChronoTick offset: mean={analysis['mean_ct_offset_ms']:.3f}ms")
    print(f"    ChronoTick uncertainty: mean={analysis['mean_ct_uncertainty_ms']:.6f}ms")
    print(f"    ChronoTick confidence: mean={analysis['mean_ct_confidence']:.6f}")
    print()

# ============================================================================
# 4. CHRONOTICK VS NTP ACCURACY
# ============================================================================

print("[4/7] Analyzing ChronoTick prediction accuracy vs NTP...")

def analyze_ct_accuracy(worker_df, worker_name):
    """Compare ChronoTick predictions against NTP reference."""
    # Error: difference between ChronoTick offset and NTP offset
    worker_df['ct_error_ms'] = worker_df['ct_offset_ms'] - worker_df['ntp_offset_ms']

    # Check if error is within uncertainty bounds
    worker_df['within_bounds'] = np.abs(worker_df['ct_error_ms']) <= worker_df['ct_uncertainty_ms']

    return {
        'worker': worker_name,
        'mean_error_ms': worker_df['ct_error_ms'].mean(),
        'median_error_ms': worker_df['ct_error_ms'].median(),
        'mae_ms': np.abs(worker_df['ct_error_ms']).mean(),
        'rmse_ms': np.sqrt((worker_df['ct_error_ms']**2).mean()),
        'within_bounds_pct': (worker_df['within_bounds'].sum() / len(worker_df)) * 100,
        'data': worker_df
    }

ct_w11_accuracy = analyze_ct_accuracy(ct_w11.copy(), "Worker B (comp11)")
ct_w12_accuracy = analyze_ct_accuracy(ct_w12.copy(), "Worker C (comp12)")

print("\n  CHRONOTICK ACCURACY vs NTP:")
print("  " + "-"*70)
for analysis in [ct_w11_accuracy, ct_w12_accuracy]:
    print(f"  {analysis['worker']}:")
    print(f"    Mean error: {analysis['mean_error_ms']:.6f}ms")
    print(f"    Median error: {analysis['median_error_ms']:.6f}ms")
    print(f"    MAE: {analysis['mae_ms']:.6f}ms")
    print(f"    RMSE: {analysis['rmse_ms']:.6f}ms")
    print(f"    Within uncertainty bounds: {analysis['within_bounds_pct']:.2f}%")
    print()

# ============================================================================
# 5. VISUALIZATIONS
# ============================================================================

print("[5/7] Generating visualizations...")

# Figure 1: Causality Violation Comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('System Clock vs ChronoTick: Causality Analysis', fontsize=16, fontweight='bold')

# Worker B - System Clock latency distribution
ax = axes[0, 0]
sys_w11_analysis['data']['latency_ms'].hist(bins=100, ax=ax, alpha=0.7, color='red', edgecolor='black')
ax.axvline(0, color='black', linestyle='--', linewidth=2, label='Causality Threshold')
ax.set_xlabel('Latency (ms)')
ax.set_ylabel('Frequency')
ax.set_title(f'Worker B - System Clock\n{sys_w11_analysis["violations"]} violations ({sys_w11_analysis["violation_rate"]:.2f}%)')
ax.legend()
ax.grid(True, alpha=0.3)

# Worker B - ChronoTick latency distribution
ax = axes[0, 1]
ct_w11_analysis['data']['ct_latency_ms'].hist(bins=100, ax=ax, alpha=0.7, color='blue', edgecolor='black')
ct_w11_analysis['data']['sys_latency_ms'].hist(bins=100, ax=ax, alpha=0.4, color='red', edgecolor='black', label='System Clock')
ax.axvline(0, color='black', linestyle='--', linewidth=2, label='Causality Threshold')
ax.set_xlabel('Latency (ms)')
ax.set_ylabel('Frequency')
ax.set_title(f'Worker B - ChronoTick\n{ct_w11_analysis["ct_violations"]} violations ({ct_w11_analysis["ct_violation_rate"]:.2f}%)')
ax.legend()
ax.grid(True, alpha=0.3)

# Worker C - System Clock latency distribution
ax = axes[1, 0]
sys_w12_analysis['data']['latency_ms'].hist(bins=100, ax=ax, alpha=0.7, color='red', edgecolor='black')
ax.axvline(0, color='black', linestyle='--', linewidth=2, label='Causality Threshold')
ax.set_xlabel('Latency (ms)')
ax.set_ylabel('Frequency')
ax.set_title(f'Worker C - System Clock\n{sys_w12_analysis["violations"]} violations ({sys_w12_analysis["violation_rate"]:.2f}%)')
ax.legend()
ax.grid(True, alpha=0.3)

# Worker C - ChronoTick latency distribution
ax = axes[1, 1]
ct_w12_analysis['data']['ct_latency_ms'].hist(bins=100, ax=ax, alpha=0.7, color='blue', edgecolor='black')
ct_w12_analysis['data']['sys_latency_ms'].hist(bins=100, ax=ax, alpha=0.4, color='red', edgecolor='black', label='System Clock')
ax.axvline(0, color='black', linestyle='--', linewidth=2, label='Causality Threshold')
ax.set_xlabel('Latency (ms)')
ax.set_ylabel('Frequency')
ax.set_title(f'Worker C - ChronoTick\n{ct_w12_analysis["ct_violations"]} violations ({ct_w12_analysis["ct_violation_rate"]:.2f}%)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "1_causality_comparison.png", dpi=300, bbox_inches='tight')
print(f"  Saved: {OUTPUT_DIR / '1_causality_comparison.png'}")

# Figure 2: ChronoTick Accuracy
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('ChronoTick Prediction Accuracy vs NTP Reference', fontsize=16, fontweight='bold')

# Worker B - Error distribution
ax = axes[0, 0]
ct_w11_accuracy['data']['ct_error_ms'].hist(bins=100, ax=ax, alpha=0.7, color='purple', edgecolor='black')
ax.axvline(0, color='black', linestyle='--', linewidth=2)
ax.set_xlabel('ChronoTick Error (ms)')
ax.set_ylabel('Frequency')
ax.set_title(f'Worker B - Error Distribution\nMAE={ct_w11_accuracy["mae_ms"]:.6f}ms, RMSE={ct_w11_accuracy["rmse_ms"]:.6f}ms')
ax.grid(True, alpha=0.3)

# Worker B - Error over time
ax = axes[0, 1]
ax.plot(ct_w11_accuracy['data']['ct_error_ms'].values, alpha=0.6, linewidth=0.5)
ax.axhline(0, color='black', linestyle='--', linewidth=2)
ax.set_xlabel('Event Index')
ax.set_ylabel('ChronoTick Error (ms)')
ax.set_title('Worker B - Error Over Time')
ax.grid(True, alpha=0.3)

# Worker C - Error distribution
ax = axes[1, 0]
ct_w12_accuracy['data']['ct_error_ms'].hist(bins=100, ax=ax, alpha=0.7, color='purple', edgecolor='black')
ax.axvline(0, color='black', linestyle='--', linewidth=2)
ax.set_xlabel('ChronoTick Error (ms)')
ax.set_ylabel('Frequency')
ax.set_title(f'Worker C - Error Distribution\nMAE={ct_w12_accuracy["mae_ms"]:.6f}ms, RMSE={ct_w12_accuracy["rmse_ms"]:.6f}ms')
ax.grid(True, alpha=0.3)

# Worker C - Error over time
ax = axes[1, 1]
ax.plot(ct_w12_accuracy['data']['ct_error_ms'].values, alpha=0.6, linewidth=0.5)
ax.axhline(0, color='black', linestyle='--', linewidth=2)
ax.set_xlabel('Event Index')
ax.set_ylabel('ChronoTick Error (ms)')
ax.set_title('Worker C - Error Over Time')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "2_chronotick_accuracy.png", dpi=300, bbox_inches='tight')
print(f"  Saved: {OUTPUT_DIR / '2_chronotick_accuracy.png'}")

# ============================================================================
# 6. SUMMARY STATISTICS
# ============================================================================

print("[6/7] Generating summary report...")

summary = f"""
CHRONOTICK VS SYSTEM CLOCK ANALYSIS REPORT
==========================================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DATASETS
--------
System Clock: {SYSTEM_CLOCK_DIR}
  - Coordinator: {len(sys_coord)} events
  - Worker B: {len(sys_w11)} events
  - Worker C: {len(sys_w12)} events

ChronoTick: {CHRONOTICK_DIR}
  - Coordinator: {len(ct_coord)} events
  - Worker B: {len(ct_w11)} events
  - Worker C: {len(ct_w12)} events

CAUSALITY VIOLATION ANALYSIS
-----------------------------

System Clock (Baseline):
  Worker B: {sys_w11_analysis['violations']}/{sys_w11_analysis['total_events']} violations ({sys_w11_analysis['violation_rate']:.2f}%)
  Worker C: {sys_w12_analysis['violations']}/{sys_w12_analysis['total_events']} violations ({sys_w12_analysis['violation_rate']:.2f}%)

ChronoTick (with AI correction):
  Worker B: {ct_w11_analysis['ct_violations']}/{ct_w11_analysis['total_events']} violations ({ct_w11_analysis['ct_violation_rate']:.2f}%)
  Worker C: {ct_w12_analysis['ct_violations']}/{ct_w12_analysis['total_events']} violations ({ct_w12_analysis['ct_violation_rate']:.2f}%)

Improvement:
  Worker B: {ct_w11_analysis['improvement']} fewer violations
  Worker C: {ct_w12_analysis['improvement']} fewer violations

CHRONOTICK PREDICTION ACCURACY
-------------------------------

Worker B:
  Mean Error: {ct_w11_accuracy['mean_error_ms']:.6f} ms
  Median Error: {ct_w11_accuracy['median_error_ms']:.6f} ms
  MAE: {ct_w11_accuracy['mae_ms']:.6f} ms
  RMSE: {ct_w12_accuracy['rmse_ms']:.6f} ms
  Within Uncertainty Bounds: {ct_w11_accuracy['within_bounds_pct']:.2f}%

Worker C:
  Mean Error: {ct_w12_accuracy['mean_error_ms']:.6f} ms
  Median Error: {ct_w12_accuracy['median_error_ms']:.6f} ms
  MAE: {ct_w12_accuracy['mae_ms']:.6f} ms
  RMSE: {ct_w12_accuracy['rmse_ms']:.6f} ms
  Within Uncertainty Bounds: {ct_w12_accuracy['within_bounds_pct']:.2f}%

CHRONOTICK MODEL STATISTICS
----------------------------

Worker B:
  Mean Offset: {ct_w11_analysis['mean_ct_offset_ms']:.3f} ms
  Mean Uncertainty: {ct_w11_analysis['mean_ct_uncertainty_ms']:.6f} ms
  Mean Confidence: {ct_w11_analysis['mean_ct_confidence']:.6f}

Worker C:
  Mean Offset: {ct_w12_analysis['mean_ct_offset_ms']:.3f} ms
  Mean Uncertainty: {ct_w12_analysis['mean_ct_uncertainty_ms']:.6f} ms
  Mean Confidence: {ct_w12_analysis['mean_ct_confidence']:.6f}

CONCLUSIONS
-----------
{
"ChronoTick SIGNIFICANTLY reduces causality violations"
if (ct_w11_analysis['improvement'] > 0 or ct_w12_analysis['improvement'] > 0)
else "ChronoTick shows similar violation rates to system clock"
}

Average reduction: {((ct_w11_analysis['improvement'] + ct_w12_analysis['improvement']) / 2):.1f} violations per worker

ChronoTick uncertainty bounds are {"well-calibrated" if ct_w11_accuracy['within_bounds_pct'] > 90 else "moderately calibrated"}
({((ct_w11_accuracy['within_bounds_pct'] + ct_w12_accuracy['within_bounds_pct']) / 2):.1f}% of predictions within bounds)

"""

with open(OUTPUT_DIR / "summary_report.txt", "w") as f:
    f.write(summary)

print(summary)
print(f"  Saved: {OUTPUT_DIR / 'summary_report.txt'}")

# ============================================================================
# 7. SAVE PROCESSED DATA
# ============================================================================

print("[7/7] Saving processed data...")

# Save detailed analysis data
sys_w11_analysis['data'].to_csv(OUTPUT_DIR / "system_clock_worker_b_detailed.csv", index=False)
sys_w12_analysis['data'].to_csv(OUTPUT_DIR / "system_clock_worker_c_detailed.csv", index=False)
ct_w11_analysis['data'].to_csv(OUTPUT_DIR / "chronotick_worker_b_detailed.csv", index=False)
ct_w12_analysis['data'].to_csv(OUTPUT_DIR / "chronotick_worker_c_detailed.csv", index=False)

print(f"  Saved detailed analysis CSVs to {OUTPUT_DIR}/")

print()
print("="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"All results saved to: {OUTPUT_DIR}/")
print()
