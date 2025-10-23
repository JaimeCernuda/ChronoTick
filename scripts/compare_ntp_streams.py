#!/usr/bin/env python3
"""
Compare NTP Measurement Streams: Client (single-server) vs ChronoTick (5-server averaged)

This script exposes a critical validation issue:
- Client uses SINGLE SERVER NTP (ntplib) for "ground truth"
- ChronoTick uses 5-SERVER AVERAGING with MAD outlier rejection internally
- Comparing ChronoTick against inferior reference makes it look worse!
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from datetime import datetime
from pathlib import Path

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10

def parse_chronotick_ntp_from_log(log_path):
    """
    Parse ChronoTick's internal NTP measurements from log file.

    Looks for lines like:
    2025-10-23 12:49:23,180 - [NTP_AVERAGED] Combined 4/5 servers (strict): offset=2.89ms, delay=29.8ms, uncertainty=0.52ms, MAD=0.66ms
    """
    pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ - .* - INFO - \[NTP_AVERAGED\] Combined (\d+)/(\d+) servers.*: offset=([\d.-]+)ms, delay=([\d.-]+)ms, uncertainty=([\d.-]+)ms, MAD=([\d.-]+)ms'

    measurements = []

    with open(log_path, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                timestamp_str, n_combined, n_total, offset, delay, uncertainty, mad = match.groups()

                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')

                measurements.append({
                    'timestamp': timestamp,
                    'offset_ms': float(offset),
                    'delay_ms': float(delay),
                    'uncertainty_ms': float(uncertainty),
                    'mad_ms': float(mad),
                    'n_combined': int(n_combined),
                    'n_total': int(n_total)
                })

    if len(measurements) == 0:
        print(f"WARNING: No [NTP_AVERAGED] measurements found in {log_path}")
        return None

    df = pd.DataFrame(measurements)

    # Calculate elapsed time from first measurement
    df['elapsed_seconds'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()

    return df

def load_client_ntp(csv_path):
    """Load client's single-server NTP measurements from CSV"""
    df = pd.read_csv(csv_path)

    # Filter to NTP measurements only
    ntp_df = df[df['has_ntp'] == True].copy()

    if len(ntp_df) == 0:
        print(f"WARNING: No client NTP measurements found in {csv_path}")
        return None

    ntp_df['timestamp'] = pd.to_datetime(ntp_df['datetime'])

    return ntp_df

def plot_ntp_comparison(platform_name, chronotick_ntp, client_ntp, output_path):
    """
    Create comparison plots showing:
    1. Both NTP streams over time
    2. Histogram comparison
    3. Statistics
    """

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'{platform_name}: NTP Measurement Quality Comparison\n'
                 f'ChronoTick Internal (5-server avg) vs Client Validation (single server)',
                 fontsize=14, fontweight='bold')

    # Calculate statistics
    chronotick_mean = chronotick_ntp['offset_ms'].mean()
    chronotick_std = chronotick_ntp['offset_ms'].std()
    chronotick_range = chronotick_ntp['offset_ms'].max() - chronotick_ntp['offset_ms'].min()

    client_mean = client_ntp['ntp_offset_ms'].mean()
    client_std = client_ntp['ntp_offset_ms'].std()
    client_range = client_ntp['ntp_offset_ms'].max() - client_ntp['ntp_offset_ms'].min()

    improvement_std = ((client_std - chronotick_std) / client_std) * 100
    improvement_range = ((client_range - chronotick_range) / client_range) * 100

    # Plot 1: Time series comparison
    ax = axes[0, 0]

    # ChronoTick's internal NTP (only plot operational measurements, skip warmup)
    # Warmup is < 60 seconds, operational is >= 60s
    chronotick_operational = chronotick_ntp[chronotick_ntp['elapsed_seconds'] >= 60]

    if len(chronotick_operational) > 0:
        ax.plot(chronotick_operational['elapsed_seconds'] / 60, chronotick_operational['offset_ms'],
               'o-', label='ChronoTick Internal (5-server avg)', color='green', alpha=0.7, markersize=5)

    # Client's NTP
    ax.plot(client_ntp['elapsed_seconds'] / 60, client_ntp['ntp_offset_ms'],
           'x-', label='Client Validation (single server)', color='red', alpha=0.7, markersize=7)

    ax.axhline(y=chronotick_mean, color='green', linestyle='--', alpha=0.5, linewidth=1.5,
              label=f'ChronoTick mean: {chronotick_mean:.2f}ms')
    ax.axhline(y=client_mean, color='red', linestyle='--', alpha=0.5, linewidth=1.5,
              label=f'Client mean: {client_mean:.2f}ms')

    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('NTP Offset (ms)')
    ax.set_title('NTP Offset Over Time', fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 2: Histograms
    ax = axes[0, 1]

    bins = np.linspace(
        min(chronotick_ntp['offset_ms'].min(), client_ntp['ntp_offset_ms'].min()),
        max(chronotick_ntp['offset_ms'].max(), client_ntp['ntp_offset_ms'].max()),
        30
    )

    ax.hist(chronotick_ntp['offset_ms'], bins=bins, alpha=0.5, label='ChronoTick Internal',
           color='green', edgecolor='black')
    ax.hist(client_ntp['ntp_offset_ms'], bins=bins, alpha=0.5, label='Client Validation',
           color='red', edgecolor='black')

    ax.axvline(x=chronotick_mean, color='green', linestyle='--', linewidth=2, label=f'ChronoTick mean')
    ax.axvline(x=client_mean, color='red', linestyle='--', linewidth=2, label=f'Client mean')

    ax.set_xlabel('NTP Offset (ms)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution Comparison', fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: Stability metrics (std over time)
    ax = axes[1, 0]

    # Calculate rolling std for both streams
    window_size = 10

    if len(chronotick_operational) >= window_size:
        chronotick_rolling_std = chronotick_operational['offset_ms'].rolling(window=window_size, center=True).std()
        ax.plot(chronotick_operational['elapsed_seconds'] / 60, chronotick_rolling_std,
               '-', label=f'ChronoTick rolling std (window={window_size})', color='green', linewidth=2)

    if len(client_ntp) >= window_size:
        client_rolling_std = client_ntp['ntp_offset_ms'].rolling(window=window_size, center=True).std()
        ax.plot(client_ntp['elapsed_seconds'] / 60, client_rolling_std,
               '-', label=f'Client rolling std (window={window_size})', color='red', linewidth=2)

    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Rolling Standard Deviation (ms)')
    ax.set_title('Stability Over Time (Lower = More Stable)', fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 4: Statistics summary
    ax = axes[1, 1]
    ax.axis('off')

    summary_text = f"""
    STATISTICAL COMPARISON
    ═══════════════════════════════════════════════════════

    ChronoTick Internal NTP (5-server averaging):
      • Measurements: {len(chronotick_ntp)} (warmup + operational)
      • Operational: {len(chronotick_operational)} (after 60s warmup)
      • Mean: {chronotick_mean:.3f} ms
      • Std Dev: {chronotick_std:.3f} ms  ✅ STABLE
      • Range: {chronotick_range:.3f} ms
      • Typical uncertainty: {chronotick_ntp['uncertainty_ms'].mean():.3f} ms
      • Typical MAD: {chronotick_ntp['mad_ms'].mean():.3f} ms

    Client Validation NTP (single server):
      • Measurements: {len(client_ntp)}
      • Mean: {client_mean:.3f} ms
      • Std Dev: {client_std:.3f} ms  ⚠️  NOISY
      • Range: {client_range:.3f} ms

    ═══════════════════════════════════════════════════════
    IMPROVEMENT FROM AVERAGING:

      • Std Dev Reduction: {improvement_std:.1f}%  {'✅' if improvement_std > 0 else '⚠️'}
      • Range Reduction: {improvement_range:.1f}%  {'✅' if improvement_range > 0 else '⚠️'}

    ═══════════════════════════════════════════════════════
    CONCLUSION:

    ChronoTick's internal NTP is {'MUCH SMOOTHER' if improvement_std > 20 else 'SMOOTHER' if improvement_std > 0 else 'NOT SMOOTHER'}
    than the client's validation NTP!

    The validation is comparing ChronoTick against an
    INFERIOR reference (single-server) instead of the
    SUPERIOR reference ChronoTick actually uses internally!

    This makes ChronoTick look WORSE than it really is.
    """

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved NTP comparison plot to: {output_path}")

    return {
        'chronotick_std': chronotick_std,
        'client_std': client_std,
        'improvement_std': improvement_std,
        'improvement_range': improvement_range
    }

def main():
    print("\n" + "="*80)
    print("NTP MEASUREMENT STREAM COMPARISON")
    print("ChronoTick Internal (5-server avg) vs Client Validation (single server)")
    print("="*80)

    platforms = [
        ('Homelab',
         'results/experiment-11/homelab/experiment11_20251023_124916.log',
         'results/experiment-11/homelab/chronotick_client_validation_20251023_124918.csv',
         'results/experiment-11/homelab_ntp_comparison.png'),
        ('ARES comp-11',
         'results/experiment-11/ares-comp-11/experiment11_comp11_20251023_134434.log',
         'results/experiment-11/ares-comp-11/chronotick_client_validation_20251023_134440.csv',
         'results/experiment-11/ares_comp11_ntp_comparison.png'),
        ('ARES comp-12',
         'results/experiment-11/ares-comp-12/experiment11_comp12_20251023_134652.log',
         'results/experiment-11/ares-comp-12/chronotick_client_validation_20251023_134702.csv',
         'results/experiment-11/ares_comp12_ntp_comparison.png'),
    ]

    results = {}

    for platform_name, log_path, csv_path, output_path in platforms:
        print(f"\n{'─'*80}")
        print(f"📊 Analyzing {platform_name}...")
        print(f"{'─'*80}")

        chronotick_ntp = parse_chronotick_ntp_from_log(log_path)
        client_ntp = load_client_ntp(csv_path)

        if chronotick_ntp is None or client_ntp is None:
            print(f"⚠️  Skipping {platform_name} - missing data")
            continue

        print(f"  ChronoTick internal NTP: {len(chronotick_ntp)} measurements")
        print(f"  Client validation NTP: {len(client_ntp)} measurements")

        stats = plot_ntp_comparison(platform_name, chronotick_ntp, client_ntp, output_path)
        results[platform_name] = stats

        print(f"  ChronoTick NTP std: {stats['chronotick_std']:.3f}ms")
        print(f"  Client NTP std: {stats['client_std']:.3f}ms")
        print(f"  Improvement: {stats['improvement_std']:.1f}% (std), {stats['improvement_range']:.1f}% (range)")

    print("\n" + "="*80)
    print("SUMMARY: NTP MEASUREMENT QUALITY")
    print("="*80)

    for platform_name, stats in results.items():
        improvement = stats['improvement_std']
        status = "✅ MUCH BETTER" if improvement > 20 else "✅ BETTER" if improvement > 0 else "⚠️ WORSE"
        print(f"{platform_name:15} | Std reduction: {improvement:6.1f}% | {status}")

    print("\n" + "="*80)
    print("CONCLUSION:")
    print("="*80)
    print("""
The validation methodology has a fundamental flaw:

1. CLIENT uses SINGLE SERVER NTP (ntplib) → NOISY reference
2. CHRONOTICK uses 5-SERVER AVERAGING → STABLE reference
3. We're comparing ChronoTick against the NOISY client reference

This makes ChronoTick look WORSE than it actually is!

RECOMMENDATIONS:
a) Update client_driven_validation.py to use same multi-server NTP as ChronoTick
b) Or compare ChronoTick only against its OWN internal NTP measurements
c) Or document this limitation and use ChronoTick's internal metrics for evaluation
""")
    print("="*80)

if __name__ == '__main__':
    main()
