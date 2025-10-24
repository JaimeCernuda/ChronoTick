#!/usr/bin/env python3
"""
Multi-Node Temporal Alignment - APPROACH COMPARISON

Shows both Approach 1 and Approach 2 side-by-side for comparison.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def calculate_approach1(df1_ntp, df2_ntp):
    """Calculate Approach 1: Wall-clock time matching."""
    agreements = []

    for idx1, row1 in df1_ntp.iterrows():
        t1 = row1['timestamp']
        time_diff = (df2_ntp['timestamp'] - t1).abs()
        close_measurements = df2_ntp[time_diff <= pd.Timedelta(seconds=60)]

        if len(close_measurements) > 0:
            closest_idx = time_diff[time_diff <= pd.Timedelta(seconds=60)].idxmin()
            row2 = df2_ntp.loc[closest_idx]

            # Both measured NTP at similar wall-clock times
            error1 = row1['chronotick_offset_ms'] - row1['ntp_offset_ms']
            error2 = row2['chronotick_offset_ms'] - row2['ntp_offset_ms']

            diff = abs(error1 - error2)
            combined_3sigma = 3 * (row1['chronotick_uncertainty_ms'] + row2['chronotick_uncertainty_ms'])

            agreements.append(diff <= combined_3sigma)

    return sum(agreements) / len(agreements) * 100 if agreements else 0, len(agreements)

def calculate_approach2(df1_ntp, df2_all):
    """Calculate Approach 2: System time matching."""
    agreements = []
    differences = []

    for idx1, row1 in df1_ntp.iterrows():
        system_time_1 = row1['system_time']
        ntp_offset_1 = row1['ntp_offset_ms']
        uncertainty_1 = row1['chronotick_uncertainty_ms']

        # Find Node 2 samples where system_time matches
        time_diff = np.abs(df2_all['system_time'] - system_time_1)
        close_samples = df2_all[time_diff <= 1.0]

        if len(close_samples) > 0:
            closest_idx = time_diff[time_diff <= 1.0].idxmin()
            row2 = df2_all.loc[closest_idx]

            chronotick_offset_2 = row2['chronotick_offset_ms']
            uncertainty_2 = row2['chronotick_uncertainty_ms']

            # Compare: |ntp_offset_1 - chronotick_offset_2|
            diff = abs(ntp_offset_1 - chronotick_offset_2)
            differences.append(diff)
            combined_3sigma = 3 * (uncertainty_1 + uncertainty_2)

            agreements.append(diff <= combined_3sigma)

    rate = sum(agreements) / len(agreements) * 100 if agreements else 0
    mean_diff = np.mean(differences) if differences else 0
    return rate, len(agreements), mean_diff

def generate_comparison_figure(node1_csv, node2_csv, output_dir):
    """Generate comparison figure showing both approaches."""

    print("="*80)
    print("APPROACH COMPARISON")
    print("="*80)

    # Load data
    df1 = pd.read_csv(node1_csv)
    df2 = pd.read_csv(node2_csv)

    df1_ntp = df1[df1['has_ntp'] == True].copy()
    df2_ntp = df2[df2['has_ntp'] == True].copy()
    df2_all = df2.copy()

    df1_ntp['timestamp'] = pd.to_datetime(df1_ntp['datetime'])
    df2_ntp['timestamp'] = pd.to_datetime(df2_ntp['datetime'])
    df2_all['timestamp'] = pd.to_datetime(df2_all['datetime'])

    # Calculate both approaches
    print("\nCalculating Approach 1 (wall-clock time matching)...")
    approach1_rate, approach1_pairs = calculate_approach1(df1_ntp, df2_ntp)

    print("Calculating Approach 2 (system time matching)...")
    approach2_rate, approach2_pairs, approach2_mean_diff = calculate_approach2(df1_ntp, df2_all)

    print(f"\n{'='*80}")
    print("RESULTS COMPARISON")
    print('='*80)
    print(f"\nApproach 1 (Wall-clock Time Matching):")
    print(f"  Agreement Rate: {approach1_rate:.1f}%")
    print(f"  Matched Pairs: {approach1_pairs}")
    print(f"  Method: Compare when both measure NTP at similar wall-clock times")
    print(f"  Data Used: Only NTP measurements from both nodes")

    print(f"\nApproach 2 (System Time Matching):")
    print(f"  Agreement Rate: {approach2_rate:.1f}%")
    print(f"  Matched Pairs: {approach2_pairs}")
    print(f"  Mean Difference: {approach2_mean_diff:.3f} ms")
    print(f"  Method: Compare when both system clocks show same time T")
    print(f"  Data Used: NTP from Node 1, ALL samples from Node 2")

    print(f"\nKey Insights:")
    print(f"  • Approach 2 has {approach2_rate - approach1_rate:+.1f}% better agreement")
    print(f"  • Approach 2 uses {len(df2_all)/len(df2_ntp):.1f}x more Node 2 data")
    print(f"  • Approach 2 directly tests distributed event timestamps")
    print(f"  • Fewer pairs in Approach 2 due to system_time overlap constraints")

    # Create comparison visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Approach 1 visualization
    ax1 = axes[0]
    categories = ['Agree', 'Disagree']
    approach1_agree = int(approach1_rate / 100 * approach1_pairs)
    approach1_disagree = approach1_pairs - approach1_agree
    values1 = [approach1_agree, approach1_disagree]
    colors1 = ['#009E73', '#E69F00']

    bars1 = ax1.bar(categories, values1, color=colors1, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Number of Matched Pairs', fontsize=11)
    ax1.set_title(f'Approach 1: Wall-Clock Time\n{approach1_rate:.1f}% Agreement ({approach1_pairs} pairs)',
                  fontsize=11, fontweight='bold')
    ax1.grid(alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars1, values1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val}\n({val/approach1_pairs*100:.1f}%)',
                ha='center', va='bottom', fontsize=10)

    # Approach 2 visualization
    ax2 = axes[1]
    approach2_agree = int(approach2_rate / 100 * approach2_pairs)
    approach2_disagree = approach2_pairs - approach2_agree
    values2 = [approach2_agree, approach2_disagree]

    bars2 = ax2.bar(categories, values2, color=colors1, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Number of Matched Pairs', fontsize=11)
    ax2.set_title(f'Approach 2: System Time\n{approach2_rate:.1f}% Agreement ({approach2_pairs} pairs)',
                  fontsize=11, fontweight='bold')
    ax2.grid(alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars2, values2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val}\n({val/approach2_pairs*100:.1f}%)',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = output_dir / "5.12_APPROACH_COMPARISON.pdf"
    png_path = output_dir / "5.12_APPROACH_COMPARISON.png"

    plt.savefig(pdf_path, bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')

    print(f"\n✓ Saved: {pdf_path}")
    print(f"✓ Saved: {png_path}")

    plt.close()

    # Create detailed comparison table
    print(f"\n{'='*80}")
    print("DETAILED COMPARISON TABLE")
    print('='*80)
    print(f"{'Metric':<40} {'Approach 1':<20} {'Approach 2':<20}")
    print('-'*80)
    print(f"{'Agreement Rate':<40} {approach1_rate:>18.1f}% {approach2_rate:>18.1f}%")
    print(f"{'Matched Pairs':<40} {approach1_pairs:>20} {approach2_pairs:>20}")
    print(f"{'Node 2 Data Used':<40} {'NTP only':>20} {'All samples':>20}")
    print(f"{'Semantic Meaning':<40} {'Wall-clock sync':>20} {'Event timestamp':>20}")
    print(f"{'Mean Difference (ms)':<40} {'N/A':>20} {approach2_mean_diff:>19.3f}")

def main():
    """Generate approach comparison figure."""

    print("="*80)
    print("MULTI-NODE TEMPORAL ALIGNMENT - APPROACH COMPARISON")
    print("="*80)

    node1_csv = Path("results/experiment-7/ares-comp-11/chronotick_client_validation_20251020_220343.csv")
    node2_csv = Path("results/experiment-7/ares-comp-12/chronotick_client_validation_20251020_220540.csv")
    output_dir = Path("results/figures/5/experiment-7")

    if node1_csv.exists() and node2_csv.exists():
        generate_comparison_figure(node1_csv, node2_csv, output_dir)
    else:
        print("\n⚠️  Dataset files not found!")
        return

    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
