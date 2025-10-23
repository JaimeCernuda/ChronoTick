#!/usr/bin/env python3
"""
Create Access Performance Bar Chart for Evaluation 1

Generates publication-quality bar chart showing:
- System clock, NTP, and ChronoTick (1, 2, 4, 8 clients) latencies
- Grouped bars with hierarchical grouping
- Error bars showing standard deviation
- Log scale Y-axis
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def load_results(json_path):
    """Load benchmark results from JSON"""
    with open(json_path, 'r') as f:
        return json.load(f)

def create_access_performance_plot(results, output_path):
    """Create bar chart with grouped bars for different methods"""

    # Prepare data with proper grouping
    groups = []
    latencies = []
    std_devs = []
    colors = []
    labels = []

    # System Clock
    groups.append('System\nClock')
    latencies.append(results['system_clock']['mean'])
    std_devs.append(results['system_clock']['std'])
    colors.append('#2ecc71')
    labels.append(f"N={results['system_clock']['num_measurements']}")

    # NTP
    if 'ntp' in results and results['ntp']['mean'] > 0:
        groups.append('NTP\n(Single)')
        latencies.append(results['ntp']['mean'])
        std_devs.append(results['ntp']['std'])
        colors.append('#f39c12')
        labels.append(f"N={results['ntp']['num_measurements']}")

    # ChronoTick (1, 2, 4, 8 clients)
    for num_clients in [1, 2, 4, 8]:
        key = f'chronotick_{num_clients}'
        if key in results:
            groups.append(f'ChronoTick\n({num_clients} client{"s" if num_clients > 1 else ""})')
            latencies.append(results[key]['mean'])
            std_devs.append(results[key]['std'])
            colors.append('#3498db')
            labels.append(f"N={results[key]['num_measurements']}")

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(groups))
    width = 0.6

    # Create bars
    bars = ax.bar(x, latencies, width, color=colors,
                  edgecolor='black', linewidth=0.5)

    # Error bars
    ax.errorbar(x, latencies, yerr=std_devs, fmt='none', ecolor='black',
                capsize=5, capthick=2, alpha=0.7)

    # Formatting
    ax.set_ylabel('Latency (ms)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time Access Method', fontsize=14, fontweight='bold')
    ax.set_title('Access Performance: System Clock vs NTP vs ChronoTick',
                fontsize=16, fontweight='bold', pad=20)

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=11)

    # Log scale for Y-axis (since we have huge differences)
    ax.set_yscale('log')

    # Grid
    ax.grid(True, alpha=0.3, which='both', axis='y', linestyle='--')
    ax.set_axisbelow(True)

    # Legend
    legend_elements = [
        mpatches.Patch(color='#2ecc71', label='System Clock', edgecolor='black', linewidth=0.5),
        mpatches.Patch(color='#f39c12', label='NTP (Network)', edgecolor='black', linewidth=0.5),
        mpatches.Patch(color='#3498db', label='ChronoTick', edgecolor='black', linewidth=0.5),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11, framealpha=0.9)

    # Add value labels on bars
    for i, (group, lat) in enumerate(zip(groups, latencies)):
        if 'System' in group:
            value_label = f'{lat:.6f} ms'
            y_pos = lat * 1.8
        elif 'NTP' in group:
            value_label = f'{lat:.2f} ms'
            y_pos = lat * 1.5
        else:  # ChronoTick
            value_label = f'{lat:.6f} ms'
            y_pos = lat * 1.8

        ax.text(i, y_pos, value_label, ha='center', va='bottom',
               fontsize=9, fontweight='bold')

    # Tight layout
    plt.tight_layout()

    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {output_path}")

    # Also save as PDF for LaTeX
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"✓ PDF saved to: {pdf_path}")

    plt.close()

def print_summary_table(results):
    """Print formatted summary table"""
    print("\n" + "="*80)
    print("ACCESS PERFORMANCE SUMMARY")
    print("="*80)
    print(f"{'Method':<25} {'Mean (ms)':>12} {'Std (ms)':>12} {'Median (ms)':>12} {'N':>6}")
    print("-"*80)

    # System Clock
    r = results['system_clock']
    print(f"{'System Clock':<25} {r['mean']:>12.6f} {r['std']:>12.6f} {r['median']:>12.6f} {r['num_measurements']:>6d}")

    # NTP
    if 'ntp' in results and results['ntp']['mean'] > 0:
        r = results['ntp']
        print(f"{'NTP (full round-trip)':<25} {r['mean']:>12.2f} {r['std']:>12.2f} {r['median']:>12.2f} {r['num_measurements']:>6d}")

    # ChronoTick
    for num_clients in [1, 2, 4, 8]:
        key = f'chronotick_{num_clients}'
        if key in results:
            r = results[key]
            print(f"{f'ChronoTick ({num_clients} clients)':<25} {r['mean']:>12.6f} {r['std']:>12.6f} {r['median']:>12.6f} {r['num_measurements']:>6d}")

    print("="*80)

    # Performance comparison
    print("\nPERFORMANCE COMPARISON:")
    print("-"*80)

    system_mean = results['system_clock']['mean']
    if 'ntp' in results and results['ntp']['mean'] > 0:
        ntp_mean = results['ntp']['mean']
        print(f"NTP is {ntp_mean/system_mean:.0f}× slower than System Clock")

    if 'chronotick_1' in results:
        ct_mean = results['chronotick_1']['mean']
        print(f"ChronoTick (1 client) is {ct_mean/system_mean:.1f}× slower than System Clock")

        if 'ntp' in results and results['ntp']['mean'] > 0:
            print(f"ChronoTick (1 client) is {ntp_mean/ct_mean:.0f}× faster than NTP")

    print("="*80)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='/tmp/eval1_results.json',
                       help='Input JSON file with benchmark results')
    parser.add_argument('--output', default='/tmp/eval1_access_performance.png',
                       help='Output PNG file')
    args = parser.parse_args()

    print("="*80)
    print("CREATING ACCESS PERFORMANCE PLOT")
    print("="*80)

    # Load results
    print(f"\nLoading results from: {args.input}")
    results = load_results(args.input)

    # Print summary table
    print_summary_table(results)

    # Create plot
    print(f"\nGenerating plot...")
    create_access_performance_plot(results, args.output)

    print("\n" + "="*80)
    print("✓ PLOT GENERATION COMPLETE!")
    print("="*80)
