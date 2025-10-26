#!/usr/bin/env python3
"""
Generate 5 Micro-Evaluation Figures with Detailed Analysis
Based on user-specified interesting cases
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import json
from pathlib import Path

# Configuration
FIGURE_DIR = Path('/home/jcernuda/tick_project/ChronoTick/results/figures/micro_evaluations')
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Output data for LaTeX
latex_data = {}

def setup_plot_style():
    """Set publication-quality plot style"""
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 13
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 14
    plt.rcParams['font.family'] = 'serif'


def figure_1_exp1_wsl2_hours3_5():
    """
    Figure 1: WSL2 Multi-Server NTP Chaos (Hours 3-5)
    Story: NTP measurements jump wildly, ChronoTick predicts through the middle
    """
    print("\n" + "="*80)
    print("FIGURE 1: WSL2 Multi-Server NTP Chaos (Hours 3-5)")
    print("="*80)

    df = pd.read_csv('/home/jcernuda/tick_project/ChronoTick/results/experiment-1/wsl2/chronotick_client_validation_20251018_020105.csv')

    # Hours 3-5
    data = df[(df['elapsed_seconds'] >= 3*3600) & (df['elapsed_seconds'] <= 5*3600)].copy()
    data['time_minutes'] = data['elapsed_seconds'] / 60

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Panel A: NTP measurements with ChronoTick prediction
    ntp_data = data[data['has_ntp'] == True]

    # Plot NTP scatter
    ax1.scatter(ntp_data['time_minutes'], ntp_data['ntp_offset_ms'],
                c='blue', s=40, alpha=0.6, label='NTP measurements', zorder=3)

    # Plot ChronoTick prediction line
    ax1.plot(data['time_minutes'], data['chronotick_offset_ms'],
             color='green', linewidth=2, label='ChronoTick prediction', zorder=2)

    # Mark large jumps
    ntp_copy = ntp_data.copy()
    ntp_copy['ntp_jump'] = ntp_copy['ntp_offset_ms'].diff().abs()
    large_jumps = ntp_copy[ntp_copy['ntp_jump'] > 200]

    for idx, row in large_jumps.iterrows():
        ax1.axvline(row['time_minutes'], color='red', alpha=0.3, linestyle='--', linewidth=1)
        ax1.annotate(f'{row["ntp_jump"]:.0f}ms jump',
                    xy=(row['time_minutes'], row['ntp_offset_ms']),
                    xytext=(10, 20), textcoords='offset points',
                    fontsize=9, color='red',
                    arrowprops=dict(arrowstyle='->', color='red', lw=1))

    ax1.set_ylabel('Offset (ms)', fontsize=12)
    ax1.set_title('Panel A: NTP Chaos vs ChronoTick Smoothing', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Panel B: Prediction error
    ntp_data_copy = ntp_data.copy()
    ntp_data_copy['error'] = ntp_data_copy['chronotick_offset_ms'] - ntp_data_copy['ntp_offset_ms']

    ax2.scatter(ntp_data_copy['time_minutes'], ntp_data_copy['error'],
                c='purple', s=40, alpha=0.6, label='ChronoTick error', zorder=3)
    ax2.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)

    # Show ±σ bands
    mae = ntp_data_copy['error'].abs().mean()
    std = ntp_data_copy['error'].std()
    ax2.axhline(std, color='orange', linestyle='--', linewidth=1.5, label=f'±σ = ±{std:.1f}ms')
    ax2.axhline(-std, color='orange', linestyle='--', linewidth=1.5)
    ax2.fill_between(data['time_minutes'], -std, std, color='orange', alpha=0.1)

    ax2.set_xlabel('Time (minutes)', fontsize=12)
    ax2.set_ylabel('Prediction Error (ms)', fontsize=12)
    ax2.set_title(f'Panel B: ChronoTick Error (MAE={mae:.1f}ms, σ={std:.1f}ms)', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / 'figure1_wsl2_ntp_chaos.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(FIGURE_DIR / 'figure1_wsl2_ntp_chaos.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Save data for LaTeX
    latex_data['figure1'] = {
        'experiment': 'experiment-1/wsl2',
        'time_window': 'Hours 3-5 (180-300 minutes)',
        'ntp_range': f'{ntp_data["ntp_offset_ms"].min():.1f}ms to {ntp_data["ntp_offset_ms"].max():.1f}ms',
        'ntp_std': f'{ntp_data["ntp_offset_ms"].std():.2f}ms',
        'ntp_jumps': len(large_jumps),
        'largest_jump': f'{large_jumps["ntp_jump"].max():.1f}ms' if len(large_jumps) > 0 else 'N/A',
        'chronotick_mae': f'{mae:.2f}ms',
        'chronotick_std': f'{std:.2f}ms',
        'chronotick_max_error': f'{ntp_data_copy["error"].abs().max():.2f}ms',
        'narrative': f'WSL2 virtualized time causes NTP chaos (σ={ntp_data["ntp_offset_ms"].std():.1f}ms, {len(large_jumps)} jumps >200ms). ChronoTick smooths through chaos with σ={std:.1f}ms.'
    }

    print(f"✓ Figure 1 saved: NTP σ={ntp_data['ntp_offset_ms'].std():.1f}ms, CT σ={std:.1f}ms")


def figure_2_exp9_bad_start_recovery():
    """
    Figure 2: Exp-9 Bad Synchronization Start and Recovery
    Story: ChronoTick starts with poor prediction, recovers within first hour
    """
    print("\n" + "="*80)
    print("FIGURE 2: Exp-9 Bad Start and Recovery (Hour 0-1)")
    print("="*80)

    df = pd.read_csv('/home/jcernuda/tick_project/ChronoTick/results/experiment-9/homelab/chronotick_client_validation_20251022_094657.csv')

    # First hour
    data = df[df['elapsed_seconds'] <= 3600].copy()
    data['time_minutes'] = data['elapsed_seconds'] / 60

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Panel A: Offset evolution
    ax1.plot(data['time_minutes'], data['chronotick_offset_ms'],
             color='green', linewidth=2, label='ChronoTick prediction', zorder=2)

    ntp_data = data[data['has_ntp'] == True]
    ax1.scatter(ntp_data['time_minutes'], ntp_data['ntp_offset_ms'],
                c='blue', s=50, alpha=0.7, label='NTP ground truth', zorder=3, marker='o')

    # Mark first NTP arrival
    first_ntp = ntp_data.iloc[0]
    ax1.axvline(first_ntp['time_minutes'], color='red', linestyle='--', alpha=0.5, linewidth=2)
    ax1.annotate('First NTP\narrival',
                xy=(first_ntp['time_minutes'], first_ntp['ntp_offset_ms']),
                xytext=(15, 20), textcoords='offset points',
                fontsize=10, color='red', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='red', lw=2))

    ax1.set_ylabel('Offset (ms)', fontsize=12)
    ax1.set_title('Panel A: Offset Evolution (Bad Start → Recovery)', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Panel B: Prediction error over time
    ntp_data_copy = ntp_data.copy()
    ntp_data_copy['error'] = ntp_data_copy['chronotick_offset_ms'] - ntp_data_copy['ntp_offset_ms']

    # Calculate rolling statistics
    ntp_data_copy['cumulative_mae'] = ntp_data_copy['error'].abs().expanding().mean()

    ax2.scatter(ntp_data_copy['time_minutes'], ntp_data_copy['error'].abs(),
                c='purple', s=50, alpha=0.6, label='Absolute error', zorder=3)
    ax2.plot(ntp_data_copy['time_minutes'], ntp_data_copy['cumulative_mae'],
             color='orange', linewidth=2.5, label='Cumulative MAE', zorder=2)

    # Mark recovery point (when cumulative MAE stabilizes)
    final_mae = ntp_data_copy['cumulative_mae'].iloc[-1]
    ax2.axhline(final_mae, color='green', linestyle='--', linewidth=1.5,
                label=f'Final MAE = {final_mae:.2f}ms')

    ax2.set_xlabel('Time (minutes)', fontsize=12)
    ax2.set_ylabel('Prediction Error (ms)', fontsize=12)
    ax2.set_title('Panel B: Error Convergence (Recovery Process)', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / 'figure2_exp9_recovery.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(FIGURE_DIR / 'figure2_exp9_recovery.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Calculate metrics
    first_10_min = ntp_data_copy[ntp_data_copy['time_minutes'] <= 10]
    last_10_min = ntp_data_copy[ntp_data_copy['time_minutes'] >= 50]

    latex_data['figure2'] = {
        'experiment': 'experiment-9/homelab',
        'time_window': 'Hour 0-1 (0-60 minutes)',
        'first_ntp_time': f'{first_ntp["time_minutes"]:.2f} minutes',
        'first_ntp_error': f'{abs(first_ntp["chronotick_offset_ms"] - first_ntp["ntp_offset_ms"]):.2f}ms',
        'first_10min_mae': f'{first_10_min["error"].abs().mean():.2f}ms' if len(first_10_min) > 0 else 'N/A',
        'last_10min_mae': f'{last_10_min["error"].abs().mean():.2f}ms' if len(last_10_min) > 0 else 'N/A',
        'final_mae': f'{final_mae:.2f}ms',
        'improvement_factor': f'{first_10_min["error"].abs().mean() / final_mae:.1f}×' if len(first_10_min) > 0 and final_mae > 0 else 'N/A',
        'narrative': f'Bad initial prediction ({first_10_min["error"].abs().mean():.2f}ms MAE in first 10min), recovers to {final_mae:.2f}ms MAE by hour end.'
    }

    print(f"✓ Figure 2 saved: Initial MAE={first_10_min['error'].abs().mean():.2f}ms → Final={final_mae:.2f}ms")


def figure_3_exp3_warmup_rejection_storm():
    """
    Figure 3: Exp-3 Homelab Warmup Rejection Storm
    Story: From paper defense story - 11 rejections in first 20 min, ChronoTick stable
    """
    print("\n" + "="*80)
    print("FIGURE 3: Exp-3 Warmup Rejection Storm (Minutes 0-20)")
    print("="*80)

    df = pd.read_csv('/home/jcernuda/tick_project/ChronoTick/results/experiment-3/homelab/data.csv')

    # First 20 minutes
    data = df[df['elapsed_seconds'] <= 1200].copy()
    data['time_minutes'] = data['elapsed_seconds'] / 60

    # Load log to find rejections
    log_path = Path('/home/jcernuda/tick_project/ChronoTick/results/experiment-3/homelab/daemon.log')
    rejections = []
    if log_path.exists():
        with open(log_path, 'r') as f:
            for line in f:
                if 'REJECTED' in line and 'z=' in line:
                    # Parse timestamp and z-score
                    parts = line.split()
                    if len(parts) > 0:
                        timestamp_str = ' '.join(parts[:2])
                        try:
                            from datetime import datetime
                            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
                            # Find z-score
                            z_str = [p for p in parts if p.startswith('z=')][0]
                            z_score = float(z_str.replace('z=', '').replace('σ', ''))
                            rejections.append({'timestamp': timestamp, 'z_score': z_score})
                        except:
                            pass

    # Filter rejections to first 20 minutes
    if rejections:
        start_time = rejections[0]['timestamp']
        warmup_rejections = [r for r in rejections
                            if (r['timestamp'] - start_time).total_seconds() <= 1200]
    else:
        warmup_rejections = []

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Panel A: NTP measurements with rejections marked
    ntp_data = data[data['has_ntp'] == True]

    # Plot accepted NTP
    ax1.scatter(ntp_data['time_minutes'], ntp_data['ntp_offset_ms'],
                c='blue', s=50, alpha=0.7, label='Accepted NTP', zorder=3, marker='o')

    # Mark rejection times (approximate from log timing)
    for i, rej in enumerate(warmup_rejections[:10]):  # First 10 rejections
        # Estimate minutes from start
        if rejections:
            rej_minutes = (rej['timestamp'] - start_time).total_seconds() / 60
            ax1.axvline(rej_minutes, color='red', alpha=0.2, linestyle='--', linewidth=1)
            if i < 3:  # Annotate first 3
                ax1.text(rej_minutes, ax1.get_ylim()[1]*0.9, f'z={rej["z_score"]:.1f}σ',
                        rotation=90, va='top', ha='right', fontsize=8, color='red')

    ax1.set_ylabel('NTP Offset (ms)', fontsize=12)
    ax1.set_title(f'Panel A: NTP Measurements ({len(warmup_rejections)} rejections marked)',
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Panel B: ChronoTick stability despite rejections
    ax2.plot(data['time_minutes'], data['chronotick_offset_ms'],
             color='green', linewidth=2, label='ChronoTick prediction', zorder=2)
    ax2.scatter(ntp_data['time_minutes'], ntp_data['ntp_offset_ms'],
                c='blue', s=30, alpha=0.5, label='NTP ground truth', zorder=3)

    # Calculate and show ChronoTick stability
    ntp_with_ct = ntp_data.copy()
    ntp_with_ct['error'] = ntp_with_ct['chronotick_offset_ms'] - ntp_with_ct['ntp_offset_ms']
    mae = ntp_with_ct['error'].abs().mean()
    std = ntp_with_ct['error'].std()

    ax2.set_xlabel('Time (minutes)', fontsize=12)
    ax2.set_ylabel('Offset (ms)', fontsize=12)
    ax2.set_title(f'Panel B: ChronoTick Stability (MAE={mae:.3f}ms, σ={std:.3f}ms despite {len(warmup_rejections)} rejections)',
                  fontsize=13, fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / 'figure3_exp3_warmup_storm.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(FIGURE_DIR / 'figure3_exp3_warmup_storm.png', dpi=300, bbox_inches='tight')
    plt.close()

    latex_data['figure3'] = {
        'experiment': 'experiment-3/homelab',
        'time_window': 'Minutes 0-20 (warmup period)',
        'ntp_rejections': len(warmup_rejections),
        'max_z_score': f'{max([r["z_score"] for r in warmup_rejections]):.1f}σ' if warmup_rejections else 'N/A',
        'chronotick_mae': f'{mae:.3f}ms',
        'chronotick_std': f'{std:.3f}ms',
        'narrative': f'{len(warmup_rejections)} NTP rejections (max z={max([r["z_score"] for r in warmup_rejections]):.1f}σ) during warmup. ChronoTick maintains σ={std:.3f}ms stability.'
    }

    print(f"✓ Figure 3 saved: {len(warmup_rejections)} rejections, CT σ={std:.3f}ms")


def figure_4_exp8_local_extreme_thermal():
    """
    Figure 4: Exp-8 Local Extreme Thermal Drift
    Story: Consumer hardware extreme drift (683 ms/h), ChronoTick tracks it
    """
    print("\n" + "="*80)
    print("FIGURE 4: Exp-8 Local Extreme Thermal Drift (Hour 0-1)")
    print("="*80)

    df = pd.read_csv('/home/jcernuda/tick_project/ChronoTick/results/experiment-8/local/chronotick_client_validation_20251021_121912.csv')

    # First hour
    data = df[df['elapsed_seconds'] <= 3600].copy()
    data['time_minutes'] = data['elapsed_seconds'] / 60

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Panel A: System clock drift (NTP offset)
    ntp_data = data[data['has_ntp'] == True]

    if len(ntp_data) > 0:
        # Linear regression for drift rate
        x = ntp_data['elapsed_seconds'].values
        y = ntp_data['ntp_offset_ms'].values
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        drift_rate_per_hour = slope * 3600

        # Plot NTP measurements
        ax1.scatter(ntp_data['time_minutes'], ntp_data['ntp_offset_ms'],
                    c='red', s=50, alpha=0.7, label='System clock drift (NTP)', zorder=3)

        # Plot linear fit
        fit_line = slope * x + intercept
        ax1.plot(ntp_data['time_minutes'], fit_line,
                color='darkred', linewidth=2, linestyle='--',
                label=f'Linear fit: {drift_rate_per_hour:.1f} ms/hour (R²={r_value**2:.3f})',
                zorder=2)

        ax1.set_ylabel('System Clock Error (ms)', fontsize=12)
        ax1.set_title(f'Panel A: Extreme Thermal Drift ({drift_rate_per_hour:.1f} ms/hour)',
                      fontsize=13, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Panel B: ChronoTick tracking
        ax2.plot(data['time_minutes'], data['chronotick_offset_ms'],
                color='green', linewidth=2, label='ChronoTick prediction', zorder=2)
        ax2.scatter(ntp_data['time_minutes'], ntp_data['ntp_offset_ms'],
                    c='red', s=30, alpha=0.5, label='NTP ground truth', zorder=3)

        # Calculate tracking error
        ntp_with_ct = ntp_data.copy()
        ntp_with_ct['error'] = ntp_with_ct['chronotick_offset_ms'] - ntp_with_ct['ntp_offset_ms']
        mae = ntp_with_ct['error'].abs().mean()
        std = ntp_with_ct['error'].std()

        ax2.set_xlabel('Time (minutes)', fontsize=12)
        ax2.set_ylabel('Offset (ms)', fontsize=12)
        ax2.set_title(f'Panel B: ChronoTick Adaptive Tracking (MAE={mae:.2f}ms, σ={std:.2f}ms)',
                      fontsize=13, fontweight='bold')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(FIGURE_DIR / 'figure4_exp8_thermal_extreme.pdf', dpi=300, bbox_inches='tight')
        fig.savefig(FIGURE_DIR / 'figure4_exp8_thermal_extreme.png', dpi=300, bbox_inches='tight')
        plt.close()

        latex_data['figure4'] = {
            'experiment': 'experiment-8/local',
            'time_window': 'Hour 0-1 (0-60 minutes)',
            'drift_rate': f'{drift_rate_per_hour:.1f} ms/hour',
            'r_squared': f'{r_value**2:.4f}',
            'total_drift': f'{ntp_data.iloc[-1]["ntp_offset_ms"] - ntp_data.iloc[0]["ntp_offset_ms"]:.1f}ms',
            'chronotick_mae': f'{mae:.2f}ms',
            'chronotick_std': f'{std:.2f}ms',
            'narrative': f'Consumer hardware shows extreme thermal drift ({drift_rate_per_hour:.0f} ms/h). ChronoTick adaptively tracks with σ={std:.1f}ms.'
        }

        print(f"✓ Figure 4 saved: Drift={drift_rate_per_hour:.1f} ms/h, CT σ={std:.1f}ms")


def figure_5_exp3_thermal_hours3_4():
    """
    Figure 5: Exp-3 Thermal Wandering Hours 3-4
    Story: Non-linear thermal drift with direction changes
    """
    print("\n" + "="*80)
    print("FIGURE 5: Exp-3 Thermal Wandering (Hours 3-4)")
    print("="*80)

    df = pd.read_csv('/home/jcernuda/tick_project/ChronoTick/results/experiment-3/homelab/data.csv')

    # Hours 3-4
    data = df[(df['elapsed_seconds'] >= 3*3600) & (df['elapsed_seconds'] <= 4*3600)].copy()
    data['time_minutes'] = data['elapsed_seconds'] / 60
    data['time_hours'] = data['elapsed_seconds'] / 3600

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Panel A: NTP offset showing wandering
    ntp_data = data[data['has_ntp'] == True]

    if len(ntp_data) > 0:
        ax1.plot(ntp_data['time_minutes'], ntp_data['ntp_offset_ms'],
                color='blue', linewidth=2, marker='o', markersize=4,
                label='System clock drift', zorder=2)

        # Calculate drift rates in 15-minute windows
        window_minutes = 15
        for i in range(0, 60, window_minutes):
            window_start = 180 + i
            window_end = 180 + i + window_minutes
            window_data = ntp_data[(ntp_data['time_minutes'] >= window_start) &
                                  (ntp_data['time_minutes'] < window_end)]

            if len(window_data) > 1:
                x = window_data['elapsed_seconds'].values
                y = window_data['ntp_offset_ms'].values
                slope, _, _, _, _ = stats.linregress(x, y)
                drift_rate = slope * 3600  # ms/hour

                # Draw segment
                mid_time = (window_start + window_end) / 2
                ax1.text(mid_time, ax1.get_ylim()[1]*0.95, f'{drift_rate:+.1f}\nms/h',
                        ha='center', va='top', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

        ax1.set_ylabel('System Clock Offset (ms)', fontsize=12)
        ax1.set_title('Panel A: Non-Linear Thermal Drift (Wandering Pattern)', fontsize=13, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Panel B: ChronoTick tracking
        ax2.plot(data['time_minutes'], data['chronotick_offset_ms'],
                color='green', linewidth=2, label='ChronoTick prediction', zorder=2)
        ax2.scatter(ntp_data['time_minutes'], ntp_data['ntp_offset_ms'],
                    c='blue', s=30, alpha=0.5, label='NTP ground truth', zorder=3)

        # Error analysis
        ntp_with_ct = ntp_data.copy()
        ntp_with_ct['error'] = ntp_with_ct['chronotick_offset_ms'] - ntp_with_ct['ntp_offset_ms']
        mae = ntp_with_ct['error'].abs().mean()
        std = ntp_with_ct['error'].std()

        ax2.set_xlabel('Time (minutes)', fontsize=12)
        ax2.set_ylabel('Offset (ms)', fontsize=12)
        ax2.set_title(f'Panel B: ChronoTick Adaptive Tracking (MAE={mae:.3f}ms, σ={std:.3f}ms)',
                      fontsize=13, fontweight='bold')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(FIGURE_DIR / 'figure5_exp3_thermal_wandering.pdf', dpi=300, bbox_inches='tight')
        fig.savefig(FIGURE_DIR / 'figure5_exp3_thermal_wandering.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Calculate overall drift metrics
        total_drift = ntp_data.iloc[-1]['ntp_offset_ms'] - ntp_data.iloc[0]['ntp_offset_ms']
        offset_range = ntp_data['ntp_offset_ms'].max() - ntp_data['ntp_offset_ms'].min()

        latex_data['figure5'] = {
            'experiment': 'experiment-3/homelab',
            'time_window': 'Hours 3-4 (180-240 minutes)',
            'total_drift': f'{total_drift:.2f}ms',
            'offset_range': f'{offset_range:.2f}ms',
            'offset_std': f'{ntp_data["ntp_offset_ms"].std():.2f}ms',
            'chronotick_mae': f'{mae:.3f}ms',
            'chronotick_std': f'{std:.3f}ms',
            'narrative': f'Thermal effects cause non-linear drift (range={offset_range:.1f}ms, σ={ntp_data["ntp_offset_ms"].std():.2f}ms). ChronoTick tracks with σ={std:.2f}ms.'
        }

        print(f"✓ Figure 5 saved: Drift range={offset_range:.2f}ms, CT σ={std:.2f}ms")


def main():
    """Generate all 5 micro-evaluation figures"""
    setup_plot_style()

    print("="*80)
    print("GENERATING 5 MICRO-EVALUATION FIGURES")
    print("="*80)

    figure_1_exp1_wsl2_hours3_5()
    figure_2_exp9_bad_start_recovery()
    figure_3_exp3_warmup_rejection_storm()
    figure_4_exp8_local_extreme_thermal()
    figure_5_exp3_thermal_hours3_4()

    # Save LaTeX data
    output_json = FIGURE_DIR / 'latex_data.json'
    with open(output_json, 'w') as f:
        json.dump(latex_data, f, indent=2)

    print("\n" + "="*80)
    print("✅ ALL FIGURES GENERATED")
    print("="*80)
    print(f"Figures saved to: {FIGURE_DIR}")
    print(f"LaTeX data saved to: {output_json}")
    print("\nFigures:")
    for fig_file in sorted(FIGURE_DIR.glob('*.pdf')):
        print(f"  - {fig_file.name}")


if __name__ == '__main__':
    main()
