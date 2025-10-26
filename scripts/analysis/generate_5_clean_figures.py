#!/usr/bin/env python3
"""
Generate 5 Clean Micro-Evaluation Figures
ONE figure per evaluation, no panels, consistent colors
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import json
from pathlib import Path

# Clean paper-quality settings
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
    'axes.linewidth': 1.0,
    'grid.alpha': 0.25,
    'grid.linewidth': 0.6,
})

# Consistent colors
COLOR_CHRONOTICK = '#2ca02c'  # Green
COLOR_NTP_ACCEPTED = '#1f77b4'  # Blue
COLOR_NTP_REJECTED = '#d62728'  # Red
COLOR_SYSTEM_DRIFT = '#ff7f0e'  # Orange/Red for system clock

FIGURE_DIR = Path('/home/jcernuda/tick_project/ChronoTick/results/figures/micro_evaluations')
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

latex_data = {}


def figure1_wsl2_chaos_clean():
    """
    Figure 1: WSL2 NTP Chaos - ONE figure showing chaos vs smoothness
    Story: NTP jumps wildly, ChronoTick smooths through the middle
    """
    print("\n" + "="*80)
    print("FIGURE 1: WSL2 Chaos (Clean, ONE figure)")
    print("="*80)

    df = pd.read_csv('/home/jcernuda/tick_project/ChronoTick/results/experiment-1/wsl2/chronotick_client_validation_20251018_020105.csv')
    data = df[(df['elapsed_seconds'] >= 3*3600) & (df['elapsed_seconds'] <= 5*3600)].copy()
    data['time_minutes'] = data['elapsed_seconds'] / 60

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    ntp_data = data[data['has_ntp'] == True].copy()

    # Plot ChronoTick smooth prediction
    ax.plot(data['time_minutes'], data['chronotick_offset_ms'],
            color=COLOR_CHRONOTICK, linewidth=2.5, label='ChronoTick', zorder=3, alpha=0.9)

    # Plot NTP measurements as scatter
    ax.scatter(ntp_data['time_minutes'], ntp_data['ntp_offset_ms'],
               c=COLOR_NTP_ACCEPTED, s=35, alpha=0.6, label='NTP measurements', zorder=2)

    # Find large jumps and shade those regions
    ntp_copy = ntp_data.copy()
    ntp_copy['ntp_jump'] = ntp_copy['ntp_offset_ms'].diff().abs()
    large_jumps = ntp_copy[ntp_copy['ntp_jump'] > 200]

    # Shade regions with jumps (subtle)
    for idx, row in large_jumps.iterrows():
        ax.axvspan(row['time_minutes']-5, row['time_minutes']+5,
                   alpha=0.08, color='red', zorder=1)

    ax.set_xlabel('Time (minutes)', fontsize=13)
    ax.set_ylabel('Offset (ms)', fontsize=13)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / 'figure1_wsl2_chaos.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(FIGURE_DIR / 'figure1_wsl2_chaos.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Metrics
    ntp_data['error'] = ntp_data['chronotick_offset_ms'] - ntp_data['ntp_offset_ms']
    mae = ntp_data['error'].abs().mean()
    std = ntp_data['error'].std()

    latex_data['figure1'] = {
        'ntp_std': f'{ntp_data["ntp_offset_ms"].std():.1f}ms',
        'ntp_jumps': len(large_jumps),
        'largest_jump': f'{large_jumps["ntp_jump"].max():.0f}ms' if len(large_jumps) > 0 else 'N/A',
        'chronotick_mae': f'{mae:.1f}ms',
        'chronotick_std': f'{std:.1f}ms',
    }

    print(f"✓ Saved: NTP σ={ntp_data['ntp_offset_ms'].std():.1f}ms, {len(large_jumps)} jumps, CT MAE={mae:.1f}ms")


def figure2_exp9_recovery_clean():
    """
    Figure 2: Exp-9 Recovery - ONE figure showing error convergence
    Story: Bad start → recovery (that's what matters!)
    """
    print("\n" + "="*80)
    print("FIGURE 2: Exp-9 Recovery (Clean, ONE figure)")
    print("="*80)

    df = pd.read_csv('/home/jcernuda/tick_project/ChronoTick/results/experiment-9/homelab/chronotick_client_validation_20251022_094657.csv')
    data = df[df['elapsed_seconds'] <= 3600].copy()
    data['time_minutes'] = data['elapsed_seconds'] / 60

    ntp_data = data[data['has_ntp'] == True].copy()
    ntp_data['error'] = (ntp_data['chronotick_offset_ms'] - ntp_data['ntp_offset_ms']).abs()
    ntp_data['cumulative_mae'] = ntp_data['error'].expanding().mean()

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # Plot individual errors as scatter
    ax.scatter(ntp_data['time_minutes'], ntp_data['error'],
               c=COLOR_NTP_ACCEPTED, s=40, alpha=0.5, label='Prediction error', zorder=2)

    # Plot cumulative MAE convergence
    ax.plot(ntp_data['time_minutes'], ntp_data['cumulative_mae'],
            color=COLOR_CHRONOTICK, linewidth=3, label='Cumulative MAE (converging)', zorder=3)

    # Mark final MAE
    final_mae = ntp_data['cumulative_mae'].iloc[-1]
    ax.axhline(final_mae, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(50, final_mae+0.3, f'Final MAE = {final_mae:.2f}ms',
            fontsize=10, va='bottom', ha='right', color='gray')

    ax.set_xlabel('Time (minutes)', fontsize=13)
    ax.set_ylabel('Error (ms)', fontsize=13)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.25)
    ax.set_ylim(0, max(ntp_data['error'].max(), 10))

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / 'figure2_exp9_recovery.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(FIGURE_DIR / 'figure2_exp9_recovery.png', dpi=300, bbox_inches='tight')
    plt.close()

    first_10 = ntp_data[ntp_data['time_minutes'] <= 10]
    latex_data['figure2'] = {
        'first_error': f'{ntp_data.iloc[0]["error"]:.2f}ms',
        'first_10min_mae': f'{first_10["error"].mean():.2f}ms' if len(first_10) > 0 else 'N/A',
        'final_mae': f'{final_mae:.2f}ms',
    }

    print(f"✓ Saved: Initial error={ntp_data.iloc[0]['error']:.2f}ms → Final MAE={final_mae:.2f}ms")


def figure3_warmup_storm_clean():
    """
    Figure 3: Warmup Rejection Storm - ONE figure with ACCEPTED + REJECTED + ChronoTick
    Story: Show the rejections! That's the point!
    """
    print("\n" + "="*80)
    print("FIGURE 3: Warmup Storm (Clean, ONE figure with rejections!)")
    print("="*80)

    df = pd.read_csv('/home/jcernuda/tick_project/ChronoTick/results/experiment-3/homelab/data.csv')
    data = df[df['elapsed_seconds'] <= 1200].copy()
    data['time_minutes'] = data['elapsed_seconds'] / 60

    # Load log for rejections
    log_path = Path('/home/jcernuda/tick_project/ChronoTick/results/experiment-3/homelab/daemon.log')
    rejections = []
    if log_path.exists():
        with open(log_path, 'r') as f:
            for line in f:
                if 'REJECTED' in line and 'z=' in line:
                    try:
                        from datetime import datetime
                        parts = line.split()
                        timestamp = datetime.strptime(' '.join(parts[:2]), '%Y-%m-%d %H:%M:%S,%f')
                        z_str = [p for p in parts if p.startswith('z=')][0]
                        z_score = float(z_str.replace('z=', '').replace('σ', ''))
                        rejections.append({'timestamp': timestamp, 'z_score': z_score})
                    except:
                        pass

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # Plot ChronoTick prediction
    ax.plot(data['time_minutes'], data['chronotick_offset_ms'],
            color=COLOR_CHRONOTICK, linewidth=2.5, label='ChronoTick', zorder=3)

    # Plot ACCEPTED NTP
    ntp_accepted = data[data['has_ntp'] == True].copy()
    ax.scatter(ntp_accepted['time_minutes'], ntp_accepted['ntp_offset_ms'],
               c=COLOR_NTP_ACCEPTED, s=50, alpha=0.7, label='NTP accepted', zorder=4, marker='o')

    # Plot REJECTED NTP (estimated from log timing)
    if rejections:
        start_time = rejections[0]['timestamp']
        for rej in rejections[:10]:  # First 10 rejections in 20min
            rej_minutes = (rej['timestamp'] - start_time).total_seconds() / 60
            if rej_minutes <= 20:
                # Mark rejection with red X
                ax.scatter(rej_minutes, ax.get_ylim()[1]*0.95,
                          marker='x', s=80, color=COLOR_NTP_REJECTED, alpha=0.8, zorder=5)

    # Add legend entry for rejected
    ax.scatter([], [], marker='x', s=80, color=COLOR_NTP_REJECTED,
              label=f'NTP rejected ({len([r for r in rejections if (r["timestamp"]-rejections[0]["timestamp"]).total_seconds()/60 <= 20])})', alpha=0.8)

    ax.set_xlabel('Time (minutes)', fontsize=13)
    ax.set_ylabel('Offset (ms)', fontsize=13)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / 'figure3_warmup_storm.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(FIGURE_DIR / 'figure3_warmup_storm.png', dpi=300, bbox_inches='tight')
    plt.close()

    ntp_with_ct = ntp_accepted.copy()
    ntp_with_ct['error'] = ntp_with_ct['chronotick_offset_ms'] - ntp_with_ct['ntp_offset_ms']
    mae = ntp_with_ct['error'].abs().mean()
    std = ntp_with_ct['error'].std()

    warmup_rejections = [r for r in rejections if (r['timestamp']-rejections[0]['timestamp']).total_seconds()/60 <= 20]
    latex_data['figure3'] = {
        'ntp_rejections': len(warmup_rejections),
        'max_z_score': f'{max([r["z_score"] for r in warmup_rejections]):.1f}σ' if warmup_rejections else 'N/A',
        'chronotick_mae': f'{mae:.3f}ms',
        'chronotick_std': f'{std:.3f}ms',
    }

    print(f"✓ Saved: {len(warmup_rejections)} rejections shown, CT MAE={mae:.3f}ms")


def figure4_thermal_extreme_clean():
    """
    Figure 4: Extreme Thermal Drift - ONE figure showing drift + tracking
    Story: System drifts 113ms/h, ChronoTick tracks it
    """
    print("\n" + "="*80)
    print("FIGURE 4: Extreme Thermal (Clean, ONE figure)")
    print("="*80)

    df = pd.read_csv('/home/jcernuda/tick_project/ChronoTick/results/experiment-8/local/chronotick_client_validation_20251021_121912.csv')
    data = df[df['elapsed_seconds'] <= 3600].copy()
    data['time_minutes'] = data['elapsed_seconds'] / 60

    ntp_data = data[data['has_ntp'] == True].copy()

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    if len(ntp_data) > 0:
        # Plot ChronoTick tracking
        ax.plot(data['time_minutes'], data['chronotick_offset_ms'],
                color=COLOR_CHRONOTICK, linewidth=2.5, label='ChronoTick', zorder=3)

        # Plot NTP (system clock drift)
        ax.scatter(ntp_data['time_minutes'], ntp_data['ntp_offset_ms'],
                   c=COLOR_SYSTEM_DRIFT, s=50, alpha=0.7, label='System clock drift (NTP)', zorder=4, marker='o')

        # Add linear fit for drift rate
        x = ntp_data['elapsed_seconds'].values
        y = ntp_data['ntp_offset_ms'].values
        slope, intercept, r_value, _, _ = stats.linregress(x, y)
        drift_rate = slope * 3600

        fit_line = slope * x + intercept
        ax.plot(ntp_data['time_minutes'], fit_line,
                color=COLOR_SYSTEM_DRIFT, linewidth=2, linestyle='--',
                label=f'Drift: {drift_rate:.1f} ms/h (R²={r_value**2:.2f})', zorder=2, alpha=0.7)

        ax.set_xlabel('Time (minutes)', fontsize=13)
        ax.set_ylabel('Offset (ms)', fontsize=13)
        ax.legend(loc='upper left', framealpha=0.9)
        ax.grid(True, alpha=0.25)

        plt.tight_layout()
        fig.savefig(FIGURE_DIR / 'figure4_thermal_extreme.pdf', dpi=300, bbox_inches='tight')
        fig.savefig(FIGURE_DIR / 'figure4_thermal_extreme.png', dpi=300, bbox_inches='tight')
        plt.close()

        ntp_with_ct = ntp_data.copy()
        ntp_with_ct['error'] = ntp_with_ct['chronotick_offset_ms'] - ntp_with_ct['ntp_offset_ms']
        mae = ntp_with_ct['error'].abs().mean()
        std = ntp_with_ct['error'].std()

        latex_data['figure4'] = {
            'drift_rate': f'{drift_rate:.1f} ms/h',
            'r_squared': f'{r_value**2:.3f}',
            'chronotick_mae': f'{mae:.1f}ms',
            'chronotick_std': f'{std:.1f}ms',
        }

        print(f"✓ Saved: Drift={drift_rate:.1f} ms/h, CT MAE={mae:.1f}ms")


def figure5_thermal_wandering_clean():
    """
    Figure 5: Thermal Wandering - ONE figure with non-linear drift
    Story: Drift rate changes over time (wandering)
    """
    print("\n" + "="*80)
    print("FIGURE 5: Thermal Wandering (Clean, ONE figure)")
    print("="*80)

    df = pd.read_csv('/home/jcernuda/tick_project/ChronoTick/results/experiment-3/homelab/data.csv')
    data = df[(df['elapsed_seconds'] >= 3*3600) & (df['elapsed_seconds'] <= 4*3600)].copy()
    data['time_minutes'] = data['elapsed_seconds'] / 60

    ntp_data = data[data['has_ntp'] == True].copy()

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    if len(ntp_data) > 0:
        # Plot ChronoTick
        ax.plot(data['time_minutes'], data['chronotick_offset_ms'],
                color=COLOR_CHRONOTICK, linewidth=2.5, label='ChronoTick', zorder=3)

        # Plot system drift (NTP)
        ax.plot(ntp_data['time_minutes'], ntp_data['ntp_offset_ms'],
                color=COLOR_SYSTEM_DRIFT, linewidth=2, marker='o', markersize=4,
                label='System drift (wandering)', zorder=2, alpha=0.8)

        # Calculate drift rates in 15-min windows and annotate
        for i in range(0, 60, 15):
            window_start = 180 + i
            window_end = 180 + i + 15
            window_data = ntp_data[(ntp_data['time_minutes'] >= window_start) &
                                  (ntp_data['time_minutes'] < window_end)]

            if len(window_data) > 1:
                x = window_data['elapsed_seconds'].values
                y = window_data['ntp_offset_ms'].values
                slope, _, _, _, _ = stats.linregress(x, y)
                rate = slope * 3600

                mid_time = (window_start + window_end) / 2
                ax.text(mid_time, ax.get_ylim()[1]*0.93, f'{rate:+.1f}\nms/h',
                       ha='center', va='top', fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.3))

        ax.set_xlabel('Time (minutes)', fontsize=13)
        ax.set_ylabel('Offset (ms)', fontsize=13)
        ax.legend(loc='upper left', framealpha=0.9)
        ax.grid(True, alpha=0.25)

        plt.tight_layout()
        fig.savefig(FIGURE_DIR / 'figure5_thermal_wandering.pdf', dpi=300, bbox_inches='tight')
        fig.savefig(FIGURE_DIR / 'figure5_thermal_wandering.png', dpi=300, bbox_inches='tight')
        plt.close()

        ntp_with_ct = ntp_data.copy()
        ntp_with_ct['error'] = ntp_with_ct['chronotick_offset_ms'] - ntp_with_ct['ntp_offset_ms']
        mae = ntp_with_ct['error'].abs().mean()
        std = ntp_with_ct['error'].std()

        offset_range = ntp_data['ntp_offset_ms'].max() - ntp_data['ntp_offset_ms'].min()

        latex_data['figure5'] = {
            'offset_range': f'{offset_range:.2f}ms',
            'chronotick_mae': f'{mae:.2f}ms',
            'chronotick_std': f'{std:.2f}ms',
        }

        print(f"✓ Saved: Offset range={offset_range:.2f}ms, CT σ={std:.2f}ms")


def main():
    """Generate all 5 clean figures"""
    print("="*80)
    print("GENERATING 5 CLEAN MICRO-EVALUATION FIGURES")
    print("="*80)

    figure1_wsl2_chaos_clean()
    figure2_exp9_recovery_clean()
    figure3_warmup_storm_clean()
    figure4_thermal_extreme_clean()
    figure5_thermal_wandering_clean()

    # Save data
    output_json = FIGURE_DIR / 'latex_data_clean.json'
    with open(output_json, 'w') as f:
        json.dump(latex_data, f, indent=2)

    print("\n" + "="*80)
    print("✅ ALL CLEAN FIGURES GENERATED")
    print("="*80)
    print(f"Saved to: {FIGURE_DIR}")
    print("\nFigures:")
    for fig_file in sorted(FIGURE_DIR.glob('figure*.pdf')):
        print(f"  - {fig_file.name}")


if __name__ == '__main__':
    main()
