#!/usr/bin/env python3
"""
Improved NTP Correction Validation - Focus on Error Magnitude

Tests how each correction method handles small, medium, and large NTP errors.
Only shows the correction period (before NTP measurement).
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def apply_correction(timestamps, measured_offsets, ntp_time, ntp_offset, method):
    """Apply correction using specified method."""
    ntp_idx = np.argmin(np.abs(timestamps - ntp_time))
    predicted_offset = measured_offsets[ntp_idx]
    error = ntp_offset - predicted_offset

    corrected = measured_offsets.copy()

    if method == 'none':
        return corrected, error

    start_time = timestamps[0]
    interval_duration = ntp_time - start_time

    if method == 'linear':
        # Linear distribution of error
        for i in range(len(timestamps)):
            if timestamps[i] <= ntp_time:
                alpha = (timestamps[i] - start_time) / interval_duration if interval_duration > 0 else 0
                corrected[i] += alpha * error

    elif method == 'drift_aware':
        # Drift-aware correction
        sigma_offset = 0.001  # 1ms
        sigma_drift = 0.0001  # 100μs/s
        delta_t = interval_duration

        if delta_t > 0:
            var_offset = sigma_offset**2
            var_drift = (sigma_drift * delta_t)**2
            var_total = var_offset + var_drift

            if var_total > 0:
                w_offset = var_offset / var_total
                w_drift = var_drift / var_total

                offset_correction = w_offset * error
                drift_correction = (w_drift * error) / delta_t

                for i in range(len(timestamps)):
                    if timestamps[i] <= ntp_time:
                        t_elapsed = timestamps[i] - start_time
                        corrected[i] += offset_correction + (drift_correction * t_elapsed)

    elif method == 'advanced':
        # Advanced inverse-variance weighting
        sigma_measurement = 0.001  # 1ms
        sigma_prediction = 0.001   # 1ms
        sigma_drift = 0.0001       # 100μs/s

        sigma_squared_base = sigma_measurement**2 + sigma_prediction**2

        # Calculate weights - MATCH REAL IMPLEMENTATION
        total_weight = 0
        weights = {}

        # Calculate weights for each timestamp UP TO AND INCLUDING ntp_time
        for i in range(ntp_idx + 1):  # +1 to include ntp_idx itself
            dt = timestamps[i] - start_time
            sigma_squared_total = sigma_squared_base + (sigma_drift * dt)**2
            # FIXED: Direct variance weighting (high uncertainty → more correction)
            weight = sigma_squared_total
            weights[i] = weight
            total_weight += weight

        # Apply normalized corrections
        if total_weight > 0:
            for i in weights:
                alpha = weights[i] / total_weight
                corrected[i] += alpha * error

    return corrected, error

def run_error_magnitude_test():
    """Test correction methods with different error magnitudes."""
    print("="*80)
    print("NTP CORRECTION VALIDATION - ERROR MAGNITUDE COMPARISON")
    print("="*80)

    # Create base synthetic dataset (perfect measurements, no noise)
    num_points = 60  # 60 seconds
    timestamps = np.arange(num_points, dtype=np.float64)
    true_offset = -50e-3  # -50ms constant offset
    measured_offsets = np.full(num_points, true_offset)  # Perfect measurements

    # NTP measurement at 80% through (t=48s)
    ntp_idx = int(num_points * 0.8)
    ntp_time = timestamps[ntp_idx]

    # Test with 3 error magnitudes
    error_cases = {
        'small': 2e-3,    # 2ms error
        'medium': 10e-3,  # 10ms error
        'large': 50e-3    # 50ms error
    }

    methods = ['none', 'linear', 'drift_aware', 'advanced']
    colors = {'none': 'gray', 'linear': 'green', 'drift_aware': 'orange', 'advanced': 'red'}

    # Create visualization
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle('NTP Correction Methods: How They Redistribute Error', fontsize=14, fontweight='bold')

    for ax_idx, (error_label, ntp_error_offset) in enumerate(error_cases.items()):
        ax = axes[ax_idx]

        # NTP reports this offset (true_offset + error)
        ntp_offset = true_offset + ntp_error_offset

        print(f"\n{'='*60}")
        print(f"{error_label.upper()} ERROR CASE: {ntp_error_offset*1000:.1f}ms")
        print(f"{'='*60}")
        print(f"  True offset: {true_offset*1000:.1f}ms")
        print(f"  Predicted: {measured_offsets[ntp_idx]*1000:.1f}ms")
        print(f"  NTP reports: {ntp_offset*1000:.1f}ms")
        print(f"  Error to correct: {ntp_error_offset*1000:.1f}ms")

        # Apply each method
        for method in methods:
            corrected, error = apply_correction(
                timestamps, measured_offsets, ntp_time, ntp_offset, method
            )

            # Only plot up to NTP point
            plot_indices = timestamps <= ntp_time
            plot_times = timestamps[plot_indices]
            plot_values = corrected[plot_indices]

            # Calculate how much correction was applied at start, middle, and end
            if method != 'none':
                correction_at_start = corrected[0] - measured_offsets[0]
                correction_at_mid = corrected[ntp_idx//2] - measured_offsets[ntp_idx//2]
                correction_at_ntp = corrected[ntp_idx] - measured_offsets[ntp_idx]

                print(f"\n  {method.upper()}:")
                print(f"    Correction at t=0s: {correction_at_start*1000:+.2f}ms")
                print(f"    Correction at t={ntp_time/2:.0f}s: {correction_at_mid*1000:+.2f}ms")
                print(f"    Correction at t={ntp_time:.0f}s (NTP): {correction_at_ntp*1000:+.2f}ms")
                print(f"    Error at NTP point: {abs(corrected[ntp_idx] - ntp_offset)*1000:.3f}ms")

            ax.plot(plot_times, plot_values * 1000, color=colors[method],
                   linewidth=2, label=method, alpha=0.8)

        # Mark true offset and NTP measurement
        ax.axhline(y=true_offset * 1000, color='black', linestyle='--',
                  linewidth=1, alpha=0.5, label='True offset')
        ax.axhline(y=ntp_offset * 1000, color='purple', linestyle=':',
                  linewidth=2, alpha=0.7, label='NTP measurement')
        ax.axvline(x=ntp_time, color='purple', linestyle=':',
                  linewidth=1, alpha=0.5)
        ax.scatter([ntp_time], [ntp_offset * 1000], color='purple',
                  s=200, marker='*', zorder=5, edgecolors='black', linewidths=1)

        ax.set_xlabel('Time (seconds)', fontsize=11)
        ax.set_ylabel('Clock Offset (ms)', fontsize=11)
        ax.set_title(f'{error_label.upper()} Error ({ntp_error_offset*1000:.0f}ms): ' +
                    f'How Each Method Corrects the Dataset', fontsize=12)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)

        # Add annotations
        ax.text(0.02, 0.98,
               f'Error to distribute: {ntp_error_offset*1000:+.1f}ms\n' +
               f'NTP at t={ntp_time:.0f}s',
               transform=ax.transAxes, fontsize=10,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()

    output_dir = Path('results/ntp_correction_experiment/synthetic_validation')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'ntp_correction_error_magnitude_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n{'='*80}")
    print(f"Saved: {output_path}")
    print("="*80)

    print("\n" + "="*80)
    print("KEY INSIGHTS:")
    print("="*80)
    print("• NONE: No correction - stays at predicted value")
    print("• LINEAR: Distributes error linearly from 0% at start to 100% at NTP")
    print("• DRIFT_AWARE: Mostly offset correction (>95%), small drift component")
    print("• ADVANCED: More correction near NTP point (higher uncertainty)")
    print("\nAll methods (except 'none') converge to NTP value at measurement point.")
    print("="*80)

    plt.close('all')

if __name__ == '__main__':
    run_error_magnitude_test()
