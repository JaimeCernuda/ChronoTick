#!/usr/bin/env python3
"""
Final NTP Correction Validation - Shows how methods handle different error magnitudes.
Tests small/medium/large errors (both positive and negative) with realistic noisy measurements.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def apply_correction(timestamps, measured, ntp_time, ntp_offset, method, debug=False):
    """Apply correction method."""
    ntp_idx = np.argmin(np.abs(timestamps - ntp_time))
    predicted = measured[ntp_idx]
    error = ntp_offset - predicted
    corrected = measured.copy()

    if method == 'none':
        return corrected

    start_time = timestamps[0]
    interval = ntp_time - start_time

    if debug:
        print(f"\n{method.upper()} Method:")
        print(f"  Error to distribute: {error*1000:.3f}ms")
        print(f"  Interval: {interval:.1f}s, Points: {ntp_idx+1}")

    if method == 'linear':
        for i in range(ntp_idx + 1):
            alpha = (timestamps[i] - start_time) / interval if interval > 0 else 0
            corrected[i] += alpha * error

        if debug:
            corr_start = 0.0 * error * 1000
            corr_end = 1.0 * error * 1000
            print(f"  Correction at t=0: {corr_start:.3f}ms")
            print(f"  Correction at NTP: {corr_end:.3f}ms")

    elif method == 'drift_aware':
        sigma_offset, sigma_drift = 0.001, 0.0001
        if interval > 0:
            var_offset = sigma_offset**2
            var_drift = (sigma_drift * interval)**2
            var_total = var_offset + var_drift
            if var_total > 0:
                w_offset = var_offset / var_total
                offset_corr = w_offset * error
                drift_corr = ((1 - w_offset) * error) / interval
                for i in range(ntp_idx + 1):
                    t_elapsed = timestamps[i] - start_time
                    corrected[i] += offset_corr + (drift_corr * t_elapsed)

                if debug:
                    corr_start = offset_corr * 1000
                    corr_end = (offset_corr + drift_corr * interval) * 1000
                    print(f"  Offset weight: {w_offset:.4f}, Drift weight: {1-w_offset:.4f}")
                    print(f"  Correction at t=0: {corr_start:.3f}ms")
                    print(f"  Correction at NTP: {corr_end:.3f}ms")

    elif method == 'advanced':
        sigma_m, sigma_p, sigma_d = 0.001, 0.001, 0.0001
        sigma_sq_base = sigma_m**2 + sigma_p**2
        weights = {}
        total_w = 0
        for i in range(ntp_idx + 1):
            dt = timestamps[i] - start_time
            sigma_sq = sigma_sq_base + (sigma_d * dt)**2
            w = sigma_sq
            weights[i] = w
            total_w += w
        if total_w > 0:
            for i in weights:
                corrected[i] += (weights[i] / total_w) * error

            if debug:
                correction_at_start = (weights[0] / total_w) * error * 1000
                correction_at_end = (weights[ntp_idx] / total_w) * error * 1000
                weight_ratio = weights[ntp_idx] / weights[0] if weights[0] > 0 else 0
                print(f"  Weight at t=0: {weights[0]:.6f}, at NTP: {weights[ntp_idx]:.6f}")
                print(f"  Weight ratio (NTP/start): {weight_ratio:.2f}x")
                print(f"  Correction at t=0: {correction_at_start:.3f}ms")
                print(f"  Correction at NTP: {correction_at_end:.3f}ms")

    return corrected

def create_scenario(base_offset, drift_rate, ntp_error, num_points=100):
    """Create synthetic scenario with specified NTP error."""
    timestamps = np.arange(num_points, dtype=np.float64)
    true_offsets = base_offset + drift_rate * timestamps
    noise = np.random.normal(0, 0.0005, num_points)
    measured = true_offsets + noise

    ntp_idx = int(num_points * 0.6)
    ntp_time = timestamps[ntp_idx]
    ntp_offset = true_offsets[ntp_idx] + ntp_error

    return timestamps, measured, true_offsets, ntp_time, ntp_offset, ntp_idx

# Create 6 test scenarios
scenarios = [
    ('Small +', 2e-3),   ('Small -', -2e-3),
    ('Medium +', 10e-3), ('Medium -', -10e-3),
    ('Large +', 50e-3),  ('Large -', -50e-3)
]

methods = ['none', 'linear', 'drift_aware', 'advanced']
colors = {'none': 'blue', 'linear': 'green', 'drift_aware': 'orange', 'advanced': 'red'}

fig, axes = plt.subplots(3, 2, figsize=(18, 14))
fig.suptitle('NTP Correction Validation: How Methods Handle Different Error Magnitudes',
             fontsize=14, fontweight='bold')

for ax_idx, ((label, ntp_error), ax) in enumerate(zip(scenarios, axes.flat)):
    timestamps, measured, true_offsets, ntp_time, ntp_offset, ntp_idx = create_scenario(
        base_offset=-50e-3, drift_rate=100e-6, ntp_error=ntp_error
    )

    # Only plot data UP TO the NTP point (not after)
    plot_mask = timestamps <= ntp_time
    plot_times = timestamps[plot_mask]

    # Plot ground truth
    ax.plot(plot_times, true_offsets[plot_mask] * 1000, 'k--', linewidth=2, alpha=0.5,
            label='True offset', zorder=1)

    # Plot measured (before correction)
    ax.plot(plot_times, measured[plot_mask] * 1000, color='gray', linewidth=1, alpha=0.5,
            label='Measured (noisy)', zorder=2)

    # Apply and plot each correction method
    for method in methods:
        corrected = apply_correction(timestamps, measured, ntp_time, ntp_offset, method)
        ax.plot(plot_times, corrected[plot_mask] * 1000, color=colors[method], linewidth=2,
                label=f'{method}', alpha=0.8, zorder=3)

    # Mark NTP measurement
    ax.axvline(x=ntp_time, color='purple', linestyle=':', linewidth=2, alpha=0.5)
    ax.scatter([ntp_time], [ntp_offset * 1000], color='purple', s=200,
              marker='*', zorder=5, edgecolors='black', linewidths=1, label='NTP')

    # Calculate errors at NTP point for each method
    error_summary = []
    for method in methods:
        corrected = apply_correction(timestamps, measured, ntp_time, ntp_offset, method)
        error_at_ntp = abs(corrected[ntp_idx] - ntp_offset) * 1000
        error_summary.append(f"{method}: {error_at_ntp:.2f}ms")

    ax.set_xlabel('Time (seconds)', fontsize=10)
    ax.set_ylabel('Clock Offset (ms)', fontsize=10)
    ax.set_title(f'{label} Error ({ntp_error*1000:+.0f}ms)\nError at NTP: ' + ', '.join(error_summary),
                fontsize=11)
    ax.legend(fontsize=8, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
output_dir = Path('results/ntp_correction_experiment/synthetic_validation')
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / 'ntp_correction_final_validation.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved: {output_path}")
print("\nKEY OBSERVATIONS:")
print("• NONE: No correction applied")
print("• LINEAR: Forces exact convergence at NTP (may over-correct)")
print("• DRIFT_AWARE: Similar to linear with small drift component")
print("• ADVANCED: Conservative correction (doesn't force exact NTP match)")
plt.close()
