#!/usr/bin/env python3
"""
Analyze what TimesFM is actually predicting vs what it should predict.
The key question: Why is TimesFM predicting small corrections when
the actual error vs NTP ground truth is 100s of ms?
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_data(summary_file, predictions_file):
    """Load both summary and predictions data."""
    df_summary = pd.read_csv(summary_file)
    df_predictions = pd.read_csv(predictions_file)

    df_summary['elapsed_hours'] = df_summary['elapsed_seconds'] / 3600
    df_predictions['elapsed_hours'] = df_predictions['timestamp'] / 3600

    return df_summary, df_predictions

def analyze_predictions_vs_truth(df_summary, df_predictions):
    """Analyze what TimesFM predicts vs actual NTP ground truth."""

    # Get only measurements with NTP ground truth
    df_with_truth = df_summary[df_summary['has_ntp'] == True].copy()

    print("\n" + "="*70)
    print("TIMESFM PREDICTION ANALYSIS")
    print("="*70)

    # For each NTP measurement, find the corresponding prediction
    results = []

    for idx, row in df_with_truth.iterrows():
        timestamp = row['timestamp']
        elapsed_h = row['elapsed_hours']

        # Find the prediction at this time (or closest)
        pred_row = df_predictions.iloc[(df_predictions['timestamp'] - timestamp).abs().argsort()[:1]]

        if len(pred_row) > 0:
            pred_row = pred_row.iloc[0]

            # What TimesFM predicted as the offset correction
            predicted_offset = pred_row['offset_correction_ms']

            # What the actual NTP ground truth offset was
            ntp_truth_offset = row['ntp_ground_truth_offset_ms']

            # The error ChronoTick made (using TimesFM prediction)
            chronotick_error = row['chronotick_error_ms']

            # The error system clock made
            system_error = row['system_error_ms']

            # TimesFM's prediction error (predicted offset - NTP truth)
            prediction_error = predicted_offset - ntp_truth_offset

            results.append({
                'elapsed_hours': elapsed_h,
                'predicted_offset': predicted_offset,
                'ntp_truth_offset': ntp_truth_offset,
                'prediction_error': prediction_error,
                'chronotick_error': chronotick_error,
                'system_error': system_error,
                'prediction_magnitude': abs(predicted_offset),
                'truth_magnitude': abs(ntp_truth_offset)
            })

    df_results = pd.DataFrame(results)

    # Summary statistics
    print(f"\nNumber of NTP measurements analyzed: {len(df_results)}")
    print(f"\nTimesFM Predicted Offset Statistics:")
    print(f"  Mean magnitude: {df_results['prediction_magnitude'].mean():.1f} ms")
    print(f"  Median magnitude: {df_results['prediction_magnitude'].median():.1f} ms")
    print(f"  Max magnitude: {df_results['prediction_magnitude'].max():.1f} ms")

    print(f"\nNTP Ground Truth Offset Statistics:")
    print(f"  Mean magnitude: {df_results['truth_magnitude'].mean():.1f} ms")
    print(f"  Median magnitude: {df_results['truth_magnitude'].median():.1f} ms")
    print(f"  Max magnitude: {df_results['truth_magnitude'].max():.1f} ms")

    print(f"\nTimesFM Prediction Error (Predicted - Truth):")
    print(f"  Mean: {df_results['prediction_error'].mean():.1f} ms")
    print(f"  Mean absolute: {df_results['prediction_error'].abs().mean():.1f} ms")
    print(f"  Median: {df_results['prediction_error'].median():.1f} ms")
    print(f"  Std: {df_results['prediction_error'].std():.1f} ms")
    print(f"  Max absolute: {df_results['prediction_error'].abs().max():.1f} ms")

    # Break down by time period
    print("\n" + "="*70)
    print("PREDICTION ERROR BY TIME PERIOD")
    print("="*70)

    periods = [
        (0, 2, "Hours 0-2"),
        (2, 4, "Hours 2-4"),
        (4, 6, "Hours 4-6"),
        (6, 8, "Hours 6-8"),
    ]

    for start, end, label in periods:
        period_data = df_results[(df_results['elapsed_hours'] >= start) &
                                 (df_results['elapsed_hours'] < end)]

        if len(period_data) > 0:
            pred_mean = period_data['prediction_magnitude'].mean()
            truth_mean = period_data['truth_magnitude'].mean()
            error_mean = period_data['prediction_error'].abs().mean()

            print(f"\n{label} ({len(period_data)} measurements):")
            print(f"  TimesFM predicted: {pred_mean:.1f} ms")
            print(f"  NTP truth: {truth_mean:.1f} ms")
            print(f"  Prediction error: {error_mean:.1f} ms")
            print(f"  Ratio (predicted/truth): {pred_mean/truth_mean:.2f}")

    return df_results

def plot_prediction_analysis(df_results, output_dir):
    """Create plots showing TimesFM predictions vs NTP ground truth."""

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Plot 1: TimesFM predicted offset vs NTP ground truth offset
    ax1 = axes[0]
    ax1.plot(df_results['elapsed_hours'], df_results['predicted_offset'],
             'b-', linewidth=2, label='TimesFM Predicted Offset', alpha=0.7, marker='o', markersize=4)
    ax1.plot(df_results['elapsed_hours'], df_results['ntp_truth_offset'],
             'r-', linewidth=2, label='NTP Ground Truth Offset', alpha=0.7, marker='s', markersize=4)
    ax1.set_ylabel('Offset (ms)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('TimesFM Predictions vs NTP Ground Truth', fontsize=13, fontweight='bold')

    # Plot 2: Prediction error magnitude over time
    ax2 = axes[1]
    ax2.plot(df_results['elapsed_hours'], df_results['prediction_error'].abs(),
             'purple', linewidth=2, alpha=0.7, marker='o', markersize=4)
    ax2.set_ylabel('Absolute Prediction Error (ms)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('TimesFM Prediction Error Magnitude (|Predicted - NTP Truth|)', fontsize=13, fontweight='bold')

    # Plot 3: Correlation plot - predicted vs truth
    ax3 = axes[2]
    ax3.scatter(df_results['truth_magnitude'], df_results['prediction_magnitude'],
                alpha=0.6, s=50, c=df_results['elapsed_hours'], cmap='viridis')

    # Add perfect prediction line
    max_val = max(df_results['truth_magnitude'].max(), df_results['prediction_magnitude'].max())
    ax3.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    ax3.set_xlabel('NTP Ground Truth Magnitude (ms)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('TimesFM Predicted Magnitude (ms)', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Prediction Accuracy: TimesFM vs NTP Truth (color = time)', fontsize=13, fontweight='bold')

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis',
                               norm=plt.Normalize(vmin=df_results['elapsed_hours'].min(),
                                                  vmax=df_results['elapsed_hours'].max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax3)
    cbar.set_label('Time (hours)', fontsize=11)

    plt.tight_layout()
    output_path = output_dir / "model_prediction_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPrediction analysis plots saved to: {output_path}")
    plt.close()

def main():
    """Main analysis function."""
    data_dir = Path("tsfm/results/ntp_correction_experiment/overnight_8hr_FIXED_20251014")
    summary_file = data_dir / "summary_backtracking_20251014_155930.csv"
    predictions_file = data_dir / "client_predictions_backtracking_20251014_155930.csv"

    print("\n" + "="*70)
    print("TIMESFM PREDICTION VS GROUND TRUTH ANALYSIS")
    print("="*70)

    # Load data
    print("\nLoading data...")
    df_summary, df_predictions = load_data(summary_file, predictions_file)

    # Analyze predictions
    df_results = analyze_predictions_vs_truth(df_summary, df_predictions)

    # Create plots
    print("\n" + "="*70)
    print("Creating prediction analysis plots...")
    plot_prediction_analysis(df_results, data_dir)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
