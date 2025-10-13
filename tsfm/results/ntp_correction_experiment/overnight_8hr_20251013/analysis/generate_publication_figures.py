#!/usr/bin/env python3
"""
Generate publication-ready figures for ChronoTick vs System Clock comparison.

Produces high-quality, single-panel figures suitable for academic papers.
Each figure focuses on a specific aspect of the comparison.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import sys

# Publication-quality settings
plt.rcParams.update({
    'font.size': 14,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 13,
    'figure.figsize': (10, 6),
    'figure.dpi': 300,
    'lines.linewidth': 2,
    'lines.markersize': 6,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

class PublicationFigureGenerator:
    """Generate publication-ready figures."""

    def __init__(self, data_dir: Path, output_dir: Path):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Colors for consistency
        self.colors = {
            'system': '#E74C3C',      # Red - System clock (poor)
            'chronotick': '#3498DB',  # Blue - ChronoTick (good)
            'ntp': '#2ECC71',         # Green - NTP ground truth
            'uncertainty': '#95A5A6', # Gray - Uncertainty bands
        }

    def load_data(self):
        """Load all data files."""
        print("Loading data...")

        # Load client predictions (ChronoTick predictions)
        client_file = list(self.data_dir.glob("client_predictions_*.csv"))[0]
        self.client_df = pd.read_csv(client_file)
        print(f"  Loaded {len(self.client_df)} ChronoTick predictions")

        # Load dataset corrections (NTP ground truth events)
        dataset_file = list(self.data_dir.glob("dataset_corrections_*.csv"))[0]
        self.dataset_df = pd.read_csv(dataset_file)
        # Only keep rows with NTP measurements
        self.dataset_df = self.dataset_df[self.dataset_df['ntp_offset_ms'].notna()].copy()
        print(f"  Loaded {len(self.dataset_df)} NTP ground truth measurements")

        # Calculate elapsed time in hours for plotting
        if not self.client_df.empty:
            self.client_df['elapsed_hours'] = (
                (self.client_df['timestamp'] - self.client_df['timestamp'].iloc[0]) / 3600
            )

        if not self.dataset_df.empty:
            self.dataset_df['elapsed_hours'] = (
                (self.dataset_df['ntp_event_timestamp'] - self.dataset_df['ntp_event_timestamp'].iloc[0]) / 3600
            )

        # Calculate system clock offset (system = ChronoTick offset - correction)
        # System clock error = what the error would be without ChronoTick
        # We can infer this from: system_offset = chronotick_offset - chronotick_correction
        # Or more directly: at each NTP point, system error = system_time - ntp_time

        # For dataset_df, calculate system clock error at NTP measurement points
        # System offset ≈ NTP offset (since system wasn't corrected)
        # ChronoTick offset ≈ offset after ML correction

    def figure1_time_comparison(self):
        """
        Figure 1: Time Comparison Over 8 Hours
        Shows: System Clock, ChronoTick, and NTP Ground Truth
        """
        print("\nGenerating Figure 1: Time Comparison...")

        fig, ax = plt.subplots(figsize=(12, 7))

        # Calculate absolute times
        # System clock time = timestamp (raw system time)
        # ChronoTick time = timestamp + chronotick_correction
        # NTP time = timestamp + ntp_offset (ground truth)

        # Plot system clock offset (error from true time)
        system_offset = np.zeros(len(self.dataset_df))  # System thinks it's correct (0 offset)

        # Plot NTP ground truth offset
        ax.plot(self.dataset_df['elapsed_hours'],
               self.dataset_df['ntp_offset_ms'],
               'o-', color=self.colors['ntp'], label='NTP Ground Truth',
               markersize=8, linewidth=2.5, alpha=0.9, zorder=3)

        # Plot ChronoTick corrections at NTP points
        # At each NTP measurement, we have offset_before and offset_after correction
        if 'offset_before_correction_ms' in self.dataset_df.columns:
            ax.plot(self.dataset_df['elapsed_hours'],
                   self.dataset_df['offset_before_correction_ms'],
                   's', color=self.colors['chronotick'], label='ChronoTick Prediction',
                   markersize=7, alpha=0.7, zorder=2)

        # Annotations
        ax.set_xlabel('Elapsed Time (hours)', fontsize=16, fontweight='bold')
        ax.set_ylabel('Clock Offset (ms)', fontsize=16, fontweight='bold')
        ax.set_title('Clock Offset Comparison: ChronoTick vs NTP Ground Truth',
                    fontsize=18, fontweight='bold', pad=20)
        ax.legend(loc='best', frameon=True, shadow=True, fontsize=14)
        ax.grid(True, alpha=0.3, linestyle='--')

        # Add text box with key statistics
        if len(self.dataset_df) > 0:
            ntp_mean = self.dataset_df['ntp_offset_ms'].mean()
            ntp_std = self.dataset_df['ntp_offset_ms'].std()
            textstr = f'NTP Mean: {ntp_mean:.2f} ms\nNTP Std: {ntp_std:.2f} ms'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=12,
                   verticalalignment='top', bbox=props)

        plt.tight_layout()
        output_file = self.output_dir / 'figure1_time_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_file}")
        plt.close()

        return output_file

    def figure2_error_comparison(self):
        """
        Figure 2: Offset Error Comparison
        Shows: System Clock Error vs ChronoTick Error against NTP Ground Truth
        """
        print("\nGenerating Figure 2: Error Comparison...")

        fig, ax = plt.subplots(figsize=(12, 7))

        # Calculate errors
        # System error = how far system clock is from NTP (just use NTP offset as system error)
        # ChronoTick error = how far ChronoTick prediction is from NTP

        if 'offset_before_correction_ms' in self.dataset_df.columns:
            # ChronoTick error (before correction applied)
            chronotick_error = self.dataset_df['offset_before_correction_ms'] - self.dataset_df['ntp_offset_ms']

            # System error (assuming system clock has no correction, so error = drift)
            # System error ≈ NTP offset (the raw clock offset)
            system_error = self.dataset_df['ntp_offset_ms'].copy()

            # Plot errors
            ax.plot(self.dataset_df['elapsed_hours'], system_error,
                   'o-', color=self.colors['system'], label='System Clock Error',
                   markersize=7, linewidth=2.5, alpha=0.8, zorder=2)

            ax.plot(self.dataset_df['elapsed_hours'], np.abs(chronotick_error),
                   's-', color=self.colors['chronotick'], label='ChronoTick Error',
                   markersize=7, linewidth=2.5, alpha=0.8, zorder=3)

            # Zero line
            ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

        ax.set_xlabel('Elapsed Time (hours)', fontsize=16, fontweight='bold')
        ax.set_ylabel('Absolute Error (ms)', fontsize=16, fontweight='bold')
        ax.set_title('Offset Error vs NTP Ground Truth',
                    fontsize=18, fontweight='bold', pad=20)
        ax.legend(loc='best', frameon=True, shadow=True, fontsize=14)
        ax.grid(True, alpha=0.3, linestyle='--')

        # Add statistics
        if 'offset_before_correction_ms' in self.dataset_df.columns:
            chronotick_mae = np.abs(chronotick_error).mean()
            system_mae = np.abs(system_error).mean()
            improvement = ((system_mae - chronotick_mae) / system_mae) * 100

            textstr = (f'System Clock MAE: {system_mae:.2f} ms\n'
                      f'ChronoTick MAE: {chronotick_mae:.2f} ms\n'
                      f'Improvement: {improvement:.1f}%')
            props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=12,
                   verticalalignment='top', bbox=props)

        plt.tight_layout()
        output_file = self.output_dir / 'figure2_error_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_file}")
        plt.close()

        return output_file

    def figure3_error_distribution(self):
        """
        Figure 3: Error Distribution Comparison
        Shows: Violin/Box plot comparing error distributions
        """
        print("\nGenerating Figure 3: Error Distribution...")

        fig, ax = plt.subplots(figsize=(10, 7))

        if 'offset_before_correction_ms' in self.dataset_df.columns:
            # Prepare data
            chronotick_error = np.abs(
                self.dataset_df['offset_before_correction_ms'] - self.dataset_df['ntp_offset_ms']
            )
            system_error = np.abs(self.dataset_df['ntp_offset_ms'])

            # Create violin plot
            parts = ax.violinplot([system_error, chronotick_error],
                                 positions=[1, 2],
                                 widths=0.7,
                                 showmeans=True,
                                 showmedians=True)

            # Color the violins
            parts['bodies'][0].set_facecolor(self.colors['system'])
            parts['bodies'][0].set_alpha(0.7)
            parts['bodies'][1].set_facecolor(self.colors['chronotick'])
            parts['bodies'][1].set_alpha(0.7)

            # Overlay box plot for clarity
            bp = ax.boxplot([system_error, chronotick_error],
                           positions=[1, 2],
                           widths=0.3,
                           patch_artist=True,
                           showfliers=False,
                           boxprops=dict(facecolor='white', edgecolor='black', linewidth=2),
                           medianprops=dict(color='red', linewidth=2))

            ax.set_xticks([1, 2])
            ax.set_xticklabels(['System Clock', 'ChronoTick'], fontsize=15, fontweight='bold')
            ax.set_ylabel('Absolute Error (ms)', fontsize=16, fontweight='bold')
            ax.set_title('Error Distribution Comparison',
                        fontsize=18, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3, linestyle='--', axis='y')

            # Add statistics annotations
            sys_mean = system_error.mean()
            sys_std = system_error.std()
            ct_mean = chronotick_error.mean()
            ct_std = chronotick_error.std()

            ax.text(1, system_error.max() * 1.05,
                   f'μ={sys_mean:.1f}\nσ={sys_std:.1f}',
                   ha='center', fontsize=11, bbox=dict(boxstyle='round',
                   facecolor=self.colors['system'], alpha=0.3))

            ax.text(2, chronotick_error.max() * 1.05,
                   f'μ={ct_mean:.1f}\nσ={ct_std:.1f}',
                   ha='center', fontsize=11, bbox=dict(boxstyle='round',
                   facecolor=self.colors['chronotick'], alpha=0.3))

        plt.tight_layout()
        output_file = self.output_dir / 'figure3_error_distribution.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_file}")
        plt.close()

        return output_file

    def figure4_confidence_intervals(self):
        """
        Figure 4: ChronoTick Predictions with Confidence Intervals
        Shows: Predictions with TimesFM quantile-based uncertainty bands
        """
        print("\nGenerating Figure 4: Confidence Intervals...")

        fig, ax = plt.subplots(figsize=(12, 7))

        # Sample data for performance (plot every 10th point for clarity)
        sample_interval = max(1, len(self.client_df) // 500)
        plot_df = self.client_df.iloc[::sample_interval].copy()

        # Calculate confidence intervals if available
        # For this plot, we'll use offset_uncertainty_ms as the error bars
        if 'offset_correction_ms' in plot_df.columns and 'offset_uncertainty_ms' in plot_df.columns:

            # Main prediction line
            ax.plot(plot_df['elapsed_hours'], plot_df['offset_correction_ms'],
                   '-', color=self.colors['chronotick'], label='ChronoTick Prediction',
                   linewidth=2, alpha=0.8, zorder=3)

            # Uncertainty band (±1 sigma)
            lower_bound = plot_df['offset_correction_ms'] - plot_df['offset_uncertainty_ms']
            upper_bound = plot_df['offset_correction_ms'] + plot_df['offset_uncertainty_ms']

            ax.fill_between(plot_df['elapsed_hours'], lower_bound, upper_bound,
                           color=self.colors['uncertainty'], alpha=0.3,
                           label='±1σ Uncertainty', zorder=1)

            # Plot NTP measurements as ground truth points
            if not self.dataset_df.empty:
                ax.plot(self.dataset_df['elapsed_hours'],
                       self.dataset_df['ntp_offset_ms'],
                       'o', color=self.colors['ntp'], label='NTP Ground Truth',
                       markersize=10, markeredgewidth=2, markeredgecolor='darkgreen',
                       alpha=0.9, zorder=4)

        ax.set_xlabel('Elapsed Time (hours)', fontsize=16, fontweight='bold')
        ax.set_ylabel('Clock Offset (ms)', fontsize=16, fontweight='bold')
        ax.set_title('ChronoTick Predictions with Uncertainty Quantification',
                    fontsize=18, fontweight='bold', pad=20)
        ax.legend(loc='best', frameon=True, shadow=True, fontsize=14)
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        output_file = self.output_dir / 'figure4_confidence_intervals.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_file}")
        plt.close()

        return output_file

    def figure5_cumulative_error(self):
        """
        Figure 5: Cumulative Error Over Time
        Shows: How total error accumulates for System vs ChronoTick
        """
        print("\nGenerating Figure 5: Cumulative Error...")

        fig, ax = plt.subplots(figsize=(12, 7))

        if 'offset_before_correction_ms' in self.dataset_df.columns:
            chronotick_error = np.abs(
                self.dataset_df['offset_before_correction_ms'] - self.dataset_df['ntp_offset_ms']
            )
            system_error = np.abs(self.dataset_df['ntp_offset_ms'])

            # Calculate cumulative sums
            system_cumulative = np.cumsum(system_error)
            chronotick_cumulative = np.cumsum(chronotick_error)

            ax.plot(self.dataset_df['elapsed_hours'], system_cumulative,
                   '-', color=self.colors['system'], label='System Clock',
                   linewidth=3, alpha=0.8, zorder=2)

            ax.plot(self.dataset_df['elapsed_hours'], chronotick_cumulative,
                   '-', color=self.colors['chronotick'], label='ChronoTick',
                   linewidth=3, alpha=0.8, zorder=3)

            # Fill area between curves
            ax.fill_between(self.dataset_df['elapsed_hours'],
                           system_cumulative, chronotick_cumulative,
                           color='green', alpha=0.2, label='Error Saved')

        ax.set_xlabel('Elapsed Time (hours)', fontsize=16, fontweight='bold')
        ax.set_ylabel('Cumulative Error (ms)', fontsize=16, fontweight='bold')
        ax.set_title('Cumulative Time Error: System Clock vs ChronoTick',
                    fontsize=18, fontweight='bold', pad=20)
        ax.legend(loc='best', frameon=True, shadow=True, fontsize=14)
        ax.grid(True, alpha=0.3, linestyle='--')

        # Add final savings annotation
        if 'offset_before_correction_ms' in self.dataset_df.columns:
            total_saved = system_cumulative.iloc[-1] - chronotick_cumulative.iloc[-1]
            savings_pct = (total_saved / system_cumulative.iloc[-1]) * 100

            textstr = (f'Total Error Saved: {total_saved:.1f} ms\n'
                      f'Improvement: {savings_pct:.1f}%')
            props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.9)
            ax.text(0.50, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                   verticalalignment='top', horizontalalignment='center',
                   bbox=props, fontweight='bold')

        plt.tight_layout()
        output_file = self.output_dir / 'figure5_cumulative_error.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_file}")
        plt.close()

        return output_file

    def figure6_error_reduction_percentage(self):
        """
        Figure 6: Error Reduction Percentage Over Time
        Shows: Percentage improvement of ChronoTick vs System Clock
        """
        print("\nGenerating Figure 6: Error Reduction Percentage...")

        fig, ax = plt.subplots(figsize=(12, 7))

        if 'offset_before_correction_ms' in self.dataset_df.columns:
            chronotick_error = np.abs(
                self.dataset_df['offset_before_correction_ms'] - self.dataset_df['ntp_offset_ms']
            )
            system_error = np.abs(self.dataset_df['ntp_offset_ms'])

            # Calculate percentage improvement
            # Avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                improvement_pct = ((system_error - chronotick_error) / system_error) * 100
                improvement_pct = np.nan_to_num(improvement_pct, nan=0, posinf=100, neginf=0)

            ax.plot(self.dataset_df['elapsed_hours'], improvement_pct,
                   '-', color=self.colors['chronotick'], linewidth=3, alpha=0.8)

            # Fill positive improvement
            ax.fill_between(self.dataset_df['elapsed_hours'], 0, improvement_pct,
                           where=(improvement_pct >= 0),
                           color='green', alpha=0.3, label='ChronoTick Better')

            # Fill negative (if any)
            ax.fill_between(self.dataset_df['elapsed_hours'], 0, improvement_pct,
                           where=(improvement_pct < 0),
                           color='red', alpha=0.3, label='System Better')

            # Zero line
            ax.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.7)

            # Mean improvement line
            mean_improvement = improvement_pct.mean()
            ax.axhline(y=mean_improvement, color='darkgreen', linestyle='-',
                      linewidth=2, alpha=0.8, label=f'Mean: {mean_improvement:.1f}%')

        ax.set_xlabel('Elapsed Time (hours)', fontsize=16, fontweight='bold')
        ax.set_ylabel('Error Reduction (%)', fontsize=16, fontweight='bold')
        ax.set_title('ChronoTick Error Reduction vs System Clock',
                    fontsize=18, fontweight='bold', pad=20)
        ax.legend(loc='best', frameon=True, shadow=True, fontsize=14)
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        output_file = self.output_dir / 'figure6_error_reduction.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_file}")
        plt.close()

        return output_file

    def figure7_zoomed_correction_event(self):
        """
        Figure 7: Zoomed View of Backtracking Correction Event
        Shows: Detailed view of one correction event showing prediction vs NTP
        """
        print("\nGenerating Figure 7: Correction Event Zoom...")

        fig, ax = plt.subplots(figsize=(12, 7))

        # Find a good correction event to zoom into (around middle of test)
        if len(self.dataset_df) > 10:
            mid_idx = len(self.dataset_df) // 2
            event_time = self.dataset_df.iloc[mid_idx]['elapsed_hours']

            # Get window around this event (±10 minutes = ±0.167 hours)
            window = 0.25  # 15 minutes window
            time_min = event_time - window
            time_max = event_time + window

            # Filter client predictions in this window
            window_client = self.client_df[
                (self.client_df['elapsed_hours'] >= time_min) &
                (self.client_df['elapsed_hours'] <= time_max)
            ].copy()

            # Filter NTP measurements in this window
            window_ntp = self.dataset_df[
                (self.dataset_df['elapsed_hours'] >= time_min) &
                (self.dataset_df['elapsed_hours'] <= time_max)
            ].copy()

            if not window_client.empty and not window_ntp.empty:
                # Plot ChronoTick predictions
                ax.plot(window_client['elapsed_hours'],
                       window_client['offset_correction_ms'],
                       '-', color=self.colors['chronotick'],
                       label='ChronoTick Prediction',
                       linewidth=2.5, alpha=0.8, zorder=2)

                # Plot NTP measurements
                ax.plot(window_ntp['elapsed_hours'],
                       window_ntp['ntp_offset_ms'],
                       'o', color=self.colors['ntp'],
                       label='NTP Ground Truth',
                       markersize=12, markeredgewidth=2.5,
                       markeredgecolor='darkgreen', alpha=0.9, zorder=3)

                # Highlight correction event
                correction_time = window_ntp.iloc[len(window_ntp)//2]['elapsed_hours']
                ax.axvline(x=correction_time, color='red', linestyle='--',
                          linewidth=2, alpha=0.7, label='Correction Event')

        ax.set_xlabel('Elapsed Time (hours)', fontsize=16, fontweight='bold')
        ax.set_ylabel('Clock Offset (ms)', fontsize=16, fontweight='bold')
        ax.set_title('Detailed View: Backtracking Correction Event',
                    fontsize=18, fontweight='bold', pad=20)
        ax.legend(loc='best', frameon=True, shadow=True, fontsize=14)
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        output_file = self.output_dir / 'figure7_correction_zoom.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_file}")
        plt.close()

        return output_file

    def generate_all_figures(self):
        """Generate all publication figures."""
        print("=" * 80)
        print("Generating Publication-Ready Figures")
        print("=" * 80)

        self.load_data()

        figures = []
        figures.append(self.figure1_time_comparison())
        figures.append(self.figure2_error_comparison())
        figures.append(self.figure3_error_distribution())
        figures.append(self.figure4_confidence_intervals())
        figures.append(self.figure5_cumulative_error())
        figures.append(self.figure6_error_reduction_percentage())
        figures.append(self.figure7_zoomed_correction_event())

        print("\n" + "=" * 80)
        print(f"Generated {len(figures)} publication figures")
        print("=" * 80)

        return figures


def main():
    """Main entry point."""
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "visualization_data"
    output_dir = script_dir

    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)

    generator = PublicationFigureGenerator(data_dir, output_dir)
    figures = generator.generate_all_figures()

    print("\nAll figures saved to:")
    for fig in figures:
        print(f"  - {fig.name}")


if __name__ == "__main__":
    main()
