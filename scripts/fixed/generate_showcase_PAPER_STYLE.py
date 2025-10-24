#!/usr/bin/env python3
"""
Paper-Styled Version - Both offsets shown with proper paper formatting

Matches paper style:
- No title (caption in LaTeX)
- Consistent colors: purple squares (system), blue/orange circles (ChronoTick)
- Simple, clean legend
- Proper font sizes
- No unnecessary styling
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Minimal paper style (no excessive styling)
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

def generate_paper_style_figure(csv_path, output_dir, output_name):
    """
    Generate paper-styled synchronized figure showing both as offsets.
    """
    print(f"\n{'='*80}")
    print(f"GENERATING: {output_name}")
    print('='*80)

    df = pd.read_csv(csv_path)
    ntp_df = df[df['has_ntp'] == True].copy()

    print(f"Total samples: {len(df)}")
    print(f"NTP samples: {len(ntp_df)}")

    ntp_df['elapsed_hours'] = ntp_df['elapsed_seconds'] / 3600

    # Both as offsets
    system_offset = ntp_df['ntp_offset_ms']
    chronotick_offset = ntp_df['chronotick_offset_ms']
    chronotick_uncertainty = ntp_df['chronotick_uncertainty_ms']

    print(f"\nOffset statistics:")
    print(f"  System offset mean: {system_offset.mean():.3f} ms")
    print(f"  ChronoTick offset mean: {chronotick_offset.mean():.3f} ms")
    print(f"  Prediction MAE: {(chronotick_offset - system_offset).abs().mean():.3f} ms")

    # Create figure - simple, no title
    fig, ax = plt.subplots(figsize=(10, 5))

    # ChronoTick uncertainty band (light blue/orange)
    chronotick_lower = chronotick_offset - chronotick_uncertainty
    chronotick_upper = chronotick_offset + chronotick_uncertainty

    ax.fill_between(ntp_df['elapsed_hours'], chronotick_lower, chronotick_upper,
                     color='#0072B2', alpha=0.15, linewidth=0, zorder=2)

    # System Clock offset (purple squares) - matching paper style
    ax.scatter(ntp_df['elapsed_hours'], system_offset,
               color='#CC79A7', marker='s', s=20, alpha=0.6,
               label='System Clock', zorder=4)

    # ChronoTick offset (blue circles) - matching paper style
    ax.scatter(ntp_df['elapsed_hours'], chronotick_offset,
               color='#0072B2', marker='o', s=20, alpha=0.7,
               label='ChronoTick', zorder=3)

    # Perfect sync reference line
    ax.axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.5,
               label='Perfect Sync', zorder=1)

    # Styling - match paper figures
    ax.set_xlabel('Time (hours)', fontsize=10)
    ax.set_ylabel('Offset from NTP (ms)', fontsize=10)
    ax.legend(loc='upper left', frameon=True, fancybox=False, shadow=False)
    ax.grid(True, alpha=0.3)

    # Set x-axis
    max_hours = int(np.ceil(ntp_df['elapsed_hours'].max()))
    ax.set_xlim(0, max_hours)
    ax.set_xticks(range(0, max_hours + 1))

    plt.tight_layout()

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = output_dir / f"{output_name}.pdf"
    png_path = output_dir / f"{output_name}.png"

    plt.savefig(pdf_path, bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')

    print(f"\n✓ Saved: {pdf_path}")
    print(f"✓ Saved: {png_path}")

    plt.close()

    return system_offset.mean(), chronotick_offset.mean()

def generate_unsynchronized_paper_style(csv_path, output_dir, output_name):
    """
    Generate paper-styled unsynchronized figure.
    """
    print(f"\n{'='*80}")
    print(f"GENERATING: {output_name}")
    print('='*80)

    df = pd.read_csv(csv_path)
    ntp_df = df[df['has_ntp'] == True].copy()

    print(f"Total samples: {len(df)}")
    print(f"NTP samples: {len(ntp_df)}")

    ntp_df['elapsed_hours'] = ntp_df['elapsed_seconds'] / 3600

    # Both as offsets
    system_offset = ntp_df['ntp_offset_ms']
    chronotick_offset = ntp_df['chronotick_offset_ms']
    chronotick_uncertainty = ntp_df['chronotick_uncertainty_ms']

    # Calculate drift
    hours = ntp_df['elapsed_hours'].values
    drift_coef = np.polyfit(hours, system_offset, 1)
    drift_rate = drift_coef[0]

    print(f"\nOffset statistics:")
    print(f"  System drift rate: {drift_rate:+.3f} ms/hour")
    print(f"  ChronoTick offset mean: {chronotick_offset.mean():.3f} ms")
    print(f"  Prediction MAE: {(chronotick_offset - system_offset).abs().mean():.3f} ms")

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))

    # ChronoTick uncertainty band
    chronotick_lower = chronotick_offset - chronotick_uncertainty
    chronotick_upper = chronotick_offset + chronotick_uncertainty

    ax.fill_between(ntp_df['elapsed_hours'], chronotick_lower, chronotick_upper,
                     color='#0072B2', alpha=0.15, linewidth=0, zorder=2)

    # System Clock offset with drift (purple squares)
    ax.scatter(ntp_df['elapsed_hours'], system_offset,
               color='#CC79A7', marker='s', s=20, alpha=0.6,
               label='System Clock', zorder=4)

    # Drift trend line
    drift_line = drift_coef[0] * hours + drift_coef[1]
    ax.plot(hours, drift_line, color='#D55E00', linewidth=2, linestyle='--',
            alpha=0.7, label='System Drift', zorder=5)

    # ChronoTick offset (blue circles)
    ax.scatter(ntp_df['elapsed_hours'], chronotick_offset,
               color='#0072B2', marker='o', s=20, alpha=0.7,
               label='ChronoTick', zorder=3)

    # Perfect sync reference
    ax.axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.5,
               label='Perfect Sync', zorder=1)

    # Styling
    ax.set_xlabel('Time (hours)', fontsize=10)
    ax.set_ylabel('Offset from NTP (ms)', fontsize=10)
    ax.legend(loc='upper left', frameon=True, fancybox=False, shadow=False)
    ax.grid(True, alpha=0.3)

    # Set x-axis
    max_hours = int(np.ceil(ntp_df['elapsed_hours'].max()))
    ax.set_xlim(0, max_hours)
    ax.set_xticks(range(0, max_hours + 1))

    plt.tight_layout()

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = output_dir / f"{output_name}.pdf"
    png_path = output_dir / f"{output_name}.png"

    plt.savefig(pdf_path, bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')

    print(f"\n✓ Saved: {pdf_path}")
    print(f"✓ Saved: {png_path}")

    plt.close()

def main():
    """Generate paper-styled showcase figures."""
    print("="*80)
    print("PAPER-STYLED SHOWCASE FIGURES (NO TITLES, CONSISTENT COLORS)")
    print("="*80)
    print("\nStyle features:")
    print("  • No titles (captions in LaTeX)")
    print("  • Purple squares: System Clock")
    print("  • Blue circles: ChronoTick")
    print("  • Light blue band: ±1σ uncertainty")
    print("  • Orange dashed: Drift trend (unsynchronized only)")
    print("  • Clean legend, minimal styling")

    output_dir = Path("results/figures_corrected/showcase")

    # Figure 3.1: Synchronized (both as offsets)
    sync_csv = Path("results/experiment-3/homelab/data.csv")
    if sync_csv.exists():
        generate_paper_style_figure(sync_csv, output_dir, "3.1_synchronized_PAPER_STYLE")
    else:
        print(f"\n⚠️  Synchronized dataset not found: {sync_csv}")

    # Figure 3.2: Unsynchronized (both as offsets)
    unsync_csv = Path("results/experiment-7/homelab/chronotick_client_validation_20251020_221631.csv")
    if unsync_csv.exists():
        generate_unsynchronized_paper_style(unsync_csv, output_dir, "3.2_unsynchronized_PAPER_STYLE")
    else:
        print(f"\n⚠️  Unsynchronized dataset not found: {unsync_csv}")

    print("\n" + "="*80)
    print("PAPER-STYLED FIGURES COMPLETE")
    print("="*80)
    print(f"\nOutput directory: {output_dir.absolute()}")
    print("\nGenerated files:")
    print("  • 3.1_synchronized_PAPER_STYLE.pdf")
    print("  • 3.2_unsynchronized_PAPER_STYLE.pdf")
    print("\nThese match the visual style of other paper figures:")
    print("  - No titles (use LaTeX captions)")
    print("  - Consistent color scheme")
    print("  - Clean, minimal styling")
    print("  - Both show offsets (not mixing offset + error)")

if __name__ == "__main__":
    main()
