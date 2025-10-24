#!/usr/bin/env python3
"""
Exact Paper Color Scheme - Matching existing 3.1 figure style

Colors from existing figures:
- #0072B2: Blue (ChronoTick line/offset)
- #D55E00: Orange (NTP markers/system clock)
- #56B4E9: Light blue (uncertainty band with label "±1σ")
- #009E73: Green (optional, for errors)

Style:
- No titles
- Label uncertainty as "±1σ" not just "uncertainty"
- Use scatter + line for appropriate elements
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def generate_synchronized_exact_colors(csv_path, output_dir, output_name):
    """Generate synchronized figure with exact paper colors."""
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

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))

    # ChronoTick ±1σ uncertainty band (light blue with label)
    chronotick_lower = chronotick_offset - chronotick_uncertainty
    chronotick_upper = chronotick_offset + chronotick_uncertainty

    ax.fill_between(ntp_df['elapsed_hours'], chronotick_lower, chronotick_upper,
                     alpha=0.2, color='#56B4E9', label='±1σ', zorder=2)

    # System Clock offset (ORANGE/YELLOW markers - matching paper)
    ax.scatter(ntp_df['elapsed_hours'], system_offset,
               color='#D55E00', marker='x', s=20, alpha=0.6,
               label='System Clock', zorder=4)

    # ChronoTick offset (BLUE line/circles - matching paper)
    ax.scatter(ntp_df['elapsed_hours'], chronotick_offset,
               color='#0072B2', marker='o', s=15, alpha=0.7,
               label='ChronoTick', zorder=3)

    # Perfect sync reference line
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.3, zorder=1)

    # Styling - match paper
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Offset (ms)')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)

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

def generate_unsynchronized_exact_colors(csv_path, output_dir, output_name):
    """Generate unsynchronized figure with exact paper colors."""
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

    # ChronoTick ±1σ uncertainty band
    chronotick_lower = chronotick_offset - chronotick_uncertainty
    chronotick_upper = chronotick_offset + chronotick_uncertainty

    ax.fill_between(ntp_df['elapsed_hours'], chronotick_lower, chronotick_upper,
                     alpha=0.2, color='#56B4E9', label='±1σ', zorder=2)

    # System Clock offset with drift (ORANGE markers)
    ax.scatter(ntp_df['elapsed_hours'], system_offset,
               color='#D55E00', marker='x', s=20, alpha=0.6,
               label='System Clock', zorder=4)

    # Drift trend line (green/teal)
    drift_line = drift_coef[0] * hours + drift_coef[1]
    ax.plot(hours, drift_line, color='#009E73', linewidth=2, linestyle='--',
            alpha=0.7, label='System Drift', zorder=5)

    # ChronoTick offset (BLUE circles)
    ax.scatter(ntp_df['elapsed_hours'], chronotick_offset,
               color='#0072B2', marker='o', s=15, alpha=0.7,
               label='ChronoTick', zorder=3)

    # Perfect sync reference
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.3, zorder=1)

    # Styling
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Offset (ms)')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)

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
    """Generate figures with EXACT paper color scheme."""
    print("="*80)
    print("EXACT PAPER COLORS - Matching Existing Figure 3.1 Style")
    print("="*80)
    print("\nColor scheme:")
    print("  • #0072B2 (Blue): ChronoTick offset")
    print("  • #D55E00 (Orange): System Clock markers")
    print("  • #56B4E9 (Light blue): ±1σ uncertainty band")
    print("  • #009E73 (Green): Drift trend line")
    print("\nStyle:")
    print("  • No titles")
    print("  • Uncertainty labeled as '±1σ'")
    print("  • Orange 'x' markers for system clock")
    print("  • Blue 'o' circles for ChronoTick")
    print("  • Simple legend")

    output_dir = Path("results/figures_corrected/showcase")

    # Figure 3.1: Synchronized
    sync_csv = Path("results/experiment-3/homelab/data.csv")
    if sync_csv.exists():
        generate_synchronized_exact_colors(sync_csv, output_dir, "3.1_synchronized_FINAL")
    else:
        print(f"\n⚠️  Synchronized dataset not found: {sync_csv}")

    # Figure 3.2: Unsynchronized
    unsync_csv = Path("results/experiment-7/homelab/chronotick_client_validation_20251020_221631.csv")
    if unsync_csv.exists():
        generate_unsynchronized_exact_colors(unsync_csv, output_dir, "3.2_unsynchronized_FINAL")
    else:
        print(f"\n⚠️  Unsynchronized dataset not found: {unsync_csv}")

    print("\n" + "="*80)
    print("EXACT PAPER COLOR FIGURES COMPLETE")
    print("="*80)
    print(f"\nOutput directory: {output_dir.absolute()}")
    print("\nGenerated files:")
    print("  • 3.1_synchronized_FINAL.pdf")
    print("  • 3.2_unsynchronized_FINAL.pdf")
    print("\nThese match EXACTLY the color scheme of existing paper figures!")

if __name__ == "__main__":
    main()
