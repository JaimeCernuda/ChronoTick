#!/usr/bin/env python3
"""
Clean evaluation points from summary CSVs to match improved logic:
1. Skip evaluations before 120s elapsed
2. Keep only one evaluation per unique NTP timestamp
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

def clean_evaluations(csv_path):
    """Remove duplicate and early evaluations from summary CSV"""
    df = pd.read_csv(csv_path)

    original_count = (df['has_ntp'] == True).sum()

    # Track which NTP timestamps we've already used for evaluation
    seen_ntp_timestamps = set()

    for idx, row in df.iterrows():
        if row['has_ntp']:
            # Skip evaluations before 120s
            if row['elapsed_seconds'] < 120:
                # Remove NTP evaluation data but keep the row
                df.at[idx, 'has_ntp'] = False
                df.at[idx, 'ntp_ground_truth_offset_ms'] = np.nan
                df.at[idx, 'ntp_uncertainty_ms'] = np.nan
                df.at[idx, 'chronotick_error_ms'] = np.nan
                df.at[idx, 'system_error_ms'] = np.nan
                continue

            # Get NTP timestamp (round to nearest second to group duplicates)
            ntp_timestamp = round(row['timestamp'] - row['elapsed_seconds'] + row['elapsed_seconds'])

            if ntp_timestamp in seen_ntp_timestamps:
                # Duplicate evaluation - remove it
                df.at[idx, 'has_ntp'] = False
                df.at[idx, 'ntp_ground_truth_offset_ms'] = np.nan
                df.at[idx, 'ntp_uncertainty_ms'] = np.nan
                df.at[idx, 'chronotick_error_ms'] = np.nan
                df.at[idx, 'system_error_ms'] = np.nan
            else:
                # First time seeing this NTP - keep it
                seen_ntp_timestamps.add(ntp_timestamp)

    # Save cleaned CSV
    output_path = str(csv_path).replace('.csv', '_cleaned.csv')
    df.to_csv(output_path, index=False)

    cleaned_count = (df['has_ntp'] == True).sum()

    print(f"Cleaned {csv_path.name}:")
    print(f"  Evaluations: {original_count} → {cleaned_count}")
    print(f"  Output: {Path(output_path).name}")

    return output_path

if __name__ == "__main__":
    results_dir = Path('results/ntp_correction_experiment/visualization_data')
    
    # Clean none and linear summaries
    none_summary = results_dir / 'summary_none_20251011_161905.csv'
    linear_summary = results_dir / 'summary_linear_20251011_164706.csv'
    
    print("Cleaning evaluation points to match improved logic...\n")
    
    clean_evaluations(none_summary)
    print()
    clean_evaluations(linear_summary)
    
    print("\n✓ Post-processing complete")
