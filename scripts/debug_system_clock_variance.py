#!/usr/bin/env python3
"""Debug why system clock errors differ across experiments."""

import pandas as pd
from pathlib import Path

experiments = {
    "25min ADVANCED (original)": "~/.local/share/Trash/files/tsfm/results/ntp_correction_experiment/experiment_4_25min_advanced/summary_advanced_20251011_174307.csv",
    "25min LINEAR": "~/.local/share/Trash/files/tsfm/results/ntp_correction_experiment/experiment_2_25min_linear/summary_linear_20251011_164706.csv",
    "25min DRIFT_AWARE": "~/.local/share/Trash/files/tsfm/results/ntp_correction_experiment/experiment_3_25min_drift_aware/summary_drift_aware_20251011_171506.csv",
    "8hr test1 DUAL+ADVANCED (first 25min)": "tsfm/results/ntp_correction_experiment/8hour_tests/test1_dual_advanced/summary_advanced_20251012_115734.csv",
    "8hr test2 SINGLE (first 25min)": "tsfm/results/ntp_correction_experiment/8hour_tests/test2_short_advanced/summary_advanced_20251012_005722.csv",
    "8hr test3 DUAL+NONE (first 25min)": "tsfm/results/ntp_correction_experiment/8hour_tests/test3_dual_none/chronotick_stability_20251011_023912.csv",
}

for name, path in experiments.items():
    filepath = Path(path).expanduser()
    if not filepath.exists():
        print(f"\n{name}: FILE NOT FOUND")
        continue

    df = pd.read_csv(filepath)

    # Apply 25-min limit for 8hr tests
    if "8hr" in name:
        df = df[df['elapsed_seconds'] <= 1500].copy()

    # Handle different formats
    if 'has_ntp' in df.columns:
        df_ntp = df[df['has_ntp'] == True]
        if len(df_ntp) > 0:
            system_sum = df_ntp['system_error_ms'].abs().sum()
            chrono_sum = df_ntp['chronotick_error_ms'].abs().sum()
    elif 'system_error_vs_ntp_ms' in df.columns:
        df_ntp = df[df['system_error_vs_ntp_ms'].notna()]
        if len(df_ntp) > 0:
            system_sum = df_ntp['system_error_vs_ntp_ms'].abs().sum()
            chrono_sum = df_ntp['chronotick_error_vs_ntp_ms'].abs().sum()
    else:
        print(f"\n{name}: Unknown format")
        continue

    print(f"\n{name}:")
    print(f"  Total rows: {len(df)}, NTP measurements: {len(df_ntp)}")
    print(f"  Duration: {df['elapsed_seconds'].iloc[-1]:.0f}s ({df['elapsed_seconds'].iloc[-1]/60:.1f} min)")
    print(f"  System accumulated error: {system_sum:.1f} ms")
    print(f"  ChronoTick accumulated error: {chrono_sum:.1f} ms")

    # Show NTP measurement times
    if len(df_ntp) > 0:
        ntp_times = df_ntp['elapsed_seconds'].values
        print(f"  NTP measurement times (s): {ntp_times[:5]}... (showing first 5)")
