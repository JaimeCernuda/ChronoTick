#!/usr/bin/env python3
"""
Enhanced NTP Correction Log Analyzer

Extracts detailed before/after replacement information from debug logs to understand
NTP correction behavior with improved detail.

Usage:
    uv run python scripts/analyze_ntp_correction.py <log_file>
"""

import sys
import re
from collections import defaultdict
from pathlib import Path


def format_value(value, fmt='.3f', unit='ms'):
    """Safely format a numeric value or return 'N/A' if missing."""
    if isinstance(value, (int, float)):
        return f"{value:{fmt}}{unit}"
    return 'N/A'


def analyze_log(log_file_path):
    """Analyze NTP correction debug log and extract key metrics with replacement details."""

    print("=" * 80)
    print("NTP CORRECTION LOG ANALYSIS")
    print("=" * 80)
    print(f"Log file: {log_file_path}\n")

    with open(log_file_path, 'r') as f:
        log_content = f.read()

    lines = log_content.split('\n')

    # Extract NTP corrections with enhanced replacement tracking
    ntp_corrections = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # Look for NTP correction blocks
        if '[NTP_CORRECTION_' in line and '═══' in line:
            correction_data = {}
            method = None

            # Extract method from header
            if match := re.search(r'\[NTP_CORRECTION_([A-Z_]+)\]', line):
                method = match.group(1)
                correction_data['method'] = method

            # Parse the correction block
            j = i + 1
            while j < len(lines) and j < i + 200:  # Look ahead max 200 lines for detailed info
                subline = lines[j]

                # NTP offset
                if 'NTP offset (ground truth):' in subline:
                    if match := re.search(r'([+-]?[\d.]+)ms', subline):
                        correction_data['ntp_offset'] = float(match.group(1))

                # Prediction offset
                elif 'Prediction offset:' in subline:
                    if match := re.search(r'([+-]?[\d.]+)ms', subline):
                        correction_data['prediction_offset'] = float(match.group(1))

                # Prediction source (CRITICAL!)
                elif 'Prediction source:' in subline:
                    if match := re.search(r'source:\s*(\S+)', subline):
                        correction_data['prediction_source'] = match.group(1)

                # Error calculation
                elif 'Error =' in subline and 'NTP_truth' not in subline and 'Error = ' in subline:
                    if match := re.search(r'Error = ([+-]?[\d.]+)ms', subline):
                        correction_data['error'] = float(match.group(1))

                # Duration
                elif 'Duration:' in subline and 'Duration:' in subline:
                    if match := re.search(r'Duration:\s*([+-]?[\d.]+)s', subline):
                        correction_data['duration'] = float(match.group(1))

                # Number of predictions replaced (backtracking)
                elif 'REPLACED' in subline and 'predictions' in subline:
                    if match := re.search(r'REPLACED\s+(\d+)\s+predictions', subline):
                        correction_data['predictions_replaced'] = int(match.group(1))

                # Enhanced: Extract BEFORE/AFTER statistics from REPLACEMENT SUMMARY
                elif '[BACKTRACKING] BEFORE correction:' in subline:
                    # Next line should have mean
                    if j + 1 < len(lines) and 'Mean offset:' in lines[j+1]:
                        if match := re.search(r'Mean offset:\s*([+-]?[\d.]+)ms', lines[j+1]):
                            correction_data['mean_before'] = float(match.group(1))
                    if j + 2 < len(lines) and 'Range:' in lines[j+2]:
                        if match := re.search(r'Range:\s*([+-]?[\d.]+)ms to ([+-]?[\d.]+)ms', lines[j+2]):
                            correction_data['range_before_min'] = float(match.group(1))
                            correction_data['range_before_max'] = float(match.group(2))

                elif '[BACKTRACKING] AFTER correction' in subline:
                    if j + 1 < len(lines) and 'Mean offset:' in lines[j+1]:
                        if match := re.search(r'Mean offset:\s*([+-]?[\d.]+)ms', lines[j+1]):
                            correction_data['mean_after'] = float(match.group(1))
                    if j + 2 < len(lines) and 'Range:' in lines[j+2]:
                        if match := re.search(r'Range:\s*([+-]?[\d.]+)ms to ([+-]?[\d.]+)ms', lines[j+2]):
                            correction_data['range_after_min'] = float(match.group(1))
                            correction_data['range_after_max'] = float(match.group(2))

                elif '[BACKTRACKING] DELTA (after - before):' in subline:
                    if j + 1 < len(lines) and 'Mean delta:' in lines[j+1]:
                        if match := re.search(r'Mean delta:\s*([+-]?[\d.]+)ms', lines[j+1]):
                            correction_data['mean_delta'] = float(match.group(1))
                    if j + 2 < len(lines) and 'Min delta:' in lines[j+2]:
                        if match := re.search(r'Min delta:\s*([+-]?[\d.]+)ms', lines[j+2]):
                            correction_data['min_delta'] = float(match.group(1))
                    if j + 3 < len(lines) and 'Max delta:' in lines[j+3]:
                        if match := re.search(r'Max delta:\s*([+-]?[\d.]+)ms', lines[j+3]):
                            correction_data['max_delta'] = float(match.group(1))

                # Stop at next correction block
                if '[NTP_CORRECTION_' in subline and subline != line:
                    break

                j += 1

            if correction_data:
                ntp_corrections.append(correction_data)

            i = j
        else:
            i += 1

    # Extract dataset entries
    dataset_ntp_count = len(re.findall(r'\[DATASET_ADD_NTP\]', log_content))
    dataset_pred_count = len(re.findall(r'\[DATASET_ADD_PRED\]', log_content))

    # Extract first ML prediction time
    first_ml_pred_time = None
    for line in lines:
        if '[DATASET_ADD_PRED]' in line and 'Timestamp:' in line:
            if match := re.search(r'Timestamp:\s*([+-]?[\d.]+)', line):
                first_ml_pred_time = float(match.group(1))
                break

    # Print summary
    print(f"📊 DATASET STATISTICS")
    print(f"{'─' * 80}")
    print(f"  NTP measurements stored:    {dataset_ntp_count}")
    print(f"  ML predictions stored:      {dataset_pred_count}")
    if first_ml_pred_time:
        print(f"  First ML prediction time:   {int(first_ml_pred_time)}")
    print()

    # Analyze NTP corrections
    if ntp_corrections:
        print(f"🔧 NTP CORRECTIONS ({len(ntp_corrections)} total)")
        print(f"{'─' * 80}")

        small_errors = []
        large_errors = []
        ntp_to_ntp = 0
        ntp_to_ml = 0
        zero_replacements = 0
        nonzero_replacements = 0

        for idx, corr in enumerate(ntp_corrections, 1):
            error = corr.get('error', 0)
            pred_source = corr.get('prediction_source', 'unknown')
            pred_replaced = corr.get('predictions_replaced', 'N/A')

            # Classify error size
            if abs(error) < 1.0:
                small_errors.append(error)
            else:
                large_errors.append(error)

            # Track prediction source
            if pred_source == 'ntp_measurement':
                ntp_to_ntp += 1
            elif pred_source.startswith('prediction_'):
                ntp_to_ml += 1

            # Track replacements
            if isinstance(pred_replaced, int):
                if pred_replaced == 0:
                    zero_replacements += 1
                else:
                    nonzero_replacements += 1

            # Print individual correction with enhanced details
            status = "✓ SMALL" if abs(error) < 1.0 else "✗ LARGE"
            source_emoji = "🚨" if pred_source == 'ntp_measurement' else "✅"
            replace_emoji = "❌" if pred_replaced == 0 else "✅"

            print(f"\n  Correction #{idx}:")
            print(f"    Method: {corr.get('method', 'unknown')}")
            print(f"    NTP offset: {format_value(corr.get('ntp_offset'), '.3f', 'ms')}")
            print(f"    Prediction offset: {format_value(corr.get('prediction_offset'), '.3f', 'ms')}")
            print(f"    {source_emoji} Prediction source: {pred_source}")
            print(f"    Error: {error:.3f}ms {status}")
            print(f"    Duration: {format_value(corr.get('duration'), '.0f', 's')}")
            if isinstance(pred_replaced, int):
                print(f"    {replace_emoji} Predictions replaced: {pred_replaced}")

            # Enhanced: Show before/after replacement details
            if 'mean_before' in corr:
                mean_before = format_value(corr.get('mean_before'), '.3f', 'ms')
                range_before = f"[{format_value(corr.get('range_before_min'), '.3f', '')}, {format_value(corr.get('range_before_max'), '.3f', '')}]ms"
                print(f"    📈 BEFORE correction: mean={mean_before}, range={range_before}")
            if 'mean_after' in corr:
                mean_after = format_value(corr.get('mean_after'), '.3f', 'ms')
                range_after = f"[{format_value(corr.get('range_after_min'), '.3f', '')}, {format_value(corr.get('range_after_max'), '.3f', '')}]ms"
                print(f"    📉 AFTER correction:  mean={mean_after}, range={range_after}")
            if 'mean_delta' in corr:
                mean_delta = format_value(corr.get('mean_delta'), '+.3f', 'ms')
                range_delta = f"[{format_value(corr.get('min_delta'), '+.3f', '')}, {format_value(corr.get('max_delta'), '+.3f', '')}]ms"
                print(f"    🔄 DELTA: mean={mean_delta}, range={range_delta}")

        print(f"\n{'─' * 80}")
        print(f"📈 CORRECTION SUMMARY")
        print(f"{'─' * 80}")
        print(f"  Total corrections:          {len(ntp_corrections)}")
        print(f"  Small errors (<1ms):        {len(small_errors)} ({len(small_errors)/len(ntp_corrections)*100:.1f}%)")
        print(f"  Large errors (≥1ms):        {len(large_errors)} ({len(large_errors)/len(ntp_corrections)*100:.1f}%)")
        print()
        print(f"  🚨 NTP-to-NTP comparisons:  {ntp_to_ntp} (BUG - should be 0!)")
        print(f"  ✅ NTP-to-ML comparisons:   {ntp_to_ml} (CORRECT)")
        print()
        print(f"  ❌ Zero replacements:       {zero_replacements} (no predictions in interval)")
        print(f"  ✅ Nonzero replacements:    {nonzero_replacements} (corrections applied)")
        print()

        if large_errors:
            print(f"  Large error stats:")
            print(f"    Mean: {sum(large_errors)/len(large_errors):.3f}ms")
            print(f"    Min:  {min(large_errors):.3f}ms")
            print(f"    Max:  {max(large_errors):.3f}ms")

        # Diagnosis
        print(f"\n{'─' * 80}")
        print(f"🩺 DIAGNOSIS")
        print(f"{'─' * 80}")

        if ntp_to_ntp > 0:
            print(f"  ⚠️  WARNING: {ntp_to_ntp} NTP-to-NTP comparisons detected!")
            print(f"      This is the bug - comparing NTP to NTP instead of NTP to ML.")
            print(f"      Expected behavior: All comparisons should be NTP-to-ML.")
        else:
            print(f"  ✅ GOOD: All comparisons are NTP-to-ML (bug fixed!)")

        if zero_replacements == len(ntp_corrections):
            print(f"  ⚠️  WARNING: ALL corrections replaced 0 predictions!")
            print(f"      Likely cause: ML predictions don't exist in correction intervals.")
            print(f"      This means corrections are not actually fixing anything.")
        elif zero_replacements > len(ntp_corrections) / 2:
            print(f"  ⚠️  WARNING: Most corrections ({zero_replacements}/{len(ntp_corrections)}) replaced 0 predictions.")
            print(f"      Check timing of ML predictions vs NTP corrections.")
        else:
            print(f"  ✅ GOOD: Most corrections ({nonzero_replacements}/{len(ntp_corrections)}) are replacing predictions.")

        if len(small_errors) == len(ntp_corrections):
            print(f"  ⚠️  WARNING: ALL errors are small (<1ms)!")
            print(f"      This suggests NTP-to-NTP comparison (bug) or perfect ML predictions (unlikely).")
        elif len(large_errors) > len(ntp_corrections) / 2:
            print(f"  ✅ GOOD: Most errors ({len(large_errors)}/{len(ntp_corrections)}) are large (≥1ms).")
            print(f"      This suggests NTP-to-ML comparison is working correctly.")

    else:
        print("❌ No NTP corrections found in log!")
        print("   This is normal if test duration was too short or warmup didn't complete.")

    print(f"\n{'=' * 80}")
    print("ANALYSIS COMPLETE")
    print(f"{'=' * 80}\n")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: uv run python scripts/analyze_ntp_correction.py <log_file>")
        sys.exit(1)

    log_file = Path(sys.argv[1])
    if not log_file.exists():
        print(f"Error: Log file not found: {log_file}")
        sys.exit(1)

    analyze_log(log_file)
