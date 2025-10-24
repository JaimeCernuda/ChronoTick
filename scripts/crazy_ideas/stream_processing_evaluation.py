#!/usr/bin/env python3
"""
Stream Processing Evaluation: Uncertainty-Aware Windowing

MOTIVATION: Apache Flink / Kafka Streams problem
- Distributed stream processors assign events to time windows (e.g., 1-second tumbling)
- Clock skew causes different nodes to assign same event to different windows
- Traditional solution: Watermarks (adds latency) + centralized coordination

CHRONOTICK SOLUTION: Fuzzy clock with bounded uncertainty
- Each event timestamped with ChronoTick: (timestamp, uncertainty)
- Identify "unambiguous" events (clearly belong to one window)
- Identify "ambiguous" events (could span window boundaries due to uncertainty)
- Enable smart buffering/replication only for ambiguous events

EVALUATION:
1. Window Assignment Agreement: Do both nodes agree on which window an event belongs to?
2. Ambiguity Detection: What % of events are ambiguous (span window boundaries)?
3. Uncertainty-Aware Windowing: Can we correctly identify ambiguous events?
4. Latency Reduction: Compare with traditional watermark-based approach
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

def load_data(node1_csv, node2_csv):
    """Load data from both nodes."""
    df1 = pd.read_csv(node1_csv)
    df2 = pd.read_csv(node2_csv)

    df1_all = df1.copy()
    df2_all = df2.copy()

    df1_all['timestamp'] = pd.to_datetime(df1_all['datetime'])
    df2_all['timestamp'] = pd.to_datetime(df2_all['datetime'])

    df1_ntp = df1[df1['has_ntp'] == True].copy()
    df2_ntp = df2[df2['has_ntp'] == True].copy()
    df1_ntp['timestamp'] = pd.to_datetime(df1_ntp['datetime'])
    df2_ntp['timestamp'] = pd.to_datetime(df2_ntp['datetime'])

    start1 = df1_ntp['timestamp'].iloc[0]
    start2 = df2_ntp['timestamp'].iloc[0]
    start_offset = (start2 - start1).total_seconds()

    df1_all['elapsed_seconds'] = (df1_all['timestamp'] - start1).dt.total_seconds()
    df2_all['elapsed_seconds'] = (df2_all['timestamp'] - start2).dt.total_seconds()
    df1_ntp['elapsed_seconds'] = (df1_ntp['timestamp'] - start1).dt.total_seconds()
    df2_ntp['elapsed_seconds'] = (df2_ntp['timestamp'] - start2).dt.total_seconds()

    return {
        'node1_all': df1_all,
        'node2_all': df2_all,
        'node1_ntp': df1_ntp,
        'node2_ntp': df2_ntp,
        'start_offset': start_offset,
        'start1': start1,
        'start2': start2
    }

def evaluate_window_assignment(data, window_size_ms=1000, output_dir=None, exp_name=''):
    """
    Evaluate 1: Window Assignment Agreement

    Scenario: Tumbling windows of `window_size_ms` milliseconds
    Question: Do both nodes assign events to the same window?

    Without ChronoTick: Clock skew causes disagreements
    With ChronoTick: Use ground truth + prediction to measure agreement
    """
    print(f"\n{'='*80}")
    print(f"EVALUATION 1: WINDOW ASSIGNMENT AGREEMENT ({window_size_ms}ms windows)")
    print('='*80)

    start_offset = data['start_offset']

    # Use Node 1 NTP as ground truth for "true" event timestamps
    node1_ntp = data['node1_ntp']
    node2_all = data['node2_all']

    results = []

    for idx1, row1 in node1_ntp.iterrows():
        elapsed1 = row1['elapsed_seconds']
        elapsed2_target = elapsed1 - start_offset

        if elapsed2_target < 0:
            continue

        # Ground truth: Node 1 NTP offset
        ntp1_ms = row1['ntp_offset_ms']

        # Find corresponding Node 2 sample
        idx2 = (node2_all['elapsed_seconds'] - elapsed2_target).abs().idxmin()
        if abs(node2_all.loc[idx2, 'elapsed_seconds'] - elapsed2_target) > 5:
            continue

        # Node 2's ChronoTick prediction offset
        pred2_ms = node2_all.loc[idx2, 'chronotick_offset_ms']

        # KEY INSIGHT: For windowing, what matters is position WITHIN a window
        # Not absolute window ID (which depends on arbitrary start time)
        # Position within window determines if event is near boundary
        position1 = ntp1_ms % window_size_ms
        position2 = pred2_ms % window_size_ms

        # Handle negative offsets
        if position1 < 0:
            position1 += window_size_ms
        if position2 < 0:
            position2 += window_size_ms

        # Check if difference causes window boundary crossing
        # Events in same window if difference < window_size_ms/2
        # Otherwise, they're in different windows
        diff = abs(position1 - position2)

        # Account for wrap-around (e.g., position1=10, position2=990 → diff=20, not 980)
        if diff > window_size_ms / 2:
            diff = window_size_ms - diff

        # Agreement if difference is small enough not to cross boundary
        # Conservative: require difference < 10ms to be "same window"
        agrees = (diff < 10)

        results.append({
            'elapsed_hours': elapsed1 / 3600,
            'position1': position1,
            'position2': position2,
            'diff_ms': diff,
            'agrees': agrees,
            'ntp1_ms': ntp1_ms,
            'pred2_ms': pred2_ms
        })

    df = pd.DataFrame(results)

    if len(df) == 0:
        print("⚠️  No valid samples")
        return None

    agree_rate = (df['agrees'].sum() / len(df) * 100)
    mean_diff = df['diff_ms'].mean()
    median_diff = df['diff_ms'].median()
    max_diff = df['diff_ms'].max()

    print(f"\nResults:")
    print(f"  Agreement (diff < 10ms): {df['agrees'].sum()}/{len(df)} = {agree_rate:.1f}%")
    print(f"  Mean offset difference: {mean_diff:.2f}ms")
    print(f"  Median offset difference: {median_diff:.2f}ms")
    print(f"  Max offset difference: {max_diff:.2f}ms")

    # Visualization
    if output_dir:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Panel 1: Offset difference over time
        agree_df = df[df['agrees']]
        disagree_df = df[~df['agrees']]

        ax1.scatter(agree_df['elapsed_hours'], agree_df['diff_ms'],
                   color='#009E73', s=30, alpha=0.6, label=f'Agreement ({len(agree_df)})')
        ax1.scatter(disagree_df['elapsed_hours'], disagree_df['diff_ms'],
                   color='#D55E00', s=30, alpha=0.6, label=f'Disagreement ({len(disagree_df)})')
        ax1.axhline(10, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='10ms threshold')

        ax1.set_xlabel('Time (hours)', fontweight='bold')
        ax1.set_ylabel('Offset Difference (ms)', fontweight='bold')
        ax1.set_title(f'Window Assignment: {agree_rate:.1f}% agreement ({len(df)} events, {window_size_ms}ms windows)',
                     fontweight='bold')
        ax1.grid(alpha=0.3)
        ax1.legend()
        ax1.set_yscale('log')

        # Panel 2: Offset difference histogram
        ax2.hist(df['diff_ms'], bins=50, color='#0072B2', alpha=0.7, edgecolor='black')
        ax2.axvline(10, color='red', linestyle='--', linewidth=2, alpha=0.7, label='10ms threshold')
        ax2.set_xlabel('Offset Difference (ms)', fontweight='bold')
        ax2.set_ylabel('Count', fontweight='bold')
        ax2.set_title(f'Distribution: mean={mean_diff:.2f}ms, median={median_diff:.2f}ms', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        ax2.legend()

        # Add percentage annotations
        ax2.text(0.7, 0.9, f'{agree_rate:.1f}% within 10ms\n({df["agrees"].sum()}/{len(df)} events)',
                transform=ax2.transAxes, fontweight='bold', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

        plt.tight_layout()
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / f'{exp_name}_eval1_window_assignment.pdf', bbox_inches='tight')
        plt.savefig(output_dir / f'{exp_name}_eval1_window_assignment.png', dpi=300, bbox_inches='tight')
        plt.close()

    return {
        'name': 'Window Assignment Agreement',
        'agree_rate': agree_rate,
        'mean_diff_ms': mean_diff,
        'median_diff_ms': median_diff,
        'max_diff_ms': max_diff,
        'samples': len(df)
    }

def evaluate_ambiguity_detection(data, window_size_ms=1000, sigma_level=3, output_dir=None, exp_name=''):
    """
    Evaluation 2: Ambiguity Detection

    Question: Can ChronoTick's uncertainty identify "ambiguous" events that span window boundaries?

    Ambiguous event: timestamp ± uncertainty crosses window boundary
    Example: window boundary at 1000ms, event at 998ms ± 5ms → spans [993, 1003] → ambiguous!

    Key insight: Traditional systems can't detect ambiguity, ChronoTick can!
    """
    print(f"\n{'='*80}")
    print(f"EVALUATION 2: AMBIGUITY DETECTION ({window_size_ms}ms windows, ±{sigma_level}σ)")
    print('='*80)

    start_offset = data['start_offset']
    node1_ntp = data['node1_ntp']
    node2_all = data['node2_all']

    results = []

    for idx1, row1 in node1_ntp.iterrows():
        elapsed1 = row1['elapsed_seconds']
        elapsed2_target = elapsed1 - start_offset

        if elapsed2_target < 0:
            continue

        # Ground truth
        ntp1_ms = row1['ntp_offset_ms']
        wallclock_ms = elapsed1 * 1000 + ntp1_ms
        true_window = int(wallclock_ms / window_size_ms)

        # Find corresponding Node 2 sample
        idx2 = (node2_all['elapsed_seconds'] - elapsed2_target).abs().idxmin()
        if abs(node2_all.loc[idx2, 'elapsed_seconds'] - elapsed2_target) > 5:
            continue

        # Node 2's ChronoTick prediction + uncertainty
        pred2_ms = node2_all.loc[idx2, 'chronotick_offset_ms']
        unc2_ms = node2_all.loc[idx2, 'chronotick_uncertainty_ms']

        wallclock2_ms = elapsed2_target * 1000 + pred2_ms

        # Calculate range: pred ± sigma_level * unc
        lower_bound = wallclock2_ms - sigma_level * unc2_ms
        upper_bound = wallclock2_ms + sigma_level * unc2_ms

        # Determine windows for lower and upper bounds
        lower_window = int(lower_bound / window_size_ms)
        upper_window = int(upper_bound / window_size_ms)

        # Is it ambiguous? (spans multiple windows)
        is_ambiguous = (lower_window != upper_window)

        # Prediction based on midpoint
        pred_window = int(wallclock2_ms / window_size_ms)

        # Did ambiguity flag help?
        # If ambiguous and wrong, we correctly identified uncertainty
        # If not ambiguous and right, we correctly identified certainty
        correct_assignment = (pred_window == true_window)

        # Categorize
        if is_ambiguous and not correct_assignment:
            category = "ambiguous_wrong"  # Good: flagged as uncertain and was wrong
        elif is_ambiguous and correct_assignment:
            category = "ambiguous_right"  # Good: flagged as uncertain but happened to be right
        elif not is_ambiguous and correct_assignment:
            category = "unambiguous_right"  # Good: confident and correct
        else:
            category = "unambiguous_wrong"  # Bad: confident but wrong

        results.append({
            'elapsed_hours': elapsed1 / 3600,
            'wallclock_ms': wallclock_ms,
            'true_window': true_window,
            'pred_window': pred_window,
            'lower_window': lower_window,
            'upper_window': upper_window,
            'is_ambiguous': is_ambiguous,
            'correct_assignment': correct_assignment,
            'category': category,
            'uncertainty_ms': unc2_ms
        })

    df = pd.DataFrame(results)

    if len(df) == 0:
        print("⚠️  No valid samples")
        return None

    ambiguous_count = df['is_ambiguous'].sum()
    ambiguous_rate = ambiguous_count / len(df) * 100

    # Calculate categories
    unambiguous_right = len(df[df['category'] == 'unambiguous_right'])
    unambiguous_wrong = len(df[df['category'] == 'unambiguous_wrong'])
    ambiguous_right = len(df[df['category'] == 'ambiguous_right'])
    ambiguous_wrong = len(df[df['category'] == 'ambiguous_wrong'])

    # Key metric: How often does ambiguity flag prevent errors?
    # If ambiguous, system can buffer/replicate → avoids wrong assignment
    prevented_errors = ambiguous_wrong
    unavoidable_errors = unambiguous_wrong

    print(f"\nResults:")
    print(f"  Ambiguous events: {ambiguous_count}/{len(df)} = {ambiguous_rate:.1f}%")
    print(f"  Unambiguous + correct: {unambiguous_right} ({unambiguous_right/len(df)*100:.1f}%)")
    print(f"  Unambiguous + wrong: {unambiguous_wrong} ({unambiguous_wrong/len(df)*100:.1f}%)")
    print(f"  Ambiguous + correct: {ambiguous_right} ({ambiguous_right/len(df)*100:.1f}%)")
    print(f"  Ambiguous + wrong: {ambiguous_wrong} ({ambiguous_wrong/len(df)*100:.1f}%)")
    print(f"\n  💡 KEY INSIGHT:")
    print(f"     Errors preventable by ambiguity detection: {prevented_errors} ({prevented_errors/len(df)*100:.1f}%)")
    print(f"     Unavoidable errors (confident but wrong): {unavoidable_errors} ({unavoidable_errors/len(df)*100:.1f}%)")

    # Visualization
    if output_dir:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Panel 1: Ambiguity over time
        ambig_df = df[df['is_ambiguous']]
        unambig_df = df[~df['is_ambiguous']]

        ax1.scatter(unambig_df['elapsed_hours'], unambig_df['uncertainty_ms'],
                   color='#0072B2', s=30, alpha=0.5, label=f'Unambiguous ({len(unambig_df)})')
        ax1.scatter(ambig_df['elapsed_hours'], ambig_df['uncertainty_ms'],
                   color='#D55E00', s=30, alpha=0.5, label=f'Ambiguous ({len(ambig_df)})')

        ax1.axhline(window_size_ms / (2 * sigma_level), color='gray', linestyle='--',
                   linewidth=1, alpha=0.6, label=f'Ambiguity threshold (~{window_size_ms/(2*sigma_level):.1f}ms)')

        ax1.set_xlabel('Time (hours)', fontweight='bold')
        ax1.set_ylabel('Uncertainty (ms)', fontweight='bold')
        ax1.set_title(f'Ambiguous Event Detection: {ambiguous_rate:.1f}% flagged', fontweight='bold')
        ax1.grid(alpha=0.3)
        ax1.legend()
        ax1.set_yscale('log')

        # Panel 2: Category breakdown
        categories = ['Unambiguous\n+ Correct', 'Unambiguous\n+ Wrong',
                     'Ambiguous\n+ Correct', 'Ambiguous\n+ Wrong']
        counts = [unambiguous_right, unambiguous_wrong, ambiguous_right, ambiguous_wrong]
        colors = ['#009E73', '#CC79A7', '#E69F00', '#D55E00']

        bars = ax2.bar(categories, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('Count', fontweight='bold')
        ax2.set_title('Event Categorization by Ambiguity & Correctness', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        # Add percentage labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}\n({count/len(df)*100:.1f}%)',
                    ha='center', va='bottom', fontweight='bold')

        # Add insight box
        ax2.text(0.5, 0.95, f'💡 {prevented_errors} errors preventable\nby ambiguity-aware buffering',
                transform=ax2.transAxes, ha='center', va='top', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

        plt.tight_layout()
        plt.savefig(output_dir / f'{exp_name}_eval2_ambiguity_detection.pdf', bbox_inches='tight')
        plt.savefig(output_dir / f'{exp_name}_eval2_ambiguity_detection.png', dpi=300, bbox_inches='tight')
        plt.close()

    return {
        'name': 'Ambiguity Detection',
        'ambiguous_rate': ambiguous_rate,
        'unambiguous_right': unambiguous_right / len(df) * 100,
        'unambiguous_wrong': unambiguous_wrong / len(df) * 100,
        'ambiguous_right': ambiguous_right / len(df) * 100,
        'ambiguous_wrong': ambiguous_wrong / len(df) * 100,
        'prevented_errors_rate': prevented_errors / len(df) * 100,
        'samples': len(df)
    }

def evaluate_stream_join(data, join_window_ms=10, sigma_level=3, output_dir=None, exp_name=''):
    """
    Evaluation 3: Uncertainty-Aware Stream Join

    Scenario: Join events from Node 1 and Node 2 if they occur within join_window_ms
    Traditional: Clock skew causes missed joins or false joins
    ChronoTick: Use uncertainty ranges to identify "potential matches" vs "certain matches"

    Question: How often do uncertainty ranges help identify correct join candidates?
    """
    print(f"\n{'='*80}")
    print(f"EVALUATION 3: UNCERTAINTY-AWARE STREAM JOIN ({join_window_ms}ms window, ±{sigma_level}σ)")
    print('='*80)

    start_offset = data['start_offset']
    node1_ntp = data['node1_ntp']
    node2_ntp = data['node2_ntp']
    node1_all = data['node1_all']
    node2_all = data['node2_all']

    results = []

    # Sample every 10th NTP event from Node 1
    for idx1 in range(0, len(node1_ntp), 10):
        row1 = node1_ntp.iloc[idx1]
        elapsed1 = row1['elapsed_seconds']

        # Ground truth: Node 1 NTP timestamp
        ntp1_ms = row1['ntp_offset_ms']
        wallclock1_ms = elapsed1 * 1000 + ntp1_ms

        # Find all Node 2 NTP events within join_window_ms (ground truth joins)
        for idx2, row2 in node2_ntp.iterrows():
            elapsed2 = row2['elapsed_seconds']
            ntp2_ms = row2['ntp_offset_ms']

            # Convert Node 2 to Node 1's timeline
            elapsed1_equivalent = elapsed2 + start_offset
            wallclock2_ms = elapsed1_equivalent * 1000 + ntp2_ms

            # Ground truth: Should these join?
            time_diff_ms = abs(wallclock1_ms - wallclock2_ms)
            true_join = (time_diff_ms <= join_window_ms)

            # Now test ChronoTick-based join
            # Node 1: Use ChronoTick prediction + uncertainty
            pred1_ms = row1['chronotick_offset_ms']
            unc1_ms = row1['chronotick_uncertainty_ms']
            range1_lower = elapsed1 * 1000 + pred1_ms - sigma_level * unc1_ms
            range1_upper = elapsed1 * 1000 + pred1_ms + sigma_level * unc1_ms

            # Node 2: Use ChronoTick prediction + uncertainty
            pred2_ms = row2['chronotick_offset_ms']
            unc2_ms = row2['chronotick_uncertainty_ms']
            range2_lower = elapsed1_equivalent * 1000 + pred2_ms - sigma_level * unc2_ms
            range2_upper = elapsed1_equivalent * 1000 + pred2_ms + sigma_level * unc2_ms

            # ChronoTick join: Do ranges overlap within join_window?
            # Ranges overlap if: range1_upper >= range2_lower AND range2_upper >= range1_lower
            ranges_overlap = (range1_upper >= range2_lower) and (range2_upper >= range1_lower)

            # Additionally check if centers are within join window
            center1 = elapsed1 * 1000 + pred1_ms
            center2 = elapsed1_equivalent * 1000 + pred2_ms
            center_diff = abs(center1 - center2)

            # Conservative join: ranges overlap
            # Aggressive join: centers within window
            conservative_join = ranges_overlap
            aggressive_join = (center_diff <= join_window_ms)

            # Categorize
            if true_join and conservative_join:
                category = "true_positive_conservative"
            elif true_join and not conservative_join:
                category = "false_negative_conservative"
            elif not true_join and conservative_join:
                category = "false_positive_conservative"
            else:
                category = "true_negative_conservative"

            results.append({
                'time_diff_ms': time_diff_ms,
                'true_join': true_join,
                'conservative_join': conservative_join,
                'aggressive_join': aggressive_join,
                'category': category,
                'range_overlap_size': min(range1_upper, range2_upper) - max(range1_lower, range2_lower)
            })

    df = pd.DataFrame(results)

    if len(df) == 0:
        print("⚠️  No valid samples")
        return None

    # Calculate metrics
    tp_cons = len(df[df['category'] == 'true_positive_conservative'])
    fn_cons = len(df[df['category'] == 'false_negative_conservative'])
    fp_cons = len(df[df['category'] == 'false_positive_conservative'])
    tn_cons = len(df[df['category'] == 'true_negative_conservative'])

    precision_cons = tp_cons / (tp_cons + fp_cons) if (tp_cons + fp_cons) > 0 else 0
    recall_cons = tp_cons / (tp_cons + fn_cons) if (tp_cons + fn_cons) > 0 else 0
    f1_cons = 2 * precision_cons * recall_cons / (precision_cons + recall_cons) if (precision_cons + recall_cons) > 0 else 0

    print(f"\nResults (Conservative Join - ranges overlap):")
    print(f"  True Positives: {tp_cons} ({tp_cons/len(df)*100:.1f}%)")
    print(f"  False Positives: {fp_cons} ({fp_cons/len(df)*100:.1f}%)")
    print(f"  False Negatives: {fn_cons} ({fn_cons/len(df)*100:.1f}%)")
    print(f"  True Negatives: {tn_cons} ({tn_cons/len(df)*100:.1f}%)")
    print(f"  Precision: {precision_cons*100:.1f}%")
    print(f"  Recall: {recall_cons*100:.1f}%")
    print(f"  F1-Score: {f1_cons*100:.1f}%")

    # Visualization
    if output_dir:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Panel 1: Confusion matrix
        confusion = [[tp_cons, fp_cons], [fn_cons, tn_cons]]
        im = ax1.imshow(confusion, cmap='Blues', alpha=0.7)
        ax1.set_xticks([0, 1])
        ax1.set_yticks([0, 1])
        ax1.set_xticklabels(['Predicted\nJoin', 'Predicted\nNo Join'])
        ax1.set_yticklabels(['Actual\nJoin', 'Actual\nNo Join'])
        ax1.set_title('Uncertainty-Aware Join: Confusion Matrix', fontweight='bold', fontsize=12)

        # Add text annotations
        for i in range(2):
            for j in range(2):
                text = ax1.text(j, i, f'{confusion[i][j]}\n({confusion[i][j]/len(df)*100:.1f}%)',
                               ha="center", va="center", color="black", fontweight='bold')

        # Panel 2: Join candidates by time difference
        join_df = df[df['true_join']]
        no_join_df = df[~df['true_join']]

        ax2.scatter(join_df['time_diff_ms'], [1]*len(join_df),
                   color='#009E73', s=50, alpha=0.6, label=f'True Joins ({len(join_df)})')
        ax2.scatter(no_join_df['time_diff_ms'], [0]*len(no_join_df),
                   color='#D55E00', s=50, alpha=0.3, label=f'Non-Joins ({len(no_join_df)})')

        ax2.axvline(join_window_ms, color='blue', linestyle='--', linewidth=2,
                   label=f'Join window ({join_window_ms}ms)')
        ax2.set_xlabel('Time Difference (ms)', fontweight='bold')
        ax2.set_ylabel('Join Status', fontweight='bold')
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(['No Join', 'Join'])
        ax2.set_title(f'Stream Join Candidates (F1={f1_cons*100:.1f}%)', fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f'{exp_name}_eval3_stream_join.pdf', bbox_inches='tight')
        plt.savefig(output_dir / f'{exp_name}_eval3_stream_join.png', dpi=300, bbox_inches='tight')
        plt.close()

    return {
        'name': 'Uncertainty-Aware Stream Join',
        'precision': precision_cons * 100,
        'recall': recall_cons * 100,
        'f1_score': f1_cons * 100,
        'true_positives_rate': tp_cons / len(df) * 100,
        'false_positives_rate': fp_cons / len(df) * 100,
        'samples': len(df)
    }

def run_stream_processing_eval(exp_name, node1_csv, node2_csv, output_base):
    """Run all stream processing evaluations."""
    print(f"\n\n{'#'*80}")
    print(f"# STREAM PROCESSING EVALUATION: {exp_name}")
    print(f"{'#'*80}")

    if not node1_csv.exists() or not node2_csv.exists():
        print(f"⚠️  Data not found for {exp_name}")
        return None

    data = load_data(node1_csv, node2_csv)
    output_dir = output_base / exp_name

    print(f"\nLoaded:")
    print(f"  Node 1: {len(data['node1_ntp'])} NTP, {len(data['node1_all'])} total")
    print(f"  Node 2: {len(data['node2_ntp'])} NTP, {len(data['node2_all'])} total")
    print(f"  Start offset: {data['start_offset']:.1f}s")

    results = {}

    # Eval 1: Window assignment (1-second windows)
    results['eval1'] = evaluate_window_assignment(data, window_size_ms=1000,
                                                   output_dir=output_dir, exp_name=exp_name)

    # Eval 2: Ambiguity detection (1-second windows, ±3σ)
    results['eval2'] = evaluate_ambiguity_detection(data, window_size_ms=1000, sigma_level=3,
                                                     output_dir=output_dir, exp_name=exp_name)

    # Eval 3: Stream join (10ms join window, ±3σ)
    results['eval3'] = evaluate_stream_join(data, join_window_ms=10, sigma_level=3,
                                            output_dir=output_dir, exp_name=exp_name)

    return results

def main():
    """Run stream processing evaluation."""
    output_base = Path("results/figures/stream_processing")

    experiments = {
        'experiment-5': {
            'node1': Path("results/experiment-5/ares-comp-11/data.csv"),
            'node2': Path("results/experiment-5/ares-comp-12/data.csv")
        }
    }

    all_results = {}

    for exp_name, paths in experiments.items():
        results = run_stream_processing_eval(exp_name, paths['node1'], paths['node2'], output_base)
        if results:
            all_results[exp_name] = results

    # Save summary
    summary_file = output_base / 'summary_results.json'
    output_base.mkdir(parents=True, exist_ok=True)
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\n{'='*80}")
    print("STREAM PROCESSING EVALUATION COMPLETE")
    print('='*80)
    print(f"\n✓ Results saved to: {output_base}/")
    print(f"✓ Summary JSON: {summary_file}")

if __name__ == "__main__":
    main()
