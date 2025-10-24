#!/usr/bin/env python3
"""
BOUNDED CLOCK EVALUATION - CORRECT VERSION

Within-node comparison of OFFSETS:
1. System clock baseline: offset = 0 (assumes no correction needed)
2. ChronoTick: offset = chronotick_offset_ms ± chronotick_uncertainty_ms
3. Ground truth: offset = ntp_offset_ms

The Google Spanner TrueTime narrative:
- System clock: "No correction needed" (single point, WRONG)
- ChronoTick: "Correction = X ± ε ms" (bounded, quantified uncertainty)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

def evaluate_within_node(csv_path, window_sizes=[100, 500, 1000, 5000]):
    """Evaluate within a single node."""
    
    df = pd.read_csv(csv_path)
    ntp_df = df[df['has_ntp'] == True].copy()
    
    print(f"Loaded {len(ntp_df)} NTP samples")
    
    all_results = {ws: [] for ws in window_sizes}
    
    for idx, row in ntp_df.iterrows():
        # Ground truth offset
        truth_offset_ms = row['ntp_offset_ms']
        
        # System clock assumption: offset = 0
        system_offset_ms = 0.0
        
        # ChronoTick prediction
        chronotick_offset_ms = row['chronotick_offset_ms']
        chronotick_unc_ms = row['chronotick_uncertainty_ms']
        
        # Bounded clock: ±3σ interval
        chronotick_lower_ms = chronotick_offset_ms - 3 * chronotick_unc_ms
        chronotick_upper_ms = chronotick_offset_ms + 3 * chronotick_unc_ms
        
        # Test each window size
        for window_ms in window_sizes:
            # Window positions
            pos_truth = truth_offset_ms % window_ms
            pos_system = system_offset_ms % window_ms  # Always 0
            pos_chronotick = chronotick_offset_ms % window_ms
            
            if pos_truth < 0:
                pos_truth += window_ms
            if pos_chronotick < 0:
                pos_chronotick += window_ms
                
            # Calculate differences
            def window_diff(p1, p2, ws):
                diff = abs(p1 - p2)
                if diff > ws / 2:
                    diff = ws - diff
                return diff
            
            diff_system = window_diff(pos_truth, pos_system, window_ms)
            diff_chronotick = window_diff(pos_truth, pos_chronotick, window_ms)
            
            # Agreement threshold
            threshold = min(10, window_ms * 0.01)
            
            agrees_system = (diff_system < threshold)
            agrees_chronotick = (diff_chronotick < threshold)
            
            # Bounded clock metrics
            # Truth within ±3σ bounds?
            truth_in_bounds = (chronotick_lower_ms <= truth_offset_ms <= chronotick_upper_ms)
            
            # Is this event ambiguous (near window boundary)?
            distance_from_boundary = min(pos_chronotick, window_ms - pos_chronotick)
            is_ambiguous = (3 * chronotick_unc_ms > distance_from_boundary)
            is_confident = not is_ambiguous
            
            all_results[window_ms].append({
                'elapsed_hours': row['elapsed_seconds'] / 3600,
                'truth_offset_ms': truth_offset_ms,
                'system_offset_ms': system_offset_ms,
                'chronotick_offset_ms': chronotick_offset_ms,
                'chronotick_unc_ms': chronotick_unc_ms,
                'diff_system': diff_system,
                'diff_chronotick': diff_chronotick,
                'agrees_system': agrees_system,
                'agrees_chronotick': agrees_chronotick,
                'truth_in_bounds': truth_in_bounds,
                'is_ambiguous': is_ambiguous,
                'is_confident': is_confident,
                'distance_from_boundary': distance_from_boundary,
            })
    
    # Calculate metrics
    results_summary = {}
    
    for window_ms in window_sizes:
        df_results = pd.DataFrame(all_results[window_ms])
        
        if len(df_results) == 0:
            continue
        
        # Agreement rates
        agree_system = (df_results['agrees_system'].sum() / len(df_results) * 100)
        agree_chronotick = (df_results['agrees_chronotick'].sum() / len(df_results) * 100)
        
        # Bounded clock metrics
        confident_df = df_results[df_results['is_confident'] == True]
        ambiguous_df = df_results[df_results['is_ambiguous'] == True]
        
        if len(confident_df) > 0:
            confident_correct = (confident_df['agrees_chronotick'].sum() / len(confident_df) * 100)
        else:
            confident_correct = 0
        
        truth_in_bounds_rate = (df_results['truth_in_bounds'].sum() / len(df_results) * 100)
        
        print(f"\n{'='*80}")
        print(f"Window {window_ms}ms ({len(df_results)} samples):")
        print('='*80)
        
        print(f"\n📊 SINGLE-POINT AGREEMENT:")
        print(f"  System clock (offset=0):  {df_results['agrees_system'].sum()}/{len(df_results)} = {agree_system:.1f}%")
        print(f"  ChronoTick:               {df_results['agrees_chronotick'].sum()}/{len(df_results)} = {agree_chronotick:.1f}%")
        print(f"  Improvement:              {agree_chronotick - agree_system:+.1f}%")
        
        print(f"\n🎯 BOUNDED CLOCK VALUE (the Google Spanner narrative!):")
        print(f"  Confident assignments:    {len(confident_df)}/{len(df_results)} = {len(confident_df)/len(df_results)*100:.1f}%")
        print(f"  Ambiguous (near boundary): {len(ambiguous_df)}/{len(df_results)} = {len(ambiguous_df)/len(df_results)*100:.1f}%")
        
        if len(confident_df) > 0:
            print(f"  When confident, correct:  {confident_df['agrees_chronotick'].sum()}/{len(confident_df)} = {confident_correct:.1f}%")
        
        print(f"\n✅ UNCERTAINTY CALIBRATION:")
        print(f"  Truth within ±3σ bounds:  {df_results['truth_in_bounds'].sum()}/{len(df_results)} = {truth_in_bounds_rate:.1f}%")
        print(f"  Median uncertainty:       {df_results['chronotick_unc_ms'].median():.2f}ms")
        
        results_summary[window_ms] = {
            'agree_system': agree_system,
            'agree_chronotick': agree_chronotick,
            'improvement': agree_chronotick - agree_system,
            'confident_rate': len(confident_df) / len(df_results) * 100,
            'ambiguous_rate': len(ambiguous_df) / len(df_results) * 100,
            'confident_correct': confident_correct,
            'truth_in_bounds_rate': truth_in_bounds_rate,
            'samples': len(df_results),
            'df': df_results
        }
    
    return results_summary

def main():
    """Run evaluation."""
    
    experiments = {
        'experiment-5 (node 1)': "results/experiment-5/ares-comp-11/data.csv",
        'experiment-5 (node 2)': "results/experiment-5/ares-comp-12/data.csv",
    }
    
    for exp_name, csv_path in experiments.items():
        print(f"\n{'#'*80}")
        print(f"# {exp_name.upper()}")
        print(f"{'#'*80}")
        
        results = evaluate_within_node(csv_path)
        
    print(f"\n\n{'='*80}")
    print("BOUNDED CLOCK EVALUATION COMPLETE")
    print('='*80)
    print("\nThe Google Spanner TrueTime narrative:")
    print("- System clock: Assumes no correction needed (offset=0)")
    print("- ChronoTick: Predicts correction with bounded uncertainty")
    print("\nValue: ChronoTick KNOWS when it's uncertain!")

if __name__ == "__main__":
    main()
