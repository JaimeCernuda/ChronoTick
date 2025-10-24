#!/usr/bin/env python3
"""
Ultra-deep narrative finder: Scans CSV + Logs for interesting micro-stories
Focuses on 10-120 minute windows, not 8-hour aggregates
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import json


class MicroStoryFinder:
    """Finds compelling narratives by combining CSV data + log analysis"""

    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.stories = []

    def parse_log_timestamp(self, log_line: str) -> datetime:
        """Extract timestamp from log line"""
        try:
            timestamp_str = log_line[:23]  # "2025-10-19 00:29:19,316"
            return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")
        except:
            return None

    def find_rejection_storms(self, log_path: Path, window_minutes: int = 15) -> List[Dict]:
        """Find periods with high NTP rejection rates"""
        if not log_path.exists():
            return []

        with open(log_path, 'r') as f:
            lines = f.readlines()

        # Parse all rejections
        rejections = []
        for line in lines:
            if 'REJECTED' in line and 'z=' in line:
                timestamp = self.parse_log_timestamp(line)
                if timestamp:
                    # Extract z-score
                    z_match = re.search(r'z=([0-9.]+)', line)
                    z_score = float(z_match.group(1)) if z_match else 0

                    # Extract offset
                    offset_match = re.search(r'offset=([-0-9.]+)', line)
                    offset = float(offset_match.group(1)) if offset_match else 0

                    rejections.append({
                        'timestamp': timestamp,
                        'z_score': z_score,
                        'offset': offset,
                        'line': line.strip()
                    })

        if not rejections:
            return []

        # Find density clusters (sliding window)
        start_time = rejections[0]['timestamp']
        end_time = rejections[-1]['timestamp']

        storms = []
        current = start_time
        while current < end_time:
            window_end = current + timedelta(minutes=window_minutes)

            # Count rejections in this window
            window_rejections = [r for r in rejections
                                if current <= r['timestamp'] < window_end]

            if len(window_rejections) >= 5:  # At least 5 rejections in window
                max_z = max([r['z_score'] for r in window_rejections])
                storms.append({
                    'start_time': current,
                    'end_time': window_end,
                    'duration_minutes': window_minutes,
                    'rejection_count': len(window_rejections),
                    'max_z_score': max_z,
                    'rejections': window_rejections
                })

            current += timedelta(minutes=5)  # Slide by 5 minutes

        return storms

    def find_source_switches(self, df: pd.DataFrame) -> List[Dict]:
        """Find when ChronoTick switches between models (error/cpu/fusion)"""
        if 'chronotick_source' not in df.columns:
            return []

        switches = []
        for i in range(1, len(df)):
            if df.iloc[i]['chronotick_source'] != df.iloc[i-1]['chronotick_source']:
                switches.append({
                    'time_seconds': df.iloc[i]['elapsed_seconds'],
                    'time_minutes': df.iloc[i]['elapsed_seconds'] / 60,
                    'from_source': df.iloc[i-1]['chronotick_source'],
                    'to_source': df.iloc[i]['chronotick_source'],
                    'offset_before': df.iloc[i-1]['chronotick_offset_ms'],
                    'offset_after': df.iloc[i]['chronotick_offset_ms'],
                    'uncertainty_before': df.iloc[i-1]['chronotick_uncertainty_ms'],
                    'uncertainty_after': df.iloc[i]['chronotick_uncertainty_ms']
                })

        return switches

    def find_uncertainty_spikes(self, df: pd.DataFrame, threshold: float = 5.0) -> List[Dict]:
        """Find periods where model uncertainty spikes"""
        if 'chronotick_uncertainty_ms' not in df.columns:
            return []

        spikes = []
        high_uncertainty = df[df['chronotick_uncertainty_ms'] > threshold]

        if len(high_uncertainty) == 0:
            return []

        # Group consecutive high-uncertainty samples
        groups = []
        current_group = [high_uncertainty.iloc[0]]

        for i in range(1, len(high_uncertainty)):
            time_gap = (high_uncertainty.iloc[i]['elapsed_seconds'] -
                       high_uncertainty.iloc[i-1]['elapsed_seconds'])

            if time_gap < 60:  # Within 1 minute
                current_group.append(high_uncertainty.iloc[i])
            else:
                if len(current_group) >= 3:  # At least 3 samples
                    groups.append(current_group)
                current_group = [high_uncertainty.iloc[i]]

        if len(current_group) >= 3:
            groups.append(current_group)

        for group in groups:
            group_df = pd.DataFrame(group)
            spikes.append({
                'start_time_minutes': group_df.iloc[0]['elapsed_seconds'] / 60,
                'end_time_minutes': group_df.iloc[-1]['elapsed_seconds'] / 60,
                'duration_minutes': (group_df.iloc[-1]['elapsed_seconds'] -
                                   group_df.iloc[0]['elapsed_seconds']) / 60,
                'max_uncertainty': group_df['chronotick_uncertainty_ms'].max(),
                'mean_uncertainty': group_df['chronotick_uncertainty_ms'].mean(),
                'sample_count': len(group)
            })

        return spikes

    def find_offset_jumps(self, df: pd.DataFrame, threshold: float = 3.0) -> List[Dict]:
        """Find sudden offset jumps in ChronoTick predictions"""
        if 'chronotick_offset_ms' not in df.columns or len(df) < 2:
            return []

        offset_changes = df['chronotick_offset_ms'].diff().abs()
        jumps = df[offset_changes > threshold]

        jump_events = []
        for idx, row in jumps.iterrows():
            if idx > 0:
                prev = df.iloc[idx - 1]
                jump_events.append({
                    'time_minutes': row['elapsed_seconds'] / 60,
                    'from_offset': prev['chronotick_offset_ms'],
                    'to_offset': row['chronotick_offset_ms'],
                    'delta': row['chronotick_offset_ms'] - prev['chronotick_offset_ms'],
                    'from_source': prev['chronotick_source'],
                    'to_source': row['chronotick_source']
                })

        return jump_events

    def find_ntp_ground_truth_divergence(self, df: pd.DataFrame, threshold: float = 2.0) -> List[Dict]:
        """Find when ChronoTick prediction diverges from NTP ground truth"""
        if 'ntp_offset_ms' not in df.columns or 'chronotick_offset_ms' not in df.columns:
            return []

        # Only look at samples where we have NTP measurements
        ntp_samples = df[df['has_ntp'] == True].copy()

        if len(ntp_samples) == 0:
            return []

        # Calculate prediction error
        ntp_samples['prediction_error'] = abs(ntp_samples['chronotick_offset_ms'] -
                                              ntp_samples['ntp_offset_ms'])

        large_errors = ntp_samples[ntp_samples['prediction_error'] > threshold]

        divergences = []
        for idx, row in large_errors.iterrows():
            divergences.append({
                'time_minutes': row['elapsed_seconds'] / 60,
                'chronotick_offset': row['chronotick_offset_ms'],
                'ntp_offset': row['ntp_offset_ms'],
                'error': row['prediction_error'],
                'source': row['chronotick_source'],
                'uncertainty': row['chronotick_uncertainty_ms']
            })

        return divergences

    def find_warmup_period_narrative(self, df: pd.DataFrame, log_path: Path) -> Dict:
        """Detailed analysis of first 30 minutes (warmup period)"""
        warmup_df = df[df['elapsed_seconds'] <= 1800].copy()  # First 30 minutes

        if len(warmup_df) == 0:
            return {}

        # Source progression
        source_progression = warmup_df.groupby('chronotick_source').size().to_dict()

        # When did first NTP arrive?
        first_ntp = warmup_df[warmup_df['has_ntp'] == True]
        first_ntp_time = first_ntp.iloc[0]['elapsed_seconds'] if len(first_ntp) > 0 else None

        # How many source switches in warmup?
        switches = self.find_source_switches(warmup_df)

        # Rejections during warmup
        rejection_count = 0
        if log_path.exists():
            with open(log_path, 'r') as f:
                for line in f:
                    if 'REJECTED' in line:
                        timestamp = self.parse_log_timestamp(line)
                        if timestamp:
                            # Check if within first 30 min of experiment
                            experiment_start = self.parse_log_timestamp(open(log_path).readline())
                            if experiment_start and (timestamp - experiment_start).total_seconds() <= 1800:
                                rejection_count += 1

        return {
            'duration_minutes': 30,
            'source_progression': source_progression,
            'first_ntp_arrival_seconds': first_ntp_time,
            'source_switches': len(switches),
            'ntp_rejections': rejection_count,
            'initial_uncertainty': warmup_df.iloc[0]['chronotick_uncertainty_ms'],
            'final_uncertainty': warmup_df.iloc[-1]['chronotick_uncertainty_ms'],
            'mean_offset': warmup_df['chronotick_offset_ms'].mean(),
            'offset_std': warmup_df['chronotick_offset_ms'].std()
        }

    def find_thermal_wandering(self, df: pd.DataFrame, window_hours: Tuple[float, float]) -> Dict:
        """Detect 'wandering' offset pattern (non-linear drift characteristic of temperature)"""
        start_hour, end_hour = window_hours
        window_df = df[(df['elapsed_seconds'] >= start_hour * 3600) &
                      (df['elapsed_seconds'] < end_hour * 3600)].copy()

        if len(window_df) < 50:
            return {}

        # Calculate drift rate in consecutive 30-minute sub-windows
        drift_rates = []
        window_duration_minutes = (end_hour - start_hour) * 60

        for start_min in range(0, int(window_duration_minutes), 30):
            sub_window = window_df[(window_df['elapsed_seconds'] >= (start_hour * 3600 + start_min * 60)) &
                                  (window_df['elapsed_seconds'] < (start_hour * 3600 + (start_min + 30) * 60))]

            if len(sub_window) >= 10:
                # Linear fit to get drift rate
                times = sub_window['elapsed_seconds'].values
                offsets = sub_window['chronotick_offset_ms'].values

                if len(times) > 1:
                    coeffs = np.polyfit(times, offsets, 1)
                    drift_rate = coeffs[0] * 3600  # ms/hour
                    drift_rates.append(drift_rate)

        if len(drift_rates) < 2:
            return {}

        # "Wandering" = changing drift rates (high variance in rates)
        drift_rate_variance = np.var(drift_rates)
        drift_rate_range = max(drift_rates) - min(drift_rates)

        # Check for sign changes (direction reversals)
        sign_changes = sum(1 for i in range(len(drift_rates)-1)
                          if np.sign(drift_rates[i]) != np.sign(drift_rates[i+1]))

        return {
            'window_hours': f'{start_hour:.1f}-{end_hour:.1f}',
            'duration_minutes': window_duration_minutes,
            'drift_rates_ms_per_hour': drift_rates,
            'drift_rate_variance': drift_rate_variance,
            'drift_rate_range': drift_rate_range,
            'direction_changes': sign_changes,
            'is_wandering': sign_changes >= 2 or drift_rate_variance > 1.0,
            'offset_mean': window_df['chronotick_offset_ms'].mean(),
            'offset_std': window_df['chronotick_offset_ms'].std(),
            'offset_range': window_df['chronotick_offset_ms'].max() - window_df['chronotick_offset_ms'].min()
        }

    def analyze_experiment(self, experiment_path: Path) -> Dict:
        """Complete analysis of one experiment combining CSV + logs"""
        # Try data.csv first, then chronotick_client_validation_*.csv
        csv_path = experiment_path / 'data.csv'
        if not csv_path.exists():
            # Look for chronotick_client_validation_*.csv
            validation_csvs = list(experiment_path.glob('chronotick_client_validation_*.csv'))
            if validation_csvs:
                csv_path = validation_csvs[0]
            else:
                return {}

        daemon_log = experiment_path / 'daemon.log'
        # Also try alternative log names
        if not daemon_log.exists():
            # Look for homelab_stdout_*.log or similar
            log_files = list(experiment_path.glob('*stdout*.log')) + list(experiment_path.glob('daemon*.log'))
            if log_files:
                daemon_log = log_files[0]

        print(f"\n🔍 Analyzing {experiment_path.name}...")

        df = pd.read_csv(csv_path)

        # All analyses
        stories = {
            'experiment': experiment_path.parent.name + '/' + experiment_path.name,
            'csv_path': str(csv_path),
            'log_path': str(daemon_log) if daemon_log.exists() else None,
            'duration_hours': df['elapsed_seconds'].max() / 3600,
            'total_samples': len(df),
        }

        # Find interesting events
        stories['rejection_storms'] = self.find_rejection_storms(daemon_log, window_minutes=15)
        stories['source_switches'] = self.find_source_switches(df)
        stories['uncertainty_spikes'] = self.find_uncertainty_spikes(df, threshold=3.0)
        stories['offset_jumps'] = self.find_offset_jumps(df, threshold=2.0)
        stories['ntp_divergences'] = self.find_ntp_ground_truth_divergence(df, threshold=2.0)
        stories['warmup_analysis'] = self.find_warmup_period_narrative(df, daemon_log)

        # Check for thermal wandering in different windows
        stories['thermal_analysis'] = {}
        for start_h, end_h in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]:
            if df['elapsed_seconds'].max() >= end_h * 3600:
                thermal = self.find_thermal_wandering(df, (start_h, end_h))
                if thermal:
                    stories['thermal_analysis'][f'hour_{start_h}-{end_h}'] = thermal

        # Print summary
        print(f"  ✅ Found:")
        print(f"     • {len(stories['rejection_storms'])} rejection storm periods")
        print(f"     • {len(stories['source_switches'])} model source switches")
        print(f"     • {len(stories['uncertainty_spikes'])} uncertainty spike events")
        print(f"     • {len(stories['offset_jumps'])} offset jump events")
        print(f"     • {len(stories['ntp_divergences'])} NTP divergence events")
        if stories['warmup_analysis']:
            print(f"     • Warmup: {stories['warmup_analysis'].get('ntp_rejections', 0)} rejections")
        wandering_periods = sum(1 for k, v in stories['thermal_analysis'].items() if v.get('is_wandering', False))
        print(f"     • {wandering_periods} thermal wandering periods detected")

        return stories

    def scan_all_experiments(self) -> List[Dict]:
        """Scan all experiments for interesting micro-stories"""
        all_stories = []

        for exp_dir in sorted(self.results_dir.iterdir()):
            if not exp_dir.is_dir() or not exp_dir.name.startswith('experiment-'):
                continue

            print(f"\n{'='*80}")
            print(f"📁 {exp_dir.name}")
            print(f"{'='*80}")

            # Check for subdirectories (platforms)
            for platform_dir in sorted(exp_dir.iterdir()):
                if platform_dir.is_dir():
                    stories = self.analyze_experiment(platform_dir)
                    if stories:
                        all_stories.append(stories)

        return all_stories


def main():
    results_dir = Path('/home/jcernuda/tick_project/ChronoTick/results')
    output_dir = results_dir / 'analysis'
    output_dir.mkdir(exist_ok=True)

    finder = MicroStoryFinder(results_dir)
    all_stories = finder.scan_all_experiments()

    # Save results
    output_path = output_dir / 'micro_stories_combined.json'
    with open(output_path, 'w') as f:
        json.dump(all_stories, f, indent=2, default=str)

    print(f"\n{'='*80}")
    print(f"✅ Analysis complete! Found {len(all_stories)} experiment stories")
    print(f"   Saved to: {output_path}")
    print(f"{'='*80}")

    # Generate summary markdown
    generate_summary_markdown(all_stories, output_dir / 'INTERESTING_MICRO_STORIES.md')


def generate_summary_markdown(stories: List[Dict], output_path: Path):
    """Generate human-readable summary of interesting stories"""
    with open(output_path, 'w') as f:
        f.write("# Interesting Micro-Stories Found in Data + Logs\n\n")
        f.write(f"**Total Experiments Analyzed**: {len(stories)}\n\n")
        f.write("---\n\n")

        for story in stories:
            f.write(f"## {story['experiment']}\n\n")
            f.write(f"**Duration**: {story['duration_hours']:.2f} hours ({story['total_samples']} samples)\n\n")

            # Rejection storms
            if story['rejection_storms']:
                f.write(f"### 🌪️ Rejection Storms ({len(story['rejection_storms'])} periods)\n\n")
                for i, storm in enumerate(story['rejection_storms'][:3], 1):  # Top 3
                    f.write(f"**Storm {i}**: {storm['start_time']} to {storm['end_time']}\n")
                    f.write(f"- Duration: {storm['duration_minutes']} minutes\n")
                    f.write(f"- Rejections: {storm['rejection_count']}\n")
                    f.write(f"- Max z-score: {storm['max_z_score']:.2f}σ\n")
                    f.write(f"- **Story**: Network quality collapsed with {storm['rejection_count']} rejections in {storm['duration_minutes']}min, z-scores up to {storm['max_z_score']:.1f}σ\n\n")

            # Source switches
            if story['source_switches']:
                f.write(f"### 🔄 Model Source Switches ({len(story['source_switches'])} events)\n\n")
                for i, switch in enumerate(story['source_switches'][:5], 1):
                    f.write(f"**Switch {i}** at {switch['time_minutes']:.1f} min: `{switch['from_source']}` → `{switch['to_source']}`\n")
                    f.write(f"- Offset change: {switch['offset_before']:.2f}ms → {switch['offset_after']:.2f}ms\n")
                    f.write(f"- Uncertainty: {switch['uncertainty_before']:.2f}ms → {switch['uncertainty_after']:.2f}ms\n\n")

            # Warmup analysis
            if story.get('warmup_analysis'):
                warmup = story['warmup_analysis']
                f.write(f"### 🚀 Warmup Period Analysis\n\n")
                f.write(f"- NTP Rejections: **{warmup.get('ntp_rejections', 0)}**\n")
                f.write(f"- First NTP arrival: {warmup.get('first_ntp_arrival_seconds') or 0:.1f} seconds\n")
                f.write(f"- Source switches: {warmup.get('source_switches', 0)}\n")
                f.write(f"- Uncertainty: {warmup.get('initial_uncertainty') or 0:.2f}ms → {warmup.get('final_uncertainty') or 0:.2f}ms\n")
                f.write(f"- Mean offset: {warmup.get('mean_offset') or 0:.2f}ms (σ={warmup.get('offset_std') or 0:.2f}ms)\n\n")

            # Thermal wandering
            wandering = [k for k, v in story.get('thermal_analysis', {}).items() if v.get('is_wandering', False)]
            if wandering:
                f.write(f"### 🌡️ Thermal Wandering Detected\n\n")
                for period in wandering:
                    thermal = story['thermal_analysis'][period]
                    f.write(f"**Period**: {thermal['window_hours']}\n")
                    f.write(f"- Direction changes: {thermal['direction_changes']}\n")
                    f.write(f"- Drift rate range: {thermal['drift_rate_range']:.2f} ms/hour\n")
                    f.write(f"- Offset σ: {thermal['offset_std']:.2f}ms, range: {thermal['offset_range']:.2f}ms\n\n")

            f.write("---\n\n")

    print(f"📝 Summary written to {output_path}")


if __name__ == '__main__':
    main()
