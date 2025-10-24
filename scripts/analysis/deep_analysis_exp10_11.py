#!/usr/bin/env python3
"""
Ultra-deep analysis of experiments 10-11 looking for:
- Recovery patterns and transients
- Long-duration stability
- Offset jumps/shifts
- Interesting micro-windows
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import List, Dict, Tuple
import re


class Experiment10_11Analyzer:
    """Deep analyzer for experiments 10 and 11"""

    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.exp10_dir = results_dir / "experiment-10"
        self.exp11_dir = results_dir / "experiment-11"

    def load_experiment_data(self, csv_path: Path) -> pd.DataFrame:
        """Load experiment CSV with flexible column handling"""
        df = pd.read_csv(csv_path)

        # Handle different time column formats
        if 'elapsed_seconds' in df.columns:
            df['time_hours'] = df['elapsed_seconds'] / 3600
            df['time_minutes'] = df['elapsed_seconds'] / 60
        elif 'time_from_start' in df.columns:
            df['time_hours'] = df['time_from_start'] / 3600
            df['time_minutes'] = df['time_from_start'] / 60
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['time_hours'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds() / 3600
            df['time_minutes'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds() / 60

        return df

    def detect_offset_jumps(self, df: pd.DataFrame, threshold_ms: float = 2.0) -> List[Dict]:
        """Detect sudden offset jumps/shifts"""
        jumps = []

        if 'chronotick_offset_ms' not in df.columns:
            return jumps

        offsets = df['chronotick_offset_ms'].values
        times = df['time_minutes'].values

        for i in range(1, len(offsets)):
            delta = abs(offsets[i] - offsets[i-1])
            if delta >= threshold_ms:
                jumps.append({
                    'time_minutes': float(times[i]),
                    'sample': int(i),
                    'old_offset': float(offsets[i-1]),
                    'new_offset': float(offsets[i]),
                    'delta': float(delta),
                    'jump_type': 'increase' if offsets[i] > offsets[i-1] else 'decrease'
                })

        return jumps

    def detect_oscillations(self, df: pd.DataFrame, window_size: int = 10) -> List[Dict]:
        """Detect oscillating behavior in offsets"""
        oscillations = []

        if 'chronotick_offset_ms' not in df.columns or len(df) < window_size * 2:
            return oscillations

        offsets = df['chronotick_offset_ms'].values
        times = df['time_minutes'].values

        for i in range(window_size, len(offsets) - window_size):
            window = offsets[i-window_size:i+window_size]

            # Count sign changes (oscillations)
            diffs = np.diff(window)
            sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)

            # High sign changes = oscillation
            if sign_changes > window_size:
                oscillations.append({
                    'time_minutes': float(times[i]),
                    'sample': int(i),
                    'sign_changes': int(sign_changes),
                    'std_ms': float(np.std(window)),
                    'range_ms': float(np.max(window) - np.min(window))
                })

        return oscillations

    def find_stability_periods(self, df: pd.DataFrame, max_std_ms: float = 0.5) -> List[Dict]:
        """Find periods of exceptional stability"""
        stable_periods = []

        if 'chronotick_offset_ms' not in df.columns:
            return stable_periods

        # Analyze in 30-minute windows
        max_hour = df['time_hours'].max()
        for start_hour in np.arange(0, max_hour, 0.5):  # 30-min steps
            end_hour = start_hour + 0.5
            window_data = df[(df['time_hours'] >= start_hour) & (df['time_hours'] < end_hour)]

            if len(window_data) < 10:
                continue

            chronotick_std = window_data['chronotick_offset_ms'].std()
            chronotick_range = window_data['chronotick_offset_ms'].max() - window_data['chronotick_offset_ms'].min()

            if chronotick_std <= max_std_ms:
                stable_periods.append({
                    'start_hour': float(start_hour),
                    'end_hour': float(end_hour),
                    'duration_minutes': 30,
                    'std_ms': float(chronotick_std),
                    'range_ms': float(chronotick_range),
                    'mean_offset_ms': float(window_data['chronotick_offset_ms'].mean()),
                    'samples': len(window_data)
                })

        return stable_periods

    def analyze_long_term_drift(self, df: pd.DataFrame) -> Dict:
        """Analyze long-term drift patterns"""
        if 'chronotick_offset_ms' not in df.columns or len(df) < 100:
            return {}

        # Fit linear trend
        times = df['time_hours'].values
        offsets = df['chronotick_offset_ms'].values

        # Remove NaN values
        mask = ~np.isnan(offsets) & ~np.isnan(times)
        times_clean = times[mask]
        offsets_clean = offsets[mask]

        if len(times_clean) < 2:
            return {}

        # Linear fit
        coeffs = np.polyfit(times_clean, offsets_clean, 1)
        drift_rate_ms_per_hour = float(coeffs[0])

        # Calculate residuals
        predicted = np.polyval(coeffs, times_clean)
        residuals = offsets_clean - predicted

        return {
            'duration_hours': float(df['time_hours'].max()),
            'initial_offset_ms': float(offsets_clean[0]),
            'final_offset_ms': float(offsets_clean[-1]),
            'total_drift_ms': float(offsets_clean[-1] - offsets_clean[0]),
            'drift_rate_ms_per_hour': drift_rate_ms_per_hour,
            'residual_std_ms': float(np.std(residuals)),
            'samples': len(df)
        }

    def scan_logs_for_events(self, log_path: Path, max_lines: int = 5000) -> Dict:
        """Scan log file for interesting events (first/last N lines)"""
        if not log_path.exists():
            return {}

        events = {
            'errors': [],
            'warnings': [],
            'rejections': 0,
            'quality_issues': 0,
            'crashes': []
        }

        try:
            # Read first and last portions of log
            with open(log_path, 'r') as f:
                lines = f.readlines()

            # Sample first 2500 and last 2500 lines
            sample_lines = lines[:2500] + lines[-2500:]

            for line in sample_lines:
                line_lower = line.lower()

                if 'error' in line_lower and 'ERROR' in line:
                    events['errors'].append(line.strip()[:200])

                if 'warning' in line_lower and 'WARNING' in line:
                    events['warnings'].append(line.strip()[:200])

                if 'reject' in line_lower:
                    events['rejections'] += 1

                if 'quality' in line_lower or 'uncertain' in line_lower:
                    events['quality_issues'] += 1

                if 'crash' in line_lower or 'died' in line_lower or 'failed' in line_lower:
                    events['crashes'].append(line.strip()[:200])

        except Exception as e:
            print(f"Error reading log {log_path}: {e}")

        # Limit stored messages
        events['errors'] = events['errors'][:10]
        events['warnings'] = events['warnings'][:10]
        events['crashes'] = events['crashes'][:5]

        return events

    def analyze_platform(self, platform_dir: Path, experiment_name: str) -> Dict:
        """Complete analysis for one platform"""
        platform_name = platform_dir.name
        print(f"\n🔍 Analyzing {experiment_name}/{platform_name}...")

        # Find CSV file
        csv_files = list(platform_dir.glob("*.csv"))
        if not csv_files:
            print(f"  ⚠️  No CSV found")
            return {}

        csv_path = csv_files[0]
        df = self.load_experiment_data(csv_path)

        print(f"  📊 {len(df)} samples, {df['time_hours'].max():.2f} hours")

        # Run all analyses
        offset_jumps = self.detect_offset_jumps(df, threshold_ms=2.0)
        oscillations = self.detect_oscillations(df, window_size=10)
        stable_periods = self.find_stability_periods(df, max_std_ms=0.5)
        long_term_drift = self.analyze_long_term_drift(df)

        # Log analysis
        log_files = list(platform_dir.glob("*.log"))
        log_events = {}
        if log_files:
            log_events = self.scan_logs_for_events(log_files[0], max_lines=5000)

        print(f"  ✅ Found {len(offset_jumps)} offset jumps, {len(stable_periods)} stable periods")

        return {
            'experiment': experiment_name,
            'platform': platform_name,
            'csv_path': str(csv_path),
            'samples': len(df),
            'duration_hours': float(df['time_hours'].max()),
            'offset_jumps': offset_jumps,
            'oscillations': oscillations,
            'stable_periods': stable_periods,
            'long_term_drift': long_term_drift,
            'log_events': log_events,
            'statistics': {
                'mean_offset_ms': float(df['chronotick_offset_ms'].mean()) if 'chronotick_offset_ms' in df.columns else None,
                'std_offset_ms': float(df['chronotick_offset_ms'].std()) if 'chronotick_offset_ms' in df.columns else None,
                'max_offset_ms': float(df['chronotick_offset_ms'].abs().max()) if 'chronotick_offset_ms' in df.columns else None,
            }
        }

    def analyze_all(self) -> Dict:
        """Analyze all platforms in experiments 10 and 11"""
        results = {
            'experiment-10': [],
            'experiment-11': []
        }

        print("\n" + "="*80)
        print("🔬 ULTRA-DEEP ANALYSIS: EXPERIMENTS 10 & 11")
        print("="*80)

        # Experiment 10
        print("\n📁 EXPERIMENT-10:")
        for platform_dir in sorted(self.exp10_dir.iterdir()):
            if platform_dir.is_dir():
                analysis = self.analyze_platform(platform_dir, "experiment-10")
                if analysis:
                    results['experiment-10'].append(analysis)

        # Experiment 11 - both collections
        print("\n📁 EXPERIMENT-11:")
        exp11_collections = ['first_collection', 'collection2']
        for collection in exp11_collections:
            collection_dir = self.exp11_dir / collection
            if not collection_dir.exists():
                continue

            print(f"\n  Collection: {collection}")
            for platform_dir in sorted(collection_dir.iterdir()):
                if platform_dir.is_dir():
                    analysis = self.analyze_platform(platform_dir, f"experiment-11/{collection}")
                    if analysis:
                        results['experiment-11'].append(analysis)

        return results


def _convert_to_native_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: _convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_native_types(item) for item in obj]
    else:
        return obj


def main():
    results_dir = Path("/home/jcernuda/tick_project/ChronoTick/results")
    output_dir = results_dir / "analysis"
    output_dir.mkdir(exist_ok=True)

    analyzer = Experiment10_11Analyzer(results_dir)
    results = analyzer.analyze_all()

    # Convert to native types
    results = _convert_to_native_types(results)

    # Save results
    output_path = output_dir / "experiment_10_11_deep_analysis.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*80)
    print(f"✅ Analysis complete! Results saved to:")
    print(f"   {output_path}")
    print("="*80)

    # Print summary
    print("\n📊 SUMMARY:")
    for exp_name, platforms in results.items():
        print(f"\n{exp_name.upper()}:")
        for platform in platforms:
            print(f"  • {platform['platform']}: {platform['duration_hours']:.1f}h, "
                  f"{len(platform['offset_jumps'])} jumps, "
                  f"{len(platform['stable_periods'])} stable periods")

            if platform['log_events'].get('rejections', 0) > 0:
                print(f"    ⚠️  {platform['log_events']['rejections']} NTP rejections")

            if platform['long_term_drift']:
                drift = platform['long_term_drift']
                print(f"    📈 Drift: {drift['drift_rate_ms_per_hour']:.3f} ms/hour")


if __name__ == "__main__":
    main()
